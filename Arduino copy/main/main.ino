#include <Arduino.h>
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include "model_data_5s.h"
#include "audio_processor.h"

// ======================= HARDWARE CONFIG =======================
// IMPORTANT: These GPIO numbers must match your physical wiring!
// On ESP32-S3, GPIO labels on PCB may differ from actual GPIO numbers.
// Check your board's pinout diagram to find which physical pins correspond
// to these GPIO numbers, then wire INMP441 accordingly:
//   INMP441 SCK -> GPIO I2S_BCK_PIN
//   INMP441 WS  -> GPIO I2S_WS_PIN
//   INMP441 SD  -> GPIO I2S_SD_PIN
//   INMP441 L/R -> GND
//   INMP441 VCC -> 3.3V
//   INMP441 GND -> GND
//
// Common safe GPIOs for I2S on ESP32-S3 (avoid GPIO 0, 3, 45, 46 which are strapping pins):
// GPIO 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, etc.
// Update these to match your actual wiring:
#define I2S_BCK_PIN 5 // Bit clock - connect to INMP441 SCK
#define I2S_WS_PIN 4  // Word select - connect to INMP441 WS
#define I2S_SD_PIN 6  // Serial data - connect to INMP441 SD
#define I2S_PORT I2S_NUM_0

// Model quantization parameters from cough_cnn_5s_base_int8.tflite
const float MODEL_INPUT_SCALE = 0.09420423954725266f;
const int MODEL_INPUT_ZERO_POINT = -1;
const float OUTPUT_SCALE = 0.00390625f;
const int OUTPUT_ZERO_POINT = -128;

// Detection threshold
const float COUGH_THRESHOLD = 0.60f;
// Set to 1 if your model outputs [non_cough, cough]
// Set to 0 if your model outputs [cough, non_cough]
const int COUGH_INDEX = 1;
#define TENSOR_ARENA_SIZE (160 * 1024)

// ======================= GLOBAL OBJECTS & BUFFERS =======================
Eloquent::TF::Sequential<30, TENSOR_ARENA_SIZE> tf;
AudioProcessor audioProcessor(I2S_BCK_PIN, I2S_WS_PIN, I2S_SD_PIN, MODEL_INPUT_SCALE, MODEL_INPUT_ZERO_POINT, I2S_PORT);

// CRITICAL: Main audio buffer allocated in PSRAM
float *g_audio_buffer_psram = nullptr;
// Model input buffer allocated in PSRAM
int8_t *g_model_input_buffer = nullptr;

// ======================= FUNCTION DECLARATIONS =======================
bool initializePSRAM();
bool initializeModel();
bool initializeAudioProcessor();
void runInference();
// void printSystemStatus();

// ======================= SETUP =======================
void setup()
{
  Serial.begin(460800);

  // CRITICAL for ESP32-S3 USB Serial
  while (!Serial)
  {
    delay(10);
  }
  delay(1000);

  Serial.println("\n==================================================");
  Serial.println("     ESP32-S3 REAL-TIME COUGH DETECTION SYSTEM");
  Serial.println("==================================================");

  // 1. Initialize PSRAM and allocate buffers
  Serial.println("\n[1/3] Initializing PSRAM...");
  if (!initializePSRAM())
  {
    Serial.println(" SYSTEM HALTED: PSRAM initialization failed.");
    while (1)
    {
      delay(1000);
    }
  }

  // CRITICAL: Connect PSRAM buffer to AudioProcessor
  audioProcessor.setAudioBuffer(g_audio_buffer_psram);

  // 2. Initialize TensorFlow Lite Model
  Serial.println("\n[2/3] Initializing TensorFlow Lite Model...");
  if (!initializeModel())
  {
    Serial.println(" SYSTEM HALTED: Model initialization failed.");
    while (1)
    {
      delay(1000);
    }
  }

  // 3. Initialize Audio Processor
  Serial.println("\n[3/3] Initializing Audio Processor...");
  if (!initializeAudioProcessor())
  {
    Serial.println(" SYSTEM HALTED: Audio processor initialization failed.");
    while (1)
    {
      delay(1000);
    }
  }

  // Print final system status
  // printSystemStatus();

  Serial.println("\n SYSTEM INITIALIZATION COMPLETE");
  Serial.println("   Listening for audio...");
  Serial.println("==================================================\n");
}

// ======================= MAIN LOOP =======================
void loop()
{
  static unsigned long lastInferenceTime = 0;
  static unsigned long lastStatusTime = 0;
  // MFCC extraction takes longer for a 5s window on ESP32-S3.
  // Keep only a short post-inference capture gap to improve responsiveness.
  const unsigned long INFERENCE_INTERVAL_MS = 1000UL;
  const unsigned long STATUS_INTERVAL_MS = 5000;

  unsigned long currentTime = millis();

  // 1. Continuously read audio from I2S microphone
  const int I2S_READ_BUFFER_SIZE = 512;
  static int32_t i2sBuffer[I2S_READ_BUFFER_SIZE];

  int samplesRead = audioProcessor.read(i2sBuffer, I2S_READ_BUFFER_SIZE);
  (void)samplesRead;


  // 2. Run inference at regular intervals
  if (currentTime - lastInferenceTime >= INFERENCE_INTERVAL_MS)
  {
    // Check if we have enough audio collected
    if (audioProcessor.hasEnoughAudio())
    {
      // Serial.printf("\n[%lu] Running inference...", currentTime / 1000);
      runInference();
      // Set timestamp after inference so we wait for fresh post-inference audio.
      lastInferenceTime = millis();
    }
    else
    {
      // Show collection progress
      // int percent = (audioProcessor._samples_collected * 100) / ANALYSIS_SAMPLES;
      // Serial.printf("\n[%lu] Collecting audio: %d%% (%d/%d samples)",
      //               currentTime / 1000,
      //               percent,
      //               audioProcessor._samples_collected,
      //               ANALYSIS_SAMPLES);
    }
  }

  // 3. Periodically print system status
  // if (currentTime - lastStatusTime >= STATUS_INTERVAL_MS)
  // {
  //   lastStatusTime = currentTime;
  //   printSystemStatus();
  // }

  // Small delay to prevent watchdog issues
  // delay(1);
}

// ======================= FUNCTION IMPLEMENTATIONS =======================

bool initializePSRAM()
{
  Serial.println("--- PSRAM Initialization ---");

  // Diagnostic: Check chip info
  Serial.printf("Chip model: %s\n", ESP.getChipModel());
  Serial.printf("Chip revision: %d\n", ESP.getChipRevision());
  Serial.printf("Free heap before allocation: %.2f KB\n", ESP.getFreeHeap() / 1024.0);

  if (psramFound())
  {
    Serial.println(" PSRAM detected by system.");
    Serial.printf("   Total PSRAM: %.2f MB\n", ESP.getPsramSize() / (1024.0 * 1024.0));
    Serial.printf("   Free PSRAM: %.2f MB\n", ESP.getFreePsram() / (1024.0 * 1024.0));

    // 1. Allocate main audio buffer in PSRAM
    size_t audio_buffer_bytes = BUFFER_SAMPLES * sizeof(float);
    Serial.printf("Allocating audio buffer: %.2f MB in PSRAM...", audio_buffer_bytes / (1024.0 * 1024.0));

    g_audio_buffer_psram = (float *)ps_malloc(audio_buffer_bytes);
    if (g_audio_buffer_psram == nullptr)
    {
      Serial.println(" FAILED!");
      Serial.println("Falling back to regular heap...");
      g_audio_buffer_psram = (float *)malloc(audio_buffer_bytes);
      if (g_audio_buffer_psram == nullptr)
      {
        Serial.println("Heap allocation also FAILED!");
        return false;
      }
      Serial.println(" SUCCESS (using heap instead of PSRAM)!");
    }
    else
    {
      Serial.println(" SUCCESS!");
    }
    memset(g_audio_buffer_psram, 0, audio_buffer_bytes);

    // 2. Allocate model input buffer in PSRAM
    size_t model_buffer_bytes = NUM_INPUTS * sizeof(int8_t);
    Serial.printf("Allocating model buffer: %.2f MB in PSRAM...", model_buffer_bytes / (1024.0 * 1024.0));

    if (psramFound())
    {
      g_model_input_buffer = (int8_t *)ps_malloc(model_buffer_bytes);
    }
    else
    {
      g_model_input_buffer = (int8_t *)malloc(model_buffer_bytes);
    }

    if (g_model_input_buffer == nullptr)
    {
      Serial.println(" FAILED!");
      free(g_audio_buffer_psram);
      g_audio_buffer_psram = nullptr;
      return false;
    }
    Serial.println(" SUCCESS!");
    memset(g_model_input_buffer, 0, model_buffer_bytes);

    if (psramFound())
    {
      Serial.printf("Free PSRAM after allocation: %.2f MB\n", ESP.getFreePsram() / (1024.0 * 1024.0));
    }
    Serial.printf("Free heap after allocation: %.2f KB\n", ESP.getFreeHeap() / 1024.0);
    return true;
  }

  // PSRAM not found - try using regular heap as fallback
  Serial.println(" PSRAM NOT FOUND!");
  Serial.println(" Attempting fallback to regular heap (may cause memory issues)...");
  Serial.printf("   Free heap: %.2f KB\n", ESP.getFreeHeap() / 1024.0);

  // 1. Allocate main audio buffer in heap
  size_t audio_buffer_bytes = BUFFER_SAMPLES * sizeof(float);
  Serial.printf("Allocating audio buffer: %.2f MB in heap...", audio_buffer_bytes / (1024.0 * 1024.0));

  g_audio_buffer_psram = (float *)malloc(audio_buffer_bytes);
  if (g_audio_buffer_psram == nullptr)
  {
    Serial.println(" FAILED!");
    Serial.println("ERROR: Not enough heap memory for audio buffer!");
    return false;
  }
  Serial.println(" SUCCESS!");
  memset(g_audio_buffer_psram, 0, audio_buffer_bytes);

  // 2. Allocate model input buffer in heap
  size_t model_buffer_bytes = NUM_INPUTS * sizeof(int8_t);
  Serial.printf("Allocating model buffer: %.2f MB in heap...", model_buffer_bytes / (1024.0 * 1024.0));

  g_model_input_buffer = (int8_t *)malloc(model_buffer_bytes);
  if (g_model_input_buffer == nullptr)
  {
    Serial.println(" FAILED!");
    free(g_audio_buffer_psram);
    g_audio_buffer_psram = nullptr;
    return false;
  }
  Serial.println(" SUCCESS!");
  memset(g_model_input_buffer, 0, model_buffer_bytes);

  Serial.printf("Free heap after allocation: %.2f KB\n", ESP.getFreeHeap() / 1024.0);
  Serial.println("WARNING: Using heap instead of PSRAM - monitor memory usage!");
  return true;
}

bool initializeModel()
{
  // Configure the model wrapper
  tf.setNumInputs(NUM_INPUTS);
  tf.setNumOutputs(2);

  // Register all operations needed by your model
  tf.resolver.AddExpandDims();
  tf.resolver.AddReshape();
  tf.resolver.AddConv2D();
  tf.resolver.AddMaxPool2D();
  tf.resolver.AddAdd();
  tf.resolver.AddMul();
  tf.resolver.AddMean();
  tf.resolver.AddFullyConnected();
  tf.resolver.AddSoftmax();

  // Load the model
  Serial.println("Loading TensorFlow Lite model...");
  auto status = tf.begin(cough_cnn_5s_base_int8_tflite);

  if (!status.isOk())
  {
    Serial.print(" Model load failed: ");
    Serial.println(status.toString());
    return false;
  }

  Serial.println(" Model loaded successfully");
  Serial.printf("  Model size: %d bytes\n", cough_cnn_5s_base_int8_tflite_len);
  Serial.printf("  Tensor arena: %d bytes\n", TENSOR_ARENA_SIZE);

  // Run a dummy inference to warm up
  Serial.println("Warming up model with dummy inference...");
  unsigned long startTime = micros();
  auto predictStatus = tf.predict(g_model_input_buffer);
  unsigned long warmupTime = micros() - startTime;

  if (!predictStatus.isOk())
  {
    Serial.print(" Warmup failed: ");
    Serial.println(predictStatus.toString());
    return false;
  }

  Serial.printf(" Model warmup complete: %lu us\n", warmupTime);
  return true;
}

bool initializeAudioProcessor()
{
  Serial.println("Initializing I2S microphone and audio processor...");

  if (!audioProcessor.begin(SAMPLE_RATE))
  {
    Serial.println(" Audio processor failed to initialize.");
    return false;
  }

  Serial.println(" Audio processor initialized");
  Serial.printf("  Sample rate: %d Hz\n", SAMPLE_RATE);
  Serial.printf("  Analysis window: %d seconds\n", ANALYSIS_SECONDS);
  Serial.printf("  Buffer samples: %d\n", BUFFER_SAMPLES);

  return true;
}

void runInference()
{
  // Timestamp (seconds since boot) for this inference
  unsigned long t_seconds = millis() / 1000;
  unsigned long window_end_seconds = t_seconds;
  unsigned long window_start_seconds = (window_end_seconds > ANALYSIS_SECONDS)
                                           ? (window_end_seconds - ANALYSIS_SECONDS)
                                           : 0;

  // 1. Extract MFCC features directly into PSRAM buffer
  unsigned long featureStart = micros();
  if (!audioProcessor.getMFCCFeatures(g_model_input_buffer))
  {
    Serial.printf("[%lus] Failed to extract MFCC features.\n", t_seconds);
    return;
  }
  unsigned long featureTime = micros() - featureStart;

  // Debug telemetry to verify that mic signal and MFCC input are changing
  float rms = 0.0f, peak = 0.0f, dc = 0.0f;
  audioProcessor.getAudioStats(rms, peak, dc);

  int q_min = 127;
  int q_max = -128;
  long q_abs_sum = 0;
  for (int i = 0; i < NUM_INPUTS; i++)
  {
    int q = (int)g_model_input_buffer[i];
    if (q < q_min)
      q_min = q;
    if (q > q_max)
      q_max = q;
    q_abs_sum += abs(q - MODEL_INPUT_ZERO_POINT);
  }
  float q_abs_mean = (float)q_abs_sum / (float)NUM_INPUTS;

  // 2. Run inference
  unsigned long startTime = micros();
  auto predictStatus = tf.predict(g_model_input_buffer);
  unsigned long inferenceTime = micros() - startTime;

  if (!predictStatus.isOk())
  {
    Serial.printf("[%lus] Inference failed: %s\n", t_seconds, predictStatus.toString().c_str());
    return;
  }

  // 3. Dequantize and process results
  int q0 = (int)round(tf.output(0));
  int q1 = (int)round(tf.output(1));

  // Dequantize
  float p0 = (q0 - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
  float p1 = (q1 - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
  float p_cough = (COUGH_INDEX == 0) ? p0 : p1;
  float p_non_cough = (COUGH_INDEX == 0) ? p1 : p0;

  // 4. Display results with timestamp and inference time
  Serial.printf("[%lus] Window:[%lu-%lu]s Feature:%lu us Inference:%lu us Class0:%.4f Class1:%.4f Probabilities:[Cough:%.4f Non-Cough:%.4f]\n",
                t_seconds, window_start_seconds, window_end_seconds,
                featureTime, inferenceTime, p0, p1, p_cough, p_non_cough);
  Serial.printf("      Audio[RMS=%.6f Peak=%.6f DC=%.6f] MFCC[qmin=%d qmax=%d mean|q-zp|=%.2f]\n",
                rms, peak, dc, q_min, q_max, q_abs_mean);

  if (peak < 0.003f || q_abs_mean < 0.75f)
  {
    Serial.println("      WARNING: Mic/features look too flat. Check INMP441 L/R pin, wiring, and I2S format.");
  }

  // 5. Detection logic (single threshold)
  if (p_cough >= COUGH_THRESHOLD)
  {
    Serial.println("  🚨🚨🚨🚨🚨🚨🚨 COUGH DETECTED! 🚨🚨🚨🚨🚨🚨🚨");

    // Add your actions here
  }
}
