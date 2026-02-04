#include <Arduino.h>
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include "model_data.h"
#include "audio_processor.h"

// ======================= HARDWARE CONFIG =======================
#define I2S_BCK_PIN 18
#define I2S_WS_PIN 19
#define I2S_SD_PIN 20
#define I2S_PORT I2S_NUM_0

// Model quantization parameters - MOVE THESE HERE (not in header)
const float MODEL_INPUT_SCALE = 0.11736483126878738f;
const int MODEL_INPUT_ZERO_POINT = 2;
const float OUTPUT_SCALE = 0.00390625f;
const int OUTPUT_ZERO_POINT = -128;

// Detection threshold
const float COUGH_THRESHOLD = 0.7f;
#define TENSOR_ARENA_SIZE (64 * 1024)


// ======================= GLOBAL OBJECTS & BUFFERS =======================
Eloquent::TF::Sequential<30, TENSOR_ARENA_SIZE> tf;
AudioProcessor audioProcessor(I2S_BCK_PIN, I2S_WS_PIN, I2S_SD_PIN,MODEL_INPUT_SCALE, MODEL_INPUT_ZERO_POINT, I2S_PORT);

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
  Serial.begin(115200);

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
  const unsigned long INFERENCE_INTERVAL_MS = 1000; // Run every second
  const unsigned long STATUS_INTERVAL_MS = 5000;    // Print status every 1s

  unsigned long currentTime = millis();

  // 1. Continuously read audio from I2S microphone
  const int I2S_READ_BUFFER_SIZE = 512;
  static int32_t i2sBuffer[I2S_READ_BUFFER_SIZE];

  int samplesRead = audioProcessor.read(i2sBuffer, I2S_READ_BUFFER_SIZE);

  // 2. Run inference at regular intervals
  if (currentTime - lastInferenceTime >= INFERENCE_INTERVAL_MS)
  {
    lastInferenceTime = currentTime;

    // Check if we have enough audio collected
    if (audioProcessor.hasEnoughAudio())
    {
      // Serial.printf("\n[%lu] Running inference...", currentTime / 1000);
      runInference();
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

  if (psramFound())
  {
    Serial.println(" PSRAM detected by system.");
    Serial.printf("   Total PSRAM: %.2f MB\n", ESP.getPsramSize() / (1024.0 * 1024.0));

    // 1. Allocate main audio buffer in PSRAM
    size_t audio_buffer_bytes = ANALYSIS_SAMPLES * sizeof(float);
    Serial.printf("Allocating audio buffer: %.2f MB...", audio_buffer_bytes / (1024.0 * 1024.0));

    g_audio_buffer_psram = (float *)ps_malloc(audio_buffer_bytes);
    if (g_audio_buffer_psram == nullptr)
    {
      Serial.println(" FAILED!");
      return false;
    }
    Serial.println(" SUCCESS!");
    memset(g_audio_buffer_psram, 0, audio_buffer_bytes);

    // 2. Allocate model input buffer in PSRAM
    size_t model_buffer_bytes = NUM_INPUTS * sizeof(int8_t);
    Serial.printf("Allocating model buffer: %.2f MB...", model_buffer_bytes / (1024.0 * 1024.0));

    g_model_input_buffer = (int8_t *)ps_malloc(model_buffer_bytes);
    if (g_model_input_buffer == nullptr)
    {
      Serial.println(" FAILED!");
      free(g_audio_buffer_psram);
      return false;
    }
    Serial.println(" SUCCESS!");
    memset(g_model_input_buffer, 0, model_buffer_bytes);

    Serial.printf("Free PSRAM after allocation: %.2f MB\n", ESP.getFreePsram() / (1024.0 * 1024.0));
    return true;
  }

  Serial.println(" PSRAM NOT FOUND!");
  return false;
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
  auto status = tf.begin(cough_cnn_int8_tflite);

  if (!status.isOk())
  {
    Serial.print(" Model load failed: ");
    Serial.println(status.toString());
    return false;
  }

  Serial.println(" Model loaded successfully");
  Serial.printf("  Model size: %d bytes\n", cough_cnn_int8_tflite_len);
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

  Serial.printf(" Model warmup complete: %lu µs\n", warmupTime);
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
  Serial.printf("  Buffer samples: %d\n", ANALYSIS_SAMPLES);

  return true;
}

void runInference()
{
  // Timestamp (seconds since boot) for this inference
  unsigned long t_seconds = millis() / 1000;

  // 1. Extract MFCC features directly into PSRAM buffer
  if (!audioProcessor.getMFCCFeatures(g_model_input_buffer))
  {
    Serial.printf("[%lus] Failed to extract MFCC features.\n", t_seconds);
    return;
  }

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
  float p_cough = (q0 - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
  float p_non_cough = (q1 - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;

  // 4. Display results with timestamp and inference time
  Serial.printf("[%lus] Inference time: %lu µs, Probabilities: [Cough: %.4f, Non-Cough: %.4f]\n",
                t_seconds, inferenceTime, p_cough, p_non_cough);

  // 5. Detection logic
  if (p_cough > COUGH_THRESHOLD)
  {
    Serial.println("  ||||||||||||||||||||| COUGH DETECTED! |||||||||||||||||||||");

    // Add your actions here
  }
}

// void printSystemStatus()
// {
//   Serial.println("\n--- SYSTEM STATUS ---");
//   Serial.printf("Uptime: %lu seconds\n", millis() / 1000);
//   Serial.printf("Audio collected: %d/%d samples\n",
//                 audioProcessor._samples_collected, ANALYSIS_SAMPLES);
//   Serial.printf("Free Heap (Internal): %.2f KB\n", ESP.getFreeHeap() / 1024.0);
//   Serial.printf("Free PSRAM: %.2f KB\n", ESP.getFreePsram() / 1024.0);

//   // Optional: Audio statistics
//   float rms, peak, dc;
//   audioProcessor.getAudioStats(rms, peak, dc);
//   Serial.printf("Audio stats: RMS=%.4f, Peak=%.4f, DC=%.4f\n", rms, peak, dc);
// }