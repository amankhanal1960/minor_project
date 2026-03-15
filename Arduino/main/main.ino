#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include "model_data_5s.h"
#include "audio_processor.h"

// ======================= NETWORK / BACKEND CONFIG =======================
// TODO: replace with your Wi‑Fi credentials
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASS = "YOUR_WIFI_PASSWORD";

// TODO: replace with your registered device ID from the backend
const char *DEVICE_ID = "YOUR_DEVICE_ID_FROM_API";

// Backend base URL (no trailing slash)
const char *API_BASE = "http://192.168.0.10:5000/api";
// ======================= END NETWORK CONFIG =======================

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

// ======================= NEW: MPU6050 MOTION GATE CONFIG =======================
// NEW: I2C pins for MPU6050. Change to match your wiring.
// NEW: Keep these different from I2S pins used by the microphone.
#define IMU_I2C_SDA_PIN 8
#define IMU_I2C_SCL_PIN 9
#define IMU_I2C_FREQ_HZ 400000UL
#define MPU6050_ADDR 0x68

// NEW: Motion sampling and fusion thresholds.
#define MOTION_SAMPLE_RATE_HZ 100
#define MOTION_RING_SECONDS 6
#define MOTION_RING_SIZE (MOTION_SAMPLE_RATE_HZ * MOTION_RING_SECONDS)
#define MOTION_FUSION_WINDOW_MS 1200UL
#define ACCEL_MOTION_THRESHOLD_G 0.12f
#define MOTION_CALIB_SAMPLES 200
// ======================= END NEW MOTION GATE CONFIG =======================

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

// ======================= NEW: MOTION PIPELINE STATE =======================
// NEW: Per-sample motion value in g (dynamic acceleration magnitude) with timestamp.
struct MotionSample
{
  float dyn_acc_g;
  uint32_t t_ms;
};

MotionSample g_motion_ring[MOTION_RING_SIZE];
int g_motion_write_idx = 0;
int g_motion_count = 0;
uint32_t g_next_motion_sample_us = 0;
bool g_motion_sensor_ready = false;

// NEW: Runtime accelerometer bias estimated during startup calibration.
float g_acc_bias_x = 0.0f;
float g_acc_bias_y = 0.0f;
float g_acc_bias_z = 0.0f;
// ======================= END NEW MOTION PIPELINE STATE =======================

// ======================= FUNCTION DECLARATIONS =======================
bool initializePSRAM();
bool initializeModel();
bool initializeAudioProcessor();
bool connectWiFi();
bool postDetection(float coughProb, float audioLevel, float motionPeakG);
// ======================= NEW: MOTION PIPELINE DECLARATIONS =======================
bool initializeMotionPipeline();
void updateMotionPipeline();
bool getMotionWindowMetric(uint32_t window_ms, float &peak_dyn_acc_g, float &mean_dyn_acc_g);
bool mpuWriteReg(uint8_t reg, uint8_t value);
bool mpuReadRegs(uint8_t start_reg, uint8_t *buffer, size_t len);
bool readAccelInG(float &ax_g, float &ay_g, float &az_g);
// ======================= END NEW MOTION PIPELINE DECLARATIONS =======================
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

  // ======================= NEW: Decision-level fusion (audio AND motion) =======================
  bool audio_hit = (p_cough >= COUGH_THRESHOLD);
  float motion_peak_dyn_g = 0.0f;
  float motion_mean_dyn_g = 0.0f;
  bool motion_window_ok = getMotionWindowMetric(MOTION_FUSION_WINDOW_MS, motion_peak_dyn_g, motion_mean_dyn_g);
  bool motion_hit = motion_window_ok && (motion_peak_dyn_g >= ACCEL_MOTION_THRESHOLD_G);
  bool fusion_hit = audio_hit && motion_hit;
  // ======================= END NEW: Decision-level fusion (audio AND motion) =======================

  // 4. Display results with timestamp and inference time
  Serial.printf("[%lus] Window:[%lu-%lu]s Feature:%lu us Inference:%lu us Class0:%.4f Class1:%.4f Probabilities:[Cough:%.4f Non-Cough:%.4f]\n",
                t_seconds, window_start_seconds, window_end_seconds,
                featureTime, inferenceTime, p0, p1, p_cough, p_non_cough);
  Serial.printf("      Audio[RMS=%.6f Peak=%.6f DC=%.6f] MFCC[qmin=%d qmax=%d mean|q-zp|=%.2f]\n",
                rms, peak, dc, q_min, q_max, q_abs_mean);
  // NEW: Motion + fusion telemetry for tuning thresholds.
  Serial.printf("      Motion[peak_dyn=%.4fg mean_dyn=%.4fg win=%lums data=%s] Fusion[audio=%s motion=%s final=%s]\n",
                motion_peak_dyn_g, motion_mean_dyn_g, MOTION_FUSION_WINDOW_MS,
                motion_window_ok ? "yes" : "no",
                audio_hit ? "hit" : "miss",
                motion_hit ? "hit" : "miss",
                fusion_hit ? "hit" : "miss");

  if (peak < 0.003f || q_abs_mean < 0.75f)
  {
    Serial.println("      WARNING: Mic/features look too flat. Check INMP441 L/R pin, wiring, and I2S format.");
  }

  // 5. Detection logic (strict fusion gate)
  if (audio_hit && !motion_hit)
  {
    Serial.println("      Audio hit but motion gate failed -> rejected.");
  }

  if (fusion_hit)
  {
    Serial.println("  >>> COUGH DETECTED (AUDIO + MOTION) <<<");

    // Send event to backend
    postDetection(p_cough, peak, motion_peak_dyn_g);
  }
}

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

  // Connect Wi‑Fi early so network is ready when detections occur
  Serial.println("\n[0/4] Connecting Wi‑Fi...");
  if (!connectWiFi())
  {
    Serial.println(" SYSTEM HALTED: Wi‑Fi connection failed.");
    while (1)
    {
      delay(1000);
    }
  }

  // 1. Initialize PSRAM and allocate buffers
  Serial.println("\n[1/4] Initializing PSRAM...");
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
  Serial.println("\n[2/4] Initializing TensorFlow Lite Model...");
  if (!initializeModel())
  {
    Serial.println(" SYSTEM HALTED: Model initialization failed.");
    while (1)
    {
      delay(1000);
    }
  }

  // 3. Initialize Audio Processor
  Serial.println("\n[3/4] Initializing Audio Processor...");
  if (!initializeAudioProcessor())
  {
    Serial.println(" SYSTEM HALTED: Audio processor initialization failed.");
    while (1)
    {
      delay(1000);
    }
  }

  // ======================= NEW: Initialize MPU6050 motion pipeline =======================
  Serial.println("\n[4/4] Initializing Motion Pipeline (MPU6050)...");
  if (!initializeMotionPipeline())
  {
    Serial.println(" SYSTEM HALTED: Motion pipeline initialization failed.");
    while (1)
    {
      delay(1000);
    }
  }
  // ======================= END NEW: Initialize MPU6050 motion pipeline =======================

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

  // ======================= NEW: Keep IMU sampled at fixed cadence =======================
  updateMotionPipeline();
  // ======================= END NEW: Keep IMU sampled at fixed cadence =======================

  // 1. Continuously read audio from I2S microphone
  const int I2S_READ_BUFFER_SIZE = 512;
  static int32_t i2sBuffer[I2S_READ_BUFFER_SIZE];

  int samplesRead = audioProcessor.read(i2sBuffer, I2S_READ_BUFFER_SIZE);
  (void)samplesRead;

  // ======================= NEW: Catch up IMU sampling after blocking I2S read =======================
  updateMotionPipeline();
  // ======================= END NEW: Catch up IMU sampling after blocking I2S read =======================

  // 2. Run inference at regular intervals
  if (currentTime - lastInferenceTime >= INFERENCE_INTERVAL_MS)
  {
    // Check if we have enough audio collected
    if (audioProcessor.hasEnoughAudio())
    {
      runInference();
      // Set timestamp after inference so we wait for fresh post-inference audio.
      lastInferenceTime = millis();
    }
  }

  // 3. Periodically print system status
  // if (currentTime - lastStatusTime >= STATUS_INTERVAL_MS)
  // {
  //   lastStatusTime = currentTime;
  //   printSystemStatus();
  // }
}

// ======================= FUNCTION IMPLEMENTATIONS =======================

bool connectWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.printf(" Connecting to Wi-Fi SSID '%s'...\n", WIFI_SSID);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < 15000)
  {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED)
  {
    Serial.printf(" Wi-Fi connected. IP: %s\n", WiFi.localIP().toString().c_str());
    return true;
  }

  Serial.println(" Wi-Fi connection timed out.");
  return false;
}

bool postDetection(float coughProb, float audioLevel, float motionPeakG)
{
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println(" Wi-Fi dropped, reconnecting...");
    if (!connectWiFi())
    {
      Serial.println(" Reconnect failed, skipping POST.");
      return false;
    }
  }

  String url = String(API_BASE) + "/detections";
  String payload;
  payload.reserve(128);
  payload += "{\"deviceId\":\"";
  payload += DEVICE_ID;
  payload += "\",\"coughProbability\":";
  payload += String(coughProb, 4);
  payload += ",\"audioLevel\":";
  payload += String(audioLevel, 4);
  payload += "}";

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");

  int code = http.POST(payload);
  if (code > 0)
  {
    Serial.printf(" POST /detections -> %d\n", code);
  }
  else
  {
    Serial.printf(" POST failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();
  return code == 200 || code == 201;
}

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

// ======================= NEW: MPU6050 MOTION PIPELINE IMPLEMENTATION =======================
bool mpuWriteReg(uint8_t reg, uint8_t value)
{
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission() == 0);
}

bool mpuReadRegs(uint8_t start_reg, uint8_t *buffer, size_t len)
{
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(start_reg);
  if (Wire.endTransmission(false) != 0)
  {
    return false;
  }

  size_t received = Wire.requestFrom((int)MPU6050_ADDR, (int)len, (int)true);
  if (received != len)
  {
    return false;
  }

  for (size_t i = 0; i < len; i++)
  {
    buffer[i] = (uint8_t)Wire.read();
  }

  return true;
}

bool readAccelInG(float &ax_g, float &ay_g, float &az_g)
{
  uint8_t raw[6];
  if (!mpuReadRegs(0x3B, raw, sizeof(raw)))
  {
    return false;
  }

  int16_t ax = (int16_t)((raw[0] << 8) | raw[1]);
  int16_t ay = (int16_t)((raw[2] << 8) | raw[3]);
  int16_t az = (int16_t)((raw[4] << 8) | raw[5]);

  // ±2g range -> 16384 LSB/g
  const float ACC_LSB_PER_G = 16384.0f;
  ax_g = ax / ACC_LSB_PER_G;
  ay_g = ay / ACC_LSB_PER_G;
  az_g = az / ACC_LSB_PER_G;
  return true;
}

bool initializeMotionPipeline()
{
  Serial.println("--- Motion Pipeline Initialization ---");
  Serial.printf("I2C pins: SDA=%d SCL=%d\n", IMU_I2C_SDA_PIN, IMU_I2C_SCL_PIN);

  Wire.begin(IMU_I2C_SDA_PIN, IMU_I2C_SCL_PIN, IMU_I2C_FREQ_HZ);
  Wire.setTimeOut(3);
  delay(20);

  // Verify WHO_AM_I
  uint8_t who_am_i = 0;
  if (!mpuReadRegs(0x75, &who_am_i, 1))
  {
    Serial.println(" Failed to read MPU6050 WHO_AM_I.");
    return false;
  }

  if (who_am_i != 0x68 && who_am_i != 0x69)
  {
    Serial.printf(" Unexpected MPU6050 WHO_AM_I: 0x%02X\n", who_am_i);
    return false;
  }

  // Reset then wake the sensor
  if (!mpuWriteReg(0x6B, 0x80))
  {
    Serial.println(" Failed to reset MPU6050.");
    return false;
  }
  delay(100);
  if (!mpuWriteReg(0x6B, 0x00))
  {
    Serial.println(" Failed to wake MPU6050.");
    return false;
  }
  delay(10);

  // DLPF and sample-rate config: 100 Hz output
  if (!mpuWriteReg(0x1A, 0x03))
  {
    Serial.println(" Failed to set MPU6050 DLPF.");
    return false;
  }
  if (!mpuWriteReg(0x19, 0x09))
  {
    Serial.println(" Failed to set MPU6050 sample divider.");
    return false;
  }

  // Accelerometer ±2g full-scale (best resolution for cough motion gate)
  if (!mpuWriteReg(0x1C, 0x00))
  {
    Serial.println(" Failed to set MPU6050 accel range.");
    return false;
  }

  // Startup calibration while stationary
  Serial.printf("Calibrating accelerometer (%d samples)... keep device still.\n", MOTION_CALIB_SAMPLES);
  float sx = 0.0f, sy = 0.0f, sz = 0.0f;
  for (int i = 0; i < MOTION_CALIB_SAMPLES; i++)
  {
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    if (!readAccelInG(ax, ay, az))
    {
      Serial.println(" Failed reading accelerometer during calibration.");
      return false;
    }
    sx += ax;
    sy += ay;
    sz += az;
    delay(10);
  }

  g_acc_bias_x = sx / MOTION_CALIB_SAMPLES;
  g_acc_bias_y = sy / MOTION_CALIB_SAMPLES;
  g_acc_bias_z = (sz / MOTION_CALIB_SAMPLES) - 1.0f;

  g_motion_write_idx = 0;
  g_motion_count = 0;
  g_next_motion_sample_us = micros() + (1000000UL / MOTION_SAMPLE_RATE_HZ);
  g_motion_sensor_ready = true;

  Serial.printf("MPU6050 ready (WHO_AM_I=0x%02X)\n", who_am_i);
  Serial.printf("Accel bias: x=%.4fg y=%.4fg z=%.4fg\n", g_acc_bias_x, g_acc_bias_y, g_acc_bias_z);
  Serial.printf("Motion gate: window=%lu ms threshold=%.3fg\n",
                MOTION_FUSION_WINDOW_MS, ACCEL_MOTION_THRESHOLD_G);
  return true;
}

void updateMotionPipeline()
{
  if (!g_motion_sensor_ready)
  {
    return;
  }

  uint32_t now_us = micros();
  const uint32_t sample_period_us = 1000000UL / MOTION_SAMPLE_RATE_HZ;

  // Catch-up loop ensures fixed-rate sampling even if loop had blocking work.
  while ((int32_t)(now_us - g_next_motion_sample_us) >= 0)
  {
    g_next_motion_sample_us += sample_period_us;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    if (!readAccelInG(ax, ay, az))
    {
      break;
    }

    // Apply calibration bias and compute dynamic acceleration magnitude.
    ax -= g_acc_bias_x;
    ay -= g_acc_bias_y;
    az -= g_acc_bias_z;
    float acc_mag_g = sqrtf(ax * ax + ay * ay + az * az);
    float dyn_acc_g = fabsf(acc_mag_g - 1.0f);

    g_motion_ring[g_motion_write_idx].dyn_acc_g = dyn_acc_g;
    g_motion_ring[g_motion_write_idx].t_ms = millis();
    g_motion_write_idx = (g_motion_write_idx + 1) % MOTION_RING_SIZE;
    if (g_motion_count < MOTION_RING_SIZE)
    {
      g_motion_count++;
    }

    now_us = micros();
  }
}

bool getMotionWindowMetric(uint32_t window_ms, float &peak_dyn_acc_g, float &mean_dyn_acc_g)
{
  peak_dyn_acc_g = 0.0f;
  mean_dyn_acc_g = 0.0f;

  if (!g_motion_sensor_ready || g_motion_count <= 0)
  {
    return false;
  }

  uint32_t now_ms = millis();
  uint32_t cutoff_ms = (now_ms > window_ms) ? (now_ms - window_ms) : 0;

  int matched = 0;
  float sum = 0.0f;
  for (int i = 0; i < g_motion_count; i++)
  {
    int idx = g_motion_write_idx - 1 - i;
    if (idx < 0)
    {
      idx += MOTION_RING_SIZE;
    }

    const MotionSample &sample = g_motion_ring[idx];
    if (sample.t_ms < cutoff_ms)
    {
      break;
    }

    if (sample.dyn_acc_g > peak_dyn_acc_g)
    {
      peak_dyn_acc_g = sample.dyn_acc_g;
    }
    sum += sample.dyn_acc_g;
    matched++;
  }

  if (matched <= 0)
  {
    return false;
  }

  mean_dyn_acc_g = sum / matched;
  return true;
}
// ======================= END NEW: MPU6050 MOTION PIPELINE IMPLEMENTATION =======================
