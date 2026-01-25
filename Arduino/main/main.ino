#include <stdint.h>
#include <Arduino.h>

#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#include "model_data.h"
#include "audio_processor.h"

// ---------- I2S PINS ----------
#define I2S_SD   7   // Mic SD
#define I2S_SCK  8   // Mic SCK
#define I2S_WS   9   // Mic WS
#define I2S_PORT I2S_NUM_0

// ---------- MODEL ----------
#define NUMBER_OF_INPUTS 11200   // 280 frames * 40 MFCCs
#define TENSOR_ARENA_SIZE (64 * 1024)

// Output quantization params
const float OUTPUT_SCALE = 0.00390625f;
const int OUTPUT_ZERO_POINT = -128;

// TF Lite wrapper
Eloquent::TF::Sequential<30, TENSOR_ARENA_SIZE> tf;

// Audio processor
AudioProcessor audio(
  I2S_SCK,
  I2S_WS,
  I2S_SD,
  I2S_NUM_0
);

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n\n----- ESP32-S3 Cough Detection Model Integration -----\n");

  tf.setNumInputs(NUMBER_OF_INPUTS);
  tf.setNumOutputs(2);

  // Register ops
  tf.resolver.AddExpandDims();
  tf.resolver.AddReshape();
  tf.resolver.AddConv2D();
  tf.resolver.AddMaxPool2D();
  tf.resolver.AddAdd();
  tf.resolver.AddMul();
  tf.resolver.AddMean();
  tf.resolver.AddFullyConnected();
  tf.resolver.AddSoftmax();

  Serial.println("Loading model...");
  auto status = tf.begin(cough_cnn_int8_tflite);

  if (!status.isOk()) {
    Serial.print("MODEL LOAD FAILED: ");
    Serial.println(status.toString());
    while (1);
  }

  Serial.println("Model loaded successfully");
  Serial.print("Model size: ");
  Serial.print(cough_cnn_int8_tflite_len);
  Serial.println(" bytes");

  Serial.print("Tensor arena size: ");
  Serial.print(TENSOR_ARENA_SIZE);
  Serial.println(" bytes");

  Serial.println("Starting microphone...");
  if (!audio.begin(16000)) {
    Serial.println("Audio start FAILED");
    while (1);
  }

  Serial.println("Microphone started");
}

void loop() {
  static int32_t i2sBuf[1024];

  int samples = audio.read(i2sBuf, 1024);
  if (samples <= 0) {
    Serial.println("No mic data");
    delay(500);
    return;
  }

  int32_t peak = 0;
  for (int i = 0; i < samples; i++) {
    int32_t s24 = i2sBuf[i] >> 8;  // extract 24-bit sample
    peak = max(peak, abs(s24));
  }

  Serial.print("Mic peak: ");
  Serial.println(peak);

  delay(300);
}
