
#include <stdint.h>
#include <Arduino.h>
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#include "model_data.h"
#include "audio_processor.h"


#define I2S_SD_PIN 10 // mic SD -> GPIO 7
#define I2S_SCK_PIN 12// mic SCK -> GPIO 8
#define I2S_WS_PIN 11// Mic WS -> GPIO 9

// the input size 280 samples *40 mfccs
#define NUMBER_OF_INPUTS 11200
// tensor arena is the number of bytes reserved for the TFML scratch memory
#define TENSOR_ARENA_SIZE (64 * 1024)

const float OUTPUT_SCALE = 0.00390625f;
const int   OUTPUT_ZERO_POINT = -128;

//Eloquent::TF::Sequential is the helper class wrapping TFML
// 30 is the library limit, max number of nodes/ops/commands allowed by the wrapper
Eloquent::TF::Sequential<30, TENSOR_ARENA_SIZE> tf;

AudioProcessor audio(I2S_SCK_PIN, I2S_WS_PIN, I2S_SD_PIN, I2S_NUM_0);

void setup()
{
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n----- ESP32-S3 Cough Detection Model Integration -----\n");

  // informs the wrapper how many input element and how many output classes are
  tf.setNumInputs(NUMBER_OF_INPUTS);
  tf.setNumOutputs(2);


  // Add...() calls register those ops so TFML knows how to execute the model
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
  // tf.begin() loads the .tflite and initializes the interpreter using the registered 
  //ops and tensor arena
  auto status = tf.begin(cough_cnn_int8_tflite);

  if (!status.isOk())
  {
    Serial.print("MODEL LOAD FAILED: ");
    //this usually gives a clear reason (missing op, arena too small, etc.).
    Serial.println(status.toString());
    while (1)
      ;
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

  // ---------- Dummy inference ----------
  // static int8_t dummyInput[NUMBER_OF_INPUTS] = {0};

  // Serial.println("\nRunning inference...");

  // unsigned long startTime = micros();
  // auto predictStatus = tf.predict(dummyInput);
  // unsigned long inferenceTime = micros() - startTime;

  // if (!predictStatus.isOk())
  // {
  //   Serial.print("INFERENCE FAILED: ");
  //   Serial.println(predictStatus.toString());
  //   return;
  // }

  // Serial.print("Inference time: ");
  // Serial.print(inferenceTime);
  // Serial.println(" microseconds");

  // int q0 = (int) round(tf.output(0));
  // int q1 = (int) round(tf.output(1));

  // // Dequantize to real probabilities
  // float p0 = (q0 - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
  // float p1 = (q1 - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;

  // Serial.print("Quantized output: ");
  // Serial.print(q0);
  // Serial.print(", ");
  // Serial.println(q1);

  // Serial.print("Probabilities: ");
  // Serial.print(p0, 6);
  // Serial.print(", ");
  // Serial.println(p1, 6);
}

void loop() {
    static int32_t i2sBuffer[1024];
    int samplesRead = audio.read(i2sBuffer, 1024);
    
    if (samplesRead > 0) {
        // Analyze first 10 samples in detail
        Serial.println("\n=== RAW I2S DATA ANALYSIS ===");
        for (int i = 0; i < min(10, samplesRead); i++) {
            uint32_t raw = (uint32_t)i2sBuffer[i];
            // 1. Show raw 32-bit value
            Serial.printf("Raw[%d]: 0x%08X", i, raw);
            
            // 2. Show as signed 32-bit integer
            Serial.printf(" | Signed32: %d", (int32_t)raw);
            
            // 3. Extract the 24-bit audio part (shift right 8 bits)
            int32_t audio24bit = (int32_t)(raw >> 8);
            // Proper sign extension for 24-bit
            if (audio24bit & 0x00800000) {
                audio24bit |= 0xFF000000;
            }
            Serial.printf(" | Audio24: %d", audio24bit);
            
            // 4. Calculate DC offset (average of first 10)
            static long long sum = 0;
            static int count = 0;
            if (i == 0) sum = 0, count = 0; // Reset each batch
            sum += audio24bit;
            count++;
            
            Serial.println();
        }
        
        // Quick stats
        int32_t minVal = INT32_MAX, maxVal = INT32_MIN;
        for (int i = 0; i < samplesRead; i++) {
            int32_t val = (int32_t)(i2sBuffer[i] >> 8);
            if (val & 0x00800000) val |= 0xFF000000;
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
        Serial.printf("Range: %d to %d (24-bit scale)\n", minVal, maxVal);
        Serial.println("==============================\n");
    }
    
    delay(2000); // Slow output for readability
}
