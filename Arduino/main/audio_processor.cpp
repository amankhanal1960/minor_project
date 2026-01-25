#include "audio_processor.h"
#include <Arduino.h>


AudioProcessor::AudioProcessor(int bckPin, int wsPin, int sdPin, i2s_port_t port) {
  _i2sPort = port;
  _sampleRate = 0;

  _pinConfig.bck_io_num = bckPin;
  _pinConfig.ws_io_num = wsPin;
  _pinConfig.data_out_num = I2S_PIN_NO_CHANGE;
  _pinConfig.data_in_num = sdPin;
}

bool AudioProcessor::begin(int sampleRate) {

  _sampleRate = sampleRate;

  i2s_config_t i2sConfig = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = _sampleRate,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // INMP441 sends 24 bit in 32 bit
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S, // Standard I2S format
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };


  esp_err_t err = i2s_driver_install(_i2sPort, &i2sConfig, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("Failed installing I2S driver: %d\n", err);
    return false;
  }
  
  err = i2s_set_pin(_i2sPort, &_pinConfig);
  if (err != ESP_OK) {
    Serial.printf("Failed setting I2S pins: %d\n", err);
    i2s_driver_uninstall(_i2sPort);
    return false;
  }
  // clearing the dma buffer
  i2s_zero_dma_buffer(_i2sPort);
  Serial.println("I2S for INMP441 initialized.");
  return true;
}

int AudioProcessor::read(int32_t* buffer, int bufferLength) {
  size_t bytesRead = 0;
  
  // Request to read 'bufferLength' samples (each sample is 4 bytes)
  esp_err_t err = i2s_read(_i2sPort, 
                          (void*)buffer, 
                          bufferLength * sizeof(int32_t), 
                          &bytesRead, 
                          portMAX_DELAY);
                          
  if (err != ESP_OK) {
    Serial.printf("I2S read error: %d\n", err);
    return 0;
  }
  
  // Return number of samples actually read
  return (int)(bytesRead / sizeof(int32_t));
}