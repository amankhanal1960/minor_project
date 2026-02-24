#include <Arduino.h>
#include <driver/i2s.h>

// ====================== USER CONFIG ======================
#define SERIAL_BAUD 460800

#define SAMPLE_RATE 16000
#define CLIP_SECONDS 5
#define SAMPLE_COUNT (SAMPLE_RATE * CLIP_SECONDS)

#define I2S_BCK_PIN 5
#define I2S_WS_PIN 4
#define I2S_SD_PIN 6
#define I2S_PORT I2S_NUM_0

#define I2S_READ_SAMPLES 256

// Protocol: magic[4] + sample_rate(uint32) + sample_count(uint32) + label(uint8) + PCM16 mono
static const uint8_t MAGIC[4] = {'A', 'U', 'D', '0'};

static bool g_i2s_format_locked = false;
static bool g_i2s_left_justified_24bit = true;
static uint32_t g_probe_count = 0;
static uint32_t g_probe_lsb_zero_count = 0;

static int16_t convertI2SToPCM16(int32_t i2s_sample) {
  // Auto-detect if incoming 24-bit data is left-justified or right-justified in 32-bit word.
  if (!g_i2s_format_locked) {
    g_probe_count++;
    if ((i2s_sample & 0xFF) == 0) {
      g_probe_lsb_zero_count++;
    }

    if (g_probe_count >= 2048) {
      g_i2s_left_justified_24bit = (g_probe_lsb_zero_count * 10 >= g_probe_count * 9);
      g_i2s_format_locked = true;
      Serial.printf("I2S sample format: %s-justified 24-bit\n",
                    g_i2s_left_justified_24bit ? "left" : "right");
    }
  }

  int32_t audio_24bit;

  if (g_i2s_left_justified_24bit) {
    audio_24bit = i2s_sample >> 8;
  }
  else {
    audio_24bit = i2s_sample & 0x00FFFFFF;
    if (audio_24bit & 0x00800000) {
      audio_24bit |= 0xFF000000;
    }
  }

  if (audio_24bit > 8388607) audio_24bit = 8388607;
  if (audio_24bit < -8388608) audio_24bit = -8388608;

  return (int16_t)(audio_24bit >> 8);
}


static bool initI2S() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0,
  };

  i2s_pin_config_t pins = {
    .bck_io_num = I2S_BCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN,
  };

  esp_err_t err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("i2s_driver_install failed: %d\n", err);
    return false;
  }

  err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) {
    Serial.printf("i2s_set_pin failed: %d\n", err);
    i2s_driver_uninstall(I2S_PORT);
    return false;
  }

  i2s_zero_dma_buffer(I2S_PORT);
  err = i2s_start(I2S_PORT);
  if (err != ESP_OK) {
    Serial.printf("i2s_start failed: %d\n", err);
    i2s_driver_uninstall(I2S_PORT);
    return false;
  }

  return true;
}

static void sendClip(uint8_t label) {
  // Flush old DMA audio so each clip starts fresh.
  i2s_zero_dma_buffer(I2S_PORT);
  delay(5);

  const uint32_t sample_rate = SAMPLE_RATE;
  const uint32_t sample_count = SAMPLE_COUNT;

  // Header
  Serial.write(MAGIC, 4);
  Serial.write((const uint8_t*)&sample_rate, sizeof(sample_rate));
  Serial.write((const uint8_t*)&sample_count, sizeof(sample_count));
  Serial.write(&label, 1);

  int32_t i2s_raw[I2S_READ_SAMPLES];
  int16_t pcm16[I2S_READ_SAMPLES];

  uint32_t sent_samples = 0;

  while (sent_samples < sample_count) {
    const uint32_t need = sample_count - sent_samples;
    const uint32_t chunk_samples = (need < I2S_READ_SAMPLES) ? need : I2S_READ_SAMPLES;

    size_t bytes_read = 0;
    esp_err_t err = i2s_read(
      I2S_PORT,
      (void*)i2s_raw,
      chunk_samples * sizeof(int32_t),
      &bytes_read,
      portMAX_DELAY
    );

    if (err != ESP_OK) {
      Serial.printf("i2s_read failed: %d\n", err);
      return;
    }

    const uint32_t got_samples = (uint32_t)(bytes_read / sizeof(int32_t));
    if (got_samples == 0) {
      continue;
    }

    for (uint32_t i = 0; i < got_samples; i++) {
      pcm16[i] = convertI2SToPCM16(i2s_raw[i]);
    }

    Serial.write((const uint8_t*)pcm16, got_samples * sizeof(int16_t));
    sent_samples += got_samples;
  }

  Serial.flush();
  Serial.printf("Sent %lu samples (%ds), label=%c\n", (unsigned long)sample_count, CLIP_SECONDS, (char)label);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
    delay(10);
  }

  delay(500);
  Serial.println("ESP32 Audio Collector (5s)");
  Serial.printf("Sample rate: %d Hz, clip: %d seconds (%d samples)\n", SAMPLE_RATE, CLIP_SECONDS, SAMPLE_COUNT);
  Serial.printf("Pins BCK=%d WS=%d SD=%d\n", I2S_BCK_PIN, I2S_WS_PIN, I2S_SD_PIN);

  if (!initI2S()) {
    Serial.println("I2S init failed. Halting.");
    while (1) {
      delay(1000);
    }
  }

  Serial.println("Ready. Send 'C' or 'N' over serial.");
}

void loop() {
  if (Serial.available() <= 0) {
    delay(1);
    return;
  }

  char cmd = (char)Serial.read();
  if (cmd >= 'a' && cmd <= 'z') {
    cmd = (char)(cmd - 'a' + 'A');
  }

  if (cmd == 'C' || cmd == 'N') {
    sendClip((uint8_t)cmd);
  }
}
