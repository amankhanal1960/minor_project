#include "audio_processor.h"
#include <Arduino.h>
#include <esp_dsp.h> // for optimized fft

// ==================== GLOBALS ==================================
arduinoFFT *MFCCExtractor::fft = nullptr;

// ===================== MFCC EXTRACTOR IMPLEMENTATION ===========

MFCCExtractor::MFCCExtractor()
    : window(nullptr), mel_filters(nullptr), dct_matrix(nullptr),
      buffer_head(0), buffer_ready(false)
{
  memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
}

MFCCExtractor::~MFCCExtractor()
{
  if (window)
    free(window);
  if (mel_filters)
    free(mel_filters);
  if (dct_matrix)
    free(dct_matrix);
  if (fft)
    delete fft;
}

bool MFCCExtractor::begin()
{
  Serial.println("Initializing MFCC Extractor...");

  if (!fft)
  {
    fft = new arduinoFFt();
    Serial.println("FFT initialized");
  }

  // Creating hte hanning window (librosa default)
  window = (float *)malloc(N_FFT * sizeof(float));
  if (!window)
  {
    Serial.println("Failed to allocate window");
    return false;
  }
  // Hanning window: 0.5 - 0.5 * cos(2pien/(n-2))
  for (int i = 0; i < N_FFT, i++)
  {
    window[i] = 0.5f - 0.5f * cos(2.0f * M_PI * i / (N_FFT - 1));
  }
  // Creating the mel filterbanks
  mel_filters = (float *)malloc(NUM_MEL_FILTERS * (N_FFT / 2 + 1) * sizeof(float));
  if (!mel_filters)
  {
    Serial.println("Failed to allocate mel filters");
    return false;
  }
  createMelFilterBank();

  // Create DCT matrix (Type II with ortho normalization)
  dct_matrix = (float *)malloc(NUM_MFCCS * NUM_MEL_FILTERS * sizeof(float));
  if (!dct_matrix)
  {
    Serial.println("Falied to allocate DCT matrix");
    return false;
  }
  createDCTmatrix();

  // INitialize circular buffer
  resetBuffer();

  Serial.println("MFCC Extractor ready: %d frames, %d coefficients\n",
                 NUM_FRAMES < NUM_MFCCS);
  return true;
}

void MFCCExtractor::createMelFilterBank()
{
  // librosa default is 0hz to sr/2 = 8000hz
  float low_freq = 0.0f;
  float high_freq = SAMPLE_RATE / 2.0f;

  // convert to mel scale
  float low_mel = hzToMel(low_freq);
  float high_mel = hzToMel(high_freq);

  // creating the equally spaced points in mel scale
  // adding 2 for triangular filters
  float mel_points[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++)
  {
    mel_points[i] = low_mel + (high_mel - low_mel) * i / (NUM_MEL_FILTERS + 1);
  }
  // converting the mel filter s back to frequency bins(FFT bins)
  float bin_points[NUM_MEL_FILTERS + 2];
  for (int i = 0; i < NUM_MEL_FILTERS + 2; i++)
  {
    float hz = melToHz(mel_points[i]);
    bin_points[i] = floorf((N_FFT + 1) * hz / SAMPLE_RATE);
  }

  // creating triangular filters (as used by librosa)
  memset(mel_filters, 0, NUM_MEL_FILTERS * (N_FFT / 2 + 1) * sizeof(float));

  for (int m = 1; m <= NUM_MEL_FILTERS; m++)
  {
    int left = (int)bin_points[m - 1];
    int center = (int)bin_points[m];
    int right = (int)bin_points[m + 1];

    for (int k = left; k < center; k++)
    {
      if (k >= 0 && k <= N_FFT / 2)
      {
        mel_filters[(m - 1) * (N_FFT / 2 + 1) + k] = (k - left) / (float)(center - left);
      }
    }
    for (int k = center; k <= right; k++)
    {
      if (k >= 0 && k <= N_FFT / 2)
      {
        mel_filters[(m - 1) * (N_FFT / 2 + 1) + k] = (right - k) / (float)(right - center);
      }
    }
  }
  Serial.println("Mel filterbank create");
}

void MFCCExtractor::createDCTMatrix()
{
  // Librosa uses DCT type II with ortho normalization
  for (int i = 0; i < NUM_MFCCS; i++)
  {
    for (int j = 0; j < NUM_MEL_FILTERS; j++)
    {
      float val = cosf(M_PI * i * (2 * j + 1) / (2.0f * NUM_MEL_FILTERS));

      // Apply orthogonal normalization
      if (i == 0)
      {
        val *= sqrtf(1.0f / NUM_MEL_FILTERS);
      }
      else
      {
        val *= sqrtf(2.0f / NUM_MEL_FILTERS);
      }

      dct_matrix[i * NUM_MEL_FILTERS + j] = val;
    }
  }
  Serial.println("DCT matrix created");
}

void MFCCExtractor::processFrame(const float *frame, float *mfcc_output)
{
  // Apply pre-emphasis (librosa default: α=0.97)
  static const float PREEMPHASIS = 0.97f;
  float processed_frame[N_FFT];

  // Pre-emphasis
  processed_frame[0] = frame[0] * (1.0f - PREEMPHASIS);
  for (int i = 1; i < N_FFT; i++)
  {
    processed_frame[i] = frame[i] - PREEMPHASIS * frame[i - 1];
  }

  // Apply window
  for (int i = 0; i < N_FFT; i++)
  {
    processed_frame[i] *= window[i];
  }

  // Prepare FFT input (real + imaginary)
  float fft_input[N_FFT * 2];
  for (int i = 0; i < N_FFT; i++)
  {
    fft_input[2 * i] = processed_frame[i]; // Real
    fft_input[2 * i + 1] = 0.0f;           // Imaginary
  }

  // Perform FFT (in-place)
  fft->compute(fft_input, N_FFT, FFT_FORWARD);

  // Compute power spectrum (magnitude squared)
  float spectrum[N_FFT / 2 + 1];
  for (int k = 0; k <= N_FFT / 2; k++)
  {
    float real = fft_input[2 * k];
    float imag = fft_input[2 * k + 1];
    spectrum[k] = (real * real + imag * imag);

    // Add epsilon to avoid log(0)
    if (spectrum[k] < 1e-10f)
      spectrum[k] = 1e-10f;
  }

  // Apply mel filterbank
  float mel_energies[NUM_MEL_FILTERS];
  memset(mel_energies, 0, NUM_MEL_FILTERS * sizeof(float));

  for (int m = 0; m < NUM_MEL_FILTERS; m++)
  {
    for (int k = 0; k <= N_FFT / 2; k++)
    {
      mel_energies[m] += spectrum[k] * mel_filters[m * (N_FFT / 2 + 1) + k];
    }
    // Natural log (librosa uses natural log for DCT)
    mel_energies[m] = logf(mel_energies[m]);
  }

  // Apply DCT to get MFCCs
  for (int i = 0; i < NUM_MFCCS; i++)
  {
    mfcc_output[i] = 0.0f;
    for (int j = 0; j < NUM_MEL_FILTERS; j++)
    {
      mfcc_output[i] += mel_energies[j] * dct_matrix[i * NUM_MEL_FILTERS + j];
    }
  }
}

AudioProcessor::AudioProcessor(int bckPin, int wsPin, int sdPin, i2s_port_t port)
{
  _i2sPort = port;
  _sampleRate = 0;

  _pinConfig.bck_io_num = bckPin;
  _pinConfig.ws_io_num = wsPin;
  _pinConfig.data_out_num = I2S_PIN_NO_CHANGE;
  _pinConfig.data_in_num = sdPin;
}

bool AudioProcessor::begin(int sampleRate)
{

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
      .fixed_mclk = 0};

  esp_err_t err = i2s_driver_install(_i2sPort, &i2sConfig, 0, NULL);
  if (err != ESP_OK)
  {
    Serial.printf("Failed installing I2S driver: %d\n", err);
    return false;
  }

  err = i2s_set_pin(_i2sPort, &_pinConfig);
  if (err != ESP_OK)
  {
    Serial.printf("Failed setting I2S pins: %d\n", err);
    i2s_driver_uninstall(_i2sPort);
    return false;
  }
  // clearing the dma buffer
  i2s_zero_dma_buffer(_i2sPort);
  Serial.println("I2S for INMP441 initialized.");
  return true;
}

int AudioProcessor::read(int32_t *buffer, int bufferLength)
{
  size_t bytesRead = 0;

  // Request to read 'bufferLength' samples (each sample is 4 bytes)
  esp_err_t err = i2s_read(_i2sPort,
                           (void *)buffer,
                           bufferLength * sizeof(int32_t),
                           &bytesRead,
                           portMAX_DELAY);

  if (err != ESP_OK)
  {
    Serial.printf("I2S read error: %d\n", err);
    return 0;
  }

  // Return number of samples actually read
  return (int)(bytesRead / sizeof(int32_t));
}