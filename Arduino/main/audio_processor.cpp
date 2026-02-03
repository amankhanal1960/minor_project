#include "audio_processor.h"
#include <Arduino.h>

// ===== MFCC EXTRACTOR IMPLEMENTATION =====
MFCCExtractor::MFCCExtractor()
    : window(nullptr), mel_filters(nullptr), dct_matrix(nullptr),
      buffer_head(0), buffer_ready(false),
      fft(vReal, vImag, N_FFT, SAMPLE_RATE) // Initialize FFT here
{
    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
    memset(vReal, 0, sizeof(vReal));
    memset(vImag, 0, sizeof(vImag));
}

MFCCExtractor::~MFCCExtractor()
{
    if (window)
        free(window);
    if (mel_filters)
        free(mel_filters);
    if (dct_matrix)
        free(dct_matrix);
    // No need to delete fft - it's not a pointer
}

bool MFCCExtractor::begin()
{
    Serial.println("Initializing MFCC Extractor...");

    // Create Hanning window (librosa default)
    window = (float *)malloc(N_FFT * sizeof(float));
    if (!window)
    {
        Serial.println("Failed to allocate window");
        return false;
    }

    // Hanning window: 0.5 - 0.5 * cos(2πn/(N-1))
    for (int i = 0; i < N_FFT; i++)
    {
        window[i] = 0.5f - 0.5f * cosf(2.0f * M_PI * i / (N_FFT - 1));
    }

    // Create mel filterbank
    int mel_filter_size = NUM_MEL_FILTERS * (N_FFT / 2 + 1);
    mel_filters = (float *)malloc(mel_filter_size * sizeof(float));
    if (!mel_filters)
    {
        Serial.println("Failed to allocate mel filters");
        free(window);
        return false;
    }
    createMelFilterBank();

    // Create DCT matrix (Type II with ortho normalization)
    int dct_size = NUM_MFCCS * NUM_MEL_FILTERS;
    dct_matrix = (float *)malloc(dct_size * sizeof(float));
    if (!dct_matrix)
    {
        Serial.println("Failed to allocate DCT matrix");
        free(window);
        free(mel_filters);
        return false;
    }
    createDCTMatrix();

    // Initialize circular buffer
    resetBuffer();

    Serial.printf("MFCC Extractor ready: %d frames, %d coefficients\n",
                  NUM_FRAMES, NUM_MFCCS);
    return true;
}

void MFCCExtractor::createMelFilterBank()
{
    // Librosa default: 0Hz to sr/2 (8000Hz)
    float low_freq = 0.0f;
    float high_freq = SAMPLE_RATE / 2.0f; // 8000Hz

    // Convert to Mel scale
    float low_mel = hzToMel(low_freq);
    float high_mel = hzToMel(high_freq);

    // Create equally spaced points in Mel scale
    float mel_points[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++)
    {
        mel_points[i] = low_mel + (high_mel - low_mel) * i / (NUM_MEL_FILTERS + 1);
    }

    // Convert Mel points back to Hz and then to FFT bins
    float bin_points[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++)
    {
        float hz = melToHz(mel_points[i]);
        bin_points[i] = floorf((N_FFT + 1) * hz / SAMPLE_RATE);
    }

    // Create triangular filters (HTK style as used by librosa)
    int half_fft = N_FFT / 2 + 1;
    memset(mel_filters, 0, NUM_MEL_FILTERS * half_fft * sizeof(float));

    for (int m = 1; m <= NUM_MEL_FILTERS; m++)
    {
        int left = (int)bin_points[m - 1];
        int center = (int)bin_points[m];
        int right = (int)bin_points[m + 1];

        for (int k = left; k < center; k++)
        {
            if (k >= 0 && k < half_fft)
            {
                mel_filters[(m - 1) * half_fft + k] = (k - left) / (float)(center - left);
            }
        }

        for (int k = center; k <= right; k++)
        {
            if (k >= 0 && k < half_fft)
            {
                mel_filters[(m - 1) * half_fft + k] = (right - k) / (float)(right - center);
            }
        }
    }

    Serial.println("Mel filterbank created");
}

void MFCCExtractor::createDCTMatrix()
{
    // Librosa uses DCT type II with ortho normalization
    // Formula matches scipy.fftpack.dct with norm='ortho'
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

    // Copy to FFT input buffers
    for (int i = 0; i < N_FFT; i++)
    {
        vReal[i] = (double)processed_frame[i];
        vImag[i] = 0.0;
    }

    // Perform FFT (in-place) - using arduinoFFT like the example
    fft.compute(FFTDirection::Forward);

    // Convert to magnitude spectrum
    fft.complexToMagnitude();

    // Compute power spectrum (magnitude squared)
    int half_fft = N_FFT / 2 + 1;
    float spectrum[half_fft];
    for (int k = 0; k < half_fft; k++)
    {
        double magnitude = vReal[k];
        spectrum[k] = (float)(magnitude * magnitude);

        // Add epsilon to avoid log(0)
        if (spectrum[k] < 1e-10f)
            spectrum[k] = 1e-10f;
    }

    // Apply mel filterbank
    float mel_energies[NUM_MEL_FILTERS];
    memset(mel_energies, 0, NUM_MEL_FILTERS * sizeof(float));

    for (int m = 0; m < NUM_MEL_FILTERS; m++)
    {
        for (int k = 0; k < half_fft; k++)
        {
            mel_energies[m] += spectrum[k] * mel_filters[m * half_fft + k];
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

void MFCCExtractor::standardizeMFCCs(float *mfccs)
{
    // Per-coefficient standardization across all frames
    // Exactly like Python: (mfcc - mean) / (std + eps)
    const float EPSILON = 1e-9f;

    // Calculate mean for each coefficient
    float means[NUM_MFCCS] = {0};
    for (int c = 0; c < NUM_MFCCS; c++)
    {
        for (int f = 0; f < NUM_FRAMES; f++)
        {
            means[c] += mfccs[f * NUM_MFCCS + c];
        }
        means[c] /= NUM_FRAMES;
    }

    // Calculate standard deviation for each coefficient
    float stddevs[NUM_MFCCS] = {0};
    for (int c = 0; c < NUM_MFCCS; c++)
    {
        float variance = 0.0f;
        for (int f = 0; f < NUM_FRAMES; f++)
        {
            float diff = mfccs[f * NUM_MFCCS + c] - means[c];
            variance += diff * diff;
        }
        stddevs[c] = sqrtf(variance / NUM_FRAMES);
    }

    // Apply standardization
    for (int c = 0; c < NUM_MFCCS; c++)
    {
        if (stddevs[c] < EPSILON)
            stddevs[c] = EPSILON; // Avoid division by zero

        for (int f = 0; f < NUM_FRAMES; f++)
        {
            mfccs[f * NUM_MFCCS + c] =
                (mfccs[f * NUM_MFCCS + c] - means[c]) / (stddevs[c] + EPSILON);
        }
    }
}

bool MFCCExtractor::extractMFCC(const float *audio, float *mfcc_output)
{
    // Verify input length (must be exactly 9 seconds)
    int expected_samples = ANALYSIS_SAMPLES; // 144000
    int actual_samples = N_FFT + (NUM_FRAMES - 1) * HOP_LENGTH;

    if (actual_samples != expected_samples)
    {
        Serial.printf("ERROR: Need %d samples for %d frames, got %d\n",
                      expected_samples, NUM_FRAMES, actual_samples);
        return false;
    }

    unsigned long start_time = micros();

    // Process each frame
    for (int frame = 0; frame < NUM_FRAMES; frame++)
    {
        int start_sample = frame * HOP_LENGTH;

        // Extract frame and process
        float frame_data[N_FFT];
        for (int i = 0; i < N_FFT; i++)
        {
            frame_data[i] = audio[start_sample + i];
        }

        // Get MFCC for this frame
        float frame_mfcc[NUM_MFCCS];
        processFrame(frame_data, frame_mfcc);

        // Store in output array
        for (int c = 0; c < NUM_MFCCS; c++)
        {
            mfcc_output[frame * NUM_MFCCS + c] = frame_mfcc[c];
        }
    }

    // Standardize across all frames
    standardizeMFCCs(mfcc_output);

    unsigned long processing_time = micros() - start_time;
    Serial.printf("MFCC extraction: %lu µs (%.1f ms)\n",
                  processing_time, processing_time / 1000.0f);

    return true;
}

void MFCCExtractor::resetBuffer()
{
    buffer_head = 0;
    buffer_ready = false;
    memset(mfcc_buffer, 0, sizeof(mfcc_buffer));
}

bool MFCCExtractor::addAudioFrame(const float *frame_512_samples, int frame_index)
{
    // This is a simplified version - you'd need to buffer N_FFT samples
    // For now, we'll just process frames when we have full context
    if (frame_index < NUM_FRAMES)
    {
        // For incremental processing, you need to maintain a buffer
        // of N_FFT samples and update it with HOP_LENGTH new samples

        // Simplified: just store placeholder
        for (int c = 0; c < NUM_MFCCS; c++)
        {
            mfcc_buffer[buffer_head][c] = 0.0f; // Placeholder
        }
        buffer_head = (buffer_head + 1) % NUM_FRAMES;

        if (frame_index == NUM_FRAMES - 1)
        {
            buffer_ready = true;
        }
        return true;
    }
    return false;
}

void MFCCExtractor::getCurrentMFCC(float *mfcc_output)
{
    if (!buffer_ready)
    {
        Serial.println("MFCC buffer not ready");
        return;
    }

    // Copy from circular buffer in correct order
    for (int f = 0; f < NUM_FRAMES; f++)
    {
        int src_idx = (buffer_head + f) % NUM_FRAMES;
        for (int c = 0; c < NUM_MFCCS; c++)
        {
            mfcc_output[f * NUM_MFCCS + c] = mfcc_buffer[src_idx][c];
        }
    }

    // Standardize
    standardizeMFCCs(mfcc_output);
}

// ===== AUDIO PROCESSOR IMPLEMENTATION =====
AudioProcessor::AudioProcessor(int bckPin, int wsPin, int sdPin, i2s_port_t port)
    : _audio_buffer(nullptr), _i2sPort(port), _sampleRate(0),
      _write_index(0), _samples_collected(0),
      _frame_position(0), _hop_counter(0)
{
    _pinConfig.bck_io_num = bckPin;
    _pinConfig.ws_io_num = wsPin;
    _pinConfig.data_out_num = I2S_PIN_NO_CHANGE;
    _pinConfig.data_in_num = sdPin;

    memset(_frame_buffer, 0, sizeof(_frame_buffer));
}

AudioProcessor::~AudioProcessor()
{
    i2s_driver_uninstall(_i2sPort);
}

bool AudioProcessor::begin(int sampleRate)
{
    _sampleRate = sampleRate;

    Serial.println("Initializing Audio Processor...");
    Serial.printf("Sample rate: %d Hz\n", _sampleRate);
    Serial.printf("Analysis window: %d seconds\n", ANALYSIS_SECONDS);
    Serial.printf("Buffer size: %d samples\n", BUFFER_SAMPLES);

    // Initialize MFCC extractor
    if (!_mfcc_extractor.begin())
    {
        Serial.println("Failed to initialize MFCC extractor");
        return false;
    }

    // Configure I2S
    i2s_config_t i2sConfig = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = _sampleRate,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0};

    // Install I2S driver
    esp_err_t err = i2s_driver_install(_i2sPort, &i2sConfig, 0, NULL);
    if (err != ESP_OK)
    {
        Serial.printf("Failed installing I2S driver: %d\n", err);
        return false;
    }

    // Set I2S pins
    err = i2s_set_pin(_i2sPort, &_pinConfig);
    if (err != ESP_OK)
    {
        Serial.printf("Failed setting I2S pins: %d\n", err);
        i2s_driver_uninstall(_i2sPort);
        return false;
    }

    // Clear DMA buffer
    i2s_zero_dma_buffer(_i2sPort);

    Serial.println("Audio Processor initialized successfully");
    return true;
}

float AudioProcessor::convertI2SToFloat(int32_t i2s_sample)
{
    // INMP441: 24-bit audio in 32-bit container
    // Extract 24-bit value (right-shift 8 bits)
    int32_t audio_24bit = i2s_sample >> 8;

    // Sign extend if negative (24-bit signed to 32-bit signed)
    if (audio_24bit & 0x00800000)
    {
        audio_24bit |= 0xFF000000;
    }

    // Convert to float in range [-1.0, 1.0]
    // 24-bit signed range: -8388608 to 8388607
    return (float)audio_24bit / 8388608.0f;
}

void AudioProcessor::quantizeMFCC(const float *mfcc_float, int8_t *mfcc_int8)
{
    // Quantize using model's exact parameters
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        // Formula: quantized = round(value / scale + zero_point)
        float scaled = mfcc_float[i] / MODEL_INPUT_SCALE;
        int32_t quantized = (int32_t)roundf(scaled + MODEL_INPUT_ZERO_POINT);

        // Clamp to int8 range (safety)
        if (quantized > 127)
            quantized = 127;
        if (quantized < -128)
            quantized = -128;

        mfcc_int8[i] = (int8_t)quantized;
    }
}

int AudioProcessor::read(int32_t *buffer, int bufferLength)
{
    size_t bytesRead = 0;

    // Read I2S data (blocking)
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

    int samplesRead = (int)(bytesRead / sizeof(int32_t));

    // Convert and store in circular buffer
    for (int i = 0; i < samplesRead; i++)
    {
        float sample = convertI2SToFloat(buffer[i]);

        // Store in circular buffer
        _audio_buffer[_write_index] = sample;
        _write_index = (_write_index + 1) % BUFFER_SAMPLES;
        _samples_collected++;

        // Keep track of the most recent samples
        if (_samples_collected > BUFFER_SAMPLES)
        {
            _samples_collected = BUFFER_SAMPLES;
        }
    }

    return samplesRead;
}

bool AudioProcessor::getMFCCFeatures(int8_t *mfcc_output)
{
    // Check if we have enough audio collected
    if (_samples_collected < ANALYSIS_SAMPLES)
    {
        Serial.printf("Not enough audio: %d/%d samples\n",
                      _samples_collected, ANALYSIS_SAMPLES);
        return false;
    }

    // Get the most recent 9 seconds of audio
    float recent_audio[ANALYSIS_SAMPLES];

    // Calculate start index (9 seconds before current write position)
    int start_idx = _write_index - ANALYSIS_SAMPLES;
    if (start_idx < 0)
    {
        start_idx += BUFFER_SAMPLES;
    }

    // Copy audio from circular buffer
    for (int i = 0; i < ANALYSIS_SAMPLES; i++)
    {
        int idx = (start_idx + i) % BUFFER_SAMPLES;
        recent_audio[i] = _audio_buffer[idx];
    }

    // Extract MFCC features (float, standardized)
    float mfcc_float[NUM_INPUTS];
    if (!_mfcc_extractor.extractMFCC(recent_audio, mfcc_float))
    {
        Serial.println("MFCC extraction failed");
        return false;
    }

    // Quantize to int8
    quantizeMFCC(mfcc_float, mfcc_output);

    // Debug: print first few values
    Serial.print("First 5 quantized MFCCs: ");
    for (int i = 0; i < 5; i++)
    {
        Serial.printf("%d ", mfcc_output[i]);
    }
    Serial.println();

    return true;
}

bool AudioProcessor::getMFCCFeaturesFromAudio(const float *audio, int audio_length, int8_t *mfcc_output)
{
    if (audio_length != ANALYSIS_SAMPLES)
    {
        Serial.printf("ERROR: Need %d samples, got %d\n",
                      ANALYSIS_SAMPLES, audio_length);
        return false;
    }

    // Extract MFCC features
    float mfcc_float[NUM_INPUTS];
    if (!_mfcc_extractor.extractMFCC(audio, mfcc_float))
    {
        return false;
    }

    // Quantize to int8
    quantizeMFCC(mfcc_float, mfcc_output);

    return true;
}

void AudioProcessor::processIncremental(const int32_t *i2s_samples, int count)
{
    for (int i = 0; i < count; i++)
    {
        // Convert to float
        float sample = convertI2SToFloat(i2s_samples[i]);

        // Store in circular buffer
        _audio_buffer[_write_index] = sample;
        _write_index = (_write_index + 1) % BUFFER_SAMPLES;
        _samples_collected++;

        // Update frame buffer for incremental processing
        _frame_buffer[_frame_position] = sample;
        _frame_position++;

        // When we have HOP_LENGTH samples, process a hop
        if (_frame_position >= HOP_LENGTH)
        {
            // Shift frame buffer (keep last N_FFT samples)
            for (int j = 0; j < N_FFT - HOP_LENGTH; j++)
            {
                _frame_buffer[j] = _frame_buffer[j + HOP_LENGTH];
            }
            _frame_position = N_FFT - HOP_LENGTH;

            // Increment hop counter
            _hop_counter++;

            // For incremental MFCC, you would process the frame here
            // This is simplified - you'd need a more complex state machine
        }
    }
}

bool AudioProcessor::getCurrentMFCC(int8_t *mfcc_output)
{
    // This is for incremental mode - get MFCC from current state
    if (_samples_collected < ANALYSIS_SAMPLES)
    {
        return false;
    }

    // For now, fall back to batch processing
    return getMFCCFeatures(mfcc_output);
}

void AudioProcessor::getAudioStats(float &rms, float &peak, float &dc_offset)
{
    if (_samples_collected == 0)
    {
        rms = 0;
        peak = 0;
        dc_offset = 0;
        return;
    }

    // Calculate statistics on the most recent second
    int num_samples = min(_samples_collected, SAMPLE_RATE);
    int start_idx = _write_index - num_samples;
    if (start_idx < 0)
        start_idx += BUFFER_SAMPLES;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    float max_val = 0.0f;

    for (int i = 0; i < num_samples; i++)
    {
        int idx = (start_idx + i) % BUFFER_SAMPLES;
        float val = _audio_buffer[idx];

        sum += val;
        sum_sq += val * val;
        if (fabs(val) > max_val)
            max_val = fabs(val);
    }

    dc_offset = sum / num_samples;
    rms = sqrtf(sum_sq / num_samples);
    peak = max_val;
}

void AudioProcessor::logAudioStats()
{
    float rms, peak, dc_offset;
    getAudioStats(rms, peak, dc_offset);

    Serial.printf("Audio stats: RMS=%.4f, Peak=%.4f, DC=%.4f\n",
                  rms, peak, dc_offset);
}