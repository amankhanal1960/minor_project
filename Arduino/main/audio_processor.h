#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <arduinoFFT.h>

#define SAMPLE_RATE 16000
#define N_FFT 1024
#define HOP_LENGTH 512
#define NUM_MFCCS 40
// Match librosa.feature.mfcc default mel bank size used in training.
#define NUM_MEL_FILTERS 128
#define ANALYSIS_SECONDS 5
#define BUFFER_SECONDS 8
#define BUFFER_SAMPLES (SAMPLE_RATE * BUFFER_SECONDS)

#define NUM_FRAMES 155

#define ANALYSIS_SAMPLES (N_FFT + (NUM_FRAMES - 1) * HOP_LENGTH) 
#define ANALYSIS_SECONDS_FLOAT (ANALYSIS_SAMPLES / (float)SAMPLE_RATE)  

#define NUM_INPUTS (NUM_FRAMES * NUM_MFCCS) // 155 * 40

// Model quantization parameters (from python model)

#define MODEL_OUTPUT_SCALE 0.00390625f
#define MODEL_OUTPUT_ZERO_POINT -128

// ===================MFCC EXTRACTOR CLASS +++++++++++++++++++++++++++
class MFCCExtractor
{
public:
    MFCCExtractor();
    ~MFCCExtractor();
    
    // Initializes FFT, window, Mel filterbank, DCT matrix, and working buffers.
    bool begin();

    // Extract MFCC from the current analysis window (~5 seconds)
    // Return float MFCCs(NUM_MFCCS x NUM_FRAMES)
    bool extractMFCC(const float *audio, float *mfcc_output);

    // Incremental processing (optimization for sliding window)
    void resetBuffer();
    bool addAudioFrame(const float *frame_512_samples, int frame_index);
    void getCurrentMFCC(float *mfcc_output);

private:
    // Window function (hanning)
    float *window;

    // Mel filterbank and DCT matrix
    float *mel_filters; // [NUM_MEL_FILTERS x (N_FFT/2 + 1)]
    float *dct_matrix;  // [NUM_MFCCS x NUM_MEL_FILTERS]

    // Working buffers moved off the stack to avoid overflow on ESP32-S3
    float *_frame_data;      // [N_FFT]
    float *_processed_frame; // [N_FFT]
    float *_spectrum;        // [N_FFT/2 + 1]
    float *_mel_energies;    // [NUM_MEL_FILTERS]
    float *_frame_mfcc;      // [NUM_MFCCS]

    // FFT buffers
    double vReal[N_FFT];
    double vImag[N_FFT];

    // FFT object - CORRECTED: using double template as in example
    ArduinoFFT<double> fft;

    // circular buffer for incremental processing
    float mfcc_buffer[NUM_FRAMES][NUM_MFCCS];
    int buffer_head;
    bool buffer_ready;

    // Internal methods
    void createMelFilterBank();
    void createDCTMatrix();

    // MFCC processing for a single frame
    void processFrame(const float *frame, float *mfcc_output);

    // Standardization (per-coefficient z-score across NUM_FRAMES frames)
    void standardizeMFCCs(float *mfccs);

    // Helper: Slaney Mel conversion (librosa default, htk=False).
    float hzToMel(float hz)
    {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp; // 15
        const float logstep = logf(6.4f) / 27.0f;

        if (hz < min_log_hz)
            return hz / f_sp;

        return min_log_mel + logf(hz / min_log_hz) / logstep;
    }

    float melToHz(float mel)
    {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp; // 15
        const float logstep = logf(6.4f) / 27.0f;

        if (mel < min_log_mel)
            return f_sp * mel;

        return min_log_hz * expf(logstep * (mel - min_log_mel));
    }

    MFCCExtractor(const MFCCExtractor &) = delete;
    MFCCExtractor &operator=(const MFCCExtractor &) = delete;
};

class AudioProcessor
{
public:
    AudioProcessor(int bckPin, int wsPin, int sdPin,float input_scale = 0.09420423954725266f,
                   int input_zero_point = -1, i2s_port_t port = I2S_NUM_0);
    ~AudioProcessor();
    // Initialize I2S and audio processing
    bool begin(int sampleRate = 16000);

    // Read the raw I2S samples
    int read(int32_t *buffer, int bufferLength);

    // Get quantized MFCC features for model inference
    // Returns true if successful and false if not enough audio
    bool getMFCCFeatures(int8_t *mfcc_output);

    // Get the quantized MFCC from provided audio buffer
    bool getMFCCFeaturesFromAudio(const float *audio, int audio_length, int8_t *mfcc_output);

    // Check if we have enough audio for inference
    bool hasEnoughAudio() { return _samples_collected >= ANALYSIS_SAMPLES; }

    // Get audio statistics for debugging
    void getAudioStats(float &rms, float &peak, float &dc_offset);

    // continuous incremental processing
    void processIncremental(const int32_t *i2s_samples, int count);
    bool getCurrentMFCC(int8_t *mfcc_output);


    void setAudioBuffer(float* external_buffer) {
        _audio_buffer = external_buffer;
    }

    int _samples_collected;

private:
    // I2S configuration
    i2s_pin_config_t _pinConfig;
    i2s_port_t _i2sPort;
    int _sampleRate;
    float _input_scale;
    int _input_zero_point;

    // Audio buffer (circular)
    float* _audio_buffer; 
    int _write_index;

    // MFCC Extractor
    MFCCExtractor _mfcc_extractor;

    // Incremental processing state
    float _frame_buffer[N_FFT];
    int _frame_position;
    int _hop_counter;

    // I2S sample packing probe (detect left-justified vs right-justified 24-bit data)
    bool _i2s_format_locked;
    bool _i2s_left_justified_24bit;
    uint32_t _i2s_probe_count;
    uint32_t _i2s_probe_lsb_zero_count;

    // 2nd-order IIR band-pass (high-pass + low-pass) to suppress rumble and hiss.
    struct BiquadSection
    {
        float b0, b1, b2;
        float a1, a2;
        float z1, z2;
    };

    BiquadSection _bp_highpass;
    BiquadSection _bp_lowpass;
    bool _bandpass_enabled;
    float _bandpass_low_hz;
    float _bandpass_high_hz;

    // Convert I2S 24-bit to float [-1, 1]
    float convertI2SToFloat(int32_t i2s_sample);

    // Band-pass filter helpers
    void initBandpassFilter();
    void configureHighpass(BiquadSection &section, float cutoff_hz, float q);
    void configureLowpass(BiquadSection &section, float cutoff_hz, float q);
    float processBiquad(BiquadSection &section, float x);
    float applyBandpass(float sample);

    // convert float to int8 using model quantization
    void quantizeMFCC(const float *mfcc_float, int8_t *mfcc_int8);

    // Debugging logic
    void logAudioStats();

    // Prevent copying
    AudioProcessor(const AudioProcessor &) = delete;
    AudioProcessor &operator=(const AudioProcessor &) = delete;
};

#endif
