#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <driver/i2s.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <arduinoFFT.h>

#define SAMPLE_RATE 16000
#define N_FFT 1024
#define HOP_LENGTH 512
#define NUM_MFCCS 40
#define NUM_MEL_FILTERS 40
#define ANALYSIS_SECONDS 9
#define BUFFER_SECONDS 10
#define BUFFER_SAMPLES (SAMPLE_RATE * BUFFER_SECONDS)

#define NUM_FRAMES 280

#define ANALYSIS_SAMPLES (N_FFT + (NUM_FRAMES - 1) * HOP_LENGTH) 
#define ANALYSIS_SECONDS_FLOAT (ANALYSIS_SAMPLES / (float)SAMPLE_RATE)  

#define NUM_INPUTS (NUM_FRAMES * NUM_MFCCS) // 280 * 40

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

    // Extract the mfcc from 9 second audio (144000 samples)
    // Return float MFCCs(40 x 280)
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

    // Standardization (per_coefficient z-score across 280 frames)
    void standardizeMFCCs(float *mfccs);

    // Helper: Mel Conversion
    float hzToMel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
    float melToHz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

    MFCCExtractor(const MFCCExtractor &) = delete;
    MFCCExtractor &operator=(const MFCCExtractor &) = delete;
};

class AudioProcessor
{
public:
    AudioProcessor(int bckPin, int wsPin, int sdPin,float input_scale = 0.11736483126878738f,
                   int input_zero_point = 2, i2s_port_t port = I2S_NUM_0);
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

    // Convert I2S 24-bit to float [-1, 1]
    float convertI2SToFloat(int32_t i2s_sample);

    // convert float to int8 using model quantization
    void quantizeMFCC(const float *mfcc_float, int8_t *mfcc_int8);

    // Debugging logic
    void logAudioStats();

    // Prevent copying
    AudioProcessor(const AudioProcessor &) = delete;
    AudioProcessor &operator=(const AudioProcessor &) = delete;
};

#endif