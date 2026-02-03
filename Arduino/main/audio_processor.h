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
#define ANALYSIS_SAMPLES (SAMPLE_RATE * ANALYSIS_SECONDS)

//Calculating the expected frames: (total_samples - n_fft) / hop_length + 1
#define NUM_FRAMES ((ANALYSIS_SAMPLES - N_FFT) / HOP_LENGTH + 1)
#define NUM_INPUTS (NUM_FRAMES * NUM_MFCCS) // 280 * 40

//Model quantization parameters (from python model)
#define MODEL_INPUT_SCALE 0.11736483126878738f
#define MODEL_INPUT_ZERO_POINT 2
#define MODEL_OUTPUT_SCALE  0.00390625f
#define MODEL_OUTPUT_ZERO_POINT -128

// ===================MFCC EXTRACTOR CLASS +++++++++++++++++++++++++++
class MFCCExtractor {
  public:
    MFCCExtractor();
    ~MFCCExtractor();

    // Initializes FFT, window, Mel filterbank, DCT matrix.
    bool begin();

    // Extract the mfcc from 9 second audio (144000 samples)
    // Return float MFCCs(40 x 280)
    bool extractMFCC(const float* audio, float* mfcc_output);

    // Incremental processing (optimization for sliding window)
    void resetBuffer();
    bool addAudioFrame(const float* frame_512_samples, int frame_index);
    void getCurrentMFCC(float* mfcc_output);

  private: 
    // WIndow function (hanning)
    float* window;

    // Mel filterbank and DCT matrix
    float* mel_filters;          // [NUM_MEL_FILTERS x (N_FFT/2 + 1)]
    float* dct_matrix;           // [NUM_MFCCS x NUM_MEL_FILTERS]

    // constexpr-> value known at compile time
    // static -> one value for the entire class, no duplicate per object
    static constexpr int FFT_SIZE = 1024;

    double vReal[FFT_SIZE];
    double vImag[FFT_SIZE];

    // FFT object
    ArduinoFFT<double> fft = ArduinoFFT<double>(vReal, vImag, FFT_SIZE, 16000);

    // circular buffer for incremental processing
    // This is for sliding window MFCC computation, 
    // so you don’t need to recompute MFCCs for all frames every time.
    float mfcc_buffer[NUM_FRAMES][NUM_MFCCS];
    int buffer_head;
    bool buffer_ready;

    // Internal mwthods
    void createMelFilterBank();
    void createDCTmatrix();

    // MFCC processing for a single frame
    void processFrame(const float* frame, float* mfcc_output);

    //Standarization (per_coefficient z-score accross 280 frames)
    void standardizeMFCCs(float* mfccs);

    // Helper: Mel Conversion
    float hzToMel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
    float melToHz (float mel) {return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

    MFCCExtractor(const MFCCExtractor&) = delete;
    MFCCExtractor& operator=(const MFCCExtractor&) = delete;
};


class AudioProcessor {
  public:
    AudioProcessor(int bckPin, int wsPin, int sdPin, i2s_port_t port = I2S_NUM_0);
    ~AudioProcessor();
    // Initialize I2S and audio processing 
    bool begin(int sampleRate = 16000);

    // REad the raw I2S samples
    int read(int32_t*buffer, int bufferLength);

    // Get quantized MFCC features for model inference
    // Returns true if successful and false if no tenough audio
    bool getMFCCFeatures(int8_t* mfcc_output);

    //Get the quantized MFCC from provided audio buffer
    bool getMFCCFeaturesFromAudio(const float* audio, int audio_length, int8_t* mfcc_output);

    // Check if we have enough audio for inference
    bool hasEnoughAudio() { return _samples_collected >= ANALYSIS_SAMPLES; }

    // Get audio statistics for debugging
    void getAudioStats(float& rms, float& peak, float& dc_offset);

    // continuous incremental processing
    void processIncremental(const int32_t* i2s_samples, int count);
    bool getCurrentMFCC(int8_t* mfcc_output);

  private:
    // I2S configuration
    i2s_pin_config_t _pinConfig;
    i2s_port_t _i2sPort;
    int _sampleRate;

    // Audio buffer (circular)
    float _audio_buffer[BUFFER_SAMPLES];
    int _write_index;
    int _samples_collected;

    // MFCC Extractor
    MFCCExtractor _mfcc_extractor;

    // INcremental processing state
    float _frame_buffer[N_FFT];
    int _frame_position;
    int _hop_counter;

    // COnvert I2S 24-bit to float [-1, 1]
    float convertI2SToFloat(int32_t i2s_sample);

    //convert float to int8 using model quantization
    void quantizeMFCC(const float* mfcc_float, int8_t* mfcc_int8);

    //DEbugging logic
    void logAudioStats();

    // Prevent copying
    AudioProcessor(const AudioProcessor&) = delete;
    AudioProcessor& operator=(const AudioProcessor&) = delete;
};

// =========================THIS WE WILL BE ADDING AT LAST==========================
// =========================AHILEKO LAGI OPTIONAL===================================
// ======================== SLIDING WINDOW DETECTOR CLASS ==========================
// class CoughDetector {
//   public:
//     CoughDetector();
    
//     // Initialize with detection threshold (0.0 to 1.0)
//     void begin(float threshold = 0.7f, int smoothing_window = 3);
    
//     // Add a new prediction result
//     void addResult(float cough_probability);
    
//     // Check if cough is detected (with smoothing logic)
//     bool isCoughDetected() const;
    
//     // Get detection statistics
//     float getCurrentProbability() const;
//     int getConsecutiveDetections() const;
    
//     // Reset detection state
//     void reset();
    
//   private:
//     float _threshold;
//     int _smoothing_window;
//     float _probability_buffer[10];  // Circular buffer for smoothing
//     int _buffer_index;
//     int _consecutive_detections;
//     bool _cough_detected;
    
//     void updateDetectionState();
// };

#endif