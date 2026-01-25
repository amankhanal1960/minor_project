#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H


#include <driver/i2s.h>

class AudioProcessor {
  public:
    AudioProcessor(int bckPin, int wsPin, int sdPin, i2s_port_t port = I2S_NUM_0);

    bool begin(int sampleRate = 16000);

    int read(int32_t*buffer, int bufferLength);

  private:
    i2s_pin_config_t _pinConfig;
    i2s_port_t _i2sPort;
    int _sampleRate;
};

#endif