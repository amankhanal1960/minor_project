<!-- Generate/update Arduino model header from TFLite -->

# Generate header for the current 5s transfer model

Run from this `model/` folder:

```bash
xxd -i cough_cnn_5s_transfer_esp32_int8.tflite > "../Arduino copy/main/model_data_5s_transfer.h"
```

This generates symbols:
- `cough_cnn_5s_transfer_esp32_int8_tflite`
- `cough_cnn_5s_transfer_esp32_int8_tflite_len`

These match `Arduino copy/main/main.ino`.

---

# Optional: Python generator (if `xxd` is not available)

```python
from pathlib import Path

src = Path("cough_cnn_5s_transfer_esp32_int8.tflite")
dst = Path("../Arduino copy/main/model_data_5s_transfer.h")

data = src.read_bytes()
arr = "cough_cnn_5s_transfer_esp32_int8_tflite"
arr_len = arr + "_len"

with dst.open("w", encoding="utf-8") as f:
    f.write("#ifndef MODEL_DATA_5S_TRANSFER_H\n")
    f.write("#define MODEL_DATA_5S_TRANSFER_H\n\n")
    f.write(f"const unsigned char {arr}[] = {{\n")

    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02x}")
        if i != len(data) - 1:
            f.write(", ")
        if i % 12 == 11:
            f.write("\n")

    if len(data) % 12 != 0:
        f.write("\n")

    f.write("};\n\n")
    f.write(f"const unsigned int {arr_len} = {len(data)};\n\n")
    f.write("#endif  // MODEL_DATA_5S_TRANSFER_H\n")
```


---

# 5-second data collection (ESP32 -> Python)

Use sketch:
- `Arduino copy/audio_collector_5s/audio_collector_5s.ino`

Then run collector from repo root:

```bash
python model/collect_esp32_audio.py --port COM10 --baud 921600 --clip-seconds 5 --out model/esp32_dataset
```

If your serial link is slow/noisy, increase timeout:

```bash
python model/collect_esp32_audio.py --port COM10 --baud 921600 --clip-seconds 5 --stall-timeout 30 --out model/esp32_dataset
```
