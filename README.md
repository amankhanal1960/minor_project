# Cough Detection Wearable – ESP32‑S3 + Backend + Dashboard

## Overview
Edge-first, privacy‑preserving cough detector built around an ESP32‑S3. Audio (INMP441, 16 kHz) and motion (MPU6050, 100 Hz) are processed on-device: MFCCs → int8 1D CNN → audio/motion fusion. Only compact event JSON is sent to the backend (Node/Express + Postgres) and visualized in a Next.js dashboard.

## System Architecture
- **Sensors:** INMP441 (I²S), MPU6050 (I²C, ±2 g, 100 Hz).
- **Edge device:** ESP32‑S3 with PSRAM.
- **Processing:** 5 s MFCC window (n_fft = 1024, hop = 512, 128 Mel, 40 MFCCs), int8 CNN (Conv1D 32→64→128, GAP, softmax).
- **Fusion:** event accepted only if `p_cough ≥ 0.60` **AND** motion peak `|‖a‖−1 g| ≥ 0.12 g` in the last 1.2 s.
- **Transport:** HTTP POST `/api/detections` (deviceId, coughProbability, audioLevel).
- **Backend:** Node.js/Express, Prisma, Postgres.
- **Frontend:** Next.js dashboard (detections list, summary stats, hourly chart).

## Repos & Paths
- Firmware: `Arduino/main/` (PlatformIO/Arduino style).
- Model training: `model/` (Python, Librosa, Keras, TFLite export).
- Backend API: `backend/` (Express + Prisma).
- Frontend: `frontend/` (Next.js).

## Firmware (Arduino/main)
- **Model in use:** `model_data_5s_transfer.h` (int8 TFLM). Symbols: `cough_cnn_5s_transfer_esp32_int8_tflite`, `_len`.
- **Audio buffering:** 8 s float ring in PSRAM; 5 s slice used per inference.
- **IMU buffering:** 6 s ring in internal RAM; peak over last 1.2 s for fusion.
- **Scheduler:** inference check every 500 ms after buffer primed; MFCC dominates compute.
- **Pins:** I²S BCK=GPIO5, WS=GPIO4, SD=GPIO6; IMU SDA=GPIO10, SCL=GPIO9.
- **Wi‑Fi/API:** edit `WIFI_SSID`, `WIFI_PASS`, `DEVICE_ID`, `API_BASE` in `main.ino`.
- **Build/flash:** open `Arduino/main` in PlatformIO or Arduino IDE; compile and upload to ESP32‑S3.

### Event Payload (current firmware)
```json
{
  "deviceId": "device-123",
  "coughProbability": 0.87,
  "audioLevel": 0.12
}
```
*(Motion metrics not sent yet.)*

## Model Training (model/)
- MFCC parity with firmware: sr=16 kHz, n_fft=1024, hop=512, n_mels=128, n_mfcc=40, Hann window, Slaney mel, z‑score per coefficient.
- Base 5 s model: Acc 0.895, AUC 0.953 (test 2 676 clips).
- Transfer model (deployed): Acc 0.752, AUC 0.834 (ESP32 dataset, 238 clips).
- Quantization: post‑training int8 with representative set; exports `*_int8.tflite` and C headers.

## Backend (backend/)
- Express + Prisma + Postgres.
- Routes: `POST /api/detections`, `GET /api/detections`.
- Configure DB and `UI_ORIGIN` in `.env`.
- Run: `npm install && npm run start` (ensure Postgres running).

## Frontend (frontend/)
- Next.js dashboard fetches `/api/detections` on load.
- Shows totals, avg probability, avg audio level, severity tags, and hourly counts.
- Run: `npm install && npm run dev` (set `NEXT_PUBLIC_API_BASE_URL` if needed).

## Security / Gaps (known)
- HTTP only (no HTTPS/auth yet).
- No offline buffering; events skipped if Wi‑Fi down.
- Motion metrics not yet posted.
- No OTA/model update channel.

## Quick Demo Script (<60 s)
1) Show hardware (ESP32‑S3 + INMP441 + MPU6050).  
2) Serial monitor: point out p_cough, motion peak, fusion result.  
3) Trigger a cough (or play clip + small shake) → see “COUGH DETECTED”.  
4) Refresh dashboard → new detection entry appears.  
5) Close: all processing on-device; only event JSON sent.

