import argparse
import json
import struct
import time
import wave
from datetime import datetime
from pathlib import Path

import serial

MAGIC = b"AUD0"
VALID_LABELS = {ord("C"), ord("N")}


def read_exact(ser: serial.Serial, n: int, stall_timeout: float) -> bytes:
    """Read exactly n bytes; fail if serial input makes no progress for stall_timeout."""
    buf = bytearray()
    last_progress = time.monotonic()

    while len(buf) < n:
        to_read = min(4096, n - len(buf))
        chunk = ser.read(to_read)

        if chunk:
            buf.extend(chunk)
            last_progress = time.monotonic()
            continue

        stalled_for = time.monotonic() - last_progress
        if stalled_for >= stall_timeout:
            raise TimeoutError(
                f"Serial timeout: received {len(buf)}/{n} bytes (stalled {stalled_for:.1f}s)"
            )

    return bytes(buf)


def sync_magic(ser: serial.Serial, stall_timeout: float, header_timeout: float) -> None:
    window = bytearray()
    started = time.monotonic()
    last_progress = started

    while True:
        now = time.monotonic()
        if now - started >= header_timeout:
            raise TimeoutError(
                f"Timed out waiting for AUD0 header after {header_timeout:.1f}s. "
                "Check that audio_collector_5s.ino is uploaded and running."
            )

        b = ser.read(1)

        if b:
            window += b
            if len(window) > 4:
                window = window[-4:]

            if bytes(window) == MAGIC:
                return

            last_progress = time.monotonic()
            continue

        stalled_for = time.monotonic() - last_progress
        if stalled_for >= stall_timeout:
            raise TimeoutError(
                f"Serial timeout while waiting for clip header (stalled {stalled_for:.1f}s)"
            )


def recv_clip(
    ser: serial.Serial,
    stall_timeout: float,
    header_timeout: float,
    expected_clip_seconds: float | None = None,
    expected_tolerance_seconds: float = 0.35,
):
    # Header format from ESP32 sketch:
    # magic[4] + sample_rate(uint32) + sample_count(uint32) + label(uint8)
    sync_magic(ser, stall_timeout, header_timeout)
    rest = read_exact(ser, 9, stall_timeout)
    sample_rate, sample_count, label = struct.unpack("<IIB", rest)

    if sample_rate <= 0 or sample_rate > 96000:
        raise ValueError(f"Invalid sample_rate in header: {sample_rate}")
    if sample_count <= 0 or sample_count > sample_rate * 10:
        raise ValueError(f"Invalid sample_count in header: {sample_count}")
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label byte in header: {label}")

    duration_sec = sample_count / float(sample_rate)
    if expected_clip_seconds is not None:
        if abs(duration_sec - expected_clip_seconds) > expected_tolerance_seconds:
            raise ValueError(
                f"Clip duration mismatch: got {duration_sec:.3f}s, expected ~{expected_clip_seconds:.3f}s"
            )

    pcm_bytes = read_exact(ser, sample_count * 2, stall_timeout)  # int16 mono
    return sample_rate, sample_count, chr(label), pcm_bytes


def save_wav(path: Path, sample_rate: int, pcm_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_name(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}"


def drain_input(
    ser: serial.Serial,
    settle_seconds: float = 0.2,
    max_total_seconds: float = 1.0,
) -> int:
    """Drain trailing bytes, but never block forever on a noisy serial stream."""
    dropped = 0
    idle_since = time.monotonic()
    started = time.monotonic()

    while True:
        waiting = ser.in_waiting
        if waiting > 0:
            dropped += len(ser.read(waiting))
            idle_since = time.monotonic()
        else:
            if time.monotonic() - idle_since >= settle_seconds:
                return dropped
            time.sleep(0.01)

        if time.monotonic() - started >= max_total_seconds:
            return dropped


def collect_loop(
    ser: serial.Serial,
    out_dir: Path,
    retries: int,
    stall_timeout: float,
    header_timeout: float,
    expected_clip_seconds: float | None,
) -> None:
    print("Connected.")
    print("Type: c = cough, n = non-cough, q = quit")
    if expected_clip_seconds is not None:
        print(f"Expected clip duration: ~{expected_clip_seconds:.2f}s")

    dropped = drain_input(ser, settle_seconds=0.05, max_total_seconds=0.5)
    if dropped:
        print(f"[INFO] Dropped {dropped} stale serial bytes at startup")

    count = 0

    while True:
        raw = input("> ").strip().lower()
        if not raw:
            continue

        cmd = raw[0]

        if cmd == "q":
            print("Exiting.")
            return

        if cmd not in ("c", "n"):
            print("Use only: c, n, q")
            continue

        tx = b"C" if cmd == "c" else b"N"
        class_name = "cough" if cmd == "c" else "non_cough"

        sample_rate = sample_count = 0
        label = "?"
        pcm = b""

        ok = False
        for attempt in range(1, retries + 1):
            ser.reset_input_buffer()
            ser.write(tx)
            ser.flush()

            try:
                sample_rate, sample_count, label, pcm = recv_clip(
                    ser,
                    stall_timeout=stall_timeout,
                    header_timeout=header_timeout,
                    expected_clip_seconds=expected_clip_seconds,
                )
                ok = True
                break
            except (TimeoutError, ValueError) as e:
                print(f"[WARN] {e} (attempt {attempt}/{retries})")
                dropped = drain_input(ser, settle_seconds=0.05, max_total_seconds=0.3)
                if dropped:
                    print(f"[INFO] Dropped {dropped} trailing bytes after failed attempt")

        if not ok:
            print("[WARN] Skipping this command after retries")
            continue

        stem = make_name(class_name)
        wav_path = out_dir / class_name / f"{stem}.wav"
        json_path = wav_path.with_suffix(".json")

        save_wav(wav_path, sample_rate, pcm)

        duration_sec = round(sample_count / float(sample_rate), 6)
        meta = {
            "label": class_name,
            "label_from_device": label,
            "sample_rate": sample_rate,
            "sample_count": sample_count,
            "duration_sec": duration_sec,
            "expected_clip_seconds": expected_clip_seconds,
            "datetime": datetime.now().isoformat(),
        }
        save_json(json_path, meta)

        count += 1
        print(f"saved #{count}: {wav_path} ({duration_sec:.3f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled audio clips from ESP32 over serial")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM10")
    parser.add_argument("--baud", type=int, default=460800, help="Serial baud rate")
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Low-level serial read timeout seconds (keep small)",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=20.0,
        help="Fail if no incoming serial data for this many seconds",
    )
    parser.add_argument(
        "--header-timeout",
        type=float,
        default=8.0,
        help="Fail if AUD0 clip header is not found within this time",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per command if clip receive fails")
    parser.add_argument("--out", default="esp32_dataset", help="Output folder")
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=5.0,
        help="Expected clip duration in seconds; set <=0 to disable check",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_clip_seconds = args.clip_seconds if args.clip_seconds > 0 else None

    print(f"Opening {args.port} @ {args.baud}...")
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        collect_loop(
            ser,
            out_dir,
            retries=max(1, args.retries),
            stall_timeout=max(1.0, args.stall_timeout),
            header_timeout=max(2.0, args.header_timeout),
            expected_clip_seconds=expected_clip_seconds,
        )


if __name__ == "__main__":
    main()



# python model/collect_esp32_audio.py --port COM10 --baud 460800 --clip-seconds 5 --out model/esp32_dataset
