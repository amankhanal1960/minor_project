"""
Collect labeled 5s audio clips from ESP32 over serial.

Device protocol per clip:
- magic: b"AUD0" (4 bytes)
- sample_rate: uint32 little-endian
- sample_count: uint32 little-endian
- label: uint8 ('C' or 'N')
- payload: PCM16 mono (sample_count * 2 bytes)
"""

import argparse
import json
import re
import struct
import time
import wave
from datetime import datetime
from pathlib import Path

import serial

MAGIC = b"AUD0"
VALID_LABELS = {ord("C"), ord("N")}


def normalize_person_id(person_id: str) -> str:
    """Return a safe person id string for filenames/metadata."""
    value = person_id.strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_-]", "", value)
    return value


def read_exact(ser: serial.Serial, num_bytes: int, stall_timeout: float) -> bytes:
    """Read exactly num_bytes from serial; fail if stream stalls."""
    data = bytearray()
    last_progress = time.monotonic()

    while len(data) < num_bytes:
        chunk = ser.read(min(4096, num_bytes - len(data)))
        if chunk:
            data.extend(chunk)
            last_progress = time.monotonic()
            continue

        if (time.monotonic() - last_progress) >= stall_timeout:
            raise TimeoutError(f"Serial stalled at {len(data)}/{num_bytes} bytes")

    return bytes(data)


def wait_for_magic(ser: serial.Serial, stall_timeout: float, header_timeout: float) -> None:
    """Scan incoming stream until MAGIC appears."""
    window = bytearray()
    started = time.monotonic()
    last_progress = started

    while True:
        now = time.monotonic()
        if (now - started) >= header_timeout:
            raise TimeoutError(f"Header timeout after {header_timeout:.1f}s")

        b = ser.read(1)
        if b:
            window += b
            if len(window) > len(MAGIC):
                window = window[-len(MAGIC):]
            if bytes(window) == MAGIC:
                return
            last_progress = now
            continue

        if (time.monotonic() - last_progress) >= stall_timeout:
            raise TimeoutError("Serial stalled while waiting for clip header")


def receive_clip(
    ser: serial.Serial,
    stall_timeout: float,
    header_timeout: float,
    expected_clip_seconds: float | None,
    expected_tolerance_seconds: float = 0.35,
) -> tuple[int, int, str, bytes]:
    """Receive one clip and validate header values."""
    wait_for_magic(ser, stall_timeout, header_timeout)

    header_rest = read_exact(ser, 9, stall_timeout)
    sample_rate, sample_count, label = struct.unpack("<IIB", header_rest)

    if sample_rate <= 0 or sample_rate > 96000:
        raise ValueError(f"Invalid sample_rate: {sample_rate}")
    if sample_count <= 0 or sample_count > sample_rate * 10:
        raise ValueError(f"Invalid sample_count: {sample_count}")
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label byte: {label}")

    duration_sec = sample_count / float(sample_rate)
    if expected_clip_seconds is not None:
        if abs(duration_sec - expected_clip_seconds) > expected_tolerance_seconds:
            raise ValueError(
                f"Duration mismatch ({duration_sec:.3f}s vs expected {expected_clip_seconds:.3f}s)"
            )

    pcm_bytes = read_exact(ser, sample_count * 2, stall_timeout)
    return sample_rate, sample_count, chr(label), pcm_bytes


def save_wav(path: Path, sample_rate: int, pcm_bytes: bytes) -> None:
    """Save PCM16 payload as WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def save_json(path: Path, payload: dict) -> None:
    """Save metadata JSON next to WAV."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_name(prefix: str) -> str:
    """Create timestamp-based file stem."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}"


def drain_input(ser: serial.Serial, settle_seconds: float = 0.05, max_total_seconds: float = 0.5) -> None:
    """Flush stale bytes quickly without blocking too long."""
    idle_since = time.monotonic()
    started = idle_since

    while True:
        waiting = ser.in_waiting
        if waiting > 0:
            ser.read(waiting)
            idle_since = time.monotonic()
        else:
            if (time.monotonic() - idle_since) >= settle_seconds:
                return
            time.sleep(0.01)

        if (time.monotonic() - started) >= max_total_seconds:
            return


def ensure_person_id(initial_person: str | None) -> str:
    """Resolve person id from CLI or prompt once."""
    if initial_person:
        pid = normalize_person_id(initial_person)
        if pid:
            return pid

    while True:
        value = input("Person ID (e.g. p01): ").strip()
        pid = normalize_person_id(value)
        if pid:
            return pid
        print("Invalid person ID. Use letters/numbers/_/- only.")


def collect_loop(
    ser: serial.Serial,
    out_dir: Path,
    retries: int,
    stall_timeout: float,
    header_timeout: float,
    expected_clip_seconds: float | None,
    initial_person: str | None,
) -> None:
    """Interactive loop: c = cough, n = non-cough, p <id> = switch person, q = quit."""
    current_person = ensure_person_id(initial_person)

    print(f"Connected. Current person: {current_person}")
    print("Commands: c (cough), n (non-cough), p <person_id> (switch), q (quit)")

    drain_input(ser)
    count = 0

    while True:
        raw = input("> ").strip()
        if not raw:
            continue

        low = raw.lower()

        if low == "q":
            print("Exiting.")
            return

        if low.startswith("p "):
            new_person = normalize_person_id(raw[2:])
            if not new_person:
                print("Invalid person ID.")
                continue
            current_person = new_person
            print(f"Switched person: {current_person}")
            continue

        cmd = low[0]
        if cmd not in ("c", "n"):
            print("Use: c, n, p <person_id>, q")
            continue

        tx = b"C" if cmd == "c" else b"N"
        class_name = "cough" if cmd == "c" else "non_cough"

        ok = False
        for _attempt in range(1, retries + 1):
            ser.reset_input_buffer()
            ser.write(tx)
            ser.flush()

            try:
                sample_rate, sample_count, label, pcm = receive_clip(
                    ser=ser,
                    stall_timeout=stall_timeout,
                    header_timeout=header_timeout,
                    expected_clip_seconds=expected_clip_seconds,
                )
                ok = True
                break
            except (TimeoutError, ValueError):
                drain_input(ser, settle_seconds=0.05, max_total_seconds=0.3)

        if not ok:
            print("Capture failed")
            continue

        stem = make_name(f"{class_name}_{current_person}")
        wav_path = out_dir / class_name / f"{stem}.wav"
        json_path = wav_path.with_suffix(".json")

        save_wav(wav_path, sample_rate, pcm)

        duration_sec = sample_count / float(sample_rate)
        metadata = {
            "label": class_name,
            "label_from_device": label,
            "person_id": current_person,
            "sample_rate": sample_rate,
            "sample_count": sample_count,
            "duration_sec": round(duration_sec, 6),
            "expected_clip_seconds": expected_clip_seconds,
            "datetime": datetime.now().isoformat(),
        }
        save_json(json_path, metadata)

        count += 1
        print(f"saved #{count}: {wav_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled audio clips from ESP32 over serial")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM10")
    parser.add_argument("--baud", type=int, default=460800, help="Serial baud rate")
    parser.add_argument("--timeout", type=float, default=1.0, help="Low-level serial read timeout (seconds)")
    parser.add_argument("--stall-timeout", type=float, default=20.0, help="Fail if serial stalls this long")
    parser.add_argument("--header-timeout", type=float, default=8.0, help="Max time waiting for AUD0 header")
    parser.add_argument("--retries", type=int, default=3, help="Retries for each command")
    parser.add_argument("--out", default="esp32_dataset", help="Output root directory")
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=5.0,
        help="Expected clip duration in seconds; use <= 0 to disable duration check",
    )
    parser.add_argument(
        "--person",
        default=None,
        help="Initial person id for this session (e.g. p01). You can switch later with 'p <id>'.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_clip_seconds = args.clip_seconds if args.clip_seconds > 0 else None

    print(f"Opening {args.port} @ {args.baud}...")
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        collect_loop(
            ser=ser,
            out_dir=out_dir,
            retries=max(1, args.retries),
            stall_timeout=max(1.0, args.stall_timeout),
            header_timeout=max(2.0, args.header_timeout),
            expected_clip_seconds=expected_clip_seconds,
            initial_person=args.person,
        )


if __name__ == "__main__":
    main()
