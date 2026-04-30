#!/usr/bin/env python3
"""
IPTV THUMBNAIL CAPTURER (OPTIMIZED + STABLE)

- Batch + parallel batch execution
- PyAV safe decoding
- WebP encoding (quality 80)
- Buffered batch saving
- Real-time progress logging
"""

import os
import json
import re
import time
import av
import numpy as np

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ───────────────── CONFIG ─────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

BATCH_SIZE = 50
TIMEOUT = 5

MAX_BATCH_WORKERS = 3
MAX_STREAM_WORKERS = 6
TOTAL_MAX_WORKERS = 16

FRAME_SAMPLE_LIMIT = 5

# batch image save buffer (for faster disk I/O)
SAVE_BUFFER_SIZE = 25


# ───────────────── COLORS ─────────────────

class C:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def log(msg, color=C.BLUE):
    print(f"{color}{msg}{C.RESET}")


def log_ok(msg):
    print(f"{C.GREEN}[OK]{C.RESET} {msg}")


def log_fail(msg):
    print(f"{C.RED}[FAIL]{C.RESET} {msg}")


# ───────────────── UTIL ─────────────────

def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_online(s):
    return str(s.get("status", "")).lower() == "online"


def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


# ───────────────── PYAV FRAME GRAB ─────────────────

def extract_frame(url):
    container = None
    try:
        container = av.open(
            url,
            timeout=TIMEOUT,
            options={
                "fflags": "nobuffer",
                "flags": "low_delay",
                "probesize": "200000",
                "analyzeduration": "200000",
                "rw_timeout": "4000000",
            }
        )

        if not container.streams.video:
            return None

        stream = container.streams.video[0]
        stream.thread_type = "NONE"

        for i, frame in enumerate(container.decode(video=0)):
            if i >= FRAME_SAMPLE_LIMIT:
                break

            img = frame.to_ndarray(format="rgb24")

            if np.mean(img) < 10:
                continue

            return img

        return None

    except Exception:
        return None

    finally:
        if container:
            try:
                container.close()
            except:
                pass


# ───────────────── SAVE (WEBP FAST) ─────────────────

def save_webp(path, frame):
    try:
        img = Image.fromarray(frame)

        img.save(
            path,
            format="WEBP",
            quality=80,
            method=0  # fastest encoding
        )
        return True
    except Exception:
        return False


# ───────────────── STREAM PROCESS ─────────────────

def process_stream(stream, index):
    url = stream.get("url")
    name = safe_name(stream.get("channel", f"stream_{index}"))

    if not url:
        return None

    frame = extract_frame(url)

    if frame is None:
        return None

    return (f"{OUTPUT_DIR}/{name}.webp", frame)


# ───────────────── BATCH PROCESSING ─────────────────

def process_batch(batch, batch_id, progress):
    workers = min(MAX_STREAM_WORKERS, len(batch), TOTAL_MAX_WORKERS)

    results = []

    log(f"Batch {batch_id} started ({len(batch)} streams)", C.YELLOW)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_stream, s, i)
            for i, s in enumerate(batch)
        ]

        for f in as_completed(futures):
            try:
                result = f.result()
                progress.update(1)

                if result:
                    results.append(result)

            except Exception:
                progress.update(1)

    log_ok(f"Batch {batch_id} done | valid: {len(results)}")

    return results


# ───────────────── MAIN ─────────────────

def main():
    start = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    batches = list(chunk_list(streams, BATCH_SIZE))

    log(f"Total streams: {len(streams)}", C.CYAN)
    log(f"Batches: {len(batches)}", C.CYAN)

    total_valid = 0
    save_buffer = []

    # global progress (per stream)
    with tqdm(total=len(streams), desc="Processing", smoothing=0.1) as pbar:

        with ThreadPoolExecutor(max_workers=min(MAX_BATCH_WORKERS, len(batches))) as batch_exec:
            futures = [
                batch_exec.submit(process_batch, batch, i, pbar)
                for i, batch in enumerate(batches, 1)
            ]

            for f in as_completed(futures):
                try:
                    results = f.result()

                    # ── buffered saving (faster disk writes) ──
                    for path, frame in results:
                        save_buffer.append((path, frame))

                        if len(save_buffer) >= SAVE_BUFFER_SIZE:
                            for p, fr in save_buffer:
                                save_webp(p, fr)
                            total_valid += len(save_buffer)
                            save_buffer.clear()

                except Exception:
                    pass

    # flush remaining buffer
    for p, fr in save_buffer:
        save_webp(p, fr)
        total_valid += 1

    duration = time.time() - start

    # ───────── SUMMARY ─────────

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    log_ok("DONE")

    print(f"{C.CYAN}Total streams:{C.RESET} {len(streams)}")
    print(f"{C.CYAN}Valid thumbnails:{C.RESET} {total_valid}")
    print(f"{C.CYAN}Failed:{C.RESET} {len(streams) - total_valid}")
    print(f"{C.CYAN}Time:{C.RESET} {duration:.2f}s")
    print("━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
