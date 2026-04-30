#!/usr/bin/env python3
"""
FAST IPTV THUMBNAIL CAPTURER (PYAV OPTIMIZED)

- Batch processing
- Parallel batches
- Thread-based workers
- Fast frame sampling (no full decode loops)
"""

import os
import json
import re
import av
import numpy as np

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

BATCH_SIZE = 50
TIMEOUT = 5
MAX_RETRIES = 1

MAX_BATCH_WORKERS = 6
MAX_STREAM_WORKERS = 16

FRAME_SAMPLE_LIMIT = 5   # 🔥 reduced (critical for speed)


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_online(stream):
    return str(stream.get("status", "")).strip().lower() == "online"


def is_black(frame):
    return np.mean(frame) < 8


def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


# ─────────────────────────────────────────────
# FAST PYAV EXTRACTION
# ─────────────────────────────────────────────

def extract_frame(url):
    try:
        container = av.open(
            url,
            timeout=TIMEOUT,
            options={
                "fflags": "nobuffer",
                "flags": "low_delay",
                "probesize": "500000",
                "analyzeduration": "500000",
                "rw_timeout": "5000000",
            }
        )

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        best_frame = None

        for i, frame in enumerate(container.decode(video=0)):
            if i >= FRAME_SAMPLE_LIMIT:
                break

            img = frame.to_ndarray(format="rgb24")

            if is_black(img):
                continue

            best_frame = img
            break  # 🔥 take first valid frame (fast)

        container.close()
        return best_frame

    except Exception:
        return None


# ─────────────────────────────────────────────
# STREAM PROCESSING
# ─────────────────────────────────────────────

def process_stream(stream, index):
    url = stream.get("url")
    name = safe_name(stream.get("channel", f"stream_{index}"))

    if not url:
        return False

    out_path = os.path.join(OUTPUT_DIR, f"{name}.jpg")

    for _ in range(MAX_RETRIES):
        frame = extract_frame(url)
        if frame is not None:
            try:
                Image.fromarray(frame).save(out_path, quality=75)
                return True
            except Exception:
                pass

    return False


# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────

def process_batch(batch, batch_id):
    workers = min(MAX_STREAM_WORKERS, len(batch))

    success = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_stream, stream, idx)
            for idx, stream in enumerate(batch)
        ]

        for f in as_completed(futures):
            try:
                if f.result():
                    success += 1
            except Exception:
                pass

    return success


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    batches = list(chunk_list(streams, BATCH_SIZE))

    print(f"Total streams: {len(streams)}")
    print(f"Batches: {len(batches)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batch workers: {min(MAX_BATCH_WORKERS, len(batches))}")
    print(f"Stream workers per batch: {MAX_STREAM_WORKERS}\n")

    total_success = 0

    # ── PARALLEL BATCH EXECUTION ──
    with ThreadPoolExecutor(max_workers=min(MAX_BATCH_WORKERS, len(batches))) as batch_executor:
        futures = [
            batch_executor.submit(process_batch, batch, i)
            for i, batch in enumerate(batches, 1)
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
            try:
                total_success += f.result()
            except Exception:
                pass

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("Done")
    print(f"Success: {total_success}/{len(streams)}")


if __name__ == "__main__":
    main()
