#!/usr/bin/env python3
"""
FAST IPTV THUMBNAIL CAPTURER (BATCH + PARALLEL OPTIMIZED)
"""

import os
import json
import re
import subprocess
import io
import multiprocessing

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
TIMEOUT = 6


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
# FAST FRAME EXTRACTION (FFMPEG)
# ─────────────────────────────────────────────

def extract_frame(url):
    try:
        cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-ss", "2",
            "-i", url,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=TIMEOUT
        )

        if not result.stdout:
            return None

        img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        frame = np.array(img)

        if is_black(frame):
            return None

        return frame

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

    frame = extract_frame(url)

    if frame is None:
        return False

    try:
        Image.fromarray(frame).save(out_path, quality=75, optimize=True)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# BATCH PROCESSING (THREAD-BASED)
# ─────────────────────────────────────────────

def process_batch(batch, batch_id):
    workers = min(32, len(batch))  # safe cap for stability

    success = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_stream, stream, idx)
            for idx, stream in enumerate(batch)
        ]

        for f in as_completed(futures):
            try:
                success += 1 if f.result() else 0
            except Exception:
                pass

    return success


# ─────────────────────────────────────────────
# MAIN (PARALLEL BATCH EXECUTION)
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    batches = list(chunk_list(streams, BATCH_SIZE))

    print(f"Total streams: {len(streams)}")
    print(f"Batches: {len(batches)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Parallel batch workers: {min(8, len(batches))}\n")

    total_success = 0

    # ── PARALLEL BATCH EXECUTION ──
    with ThreadPoolExecutor(max_workers=min(8, len(batches))) as batch_executor:
        futures = [
            batch_executor.submit(process_batch, batch, i)
            for i, batch in enumerate(batches, 1)
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
            total_success += f.result()

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("Done")
    print(f"Success: {total_success}/{len(streams)}")


if __name__ == "__main__":
    main()