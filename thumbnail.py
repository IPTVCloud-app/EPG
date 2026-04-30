#!/usr/bin/env python3
"""
IPTV THUMBNAIL CAPTURER (PYAV VERSION - CLEAN & FAST)
"""

import os
import json
import re
import time

import numpy as np
import psutil
import av
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

TIMEOUT_SECONDS = 10
BATCH_SIZE = 50
MAX_BATCH_WORKERS = 4

RETRIES = 2


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
    return np.mean(frame) < 10


def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield i, data[i:i + size]


# ─────────────────────────────────────────────
# CPU-aware workers
# ─────────────────────────────────────────────

def get_workers():
    cpu = psutil.cpu_percent(interval=0.2)

    if cpu < 60:
        return 12
    elif cpu < 80:
        return 8
    elif cpu < 90:
        return 5
    else:
        return 3


# ─────────────────────────────────────────────
# PYAV FRAME EXTRACTION (CORE FIX)
# ─────────────────────────────────────────────

def pyav_extract(url):
    try:
        # Open stream (auto demux + decode)
        container = av.open(
            url,
            options={
                "user_agent": "Mozilla/5.0",
                "reconnect": "1",
                "reconnect_streamed": "1",
                "reconnect_delay_max": "2",
            },
            timeout=TIMEOUT_SECONDS
        )

        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")

            if is_black(img):
                continue

            container.close()
            return img

        container.close()
        return None

    except Exception:
        return None


# ─────────────────────────────────────────────
# RETRY WRAPPER
# ─────────────────────────────────────────────

def extract_with_retry(url):
    for _ in range(RETRIES):
        frame = pyav_extract(url)
        if frame is not None:
            return frame
        time.sleep(0.5)
    return None


# ─────────────────────────────────────────────
# STREAM PROCESSING
# ─────────────────────────────────────────────

def process_stream(stream, idx):
    url = stream.get("url")
    if not url:
        return False

    name = safe_name(stream.get("channel", f"stream_{idx}"))
    out_path = os.path.join(OUTPUT_DIR, f"{name}.webp")

    frame = extract_with_retry(url)

    if frame is None:
        return False

    try:
        Image.fromarray(frame).save(
            out_path,
            "WEBP",
            quality=82,
            method=6
        )
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────

def process_batch(offset, batch):
    workers = get_workers()
    success = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_stream, stream, offset + i)
            for i, stream in enumerate(batch)
        ]

        for f in as_completed(futures):
            try:
                success += 1 if f.result() else 0
            except Exception:
                pass

    return success


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield i, data[i:i + size]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    batches = list(chunk_list(streams, BATCH_SIZE))

    print(f"Streams: {len(streams)}")
    print(f"Batches: {len(batches)}\n")

    total = 0

    with ThreadPoolExecutor(max_workers=MAX_BATCH_WORKERS) as batch_exec:
        futures = [
            batch_exec.submit(process_batch, offset, batch)
            for offset, batch in batches
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
            total += f.result()

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("DONE")
    print(f"Success: {total}/{len(streams)}")


if __name__ == "__main__":
    main()
