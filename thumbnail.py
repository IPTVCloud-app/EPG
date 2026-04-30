#!/usr/bin/env python3
"""
FAST IPTV THUMBNAIL CAPTURER (WEBP + CPU-AWARE + FIXED)
"""

import os
import json
import re
import subprocess
import io
import time

import numpy as np
import psutil
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

BATCH_SIZE = 50
TIMEOUT = 15


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
# CPU-AWARE SCALING (psutil)
# ─────────────────────────────────────────────

def get_dynamic_workers():
    cpu = psutil.cpu_percent(interval=0.2)

    if cpu < 50:
        return 16, 6   # stream_workers, batch_workers
    elif cpu < 75:
        return 10, 4
    elif cpu < 90:
        return 6, 3
    else:
        return 3, 2


# ─────────────────────────────────────────────
# FRAME EXTRACTION (FFMPEG → WEBP PIPE)
# ─────────────────────────────────────────────

def extract_frame(url):
    try:
        cmd = [
            "ffmpeg",
            "-loglevel", "error",

            # required for most IPTV streams
            "-headers",
            "User-Agent: Mozilla/5.0\r\nReferer: https://google.com\r\n",

            "-ss", "2",
            "-i", url,

            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "libwebp",   # WEBP output

            "-"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT
        )

        # IMPORTANT: validate ffmpeg success
        if result.returncode != 0 or not result.stdout:
            return None

        if len(result.stdout) < 500:
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

def process_stream(stream, global_idx):
    url = stream.get("url")
    name = safe_name(stream.get("channel", f"stream_{global_idx}"))

    if not url:
        return False

    out_path = os.path.join(OUTPUT_DIR, f"{name}.webp")

    frame = extract_frame(url)

    if frame is None:
        return False

    try:
        Image.fromarray(frame).save(
            out_path,
            "WEBP",
            quality=80,
            method=6
        )
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# BATCH PROCESSING (DYNAMIC THREADS)
# ─────────────────────────────────────────────

def process_batch_dynamic(batch_offset, batch, stream_workers):
    success = 0

    with ThreadPoolExecutor(max_workers=stream_workers) as executor:
        futures = [
            executor.submit(process_stream, stream, batch_offset + i)
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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    batches = list(chunk_list(streams, BATCH_SIZE))

    print(f"Total streams: {len(streams)}")
    print(f"Batches: {len(batches)}\n")

    total_success = 0

    # batch executor is lightweight
    with ThreadPoolExecutor(max_workers=4) as batch_executor:

        futures = []

        for offset, batch in batches:
            stream_workers, _ = get_dynamic_workers()

            futures.append(
                batch_executor.submit(process_batch_dynamic, offset, batch, stream_workers)
            )

        for f in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
            total_success += f.result()

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("Done")
    print(f"Success: {total_success}/{len(streams)}")


if __name__ == "__main__":
    main()
