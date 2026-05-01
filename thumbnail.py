#!/usr/bin/env python3
"""
IPTV THUMBNAIL CAPTURER FOR IPTVCLOUD.APP
"""

import os
import json
import re
import subprocess
import io
import time
import hashlib

import numpy as np
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

BATCH_SIZE = 10
WORKERS_PER_BATCH = 6
TIMEOUT = 1
RETRIES = 2


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────

def safe_name(name: str):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def unique_name(stream, idx):
    base = safe_name(stream.get("channel", f"stream_{idx}"))
    url = stream.get("url", "")
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{base}_{h}"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_online(stream):
    return str(stream.get("status", "")).lower() == "online"


def is_black(frame):
    return np.mean(frame) < 10


def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield i, data[i:i + size]


def deduplicate_streams(streams):
    seen = set()
    unique = []
    for s in streams:
        url = s.get("url")
        if url and url not in seen:
            seen.add(url)
            unique.append(s)
    return unique


# ─────────────────────────────────────────────
# FFMPEG FRAME CAPTURE
# ─────────────────────────────────────────────

def ffmpeg_capture(url):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "1",
        "-rw_timeout", "8000000",
        "-ss", "00:00:01",
        "-i", url,
        "-reorder_queue_size", "0",
        "-vf", "scale=iw/2:ih/2",
        "-vframes", "1",  # FIXED
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-"
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT
        )

        if result.returncode != 0:
            return None

        if not result.stdout or len(result.stdout) < 1000:
            return None

        img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        frame = np.array(img)

        if is_black(frame):
            return None

        return frame

    except Exception:
        return None


# ─────────────────────────────────────────────
# RETRY WRAPPER
# ─────────────────────────────────────────────

def capture_with_retry(url):
    for _ in range(RETRIES):
        frame = ffmpeg_capture(url)
        if frame is not None:
            return frame
        time.sleep(0.2)
    return None


# ─────────────────────────────────────────────
# STREAM PROCESSING
# ─────────────────────────────────────────────

def process_stream(stream, idx):
    url = stream.get("url")
    if not url:
        return False

    name = unique_name(stream, idx)
    out_path = os.path.join(OUTPUT_DIR, f"{name}.webp")

    frame = capture_with_retry(url)
    if frame is None:
        return False

    try:
        tmp_path = out_path + ".tmp"

        # Save to temp file first
        Image.fromarray(frame).save(
            tmp_path,
            format="WEBP",
            quality=80,
            method=4
        )

        # Atomic overwrite (safe even with threads)
        os.replace(tmp_path, out_path)

        return True

    except Exception:
        return False


# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────

def process_batch(batch, batch_id):
    success = 0

    with ThreadPoolExecutor(max_workers=WORKERS_PER_BATCH) as executor:
        futures = [
            executor.submit(process_stream, stream, batch_id * 1000 + i)
            for i, stream in enumerate(batch)
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

    # Remove duplicate URLs
    streams = deduplicate_streams(streams)

    batches = list(chunk_list(streams, BATCH_SIZE))

    print(f"Total streams: {len(streams)}")
    print(f"Batches: {len(batches)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers per batch: {WORKERS_PER_BATCH}\n")

    total_success = 0

    for i, (offset, batch) in enumerate(tqdm(batches, desc="Batches")):
        total_success += process_batch(batch, i)

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("DONE")
    print(f"Success: {total_success}/{len(streams)}")


if __name__ == "__main__":
    main()
