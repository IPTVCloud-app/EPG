#!/usr/bin/env python3
"""
IPTV THUMBNAIL CAPTURER
"""

import os
import json
import re
import subprocess
import io
import time

import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

MAX_WORKERS = 8          # safe pool size (adjust 4–12)
TIMEOUT = 12
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
    return str(stream.get("status", "")).lower() == "online"


def is_black(frame):
    return np.mean(frame) < 10


# ─────────────────────────────────────────────
# FFMPEG FRAME CAPTURE
# ─────────────────────────────────────────────

def ffmpeg_capture(url):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "2",

        "-rw_timeout", "15000000",

        "-headers",
        "User-Agent: Mozilla/5.0\r\nReferer: https://google.com\r\n",

        "-ss", "2",
        "-i", url,

        "-frames:v", "1",
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
        time.sleep(0.3)
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

    frame = capture_with_retry(url)

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
# MAIN POOL EXECUTION
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    print(f"Streams: {len(streams)}")
    print(f"Workers: {MAX_WORKERS}\n")

    success = 0

    # 🔥 FFmpeg SUBPROCESS POOL
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(process_stream, stream, i)
            for i, stream in enumerate(streams)
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                success += 1 if f.result() else 0
            except Exception:
                pass

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("DONE")
    print(f"Success: {success}/{len(streams)}")


if __name__ == "__main__":
    main()
