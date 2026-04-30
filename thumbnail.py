#!/usr/bin/env python3

import os
import json
import re
import time
import subprocess
import numpy as np

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ───────── CONFIG ─────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

WORKERS = 48          # safe for subprocess (NOT PyAV)
TIMEOUT = 6


# ───────── UTIL ─────────

def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_online(s):
    return str(s.get("status", "")).lower() == "online"


# ───────── FAST FFMEG (ISOLATED = NO CRASH) ─────────

def extract_frame(url):
    try:
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-rw_timeout", "5000000",

            "-i", url,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-"
        ]

        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=TIMEOUT
        )

        if not p.stdout:
            return None

        img = Image.open(io.BytesIO(p.stdout)).convert("RGB")
        frame = np.array(img)

        if np.mean(frame) < 10:
            return None

        return frame

    except:
        return None


# ───────── PROCESS ─────────

def process(stream, idx):
    url = stream.get("url")
    name = safe_name(stream.get("channel", f"stream_{idx}"))

    if not url:
        return False

    out = os.path.join(OUTPUT_DIR, f"{name}.webp")

    frame = extract_frame(url)
    if frame is None:
        return False

    try:
        Image.fromarray(frame).save(
            out,
            format="WEBP",
            quality=80,
            method=0
        )
        return True
    except:
        return False


# ───────── MAIN ─────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    print(f"Streams: {len(streams)}")
    print(f"Workers: {WORKERS}\n")

    start = time.time()
    success = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(process, s, i): i
            for i, s in enumerate(streams)
        }

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                success += 1 if f.result() else 0
            except:
                pass

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Done")
    print(f"Success: {success}/{len(streams)}")
    print(f"Time: {time.time() - start:.2f}s")
    print("━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
