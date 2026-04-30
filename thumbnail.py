#!/usr/bin/env python3

import os
import json
import re
import time
import av
import numpy as np
import threading

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ───────────────── CONFIG ─────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

BATCH_SIZE = 50
TIMEOUT = 5
MAX_RETRIES = 1

MAX_BATCH_WORKERS = 3
MAX_STREAM_WORKERS = 6
TOTAL_MAX_WORKERS = 16

FRAME_SAMPLE_LIMIT = 5


# ───────────────── COLOR ─────────────────

class C:
    RESET="\033[0m"; GREEN="\033[92m"; RED="\033[91m"
    YELLOW="\033[93m"; BLUE="\033[94m"; CYAN="\033[96m"


# ───────────────── UTIL ─────────────────

def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_online(s):
    return str(s.get("status","")).lower()=="online"

def is_black(frame):
    return np.mean(frame) < 8

def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i+size]


# ───────────────── PYAV ─────────────────

def extract_frame(url):
    container=None
    try:
        container = av.open(
            url,
            timeout=TIMEOUT,
            options={
                "fflags":"nobuffer",
                "flags":"low_delay",
                "probesize":"500000",
                "analyzeduration":"500000",
                "rw_timeout":"5000000",
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
            if is_black(img):
                continue

            return img

        return None

    except:
        return None

    finally:
        if container:
            try: container.close()
            except: pass


# ───────────────── STREAM ─────────────────

def process_stream(stream, index):
    url = stream.get("url")
    name = safe_name(stream.get("channel", f"stream_{index}"))

    if not url:
        return False

    out = os.path.join(OUTPUT_DIR, f"{name}.jpg")

    for _ in range(MAX_RETRIES):
        frame = extract_frame(url)
        if frame is not None:
            try:
                Image.fromarray(frame).save(out, quality=75)
                return True
            except:
                return False

    return False


# ───────────────── BATCH ─────────────────

def process_batch(batch, progress_bar, lock):
    workers = min(MAX_STREAM_WORKERS, len(batch), TOTAL_MAX_WORKERS)

    success = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_stream, s, i) for i, s in enumerate(batch)]

        for f in as_completed(futures):
            try:
                if f.result():
                    success += 1
            except:
                pass

            # 🔥 real-time progress update
            with lock:
                progress_bar.update(1)

    return success


# ───────────────── MAIN ─────────────────

def main():
    start = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    batches = list(chunk_list(streams, BATCH_SIZE))

    print(f"{C.BLUE}Streams:{C.RESET} {len(streams)}")
    print(f"{C.BLUE}Batches:{C.RESET} {len(batches)}\n")

    total_success = 0
    lock = threading.Lock()

    # 🔥 global progress (per stream, not batch)
    with tqdm(total=len(streams), desc="Processing", smoothing=0.1) as pbar:

        with ThreadPoolExecutor(max_workers=min(MAX_BATCH_WORKERS, len(batches))) as batch_exec:
            futures = [
                batch_exec.submit(process_batch, batch, pbar, lock)
                for batch in batches
            ]

            for f in as_completed(futures):
                try:
                    total_success += f.result()
                except:
                    pass

    duration = time.time() - start
    success_rate = (total_success / len(streams) * 100) if streams else 0

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{C.GREEN if total_success else C.RED}Done{C.RESET}")
    print(f"{C.CYAN}Success:{C.RESET} {total_success}")
    print(f"{C.CYAN}Failed:{C.RESET} {len(streams)-total_success}")
    print(f"{C.CYAN}Rate:{C.RESET} {success_rate:.2f}%")
    print(f"{C.CYAN}Time:{C.RESET} {duration:.2f}s")
    print("━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
