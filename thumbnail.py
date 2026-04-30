#!/usr/bin/env python3
"""
STABLE IPTV THUMBNAIL CAPTURER (PYAV + COLOR LOGS)

- Batch + parallel processing
- Safe PyAV decoding (no segfaults)
- Colored console logs
- Final execution summary
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


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

BATCH_SIZE = 50

TIMEOUT = 5
MAX_RETRIES = 1

MAX_BATCH_WORKERS = 3
MAX_STREAM_WORKERS = 6
TOTAL_MAX_WORKERS = 16

FRAME_SAMPLE_LIMIT = 5


# ─────────────────────────────────────────────
# COLOR LOGGING
# ─────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def log_info(msg):
    print(f"{C.BLUE}[INFO]{C.RESET} {msg}")


def log_success(msg):
    print(f"{C.GREEN}[SUCCESS]{C.RESET} {msg}")


def log_warn(msg):
    print(f"{C.YELLOW}[WARN]{C.RESET} {msg}")


def log_error(msg):
    print(f"{C.RED}[ERROR]{C.RESET} {msg}")


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
# SAFE PYAV EXTRACTION
# ─────────────────────────────────────────────

def extract_frame(url):
    container = None

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

        if not container.streams.video:
            return None

        stream = container.streams.video[0]
        stream.thread_type = "NONE"  # critical

        for i, frame in enumerate(container.decode(video=0)):
            if i >= FRAME_SAMPLE_LIMIT:
                break

            img = frame.to_ndarray(format="rgb24")

            if is_black(img):
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
            except:
                return False

    return False


# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────

def process_batch(batch, batch_id):
    workers = min(MAX_STREAM_WORKERS, len(batch), TOTAL_MAX_WORKERS)

    success = 0

    for _ in range(1):  # simple wrapper to allow future extension
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_stream, stream, idx)
                for idx, stream in enumerate(batch)
            ]

            for f in as_completed(futures):
                try:
                    if f.result():
                        success += 1
                except:
                    pass

    return success


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_info("Loading streams...")
    data = load_json(STREAMS_FILE)

    streams = data.get("streams", [])
    online_streams = [s for s in streams if is_online(s)]

    batches = list(chunk_list(online_streams, BATCH_SIZE))

    log_info(f"Total streams: {len(streams)}")
    log_info(f"Online streams: {len(online_streams)}")
    log_info(f"Batches: {len(batches)}")
    log_info(f"Batch size: {BATCH_SIZE}")
    log_info(f"Batch workers: {min(MAX_BATCH_WORKERS, len(batches))}")
    log_info(f"Stream workers per batch: {MAX_STREAM_WORKERS}\n")

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
            except:
                pass

    duration = time.time() - start_time
    success_rate = (total_success / len(online_streams) * 100) if online_streams else 0

    # ─────────────────────────────
    # SUMMARY
    # ─────────────────────────────

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if total_success > 0:
        log_success("Thumbnail generation completed")
    else:
        log_error("No thumbnails generated")

    print(f"{C.CYAN}Total:{C.RESET} {len(streams)}")
    print(f"{C.CYAN}Online:{C.RESET} {len(online_streams)}")
    print(f"{C.CYAN}Success:{C.RESET} {total_success}")
    print(f"{C.CYAN}Failed:{C.RESET} {len(online_streams) - total_success}")
    print(f"{C.CYAN}Success Rate:{C.RESET} {success_rate:.2f}%")
    print(f"{C.CYAN}Time:{C.RESET} {duration:.2f}s")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
