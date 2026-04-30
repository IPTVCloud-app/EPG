import os
import json
import av
import numpy as np
import re
import multiprocessing
import psutil
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

"""
THUMBNAIL CAPTURER FROM STREAMS

This scripts handles updates every 15 minutes on cron github workers to capture thumbnails used in IPTVCloud.
"""

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

BATCH_SIZE = 50
SAMPLE_LIMIT = 30
MAX_RETRIES = 2
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


def frame_score(frame):
    gray = np.mean(frame, axis=2)
    return np.var(gray)


def is_black(frame):
    return np.mean(frame) < 8


# ─────────────────────────────────────────────
# CPU AWARE TUNING
# ─────────────────────────────────────────────

def get_dynamic_workers():
    cpu_count = multiprocessing.cpu_count()

    cpu_usage = psutil.cpu_percent(interval=0.5)

    # reduce workers if system is busy
    if cpu_usage > 80:
        return max(1, cpu_count // 4)
    elif cpu_usage > 50:
        return max(2, cpu_count // 2)
    else:
        return cpu_count


# ─────────────────────────────────────────────
# THUMBNAIL ENGINE
# ─────────────────────────────────────────────

def extract_best_frame(url):
    container = av.open(url, timeout=TIMEOUT)

    best = None
    best_score = -1

    for i, frame in enumerate(container.decode(video=0)):
        if i >= SAMPLE_LIMIT:
            break

        img = frame.to_ndarray(format="rgb24")

        if is_black(img):
            continue

        score = frame_score(img)

        if score > best_score:
            best_score = score
            best = img

    container.close()
    return best


def process_stream(stream, index):
    url = stream.get("url")
    name = safe_name(stream.get("channel", f"stream_{index}"))

    if not url:
        return False

    out_path = os.path.join(OUTPUT_DIR, f"{name}.jpg")

    try:
        for _ in range(MAX_RETRIES):
            frame = extract_best_frame(url)
            if frame is not None:
                Image.fromarray(frame).save(out_path)
                return True
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────

def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


def process_batch(batch, batch_id, global_bar):
    workers = get_dynamic_workers()

    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_stream, stream, idx): idx
            for idx, stream in enumerate(batch, 1)
        }

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Batch {batch_id}",
            leave=False
        ):
            try:
                results.append(f.result())
            except Exception:
                results.append(False)

    global_bar.update(len(batch))
    return sum(results)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = data.get("streams", [])

    online_streams = [s for s in streams if is_online(s)]

    print(f"Total streams: {len(streams)}")
    print(f"Online only: {len(online_streams)}")
    print(f"Batches: {len(online_streams)//BATCH_SIZE + 1}")
    print(f"Initial CPU workers: {multiprocessing.cpu_count()} (dynamic tuning enabled)\n")

    batches = list(chunk_list(online_streams, BATCH_SIZE))

    total_success = 0

    with tqdm(total=len(online_streams), desc="Overall Progress") as global_bar:
        for i, batch in enumerate(batches, 1):
            total_success += process_batch(batch, i, global_bar)

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Done")
    print(f"Success: {total_success}/{len(online_streams)}")


if __name__ == "__main__":
    main()