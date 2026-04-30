#!/usr/bin/env python3

import os
import json
import re
import time
import av
import numpy as np
import queue
import threading

from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# ───────────────── CONFIG ─────────────────

STREAMS_FILE = "streams.json"
OUTPUT_DIR = "thumbnails"

TIMEOUT = 4
FRAME_LIMIT = 3

DECODE_WORKERS = 12     # CPU/network bound
ENCODE_WORKERS = 6      # disk bound

QUEUE_SIZE = 500


# ───────────────── UTIL ─────────────────

def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_online(s):
    return str(s.get("status", "")).lower() == "online"


# ───────────────── QUEUES ─────────────────

frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
done_queue = queue.Queue()


# ───────────────── DECODE WORKER ─────────────────

def decode_worker(streams, pbar):
    for i, stream in enumerate(streams):
        url = stream.get("url")
        name = safe_name(stream.get("channel", f"stream_{i}"))

        if not url:
            pbar.update(1)
            continue

        try:
            container = av.open(
                url,
                timeout=TIMEOUT,
                options={
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                    "probesize": "200000",
                    "analyzeduration": "200000",
                }
            )

            if not container.streams.video:
                raise Exception()

            stream_v = container.streams.video[0]
            stream_v.thread_type = "NONE"

            frame = None

            for j, f in enumerate(container.decode(video=0)):
                if j >= FRAME_LIMIT:
                    break

                img = f.to_ndarray(format="rgb24")

                if np.mean(img) > 10:
                    frame = img
                    break

            container.close()

            if frame is not None:
                frame_queue.put((name, frame))

        except:
            pass

        pbar.update(1)

    # signal end
    for _ in range(ENCODE_WORKERS):
        frame_queue.put(None)


# ───────────────── ENCODE WORKER ─────────────────

def encode_worker():
    while True:
        item = frame_queue.get()

        if item is None:
            break

        name, frame = item

        path = os.path.join(OUTPUT_DIR, f"{name}.webp")

        try:
            img = Image.fromarray(frame)

            img.save(
                path,
                format="WEBP",
                quality=80,
                method=0
            )

            done_queue.put(True)

        except:
            done_queue.put(False)


# ───────────────── MAIN ─────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_json(STREAMS_FILE)
    streams = [s for s in data.get("streams", []) if is_online(s)]

    total = len(streams)

    print(f"Streams: {total}")
    print(f"Decode workers: {DECODE_WORKERS}")
    print(f"Encode workers: {ENCODE_WORKERS}\n")

    start = time.time()

    with tqdm(total=total, desc="Decoding") as pbar:

        # start encode workers
        encoders = []
        for _ in range(ENCODE_WORKERS):
            t = threading.Thread(target=encode_worker, daemon=True)
            t.start()
            encoders.append(t)

        # run decode workers
        decode_worker(streams, pbar)

        # wait encoders
        for t in encoders:
            t.join()

    # summary
    success = 0
    while not done_queue.empty():
        if done_queue.get():
            success += 1

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Done")
    print(f"Success: {success}/{total}")
    print(f"Time: {time.time() - start:.2f}s")
    print("━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
