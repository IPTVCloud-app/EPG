#!/usr/bin/env python3
"""
STABLE IPTV THUMBNAIL CAPTURER
- GitHub Actions optimized
- Colored logging (ANSI + GHA annotations)
- Batch parallel processing
- Safe PyAV / FFmpeg threading
"""

import os
import sys
import json
import re
import time
import av
import numpy as np

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from tqdm import tqdm


# ─────────────────────────────────────────────
# ENVIRONMENT DETECTION
# ─────────────────────────────────────────────

IS_GHA = os.getenv("GITHUB_ACTIONS") == "true"
NO_COLOR = not sys.stdout.isatty() and not IS_GHA


# ─────────────────────────────────────────────
# COLORED LOGGER
# ─────────────────────────────────────────────

class Color:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"


def _colorize(text: str, *codes: str) -> str:
    if NO_COLOR:
        return text
    return "".join(codes) + text + Color.RESET


class Log:
    """Unified logger: ANSI colors + GitHub Actions workflow commands."""

    @staticmethod
    def info(msg: str):
        prefix = _colorize("ℹ INFO ", Color.CYAN, Color.BOLD)
        print(f"{prefix} {msg}")

    @staticmethod
    def success(msg: str):
        prefix = _colorize("✔ OK   ", Color.GREEN, Color.BOLD)
        print(f"{prefix} {msg}")

    @staticmethod
    def warn(msg: str):
        prefix = _colorize("⚠ WARN ", Color.YELLOW, Color.BOLD)
        print(f"{prefix} {msg}")
        if IS_GHA:
            print(f"::warning::{msg}")

    @staticmethod
    def error(msg: str):
        prefix = _colorize("✖ ERR  ", Color.RED, Color.BOLD)
        print(f"{prefix} {msg}", file=sys.stderr)
        if IS_GHA:
            print(f"::error::{msg}", file=sys.stderr)

    @staticmethod
    def debug(msg: str):
        if os.getenv("RUNNER_DEBUG") == "1" or os.getenv("DEBUG"):
            prefix = _colorize("⋯ DEBUG", Color.GRAY, Color.BOLD)
            print(f"{prefix} {msg}")
            if IS_GHA:
                print(f"::debug::{msg}")

    @staticmethod
    def group(title: str):
        if IS_GHA:
            print(f"::group::{title}")
        else:
            print(_colorize(f"\n┌─ {title} ", Color.MAGENTA, Color.BOLD))

    @staticmethod
    def endgroup():
        if IS_GHA:
            print("::endgroup::")
        else:
            print(_colorize("└─────────────────────────────\n", Color.MAGENTA))

    @staticmethod
    def summary_line(label: str, value: str, ok: bool = True):
        color = Color.GREEN if ok else Color.RED
        label_str = _colorize(f"{label:<25}", Color.BOLD)
        value_str = _colorize(value, color, Color.BOLD)
        print(f"  {label_str} {value_str}")

    @staticmethod
    def set_output(name: str, value: str):
        """Write GitHub Actions step output."""
        if IS_GHA:
            gh_output = os.getenv("GITHUB_OUTPUT")
            if gh_output:
                with open(gh_output, "a") as f:
                    f.write(f"{name}={value}\n")
            else:
                print(f"::set-output name={name}::{value}")

    @staticmethod
    def step_summary(lines: list[str]):
        """Write to GitHub Actions step summary markdown."""
        if IS_GHA:
            summary_file = os.getenv("GITHUB_STEP_SUMMARY")
            if summary_file:
                with open(summary_file, "a") as f:
                    f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────
# CONFIG (ENV-OVERRIDABLE FOR GHA)
# ─────────────────────────────────────────────

def env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


STREAMS_FILE        = os.getenv("STREAMS_FILE", "streams.json")
OUTPUT_DIR          = os.getenv("OUTPUT_DIR", "thumbnails")
BATCH_SIZE          = env_int("BATCH_SIZE", 50)
TIMEOUT             = env_int("TIMEOUT", 5)
MAX_RETRIES         = env_int("MAX_RETRIES", 1)
MAX_BATCH_WORKERS   = env_int("MAX_BATCH_WORKERS", 3)
MAX_STREAM_WORKERS  = env_int("MAX_STREAM_WORKERS", 6)
TOTAL_MAX_WORKERS   = env_int("TOTAL_MAX_WORKERS", 16)
FRAME_SAMPLE_LIMIT  = env_int("FRAME_SAMPLE_LIMIT", 5)
FAIL_THRESHOLD      = env_int("FAIL_THRESHOLD", 100)   # exit 1 if success% < this


# ─────────────────────────────────────────────
# STATS TRACKER
# ─────────────────────────────────────────────

@dataclass
class Stats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> str:
        s = int(time.time() - self.start_time)
        return f"{s // 60}m {s % 60}s"

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success / self.total * 100


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def load_json(path: str):
    if not os.path.exists(path):
        Log.error(f"Streams file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_online(stream: dict) -> bool:
    return str(stream.get("status", "")).strip().lower() == "online"


def is_black(frame: np.ndarray) -> bool:
    return np.mean(frame) < 8


def chunk_list(data: list, size: int):
    for i in range(0, len(data), size):
        yield data[i:i + size]


# ─────────────────────────────────────────────
# SAFE PYAV EXTRACTION
# ─────────────────────────────────────────────

def extract_frame(url: str) -> np.ndarray | None:
    container = None
    try:
        container = av.open(
            url,
            timeout=TIMEOUT,
            options={
                "fflags":           "nobuffer",
                "flags":            "low_delay",
                "probesize":        "500000",
                "analyzeduration":  "500000",
                "rw_timeout":       "5000000",
            },
        )

        if not container.streams.video:
            Log.debug(f"No video stream: {url}")
            return None

        stream = container.streams.video[0]
        stream.thread_type = "NONE"   # 🔥 prevent segfaults

        for i, frame in enumerate(container.decode(video=0)):
            if i >= FRAME_SAMPLE_LIMIT:
                break
            img = frame.to_ndarray(format="rgb24")
            if is_black(img):
                Log.debug(f"Black frame {i} skipped: {url}")
                continue
            return img

        return None

    except Exception as exc:
        Log.debug(f"extract_frame error [{url}]: {exc}")
        return None

    finally:
        if container:
            try:
                container.close()
            except Exception:
                pass


# ─────────────────────────────────────────────
# STREAM PROCESSING
# ─────────────────────────────────────────────

def process_stream(stream: dict, index: int) -> bool:
    url  = stream.get("url")
    name = safe_name(stream.get("channel", f"stream_{index}"))

    if not url:
        Log.warn(f"Stream #{index} has no URL — skipped")
        return False

    out_path = os.path.join(OUTPUT_DIR, f"{name}.jpg")

    for attempt in range(1, MAX_RETRIES + 1):
        frame = extract_frame(url)
        if frame is not None:
            try:
                Image.fromarray(frame).save(out_path, quality=75)
                Log.debug(f"Saved {out_path}")
                return True
            except Exception as exc:
                Log.warn(f"Save failed [{name}] attempt {attempt}: {exc}")

    Log.debug(f"Failed after {MAX_RETRIES} retries: {name}")
    return False


# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────

def process_batch(batch: list, batch_id: int) -> int:
    workers  = min(MAX_STREAM_WORKERS, len(batch), TOTAL_MAX_WORKERS)
    success  = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_stream, stream, idx): stream
            for idx, stream in enumerate(batch)
        }
        for f in as_completed(futures):
            try:
                if f.result():
                    success += 1
            except Exception as exc:
                Log.debug(f"Batch {batch_id} future error: {exc}")

    return success


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    stats = Stats()

    Log.group("📡 IPTV Thumbnail Capturer")
    Log.info(f"Streams file : {_colorize(STREAMS_FILE, Color.CYAN)}")
    Log.info(f"Output dir   : {_colorize(OUTPUT_DIR, Color.CYAN)}")
    Log.info(f"Batch size   : {_colorize(str(BATCH_SIZE), Color.CYAN)}")
    Log.info(f"Timeout      : {_colorize(str(TIMEOUT) + 's', Color.CYAN)}")
    Log.info(f"GHA mode     : {_colorize(str(IS_GHA), Color.CYAN)}")
    Log.endgroup()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data    = load_json(STREAMS_FILE)
    all_streams = data.get("streams", [])

    online  = [s for s in all_streams if is_online(s)]
    offline = len(all_streams) - len(online)

    stats.total   = len(online)
    stats.skipped = offline

    batches = list(chunk_list(online, BATCH_SIZE))
    batch_workers = min(MAX_BATCH_WORKERS, len(batches))

    Log.group("📊 Stream Summary")
    Log.summary_line("Total in file",   str(len(all_streams)))
    Log.summary_line("Online",          str(len(online)),  ok=len(online) > 0)
    Log.summary_line("Offline/skipped", str(offline),      ok=offline == 0)
    Log.summary_line("Batches",         str(len(batches)))
    Log.summary_line("Batch workers",   str(batch_workers))
    Log.summary_line("Stream workers",  str(MAX_STREAM_WORKERS))
    Log.endgroup()

    if not online:
        Log.warn("No online streams found. Nothing to do.")
        sys.exit(0)

    # ── PARALLEL BATCH EXECUTION ──
    Log.group("🚀 Processing")

    with ThreadPoolExecutor(max_workers=batch_workers) as batch_executor:
        futures = {
            batch_executor.submit(process_batch, batch, i): i
            for i, batch in enumerate(batches, 1)
        }

        with tqdm(
            total=len(futures),
            desc=_colorize("Batches", Color.CYAN, Color.BOLD),
            unit="batch",
            dynamic_ncols=True,
            disable=IS_GHA,   # GHA doesn't render TTY progress bars nicely
        ) as pbar:
            for f in as_completed(futures):
                batch_id = futures[f]
                try:
                    result = f.result()
                    stats.success += result
                    pbar.set_postfix_str(
                        _colorize(f"✔ {stats.success}/{stats.total}", Color.GREEN)
                    )
                except Exception as exc:
                    Log.error(f"Batch {batch_id} crashed: {exc}")
                finally:
                    pbar.update(1)

                # GHA: print per-batch line instead of progress bar
                if IS_GHA:
                    Log.info(f"Batch {batch_id}/{len(batches)} done — running total: {stats.success} captured")

    stats.failed = stats.total - stats.success

    Log.endgroup()

    # ── FINAL REPORT ──
    rate_ok = stats.success_rate >= FAIL_THRESHOLD

    Log.group("📋 Results")
    Log.summary_line("Elapsed",       stats.elapsed)
    Log.summary_line("Total online",  str(stats.total))
    Log.summary_line("Captured",      str(stats.success),         ok=stats.success > 0)
    Log.summary_line("Failed",        str(stats.failed),           ok=stats.failed == 0)
    Log.summary_line("Skipped",       str(stats.skipped),          ok=True)
    Log.summary_line("Success rate",  f"{stats.success_rate:.1f}%", ok=rate_ok)
    Log.endgroup()

    # ── GHA OUTPUTS & SUMMARY ──
    Log.set_output("total",        str(stats.total))
    Log.set_output("success",      str(stats.success))
    Log.set_output("failed",       str(stats.failed))
    Log.set_output("success_rate", f"{stats.success_rate:.1f}")

    Log.step_summary([
        "## 📡 IPTV Thumbnail Capturer Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| ⏱ Elapsed | {stats.elapsed} |",
        f"| 📺 Online streams | {stats.total} |",
        f"| ✅ Captured | {stats.success} |",
        f"| ❌ Failed | {stats.failed} |",
        f"| ⏭ Skipped (offline) | {stats.skipped} |",
        f"| 📊 Success rate | {stats.success_rate:.1f}% |",
    ])

    if not rate_ok:
        Log.error(
            f"Success rate {stats.success_rate:.1f}% is below threshold {FAIL_THRESHOLD}%"
        )
        sys.exit(1)

    Log.success(f"Done! {stats.success}/{stats.total} thumbnails captured in {stats.elapsed}")


if __name__ == "__main__":
    main()
