#!/usr/bin/env python3
"""
IPTV Stream Status Checker
Fetches streams from iptv-org API and checks if each is online/viewable.
"""

import asyncio
import aiohttp
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse


# ── Config ────────────────────────────────────────────────────────────────────
API_URL        = "https://iptv-org.github.io/api/streams.json"
CONCURRENCY    = 50       # simultaneous checks
TIMEOUT_SEC    = 8        # per-stream GET timeout (seconds)
HEAD_TIMEOUT   = 5        # HEAD request timeout (seconds)
API_TIMEOUT    = 30       # timeout for fetching the API stream list
OUTPUT_FILE    = "stream_results.json"
SHOW_OFFLINE   = False    # set True to also print offline streams
MAX_REDIRECTS  = 10       # redirect limit per stream


# ── Logging (warnings+ to stderr, keeps stdout clean) ─────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class StreamResult:
    channel:    str
    url:        str
    status:     str               # "online" | "offline"
    http_code:  Optional[int]   = None
    latency_ms: Optional[float] = None
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_viewable(status_code: int) -> bool:
    """2xx / 3xx responses are considered viewable (HLS streams commonly redirect)."""
    return 200 <= status_code < 400


def is_valid_url(url: str) -> bool:
    """Reject obviously malformed URLs before attempting a connection."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def make_offline(channel: str, url: str, http_code: Optional[int] = None) -> StreamResult:
    return StreamResult(channel=channel, url=url, status="offline", http_code=http_code)


# ── Per-stream checker ────────────────────────────────────────────────────────
async def check_stream(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    channel: str,
    url: str,
) -> StreamResult:
    """
    Try HEAD → GET for each stream.
    Every known failure mode is caught and mapped to "offline".
    """
    if not is_valid_url(url):
        return make_offline(channel, url)

    async with semaphore:
        t0 = time.perf_counter()

        for method in ("HEAD", "GET"):
            timeout = aiohttp.ClientTimeout(
                total=HEAD_TIMEOUT if method == "HEAD" else TIMEOUT_SEC,
                connect=5,
                sock_connect=5,
                sock_read=TIMEOUT_SEC,
            )
            try:
                async with session.request(
                    method, url,
                    timeout=timeout,
                    allow_redirects=True,
                    max_redirects=MAX_REDIRECTS,
                    ssl=False,          # many streams use self-signed certs
                ) as resp:
                    latency = (time.perf_counter() - t0) * 1000
                    return StreamResult(
                        channel=channel,
                        url=url,
                        status="online" if is_viewable(resp.status) else "offline",
                        http_code=resp.status,
                        latency_ms=round(latency, 1),
                    )

            # ── HEAD rejected → retry with GET ────────────────────────────────
            except aiohttp.ClientResponseError as e:
                if method == "HEAD":
                    continue
                return make_offline(channel, url, http_code=e.status)

            # ── Redirect loop ──────────────────────────────────────────────────
            except aiohttp.TooManyRedirects:
                return make_offline(channel, url)

            # ── DNS / TCP connection failures ──────────────────────────────────
            except (
                aiohttp.ClientConnectorError,
                aiohttp.ClientConnectorDNSError,
                aiohttp.ClientConnectorSSLError,
                aiohttp.ClientConnectorCertificateError,
            ):
                return make_offline(channel, url)

            # ── Server-side connection problems ────────────────────────────────
            except (
                aiohttp.ServerDisconnectedError,
                aiohttp.ServerConnectionError,
                aiohttp.ServerTimeoutError,
            ):
                return make_offline(channel, url)

            # ── Timeout (connect or read) ──────────────────────────────────────
            except asyncio.TimeoutError:
                return make_offline(channel, url)

            # ── DNS resolution failure ─────────────────────────────────────────
            except socket.gaierror:
                return make_offline(channel, url)

            # ── Payload / encoding errors ──────────────────────────────────────
            except (aiohttp.ClientPayloadError, UnicodeDecodeError):
                return make_offline(channel, url)

            # ── OS-level socket errors ─────────────────────────────────────────
            except (aiohttp.ClientOSError, OSError):
                return make_offline(channel, url)

            # ── Malformed URL passed through ───────────────────────────────────
            except (aiohttp.InvalidURL, ValueError):
                return make_offline(channel, url)

            # ── Task cancellation — propagate, do not swallow ──────────────────
            except asyncio.CancelledError:
                raise

            # ── Absolute safety net ────────────────────────────────────────────
            except Exception as e:
                log.warning("Unexpected error for %s: %s: %s", url, type(e).__name__, e)
                return make_offline(channel, url)

    # Exhausted both methods without returning (should not happen)
    return make_offline(channel, url)


# ── Progress printer ──────────────────────────────────────────────────────────
def print_result(r: StreamResult, idx: int, total: int) -> None:
    if r.status != "online" and not SHOW_OFFLINE:
        return
    icon = "✅" if r.status == "online" else "❌"
    ch   = r.channel or "unknown"
    name = (ch[:35] + "…") if len(ch) > 36 else ch
    lat  = f"{r.latency_ms}ms" if r.latency_ms is not None else ""
    code = f"HTTP {r.http_code}" if r.http_code is not None else "—"
    pct  = f"{idx}/{total}"
    print(f"[{pct:>12}] {icon}  {name:<36}  {code:<10}  {lat}")


# ── API fetch ─────────────────────────────────────────────────────────────────
async def fetch_stream_list() -> list[tuple[str, str]]:
    """Download and parse the iptv-org stream list. Returns (channel, url) pairs."""
    connector = aiohttp.TCPConnector(force_close=True)
    timeout   = aiohttp.ClientTimeout(total=API_TIMEOUT)

    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(API_URL) as resp:
                resp.raise_for_status()
                try:
                    raw = await resp.json(content_type=None)
                except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                    sys.exit(f"❌  Failed to parse API response as JSON: {e}")

    except aiohttp.ClientConnectorError as e:
        sys.exit(f"❌  Cannot reach API ({API_URL}): {e}")
    except asyncio.TimeoutError:
        sys.exit(f"❌  API request timed out after {API_TIMEOUT}s.")
    except aiohttp.ClientResponseError as e:
        sys.exit(f"❌  API returned HTTP {e.status}: {e.message}")
    except Exception as e:
        sys.exit(f"❌  Unexpected error fetching API: {type(e).__name__}: {e}")

    if not isinstance(raw, list) or len(raw) == 0:
        sys.exit("❌  API returned an empty or unexpected payload.")

    streams = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not url or not isinstance(url, str) or not url.strip():
            continue
        channel = (entry.get("channel") or "unknown").strip() or "unknown"
        streams.append((channel, url.strip()))

    if not streams:
        sys.exit("❌  No valid stream URLs found in API response.")

    return streams


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(results: list[StreamResult]) -> None:
    online  = [r for r in results if r.status == "online"]
    offline = [r for r in results if r.status != "online"]

    output = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total":   len(results),
            "online":  len(online),
            "offline": len(offline),
        },
        "streams": [
            {
                "channel":    r.channel,
                "url":        r.url,
                "status":     r.status,
                "http_code":  r.http_code,
                "latency_ms": r.latency_ms,
            }
            for r in sorted(results, key=lambda x: (x.status != "online", x.channel or ""))
        ],
    }

    try:
        tmp = OUTPUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        os.replace(tmp, OUTPUT_FILE)   # atomic write — avoids partial file on crash
    except OSError as e:
        print(f"\n⚠️  Could not save results to {OUTPUT_FILE}: {e}", file=sys.stderr)
        return

    print(f"\n💾  Results saved → {OUTPUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    print("⬇️  Fetching stream list from iptv-org API …")
    streams = await fetch_stream_list()
    total   = len(streams)

    print(f"📋  {total:,} streams found. Starting checks "
          f"({CONCURRENCY} workers, timeout {TIMEOUT_SEC}s / HEAD {HEAD_TIMEOUT}s) …\n")
    print(f"{'':>14}  {'Channel':<36}  {'HTTP':<10}  Latency")
    print("─" * 72)

    semaphore  = asyncio.Semaphore(CONCURRENCY)
    connector  = aiohttp.TCPConnector(limit=CONCURRENCY, force_close=True, enable_cleanup_closed=True)
    results: list[StreamResult] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            check_stream(session, semaphore, channel, url)
            for channel, url in streams
        ]
        for idx, coro in enumerate(asyncio.as_completed(tasks), start=1):
            try:
                result = await coro
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Should never reach here, but log and continue
                log.warning("Unhandled task exception at index %d: %s", idx, e)
                continue
            results.append(result)
            print_result(result, idx, total)

    # ── Summary ───────────────────────────────────────────────────────────────
    online  = [r for r in results if r.status == "online"]
    offline = [r for r in results if r.status != "online"]

    print("\n" + "═" * 72)
    print(f"  ✅  Online  : {len(online):,}")
    print(f"  ❌  Offline : {len(offline):,}")
    print(f"  📊  Total   : {total:,}")
    if online:
        lats    = [r.latency_ms for r in online if r.latency_ms is not None]
        avg_lat = sum(lats) / len(lats) if lats else 0
        print(f"  ⚡  Avg latency (online): {avg_lat:.0f} ms")
    print("═" * 72)

    save_results(results)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.", file=sys.stderr)
        sys.exit(0)
    except MemoryError:
        sys.exit("❌  Out of memory — try reducing CONCURRENCY.")
    except Exception as e:
        sys.exit(f"❌  Fatal error: {type(e).__name__}: {e}")