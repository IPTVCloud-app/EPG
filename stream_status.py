#!/usr/bin/env python3
"""
IPTV Stream Status Checker
Fetches streams from iptv-org API and checks if each is online/viewable.

Usage:
    python stream_status.py [--concurrency 50] [--timeout 8] [--out status.json]
                            [--hide-offline] [--limit 500]
"""

import argparse
import asyncio
import aiohttp
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ── Config defaults ───────────────────────────────────────────────────────────
API_URL      = "https://iptv-org.github.io/api/streams.json"
CONCURRENCY  = 50    # simultaneous checks
TIMEOUT_SEC  = 8     # per-stream timeout (GET fallback)
HEAD_TIMEOUT = 5     # HEAD request timeout (faster first-pass)
OUTPUT_FILE  = "status.json"


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class StreamResult:
    channel:    str
    url:        str
    status:     str                  # "online" | "offline" | "error"
    http_code:  Optional[int]   = None
    latency_ms: Optional[float] = None
    error:      Optional[str]   = None
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_viewable(status_code: int) -> bool:
    """2xx / 3xx are considered viewable (redirects are common for HLS)."""
    return 200 <= status_code < 400


async def check_stream(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    channel:   str,
    url:       str,
    timeout:   float,
    head_timeout: float,
) -> StreamResult:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            for method in ("HEAD", "GET"):
                try:
                    async with session.request(
                        method, url,
                        timeout=aiohttp.ClientTimeout(
                            total=head_timeout if method == "HEAD" else timeout
                        ),
                        allow_redirects=True,
                        ssl=False,   # many IPTV streams use self-signed certs
                    ) as resp:
                        latency = (time.perf_counter() - t0) * 1000

                        # Retry with GET if server rejects HEAD (405) or
                        # returns other client errors that may not apply to GET
                        if method == "HEAD" and resp.status in (405, 501):
                            continue

                        viewable = is_viewable(resp.status)
                        return StreamResult(
                            channel=channel,
                            url=url,
                            status="online" if viewable else "offline",
                            http_code=resp.status,
                            latency_ms=round(latency, 1),
                        )

                except aiohttp.ClientResponseError:
                    if method == "HEAD":
                        continue   # retry with GET
                    raise

        except asyncio.TimeoutError:
            return StreamResult(channel=channel, url=url,
                                status="offline", error="timeout")
        except aiohttp.ClientConnectorError as e:
            return StreamResult(channel=channel, url=url,
                                status="offline", error=f"connection: {e}")
        except Exception as e:
            return StreamResult(channel=channel, url=url,
                                status="error", error=str(e))


# ── Progress printer ──────────────────────────────────────────────────────────
def print_result(r: StreamResult, idx: int, total: int,
                 show_offline: bool) -> None:
    if r.status != "online" and not show_offline:
        return

    pct  = f"{idx}/{total}"
    icon = "✅" if r.status == "online" else "❌"
    name = (r.channel[:35] + "…") if len(r.channel) > 36 else r.channel
    lat  = f"{r.latency_ms}ms" if r.latency_ms is not None else ""
    code = f"HTTP {r.http_code}" if r.http_code is not None else (r.error or "")

    print(f"[{pct:>13}] {icon}  {name:<36}  {code:<12}  {lat}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check IPTV stream availability.")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY,
                   help=f"Parallel workers (default: {CONCURRENCY})")
    p.add_argument("--timeout", type=float, default=TIMEOUT_SEC,
                   help=f"GET timeout in seconds (default: {TIMEOUT_SEC})")
    p.add_argument("--head-timeout", type=float, default=HEAD_TIMEOUT,
                   help=f"HEAD timeout in seconds (default: {HEAD_TIMEOUT})")
    p.add_argument("--out", default=OUTPUT_FILE,
                   help=f"Output JSON file (default: {OUTPUT_FILE})")
    p.add_argument("--hide-offline", action="store_true",
                   help="Suppress offline/error streams from console output")
    p.add_argument("--limit", type=int, default=0,
                   help="Only check the first N streams (0 = all)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args: argparse.Namespace) -> None:
    print("⬇️  Fetching stream list from iptv-org API …")

    # Single shared connector for the whole run
    connector = aiohttp.TCPConnector(limit=args.concurrency, force_close=True)

    async with aiohttp.ClientSession(connector=connector) as session:

        # 1. Download stream list
        try:
            async with session.get(
                API_URL, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                resp.raise_for_status()
                raw = await resp.json(content_type=None)
        except Exception as e:
            print(f"❌  Failed to fetch stream list: {e}", file=sys.stderr)
            sys.exit(1)

        streams = [
            (entry.get("channel", "unknown"), entry["url"])
            for entry in raw
            if entry.get("url")
        ]

        if args.limit > 0:
            streams = streams[: args.limit]

        total = len(streams)
        show_offline = not args.hide_offline

        print(f"📋  {total:,} streams to check "
              f"({args.concurrency} workers, "
              f"timeout {args.timeout}s / HEAD {args.head_timeout}s) …\n")
        print(f"{'':>15}  {'Channel':<36}  {'HTTP':<12}  Latency")
        print("─" * 75)

        semaphore = asyncio.Semaphore(args.concurrency)
        results: list[StreamResult] = []

        tasks = [
            check_stream(session, semaphore, channel, url,
                         args.timeout, args.head_timeout)
            for channel, url in streams
        ]

        for idx, coro in enumerate(asyncio.as_completed(tasks), start=1):
            result = await coro
            results.append(result)
            print_result(result, idx, total, show_offline)

    # ── Summary ───────────────────────────────────────────────────────────────
    online  = [r for r in results if r.status == "online"]
    offline = [r for r in results if r.status != "online"]

    print("\n" + "═" * 75)
    print(f"  ✅  Online  : {len(online):,}")
    print(f"  ❌  Offline : {len(offline):,}")
    print(f"  📊  Total   : {total:,}")

    latencies = [r.latency_ms for r in online if r.latency_ms is not None]
    if latencies:
        print(f"  ⚡  Avg latency (online): {sum(latencies)/len(latencies):.0f} ms")

    print("═" * 75)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total":   total,
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
                "error":      r.error,
                "checked_at": r.checked_at,
            }
            for r in sorted(results, key=lambda x: (x.status != "online", x.channel))
        ],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾  Results saved → {args.out}")


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        sys.exit(0)