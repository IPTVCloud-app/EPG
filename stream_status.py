#!/usr/bin/env python3
"""
IPTV Stream Status Checker
Fetches streams from iptv-org API and checks if each is online/viewable.
"""

import asyncio
import aiohttp
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────
API_URL        = "https://iptv-org.github.io/api/streams.json"
CONCURRENCY    = 50       # simultaneous checks
TIMEOUT_SEC    = 8        # per-stream timeout
HEAD_TIMEOUT   = 5        # HEAD request timeout (faster first-pass)
OUTPUT_FILE    = "status.json"
SHOW_OFFLINE   = True    # set True to also print offline streams


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class StreamResult:
    channel:  str
    url:      str
    status:   str          # "online" | "offline" | "error"
    http_code: Optional[int] = None
    latency_ms: Optional[float] = None
    error:    Optional[str] = None
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_viewable(status_code: int) -> bool:
    """2xx / 3xx are considered viewable (redirects are common for HLS)."""
    return 200 <= status_code < 400


async def check_stream(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    channel: str,
    url: str,
) -> StreamResult:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            # Try HEAD first (cheap); fall back to GET for servers that reject HEAD
            for method in ("HEAD", "GET"):
                try:
                    async with session.request(
                        method, url,
                        timeout=aiohttp.ClientTimeout(
                            total=HEAD_TIMEOUT if method == "HEAD" else TIMEOUT_SEC
                        ),
                        allow_redirects=True,
                        ssl=False,          # many IPTV streams use self-signed certs
                    ) as resp:
                        latency = (time.perf_counter() - t0) * 1000
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
                        continue      # retry with GET
                    raise

        except asyncio.TimeoutError:
            return StreamResult(channel=channel, url=url, status="offline", error="timeout")
        except aiohttp.ClientConnectorError as e:
            return StreamResult(channel=channel, url=url, status="offline", error=f"connection: {e}")
        except Exception as e:
            return StreamResult(channel=channel, url=url, status="error", error=str(e))


# ── Progress printer ──────────────────────────────────────────────────────────
def print_result(r: StreamResult, idx: int, total: int) -> None:
    pct   = f"{idx}/{total}"
    icon  = "✅" if r.status == "online" else "❌"
    name  = (r.channel[:35] + "…") if len(r.channel) > 36 else r.channel
    lat   = f"{r.latency_ms}ms" if r.latency_ms else ""
    code  = f"HTTP {r.http_code}" if r.http_code else r.error or ""

    if r.status == "online" or SHOW_OFFLINE:
        print(f"[{pct:>12}] {icon}  {name:<36}  {code:<10}  {lat}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"⬇️  Fetching stream list from iptv-org API …")

    connector = aiohttp.TCPConnector(limit=CONCURRENCY, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:

        # 1. Download stream list
        async with session.get(API_URL, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            raw = await resp.json(content_type=None)

    streams = [
        (entry.get("channel", "unknown"), entry["url"])
        for entry in raw
        if entry.get("url")
    ]

    total = len(streams)
    print(f"📋  {total:,} streams found. Starting checks with {CONCURRENCY} workers …\n")
    print(f"{'':>14}  {'Channel':<36}  {'HTTP':<10}  Latency")
    print("─" * 72)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector2 = aiohttp.TCPConnector(limit=CONCURRENCY, force_close=True)

    results: list[StreamResult] = []

    async with aiohttp.ClientSession(connector=connector2) as session:
        tasks = [
            check_stream(session, semaphore, channel, url)
            for channel, url in streams
        ]

        for idx, coro in enumerate(asyncio.as_completed(tasks), start=1):
            result = await coro
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
        avg_lat = sum(r.latency_ms for r in online if r.latency_ms) / len(online)
        print(f"  ⚡  Avg latency (online): {avg_lat:.0f} ms")
    print("═" * 72)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "checked_at": datetime.utcnow().isoformat(),
        "summary": {
            "total": total,
            "online": len(online),
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
            }
            for r in sorted(results, key=lambda x: (x.status != "online", x.channel))
        ],
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾  Results saved to → {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        sys.exit(0)