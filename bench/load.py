#!/usr/bin/env python3
"""Concurrent load probe for a running decision-toolkit server.

Start the server first, then:

    PYTHONPATH=src python bench/load.py \
      --base-url http://127.0.0.1:8011 \
      --model gliner2.5-multi \
      --concurrency 10 --requests 50 \
      --output bench/results/2026-10-20-load.json

The probe deliberately does not start a model itself. It reports status mix,
latency percentiles and throughput; 429/503 are expected when rate limiting or
backpressure is configured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError as error:  # pragma: no cover - optional dev dependency
    raise SystemExit("bench/load.py needs httpx: pip install httpx") from error

STATE = "I was charged twice for my order last week and nobody replied."
QUESTIONS = {
    "urgent": {
        "type": "noul",
        "instructions": "Does this need a reply today?",
    }
}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


async def probe(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if args.api_key:
        headers["authorization"] = f"Bearer {args.api_key}"
    payload = {"model": args.model, "state": STATE, "questions": QUESTIONS}
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    statuses: Counter[int] = Counter()
    latencies: list[float] = []
    errors: list[str] = []
    retry_after = 0
    warmup_status: int | None = None

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        headers=headers,
    ) as client:
        try:
            warmup = await client.post("/v1/systemone", json=payload)
            warmup_status = warmup.status_code
        except Exception as error:  # noqa: BLE001 - report connectivity
            errors.append(f"warmup: {type(error).__name__}: {error}")

        async def one(index: int) -> None:
            nonlocal retry_after
            async with semaphore:
                started = asyncio.get_running_loop().time()
                try:
                    response = await client.post("/v1/systemone", json=payload)
                    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000.0
                    statuses[response.status_code] += 1
                    latencies.append(elapsed_ms)
                    if "retry-after" in response.headers:
                        retry_after += 1
                except Exception as error:  # noqa: BLE001 - report connectivity
                    errors.append(f"request {index}: {type(error).__name__}: {error}")

        started = asyncio.get_running_loop().time()
        await asyncio.gather(*(one(index) for index in range(args.requests)))
        elapsed = asyncio.get_running_loop().time() - started

    success = statuses[200]
    ok = success > 0 and statuses[500] == 0 and not errors
    return {
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "server_settings": {
            "max_concurrent": args.server_max_concurrent,
            "rate_limit": args.server_rate_limit,
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "warmup_status": warmup_status,
        "status_counts": {str(code): count for code, count in sorted(statuses.items())},
        "errors": errors[:20],
        "retry_after_count": retry_after,
        "latency_ms": {
            "n": len(latencies),
            "p50": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "elapsed_seconds": round(elapsed, 3),
        "throughput_rps": round(args.requests / elapsed, 3) if elapsed else None,
        "ok": ok,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="gliner2.5-multi")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--server-max-concurrent",
        type=int,
        default=4,
        help="Expected server --max-concurrent, recorded in the report.",
    )
    parser.add_argument(
        "--server-rate-limit",
        type=int,
        default=0,
        help="Expected server --rate-limit, recorded in the report.",
    )
    parser.add_argument("--api-key")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = asyncio.run(probe(args))
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
