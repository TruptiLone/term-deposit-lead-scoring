"""Async load test for the lead-scoring stack.

Measures latency percentiles and throughput at several concurrency levels, either against
Triton directly (to compare dynamic batching on/off) or against the FastAPI gateway.

Run from the repo root, with Triton (and, for --target gateway, the gateway) already running:

    python benchmark/bench.py --target triton --models lead_scorer lead_scorer_nobatch
    python benchmark/bench.py --target gateway
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

# Reuse the gateway's feature encoding so the load test sends realistic, valid inputs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
import app as gw  # noqa: E402

HEADERS = {"Content-Type": "application/json"}


def make_customers(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    customers = []
    for _ in range(n):
        c = {field: rng.choice(gw.CATEGORIES[field]) for field in gw.CATEGORICAL_FIELDS}
        c.update(age=rng.randint(18, 90), balance=rng.randint(-500, 8000), day=rng.randint(1, 28),
                 campaign=rng.randint(1, 6), pdays=-1, previous=0)
        customers.append(c)
    return customers


def triton_body(customer: dict) -> bytes:
    row = gw.encode(gw.Customer(**customer))
    return json.dumps({"inputs": [{"name": "input", "shape": [1, gw.N_FEATURES],
                                   "datatype": "FP32", "data": row}]}).encode()


def percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


async def run_level(client: httpx.AsyncClient, path: str, bodies: list[bytes],
                    concurrency: int, total: int) -> dict:
    """Send `total` requests with `concurrency` requests in flight at once."""
    latencies: list[float] = []
    errors = 0
    next_index = 0

    async def worker() -> None:
        nonlocal next_index, errors
        while True:
            i = next_index
            if i >= total:
                return
            next_index += 1
            t0 = time.perf_counter()
            try:
                resp = await client.post(path, content=bodies[i % len(bodies)], headers=HEADERS)
                ok = resp.status_code == 200
            except httpx.HTTPError:
                ok = False
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if ok:
                latencies.append(elapsed_ms)
            else:
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - start

    latencies.sort()
    if not latencies:
        return {"concurrency": concurrency, "requests": total, "errors": errors,
                "rps": 0.0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "concurrency": concurrency, "requests": total, "errors": errors,
        "rps": round(len(latencies) / wall, 1),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--target", choices=["triton", "gateway"], default="triton")
    parser.add_argument("--url", help="base URL (default: localhost:8000 for triton, :8080 for gateway)")
    parser.add_argument("--models", nargs="+", default=["lead_scorer"], help="Triton model names to test")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--requests", type=int, default=2000, help="requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=100, help="discarded requests before each run")
    parser.add_argument("--json", help="also write raw results to this file")
    args = parser.parse_args()

    url = args.url or ("http://localhost:8000" if args.target == "triton" else "http://localhost:8080")
    customers = make_customers(200)
    if args.target == "triton":
        bodies = [triton_body(c) for c in customers]
        runs = [(m, f"/v2/models/{m}/infer") for m in args.models]
        ready_path = "/v2/health/ready"
    else:
        bodies = [json.dumps(c).encode() for c in customers]
        runs = [("gateway /score", "/score")]
        ready_path = "/health"

    limits = httpx.Limits(max_connections=max(args.concurrency) + 4,
                          max_keepalive_connections=max(args.concurrency) + 4)
    results = []
    async with httpx.AsyncClient(base_url=url, timeout=30.0, limits=limits) as client:
        try:
            if (await client.get(ready_path)).status_code != 200:
                print(f"{url}{ready_path} is not ready", file=sys.stderr)
                return 1
        except httpx.HTTPError as exc:
            print(f"Cannot reach {url}: {exc!r}", file=sys.stderr)
            return 1

        for label, path in runs:
            for concurrency in args.concurrency:
                await run_level(client, path, bodies, min(8, concurrency), args.warmup)  # warm up
                row = await run_level(client, path, bodies, concurrency, args.requests)
                row["name"] = label
                results.append(row)
                print(f"done: {label} @ concurrency {concurrency}", file=sys.stderr)

    print("\n| Name | Concurrency | Requests | Errors | Req/s | p50 ms | p95 ms | p99 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['concurrency']} | {r['requests']} | {r['errors']} | {r['rps']} "
              f"| {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} |")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nraw results written to {args.json}", file=sys.stderr)
    return 1 if any(r["errors"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))