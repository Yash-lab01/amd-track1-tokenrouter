"""
main.py
-------
Entry point for the AMD Hackathon submission.

Expected Docker environment:
  - Input:  /input/tasks.json   (injected by eval harness)
  - Output: /output/results.json (read by eval harness)

Environment variables injected by harness:
  - FIREWORKS_API_KEY
  - FIREWORKS_BASE_URL
  - ALLOWED_MODELS  (comma-separated)

Local dev override:
  python main.py <input_path> <output_path>
"""
import asyncio
import json
import os
import sys
import time

from agent.router import HybridRouter


async def process_task(task: dict, router: HybridRouter, semaphore: asyncio.Semaphore) -> dict:
    """Process a single task with concurrency control."""
    async with semaphore:
        start = time.monotonic()
        answer = await router.route_async(task["prompt"])
        elapsed = time.monotonic() - start
        print(f"  [{elapsed:.2f}s] task_id={task['task_id']} domain=? len={len(answer)}", flush=True)
        return {"task_id": task["task_id"], "answer": answer}


async def main():
    # ── Paths ──────────────────────────────────────────────────────────
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "/input/tasks.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/output/results.json"

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load tasks ─────────────────────────────────────────────────────
    with open(input_path) as f:
        tasks = json.load(f)
    print(f"[main] Loaded {len(tasks)} tasks from {input_path}", flush=True)

    # ── Credentials from env (injected by harness) ────────────────────
    api_key        = os.environ.get("FIREWORKS_API_KEY", "")
    base_url       = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models = os.environ.get("ALLOWED_MODELS", "accounts/fireworks/models/gemma2-9b-it").split(",")

    if not api_key:
        print("[WARNING] FIREWORKS_API_KEY not set — remote calls will fail.", file=sys.stderr)

    # ── Router ─────────────────────────────────────────────────────────
    router = HybridRouter(api_key=api_key, base_url=base_url, allowed_models=allowed_models)

    # ── Run concurrently ───────────────────────────────────────────────
    # Semaphore: limits simultaneous remote API calls (I/O bound, can run 10+ concurrently)
    # Local calls are serialized inside the router via ThreadPoolExecutor(max_workers=1)
    semaphore = asyncio.Semaphore(10)
    t_start   = time.monotonic()

    futures = [process_task(t, router, semaphore) for t in tasks]
    results = await asyncio.gather(*futures)

    elapsed = time.monotonic() - t_start
    print(f"\n[main] Completed {len(results)} tasks in {elapsed:.1f}s", flush=True)

    # ── Write output ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(list(results), f, indent=2)

    print(f"[main] Results written to {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
