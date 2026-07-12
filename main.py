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
    """Process a single task with concurrency control, catching errors to avoid batch crashes."""
    async with semaphore:
        task_id = str(task.get("task_id", "unknown"))
        prompt  = str(task.get("prompt", ""))
        if not prompt:
            return {"task_id": task_id, "answer": "Error: Empty prompt"}

        try:
            start = time.monotonic()
            answer, metadata = await router.route_async(prompt)
            elapsed = time.monotonic() - start
            print(f"  [{elapsed:.2f}s] task_id={task_id} len={len(answer)}", flush=True)
            return {"task_id": task_id, "answer": answer}
        except Exception as e:
            print(f"[ERROR] Failed processing task {task_id}: {e}", file=sys.stderr, flush=True)
            return {"task_id": task_id, "answer": f"Error: {e}"}


async def main():
    # ── Hugging Face Space Auto-detection ──────────────────────────────
    is_hf = "SPACE_ID" in os.environ or "SPACE_REPO_NAME" in os.environ or "HF_SPACE" in os.environ
    if is_hf:
        print("[main] Hugging Face Space detected. Launching Streamlit UI...", flush=True)
        import subprocess
        port = os.environ.get("PORT", "7860")
        subprocess.run([
            "streamlit", "run", "streamlit_app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0"
        ])
        return

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
    default_allowed = (
        "accounts/fireworks/models/minimax-m3,"
        "accounts/fireworks/models/kimi-k2p7-code,"
        "accounts/fireworks/models/gemma-4-31b-it,"
        "accounts/fireworks/models/gemma-4-26b-a4b-it,"
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
    )
    allowed_models = os.environ.get("ALLOWED_MODELS", default_allowed).split(",")

    if not api_key:
        print("[WARNING] FIREWORKS_API_KEY not set — remote calls will fail.", file=sys.stderr)

    # ── Router ─────────────────────────────────────────────────────────
    router = HybridRouter(api_key=api_key, base_url=base_url, allowed_models=allowed_models)

    # Semaphore: limits simultaneous remote API calls.
    # 8 concurrent tasks keeps the pipeline moving faster to avoid 10-minute timeout
    # while still avoiding HTTP 429 Too Many Requests.
    semaphore = asyncio.Semaphore(8)
    t_start   = time.monotonic()

    futures = [process_task(t, router, semaphore) for t in tasks]
    results = await asyncio.gather(*futures)

    elapsed = time.monotonic() - t_start
    print(f"\n[main] Completed {len(results)} tasks in {elapsed:.1f}s", flush=True)

    # ── Write output atomically ───────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(list(results), f, indent=2)
        os.replace(tmp_path, output_path)
        print(f"[main] Results successfully written to {output_path}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed writing output to {output_path}: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())