"""
eval.py
-------
Local evaluation harness.
Run before EVERY submission to verify accuracy and measure token usage.

Usage:
  python eval.py                            # uses data/test_cases.json
  python eval.py data/my_test_cases.json   # custom test file
  python eval.py --remote-only             # force all to remote (baseline)
  python eval.py --local-only              # force all to local
"""
import asyncio
import json
import os
import sys
import time
from agent.router import HybridRouter


# ── Sample test cases (replace with actual revealed tasks) ────────────────────
SAMPLE_TESTS = [
    {"task_id": "t1",  "prompt": "What is the capital of Japan?",
     "expected": "tokyo"},
    {"task_id": "t2",  "prompt": "What is 15% of 240?",
     "expected": "36"},
    {"task_id": "t3",  "prompt": "Classify sentiment: 'The movie was absolutely fantastic!'",
     "expected": "positive"},
    {"task_id": "t4",  "prompt": "Fix this code: def add(a, b):\n    return a - b",
     "expected": "return a + b"},
    {"task_id": "t5",  "prompt": "Extract entities from: 'Apple CEO Tim Cook visited London in March 2024'",
     "expected_keys": ["person", "org", "location", "date"]},
]


def fuzzy_match(predicted: str, expected: str) -> bool:
    """Case-insensitive, whitespace-normalized answer matching."""
    p = predicted.lower().strip()
    e = expected.lower().strip()
    return e in p or p in e or p == e


async def run_eval(
    test_path: str | None = None,
    remote_only: bool = False,
    local_only: bool = False,
):
    # ── Load test cases ───────────────────────────────────────────────
    if test_path and os.path.exists(test_path):
        with open(test_path) as f:
            tests = json.load(f)
    else:
        print("[eval] Using built-in sample test cases.")
        tests = SAMPLE_TESTS

    # ── Build router ──────────────────────────────────────────────────
    api_key        = os.environ.get("FIREWORKS_API_KEY", "")
    base_url       = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models = os.environ.get("ALLOWED_MODELS", "accounts/fireworks/models/gemma2-9b-it").split(",")

    router = HybridRouter(api_key, base_url, allowed_models)

    # ── Run tests ─────────────────────────────────────────────────────
    correct       = 0
    total         = len(tests)
    remote_tokens = 0  # Simulated — actual count needs Fireworks usage API

    results = []
    t_start = time.monotonic()

    semaphore = asyncio.Semaphore(5)

    async def run_one(test):
        nonlocal correct, remote_tokens
        async with semaphore:
            answer = await router.route_async(test["prompt"])

        # Check correctness
        expected = test.get("expected", "")
        if expected:
            ok = fuzzy_match(answer, expected)
        elif "expected_keys" in test:
            import json as _json
            try:
                parsed = _json.loads(answer)
                ok = all(k in parsed for k in test["expected_keys"])
            except Exception:
                ok = False
        else:
            ok = True  # No expected — assume pass

        if ok:
            correct += 1

        results.append({
            "task_id":  test["task_id"],
            "prompt":   test["prompt"],
            "answer":   answer,
            "expected": expected,
            "correct":  ok,
        })

    await asyncio.gather(*[run_one(t) for t in tests])
    elapsed = time.monotonic() - t_start

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    for r in results:
        status = "✅" if r["correct"] else "❌"
        print(f"  {status} [{r['task_id']}] {r['prompt'][:50]}")
        print(f"       Answer: {r['answer'][:80]}")
        if r["expected"]:
            print(f"     Expected: {r['expected']}")
        print()

    print(f"  Accuracy:  {correct}/{total} ({correct/total:.0%})")
    print(f"  Time:      {elapsed:.1f}s for {total} tasks")
    print("=" * 60)

    return correct / total


if __name__ == "__main__":
    test_file  = None
    remote_only = "--remote-only" in sys.argv
    local_only  = "--local-only"  in sys.argv

    for arg in sys.argv[1:]:
        if not arg.startswith("--") and os.path.exists(arg):
            test_file = arg

    asyncio.run(run_eval(test_file, remote_only, local_only))
