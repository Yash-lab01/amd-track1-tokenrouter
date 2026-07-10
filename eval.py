"""
eval.py
-------
Local evaluation harness.

Usage:
  python eval.py
  python eval.py data/my_test_cases.json
  python eval.py --remote-only
  python eval.py --local-only
"""
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv

from agent.router import HybridRouter

load_dotenv()


SAMPLE_TESTS = [
    {
        "task_id": "t1",
        "prompt": "What is the capital of Japan?",
        "expected": "tokyo",
        "domain": "factual",
    },
    {
        "task_id": "t2",
        "prompt": "What is 15% of 240?",
        "expected": "36",
        "domain": "math",
    },
    {
        "task_id": "t3",
        "prompt": "Classify sentiment: 'The movie was absolutely fantastic!'",
        "expected": "positive",
        "domain": "sentiment",
    },
    {
        "task_id": "t4",
        "prompt": "Fix this code: def add(a, b):\n    return a - b",
        "expected": "return a + b",
        "domain": "debugging",
    },
    {
        "task_id": "t5",
        "prompt": "Extract entities from: 'Apple CEO Tim Cook visited London in March 2024'",
        "expected_keys": ["person", "org", "location", "date"],
        "domain": "ner",
    },
]


def fuzzy_match(predicted: str, expected: str) -> bool:
    """Case-insensitive, whitespace-normalized answer matching."""
    p = predicted.lower().strip()
    e = expected.lower().strip()
    if not p or not e:
        return False
    return e in p or p in e or p == e


def check_correct(test: dict, answer: str) -> bool:
    expected = test.get("expected", "")
    if expected:
        return fuzzy_match(answer, expected)
    if "expected_keys" in test:
        try:
            parsed = json.loads(answer)
            return all(k in parsed for k in test["expected_keys"])
        except Exception:
            return False
    return True


async def run_eval(
    test_path: str | None = None,
    remote_only: bool = False,
    local_only: bool = False,
):
    if test_path and os.path.exists(test_path):
        with open(test_path, encoding="utf-8") as f:
            tests = json.load(f)
    else:
        print("[eval] Using built-in sample test cases.")
        tests = SAMPLE_TESTS

    if remote_only and local_only:
        raise ValueError("Use only one of --remote-only or --local-only.")

    mode = "hybrid"
    if remote_only:
        mode = "remote"
    elif local_only:
        mode = "local"
    print(f"[eval] Router mode: {mode}")

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    default_allowed = (
        "accounts/fireworks/models/minimax-m3,"
        "accounts/fireworks/models/kimi-k2p7-code,"
        "accounts/fireworks/models/gemma-4-31b-it,"
        "accounts/fireworks/models/gemma-4-26b-a4b-it,"
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
    )
    allowed_models = os.environ.get("ALLOWED_MODELS", default_allowed).split(",")
    router = HybridRouter(api_key, base_url, allowed_models, mode=mode)

    correct = 0
    total = len(tests)
    results = []
    domain_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    t_start = time.monotonic()
    semaphore = asyncio.Semaphore(5)

    async def run_one(test):
        nonlocal correct
        async with semaphore:
            answer, metadata = await router.route_async(test["prompt"])

        ok = check_correct(test, answer)
        if ok:
            correct += 1

        domain = test.get("domain", "unknown")
        domain_stats[domain]["total"] += 1
        if ok:
            domain_stats[domain]["correct"] += 1

        results.append(
            {
                "task_id": test["task_id"],
                "domain": domain,
                "prompt": test["prompt"],
                "answer": answer,
                "expected": test.get("expected", ""),
                "correct": ok,
                "metadata": metadata,
            }
        )

    await asyncio.gather(*[run_one(t) for t in tests])
    elapsed = time.monotonic() - t_start

    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"  {status} [{r['task_id']}] ({r['domain']}) {r['prompt'][:50]}")
        print(f"       Answer: {r['answer'][:120]}")
        if r["expected"]:
            print(f"     Expected: {r['expected']}")
        print()

    print(f"  Accuracy:  {correct}/{total} ({correct / total:.0%})")
    print(f"  Time:      {elapsed:.1f}s for {total} tasks")
    print("\n  Per-domain accuracy:")
    for domain in sorted(domain_stats):
        stats = domain_stats[domain]
        pct = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"    {domain:15s} {stats['correct']}/{stats['total']} ({pct:.0%})")
    print("=" * 60)

    report_path = os.path.join(os.path.dirname(__file__), "data", "eval_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "accuracy": correct / total if total else 0,
                "correct": correct,
                "total": total,
                "elapsed_s": elapsed,
                "domain_stats": dict(domain_stats),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[eval] Full results saved to {report_path}")

    return correct / total if total else 0


if __name__ == "__main__":
    test_file = None
    remote_only = "--remote-only" in sys.argv
    local_only = "--local-only" in sys.argv

    for arg in sys.argv[1:]:
        if not arg.startswith("--") and os.path.exists(arg):
            test_file = arg

    asyncio.run(run_eval(test_file, remote_only, local_only))
