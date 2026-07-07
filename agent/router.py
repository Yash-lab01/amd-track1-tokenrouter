"""
router.py
---------
Three-tier hybrid routing engine.

Tier 0 — Direct math solver (0 tokens, no model needed at all)
Tier 1 — TF-IDF domain classifier  (0 tokens, <1ms)
Tier 2 — Local model + programmatic validator (0 remote tokens)
Tier 3 — Remote Fireworks AI via speculative correction (minimal tokens)
"""
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from agent.classifier  import DomainClassifier
from agent.evaluator   import validate, try_solve_locally
from agent.local_model import LocalModel
from agent.remote_model import RemoteModel

# Confidence threshold for trusting the domain classifier
CLASSIFIER_CONF_THRESHOLD = 0.65

# Domains where we skip local model entirely (direct solve or ultra-safe remote)
ALWAYS_LOCAL_DOMAINS = {"sentiment"}


class HybridRouter:
    def __init__(self, api_key: str, base_url: str, allowed_models: list[str]):
        self.classifier = DomainClassifier()
        self.local      = LocalModel()
        self.remote     = RemoteModel(api_key, base_url, allowed_models)

        # Single-worker executor keeps local CPU calls sequential (prevents thrashing)
        self._local_executor = ThreadPoolExecutor(max_workers=1)

        # LRU cache for exact-match prompt deduplication
        self._answer_cache: dict[str, str] = {}

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    async def route_async(self, prompt: str) -> str:
        """
        Main routing coroutine. Returns the answer string.
        """
        # ── Cache hit (exact match) ────────────────────────────────────
        if prompt in self._answer_cache:
            return self._answer_cache[prompt]

        # ── Tier 0: Programmatic solver (math, etc.) ───────────────────
        domain, conf = self.classifier.classify(prompt)
        direct = try_solve_locally(domain, prompt)
        if direct is not None:
            self._answer_cache[prompt] = direct
            return direct

        # ── Tier 1: Local model ────────────────────────────────────────
        loop = asyncio.get_event_loop()
        local_answer = await loop.run_in_executor(
            self._local_executor,
            self.local.generate,
            prompt,
            domain,
        )

        # ── Validate local output ──────────────────────────────────────
        is_valid, cleaned = validate(domain, prompt, local_answer)

        if is_valid:
            self._answer_cache[prompt] = cleaned
            return cleaned

        # ── Tier 2: Speculative remote correction ──────────────────────
        # For code/logic — send draft, ask remote to verify/fix (saves tokens)
        if domain in ("debugging", "codegen", "logic", "ner"):
            remote_answer = await self.remote.speculative_correct(
                prompt, local_answer, domain
            )
        else:
            # Full remote generation for summarization / factual fallbacks
            remote_answer = await self.remote.generate(prompt, domain)

        self._answer_cache[prompt] = remote_answer
        return remote_answer
