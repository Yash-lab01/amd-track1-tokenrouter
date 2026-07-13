"""
router.py
---------
Pure Remote-First Token-Efficient Router.

Strategy:
1. Tier 0: Deterministic solvers (0 tokens) — math, facts, sentiment rules, code fixes
2. Tier 1: Remote Fireworks AI (remote tokens) — ALL other domains
3. Emergency: Local model (loaded at startup) if remote completely fails

Local model is loaded at startup and always available for emergency fallback.
"""
import asyncio
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from agent.classifier import DomainClassifier
from agent.evaluator import postprocess, try_solve_locally, validate
from agent.remote_model import RemoteModel
from agent.trap_detector import TrapDetector

# Local model is optional — lazy-loaded only for emergency fallback.
_LOCAL_MODEL_AVAILABLE = True
try:
    from agent.local_model import LocalModel
except (ImportError, ModuleNotFoundError):
    LocalModel = None
    _LOCAL_MODEL_AVAILABLE = False
    print("[Router] llama_cpp not available — running in remote-only mode (no local fallback).")


CLASSIFIER_CONF_THRESHOLD = 0.65

# Domains that should retry on validation failure (malformed output only)
RETRY_DOMAINS = {"ner", "debugging", "codegen", "logic"}

_SPATIAL_CONSTRAINT_KEYWORDS = [
    "immediately to the left", "immediately to the right",
    "directly to the left", "directly to the right",
    "left of", "right of", "adjacent to", "next to",
    "in a row", "sit in a row", "sit in a line", "stand in a line",
    "sits beside", "sit beside", "seated beside",
    "rightmost", "leftmost", "far left", "far right",
    "at one end", "at either end", "in position",
    "directly behind", "immediately behind", "directly in front of"
]

def _is_spatial_puzzle(prompt: str) -> bool:
    lowered = prompt.lower()
    if any(marker in lowered for marker in ["function", "python", "code", "implement", "def ", "class ", "algorithm", "bug"]):
        return False
    return any(keyword in lowered for keyword in _SPATIAL_CONSTRAINT_KEYWORDS)

_HALLUCINATION_TRAP_KEYWORDS = [
    "knight and knave", "always lie", "always tell the truth",
    "riddle", "60 more"
]

def _is_hallucination_trap(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(keyword in lowered for keyword in _HALLUCINATION_TRAP_KEYWORDS)


def _normalize_cache_prompt(prompt: str) -> str:
    """Cache duplicate prompts without changing the prompt sent to models."""
    return re.sub(r"\s+", " ", prompt.strip())


def _prompt_asks_explanation(prompt: str) -> bool:
    """Check if a factual prompt asks for explanation, not just a one-word answer."""
    lowered = prompt.lower()
    return bool(re.search(
        r"\b(explain|describe|analyze|compare|contrast|why|how does|elaborate|"
        r"what is the difference|briefly explain|in detail)\b",
        lowered
    ))



class HybridRouter:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        allowed_models: list[str],
        mode: str = "hybrid",
    ):
        self.classifier = DomainClassifier()
        # Load local model at startup for emergency fallback
        self._local_model = None
        if _LOCAL_MODEL_AVAILABLE and LocalModel is not None:
            try:
                self._local_model = LocalModel()
                print("[Router] Local model loaded at startup for emergency fallback.", flush=True)
            except Exception as e:
                print(f"[Router] Failed to load local model: {e}. Running without local fallback.", flush=True)
                self._local_model = None
        else:
            print("[Router] Local model not available — running without local fallback.", flush=True)
        self.remote = RemoteModel(api_key, base_url, allowed_models)
        self.traps = TrapDetector()
        self.mode = mode

        self._local_executor = ThreadPoolExecutor(max_workers=1)
        self._active_local_tasks = 0
        self._answer_cache: dict[str, tuple[str, dict]] = {}

    def _get_local_model(self):
        """Return the pre-loaded local model for emergency fallback."""
        if self._local_model is None:
            raise RuntimeError("Local model not available")
        return self._local_model

    async def route_async(self, prompt: str) -> tuple[str, dict]:
        """Route one prompt and return (answer_string, metadata_dict).

        Pure Remote-First flow:
        1. Check cache
        2. Classify domain
        3. Check for traps (spatial/hallucination) -> force remote
        4. Tier 0: Deterministic solver (exact only) -- 0 tokens
        5. Tier 1: Remote accuracy engine (ALL domains)
        6. Validate + postprocess
        7. Retry only on malformed NER/code/logic
        8. Emergency local fallback if remote fails (lazy-loaded)
        """
        deadline = time.monotonic() + 28.0
        cache_key = f"{self.mode}:{_normalize_cache_prompt(prompt)}"
        if cache_key in self._answer_cache:
            return self._answer_cache[cache_key]

        domain, conf = self.classifier.classify(prompt)

        is_semantic_trap = False
        trap_reason, trap_score = "", 0.0
        if os.environ.get("ENABLE_TFIDF_TRAP_DETECTOR", "0") == "1":
            is_semantic_trap, trap_reason, trap_score = self.traps.is_trap(prompt)

        # SPATIAL PUZZLE & HALLUCINATION INTERCEPT:
        if _is_spatial_puzzle(prompt) or _is_hallucination_trap(prompt) or is_semantic_trap:
            domain = "logic"
            conf = 1.0

        # -- Tier 0: Deterministic solver (exact only) --
        direct = try_solve_locally(domain, prompt)
        if direct is not None:
            metadata = self._trace(domain, conf, "direct", "deterministic", model="local")
            self._answer_cache[cache_key] = (direct, metadata)
            return direct, metadata

        # -- Tier 1: Remote accuracy engine (ALL non-deterministic tasks) --
        answer, metadata = await self._remote_with_retry(
            prompt, domain, conf, deadline
        )
        self._answer_cache[cache_key] = (answer, metadata)
        return answer, metadata

    async def _remote_with_retry(
        self,
        prompt: str,
        domain: str,
        conf: float,
        deadline: float,
    ) -> tuple[str, dict]:
        """Remote generation with one retry on malformed output."""
        try:
            answer, model = await self.remote.generate(prompt, domain, conf=conf)
            is_valid, cleaned = validate(domain, prompt, answer)

            # Retry only for domains with strict format requirements (malformed output)
            should_retry = (not answer.strip()) or (not is_valid and domain in RETRY_DOMAINS)
            if should_retry:
                try:
                    retry_answer, retry_model = await self.remote.generate_correction(
                        prompt, domain, cleaned, conf=conf
                    )
                    retry_valid, retry_cleaned = validate(domain, prompt, retry_answer)
                    if retry_valid:
                        metadata = self._trace(domain, conf, "remote", "retry-success", model=retry_model)
                        return retry_cleaned, metadata
                    # Use retry output if it's non-empty, else fall back to first
                    retry_cleaned = postprocess(domain, retry_answer)
                    if retry_cleaned.strip():
                        cleaned = retry_cleaned
                        model = retry_model
                except Exception as retry_exc:
                    print(f"[WARNING] Retry failed: {retry_exc}", flush=True)
            elif not is_valid:
                cleaned = postprocess(domain, answer)

            # Safety net: never return empty string
            if not cleaned.strip():
                print("[WARNING] Remote returned empty after postprocess.", flush=True)
                cleaned = "Unable to determine"

            metadata = self._trace(domain, conf, "remote", "remote-first", model=model)
            return cleaned, metadata

        except Exception as exc:
            # Remote completely failed -- emergency local fallback
            print(f"[WARNING] Remote call failed: {exc}. Emergency local fallback.", flush=True)
            try:
                local_ans = await self._generate_local(prompt, domain, deadline=deadline)
                is_val, cleaned_local = validate(domain, prompt, local_ans)
                fallback = cleaned_local if is_val else postprocess(domain, local_ans)
            except Exception as local_exc:
                print(f"[WARNING] Local fallback also failed: {local_exc}", flush=True)
                fallback = "Unable to determine"
            metadata = self._trace(domain, conf, "local", "remote-failed", model="local")
            return fallback, metadata

    # Timeout for local model -- emergency fallback only
    _LOCAL_TIMEOUT_S = 10

    async def _generate_local(self, prompt: str, domain: str, temperature: float = 0.1, deadline: float = 0.0, timeout: float = 0.0) -> str:
        """Local model generation -- used for emergency fallback only. Lazy-loads model."""
        local = self._get_local_model()
        if local is None:
            raise RuntimeError("Local model not available")

        if deadline > 0 and time.monotonic() > deadline - 4.0:
            raise asyncio.TimeoutError("Time budget low")

        if self._active_local_tasks >= 2:
            raise asyncio.TimeoutError("Local queue full")

        actual_timeout = timeout if timeout > 0 else self._LOCAL_TIMEOUT_S

        self._active_local_tasks += 1
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._local_executor,
                    local.generate,
                    prompt,
                    domain,
                    temperature
                ),
                timeout=actual_timeout,
            )
        except asyncio.TimeoutError:
            raise
        finally:
            self._active_local_tasks -= 1

    def _trace(
        self,
        domain: str,
        conf: float,
        tier: str,
        reason: str,
        model: str = "",
    ) -> dict:
        model_part = f" model={model}" if model else ""
        print(
            f"[route] domain={domain} conf={conf:.3f} tier={tier} reason={reason}{model_part}",
            flush=True,
        )
        return {
            "domain": domain,
            "conf": conf,
            "tier": tier,
            "reason": reason,
            "model": model,
        }
