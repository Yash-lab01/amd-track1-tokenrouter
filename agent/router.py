"""
router.py
---------
Remote-First Accuracy Router.

Per HACKATHON_WINNING_PLAN.md:
- Convert routing to remote-first for all non-exact tasks.
- Disable local LLM generation from the normal path.
- Keep exact deterministic solvers before remote.
- Use domain-specific remote prompts.
- Use model specialization aggressively.
- Retry only NER/code/logic malformed outputs.
- Preserve full prompts for reasoning and extraction tasks.
- Keep postprocessing strict.
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

# Local model is optional — only used as emergency fallback.
# If llama_cpp is not installed, the router runs in remote-only mode.
try:
    from agent.local_model import LocalModel
    _LOCAL_MODEL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    LocalModel = None
    _LOCAL_MODEL_AVAILABLE = False
    print("[Router] llama_cpp not available — running in remote-only mode (no local fallback).")


CLASSIFIER_CONF_THRESHOLD = 0.65
# No domains trust the local model for generation — remote-first.
LOCAL_TRUST_DOMAINS = set()
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



class HybridRouter:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        allowed_models: list[str],
        mode: str = "hybrid",
    ):
        self.classifier = DomainClassifier()
        self.local = LocalModel() if _LOCAL_MODEL_AVAILABLE else None
        self.remote = RemoteModel(api_key, base_url, allowed_models)
        self.traps = TrapDetector()
        self.mode = mode

        self._local_executor = ThreadPoolExecutor(max_workers=1)
        self._active_local_tasks = 0
        self._answer_cache: dict[str, tuple[str, dict]] = {}

    async def route_async(self, prompt: str) -> tuple[str, dict]:
        """Route one prompt and return (answer_string, metadata_dict).

        Remote-First flow:
        1. Check cache
        2. Classify domain
        3. Check for traps (spatial/hallucination) → force remote
        4. Try deterministic solver (Tier 0) — exact only
        5. Route to remote (Tier 1 — accuracy engine)
        6. Validate + postprocess
        7. Retry only on malformed NER/code/logic
        8. Local model only as emergency fallback if remote fails
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
        # If detected, force the domain to 'logic' (REMOTE) to bypass local.
        if _is_spatial_puzzle(prompt) or _is_hallucination_trap(prompt) or is_semantic_trap:
            domain = "logic"
            conf = 1.0

        # ── Tier 0: Deterministic solver (exact only) ──────────────────
        direct = try_solve_locally(domain, prompt)
        if direct is not None:
            metadata = self._trace(domain, conf, "direct", "deterministic", model="local")
            self._answer_cache[cache_key] = (direct, metadata)
            return direct, metadata

        # ── Tier 1: Remote accuracy engine ─────────────────────────────
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

            # Retry only for domains with strict format requirements
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
                print("[WARNING] Remote returned empty after postprocess. Trying local fallback.", flush=True)
                try:
                    local_ans = await self._generate_local(prompt, domain, deadline=deadline)
                    cleaned = postprocess(domain, local_ans)
                except Exception as local_exc:
                    print(f"[WARNING] Local fallback also failed: {local_exc}", flush=True)
                    cleaned = "Unable to determine"

            metadata = self._trace(domain, conf, "remote", "remote-first", model=model)
            return cleaned, metadata

        except Exception as exc:
            # Remote completely failed — emergency local fallback
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

    # Max seconds to wait for local model inference (emergency fallback only)
    _LOCAL_TIMEOUT_S = 12

    async def _generate_local(self, prompt: str, domain: str, temperature: float = 0.1, deadline: float = 0.0) -> str:
        """Emergency local fallback only — not used in normal routing."""
        if self.local is None:
            raise RuntimeError("Local model not available")

        if deadline > 0 and time.monotonic() > deadline - 4.0:
            raise asyncio.TimeoutError("Time budget low")

        if self._active_local_tasks >= 2:
            raise asyncio.TimeoutError("Local queue full")

        self._active_local_tasks += 1
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._local_executor,
                    self.local.generate,
                    prompt,
                    domain,
                    temperature
                ),
                timeout=self._LOCAL_TIMEOUT_S,
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