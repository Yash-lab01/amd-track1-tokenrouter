"""
router.py
---------
Accuracy-first hybrid routing engine.

The leaderboard gates on accuracy before token efficiency, so this router only
trusts local answers when a deterministic solver or strict validator can prove
the output shape. Risky domains go remote first with short prompts.
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from agent.classifier import DomainClassifier
from agent.evaluator import postprocess, try_solve_locally, validate
from agent.local_model import LocalModel
from agent.remote_model import RemoteModel


CLASSIFIER_CONF_THRESHOLD = 0.65
# NER removed from LOCAL_TRUST: on 2 vCPU, generating 60-100 token JSON takes
# 12-30 seconds per task and serialises through max_workers=1, causing TIMEOUT.
# Remote NER with strict JSON system prompt is faster and more accurate.
LOCAL_TRUST_DOMAINS = {"sentiment"}
REMOTE_FIRST_DOMAINS = {"factual", "summarization", "logic", "debugging", "codegen", "ner"}
RETRY_DOMAINS = {"ner", "sentiment", "debugging", "codegen", "math"}

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



class HybridRouter:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        allowed_models: list[str],
        mode: str = "hybrid",
    ):
        self.classifier = DomainClassifier()
        self.local = LocalModel()
        self.remote = RemoteModel(api_key, base_url, allowed_models)
        self.mode = mode

        self._local_executor = ThreadPoolExecutor(max_workers=1)
        self._active_local_tasks = 0
        self._answer_cache: dict[str, str] = {}

    async def route_async(self, prompt: str) -> str:
        """Route one prompt and return only the answer string."""
        deadline = time.monotonic() + 28.0
        cache_key = f"{self.mode}:{prompt}"
        if cache_key in self._answer_cache:
            return self._answer_cache[cache_key]

        domain, conf = self.classifier.classify(prompt)

        # SPATIAL PUZZLE & HALLUCINATION INTERCEPT:
        # If detected, force the domain to 'logic' (which is REMOTE_FIRST) to bypass local completely.
        if _is_spatial_puzzle(prompt) or _is_hallucination_trap(prompt):
            domain = "logic"
            conf = 1.0


        if self.mode == "remote":
            answer = await self._remote_or_local_fallback(prompt, domain, conf, None, "remote-only", deadline)
            self._answer_cache[cache_key] = answer
            return answer

        direct = try_solve_locally(domain, prompt)
        if direct is not None:
            self._trace(domain, conf, "direct", "deterministic", model="local")
            self._answer_cache[cache_key] = direct
            return direct

        local_answer = None
        if self.mode == "local" or self._should_try_local_first(domain, conf):
            try:
                local_answer = await self._generate_local(prompt, domain, deadline=deadline)

                # Math Self-Consistency Check (Two-Pass Verification)
                if domain == "math" and local_answer:
                    import re
                    ans1_nums = re.findall(r"[-+]?\d*\.\d+|\d+", local_answer)
                    ans1 = ans1_nums[-1] if ans1_nums else None
                    if ans1:
                        try:
                            second_ans = await self._generate_local(prompt, domain, temperature=0.7, deadline=deadline)
                            ans2_nums = re.findall(r"[-+]?\d*\.\d+|\d+", second_ans)
                            ans2 = ans2_nums[-1] if ans2_nums else None
                            if ans1 != ans2:
                                # Answers disagree, local model is hallucinating. Discard local answer.
                                self._trace(domain, conf, "local", "math-consistency-failed", model="local")
                                local_answer = None
                        except asyncio.TimeoutError:
                            # Not enough time for second pass; trust the first answer to save tokens
                            pass

                if local_answer is not None:
                    is_valid, cleaned = validate(domain, prompt, local_answer)
                if is_valid:
                    self._trace(domain, conf, "local", "validator-pass", model="local")
                    self._answer_cache[cache_key] = cleaned
                    return cleaned
                if self.mode == "local":
                    cleaned = postprocess(domain, local_answer)
                    self._trace(domain, conf, "local", "validator-fail-local-only", model="local")
                    self._answer_cache[cache_key] = cleaned
                    return cleaned
            except asyncio.TimeoutError:
                # Local model was too slow — skip it and go remote
                local_answer = None
                if self.mode == "local":
                    # local-only mode with no remote: return empty rather than hang
                    self._trace(domain, conf, "local", "timeout-local-only", model="local")
                    self._answer_cache[cache_key] = ""
                    return ""

        answer = await self._remote_or_local_fallback(
            prompt,
            domain,
            conf,
            local_answer,
            "remote-first" if local_answer is None else "local-validator-fail",
            deadline
        )
        self._answer_cache[cache_key] = answer
        return answer

    def _should_try_local_first(self, domain: str, conf: float) -> bool:
        if domain in LOCAL_TRUST_DOMAINS:
            return True
        if conf < CLASSIFIER_CONF_THRESHOLD:
            return False
        if domain in REMOTE_FIRST_DOMAINS:
            return False
        return False

    # Max seconds to wait for local model inference before giving up and going remote.
    # On 2 vCPU grading env, NER/code can take 20-30s — this prevents TIMEOUT.
    _LOCAL_TIMEOUT_S = 12

    async def _generate_local(self, prompt: str, domain: str, temperature: float = 0.1, deadline: float = 0.0) -> str:
        # Time Budgeting: Skip local if we don't have enough time before global container timeout
        if deadline > 0 and time.monotonic() > deadline - 4.0:
            print(f"[WARNING] Global time budget critically low, skipping local for {domain}.", flush=True)
            raise asyncio.TimeoutError("Time budget low")
            
        # Load Shedding: If the local thread pool is busy, instantly bypass it to avoid hanging the container
        if self._active_local_tasks >= 2:
            print(f"[WARNING] Local model queue full ({self._active_local_tasks}), fast-falling back to remote for {domain}.", flush=True)
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
            print(
                f"[WARNING] Local model timed out (or queue full) after {self._LOCAL_TIMEOUT_S}s "
                f"for domain={domain}. Falling back to remote.",
                flush=True,
            )
            raise  # caller (_remote_or_local_fallback) will catch and use remote
        finally:
            self._active_local_tasks -= 1

    async def _remote_or_local_fallback(
        self,
        prompt: str,
        domain: str,
        conf: float,
        local_answer: str | None,
        reason: str,
        deadline: float = 0.0,
    ) -> str:
        # Phase 8: The Logprob Gate
        # We no longer use the Lean Auditor to double check local answers.
        # If we have a local_answer here, it means it failed local validation (`is_valid` was false).
        # We should NOT return it automatically, we must fall back to remote.
        
        try:
            # We don't audit anymore, we just generate remote
            answer, model = await self.remote.generate(prompt, domain, conf=conf)

            is_valid, cleaned = validate(domain, prompt, answer)

            if not is_valid and domain in RETRY_DOMAINS:
                retry_answer, retry_model = await self.remote.generate_correction(
                    prompt, domain, cleaned, conf=conf
                )
                retry_valid, retry_cleaned = validate(domain, prompt, retry_answer)
                if retry_valid:
                    self._trace(domain, conf, "remote", f"{reason}-retry", model=retry_model)
                    return retry_cleaned
                cleaned = postprocess(domain, retry_answer)
                model = retry_model

            if not is_valid:
                cleaned = postprocess(domain, answer)

            self._trace(domain, conf, "remote", reason, model=model)
            return cleaned
        except Exception as exc:
            print(f"[WARNING] Remote call failed: {exc}. Falling back to local.", flush=True)
            if local_answer is not None:
                fallback = local_answer
            else:
                # We do not have a local answer because it was a spatial puzzle, or local timed out earlier.
                # We MUST generate one now, or else we return an empty string and score 0.
                try:
                    local_ans = await self._generate_local(prompt, domain, deadline=deadline)
                    is_val, cleaned_local = validate(domain, prompt, local_ans)
                    fallback = cleaned_local if is_val else postprocess(domain, local_ans)
                except Exception as local_exc:
                    print(f"[WARNING] Local fallback also failed/timed out: {local_exc}", flush=True)
                    fallback = ""
            self._trace(domain, conf, "local", "remote-failed", model="local")
            return fallback

    def _trace(
        self,
        domain: str,
        conf: float,
        tier: str,
        reason: str,
        model: str = "",
    ) -> None:
        model_part = f" model={model}" if model else ""
        print(
            f"[route] domain={domain} conf={conf:.3f} tier={tier} reason={reason}{model_part}",
            flush=True,
        )
