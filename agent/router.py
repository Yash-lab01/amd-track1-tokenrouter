"""
router.py
---------
Accuracy-first hybrid routing engine.

The leaderboard gates on accuracy before token efficiency, so this router only
trusts local answers when a deterministic solver or strict validator can prove
the output shape. Risky domains go remote first with short prompts.
"""
import asyncio
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
        self._answer_cache: dict[str, str] = {}

    async def route_async(self, prompt: str) -> str:
        """Route one prompt and return only the answer string."""
        cache_key = f"{self.mode}:{prompt}"
        if cache_key in self._answer_cache:
            return self._answer_cache[cache_key]

        domain, conf = self.classifier.classify(prompt)

        if self.mode == "remote":
            answer = await self._remote_or_local_fallback(prompt, domain, conf, None, "remote-only")
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
                local_answer = await self._generate_local(prompt, domain)
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

    async def _generate_local(self, prompt: str, domain: str) -> str:
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._local_executor,
                    self.local.generate,
                    prompt,
                    domain,
                ),
                timeout=self._LOCAL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            print(
                f"[WARNING] Local model timed out after {self._LOCAL_TIMEOUT_S}s "
                f"for domain={domain}. Falling back to remote.",
                flush=True,
            )
            raise  # caller (_remote_or_local_fallback) will catch and use remote

    async def _remote_or_local_fallback(
        self,
        prompt: str,
        domain: str,
        conf: float,
        local_answer: str | None,
        reason: str,
    ) -> str:
        try:
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
            # Use whatever local answer we already have; do NOT call _generate_local
            # again here — it could timeout and create an infinite wait.
            if local_answer is not None:
                fallback = postprocess(domain, local_answer)
            else:
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
