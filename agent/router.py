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
LOCAL_TRUST_DOMAINS = {"sentiment", "ner"}
REMOTE_FIRST_DOMAINS = {"factual", "summarization", "logic", "debugging", "codegen"}
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

    async def _generate_local(self, prompt: str, domain: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._local_executor,
            self.local.generate,
            prompt,
            domain,
        )

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
            if local_answer is None:
                local_answer = await self._generate_local(prompt, domain)
            fallback = postprocess(domain, local_answer)
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
