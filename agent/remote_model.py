"""
remote_model.py
---------------
Async Fireworks AI client with:
- Ranked model selection from ALLOWED_MODELS
- Prompt difficulty scoring for model upgrades
- Exponential backoff retries via tenacity
- Domain-specific max_tokens caps (output token saver)
- Correction retry for invalid outputs
"""
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

REMOTE_MAX_TOKENS = {
    "sentiment":     3,
    "factual":      50,
    "math":         60,
    "ner":          120,
    "summarization": 130,
    "debugging":    350,
    "codegen":      450,
    "logic":        160,
}

REMOTE_SYSTEM_PROMPTS = {
    "ner":           'Return ONLY a JSON object with exactly these keys: {"person":[],"org":[],"location":[],"date":[]}. Use empty arrays for missing entities. No prose, no markdown.',
    "sentiment":     "Return exactly one word: positive, negative, or neutral. Nothing else.",
    "math":          "Solve the problem. Return only the final numeric answer, no units unless asked.",
    "summarization": "Write a concise summary only. No preamble like 'Here is a summary:'.",
    "debugging":     "Return only the corrected code. No explanation.",
    "codegen":       "Return only working Python code. No explanation or markdown unless explicitly requested.",
    "logic":         "Answer accurately and concisely. State your final answer clearly on the last line.",
    "factual":       "Return the direct answer only. One sentence or less. No extra facts.",
}

RETRY_SYSTEM_PROMPTS = {
    "ner": 'Return ONLY valid JSON with keys person, org, location, date. Use empty arrays when missing.',
    "sentiment": "Return exactly one word: positive, negative, or neutral.",
    "math": "Return only the final numeric answer.",
    "debugging": "Return only syntactically valid corrected Python code.",
    "codegen": "Return only syntactically valid Python code.",
}

# Preference tags matched against ALLOWED_MODELS (first match wins per tier)
# Prioritize 26b (serverless), kimi, minimax. Push 31b (throws 404) to the end.
DOMAIN_MODEL_PREFS: dict[str, list[list[str]]] = {
    "sentiment": [["gemma", "26b"], ["gemma"]],
    "math": [["gemma", "26b"], ["gemma"]],
    "ner": [["gemma", "26b"], ["gemma"]],
    "factual": [["gemma", "26b"], ["minimax"], ["gemma", "31b"]],
    "summarization": [["gemma", "26b"], ["gemma", "nvfp4"], ["gemma", "31b"]],
    "logic": [["minimax"], ["gemma", "26b"], ["gemma", "31b"], ["gemma"]],
    "debugging": [["kimi"], ["gemma", "26b"], ["gemma", "31b"], ["gemma"]],
    "codegen": [["kimi"], ["gemma", "26b"], ["gemma", "31b"], ["gemma"]],
}

HARD_DOMAIN_UPGRADE: dict[str, list[list[str]]] = {
    "factual": [["minimax"], ["gemma", "26b"], ["gemma", "31b"]],
    "logic": [["minimax"], ["gemma", "26b"], ["gemma", "31b"]],
    "summarization": [["gemma", "26b"], ["gemma", "31b"]],
    "math": [["gemma", "26b"], ["gemma", "31b"]],
}


class RemoteModel:
    def __init__(self, api_key: str, base_url: str, allowed_models: list[str]):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.allowed_models = [m.strip() for m in allowed_models if m.strip()]
        print(f"[RemoteModel] Initialized with {len(self.allowed_models)} allowed models.")

    async def generate(
        self,
        prompt: str,
        domain: str = "factual",
        conf: float = 1.0,
        upgrade: bool = False,
    ) -> tuple[str, str]:
        """Full remote generation. Returns (answer, model_id)."""
        compressed = self._prune_prompt(prompt)
        system = REMOTE_SYSTEM_PROMPTS.get(domain, "Answer concisely.")
        max_tok = REMOTE_MAX_TOKENS.get(domain, 100)
        model = self._pick_model(domain, prompt, conf, upgrade=upgrade)

        # Append domain-specific format hints to the user prompt
        format_hint = ""
        if domain == "factual":
            format_hint = "\n\nDirect answer only (one word or short phrase):"
        elif domain == "logic":
            format_hint = "\n\nFinal answer (Yes or No, then brief reason if needed):"
        elif domain == "sentiment":
            format_hint = "\n\nLabel (positive, negative, or neutral):"
        elif domain == "ner":
            # For NER, we use native JSON mode
            pass

        result = await self._call(
            system=system,
            user=f"Task:\n{compressed}{format_hint}\n\nAnswer:",
            max_tokens=max_tok,
            model=model,
        )
        return result, model

    async def audit(
        self,
        prompt: str,
        domain: str,
        local_answer: str,
        conf: float = 1.0,
    ) -> tuple[str, str]:
        """Audits a local answer. Returns ([APPROVE] or corrected answer, model_id)."""
        model = self._pick_model(domain, prompt, conf, upgrade=True)
        system = (
            "You are an Auditor. Check if the provided local answer is 100% correct and follows all formatting rules exactly. "
            "If it is perfect, reply exactly with '[APPROVE]' and nothing else. "
            "If it is flawed, reply with the fully corrected answer and nothing else."
        )
        max_tok = REMOTE_MAX_TOKENS.get(domain, 100)
        result = await self._call(
            system=system,
            user=(
                f"Task:\n{self._prune_prompt(prompt)}\n\n"
                f"Local answer:\n{local_answer}\n\nDecision:"
            ),
            max_tokens=max_tok,
            model=model,
        )
        return result, model

    async def generate_correction(
        self,
        prompt: str,
        domain: str,
        bad_answer: str,
        conf: float = 1.0,
    ) -> tuple[str, str]:
        """Retry with a stricter prompt and stronger model."""
        model = self._pick_model(domain, prompt, conf, upgrade=True)
        system = RETRY_SYSTEM_PROMPTS.get(
            domain,
            "Fix the answer. Return only the corrected answer with no explanation.",
        )
        max_tok = REMOTE_MAX_TOKENS.get(domain, 100)
        result = await self._call(
            system=system,
            user=(
                f"Task:\n{self._prune_prompt(prompt)}\n\n"
                f"Bad answer:\n{bad_answer}\n\nCorrected answer:"
            ),
            max_tokens=max_tok,
            model=model,
        )
        return result, model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def _call(
        self, system: str, user: str, max_tokens: int, model: str
    ) -> str:
        async with httpx.AsyncClient(timeout=20.0) as client:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
            }

            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    def _score_difficulty(self, prompt: str, conf: float) -> int:
        """Score prompt difficulty to decide whether to upgrade model tier."""
        score = 0
        # Long prompt = more context = harder
        if len(prompt) > 700:
            score += 1
        # Contains code block
        if "```" in prompt:
            score += 1
        # Complex reasoning or constraint language
        if re.search(
            r"\b(proof|puzzle|constraint|reasoning|step by step|multiple|"
            r"all of the following|if and only if|necessarily|must be|cannot be)\b",
            prompt, re.IGNORECASE,
        ):
            score += 1
        # Explanation / justification requested (needs more output)
        if re.search(
            r"\b(explain|why|how does|prove|justify|elaborate|describe|analyze)\b",
            prompt, re.IGNORECASE,
        ):
            score += 1
        # Multipart / multiple sub-questions
        if re.search(r"\b(and also|additionally|furthermore|as well as|second|third)\b"
                     r"|\(a\)|\(b\)|\d+\.\s+\w",
                     prompt, re.IGNORECASE):
            score += 1
        # Very short prompt — could be a trick question or ambiguous
        if len(prompt.strip()) < 40:
            score += 1
        # Low classifier confidence
        if conf < 0.65:
            score += 1
        return score

    def _pick_model(
        self,
        domain: str,
        prompt: str,
        conf: float,
        upgrade: bool = False,
    ) -> str:
        difficulty = self._score_difficulty(prompt, conf)
        
        if domain == "logic":
            upgrade = True

        use_upgrade = upgrade or difficulty >= 2

        if use_upgrade and domain in HARD_DOMAIN_UPGRADE:
            tiers = HARD_DOMAIN_UPGRADE[domain]
        else:
            tiers = DOMAIN_MODEL_PREFS.get(domain, [["gemma", "26b"], ["gemma"]])

        for tags in tiers:
            model = self._find_model(tags)
            if model:
                return model

        return self.allowed_models[0] if self.allowed_models else "accounts/fireworks/models/gemma-4-26b-a4b-it"

    def _find_model(self, tags: list[str]) -> str | None:
        for model in self.allowed_models:
            lower = model.lower()
            if all(tag.lower() in lower for tag in tags):
                return model
        return None

    def _prune_prompt(self, prompt: str, max_chars: int = 1200) -> str:
        if len(prompt) <= max_chars:
            return prompt
        keep_start = int(max_chars * 0.6)
        keep_end = max_chars - keep_start
        return prompt[:keep_start] + "\n[...truncated...]\n" + prompt[-keep_end:]
