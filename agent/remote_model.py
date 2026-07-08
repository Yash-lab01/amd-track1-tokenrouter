"""
remote_model.py
---------------
Async Fireworks AI client with:
- Dynamic model selection from ALLOWED_MODELS (Gemma-first)
- Exponential backoff retries via tenacity
- Domain-specific max_tokens caps (output token saver)
- Speculative correction (send local draft, ask remote to fix only)
- Input prompt pruning to avoid paying for fluff tokens
"""
import os
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Remote max_tokens per domain — aggressive caps to save output tokens
REMOTE_MAX_TOKENS = {
    "sentiment":     3,     # "positive" / "negative" / "neutral"
    "factual":      60,
    "math":         50,
    "ner":          100,
    "summarization":120,
    "debugging":    300,
    "codegen":      350,
    "logic":        150,
}

# Compressed remote system prompts (minimal tokens — stripped of examples)
REMOTE_SYSTEM_PROMPTS = {
    "ner":           'Output JSON only: {"person":[],"org":[],"location":[],"date":[]}',
    "sentiment":     "Reply with one word: positive, negative, or neutral.",
    "math":          "Solve. Output: 'Answer: <number>'",
    "summarization": "Summarize in 2 sentences.",
    "debugging":     "Fix the code. Output corrected code only in ```python block.",
    "codegen":       "Write working Python code in ```python block.",
    "logic":         "Reason step by step, then give a concise conclusion.",
    "factual":       "Answer concisely.",
}

# Max input tokens before pruning (saves input token cost)
MAX_INPUT_TOKENS = 300


class RemoteModel:
    def __init__(self, api_key: str, base_url: str, allowed_models: list[str]):
        self.api_key       = api_key
        self.base_url      = base_url.rstrip("/")
        self.allowed_models = [m.strip() for m in allowed_models if m.strip()]

        print(f"[RemoteModel] Initialized with {len(self.allowed_models)} allowed models.")

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    async def generate(self, prompt: str, domain: str = "factual") -> str:
        """Full remote generation. Uses minimal compressed prompt."""
        compressed = self._prune_prompt(prompt)
        system     = REMOTE_SYSTEM_PROMPTS.get(domain, "Answer concisely.")
        max_tok    = REMOTE_MAX_TOKENS.get(domain, 100)
        model      = self._get_model_for_domain(domain)

        return await self._call(
            system=system,
            user=compressed,
            max_tokens=max_tok,
            model=model,
        )

    async def speculative_correct(
        self, prompt: str, local_draft: str, domain: str = "factual"
    ) -> str:
        """
        Speculative correction: send local draft and ask remote to fix ONLY if wrong.
        Saves ~70% output tokens vs full generation for code/logic tasks.
        """
        max_tok = REMOTE_MAX_TOKENS.get(domain, 100)
        model   = self._get_model_for_domain(domain)

        if domain in ("debugging", "codegen"):
            system = "If draft code has errors output ONLY the corrected code in ```python block. If correct reply VALID."
            user   = f"Task: {self._prune_prompt(prompt)}\nDraft:\n{local_draft}"
        else:
            system = "If draft answer is correct reply VALID. If wrong output only the correct answer, no explanation."
            user   = f"Task: {self._prune_prompt(prompt)}\nDraft: {local_draft}"

        result = await self._call(
            system=system,
            user=user,
            max_tokens=max_tok,
            model=model,
        )

        # If remote says the draft is valid, return the local draft (0 output tokens wasted)
        if "VALID" in result.upper() and len(result.split()) < 5:
            return local_draft

        return result

    # ──────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def _call(
        self, system: str, user: str, max_tokens: int, model: str
    ) -> str:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       model,
                    "messages":    [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "max_tokens":  max_tokens,
                    "temperature": 0.1,
                    "top_p":       0.9,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    def _get_model_for_domain(self, domain: str) -> str:
        """
        Dynamically choose the most token-efficient/accurate model from ALLOWED_MODELS.
        - For code/debugging: Prefer 'kimi' (specialized coding model)
        - For general tasks: Prefer 'gemma' (qualifies for Best Use of Gemma)
          - Prefer the MoE 'gemma-4-26b-a4b-it' (highly efficient 4B active params) if available
          - Otherwise fallback to the dense 'gemma-4-31b-it'
        """
        if not self.allowed_models:
            return "accounts/fireworks/models/gemma-4-31b-it"

        # 1. Coding tasks -> route to Kimi
        if domain in ("debugging", "codegen"):
            for m in self.allowed_models:
                if "kimi" in m.lower():
                    return m

        # 2. General tasks -> route to Gemma MoE (26B with 4B active parameters)
        for m in self.allowed_models:
            if "gemma" in m.lower() and "26b" in m.lower():
                return m

        # 3. Fallback to any Gemma
        for m in self.allowed_models:
            if "gemma" in m.lower():
                return m

        # 4. Global fallback -> first allowed model
        return self.allowed_models[0]

    def _prune_prompt(self, prompt: str, max_chars: int = 1200) -> str:
        """
        Prune input prompt to avoid paying for excessive input tokens.
        Simple heuristic: truncate at max_chars if over limit.
        (Token-level pruning requires llama tokenizer — done in router.py)
        """
        if len(prompt) <= max_chars:
            return prompt

        # Keep first 60% and last 40% — preserves question + context
        keep_start = int(max_chars * 0.6)
        keep_end   = max_chars - keep_start
        return prompt[:keep_start] + "\n[...truncated...]\n" + prompt[-keep_end:]
