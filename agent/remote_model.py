"""
remote_model.py
---------------
Async Fireworks AI client — Remote-First Accuracy Engine with:
- Chain-of-Thought prompting with answer extraction
- Self-consistency voting for math/logic (3 parallel calls)
- Few-shot examples in system prompts
- Judge-aware prompts (sentiment needs reason, factual needs explanation)
- Model specialization per domain
- Dynamic max_tokens based on prompt complexity
"""
import re
import asyncio
from collections import Counter

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.compressor import DomainCompressor

# Output token caps — accuracy-first with dynamic scaling
# Increased for judge-aware prompts (sentiment needs reason, factual needs explanation)
REMOTE_MAX_TOKENS = {
    "sentiment":      100,  # Needs label + one-sentence reason
    "factual":       300,   # Needs answer + brief explanation
    "math":          500,   # CoT
    "ner":           300,   # Needs all entities with labels
    "summarization": 400,   # Needs exact format (sentences/bullets)
    "debugging":     600,
    "codegen":       700,
    "logic":         600,   # CoT
}

# Judge-aware system prompts with few-shot examples
# Key insight: the judge is an LLM that checks semantic completeness
REMOTE_SYSTEM_PROMPTS = {
    "ner":           'Extract ALL named entities from the text. Return ONLY a JSON object with exactly these keys: {"person":[],"org":[],"location":[],"date":[]}. List every person, organization, location, and date. Use empty arrays for missing types. No prose, no markdown.\nExample: Input: "Tim Cook visited Paris in 2024" → {"person":["Tim Cook"],"org":[],"location":["Paris"],"date":["2024"]}',
    
    "sentiment":     "Classify the sentiment as Positive, Negative, Neutral, or Mixed. Then give a one-sentence reason that acknowledges ALL aspects of the text (both positive and negative if present). Format: 'Label: <one-sentence reason>'. If the text has both positive and negative elements, use Mixed or Neutral and acknowledge both sides.\nExample: 'The food was great but service was slow.' → Neutral: The review acknowledges both positive food quality and negative service speed.\nExample: 'I love this product!' → Positive: The review expresses clear satisfaction with the product.",
    
    "math":          "Solve the problem step by step. Show your reasoning. End with 'Final Answer: <number>'. No units unless asked.\nExample: 'What is 15% of 200?' → 15% = 0.15, 0.15 * 200 = 30. Final Answer: 30",
    
    "summarization": "Summarize the passage following the EXACT format requested (number of sentences, bullet points, word limits). Capture both the main points AND any challenges or concerns mentioned. Do not omit either side. No preamble.\nExample: 'Summarize in 2 sentences' → Write exactly 2 sentences covering both opportunities and challenges.",
    
    "debugging":     "Return only the corrected code. No explanation.\nExample: Fix 'def add(a,b): return a-b' → def add(a, b):\n    return a + b",
    
    "codegen":       "Return only working Python code. No explanation or markdown unless explicitly requested.\nExample: 'Write a function to reverse a string' → def reverse_string(s):\n    return s[::-1]",
    
    "logic":         "Think through the problem step by step. Show your reasoning. End with 'Final Answer: <answer>'. Do not force yes/no unless the task asks yes/no.\nExample: 'All cats are animals. Fluffy is a cat. Is Fluffy an animal?' → All cats are animals. Fluffy is a cat. Therefore Fluffy is an animal. Final Answer: Yes",
    
    "factual":       "Answer the question directly and completely. If the question asks to explain or compare, provide a brief but complete explanation. Do not give just one word if an explanation is requested. Be concise but thorough.\nExample: 'What is the capital of France?' → Paris\nExample: 'Explain the difference between RAM and ROM' → RAM is volatile memory used for temporary storage of active programs, while ROM is non-volatile memory that stores permanent firmware. RAM is fast and loses data when powered off; ROM retains data without power.",
}

RETRY_SYSTEM_PROMPTS = {
    "ner": 'Return ONLY valid JSON with keys person, org, location, date. List ALL entities. Use empty arrays when missing.',
    "sentiment": "Classify as Positive, Negative, Neutral, or Mixed. Give a one-sentence reason acknowledging all aspects. Format: 'Label: reason'.",
    "math": "Solve step by step. End with 'Final Answer: <number>'.",
    "debugging": "Return only syntactically valid corrected Python code.",
    "codegen": "Return only syntactically valid Python code.",
    "logic": "Think step by step. End with 'Final Answer: <answer>'.",
    "factual": "Answer directly and completely. Explain if asked to explain.",
    "summarization": "Follow the exact format requested. Capture both main points and challenges.",
}

# Domains that use self-consistency voting (3 parallel calls, majority vote)
SELF_CONSISTENCY_DOMAINS = {"math", "logic"}

# Model specialization per the plan
DOMAIN_MODEL_PREFS: dict[str, list[list[str]]] = {
    "sentiment": [["gemma", "26b"], ["gemma", "nvfp4"], ["gemma", "31b"], ["gemma"], ["minimax"]],
    "math": [["gemma", "26b"], ["minimax"], ["gemma", "31b"], ["gemma"]],
    "ner": [["gemma", "26b"], ["gemma", "nvfp4"], ["gemma", "31b"], ["gemma"], ["minimax"]],
    "factual": [["gemma", "26b"], ["minimax"], ["gemma", "31b"], ["gemma"]],
    "summarization": [["gemma", "26b"], ["gemma", "nvfp4"], ["gemma", "31b"], ["gemma"], ["minimax"]],
    "logic": [["minimax"], ["gemma", "31b"], ["gemma", "26b"], ["gemma"]],
    "debugging": [["kimi"], ["gemma", "26b"], ["gemma", "31b"], ["gemma"], ["minimax"]],
    "codegen": [["kimi"], ["gemma", "26b"], ["gemma", "31b"], ["gemma"], ["minimax"]],
}

HARD_DOMAIN_UPGRADE: dict[str, list[list[str]]] = {
    "factual": [["minimax"], ["gemma", "26b"], ["gemma", "31b"]],
    "logic": [["minimax"], ["gemma", "31b"], ["gemma", "26b"]],
    "summarization": [["gemma", "26b"], ["gemma", "31b"], ["minimax"]],
    "math": [["minimax"], ["gemma", "26b"], ["gemma", "31b"]],
    "ner": [["gemma", "26b"], ["gemma", "31b"], ["minimax"]],
}


class RemoteModel:
    def __init__(self, api_key: str, base_url: str, allowed_models: list[str]):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.allowed_models = [m.strip() for m in allowed_models if m.strip()]
        self.compressor = DomainCompressor()
        self.bad_models = set()
        print(f"[RemoteModel] Initialized with {len(self.allowed_models)} allowed models.")

    async def generate(
        self,
        prompt: str,
        domain: str = "factual",
        conf: float = 1.0,
        upgrade: bool = False,
    ) -> tuple[str, str]:
        """Full remote generation with self-consistency for math/logic."""
        compressed = self.compressor.compress(prompt, domain)
        system = REMOTE_SYSTEM_PROMPTS.get(domain, "Answer concisely.")
        max_tok = self._dynamic_max_tokens(domain, compressed)
        models = self._pick_models(domain, prompt, conf, upgrade=upgrade)

        # Self-consistency voting for math and logic
        if domain in SELF_CONSISTENCY_DOMAINS:
            return await self._generate_with_consistency(
                compressed, system, domain, max_tok, models, conf
            )

        # Standard single-call generation for other domains
        format_hint = self._get_format_hint(domain)
        last_exc = None
        for model in models:
            try:
                result = await self._call(
                    system=system,
                    user=f"Task:\n{compressed}{format_hint}\n\nAnswer:",
                    max_tokens=max_tok,
                    model=model,
                    reasoning=self._use_reasoning(domain),
                )
                return result, model
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (404, 401, 403, 400):
                    print(f"[RemoteModel] Model {model} returned {e.response.status_code}. Adding to bad_models.", flush=True)
                    self.bad_models.add(model)
                    last_exc = e
                    continue
                last_exc = e
                print(f"[RemoteModel] Model {model} failed with {e.response.status_code} after retries. Cascading.", flush=True)
                continue
            except Exception as e:
                print(f"[RemoteModel] Model {model} failed: {e}. Cascading.", flush=True)
                last_exc = e
                continue
        
        raise last_exc or Exception("All remote models failed")

    async def _generate_with_consistency(
        self,
        compressed: str,
        system: str,
        domain: str,
        max_tok: int,
        models: list[str],
        conf: float,
    ) -> tuple[str, str]:
        """Generate 3 responses in parallel and take majority vote on extracted answer."""
        format_hint = self._get_format_hint(domain)
        user_prompt = f"Task:\n{compressed}{format_hint}\n\nAnswer:"
        
        # Use the best available model for all 3 calls
        model = models[0] if models else "accounts/fireworks/models/gemma-4-26b-a4b-it"
        
        # Launch 3 parallel calls with slightly different temperatures
        tasks = []
        temps = [0.1, 0.3, 0.5]  # Low variance for consistency
        for temp in temps:
            tasks.append(self._call_with_temp(system, user_prompt, max_tok, model, temp))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Extract final answers from each response
            answers = []
            raw_responses = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[RemoteModel] Consistency call {i} failed: {result}", flush=True)
                    continue
                raw_responses.append(result)
                extracted = self._extract_final_answer(result, domain)
                if extracted:
                    answers.append(extracted)
            
            if not answers:
                # All calls failed or no answers extracted
                if raw_responses:
                    return raw_responses[0], model
                raise Exception("All consistency calls failed")
            
            # Majority vote on extracted answers
            answer_counts = Counter(answers)
            best_answer, count = answer_counts.most_common(1)[0]
            
            if count >= 2:
                # Majority agreement — return the raw response that contains this answer
                for raw in raw_responses:
                    if best_answer in raw:
                        return raw, model
                return best_answer, model
            else:
                # No majority — return the first response (most likely correct at temp=0.1)
                return raw_responses[0] if raw_responses else answers[0], model
                
        except Exception as e:
            print(f"[RemoteModel] Consistency voting failed: {e}. Falling back to single call.", flush=True)
            # Fallback to single call
            result = await self._call(
                system=system,
                user=user_prompt,
                max_tokens=max_tok,
                model=model,
                reasoning=self._use_reasoning(domain),
            )
            return result, model

    def _extract_final_answer(self, response: str, domain: str) -> str:
        """Extract the final answer from a CoT response."""
        # Look for "Final Answer: X" pattern
        match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for "Answer: X"
        match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # For math: extract last number
        if domain == "math":
            nums = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
            if nums:
                return nums[-1]
        
        # For logic: take last non-empty line
        if domain == "logic":
            lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
            if lines:
                return lines[-1].rstrip(".")
        
        return response.strip()

    async def _call_with_temp(
        self, system: str, user: str, max_tokens: int, model: str, temperature: float
    ) -> str:
        """Make a call with a specific temperature (for self-consistency)."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
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

    def _get_format_hint(self, domain: str) -> str:
        """Domain-specific format hints appended to user prompt."""
        if domain == "factual":
            return "\n\nAnswer directly and completely:"
        elif domain == "logic":
            return "\n\nThink step by step. Final Answer:"
        elif domain == "sentiment":
            return "\n\nClassification with reason:"
        elif domain == "math":
            return "\n\nSolve step by step. Final Answer:"
        elif domain == "summarization":
            return "\n\nSummary following the exact format requested:"
        return ""

    def _dynamic_max_tokens(self, domain: str, prompt: str) -> int:
        """Scale max_tokens based on prompt complexity."""
        base = REMOTE_MAX_TOKENS.get(domain, 150)
        # Add tokens for longer prompts (more context = more output needed)
        extra = min(len(prompt) // 100 * 5, 200)
        return base + extra

    async def generate_correction(
        self,
        prompt: str,
        domain: str,
        bad_answer: str,
        conf: float = 1.0,
    ) -> tuple[str, str]:
        """Retry with a stricter prompt and stronger model."""
        models = self._pick_models(domain, prompt, conf, upgrade=True)
        system = RETRY_SYSTEM_PROMPTS.get(
            domain,
            "Fix the answer. Return only the corrected answer with no explanation.",
        )
        max_tok = self._dynamic_max_tokens(domain, self.compressor.compress(prompt, domain))
        
        last_exc = None
        for model in models:
            try:
                result = await self._call(
                    system=system,
                    user=(
                        f"Task:\n{self.compressor.compress(prompt, domain)}\n\n"
                        f"Bad answer:\n{bad_answer}\n\nCorrected answer:"
                    ),
                    max_tokens=max_tok,
                    model=model,
                    reasoning=True,
                )
                return result, model
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (404, 401, 403, 400):
                    print(f"[RemoteModel] Correction model {model} returned {e.response.status_code}. Adding to bad_models.", flush=True)
                    self.bad_models.add(model)
                    last_exc = e
                    continue
                last_exc = e
                print(f"[RemoteModel] Correction model {model} failed with {e.response.status_code}. Cascading.", flush=True)
                continue
            except Exception as e:
                print(f"[RemoteModel] Correction model {model} failed: {e}. Cascading.", flush=True)
                last_exc = e
                continue
                
        raise last_exc or Exception("All remote correction models failed")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def _call(
        self, system: str, user: str, max_tokens: int, model: str, reasoning: bool = False, disable_reasoning_param: bool = False
    ) -> str:
        async with httpx.AsyncClient(timeout=15.0) as client:
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
            
            if not reasoning and not disable_reasoning_param:
                payload["reasoning_effort"] = "none"

            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            
            if resp.status_code == 400 and not reasoning and not disable_reasoning_param:
                print(f"[RemoteModel] 400 Bad Request with reasoning_effort=none, retrying without it for {model}", flush=True)
                return await self._call(system, user, max_tokens, model, reasoning=False, disable_reasoning_param=True)
                
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    def _score_difficulty(self, prompt: str, conf: float) -> int:
        """Score prompt difficulty to decide whether to upgrade model tier."""
        score = 0
        if len(prompt) > 700:
            score += 1
        if "```" in prompt:
            score += 1
        if re.search(
            r"\b(proof|puzzle|constraint|reasoning|step by step|multiple|"
            r"all of the following|if and only if|necessarily|must be|cannot be)\b",
            prompt, re.IGNORECASE,
        ):
            score += 1
        if re.search(
            r"\b(explain|why|how does|prove|justify|elaborate|describe|analyze|difference|compare)\b",
            prompt, re.IGNORECASE,
        ):
            score += 1
        if re.search(r"\b(and also|additionally|furthermore|as well as|second|third)\b"
                     r"|\(a\)|\(b\)|\d+\.\s+\w",
                     prompt, re.IGNORECASE):
            score += 1
        if len(prompt.strip()) < 40:
            score += 1
        if conf < 0.65:
            score += 1
        return score

    def _use_reasoning(self, domain: str) -> bool:
        """Enable reasoning for domains that benefit from step-by-step thinking."""
        return domain in {"logic", "math", "debugging", "codegen"}

    def _pick_models(
        self,
        domain: str,
        prompt: str,
        conf: float,
        upgrade: bool = False,
    ) -> list[str]:
        difficulty = self._score_difficulty(prompt, conf)
        
        if domain == "logic":
            upgrade = True

        use_upgrade = upgrade or difficulty >= 2

        if use_upgrade and domain in HARD_DOMAIN_UPGRADE:
            tiers = HARD_DOMAIN_UPGRADE[domain]
        else:
            tiers = DOMAIN_MODEL_PREFS.get(domain, [["gemma", "26b"], ["gemma"]])

        selected = []
        for tags in tiers:
            model = self._find_model(tags)
            if model and model not in self.bad_models and model not in selected:
                selected.append(model)

        for m in self.allowed_models:
            if m not in self.bad_models and m not in selected:
                selected.append(m)

        if not selected:
            selected.append(self.allowed_models[0] if self.allowed_models else "accounts/fireworks/models/gemma-4-26b-a4b-it")

        return selected

    def _find_model(self, tags: list[str]) -> str | None:
        for model in self.allowed_models:
            lower = model.lower()
            if all(tag.lower() in lower for tag in tags):
                return model
        return None