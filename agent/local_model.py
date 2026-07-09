"""
local_model.py
--------------
Thread-safe wrapper around llama-cpp-python for CPU inference.
Uses a lock so concurrent async tasks don't cause CPU thrashing.
"""
import os
import threading
from llama_cpp import Llama

# Domain-specific max_tokens for local generation (free, so be generous)
LOCAL_MAX_TOKENS = {
    "sentiment":     32,
    "factual":       80,
    "math":         100,
    "ner":          256,
    "summarization":256,
    "debugging":    384,
    "codegen":      512,
    "logic":        200,
}

# Long, detailed system prompts per domain (free tokens — go all out)
SYSTEM_PROMPTS = {
    "ner": (
        "You are a named entity recognition expert. "
        "Always output a JSON object with exactly four keys: "
        '"person" (list of person names), "org" (list of organization names), '
        '"location" (list of place names), "date" (list of date/time expressions). '
        "If a category has no entities, output an empty list []. "
        "Output ONLY the JSON object, no explanation.\n"
        "Example: {\"person\": [\"Alice\"], \"org\": [\"OpenAI\"], \"location\": [\"San Francisco\"], \"date\": [\"2024\"]}"
    ),
    "sentiment": (
        "You are a sentiment classifier. "
        "Reply with EXACTLY one word: positive, negative, or neutral. "
        "No explanation, no punctuation, just the single label."
    ),
    "math": (
        "You are a math solver. "
        "Solve the problem step by step, then output the final numeric answer on the last line. "
        "Format: 'Answer: <number>'"
    ),
    "summarization": (
        "You are a summarization expert. "
        "Write a concise summary of 2-3 sentences. "
        "Capture the main idea only. Do not copy sentences verbatim."
    ),
    "debugging": (
        "You are an expert Python debugger. "
        "Identify the bug, then output ONLY the corrected code inside a ```python block. "
        "No explanation needed unless the fix is non-obvious."
    ),
    "codegen": (
        "You are an expert Python programmer. "
        "Write clean, working Python code inside a ```python block. "
        "Include a docstring. Do not include example usage unless asked."
    ),
    "logic": (
        "You are a logical reasoning expert. "
        "Think step by step, then give a clear, concise conclusion."
    ),
    "factual": (
        "You are a knowledgeable assistant. "
        "Answer factual questions concisely and accurately. "
        "Give the direct answer first, then a brief explanation if needed."
    ),
}


class LocalModel:
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = os.environ.get(
                "LOCAL_MODEL_PATH",
                "./models/gemma-2b-instruct-q4.gguf"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Local model not found at: {model_path}\n"
                f"Download from HuggingFace and place in ./models/"
            )

        print(f"[LocalModel] Loading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=1280,              # Reduced from 2048 to save KV cache RAM (plenty for hackathon tasks)
            n_threads=min(4, os.cpu_count() or 4),
            n_batch=256,             # Reduced from 512 to lower peak memory during prompt processing
            use_mmap=True,           # Use memory-mapping to let OS page weights in/out dynamically
            use_mlock=False,         # Do not lock memory (keeps physical RAM usage minimal)
            verbose=False,
        )
        self._lock = threading.Lock()
        print("[LocalModel] Model loaded.")

    def generate(self, prompt: str, domain: str = "factual", temperature: float = 0.1) -> str:
        """
        Thread-safe local generation.
        Uses domain-specific system prompt and token caps.
        """
        system = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["factual"])
        max_tokens = LOCAL_MAX_TOKENS.get(domain, 128)

        full_prompt = f"<start_of_turn>system\n{system}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        with self._lock:
            output = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                echo=False,
                stop=["<end_of_turn>", "<start_of_turn>"],
            )

        return output["choices"][0]["text"].strip()

    def count_tokens(self, text: str) -> int:
        """Count tokens locally — used to prune remote prompts."""
        return len(self.llm.tokenize(text.encode("utf-8", errors="replace")))
