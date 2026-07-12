"""
local_model.py
--------------
Thread-safe wrapper around llama-cpp-python for CPU inference.
Uses a lock so concurrent async tasks don't cause CPU thrashing.

Used in hybrid mode for:
- NER: Generate JSON → validate schema → 0 remote tokens if valid
- Code: Generate code → AST check → 0 remote tokens if valid
- Sentiment: Generate label → validate → 0 remote tokens if valid
- Emergency fallback if remote fails
"""
import os
import threading
from llama_cpp import Llama

try:
    from llama_cpp import LlamaGrammar
except Exception:  # Depends on llama-cpp-python build/version.
    LlamaGrammar = None

NER_GRAMMAR_STRING = r'''
root ::= ws "{" ws "\"person\"" ws ":" ws stringlist "," ws "\"org\"" ws ":" ws stringlist "," ws "\"location\"" ws ":" ws stringlist "," ws "\"date\"" ws ":" ws stringlist "}"
stringlist ::= "[" ws "]" | "[" ws string ("," ws string)* "]" ws
string ::= "\"" [^"]* "\"" ws
ws ::= [ \t\n]*
'''

SENTIMENT_GRAMMAR_STRING = r'''
root ::= ws ("positive" | "negative" | "neutral" | "Positive" | "Negative" | "Neutral")
ws ::= [ \t\n]*
'''

# Domain-specific max_tokens for local generation (free, so be generous)
LOCAL_MAX_TOKENS = {
    "sentiment":      80,   # Increased for "Label: reason" format
    "factual":       150,
    "math":          150,
    "ner":           300,   # Increased for better entity extraction
    "summarization": 400,   # Increased for format-compliant summaries
    "debugging":     450,   # Increased for code fixes
    "codegen":       600,   # Increased for code generation
    "logic":         250,
}

# Judge-aware system prompts — local tokens are FREE, so go all out
SYSTEM_PROMPTS = {
    "ner": (
        "You are a named entity recognition expert. "
        "Extract ALL named entities from the text. "
        "Always output a JSON object with exactly four keys: "
        '"person" (list of person names), "org" (list of organization names), '
        '"location" (list of place names), "date" (list of date/time expressions). '
        "If a category has no entities, output an empty list []. "
        "Output ONLY the JSON object, no explanation.\n"
        "Example: {\"person\": [\"Alice\"], \"org\": [\"OpenAI\"], \"location\": [\"San Francisco\"], \"date\": [\"2024\"]}"
    ),
    "sentiment": (
        "You are a sentiment classifier. "
        "Classify the sentiment as Positive, Negative, Neutral, or Mixed. "
        "If the text has BOTH positive and negative elements, use Mixed or Neutral. "
        "Format: 'Label: <one-sentence reason that acknowledges BOTH sides>'. "
        "Example: 'Mixed: The review praises the product quality but criticizes the slow delivery.'"
    ),
    "math": (
        "You are a math solver. "
        "Solve the problem step by step, then output the final numeric answer on the last line. "
        "Format: 'Answer: <number>'"
    ),
    "summarization": (
        "You are a summarization expert. "
        "Follow the EXACT format requested (number of sentences, bullet points, word limits). "
        "Capture BOTH the main points AND any challenges or concerns mentioned. "
        "Do not omit either side. No preamble, no 'Here is a summary:' prefix."
    ),
    "debugging": (
        "You are an expert Python debugger. "
        "Identify the bug, then output ONLY the corrected code. "
        "No explanation needed. Output just the fixed code."
    ),
    "codegen": (
        "You are an expert Python programmer. "
        "Write clean, working Python code. "
        "Include a docstring. Do not include example usage unless asked. "
        "Output only the code, no markdown fences."
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
                "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
            )

        if not os.path.exists(model_path):
            # Try 1.5B as fallback if 3B not available
            fallback_path = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
            if os.path.exists(fallback_path):
                print(f"[LocalModel] 3B model missing, using 1.5B fallback: {fallback_path}")
                model_path = fallback_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Local model not found at: {model_path}\n"
                f"Download from HuggingFace and place in ./models/"
            )

        print(f"[LocalModel] Loading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,              # Increased from 1280 for better NER/code context
            n_threads=min(4, os.cpu_count() or 4),
            n_batch=256,             # Reduced from 512 to lower peak memory during prompt processing
            use_mmap=True,           # Use memory-mapping to let OS page weights in/out dynamically
            use_mlock=False,         # Do not lock memory (keeps physical RAM usage minimal)
            verbose=False,
        )
        self._lock = threading.Lock()
        
        self.grammars = self._load_grammars()
        
        print("[LocalModel] Model loaded.")

    def generate(self, prompt: str, domain: str = "factual", temperature: float = 0.1, min_confidence: float = 0.75) -> str:
        """
        Thread-safe local generation.
        Uses domain-specific system prompt and token caps.
        GBNF grammars force valid JSON for NER and exact labels for sentiment.
        """
        system = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["factual"])
        max_tokens = LOCAL_MAX_TOKENS.get(domain, 128)

        full_prompt = f"<start_of_turn>system\n{system}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        grammar = self.grammars.get(domain)

        with self._lock:
            try:
                output = self.llm(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    echo=False,
                    grammar=grammar,
                    stop=["<end_of_turn>", "<start_of_turn>"],
                )
                choice = output["choices"][0]
                text = choice["text"].strip()
                return text
            except Exception as e:
                print(f"[LocalModel] Generation failed: {e}", flush=True)
                return ""

    def count_tokens(self, text: str) -> int:
        """Count tokens locally — used to prune remote prompts."""
        return len(self.llm.tokenize(text.encode("utf-8", errors="replace")))

    def _load_grammars(self) -> dict[str, object]:
        if LlamaGrammar is None:
            print("[LocalModel] LlamaGrammar unavailable; strict local grammars disabled.", flush=True)
            return {}

        grammars = {}
        for domain, grammar_text in {
            "ner": NER_GRAMMAR_STRING,
            # Sentiment grammar removed — we now need "Label: reason" format, not single label
        }.items():
            try:
                grammars[domain] = LlamaGrammar.from_string(grammar_text)
            except Exception as exc:
                print(f"[LocalModel] Failed to load {domain} grammar: {exc}", flush=True)
        return grammars