import re
from sklearn.feature_extraction.text import TfidfVectorizer

class DomainCompressor:
    """
    Safe prompt compressor — accuracy-first.

    Per HACKATHON_WINNING_PLAN.md:
    - Keep the full original prompt for logic, math, NER, and code unless extremely long.
    - Remove only whitespace, markdown noise, and irrelevant trace clutter.
    - Do NOT use TF-IDF sentence deletion for constraint-heavy tasks.
    - Do NOT over-compress summarization input.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def compress(self, prompt: str, domain: str, max_chars: int = 4000) -> str:
        # Domains where we must preserve the full prompt (reasoning/extraction critical)
        if domain in {"logic", "math", "ner"}:
            return self._preserve_reasoning_prompt(prompt, max_chars=max_chars)

        # 1. Base Cleanup (Whitespace & Markdown) — safe for all domains
        cleaned = self._safe_cleanup(prompt)

        # 2. Domain-Aware Compression (only for very long prompts)
        if len(cleaned) > max_chars:
            if domain in ["codegen", "debugging"]:
                cleaned = self._compress_code_prompt(cleaned, max_chars)
            # summarization: do NOT TF-IDF delete sentences (can remove key facts)
            # just truncate if truly huge

        # 3. Final Failsafe Truncation (70/30 Split — keep more context)
        if len(cleaned) > max_chars:
            keep_start = int(max_chars * 0.7)
            keep_end = max_chars - keep_start
            cleaned = cleaned[:keep_start] + "\n\n[...truncated...]\n\n" + cleaned[-keep_end:]

        return cleaned

    def _preserve_reasoning_prompt(self, prompt: str, max_chars: int) -> str:
        """Keep logic/math/NER wording intact; only trim if it is truly huge."""
        text = prompt.strip()
        if len(text) <= max_chars:
            return text
        keep_start = int(max_chars * 0.6)
        keep_end = max_chars - keep_start
        return text[:keep_start] + "\n\n[...middle omitted...]\n\n" + text[-keep_end:]

    def _safe_cleanup(self, prompt: str) -> str:
        """Removes excessive markdown and whitespace without touching code blocks."""
        # Extract code blocks
        code_blocks = []
        def _replace_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"

        text = re.sub(r"```.*?```", _replace_code, prompt, flags=re.DOTALL)

        # Collapse excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {3,}', '  ', text)

        # Strip excessive markdown decoration (bolding, headers) but leave structure
        text = re.sub(r'\*{2,}|_{2,}|#{1,}', '', text)

        # Re-inject code blocks
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)

        return text.strip()

    def _compress_code_prompt(self, text: str, target_chars: int) -> str:
        """Trims excessive tracebacks but protects code."""
        # Find traceback
        if "Traceback (most recent call last):" in text:
            parts = text.split("Traceback (most recent call last):")
            traceback = parts[1]
            lines = traceback.split('\n')
            if len(lines) > 20:
                # Keep top 5 and bottom 10 lines of traceback
                trimmed_tb = "\n".join(lines[:5]) + "\n...[traceback truncated]...\n" + "\n".join(lines[-10:])
                text = parts[0] + "Traceback (most recent call last):\n" + trimmed_tb
        return text