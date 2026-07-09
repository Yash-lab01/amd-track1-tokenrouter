import re
from sklearn.feature_extraction.text import TfidfVectorizer

class DomainCompressor:
    """
    A zero-dependency prompt compressor.
    Safely reduces input tokens by cleaning markdown/whitespace and intelligently
    truncating prose using TF-IDF for summarization tasks, while protecting code blocks.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def compress(self, prompt: str, domain: str, max_chars: int = 1200) -> str:
        # 1. Base Cleanup (Whitespace & Markdown)
        cleaned = self._safe_cleanup(prompt)

        # 2. Domain-Aware Compression
        if len(cleaned) > max_chars:
            if domain == "summarization":
                cleaned = self._tfidf_summarize(cleaned, max_chars)
            elif domain in ["codegen", "debugging"]:
                cleaned = self._compress_code_prompt(cleaned, max_chars)
            
        # 3. Final Failsafe Truncation (60/40 Split)
        if len(cleaned) > max_chars:
            keep_start = int(max_chars * 0.6)
            keep_end = max_chars - keep_start
            cleaned = cleaned[:keep_start] + "\n\n[...truncated...]\n\n" + cleaned[-keep_end:]

        return cleaned

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

    def _tfidf_summarize(self, text: str, target_chars: int) -> str:
        """Uses TF-IDF to rank sentences and discard the least important ones."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 5:
            return text

        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            scores = tfidf_matrix.sum(axis=1).A1
            
            # Rank sentences by score
            ranked_sentences = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            
            # Keep picking the top sentences until we hit the budget
            kept_indices = []
            current_chars = 0
            for idx, score in ranked_sentences:
                # Always keep the first and last sentence for context
                if idx == 0 or idx == len(sentences) - 1:
                    continue
                
                if current_chars + len(sentences[idx]) > target_chars:
                    break
                kept_indices.append(idx)
                current_chars += len(sentences[idx])
            
            # Add back first and last sentence
            kept_indices.append(0)
            kept_indices.append(len(sentences) - 1)
            
            # Reconstruct in original order
            kept_indices = sorted(list(set(kept_indices)))
            return " ".join([sentences[i] for i in kept_indices])
        except Exception:
            # Fallback if TF-IDF fails (e.g. no vocab)
            return text

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
