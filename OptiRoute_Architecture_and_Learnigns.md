# OptiRoute: AMD Hackathon Architecture & Learnings

This document is a comprehensive summary of the `OptiRoute` AI routing agent built for the AMD Developer Hackathon (Track 1). It is designed to provide maximum context to other frontier AI models that may review this project to offer suggestions.

---

## 1. Hackathon Overview & Constraints

The goal of Track 1 is to build an intelligent, token-efficient routing agent. The leaderboard scores submissions based on two metrics:

1. **Accuracy (Primary):** The model must get the correct answer.
2. **Token Efficiency (Secondary):** The model must minimize remote API token usage (by solving questions locally when possible).

### Strict Environment Constraints

- **Compute:** The Hugging Face Spaces grading environment provides a severely limited CPU-only environment (2 vCPUs, 4GB RAM). No GPU is available.
- **Time Budget:** There is a strict 30-second execution time limit per prompt. If the agent takes longer than 30s to return an answer, it scores a `0`.
- **Network Isolation Rule:** Competitors are only allowed to route remote requests to the provided Fireworks AI API. Routing to OpenAI, Anthropic, or any other external API during the final judging period will result in immediate disqualification.
- **Hidden Test Set:** While the public leaderboard uses a fixed set of questions, the final judging will use **randomized, unseen prompts**. This means "0-token" strategies that overfit a local 1.5B model to the public leaderboard questions will catastrophically fail during final judging.

---

## 2. Current Architecture (OptiRoute)

OptiRoute uses a 3-Tier Hybrid Routing system designed to maximize accuracy on a hidden test set while minimizing token usage.

### Tier 0: Direct Solver (Deterministic)

Before touching any LLM, `agent/evaluator.py` intercepts the prompt.

- Uses regex and `SymPy` to directly solve deterministic math (factorials, algebra, GCD, LCM, Mean/Median, area/perimeter).
- Uses Python `ast.parse` and regex to fix trivial code bugs (e.g., swapping `^` for `**`).
- Uses a rule-based lexicon lookup to solve obvious Sentiment Analysis queries.
- **Cost:** 0 tokens, ~5ms latency.

### Tier 1: Local Qwen2.5-1.5B (GGUF)

If Tier 0 fails, the prompt is classified into a domain (e.g., `ner`, `sentiment`, `factual`, `math`, `logic`).

- The local model (`qwen2.5-1.5b-instruct-q4_k_m.gguf`) runs purely on the CPU using `llama-cpp-python`.
- The local model is trusted for easy domains (`sentiment`, `ner`, `factual`).
- **Cost:** 0 remote tokens, 1-3 seconds latency.

### Tier 2: Speculative Remote (Fireworks AI)

If the local model fails, is too slow, or detects a domain it is historically bad at (`logic`, `math`), it falls back to the remote Fireworks API (usually accessing a massive model like `gemma-4-26b-a4b-it` or `deepseek-v4-pro`).

- Remote inputs are heavily compressed by removing conversational fluff before sending.
- We enforce strict `max_tokens` limits on the remote generation to cut output costs.
- **Cost:** Variable tokens, highly accurate.

---

## 3. What We Tried & Succeeded

### The Hallucination Blacklist

We discovered that 1.5B CPU models silently fail on spatial reasoning ("directly left of") and logic riddles ("knight and knave"). We added an interceptor in `router.py` to regex-match these traps and **instantly route them to the remote model**, bypassing the local model entirely to save time.

### Two-Pass Math Consistency

Small local models hallucinate math. We implemented a Two-Pass validation loop:

1. Generate an answer locally at `temperature=0.1`.
2. Generate a second answer locally at `temperature=0.7`.
3. If the answers differ, the local model is hallucinating. We discard the local answer and fall back to the remote API.

### Docker Cache Optimization

Modifying python files in `agent/` was forcing Docker to re-download the 1.1GB Qwen GGUF model, causing 90-second rebuilds. We fixed this by moving the `wget` model download step _above_ the `COPY agent/` step in the `Dockerfile`.

### Concurrency Bottleneck Fix

The Hugging Face grader hits the agent with 20 concurrent requests simultaneously.

- Originally, our `asyncio.Semaphore(20)` passed all 20 requests to the remote API at once, triggering a `429 Too Many Requests` rate-limit ban from Fireworks AI, resulting in empty string outputs and a 36.8% score.
- We reduced the semaphore to `asyncio.Semaphore(5)`, throttling the pipeline to avoid API rate limits and keep the local CPU queue from overflowing.

---

## 4. What We Tried & Failed

### Logprob Confidence Gating (Catastrophic Failure)

**The Idea:** Use the local model's `logprobs` to mathematically measure its confidence. If the average token probability drops below 75%, fall back to remote.
**The Failure:** In `llama-cpp-python`, fetching logprobs requires setting `logits_all=True` during initialization. This forces the CPU to evaluate logits for _every single token in the prompt_, inflating prompt processing time from 0.5s to 15s+. This caused our agent to constantly hit the 30-second timeout. When we tried removing `logits_all=True` but left the `logprobs=1` request, the internal C++ engine threw a `ValueError` on every request, crashing the local model and forcing 100% of tasks to the remote API. We entirely stripped logprobs out of the system.

### Remote Auditor Pattern (Token Waste)

**The Idea:** Use the local model to generate an answer, then send that answer to a cheap remote model (like `gemma-2b-it`) to double-check if the local answer is correct.
**The Failure:** It doubled the remote output tokens because the remote model had to read the prompt _and_ the local answer. The Two-Pass Math Consistency check (running the local model twice) proved to be much more efficient.

### Phase 9: Fine-Tuning the Local Model (Abandoned)

**The Idea:** Generate a synthetic dataset of 200 logic/math questions, use Unsloth to LoRA fine-tune the Qwen 1.5B model, and achieve a 0-token 100% accuracy score.
**The Failure:** Abandoned immediately upon reading the Discord announcement. The organizers stated the final test set will use hidden, randomized prompts to punish models that overfit to the public leaderboard. Fine-tuning a 1.5B model to solve logic puzzles generally is nearly impossible; it would just memorize the 200 synthetic questions. Our Hybrid architecture (falling back to a 27B+ parameter remote model for novel logic) is the only mathematically sound strategy to pass the hidden test set.

---

## 5. Highest-ROI Routing Overhaul (Top Priority AI Review)

A highly pragmatic review concluded that completely rewriting the project is a mistake, but overhauling the routing guardrails yields the highest Return On Investment (ROI). These suggestions should be prioritized:

### 1. Expanded Hallucination Blacklist (The Fast-Path Trap)
Small local models fail multi-step logic and ambiguous factuals silently. We must drastically expand our regex blacklist to force remote routing for:
* **Spatial:** `left of`, `right of`, `between`, `seated`, `arranged`
* **Paradox:** `knight/knave`, `truth-teller/liar`
* **Multi-step Logic:** `if and only if`, `unless`, `all/some/no`, `must be true`
* **Ambiguous Factual:** `which of the following`, `most likely`, `except`

### 2. Ranked Remote Model Routing
Instead of sending every remote request to a single fallback model, we should dynamically select the most capable remote model based on the domain (if the API key allows it):
* **Code/Debugging:** `kimi-k2p7-code` or similar specialized model.
* **Logic/Hard Factual:** `minimax-m3` or `gemma-4-31b-it`.
* **General/Summarization:** `gemma-4-26b-a4b-it` or `deepseek-v4-pro`.

### 3. Safe Prompt Cleanup (No Stop-word Stripping)
When compressing prompts for the remote API, we must **never strip stop-words**. Stripping words can accidentally delete critical constraints like `not`, `unless`, `only`, and `except`. Compression should only collapse whitespace, strip markdown decoration, and truncate massively long prompts, preserving all semantic negations.

### 4. Retry Only on Strict Validation Failures
To save remote tokens, do not blindly retry answers that "look" wrong. Only retry when the validator proves it is broken (e.g., Invalid JSON, syntax error in Python, completely empty answer, or a massive paragraph when only a single word was requested for Sentiment).

---

## 6. Further Strategic Improvements (Additional AI Insights)

After reviewing this architecture against the hidden-test-set constraints, a frontier AI suggested several high-leverage architectural improvements:

### 1. Extend Self-Consistency Beyond Math
Currently, the Two-Pass validation check (temperature 0.1 vs 0.7) is only used for `math`. We should expand this to `sentiment` and `ner`. If the local 1.5B model gives two different answers across temperatures, it's a silent hallucination. Falling back to remote costs almost no extra time (since local generation is fast) and provides a powerful confidence proxy without the massive `logits_all` penalty.

### 2. Never Return Empty Strings (The Ultimate Safety Net)
Our 36.8% score crash was caused by the agent returning empty strings when the remote API hit a `429` rate limit. We must enforce a final, ironclad fallback rule: **If Tier 2 fails, always return the discarded Tier 1 answer (even if it failed validation) instead of an empty string.** A wrong answer scores the same as an empty string, but an unvalidated guess still has a >0% chance of being right.

### 3. Replace Regex Traps with Semantic Embedding
Our hallucination blacklist relies on exact keyword matching (`"knight and knave"`). A hidden test set will use phrasing variance to break regex (e.g., `"a person who always tells the truth and a person who always lies"`). We should replace pure regex with a tiny, ultra-fast embedding model (like `all-MiniLM-L6-v2`) to check semantic similarity against a database of known logic traps.

### 4. Split Concurrency Semaphores
Currently, `main.py` uses a single `Semaphore(5)` for all tasks. Because local inference uses a separate `ThreadPoolExecutor(max_workers=1)` (due to the 2 vCPU limit), we should decouple the semaphores: one specifically for tracking local queue depth, and a separate, dynamically tuned semaphore purely for remote API requests. *(Note: Our `LocalModel` already correctly implements a `threading.Lock()` to serialize CPU usage and prevent context-thrashing, which aligns perfectly with this advice).*

### 5. Shift to Signal-Based Trust, Not Domain-Based
Rather than declaring "sentiment is always trustworthy locally," we should treat *every* domain with suspicion. The classifier should become a fast-path heuristic, while the mesh of validators (GBNF Grammars + Two-Pass Consistency) becomes the ultimate decider of whether a local answer is trusted. (Note: GBNF Grammars are already successfully implemented in `local_model.py` for NER and Sentiment!)

---

## 7. Major Structural Overhauls (Alternative Perspectives)

A second frontier model reviewed the architecture and provided deeper structural suggestions targeting the hidden test set vulnerabilities:

### 1. TF-IDF for Hallucination Trapping
Relying on exact regex keyword matching (`"knight and knave"`) is too brittle for randomized prompts. Since we already use a lightweight TF-IDF `DomainClassifier` (which takes ~10MB RAM and 5ms to run), we should train a separate binary classifier to detect "Semantic Complexity / Trick Questions" and use that to gate remote routing instead of the regex blacklist.

### 2. Code-Execution Math (Zero-Shot Sandbox)
The Two-Pass Math Consistency check (temp 0.1 vs 0.7) has a flaw: a small 1.5B model might hallucinate the exact same wrong answer twice. Instead of asking the local model for the final number, we should prompt it to write a Python script that solves the math problem. We extract the code and run it locally via `ast` or `RestrictedPython`. Code generation forces logical reasoning. If the code throws a `SyntaxError` or fails, we instantly route to the remote API.

### 3. Conditional "Fluff" Compression
Our `DomainCompressor` heavily strips "conversational fluff" to save tokens before sending to the remote Fireworks API. However, for complex logic and math riddles, what looks like "fluff" is often crucial semantic context needed to solve the trick. We should apply heavy compression *only* to deterministic domains (NER, Factual), and send the raw, unedited prompt for Logic and Math tasks to maximize the 27B+ remote model's accuracy.

---

## 8. Phase 2 Hardening (Immediate Next Steps)

Based on the implemented features (TF-IDF Trap Detector, Domain Compressor, Ranked Remote Routing), the final step before a major leaderboard push is "Phase 2 Hardening". This ensures the new systems don't introduce fragile edge cases:

### 1. Remote Model Fallback Cascade & Bad-Model Cache
Currently, if a ranked remote model (e.g., `gemma-4-31b-it`) returns a 404 or fails, the system might crash or return an empty string. 
**Action:** Implement a fallback cascade. If the primary ranked model fails, instantly try the next best model. Additionally, implement a `bad-model cache` (a session-level set of failed model IDs) so that if a model 404s once, we never waste time trying to call it again during the 30-second run.

### 2. NER Schema Normalization (Key Aliasing)
If the remote model outputs prose around the JSON, our postprocessor extracts it. However, remote models often hallucinate the JSON keys (e.g., using `people` instead of `person`, or `ORG` instead of `org`). 
**Action:** Add robust key aliasing in `evaluator.py` to coerce common hallucinations back into the strict `NEROutput` schema (e.g., `{"people":...}` -> `{"person":...}`).

### 3. Trap Detector Kill Switch (Env Var)
The new TF-IDF `trap_detector.py` is powerful but risks false positives (e.g., classifying "What is the capital of Japan?" as a logic puzzle).
**Action:** Add an environment variable toggle (e.g., `ENABLE_TFIDF_TRAP_DETECTOR=1`). If leaderboard testing shows massive false positives wasting remote tokens, we can instantly kill the TF-IDF layer and revert to the regex blacklist without altering code.

### 4. Docker vs. Local Model Consistency
The Dockerfile and local environment sometimes drift (e.g., Docker pulling Qwen 1.5B, while local testing uses Gemma 2B). 
**Action:** Ensure the `Dockerfile` exactly matches the intended strategy. For the hidden test set, picking one stable model and ensuring it loads flawlessly in the container is critical for Tier 1 stability.
