# AMD Developer Hackathon: ACT II — Track 1

# Full Hackathon Details & Development Log

> **Track:** Hybrid Token-Efficient Routing Agent  
> **Deadline:** July 11, 2026 at 15:00 UTC (8:30 PM IST)  
> **Scoring:** Leaderboard-ranked (Remote token count + Output accuracy)  
> **Compute:** Local Model (Qwen 2.5-1.5B CPU GGUF) + Remote Models (via Fireworks AI API)  
> **Constraints:** Max 10GB Docker size, 60s container boot time, 30s response time per prompt, 10m overall runtime

---

## 1. Hackathon Overview

### Track 1: Hybrid Token-Efficient Routing Agent

Build an AI agent that completes a fixed set of tasks autonomously by deciding in real time whether to use a local model or call a remote model via Fireworks AI credits. The goal: pick the cheapest option every time, without falling below the accuracy threshold.

### Scoring Formula

```
Score = minimize(remote_tokens) + pass(accuracy_threshold)
```

- **Local tokens = FREE** (count as zero toward final score)
- **Remote tokens = EXPENSIVE** (count toward final score)
- **Top performers:** 100% accuracy with only 2.5k-5k tokens

### Strict Environment Constraints

- **Compute:** Hugging Face Spaces grading environment — CPU-only (2 vCPUs, 4GB RAM). No GPU.
- **Time Budget:** 30-second execution time limit per prompt. Over 30s = score 0.
- **Network Isolation:** Only allowed to route remote requests to Fireworks AI API. No OpenAI, Anthropic, etc.
- **Hidden Test Set:** Final judging uses randomized, unseen prompts. Overfitting to public leaderboard fails.
- **8 Capability Domains:** Factual Knowledge, Math, Sentiment, Summarization, NER, Code Debugging, Logical Reasoning, Code Generation

### Judging (from Discord FAQ)

- The judge is an **LLM** that checks **semantic completeness**, not exact string matching
- **Sentiment** needs a label + one-sentence reason acknowledging both sides for mixed reviews
- **Factual** needs complete answers with explanations when requested
- **Summarization** has strict format requirements (exact number of sentences/bullets, word limits)
- **NER** needs ALL entities — missing one fails
- **Math** needs correct final answer with minor reasoning shown

---

## 2. Current Architecture (v3 — Remote-First with Advanced Techniques)

### Flow

```
Prompt → Cache → Classify → Traps → Tier 0 (Deterministic) → Tier 1 (Remote with CoT) → Validate → Postprocess → Retry → Emergency Local Fallback
```

### Tier 0: Deterministic Solvers (0 tokens)

- **59 stable facts** (capitals, chemistry, people, history, science, geography, landmarks, math constants)
- **25+ math patterns** (percentages, factorials, GCD/LCM, mean/median, equations, geometry, word problems, unit conversions, probability, permutations, compound interest, population doubling, two trains, "more than" problems)
- **Sentiment rules** (lexicon-based for obvious cases)
- **Code fixes** (AST pattern matching for trivial bugs)

### Tier 1: Remote Fireworks AI (Remote tokens)

- **Chain-of-Thought prompts** with "Final Answer: X" extraction
- **Self-consistency voting** (3 parallel calls for math/logic, majority vote)
- **Few-shot examples** in all system prompts
- **Judge-aware prompts** (sentiment with reason, factual with explanation, summarization with exact format)
- **Model specialization:**
  - Kimi for code/debugging
  - Minimax for logic
  - Gemma 26B for factual/NER/sentiment/summarization
- **Dynamic max_tokens** based on prompt complexity
- **Ranked fallback cascade** with bad-model cache

### Fallback: Local Qwen 1.5B (0 remote tokens)

- Only used if ALL remote models fail
- Emergency best-effort answer (better than empty string)

---

## 3. What We Tried & Succeeded

### ✅ Remote-First Architecture

Converted from local-first to remote-first. Local 1.5B model was hallucinating on hidden test questions. Remote models are the accuracy engine.

### ✅ Chain-of-Thought with Answer Extraction

Math and logic prompts ask models to "Think step by step. End with 'Final Answer: X'". Postprocessing extracts only the final answer. Models reason much better with CoT.

### ✅ Self-Consistency Voting

3 parallel calls at temperatures [0.1, 0.3, 0.5] for math and logic. Majority vote on extracted answers. If 2+ agree, that's the answer. Falls back to first response if no majority.

### ✅ Few-Shot Prompting

Every domain's system prompt includes 1-2 example Q&A pairs. Models follow patterns much better with examples.

### ✅ Judge-Aware Prompts

After reading the judging FAQ from Discord, updated prompts:

- Sentiment: "Label: one-sentence reason" acknowledging both sides
- Factual: Complete answers with explanations when requested
- Summarization: Follows EXACT format (number of sentences, bullet points, word limits)
- NER: "Extract ALL named entities" — missing one fails

### ✅ Expanded Deterministic Solvers

- 59 stable facts (was 22)
- 25+ math patterns (was ~10)
- All tested and verified working

### ✅ Model Specialization

- Kimi for code/debugging
- Minimax for logic (always upgrades)
- Gemma 26B for factual/NER/sentiment/summarization

### ✅ Ranked Fallback Cascade

If a model 404s, instantly cascade to next best. Bad-model cache prevents retrying failed models.

### ✅ Safe Prompt Compression

Preserves full prompts for logic/math/NER. Only cleans whitespace/markdown for other domains. No TF-IDF sentence deletion.

### ✅ Docker Cache Optimization

Model download step moved above `COPY agent/` in Dockerfile to prevent 90-second rebuilds.

### ✅ Concurrency Throttling

`asyncio.Semaphore(5)` to avoid HTTP 429 rate limits from Fireworks AI.

### ✅ Emergency Local Fallback

If all remote models fail, local Qwen 1.5B generates a best-effort answer. Better than returning empty string (which scores 0).

### ✅ NER Schema Normalization

Aliases LLM hallucinations like `"people"` → `"person"`, `"places"` → `"location"` back to strict schema.

### ✅ GBNF Grammars (for local model)

Forces local model to output 100% valid JSON and exact sentiment labels when used.

---

## 4. What We Tried & Failed

### ❌ Logprob Confidence Gating (Catastrophic Failure)

**Idea:** Use local model's `logprobs` to measure confidence. If below 75%, fall back to remote.  
**Failure:** `llama-cpp-python` requires `logits_all=True` which forces CPU to evaluate logits for every token, inflating processing time from 0.5s to 15s+. Caused constant 30-second timeouts. When we tried removing `logits_all=True` but left `logprobs=1`, the C++ engine threw `ValueError` on every request.  
**Result:** Entirely stripped logprobs out of the system.

### ❌ Remote Auditor Pattern (Token Waste)

**Idea:** Use local model to generate an answer, then send to a cheap remote model to double-check.  
**Failure:** Doubled remote output tokens because the remote model had to read both the prompt AND the local answer. Two-Pass Math Consistency (running local twice) proved more efficient.  
**Result:** Removed remote auditor entirely.

### ❌ Fine-Tuning the Local Model (Abandoned)

**Idea:** Generate synthetic dataset of 200 logic/math questions, use Unsloth to LoRA fine-tune Qwen 1.5B.  
**Failure:** Abandoned after Discord announcement that final test set uses hidden, randomized prompts. Fine-tuning a 1.5B model to solve logic puzzles generally is nearly impossible; it would just memorize the synthetic questions.  
**Result:** Hybrid architecture (remote fallback for novel logic) is the only sound strategy.

### ❌ TF-IDF Sentence Deletion for Compression

**Idea:** Use TF-IDF to rank sentences and discard least important ones before sending to remote.  
**Failure:** For complex logic and math riddles, what looks like "fluff" is often crucial semantic context. Removing sentences broke accuracy on constraint-heavy tasks.  
**Result:** Removed TF-IDF summarization. Only do safe cleanup (whitespace, markdown).

### ❌ Local-First Routing for Risky Domains

**Idea:** Trust local model for factual, logic, summarization, NER, debugging, codegen.  
**Failure:** Local 1.5B model answers may be plausible but wrong. Hidden test questions punish local hallucination. Accuracy dropped from 47.4% to 42%.  
**Result:** Converted to remote-first. Local model only for emergency fallback.

### ❌ Broad Factual Regex Shortcuts

**Idea:** Expand fact table aggressively to cover more questions with 0 tokens.  
**Failure:** Broad facts can be stale or wrong. Risk of returning incorrect answers for edge cases.  
**Result:** Kept fact table small and stable (59 facts, all 100% verified).

---

## 5. What We Removed

### Removed from Normal Routing

- **Local LLM generation** — removed from normal decision path, only emergency fallback
- **Two-Pass Math Consistency** (temp 0.1 vs 0.7) — replaced by remote self-consistency voting
- **TF-IDF sentence deletion** — too risky for constraint-heavy tasks
- **Remote Auditor pattern** — wasted tokens
- **Logprob confidence gating** — caused timeouts
- **Multi-step local consensus** — unreliable on 1.5B model

### Disabled by Default

- **TF-IDF Trap Detector** — `ENABLE_TFIDF_TRAP_DETECTOR=0` by default. False positives could force wrong routing. Can be enabled via env var if needed.

### Kept but Simplified

- **Prompt compression** — only safe cleanup now (whitespace, markdown). No semantic compression.
- **Retry logic** — only retries NER/code/logic malformed outputs. No blind retries.

---

## 6. What We Didn't Use (And Why)

### ❌ PyTorch / Transformers

**Why not:** Bloats Docker image by 2-3GB, slow startup, high RAM usage. TF-IDF + scikit-learn is 10MB and <1ms inference.

### ❌ Embedding-Based Semantic Cache

**Why not:** Would need `sentence-transformers` (PyTorch dependency). The strategy guide recommends it, but Docker size constraints make it risky. Could add in future if needed.

### ❌ ONNX Runtime + Mini Transformer

**Why not:** Requires one-time export step. TF-IDF classifier is sufficient for 8-domain classification.

### ❌ LLMLingua Prompt Compression

**Why not:** Additional dependency. Our safe cleanup (whitespace + markdown) is sufficient and doesn't risk removing critical context.

### ❌ Speculative Execution (Draft & Fix)

**Why not:** Sending local draft to remote for correction doubles tokens (remote reads prompt + draft). Self-consistency voting is more reliable.

### ❌ Task Decomposition

**Why not:** Breaking queries into sub-tasks adds complexity. Remote models handle multi-part questions well with CoT.

### ❌ Code-Execution Math

**Why not:** Asking model to write Python code that solves the problem, then executing it. Risky — model might write incorrect code. Self-consistency voting is safer. Could add in future.

### ❌ Parallel Multi-Model Ensemble

**Why not:** Querying 2-3 different models in parallel and picking majority answer. High token cost. Self-consistency with same model is more token-efficient.

---

## 7. Development Timeline

### Phase 1: Initial Setup

- Built 3-tier hybrid routing (deterministic → local → remote)
- TF-IDF domain classifier
- Basic validators (NER JSON, Python syntax, sentiment labels)
- Docker containerization

### Phase 2: Local Model Optimization

- Added GBNF grammars for NER and sentiment
- Two-Pass Math Consistency check
- Hallucination blacklist (regex traps for knight/knave, spatial puzzles)
- Docker cache optimization

### Phase 3: Concurrency & Stability

- Fixed 429 rate limit crash (reduced Semaphore from 20 to 5)
- Added "never return empty string" safety net
- Remote model fallback cascade
- Bad-model cache

### Phase 4: Accuracy Improvements

- NER schema normalization (key aliasing)
- TF-IDF trap detector
- Domain compressor (TF-IDF + markdown cleanup)
- Expanded hallucination blacklist

### Phase 5: Remote-First Overhaul (Current)

- Removed local LLM from normal routing
- Added Chain-of-Thought prompts with answer extraction
- Added self-consistency voting (3 parallel calls for math/logic)
- Added few-shot examples in all system prompts
- Added judge-aware prompts (from Discord FAQ analysis)
- Expanded deterministic solvers (59 facts, 25+ math patterns)
- Dynamic max_tokens based on prompt complexity
- Improved postprocessing for CoT outputs

---

## 8. Key Files

| File                     | Purpose                                                                      | Lines |
| ------------------------ | ---------------------------------------------------------------------------- | ----- |
| `agent/router.py`        | Core routing logic — remote-first with deterministic shortcuts               | ~200  |
| `agent/remote_model.py`  | Remote API client with CoT, self-consistency, few-shot, model specialization | ~300  |
| `agent/evaluator.py`     | Deterministic solvers, validators, judge-aware postprocessing                | ~900  |
| `agent/compressor.py`    | Safe prompt compressor (preserves reasoning prompts)                         | ~100  |
| `agent/classifier.py`    | TF-IDF domain classifier                                                     | ~120  |
| `agent/trap_detector.py` | Semantic trap detector for logic puzzles                                     | ~80   |
| `agent/local_model.py`   | Thread-safe Qwen 1.5B wrapper (emergency fallback)                           | ~170  |
| `main.py`                | Entry point (batch eval + Hugging Face Space auto-detection)                 | ~117  |
| `eval.py`                | Evaluation harness against ground truth                                      | ~200  |
| `streamlit_app.py`       | Streamlit web UI with live diagnostics                                       | ~440  |
| `Dockerfile`             | CPU-optimized Docker build                                                   | ~57   |

---

## 9. Environment Variables

| Variable                     | Required | Default                                      | Description                                |
| ---------------------------- | -------- | -------------------------------------------- | ------------------------------------------ |
| `FIREWORKS_API_KEY`          | ✅ Yes   | —                                            | Fireworks AI API key (injected by harness) |
| `FIREWORKS_BASE_URL`         | No       | `https://api.fireworks.ai/inference/v1`      | Fireworks endpoint                         |
| `ALLOWED_MODELS`             | No       | 5 Fireworks models                           | Comma-separated allowed model IDs          |
| `LOCAL_MODEL_PATH`           | No       | `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | Path to local GGUF model                   |
| `ENABLE_TFIDF_TRAP_DETECTOR` | No       | `0`                                          | Toggle semantic trap detector              |
| `PORT`                       | No       | `7860`                                       | Port for Streamlit web UI                  |

---

## 10. Allowed Fireworks Models

| Model                                            | Specialization    | Use Case                               |
| ------------------------------------------------ | ----------------- | -------------------------------------- |
| `accounts/fireworks/models/minimax-m3`           | Strong reasoning  | Logic, hard factual                    |
| `accounts/fireworks/models/kimi-k2p7-code`       | Code specialist   | Debugging, codegen                     |
| `accounts/fireworks/models/gemma-4-31b-it`       | General reasoning | Logic fallback, hard tasks             |
| `accounts/fireworks/models/gemma-4-26b-a4b-it`   | Fast general      | Factual, NER, sentiment, summarization |
| `accounts/fireworks/models/gemma-4-31b-it-nvfp4` | Quantized general | Fallback for any domain                |

---

## 11. Future Improvements (If Time Permits)

### High Priority

1. **Hybrid local-first with validation** — Add local model back for NER/code/sentiment with validation. If local output passes validation, use it (0 tokens). This could cut token usage 40-60%.
2. **Fine-tune local model** — Use AMD Developer Cloud GPU to LoRA fine-tune Qwen on task-specific data. "The single highest-ROI action" per strategy guide.
3. **Embedding-based semantic cache** — Cache similar queries using cosine similarity > 0.96. Bypasses inference entirely for prompt variants.

### Medium Priority

4. **Code-execution math** — Ask model to write Python code that solves the problem, execute locally. More reliable for complex word problems.
5. **Parallel multi-model ensemble** — For hard logic, query 2-3 different models and pick majority answer.
6. **Dynamic semaphore tuning** — Adjust concurrency based on API response times.

### Low Priority

7. **LLMLingua prompt compression** — Compress prompts 2-10x while preserving meaning.
8. **Task decomposition** — Break complex queries into sub-tasks, route each independently.
9. **ONNX runtime classifier** — Replace TF-IDF with mini transformer for better semantic understanding.

---

## 12. Lessons Learned

1. **Accuracy first, tokens second** — The leaderboard gates on accuracy before token efficiency. Don't optimize tokens before clearing the accuracy gate.
2. **The judge is an LLM** — It checks semantic completeness, not exact string matching. Answers need to be complete, not just correct.
3. **Local 1.5B model hallucinates** — It can produce plausible but wrong answers. Validation is essential if using local.
4. **Hidden test set punishes overfitting** — Don't memorize public leaderboard questions. Build generalizable systems.
5. **30-second timeout is real** — Local model on 2 vCPU can take 12-30s for NER/code. Always have a timeout fallback.
6. **Rate limits are real** — 20 concurrent requests → 429 ban. Throttle to 5 concurrent.
7. **Never return empty strings** — A wrong answer scores the same as empty, but has >0% chance of being right.
8. **CoT significantly improves reasoning** — Models reason much better when asked to think step by step.
9. **Self-consistency voting works** — 3 parallel calls with majority vote is a proven accuracy boost.
10. **Prompt compression is risky** — Removing "fluff" can delete critical constraints. Only do safe cleanup.
