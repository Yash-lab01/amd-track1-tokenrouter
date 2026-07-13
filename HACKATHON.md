# AMD Developer Hackathon: ACT II — Track 1
# Complete Development Log & Knowledge Base

> **Track:** Hybrid Token-Efficient Routing Agent
> **Deadline:** July 13, 2026 at 5:30 PM IST
> **Scoring:** Leaderboard-ranked (Remote token count + Output accuracy)
> **Compute:** Local Model (Qwen 2.5-1.5B CPU GGUF) + Remote Models (via Fireworks AI API)
> **Constraints:** Max 10GB Docker size, 60s container boot time, 30s response time per prompt, 600s overall runtime

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
- **Accuracy gate:** Must pass accuracy threshold before token efficiency matters

### Strict Environment Constraints

- **Compute:** Hugging Face Spaces grading environment — CPU-only (2 vCPUs, 4GB RAM). No GPU.
- **Time Budget:** 30-second execution time limit per prompt. Over 30s = score 0.
- **Total Runtime:** 600 seconds (10 minutes) for all tasks combined.
- **Network Isolation:** Only allowed to route remote requests to Fireworks AI API. No OpenAI, Anthropic, etc.
- **Hidden Test Set:** Final judging uses randomized, unseen prompts. Overfitting to public leaderboard fails.
- **8 Capability Domains:** Factual Knowledge, Math, Sentiment, Summarization, NER, Code Debugging, Logical Reasoning, Code Generation

### Judging (from Discord FAQ)

- The judge is an **LLM** that checks **semantic completeness**, not exact string matching
- **Sentiment** needs a label + one-sentence reason acknowledging both sides for mixed reviews
  - Mixed reviews: judge accepts Mixed/Neutral/Positive, **rejects Negative**
  - Reason must acknowledge BOTH positive and negative aspects
- **Factual** needs complete answers with explanations when requested
- **Summarization** has strict format requirements (exact number of sentences/bullets, word limits)
- **NER** needs ALL entities — missing one fails
- **Math** needs correct final answer with minor reasoning shown

### Public Validation Examples (from Judging FAQ)

**T01 - factual_knowledge:**
Prompt: "Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB."
Expected: Identifies red, green, blue; explains additive color mixing vs subtractive.

**T02 - mathematical_reasoning:**
Prompt: "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?"
Expected: 1672 units. (2400 - 888 + 800 - 640 = 1672)

**T03 - sentiment_classification:**
Prompt: "Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'"
Expected: Mixed/Neutral/Positive (NOT Negative). Reason must acknowledge both sides.

**T04 - text_summarization:**
Prompt: "Summarize the following passage in exactly two sentences: ..."
Expected: Exactly 2 sentences. Must capture both opportunities AND challenges. Wrong sentence count = fail.

**T04b - text_summarization:**
Prompt: "Summarize the following passage in exactly three bullet points, each no longer than 15 words: ..."
Expected: Exactly 3 bullets, each under 15 words. Wrong count or word limit = fail.

**T05 - named_entity_recognition:**
Prompt: "Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE: 'On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich to focus on large language model safety.'"
Expected: All 5 entities: Sundar Pichai (PERSON), March 15 2023 (DATE), Google (ORG), Zurich (LOCATION), ETH Zurich (ORG). Missing one = fail.

---

## 2. Allowed Fireworks Models

| Model                                            | Specialization    | Use Case                               |
| ------------------------------------------------ | ----------------- | -------------------------------------- |
| `accounts/fireworks/models/minimax-m3`           | Strong reasoning  | Logic, hard factual                    |
| `accounts/fireworks/models/kimi-k2p7-code`       | Code specialist   | Debugging, codegen                     |
| `accounts/fireworks/models/gemma-4-31b-it`       | General reasoning | Logic fallback, hard tasks             |
| `accounts/fireworks/models/gemma-4-26b-a4b-it`   | Fast general      | Factual, NER, sentiment, summarization |
| `accounts/fireworks/models/gemma-4-31b-it-nvfp4` | Quantized general | Fallback for any domain                |

---

## 3. Current Architecture (Latest Version)

### Flow

```
Prompt → Cache → Classify → Traps → Tier 0 (Deterministic) → Tier 1 (Remote) → Validate → Postprocess → Retry → Emergency Local Fallback
```

### Tier 0: Deterministic Solvers (0 tokens)

- **59 stable facts** (capitals, chemistry, people, history, science, geography, landmarks, math constants)
- **25+ math patterns** (percentages, factorials, GCD/LCM, mean/median, equations, geometry, word problems, unit conversions, probability, permutations, compound interest, population doubling, two trains, "more than" problems)
- **Sentiment rules** (lexicon-based for obvious cases)
- **Code fixes** (AST pattern matching for trivial bugs)

### Tier 1: Remote Fireworks AI (Remote tokens)

- **Chain-of-Thought prompts** with "Final Answer: X" extraction
- **Self-consistency voting** (3 parallel calls for MATH ONLY, majority vote)
  - Logic uses single call to prevent timeout (3 calls × 20s = potential 60s)
- **Few-shot examples** in all system prompts
- **Judge-aware prompts** (sentiment with reason, factual with explanation, summarization with exact format)
- **Model specialization:**
  - Kimi for code/debugging
  - Minimax for logic (always upgrades)
  - Gemma 26B for factual/NER/sentiment/summarization
- **Dynamic max_tokens** based on prompt complexity
- **Ranked fallback cascade** with bad-model cache

### Fallback: Local Qwen 1.5B (0 remote tokens)

- Loaded at startup (600s total budget is enough for 10-30s boot)
- Only used if ALL remote models fail
- Emergency best-effort answer (better than empty string)

### Key Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| API timeout | 20s | Fits in 30s per-task limit |
| `_call` retries | 2 (tenacity) | Prevents excessive retry stacking |
| `_call_with_temp` retries | 2 (manual) | For self-consistency calls |
| Self-consistency | Math only (3 parallel calls) | High value, parallel so no timeout |
| Concurrency | 3 | Prevents 429 rate limits |
| Task deadline | 28s | Ensures no task exceeds 30s |
| Local model timeout | 12s | Emergency fallback only |
| max_tokens | Full values (math: 500, logic: 600) | Enough for CoT reasoning |

---

## 4. File Structure

| File                     | Purpose                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| `agent/classifier.py`    | TF-IDF domain classifier (< 10MB, < 1ms inference)                           |
| `agent/trap_detector.py` | Semantic trap detector for logic puzzles                                     |
| `agent/compressor.py`    | Safe prompt compressor (preserves reasoning prompts)                         |
| `agent/local_model.py`   | Thread-safe Qwen 1.5B wrapper (emergency fallback)                           |
| `agent/remote_model.py`  | Remote API client with CoT, self-consistency, few-shot, model specialization |
| `agent/router.py`        | Core routing logic — remote-first with deterministic shortcuts               |
| `agent/evaluator.py`     | Deterministic solvers, validators, judge-aware postprocessing                |
| `main.py`                | Entry point (batch eval + Hugging Face Space auto-detection)                 |
| `eval.py`                | Evaluation harness against ground truth                                      |
| `streamlit_app.py`       | Streamlit web UI with live diagnostics                                       |
| `Dockerfile`             | CPU-optimized Docker build                                                   |

---

## 5. What We Tried & Succeeded

### ✅ Remote-First Architecture

Converted from local-first to remote-first. Local 1.5B model was hallucinating on hidden test questions. Remote models are the accuracy engine.

### ✅ Chain-of-Thought with Answer Extraction

Math and logic prompts ask models to "Think step by step. End with 'Final Answer: X'". Postprocessing extracts only the final answer. Models reason much better with CoT.

### ✅ Self-Consistency Voting (Math Only)

3 parallel calls at temperatures [0.1, 0.3, 0.5] for math. Majority vote on extracted answers. If 2+ agree, that's the answer. Falls back to first response if no majority.
- **Key insight:** Calls run in parallel via `asyncio.gather`, so total time = max(call times), NOT sum. This fits within 30s.
- **Originally tried for both math AND logic** — removed for logic to prevent timeout.

### ✅ Few-Shot Prompting

Every domain's system prompt includes 1-2 example Q&A pairs. Models follow patterns much better with examples.

### ✅ Judge-Aware Prompts

After reading the judging FAQ from Discord:
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

`asyncio.Semaphore(3)` to avoid HTTP 429 rate limits from Fireworks AI. (Was 5, then 8 — both caused issues)

### ✅ Emergency Local Fallback

If all remote models fail, local Qwen 1.5B generates a best-effort answer. Better than returning empty string (which scores 0).

### ✅ NER Schema Normalization

Aliases LLM hallucinations like `"people"` → `"person"`, `"places"` → `"location"` back to strict schema.

### ✅ GBNF Grammars (for local model)

Forces local model to output 100% valid JSON for NER.

### ✅ `reasoning_effort` Parameter Removal

Removed the non-standard `reasoning_effort: "none"` parameter that was causing 400 errors on Fireworks API. This was wasting API calls and latency on failed-then-retried requests.

### ✅ 28s Task Deadline

Added a 28s deadline in the router to ensure no single task exceeds the 30s per-task limit. If approaching deadline, skips retry and returns what it has.

---

## 6. What We Tried & Failed

### ❌ Logprob Confidence Gating (Catastrophic Failure)

**Idea:** Use local model's `logprobs` to measure confidence. If below 75%, fall back to remote.
**Failure:** `llama-cpp-python` requires `logits_all=True` which forces CPU to evaluate logits for every token, inflating processing time from 0.5s to 15s+. Caused constant 30-second timeouts.
**Result:** Entirely stripped logprobs out of the system.

### ❌ Remote Auditor Pattern (Token Waste)

**Idea:** Use local model to generate an answer, then send to a cheap remote model to double-check.
**Failure:** Doubled remote output tokens because the remote model had to read both the prompt AND the local answer.
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

### ❌ 3B Model Upgrade (Caused TIMEOUT)

**Idea:** Upgrade from Qwen 1.5B to Qwen 3B for better local reasoning.
**Failure:** 3B model (~2GB) took too long to load on 2 vCPU/4GB RAM environment, causing container TIMEOUT.
**Result:** Reverted to 1.5B model.

### ❌ Concurrency 8 (Caused 429 Rate Limits)

**Idea:** Increase concurrency from 5 to 8 for faster processing.
**Failure:** 8 concurrent tasks × 3 self-consistency calls = 24 concurrent API calls → HTTP 429 rate limit ban from Fireworks AI.
**Result:** Reduced to concurrency 3.

### ❌ Concurrency 5 (Caused TIMEOUT)

**Idea:** Original concurrency was 5.
**Failure:** 5 concurrent tasks × 3 self-consistency calls = 15 concurrent API calls → some 429s → retries stack up → 10-minute overall timeout.
**Result:** Reduced to concurrency 3.

### ❌ Tight max_tokens (Caused 42% Accuracy)

**Idea:** Reduce max_tokens to save tokens (math: 160, logic: 260).
**Failure:** Models couldn't finish step-by-step reasoning before being cut off. Math at 160 tokens means the model can't even complete a simple CoT. Accuracy crashed from 68.4% to 42%.
**Result:** Restored to full values (math: 500, logic: 600).

### ❌ 10s API Timeout (Caused 42% Accuracy)

**Idea:** Reduce API timeout from 15s to 10s to prevent overall timeout.
**Failure:** 10s was too short for complex prompts — the model got cut off mid-generation.
**Result:** Restored to 20s (sweet spot between 15s and 25s).

### ❌ Disabled Local Emergency Fallback (Caused 42% Accuracy)

**Idea:** Disable local model fallback to save boot time (lazy-load only when env var set).
**Failure:** When remote API failed, returned "Unable to determine" instead of using local model. Many tasks scored 0 that could have had >0% chance with local model.
**Result:** Re-enabled local model at startup, always available for emergency fallback.

### ❌ Self-Consistency for Logic (Caused TIMEOUT)

**Idea:** Use 3-call self-consistency for both math AND logic.
**Failure:** Logic tasks with 3 calls × 15s = potential 45s per task → exceeded 30s per-task limit.
**Result:** Self-consistency for math only. Logic uses single call.

### ❌ Model Cascade Limit (Caused 42% Accuracy)

**Idea:** Limit model cascade to 2 models to save time.
**Failure:** If first 2 models failed, gave up instead of trying all available models.
**Result:** Removed cascade limit — try all available models.

### ❌ "Safety Override" Code Block (Caused 42% Accuracy)

**Idea:** Another AI added a "safety override" block that slashed max_tokens, disabled self-consistency for math, reduced timeout, limited cascade, and disabled local fallback.
**Failure:** ALL of these changes combined destroyed accuracy from 68.4% to 42%.
**Result:** Removed ALL safety overrides. Restored original working values.

---

## 7. Leaderboard History

| Date | Accuracy | Tokens | Notes |
|------|----------|--------|-------|
| Jul 9 | 47.4% | ~2k | Initial local-first architecture |
| Jul 11 | 68.4% | ~3k | Remote-first with CoT + self-consistency |
| Jul 12 | 47.4% | ~2k | After removing local-first domains (broke something) |
| Jul 12 | TIMEOUT | - | After 3B model upgrade + concurrency 8 |
| Jul 12 | 42% | ~1k | After "safety overrides" (slashed max_tokens, disabled fallback) |
| Jul 13 | TBD | TBD | After restoring original values + math-only consistency + concurrency 3 |

### Top Leaderboard Teams (for reference)

| Rank | Team | Tokens | Accuracy |
|------|------|--------|----------|
| 14 | AdversaryAI | 3,459 | 94.7% |
| 10 | TOKENMAN | 3,101 | 89.5% |
| 13 | Token Frontier | 3,414 | 89.5% |
| 6 | k2k Kartik | 1,578 | 84.2% |
| 7 | YOLOAI | 2,024 | 84.2% |
| 5 | ZeroFire | 642 | 57.9% (failed gate) |

---

## 8. What We Didn't Use (And Why)

### ❌ PyTorch / Transformers

**Why not:** Bloats Docker image by 2-3GB, slow startup, high RAM usage. TF-IDF + scikit-learn is 10MB and <1ms inference.

### ❌ Embedding-Based Semantic Cache

**Why not:** Would need `sentence-transformers` (PyTorch dependency). Docker size constraints make it risky.

### ❌ ONNX Runtime + Mini Transformer

**Why not:** Requires one-time export step. TF-IDF classifier is sufficient for 8-domain classification.

### ❌ LLMLingua Prompt Compression

**Why not:** Additional dependency. Our safe cleanup (whitespace + markdown) is sufficient and doesn't risk removing critical context.

### ❌ Speculative Execution (Draft & Fix)

**Why not:** Sending local draft to remote for correction doubles tokens (remote reads prompt + draft). Self-consistency voting is more reliable.

### ❌ Task Decomposition

**Why not:** Breaking queries into sub-tasks adds complexity. Remote models handle multi-part questions well with CoT.

### ❌ Code-Execution Math

**Why not:** Asking model to write Python code that solves the problem, then executing it. Risky — model might write incorrect code. Self-consistency voting is safer.

### ❌ Parallel Multi-Model Ensemble

**Why not:** Querying 2-3 different models in parallel and picking majority answer. High token cost. Self-consistency with same model is more token-efficient.

### ❌ VADER/TextBlob for Sentiment

**Why not:** Would need additional dependencies. Our lexicon-based rules work for obvious cases, and remote models handle complex sentiment with judge-aware prompts.

### ❌ spaCy for NER

**Why not:** Would need `spaCy` + `en_core_web_sm` model (12MB). Remote models handle NER well with JSON-only prompts. Missing entities is the failure mode, and remote models are better at finding ALL entities.

### ❌ Critic Agent Loop (LLM-as-a-Judge)

**Why not:** Second remote call to check first answer doubles token cost. Not worth it for most tasks.

### ❌ Dynamic Few-Shot Injection (RAG for Prompts)

**Why not:** Would need a database of examples + similarity search. Static few-shot examples in system prompts work well enough.

---

## 9. Key Lessons Learned

1. **Accuracy first, tokens second** — The leaderboard gates on accuracy before token efficiency. Don't optimize tokens before clearing the accuracy gate.
2. **The judge is an LLM** — It checks semantic completeness, not exact string matching. Answers need to be complete, not just correct.
3. **Local 1.5B model hallucinates** — It can produce plausible but wrong answers. Validation is essential if using local.
4. **Hidden test set punishes overfitting** — Don't memorize public leaderboard questions. Build generalizable systems.
5. **30-second timeout is real** — Local model on 2 vCPU can take 12-30s for NER/code. Always have a timeout fallback.
6. **Rate limits are real** — 5+ concurrent requests → 429 ban. Throttle to 3 concurrent.
7. **Never return empty strings** — A wrong answer scores the same as empty, but has >0% chance of being right.
8. **CoT significantly improves reasoning** — Models reason much better when asked to think step by step.
9. **Self-consistency voting works** — 3 parallel calls with majority vote is a proven accuracy boost.
10. **Prompt compression is risky** — Removing "fluff" can delete critical constraints. Only do safe cleanup.
11. **Don't over-restrict remote models** — Slashing max_tokens or timeout to save tokens/timer destroys accuracy. Let the model finish reasoning.
12. **Parallel calls = max time, not sum** — `asyncio.gather` runs calls in parallel, so 3 calls × 20s = 20s total, not 60s.
13. **Safety overrides are dangerous** — Adding restrictions to "prevent timeout" can destroy accuracy. Find the sweet spot.
14. **3B model too slow for 4GB RAM** — Stick with 1.5B for the grading environment.

---

## 10. Environment Variables

| Variable                     | Required | Default                                      | Description                                |
| ---------------------------- | -------- | -------------------------------------------- | ------------------------------------------ |
| `FIREWORKS_API_KEY`          | ✅ Yes   | —                                            | Fireworks AI API key (injected by harness) |
| `FIREWORKS_BASE_URL`         | No       | `https://api.fireworks.ai/inference/v1`      | Fireworks endpoint                         |
| `ALLOWED_MODELS`             | No       | 5 Fireworks models                           | Comma-separated allowed model IDs          |
| `LOCAL_MODEL_PATH`           | No       | `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | Path to local GGUF model                   |
| `ENABLE_TFIDF_TRAP_DETECTOR` | No       | `0`                                          | Toggle semantic trap detector              |
| `PORT`                       | No       | `7860`                                       | Port for Streamlit web UI                  |

---

## 11. Future Improvements (If Time Permits)

### High Priority

1. **Expand deterministic math solvers** — Add multi-step word problems, fraction problems, cost calculations, derivatives
2. **Add more stable facts** — But only 100% verified facts
3. **Improve sentiment lexicon** — More words for better rule-based classification

### Medium Priority

4. **Code-execution math** — Ask model to write Python code that solves the problem, execute locally. More reliable for complex word problems.
5. **Dynamic semaphore tuning** — Adjust concurrency based on API response times.
6. **Better prompt engineering** — Experiment with different system prompts for each domain.

### Low Priority

7. **LLMLingua prompt compression** — Compress prompts 2-10x while preserving meaning.
8. **Task decomposition** — Break complex queries into sub-tasks, route each independently.
9. **ONNX runtime classifier** — Replace TF-IDF with mini transformer for better semantic understanding.

---

## 12. Domain Routing Strategy

| Domain            | Tier 0 (0 tokens)             | Tier 1 (Remote)                  | Model Preference    | Self-Consistency |
| ----------------- | ----------------------------- | -------------------------------- | ------------------- | ---------------- |
| **Math**          | 25+ deterministic patterns    | CoT + self-consistency (3 calls) | Gemma 26B → Minimax | ✅ Yes (3 calls) |
| **Factual**       | 59 stable facts               | Complete answer with explanation | Gemma 26B → Minimax | ❌ No            |
| **Sentiment**     | Lexicon rules (obvious cases) | Label + one-sentence reason      | Gemma 26B           | ❌ No            |
| **Logic**         | —                             | CoT single call                  | Minimax → Gemma 31B | ❌ No (timeout)  |
| **NER**           | —                             | JSON with all entities           | Gemma 26B           | ❌ No            |
| **Debugging**     | AST pattern fixes             | Corrected code only              | Kimi → Gemma 26B    | ❌ No            |
| **Codegen**       | AST pattern fixes             | Working code only                | Kimi → Gemma 26B    | ❌ No            |
| **Summarization** | —                             | Exact format (sentences/bullets) | Gemma 26B           | ❌ No            |

---

## 13. Token Budget Analysis

| Domain            | % of Tasks | Strategy             | Remote Tokens |
| ----------------- | ---------- | -------------------- | ------------- |
| Math (simple)     | ~15%       | Deterministic        | 0             |
| Math (complex)    | ~10%       | Remote (3 calls)     | ~600          |
| Factual (simple)  | ~10%       | Fact table           | 0             |
| Factual (complex) | ~10%       | Remote               | ~300          |
| Sentiment         | ~15%       | Remote               | ~100          |
| Summarization     | ~10%       | Remote               | ~400          |
| NER               | ~10%       | Remote               | ~300          |
| Logic             | ~10%       | Remote               | ~600          |
| Debugging         | ~5%        | Remote (Kimi)        | ~600          |
| Codegen           | ~5%        | Remote (Kimi)        | ~700          |
| **Total**         | **100%**   |                      | **~3,000-5,000** |

---

## 14. Common Errors (from Judging FAQ)

| Error | Cause | Fix |
|-------|-------|-----|
| PULL_ERROR | Docker image not public | Make image public, check tag |
| RUNTIME_ERROR | Container crashed | Check dependencies, entrypoint |
| TIMEOUT | Container too slow | Avoid large downloads at runtime, test worst-case |
| OUTPUT_MISSING | No output file | Check output path, write before exit |
| INVALID_RESULTS_SCHEMA | Wrong JSON structure | Validate JSON, check field names |
| MISSING_TASKS | Skipped tasks | Return one result per task, preserve task IDs |
| ACCURACY_GATE_FAILED | Answers wrong | Prioritize correctness, avoid generic answers |
| INFRA_ERROR | Backend issue | Not your fault, handled by judging process |