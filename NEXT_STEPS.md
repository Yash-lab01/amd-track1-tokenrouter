# AMD Track 1 — Next Steps Checklist
> **Deadline: July 11, 2026 at 15:00 UTC (8:30 PM IST)**

---

## ✅ DONE — Project Scaffold Built
All source files are in: `c:\Users\yashp\Desktop\AMD Hackathon\amd-track1-router\`

```
amd-track1-router/
├── agent/
│   ├── __init__.py        ✅
│   ├── classifier.py      ✅  TF-IDF domain router
│   ├── evaluator.py       ✅  Programmatic validators + math solver
│   ├── local_model.py     ✅  Thread-safe llama-cpp wrapper
│   ├── remote_model.py    ✅  Async Fireworks AI client
│   └── router.py          ✅  3-tier routing engine
├── data/
│   └── router_training.json  ✅  Classifier training data
├── main.py                ✅  Async entry point
├── eval.py                ✅  Local evaluation harness
├── Dockerfile             ✅  CPU-optimized container
├── requirements.txt       ✅  Minimal dependencies
├── .env.example           ✅  API key template
├── .gitignore             ✅
└── README.md              ✅
```

---

## 🔴 STEP 1 — Install Dependencies (Do This Now)

Open a terminal in the project folder:

```powershell
cd "C:\Users\yashp\Desktop\AMD Hackathon\amd-track1-router"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ `llama-cpp-python` compiles from source on first install.
> It may take 5-10 minutes. Make sure you have Visual Studio Build Tools installed.
> If it fails: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`

---

## 🔴 STEP 2 — Get Fireworks API Key

1. Go to: https://fireworks.ai
2. Sign in / register with your AMD AI Developer Program account
3. Navigate to: Profile → API Keys → Create New Key
4. Copy the key

Then:
```powershell
copy .env.example .env
# Open .env and paste your key into FIREWORKS_API_KEY=
```

Test it immediately:
```powershell
python -c "
import httpx, os
from dotenv import load_dotenv
load_dotenv()
r = httpx.post(
    'https://api.fireworks.ai/inference/v1/chat/completions',
    headers={'Authorization': f'Bearer {os.environ[\"FIREWORKS_API_KEY\"]}'},
    json={'model': 'accounts/fireworks/models/gemma2-9b-it',
          'messages': [{'role': 'user', 'content': 'Say hello.'}],
          'max_tokens': 10}
)
print(r.status_code, r.json()['choices'][0]['message']['content'])
"
```
Expected: `200  Hello!` (or similar)

---

## 🔴 STEP 3 — Download Local Gemma Model

1. Go to: https://huggingface.co/bartowski/gemma-2-2b-it-GGUF
2. Download: `gemma-2-2b-it-Q4_K_M.gguf` (~1.5 GB)
3. Create the models folder and place the file:

```powershell
mkdir models
# Move/copy the downloaded file to:
# C:\Users\yashp\Desktop\AMD Hackathon\amd-track1-router\models\gemma-2b-instruct-q4.gguf
```

Test local model loads correctly:
```powershell
python -c "
from agent.local_model import LocalModel
m = LocalModel()
print(m.generate('What is 2+2?', 'math'))
"
```
Expected: Some response containing "4"

---

## 🔴 STEP 4 — Pre-Bake the Classifier

Run this once before building Docker (saves build time, ensures classifier is ready):
```powershell
python -m agent.classifier
```

Expected output:
```
Training and saving classifier pickle ...
  [0.95] factual         ← What is the capital of France?
  [0.88] debugging       ← Fix this bug: def add(a, b): return a - b
  [0.92] summarization   ← Summarize the passage in 3 sentences.
  ...
Classifier saved to: agent/classifier.pkl
```

---

## 🟡 STEP 5 — Run Local Tests

```powershell
python eval.py
```

Check:
- Accuracy should be > 70% even before any tuning
- Timing: all tasks should complete in < 10 minutes total

Once actual tasks are revealed (they were revealed on July 6), add them to:
`data/test_cases.json` and re-run eval.

---

## 🟡 STEP 6 — Tune Based on Actual Tasks

After you see the real evaluation tasks:

1. **Categorize them** — which of the 8 domains do they belong to?
2. **Update `data/router_training.json`** — add real task examples for each domain
3. **Re-run classifier pre-bake**: `python -m agent.classifier`
4. **Adjust max_tokens** in `remote_model.py` if actual answers are longer/shorter
5. **Adjust validator logic** in `evaluator.py` if NER schema keys differ from expected
6. **Re-run eval**: `python eval.py`

---

## 🟡 STEP 7 — Build Docker Container

```powershell
# Make sure Docker Desktop is running
docker build -t tokrouter .
```

Expected build time: 5-15 minutes (llama-cpp-python compiles)

> ⚠️ The Gemma model must be in `./models/` before building.
> Uncomment the `COPY models/ ./models/` line in the Dockerfile.

Test the container locally:
```powershell
# Create test input
echo '[{"task_id": "t1", "prompt": "What is the capital of Japan?"}]' > data\test_input.json
mkdir output

docker run ^
  -e FIREWORKS_API_KEY=your_key_here ^
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 ^
  -e ALLOWED_MODELS=accounts/fireworks/models/gemma2-9b-it ^
  -v "%cd%\data:/input" ^
  -v "%cd%\output:/output" ^
  tokrouter
```

Check `output/results.json` — should contain the answers.

---

## 🟡 STEP 8 — Check Docker Image Size

```powershell
docker image ls tokrouter
```

**Must be under 10GB compressed.** Expected: ~4-6GB (Python base + model + llama).

If too large:
- Switch to `python:3.11-alpine` base image (risky — gcc issues)
- Use a smaller quantization: `Q2_K` instead of `Q4_K_M` (~800MB instead of 1.5GB)

---

## 🟢 STEP 9 — GitHub Repository

1. Create a new PUBLIC repository on GitHub: `amd-track1-tokrouter`
2. Push the project:

```powershell
cd "C:\Users\yashp\Desktop\AMD Hackathon\amd-track1-router"
git init
git add .
git commit -m "Initial submission: TokRouter — Hybrid Token-Efficient Agent"
git remote add origin https://github.com/YOUR_USERNAME/amd-track1-tokrouter.git
git push -u origin main
```

> ⚠️ Make sure `.env` and `models/*.gguf` are in `.gitignore` (they are).
> ⚠️ Repository MUST be public for judges to review.

---

## 🟢 STEP 10 — Deploy Demo Application

The submission requires a live Application URL. Options (all free):

### Option A: HuggingFace Spaces (Recommended — Docker support)
1. Go to: https://huggingface.co/spaces
2. Click "Create new Space" → Select "Docker" type
3. Upload your code or connect GitHub repo
4. Set secrets: FIREWORKS_API_KEY, ALLOWED_MODELS

### Option B: Render.com (Free tier)
1. Go to: https://render.com
2. New → Web Service → Connect GitHub repo
3. Environment: Docker
4. Add env vars: FIREWORKS_API_KEY, etc.
5. Free tier gives you a public URL

### Option C: Streamlit Cloud (If you add a simple Streamlit UI)
Add a `streamlit_app.py` with a simple text input form, then deploy to:
https://share.streamlit.io

> The demo URL just needs to be accessible — a simple web interface showing the agent working is sufficient.

---

## 🟢 STEP 11 — Create Submission Assets

### Video (Max 5 minutes)
Record using OBS or Windows Game Bar (Win + G):
- **0:00-0:30** — Problem statement: token efficiency challenge
- **0:30-1:30** — Architecture walkthrough: 3 tiers, routing logic
- **1:30-3:30** — Live demo: show tasks being routed to local vs remote
- **3:30-4:30** — Results: token count, accuracy, local-hit-rate stats
- **4:30-5:00** — Gemma usage, closing

Upload to YouTube (unlisted) or Loom.

### Slide PDF (Required)
Create 6-8 slides in Canva or Google Slides:
1. Title + Team
2. Problem: token cost challenge
3. Solution: 3-tier routing architecture diagram
4. Key innovations (Gemma local, programmatic validators, speculative correction)
5. Results: accuracy %, tokens saved vs baseline
6. Tech stack
7. Live demo screenshot

Export as PDF.

### Cover Image (16:9 PNG)
Create in Canva: project name + architecture diagram thumbnail.

---

## 🟢 STEP 12 — Submit on lablab.ai

1. Go to: https://lablab.ai → Your hackathon dashboard
2. Click "Submit project"
3. Fill in:
   - **Project Title**: TokRouter — Hybrid Token-Efficient Routing Agent
   - **Short Description** (max 255 chars): "A 3-tier AI routing agent that uses local Gemma-2B + TF-IDF classification + programmatic validation to minimize Fireworks API token usage while maintaining accuracy."
   - **Long Description**: Copy from README, expand with results
   - **Tags**: AI Agent, Token Optimization, Fireworks AI, Gemma, AMD, ROCm, Python
   - **Cover Image**: Upload 16:9 PNG
   - **Video URL**: YouTube/Loom link
   - **Slide PDF**: Upload PDF
   - **GitHub URL**: Your public repo
   - **Application URL**: HuggingFace Spaces or Render URL

**Submit by: July 11, 2026 at 15:00 UTC (8:30 PM IST)**

---

## Timeline Summary

| When | What |
|---|---|
| **Today (July 7)** | Steps 1-4: Install deps, get API key, download model, pre-bake classifier |
| **July 7-8** | Step 5-6: Run evals with real tasks, tune routing logic |
| **July 9** | Step 7-8: Build & test Docker container |
| **July 10** | Steps 9-10: Push to GitHub, deploy demo |
| **July 11 morning** | Steps 11-12: Record video, create slides, SUBMIT |
| **July 11 by 8:30 PM IST** | ✅ Submission complete |

---

## Key URLs

| Resource | URL |
|---|---|
| Fireworks AI Platform | https://fireworks.ai |
| Fireworks API Docs | https://docs.fireworks.ai |
| Gemma GGUF Models | https://huggingface.co/bartowski/gemma-2-2b-it-GGUF |
| HuggingFace Spaces | https://huggingface.co/spaces |
| lablab.ai Dashboard | https://lablab.ai |
| AMD Developer Cloud | https://developer.amd.com/amd-developer-cloud |
