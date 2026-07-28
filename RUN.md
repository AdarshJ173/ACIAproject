# ACIA — Run Commands

## Setup (first time / after moving the project)

```bash
cd /Users/yashaswinsharma/Desktop/ACIAproject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Activate the venv (every new terminal)

```bash
source .venv/bin/activate
```

> On this Mac use `python` **after** activating the venv, or always use `python3`.

---

## Main commands

```bash
# Full pipeline — generate data, train models, run agent cycle
python main.py

# With OpenRouter LLM planning
python main.py --llm

# Quiet output
python main.py --llm --quiet
```

## Individual stages

```bash
# Generate data + train models only
python main.py --setup

# One agent decision + execution cycle
python main.py --cycle

# Cycle with LLM (max 10 API calls)
python main.py --cycle --llm --llm-budget 10
```

## API server

```bash
python main.py --api
# Swagger UI → http://localhost:8000/docs
# ReDoc      → http://localhost:8000/redoc

# Custom host/port
python main.py --api --host 127.0.0.1 --port 9000
```

## Optional LLM env vars

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
export OPENROUTER_MODEL=openrouter/free
```

Works without a key — falls back to rule-based decisions.
