# Adversarial Arena — Adaptive Multi-Agent Belief Modeling

## Problem

LLMs often fail in adversarial settings because they react to local patterns instead of maintaining uncertainty-aware beliefs over opponent behavior.  
This project addresses that by forcing policy decisions to depend on evolving belief distributions, not single-step heuristics.

## Solution

- Environment: `env/environement.py` with `reset()`, `step()`, `state()`
- Tasks: static, adaptive, and deceptive opponents
- Observation includes belief state + opponent history
- Reward combines performance, adaptation quality, diversity, exploitability
- Reward shaping explicitly adds:
  - adaptation bonus for countering dominant belief
  - repetition penalty for overused actions
- Inference protocol is strict and machine-gradable

## LLM + Policy Stack

- `agents/llm_agent.py`
  - loads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` with dotenv
  - OpenAI-compatible `/chat/completions` call
  - strict JSON parse: `{"action":"..."}`
  - debug logs for prompt/raw response
  - fallback is random (never fixed)
- `agents/trained_agent.py`
  - loads learned policy from `trained_model/policy.json` (or artifacts)
  - belief-threshold adaptation (e.g. strong `bouncer` belief -> counter)
  - anti-repetition diversification

## Baseline and Training

- Baselines:
  - `RandomAgent` in `agents/baseline.py`
  - optional weak rule baseline
- Trajectories:
  - `training/collect_trajectories.py` -> `data/trajectories.json`
  - collects state, belief, history, action, reward, opponent action (320+ steps)
- Local train:
  - `training/train_local.py`
  - high-quality filtering (`reward > 0.5`) + reward-weighted learning
  - writes `training/artifacts/trained_policy.json` and `trained_model/policy.json`
- Colab train:
  - `training/train_unsloth.ipynb` (Unsloth + TRL)

## Evidence of Learning

- `training/evaluate_and_plot.py` generates:
  - `plots/reward_curve.png`
  - `plots/comparison.png`
  - `training/artifacts/evaluation_report.json`
- Measured result (latest local run):
  - baseline average score: `0.4569`
  - trained average score: `0.5139`
  - improvement: `+0.0570`
- Key claim: **trained agent adapts to opponent patterns and outperforms random baseline with clear margin.**

## Strict Inference Output

`inference.py` prints exactly:

```text
[START]
task=...
episode=...

[STEP]
state=...
belief=...
action=...
reward=...
commentary=...

[END]
score=...
```

## Run

```bash
cd "/Users/himatejagudi/Desktop/SRS"
pip install -r requirements.txt
PYTHONPATH=. python training/train_local.py
PYTHONPATH=. python training/evaluate_and_plot.py
PYTHONPATH=. python inference.py --task adaptive_opponent --agent trained --episode 1
```

API:

```bash
PYTHONPATH=. python app.py
curl -i http://127.0.0.1:7860/health
curl -i -X POST http://127.0.0.1:7860/reset
```
