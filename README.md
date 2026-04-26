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
- `agents/trained_model_agent.py`
  - loads full fine-tuned model + tokenizer from `trained_model/` or `trained_models/model_v*/`
  - inference-first policy source for `inference.py --agent auto`
  - strict action validation with explicit fallback logging

## Baseline and Training

- Baselines:
  - `RandomAgent` in `agents/baseline.py`
  - optional weak rule baseline
- Trajectories:
  - `training/collect_trajectories.py` -> `data/trajectories.json`
  - collects state, belief, history, action, reward, opponent action (320+ steps)
  - filtered high-quality subset saved to `data/filtered_trajectories.json`
- Local train:
  - `training/train_local.py`
  - top-30%-or-`reward > 0.5` filtering + reward-weighted oversampling
  - writes `training/artifacts/trained_policy.json` and `trained_model/policy.json`
- Colab train:
  - `training/train_unsloth.ipynb` (Unsloth + TRL)
  - saves full model to `trained_model/`
  - versions each run to `trained_models/model_v*/` with `metadata.json`
  - logs `training/artifacts/training_logs.json` (reward history, rolling mean, action distribution, belief-action alignment)

## Evidence of Learning

- `training/evaluate_and_plot.py` generates:
  - `plots/reward_curve.png`
  - `plots/comparison.png`
  - `training/artifacts/evaluation_report.json`
- Measured result (latest local run):
  - before baseline avg reward: `0.2962`
  - after training avg reward: `0.4315`
  - improvement: `+0.1353`
- Behavior validation (`training/validate_behavior.py`):
  - when `belief["bouncer"] >= 0.6`, agent selects `balanced` counter strategy
- Key claim: **trained agent explicitly links belief to action and outperforms baseline by a strong margin.**

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

Each step additionally reports `policy_source` to verify whether actions come from `trained_model`, `trained_policy_map`, `llm_fallback`, or `random_fallback`.

## Run

```bash
cd "/Users/himatejagudi/Desktop/SRS"
pip install -r requirements.txt
PYTHONPATH=. python training/train_local.py
PYTHONPATH=. python training/evaluate_and_plot.py
python -m inference --task adaptive_opponent --agent auto --episode 1
python -m inference --task adaptive_opponent --agent auto --episodes 20
```

API:

```bash
PYTHONPATH=. python app.py
curl -i http://127.0.0.1:7860/health
curl -i -X POST http://127.0.0.1:7860/reset
```
