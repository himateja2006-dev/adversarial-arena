from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import time

from dotenv import load_dotenv

from agents.adaptive_fallback_agent import AdaptiveFallbackAgent
from agents.baseline import BaselineAgent, RandomAgent
from agents.llm_agent import LLMBeliefAgent
from agents.trained_model_agent import TrainedAgent
from agents.trained_agent import TrainedBeliefAgent
from env.environement import AdversarialArenaEnv
from graders.grader import grade_episode
from tasks.scenarios import TASKS


def _best_versioned_model_dir() -> Path | None:
    root = Path("trained_models")
    if not root.exists():
        return None
    candidates = []
    for d in root.glob("model_v*"):
        meta = d / "metadata.json"
        if meta.exists():
            try:
                score = float(json.loads(meta.read_text(encoding="utf-8")).get("avg_reward", -1.0))
            except Exception:
                score = -1.0
            candidates.append((score, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _resolve_trained_model_dir() -> Path | None:
    best = _best_versioned_model_dir()
    if best is not None:
        return best
    fallback = Path("trained_model")
    if (fallback / "config.json").exists() or (fallback / "adapter_config.json").exists() or (fallback / "tokenizer_config.json").exists():
        return fallback
    return None


def _build_agent(agent_name: str, episode_idx: int, strict_model: bool = False):
    if agent_name == "baseline":
        return BaselineAgent(seed=episode_idx), "baseline_rule"
    if agent_name == "random":
        return RandomAgent(), "baseline_random"
    if agent_name == "trained":
        model_dir = _resolve_trained_model_dir()
        if model_dir is None:
            raise RuntimeError("Trained model not found (no valid trained_model/ or trained_models/model_v*/)")
        agent = TrainedAgent(str(model_dir))
        if not agent.available:
            raise RuntimeError("Trained model not loaded — aborting inference")
        return agent, "trained_model"
    if agent_name == "llm":
        return LLMBeliefAgent(), "llm_policy"
    if agent_name == "auto":
        model_dir = _resolve_trained_model_dir()
        if model_dir is not None:
            agent = TrainedAgent(str(model_dir))
            if agent.available:
                return agent, "trained_model"
            print("[INFERENCE] trained model exists but failed to load", file=sys.stderr)
        # fallback chain with LLM first
        llm = LLMBeliefAgent()
        if llm.enabled:
            return llm, "llm_policy"
        # strict mode allows trained_model or llm only.
        if strict_model:
            raise RuntimeError("Strict mode enabled: no loadable trained model and no LLM fallback.")
        print("[INFERENCE] fallback policy source=adaptive_rule_fallback", file=sys.stderr)
        return AdaptiveFallbackAgent(), "adaptive_rule_fallback"
    if agent_name == "auto_llm":
        model_dir = _resolve_trained_model_dir()
        if model_dir is not None:
            agent = TrainedAgent(str(model_dir))
            if agent.available:
                return agent, "trained_model"
            print("[INFERENCE] trained model exists but failed to load", file=sys.stderr)
            if strict_model:
                raise RuntimeError("Strict mode enabled: trained model exists but failed to load.")
        # fallback chain with LLM first
        llm = LLMBeliefAgent()
        if llm.enabled:
            return llm, "llm_policy"
        if strict_model:
            raise RuntimeError("Strict mode enabled: no loadable trained model and no LLM fallback.")
        print("[INFERENCE] fallback policy source=adaptive_rule_fallback", file=sys.stderr)
        return AdaptiveFallbackAgent(), "adaptive_rule_fallback"
    raise ValueError(f"Unknown agent: {agent_name}")


def run_episode(
    task_name: str,
    agent_name: str,
    episode_idx: int = 1,
    emit_logs: bool = True,
    run_seed: int | None = None,
    strict_model: bool = False,
    agent_override=None,
    policy_source_override: str | None = None,
) -> float:
    task = TASKS[task_name]
    seed = run_seed if run_seed is not None else int(time() * 1000) % 10_000_000
    env = AdversarialArenaEnv(opponent_policy=task.opponent_policy_factory(seed=seed + 99), seed=seed + 11)
    if agent_override is not None and policy_source_override is not None:
        agent, policy_source = agent_override, policy_source_override
    else:
        agent, policy_source = _build_agent(agent_name, episode_idx, strict_model=strict_model)

    state = env.reset(seed=seed)
    steps = []
    if emit_logs:
        print("[START]")
        print(f"task={task_name}")
        print(f"episode={episode_idx}")
    while True:
        action = agent.act(state)
        sr = env.step(action)
        steps.append(sr)
        short_state = {
            "ball_number": sr.state.ball_number,
            "runs": sr.state.runs,
            "wickets": sr.state.wickets,
            "required_run_rate": round(sr.state.required_run_rate, 3),
        }
        rounded_belief = {
            k: round(float(v), 3) for k, v in sr.state.belief_distribution_over_opponent_actions.items()
        }
        commentary = (
            f"Opponent={sr.info['opponent_action']}, runs={int(sr.info['outcome_runs'])}, "
            f"wicket={bool(sr.info['wicket'])}"
        )
        if emit_logs:
            print("")
            print("[STATE]")
            print(f"state={json.dumps(short_state, ensure_ascii=True)}")
            print(f"belief={json.dumps(rounded_belief, ensure_ascii=True)}")
            print(f"action={action}")
            print(f"reward={sr.reward:.4f}")
            print(f"commentary={commentary}")
        state = sr.state
        if sr.done:
            break
    metrics = grade_episode(steps)
    if emit_logs:
        print("")
        print("[END]")
        print(f"score={metrics.score:.4f}")
    return metrics.score


def main() -> None:
    load_dotenv()
    random.seed(None)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="adaptive_opponent", choices=list(TASKS.keys()))
    parser.add_argument("--agent", type=str, default="auto", choices=["baseline", "random", "trained", "llm", "auto", "auto_llm"])
    parser.add_argument("--episode", type=int, default=1, help="Starting episode index")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run sequentially")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strict-model", action="store_true", help="Fail if trained model is unavailable in auto mode")
    args = parser.parse_args()
    scores = []
    shared_agent = None
    shared_policy_source = None
    if args.episodes > 1:
        shared_agent, shared_policy_source = _build_agent(args.agent, args.episode, strict_model=args.strict_model)
    for i in range(args.episodes):
        ep = args.episode + i
        emit = args.episodes == 1
        score = run_episode(
            task_name=args.task,
            agent_name=args.agent,
            episode_idx=ep,
            run_seed=(None if args.seed is None else args.seed + i),
            emit_logs=emit,
            strict_model=args.strict_model,
            agent_override=shared_agent,
            policy_source_override=shared_policy_source,
        )
        scores.append(score)
    if args.episodes > 1:
        avg = sum(scores) / len(scores)
        print(json.dumps({"task": args.task, "agent": args.agent, "episodes": args.episodes, "avg_score": round(avg, 4)}))


if __name__ == "__main__":
    main()
