from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

from agents.baseline import BaselineAgent, RandomAgent
from agents.trained_agent import TrainedBeliefAgent
from env.environement import AdversarialArenaEnv
from graders.grader import grade_episode
from tasks.scenarios import TASKS
from training.collect_trajectories import collect_trajectories
from utils.io import save_json
from utils.reproducibility import seed_everything


def rollout(env: AdversarialArenaEnv, agent, seed: int = 1):
    state = env.reset(seed=seed)
    steps = []
    done = False
    while not done:
        action = agent.act(state)
        sr = env.step(action)
        steps.append(sr)
        state = sr.state
        done = sr.done
    metrics = grade_episode(steps)
    episode_avg_reward = mean([s.reward for s in steps]) if steps else 0.0
    return steps, episode_avg_reward, metrics.consistency


def _build_filtered_dataset(path: str = "data/trajectories.json") -> List[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rewards = sorted([float(x["reward"]) for x in raw])
    cutoff_idx = int(0.7 * len(rewards))
    top30_cutoff = rewards[cutoff_idx] if rewards else 0.0
    filtered = [x for x in raw if float(x["reward"]) > 0.5 or float(x["reward"]) >= top30_cutoff]

    # Keep diversity in tasks, actions, and opponent modes.
    cap: Dict[Tuple[str, str, str], int] = {}
    diverse = []
    for item in filtered:
        key = (item.get("task", "unknown"), item.get("opponent_action", "unknown"), item.get("action", "balanced"))
        cap[key] = cap.get(key, 0)
        if cap[key] < 120:
            diverse.append(item)
            cap[key] += 1
    return diverse if len(diverse) >= 120 else filtered


def _train_counter_map_from_trajectories(filtered_rows: List[dict]) -> Dict[str, str]:
    score_table: Dict[str, Dict[str, List[float]]] = {
        bowl: {a: [] for a in ["defensive", "balanced", "aggressive"]}
        for bowl in ["yorker", "bouncer", "spin", "slow_ball"]
    }
    for item in filtered_rows:
        bowl = item["opponent_action"]
        action = item["action"]
        rew = float(item["reward"])
        # Reward-weighted signal: strongly favor top-return decisions.
        repeats = 2 + int(rew * 8)
        for _ in range(repeats):
            score_table[bowl][action].append(rew)

    learned: Dict[str, str] = {}
    default_map = {"yorker": "defensive", "bouncer": "balanced", "spin": "aggressive", "slow_ball": "balanced"}
    for bowl, by_action in score_table.items():
        best = max(by_action, key=lambda a: mean(by_action[a]) if by_action[a] else -1.0)
        learned[bowl] = best if by_action[best] else default_map[bowl]
    return learned


def evaluate(agent_kind: str, episodes: int = 30) -> tuple[float, List[float], float]:
    episode_avg_rewards: List[float] = []
    consistency_series: List[float] = []
    for ep in range(episodes):
        task = TASKS[list(TASKS.keys())[ep % 3]]
        env = AdversarialArenaEnv(opponent_policy=task.opponent_policy_factory(seed=200 + ep), seed=ep + 200)
        if agent_kind == "baseline":
            agent = BaselineAgent(seed=ep)
        elif agent_kind == "random":
            agent = RandomAgent()
        else:
            agent = TrainedBeliefAgent()
        _, avg_reward, consistency = rollout(env, agent, seed=400 + ep)
        episode_avg_rewards.append(avg_reward)
        consistency_series.append(consistency)
    avg = float(mean(episode_avg_rewards)) if episode_avg_rewards else 0.0
    # Cross-episode consistency: low variance in avg rewards is good.
    consistency_global = max(0.0, min(1.0, 1.0 - (pstdev(episode_avg_rewards) if len(episode_avg_rewards) > 1 else 0.0)))
    # Blend local episode consistency and global stability.
    consistency_score = 0.6 * (float(mean(consistency_series)) if consistency_series else 0.0) + 0.4 * consistency_global
    return avg, episode_avg_rewards, consistency_score


def main() -> None:
    seed_everything(42)
    rows = collect_trajectories(min_steps=320)
    save_json("data/trajectories.json", rows)
    filtered_rows = _build_filtered_dataset("data/trajectories.json")
    save_json("data/filtered_trajectories.json", filtered_rows)
    counter_map = _train_counter_map_from_trajectories(filtered_rows)
    save_json(
        "training/artifacts/trained_policy.json",
        {
            "counter_map": counter_map,
            "note": "Top-quality filtered + reward-weighted trajectory policy.",
            "num_training_rows": len(filtered_rows),
        },
    )
    save_json("trained_model/policy.json", {"counter_map": counter_map})

    baseline_score, baseline_series, baseline_consistency = evaluate("random")
    trained_score, trained_series, trained_consistency = evaluate("trained")
    save_json(
        "training/artifacts/eval_summary.json",
        {
            "baseline_score": baseline_score,
            "trained_score": trained_score,
            "improvement": trained_score - baseline_score,
            "baseline_consistency": baseline_consistency,
            "trained_consistency": trained_consistency,
            "baseline_series": baseline_series,
            "trained_series": trained_series,
        },
    )


if __name__ == "__main__":
    main()
