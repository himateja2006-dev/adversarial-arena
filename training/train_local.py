from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

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
    return steps, grade_episode(steps).score


def _train_counter_map_from_trajectories(path: str = "data/trajectories.json") -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    score_table: Dict[str, Dict[str, List[float]]] = {
        bowl: {a: [] for a in ["defensive", "balanced", "aggressive"]}
        for bowl in ["yorker", "bouncer", "spin", "slow_ball"]
    }
    # High-quality filtering first; fallback to weighted full data if too small.
    high_reward = [x for x in raw if float(x["reward"]) > 0.5]
    usable = high_reward if len(high_reward) >= 80 else raw
    for item in usable:
        bowl = item["opponent_action"]
        action = item["action"]
        rew = float(item["reward"])
        # Reward-weighted signal: high reward samples count more.
        repeats = 1 + int(rew * 5)
        for _ in range(repeats):
            score_table[bowl][action].append(rew)

    learned: Dict[str, str] = {}
    default_map = {"yorker": "defensive", "bouncer": "balanced", "spin": "aggressive", "slow_ball": "balanced"}
    for bowl, by_action in score_table.items():
        best = max(by_action, key=lambda a: mean(by_action[a]) if by_action[a] else -1.0)
        learned[bowl] = best if by_action[best] else default_map[bowl]
    return learned


def evaluate(agent_kind: str, episodes: int = 20) -> tuple[float, List[float]]:
    scores: List[float] = []
    for ep in range(episodes):
        task = TASKS[list(TASKS.keys())[ep % 3]]
        env = AdversarialArenaEnv(opponent_policy=task.opponent_policy_factory(seed=200 + ep), seed=ep + 200)
        if agent_kind == "baseline":
            agent = BaselineAgent(seed=ep)
        elif agent_kind == "random":
            agent = RandomAgent()
        else:
            agent = TrainedBeliefAgent()
        _, score = rollout(env, agent, seed=400 + ep)
        scores.append(score)
    return (float(mean(scores)) if scores else 0.0, scores)


def main() -> None:
    seed_everything(42)
    rows = collect_trajectories(min_steps=320)
    save_json("data/trajectories.json", rows)
    counter_map = _train_counter_map_from_trajectories("data/trajectories.json")
    save_json(
        "training/artifacts/trained_policy.json",
        {
            "counter_map": counter_map,
            "note": "Reward-weighted trajectory policy.",
        },
    )
    save_json("trained_model/policy.json", {"counter_map": counter_map})

    baseline_score, baseline_series = evaluate("random")
    trained_score, trained_series = evaluate("trained")
    save_json(
        "training/artifacts/eval_summary.json",
        {
            "baseline_score": baseline_score,
            "trained_score": trained_score,
            "improvement": trained_score - baseline_score,
            "baseline_series": baseline_series,
            "trained_series": trained_series,
        },
    )


if __name__ == "__main__":
    main()
