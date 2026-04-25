from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import time

from dotenv import load_dotenv

from agents.baseline import BaselineAgent, RandomAgent
from agents.llm_agent import LLMBeliefAgent
from agents.trained_agent import TrainedBeliefAgent
from env.environement import AdversarialArenaEnv
from graders.grader import grade_episode
from tasks.scenarios import TASKS


def run_episode(
    task_name: str,
    agent_name: str,
    episode_idx: int = 1,
    emit_logs: bool = True,
    run_seed: int | None = None,
) -> float:
    task = TASKS[task_name]
    seed = run_seed if run_seed is not None else int(time() * 1000) % 10_000_000
    env = AdversarialArenaEnv(opponent_policy=task.opponent_policy_factory(seed=seed + 99), seed=seed + 11)

    if agent_name == "baseline":
        agent = BaselineAgent(seed=episode_idx)
    elif agent_name == "random":
        agent = RandomAgent()
    elif agent_name == "trained":
        agent = TrainedBeliefAgent()
    elif agent_name == "llm":
        agent = LLMBeliefAgent()
    elif agent_name == "auto":
        agent = TrainedBeliefAgent() if Path("trained_model/policy.json").exists() else LLMBeliefAgent()
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

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
        commentary = (
            f"Opponent={sr.info['opponent_action']}, runs={int(sr.info['outcome_runs'])}, "
            f"wicket={bool(sr.info['wicket'])}"
        )
        if emit_logs:
            print("")
            print("[STEP]")
            print(f"state={json.dumps(short_state, ensure_ascii=True)}")
            print(f"belief={json.dumps(sr.state.belief_distribution_over_opponent_actions, ensure_ascii=True)}")
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
    parser.add_argument("--agent", type=str, default="auto", choices=["baseline", "random", "trained", "llm", "auto"])
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_episode(task_name=args.task, agent_name=args.agent, episode_idx=args.episode, run_seed=args.seed)


if __name__ == "__main__":
    main()
