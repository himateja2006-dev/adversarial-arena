from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from agents.baseline import BaselineAgent, RandomAgent
from env.environement import AdversarialArenaEnv
from tasks.scenarios import TASKS


def collect_trajectories(min_steps: int = 300, seed_offset: int = 1000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    episode = 0
    random.seed(None)
    while len(rows) < min_steps:
        task = TASKS[list(TASKS.keys())[episode % len(TASKS)]]
        env = AdversarialArenaEnv(opponent_policy=task.opponent_policy_factory(seed=seed_offset + episode), seed=seed_offset + episode)
        agent = RandomAgent() if episode % 2 == 0 else BaselineAgent(seed=episode + 123)
        state = env.reset(seed=seed_offset + episode)
        done = False
        while not done:
            action = agent.act(state)
            # Inject mild exploration for action diversity.
            if random.random() < 0.15:
                action = random.choice(["defensive", "balanced", "aggressive"])
            sr = env.step(action)
            rows.append(
                {
                    "task": task.name,
                    "episode": episode,
                    "step": sr.state.ball_number,
                    "state": {
                        "ball_number": sr.state.ball_number,
                        "runs": sr.state.runs,
                        "wickets": sr.state.wickets,
                        "required_run_rate": sr.state.required_run_rate,
                        "pitch_condition": sr.state.pitch_condition,
                        "batter_form": sr.state.batter_form,
                        "bowler_fatigue": sr.state.bowler_fatigue,
                        "agent_personality": sr.state.agent_personality,
                    },
                    "belief": sr.state.belief_distribution_over_opponent_actions,
                    "history": sr.state.opponent_history,
                    "action": action,
                    "reward": sr.reward,
                    "opponent_action": sr.info["opponent_action"],
                }
            )
            state = sr.state
            done = sr.done
        episode += 1
    return rows


def main() -> None:
    data = collect_trajectories(min_steps=320)
    out = Path("data/trajectories.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"saved_steps={len(data)} path={out}")


if __name__ == "__main__":
    main()
