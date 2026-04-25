from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

from env.types import ArenaState, BatterAction


class TrainedBeliefAgent:
    """
    Lightweight trained-policy reader.
    The policy file stores per-opponent-action best counters learned from trajectories.
    """

    def __init__(self, policy_path: str = "training/artifacts/trained_policy.json") -> None:
        self.policy_path = Path(policy_path)
        self.model_policy_path = Path("trained_model/policy.json")
        self.default_counter: Dict[str, BatterAction] = {
            "yorker": "defensive",
            "bouncer": "balanced",
            "spin": "aggressive",
            "slow_ball": "balanced",
        }
        self.last_actions = []
        random.seed(None)
        source = self.model_policy_path if self.model_policy_path.exists() else self.policy_path
        if source.exists():
            with source.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            learned = payload.get("counter_map", {})
            self.default_counter.update(learned)

    def act(self, state: ArenaState) -> BatterAction:
        belief = state.belief_distribution_over_opponent_actions
        likely = max(belief, key=belief.get)
        likely_prob = belief.get(likely, 0.0)
        action = self.default_counter.get(likely, "balanced")

        # Stronger adaptation under high-confidence belief.
        if likely == "bouncer" and likely_prob > 0.45:
            action = "balanced"
        elif likely == "spin" and likely_prob > 0.40:
            action = "aggressive"
        elif likely == "yorker" and likely_prob > 0.40:
            action = "defensive"
        elif likely_prob >= 0.30:
            action = self.default_counter.get(likely, action)
        else:
            action = "balanced"

        # Penalize repetitive actions by injecting controlled diversity.
        if len(self.last_actions) >= 3 and self.last_actions[-1] == self.last_actions[-2] == self.last_actions[-3] == action:
            action = random.choice([a for a in ["defensive", "balanced", "aggressive"] if a != action])

        # Dynamic pace adjustment.
        if state.required_run_rate > 11.0 and state.wickets < 6:
            action = "aggressive"
        if state.wickets >= 7:
            action = "defensive"
        self.last_actions.append(action)
        if len(self.last_actions) > 5:
            self.last_actions = self.last_actions[-5:]
        return action
