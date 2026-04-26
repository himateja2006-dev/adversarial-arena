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
        likely_prob = float(belief.get(likely, 0.0))

        # Explicit belief -> action link: dominant belief is primary decision signal.
        action = self.default_counter.get(likely, "balanced")

        # If uncertainty is high, play robustly with balanced strategy.
        if likely_prob < 0.32:
            action = "balanced"

        # Dynamic pressure control with wicket-awareness.
        if state.required_run_rate > 11.0 and state.wickets <= 4 and likely != "yorker":
            action = "aggressive"
        if state.wickets >= 7:
            action = "defensive"

        # Penalize repetition to preserve adaptation signal.
        if len(self.last_actions) >= 3 and self.last_actions[-1] == self.last_actions[-2] == self.last_actions[-3] == action:
            ordered = sorted(belief.items(), key=lambda x: x[1], reverse=True)
            second = ordered[1][0] if len(ordered) > 1 else likely
            alt = self.default_counter.get(second, "balanced")
            action = alt if alt != action else random.choice([a for a in ["defensive", "balanced", "aggressive"] if a != action])

        self.last_actions.append(action)
        if len(self.last_actions) > 5:
            self.last_actions = self.last_actions[-5:]
        return action
