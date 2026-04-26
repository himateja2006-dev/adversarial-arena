from __future__ import annotations

import random
from typing import List

from env.types import ArenaState, BatterAction


class AdaptiveFallbackAgent:
    """
    Belief-driven heuristic policy used when trained model/LLM are unavailable.
    It mimics an RL-style policy by optimizing counter-actions under uncertainty,
    while adapting to chase pressure and wicket risk.
    """

    def __init__(self) -> None:
        random.seed(None)
        self.last_actions: List[BatterAction] = []
        self.counter_map = {
            "yorker": "defensive",
            "bouncer": "balanced",
            "spin": "aggressive",
            "slow_ball": "balanced",
        }

    def act(self, state: ArenaState) -> BatterAction:
        belief = state.belief_distribution_over_opponent_actions
        dominant = max(belief, key=belief.get)
        dominant_prob = float(belief[dominant])

        # Expected value under belief; subtract approximate wicket risk.
        expected_runs = {
            "defensive": (
                0.8 * belief["yorker"]
                + 0.7 * belief["bouncer"]
                + 1.0 * belief["spin"]
                + 1.1 * belief["slow_ball"]
            ),
            "balanced": (
                1.2 * belief["yorker"]
                + 1.4 * belief["bouncer"]
                + 1.6 * belief["spin"]
                + 1.8 * belief["slow_ball"]
            ),
            "aggressive": (
                1.6 * belief["yorker"]
                + 2.2 * belief["bouncer"]
                + 2.4 * belief["spin"]
                + 2.0 * belief["slow_ball"]
            ),
        }
        risk = {
            "defensive": 0.04,
            "balanced": 0.08,
            "aggressive": 0.16,
        }
        utility = {
            a: expected_runs[a] - risk[a] * (1.0 + state.wickets / 7.0)
            for a in ["defensive", "balanced", "aggressive"]
        }

        # Belief-action coupling boost.
        utility[self.counter_map[dominant]] += 0.22 if dominant_prob >= 0.35 else 0.08

        action: BatterAction = max(utility, key=utility.get)  # type: ignore[assignment]

        # Match pressure adjustments.
        if state.required_run_rate > 10.5 and state.wickets <= 4:
            action = "aggressive"
        if state.wickets >= 7:
            action = "defensive"

        # Avoid repetitive collapse.
        if len(self.last_actions) >= 3 and self.last_actions[-1] == self.last_actions[-2] == self.last_actions[-3] == action:
            alternatives = [x for x in ["defensive", "balanced", "aggressive"] if x != action]
            action = random.choice(alternatives)  # type: ignore[assignment]

        self.last_actions.append(action)
        if len(self.last_actions) > 6:
            self.last_actions = self.last_actions[-6:]
        return action
