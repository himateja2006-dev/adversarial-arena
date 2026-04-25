from __future__ import annotations

import random

from env.types import ArenaState, BatterAction


class BaselineAgent:
    """Rule-based baseline that ignores belief uncertainty."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def act(self, state: ArenaState) -> BatterAction:
        if state.required_run_rate > 10.0:
            return "aggressive"
        if state.wickets >= 6:
            return "defensive"
        # Weak policy: mostly balanced, occasional random move.
        return self.rng.choice(["balanced", "balanced", "defensive", "aggressive"])


class RandomAgent:
    """Pure random baseline for robust comparison."""

    def __init__(self) -> None:
        random.seed(None)

    def act(self, state: ArenaState) -> BatterAction:
        _ = state
        return random.choice(["defensive", "balanced", "aggressive"])
