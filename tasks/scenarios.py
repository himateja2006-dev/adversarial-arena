from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict

from env.environement import AdversarialArenaEnv
from env.types import BowlerAction


@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    opponent_policy_factory: Callable[[int], Callable[[AdversarialArenaEnv], BowlerAction]]


def static_opponent_factory(seed: int) -> Callable[[AdversarialArenaEnv], BowlerAction]:
    rng = random.Random(seed)
    fixed = rng.choice(["yorker", "spin"])

    def policy(_: AdversarialArenaEnv) -> BowlerAction:
        return fixed  # type: ignore[return-value]

    return policy


def adaptive_opponent_factory(seed: int) -> Callable[[AdversarialArenaEnv], BowlerAction]:
    rng = random.Random(seed)

    def policy(env: AdversarialArenaEnv) -> BowlerAction:
        if not env._last_actions:
            return rng.choice(env.bowler_actions)  # type: ignore[return-value]
        last_batter = env._last_actions[-1]
        if last_batter == "aggressive":
            return "yorker"
        if last_batter == "defensive":
            return "spin"
        return "bouncer"

    return policy


def deceptive_opponent_factory(seed: int) -> Callable[[AdversarialArenaEnv], BowlerAction]:
    rng = random.Random(seed)

    def policy(env: AdversarialArenaEnv) -> BowlerAction:
        history = list(env._history)
        if len(history) < 5:
            return "slow_ball"
        # Deception: establish a pattern then break it.
        if history[-1] == "slow_ball" and rng.random() < 0.65:
            return "yorker"
        if rng.random() < 0.5:
            return "slow_ball"
        return rng.choice(env.bowler_actions)  # type: ignore[return-value]

    return policy


TASKS: Dict[str, TaskConfig] = {
    "static_opponent": TaskConfig(
        name="static_opponent",
        description="Opponent mostly repeats one bowling style.",
        opponent_policy_factory=static_opponent_factory,
    ),
    "adaptive_opponent": TaskConfig(
        name="adaptive_opponent",
        description="Opponent reacts to recent batter action.",
        opponent_policy_factory=adaptive_opponent_factory,
    ),
    "deceptive_opponent": TaskConfig(
        name="deceptive_opponent",
        description="Opponent fakes a stable pattern then shifts.",
        opponent_policy_factory=deceptive_opponent_factory,
    ),
}
