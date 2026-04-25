from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List

from env.types import StepResult


@dataclass
class EpisodeMetrics:
    performance: float
    adaptation: float
    consistency: float
    score: float


def _consistency_from_rewards(rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    instability = sum(abs(rewards[i] - rewards[i - 1]) for i in range(1, len(rewards))) / max(1, len(rewards) - 1)
    return max(0.0, min(1.0, 1.0 - instability))


def grade_episode(steps: List[StepResult]) -> EpisodeMetrics:
    if not steps:
        return EpisodeMetrics(0.0, 0.0, 0.0, 0.0)
    rewards = [s.reward for s in steps]
    infos: List[Dict] = [s.info for s in steps]
    performance = max(0.0, min(1.0, mean(rewards)))
    adaptation = mean([1.0 if x.get("outcome_runs", 0) >= 2 else 0.0 for x in infos])
    consistency = _consistency_from_rewards(rewards)
    score = max(0.0, min(1.0, 0.5 * performance + 0.3 * adaptation + 0.2 * consistency))
    return EpisodeMetrics(performance, adaptation, consistency, score)
