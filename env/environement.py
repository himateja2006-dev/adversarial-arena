from __future__ import annotations

import random
from collections import Counter, deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from env.types import ArenaState, BatterAction, BowlerAction, StepResult


class AdversarialArenaEnv:
    """Gym-style OpenEnv environment for adversarial belief modeling."""

    batter_actions: Tuple[BatterAction, ...] = ("defensive", "balanced", "aggressive")
    bowler_actions: Tuple[BowlerAction, ...] = ("yorker", "bouncer", "spin", "slow_ball")

    def __init__(
        self,
        opponent_policy: Callable[["AdversarialArenaEnv"], BowlerAction],
        max_balls: int = 30,
        target_runs: int = 45,
        history_len: int = 8,
        seed: int = 42,
    ) -> None:
        self.opponent_policy = opponent_policy
        self.max_balls = max_balls
        self.target_runs = target_runs
        self.history_len = history_len
        self.rng = random.Random(seed)

        self.ball_number = 0
        self.runs = 0
        self.wickets = 0
        self.pitch_condition = "flat"
        self.batter_form = 0.5
        self.bowler_fatigue = 0.1
        self.agent_personality = "analytical"
        self._history: Deque[BowlerAction] = deque(maxlen=history_len)
        self._belief_counts: Counter = Counter({a: 1 for a in self.bowler_actions})
        self._last_actions: Deque[BatterAction] = deque(maxlen=5)

    def reset(self, seed: Optional[int] = None) -> ArenaState:
        if seed is not None:
            self.rng.seed(seed)
        self.ball_number = 0
        self.runs = 0
        self.wickets = 0
        self.pitch_condition = self.rng.choice(["green", "flat", "dusty"])
        self.batter_form = self.rng.uniform(0.35, 0.65)
        self.bowler_fatigue = self.rng.uniform(0.05, 0.2)
        self.agent_personality = self.rng.choice(["calm", "analytical", "aggressive"])
        self._history.clear()
        self._belief_counts = Counter({a: 1 for a in self.bowler_actions})
        self._last_actions.clear()
        return self.state()

    def state(self) -> ArenaState:
        balls_left = max(1, self.max_balls - self.ball_number)
        runs_left = max(0, self.target_runs - self.runs)
        required_run_rate = float(runs_left * 6.0 / balls_left)
        total = sum(self._belief_counts.values())
        beliefs: Dict[BowlerAction, float] = {
            action: self._belief_counts[action] / total for action in self.bowler_actions
        }
        return ArenaState(
            ball_number=self.ball_number,
            runs=self.runs,
            wickets=self.wickets,
            required_run_rate=required_run_rate,
            pitch_condition=self.pitch_condition,  # type: ignore[arg-type]
            batter_form=round(self.batter_form, 4),
            bowler_fatigue=round(self.bowler_fatigue, 4),
            belief_distribution_over_opponent_actions=beliefs,
            opponent_history=list(self._history),
            agent_personality=self.agent_personality,  # type: ignore[arg-type]
        )

    def step(self, action: BatterAction) -> StepResult:
        if action not in self.batter_actions:
            raise ValueError(f"Invalid action: {action}")
        if self.ball_number >= self.max_balls or self.wickets >= 10:
            return StepResult(state=self.state(), reward=0.0, done=True, info={"terminal": True})

        current_belief = self.state().belief_distribution_over_opponent_actions
        dominant_belief_action = max(current_belief, key=current_belief.get)
        dominant_belief_prob = current_belief[dominant_belief_action]
        opponent_action = self.opponent_policy(self)
        outcome_runs, wicket = self._sample_outcome(action, opponent_action)

        self.ball_number += 1
        self.runs += outcome_runs
        self.wickets += int(wicket)
        self.bowler_fatigue = min(1.0, self.bowler_fatigue + 0.02)
        self.batter_form = min(1.0, max(0.0, self.batter_form + (0.01 if outcome_runs > 0 else -0.02)))

        self._history.append(opponent_action)
        self._belief_counts[opponent_action] += 1
        self._last_actions.append(action)

        reward = self._compute_reward(
            action,
            opponent_action,
            outcome_runs,
            wicket,
            dominant_belief_action=dominant_belief_action,  # type: ignore[arg-type]
            dominant_belief_prob=dominant_belief_prob,
        )
        done = self.ball_number >= self.max_balls or self.wickets >= 10 or self.runs >= self.target_runs
        info = {
            "opponent_action": opponent_action,
            "outcome_runs": outcome_runs,
            "wicket": wicket,
            "current_run_rate": self.runs * 6.0 / max(1, self.ball_number),
        }
        return StepResult(state=self.state(), reward=reward, done=done, info=info)

    def _sample_outcome(self, batter_action: BatterAction, bowler_action: BowlerAction) -> Tuple[int, bool]:
        base = {
            ("defensive", "yorker"): (0.8, 0.05),
            ("defensive", "bouncer"): (0.7, 0.04),
            ("defensive", "spin"): (1.0, 0.03),
            ("defensive", "slow_ball"): (1.1, 0.03),
            ("balanced", "yorker"): (1.2, 0.06),
            ("balanced", "bouncer"): (1.4, 0.06),
            ("balanced", "spin"): (1.6, 0.05),
            ("balanced", "slow_ball"): (1.8, 0.05),
            ("aggressive", "yorker"): (1.6, 0.12),
            ("aggressive", "bouncer"): (2.2, 0.13),
            ("aggressive", "spin"): (2.4, 0.11),
            ("aggressive", "slow_ball"): (2.0, 0.10),
        }
        mean_runs, wicket_p = base[(batter_action, bowler_action)]

        if self.pitch_condition == "green" and bowler_action in ("yorker", "bouncer"):
            mean_runs -= 0.25
            wicket_p += 0.015
        elif self.pitch_condition == "dusty" and bowler_action == "spin":
            mean_runs -= 0.35
            wicket_p += 0.02

        mean_runs += (self.batter_form - 0.5) * 1.0
        mean_runs -= self.bowler_fatigue * 0.4
        mean_runs = max(0.0, mean_runs)

        wicket = self.rng.random() < min(0.35, max(0.01, wicket_p))
        if wicket:
            return 0, True
        runs_bucket = [0, 1, 2, 3, 4, 6]
        probs = self._runs_distribution(mean_runs)
        sampled = self.rng.choices(runs_bucket, weights=probs, k=1)[0]
        return sampled, False

    def _runs_distribution(self, mean_runs: float) -> List[float]:
        # Smooth mapping from target expected runs to cricket-like outcomes.
        p0 = max(0.05, 0.35 - 0.1 * mean_runs)
        p6 = min(0.18, 0.02 + 0.05 * mean_runs)
        p4 = min(0.28, 0.10 + 0.06 * mean_runs)
        p1 = max(0.10, 0.25 - 0.03 * mean_runs)
        p2 = max(0.08, 0.20 - 0.02 * mean_runs)
        p3 = max(0.02, 1.0 - (p0 + p1 + p2 + p4 + p6))
        raw = [p0, p1, p2, p3, p4, p6]
        total = sum(raw)
        return [x / total for x in raw]

    def _compute_reward(
        self,
        batter_action: BatterAction,
        opponent_action: BowlerAction,
        outcome_runs: int,
        wicket: bool,
        dominant_belief_action: BowlerAction,
        dominant_belief_prob: float,
    ) -> float:
        perf = min(1.0, max(0.0, (outcome_runs / 6.0) - (0.4 if wicket else 0.0)))

        counter_map = {
            "yorker": "defensive",
            "bouncer": "balanced",
            "spin": "aggressive",
            "slow_ball": "balanced",
        }
        adaptation = 1.0 if batter_action == counter_map[opponent_action] else 0.0
        belief_counter_bonus = 0.15 if batter_action == counter_map[dominant_belief_action] and dominant_belief_prob >= 0.35 else 0.0

        repeats = self._last_actions.count(batter_action)
        diversity_penalty = min(0.25, max(0.0, (repeats - 3) * 0.05))

        belief = self.state().belief_distribution_over_opponent_actions[opponent_action]
        exploitability_penalty = max(0.0, belief - 0.65) * 0.3

        reward = (
            0.5 * perf
            + 0.4 * adaptation
            + belief_counter_bonus
            - diversity_penalty
            - exploitability_penalty
        )
        return max(0.0, min(1.0, reward))
