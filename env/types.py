from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal


BatterAction = Literal["defensive", "balanced", "aggressive"]
BowlerAction = Literal["yorker", "bouncer", "spin", "slow_ball"]
PitchCondition = Literal["green", "flat", "dusty"]
AgentPersonality = Literal["calm", "analytical", "aggressive"]


@dataclass
class ArenaState:
    ball_number: int
    runs: int
    wickets: int
    required_run_rate: float
    pitch_condition: PitchCondition
    batter_form: float
    bowler_fatigue: float
    belief_distribution_over_opponent_actions: Dict[BowlerAction, float]
    agent_personality: AgentPersonality
    opponent_history: List[BowlerAction] = field(default_factory=list)

    def model_dump(self):
        return asdict(self)


@dataclass
class StepResult:
    state: ArenaState
    reward: float
    done: bool
    info: Dict[str, float | int | str | bool]
