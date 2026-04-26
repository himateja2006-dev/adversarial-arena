from __future__ import annotations

import json

from agents.trained_agent import TrainedBeliefAgent
from env.types import ArenaState


def make_state(bouncer_prob: float) -> ArenaState:
    rem = (1.0 - bouncer_prob) / 3.0
    belief = {
        "yorker": rem,
        "bouncer": bouncer_prob,
        "spin": rem,
        "slow_ball": rem,
    }
    return ArenaState(
        ball_number=12,
        runs=20,
        wickets=2,
        required_run_rate=8.0,
        pitch_condition="flat",
        batter_form=0.55,
        bowler_fatigue=0.3,
        belief_distribution_over_opponent_actions=belief,  # type: ignore[arg-type]
        agent_personality="analytical",
        opponent_history=["bouncer", "bouncer", "spin", "bouncer"],
    )


def main() -> None:
    actions = []
    for p in [0.2, 0.4, 0.6, 0.75]:
        agent = TrainedBeliefAgent()
        s = make_state(p)
        actions.append({"bouncer_belief": p, "chosen_action": agent.act(s)})
    print(json.dumps({"behavior_check": actions}, indent=2))


if __name__ == "__main__":
    main()
