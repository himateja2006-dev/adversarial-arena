from __future__ import annotations

import json
from statistics import mean

from agents.baseline import BaselineAgent, RandomAgent
from agents.llm_agent import LLMBeliefAgent
from agents.trained_agent import TrainedBeliefAgent
from env.environement import AdversarialArenaEnv
from graders.grader import grade_episode
from tasks.scenarios import TASKS


def evaluate(agent_name: str, episodes: int = 15) -> float:
    scores = []
    for ep in range(episodes):
        task = TASKS[list(TASKS.keys())[ep % 3]]
        env = AdversarialArenaEnv(opponent_policy=task.opponent_policy_factory(seed=900 + ep), seed=300 + ep)
        if agent_name == "baseline":
            agent = BaselineAgent(seed=ep)
        elif agent_name == "random":
            agent = RandomAgent()
        elif agent_name == "trained":
            agent = TrainedBeliefAgent()
        elif agent_name == "llm":
            agent = LLMBeliefAgent()
        else:
            raise ValueError(agent_name)

        state = env.reset(seed=ep)
        steps = []
        done = False
        while not done:
            sr = env.step(agent.act(state))
            steps.append(sr)
            state = sr.state
            done = sr.done
        scores.append(grade_episode(steps).score)
    return float(mean(scores))


def main() -> None:
    report = {
        "baseline_rule": evaluate("baseline"),
        "baseline_random": evaluate("random"),
        "trained": evaluate("trained"),
        "llm_fallback_or_live": evaluate("llm"),
    }
    report["improvement_over_random_baseline"] = report["trained"] - report["baseline_random"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
