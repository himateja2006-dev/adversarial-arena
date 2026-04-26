from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, Optional
from urllib import request

from dotenv import load_dotenv

from env.types import ArenaState, BatterAction


class LLMBeliefAgent:
    """OpenAI-compatible LLM policy with strict JSON parsing and fallback."""

    VALID_ACTIONS = {"defensive", "balanced", "aggressive"}

    def __init__(self) -> None:
        load_dotenv()
        random.seed(None)
        self.base_url = os.getenv("API_BASE_URL", "").strip().rstrip("/")
        model = os.getenv("MODEL_NAME", "").strip()
        hf_token = os.getenv("HF_TOKEN", "").strip()
        self.model = model
        self.hf_token = hf_token
        self.enabled = bool(self.base_url and model and hf_token)
        self.debug = os.getenv("LLM_DEBUG", "0").strip() == "1"
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.55").strip() or "0.55")
        self.last_decision_source = "llm_uninitialized"
        self._last_state: Optional[ArenaState] = None
        self._last_action: Optional[BatterAction] = None
        self._recent_actions: Deque[BatterAction] = deque(maxlen=6)
        self._q_values: Dict[BatterAction, float] = {"defensive": 0.0, "balanced": 0.0, "aggressive": 0.0}
        self._action_counts: Dict[BatterAction, int] = {"defensive": 0, "balanced": 0, "aggressive": 0}
        self._transition_counts: Dict[BatterAction, Counter] = defaultdict(Counter)
        self._alpha = 0.25

    def act(self, state: ArenaState) -> BatterAction:
        self._ingest_observation(state)
        if not self.enabled:
            return self._fallback("Missing API_BASE_URL/MODEL_NAME/HF_TOKEN")
        prompt = self._build_prompt(state)
        if self.debug:
            print(f"[LLM_DEBUG] request={self._debug_state_summary(state)}", file=sys.stderr)
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an RL policy. Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": 80,
            }
            req = request.Request(
                url=f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.hf_token}",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            if self.debug:
                compact = text.strip().replace("\n", " ")
                if len(compact) > 180:
                    compact = compact[:177] + "..."
                print(f"[LLM_DEBUG] raw_response={compact}", file=sys.stderr)
            parsed = self._strict_parse(text)
            if parsed in self.VALID_ACTIONS:
                self.last_decision_source = "llm_api"
                action = parsed  # type: ignore[assignment]
                self._register_action(action)
                return action  # type: ignore[return-value]
            return self._fallback("Parsed action not in valid set")
        except Exception as exc:
            return self._fallback(f"LLM call failed: {exc}")

    def _build_prompt(self, state: ArenaState) -> str:
        belief = state.belief_distribution_over_opponent_actions
        likely = max(belief, key=belief.get)
        likely_prob = round(float(belief[likely]) * 100, 1)
        payload: Dict[str, Any] = {
            "state": state.model_dump(),
            "belief_distribution": belief,
            "history": state.opponent_history,
            "personality": state.agent_personality,
            "belief_guidance": (
                f"Opponent most likely action: {likely} ({likely_prob}%). "
                "Choose the best counter action. Avoid repeating the same action unless optimal."
            ),
            "instruction": (
                "Choose one action from defensive|balanced|aggressive to maximize long-term reward "
                "under uncertainty. Output ONLY valid JSON: {\"action\":\"...\"}"
            ),
        }
        return json.dumps(payload, ensure_ascii=True)

    def _strict_parse(self, text: str) -> str:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found.")
        data = json.loads(text[start : end + 1])
        action = data.get("action", "")
        if not isinstance(action, str):
            raise ValueError("Invalid action type")
        return action.strip()

    def _fallback(self, reason: str) -> BatterAction:
        self.last_decision_source = "llm_internal_fallback"
        if self.debug:
            print(f"[LLM_DEBUG] fallback_reason={reason}", file=sys.stderr)
        action = self._adaptive_fallback_action()
        self._register_action(action)
        return action

    def _debug_state_summary(self, state: ArenaState) -> str:
        belief = state.belief_distribution_over_opponent_actions
        dominant = max(belief, key=belief.get)
        return (
            f"ball={state.ball_number}, runs={state.runs}, wickets={state.wickets}, "
            f"rrr={state.required_run_rate:.2f}, dominant_belief={dominant}:{belief[dominant]:.2f}, "
            f"history_len={len(state.opponent_history)}"
        )

    def _ingest_observation(self, state: ArenaState) -> None:
        if self._last_state is None or self._last_action is None:
            self._last_state = state
            return
        if len(state.opponent_history) == 0:
            self._last_state = state
            return
        observed_opp = state.opponent_history[-1]
        prev = self._last_state
        outcome_runs = max(0, state.runs - prev.runs)
        wicket = state.wickets > prev.wickets
        pseudo_reward = self._estimate_reward(self._last_action, observed_opp, outcome_runs, wicket, state)
        self._q_values[self._last_action] += self._alpha * (pseudo_reward - self._q_values[self._last_action])
        self._transition_counts[self._last_action][observed_opp] += 1
        self._last_state = state

    def _register_action(self, action: BatterAction) -> None:
        self._last_action = action
        self._recent_actions.append(action)
        self._action_counts[action] += 1

    def _adaptive_fallback_action(self) -> BatterAction:
        if self._last_state is None:
            return random.choice(tuple(self.VALID_ACTIONS))  # type: ignore[return-value]
        state = self._last_state
        predicted_belief = self._predict_opponent_distribution(state)
        dominant = max(predicted_belief, key=predicted_belief.get)
        dominant_prob = float(predicted_belief[dominant])
        # Under conflicting/uncertain belief, balanced is robust in this environment.
        if dominant_prob < 0.55 and state.required_run_rate <= 9.0 and state.wickets <= 6:
            return "balanced"
        total_actions = sum(self._action_counts.values()) + 1

        scores: Dict[BatterAction, float] = {}
        for action in ("defensive", "balanced", "aggressive"):
            exp_runs = self._expected_runs(action, predicted_belief, state)
            perf = min(1.0, max(0.0, exp_runs / 6.0))
            counter_map = {"yorker": "defensive", "bouncer": "balanced", "spin": "aggressive", "slow_ball": "balanced"}
            adaptation = 1.0 if action == counter_map[dominant] else 0.0
            belief_bonus = 0.15 if adaptation > 0.0 and dominant_prob >= 0.35 else 0.0
            repeats = self._recent_actions.count(action)
            repeat_penalty = min(0.2, max(0.0, (repeats - 2) * 0.05))
            exploration = (
                math.sqrt(math.log(total_actions + 1) / (1 + self._action_counts[action])) * 0.04 / math.sqrt(total_actions)
            )
            q_bonus = 0.35 * self._q_values[action]
            stability_bonus = 0.12 if action == "balanced" else 0.0
            score = 0.48 * perf + 0.34 * adaptation + belief_bonus + q_bonus + exploration + stability_bonus - repeat_penalty
            scores[action] = score

        best = max(scores, key=scores.get)
        if state.required_run_rate > 10.5 and state.wickets <= 4:
            best = "aggressive"
        if state.wickets >= 7:
            best = "defensive"
        return best  # type: ignore[return-value]

    def _predict_opponent_distribution(self, state: ArenaState) -> Dict[str, float]:
        belief = dict(state.belief_distribution_over_opponent_actions)
        if self._last_action is None:
            return belief
        counts = self._transition_counts[self._last_action]
        total = sum(counts.values())
        if total <= 0:
            return belief
        mixed: Dict[str, float] = {}
        for k in belief:
            empirical = counts[k] / total
            mixed[k] = 0.65 * empirical + 0.35 * float(belief[k])
        s = sum(mixed.values()) or 1.0
        return {k: v / s for k, v in mixed.items()}

    def _expected_runs(self, action: BatterAction, belief: Dict[str, float], state: ArenaState) -> float:
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
        exp = 0.0
        for opp, p in belief.items():
            mean, risk = base[(action, opp)]
            if state.pitch_condition == "green" and opp in ("yorker", "bouncer"):
                mean -= 0.25
                risk += 0.015
            elif state.pitch_condition == "dusty" and opp == "spin":
                mean -= 0.35
                risk += 0.02
            mean += (float(state.batter_form) - 0.5)
            mean -= float(state.bowler_fatigue) * 0.4
            mean = max(0.0, mean)
            exp += p * (mean - 2.5 * risk)
        return max(0.0, exp)

    def _estimate_reward(
        self,
        action: BatterAction,
        opponent_action: str,
        outcome_runs: int,
        wicket: bool,
        state: ArenaState,
    ) -> float:
        counter_map = {"yorker": "defensive", "bouncer": "balanced", "spin": "aggressive", "slow_ball": "balanced"}
        perf = min(1.0, max(0.0, (outcome_runs / 6.0) - (0.4 if wicket else 0.0)))
        adaptation = 1.0 if action == counter_map.get(opponent_action, "balanced") else 0.0
        belief = state.belief_distribution_over_opponent_actions
        dominant = max(belief, key=belief.get)
        bonus = 0.15 if action == counter_map.get(dominant, "balanced") and belief[dominant] >= 0.35 else 0.0
        reward = 0.5 * perf + 0.4 * adaptation + bonus
        return max(0.0, min(1.0, reward))
