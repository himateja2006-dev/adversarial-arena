from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict
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

    def act(self, state: ArenaState) -> BatterAction:
        if not self.enabled:
            return self._fallback("Missing API_BASE_URL/MODEL_NAME/HF_TOKEN")
        prompt = self._build_prompt(state)
        if self.debug:
            print(f"[LLM_DEBUG] prompt={prompt}", file=sys.stderr)
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an RL policy. Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
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
                print(f"[LLM_DEBUG] raw_response={text}", file=sys.stderr)
            parsed = self._strict_parse(text)
            if parsed in self.VALID_ACTIONS:
                return parsed  # type: ignore[return-value]
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
        if self.debug:
            print(f"[LLM_DEBUG] fallback_reason={reason}", file=sys.stderr)
        return random.choice(tuple(self.VALID_ACTIONS))  # no deterministic fallback
