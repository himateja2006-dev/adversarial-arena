from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

from env.types import ArenaState, BatterAction

VALID_ACTIONS = ("defensive", "balanced", "aggressive")


class TrainedAgent:
    """Inference-time policy that loads a fine-tuned local model directory."""

    def __init__(self, model_path: str, sampling: bool = False) -> None:
        self.model_path = Path(model_path)
        self.sampling = sampling
        self.available = False
        self.model = None
        self.tokenizer = None
        random.seed(None)
        print(f"[TRAINED_AGENT] Loading trained model from: {self.model_path}", file=sys.stderr)
        if self.model_path.exists():
            print(f"[TRAINED_AGENT] Model files: {sorted(os.listdir(self.model_path))}", file=sys.stderr)
        else:
            print("[TRAINED_AGENT] Model path does not exist.", file=sys.stderr)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            adapter_cfg = self.model_path / "adapter_config.json"
            if adapter_cfg.exists():
                # LoRA/adapter model path.
                from peft import PeftModel

                cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                base_model_name = cfg.get("base_model_name_or_path", "").strip()
                if not base_model_name:
                    raise RuntimeError("adapter_config.json missing base_model_name_or_path")
                # Adapter dirs may contain tokenizer metadata incompatible with local env.
                # Always use base-model tokenizer for robust loading.
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

                adapter_weights = self.model_path / "adapter_model.safetensors"
                adapter_weights_bin = self.model_path / "adapter_model.bin"
                if not adapter_weights.exists() and not adapter_weights_bin.exists():
                    raise RuntimeError(
                        "Adapter weights missing: expected adapter_model.safetensors or adapter_model.bin in model_path"
                    )
                base_model = self._load_base_model_with_retry(AutoModelForCausalLM, base_model_name)
                self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            else:
                # Full model path.
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), use_fast=False)
                self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path))
            self.available = True
        except Exception as exc:
            print(f"[TRAINED_AGENT] failed_to_load_model path={self.model_path} err={exc}", file=sys.stderr)
        print(f"[TRAINED_AGENT] Model loaded successfully: {self.available}", file=sys.stderr)

    def _load_base_model_with_retry(self, auto_model_cls, base_model_name: str):
        """
        Load base model robustly:
        - first try adapter-declared base model
        - if bitsandbytes/4bit dependency error appears, retry with non-bnb variant
        """
        try:
            return auto_model_cls.from_pretrained(base_model_name)
        except Exception as exc:
            err_txt = str(exc).lower()
            needs_retry = ("bitsandbytes" in err_txt) or ("4bit" in base_model_name.lower()) or ("bnb" in base_model_name.lower())
            if not needs_retry:
                raise
            fallback = self._dequantized_base_name(base_model_name)
            print(
                f"[TRAINED_AGENT] retrying base model load without bnb. original={base_model_name} fallback={fallback}",
                file=sys.stderr,
            )
            return auto_model_cls.from_pretrained(fallback)

    def _dequantized_base_name(self, name: str) -> str:
        n = name.strip()
        # Common Unsloth bnb naming used by this project.
        if "mistral-7b-instruct-v0.2" in n.lower():
            return "mistralai/Mistral-7B-Instruct-v0.2"
        # Generic cleanup for bnb/4bit suffixes.
        n = n.replace("-bnb-4bit", "").replace("-4bit", "").replace("bnb-4bit", "")
        return n

    def act(self, state: ArenaState) -> BatterAction:
        if not self.available or self.model is None or self.tokenizer is None:
            raise RuntimeError("Trained model not loaded — aborting inference")
        prompt = self._format_prompt(state)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=self.sampling,
                temperature=0.7 if self.sampling else 1.0,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            action = self._parse_action(text)
            if action in VALID_ACTIONS:
                return action  # type: ignore[return-value]
        except Exception as exc:
            print(f"[TRAINED_AGENT] generate_error err={exc}", file=sys.stderr)
        raise RuntimeError("Trained model failed to generate valid action")

    def _format_prompt(self, state: ArenaState) -> str:
        belief = state.belief_distribution_over_opponent_actions
        likely = max(belief, key=belief.get)
        likely_prob = round(100.0 * float(belief[likely]), 1)
        state_txt = json.dumps(
            {
                "ball_number": state.ball_number,
                "runs": state.runs,
                "wickets": state.wickets,
                "required_run_rate": state.required_run_rate,
                "pitch_condition": state.pitch_condition,
                "batter_form": state.batter_form,
                "bowler_fatigue": state.bowler_fatigue,
            },
            ensure_ascii=True,
        )
        belief_txt = json.dumps(belief, ensure_ascii=True)
        hist_txt = json.dumps(state.opponent_history, ensure_ascii=True)
        return (
            f"State: {state_txt}\n"
            f"Belief: {belief_txt}\n"
            f"History: {hist_txt}\n"
            f"Opponent most likely action: {likely} ({likely_prob}%).\n"
            "Choose the BEST counter strategy.\n"
            "Avoid repeating the same action unless optimal.\n"
            "Maximize long-term reward.\n"
            "Output strict JSON only."
        )

    def _parse_action(self, text: str) -> Optional[str]:
        # Parse first JSON object if present.
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                action = data.get("action")
                if isinstance(action, str):
                    return action.strip()
            except Exception:
                pass
        # lightweight fallback parse
        for a in VALID_ACTIONS:
            if a in text:
                return a
        return None
