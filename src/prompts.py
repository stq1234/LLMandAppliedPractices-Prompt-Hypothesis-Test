"""Prompt construction with guardrails against gold leakage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "arms.json"


def load_prompt_spec(path: Path = PROMPT_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    decompose = spec["arms"]["decompose"]
    control = spec["arms"]["length_control"]
    decompose_len = len(decompose["extra_instruction"] + "\n" + decompose["instruction"])
    control_len = len(control["extra_instruction"] + "\n" + control["instruction"])
    if decompose_len != control_len:
        raise ValueError(
            "complete decompose and length_control user prompts must have equal Unicode character length"
        )
    return spec


def build_messages(record: dict[str, Any], arm: str, spec: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Build messages from only the model-visible dataset fields."""
    spec = spec or load_prompt_spec()
    if arm not in spec["arms"]:
        raise ValueError(f"unknown arm: {arm}")
    required = ("scenario", "rules_text", "prohibitions_text", "facts_text")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"dataset record lacks model-visible fields: {missing}")

    user = spec["base_user"].format(**{key: record[key] for key in required})
    arm_spec = spec["arms"][arm]
    if arm != "direct":
        user += arm_spec["extra_instruction"] + "\n"
    user += arm_spec["instruction"]
    return [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": user},
    ]

