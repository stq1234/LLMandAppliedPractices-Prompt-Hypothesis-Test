"""Strict primary scoring and permissive JSON-extraction sensitivity scoring."""

from __future__ import annotations

import json
from typing import Any

from .prompts import load_prompt_spec


class DuplicateKeyError(ValueError):
    pass


class InvalidConstantError(ValueError):
    pass


def _reject_non_json_constant(value: str) -> None:
    raise InvalidConstantError(f"non-JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_object(text: str | None) -> tuple[bool, dict[str, Any] | None]:
    if not isinstance(text, str):
        return False, None
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_non_json_constant)
    except (json.JSONDecodeError, DuplicateKeyError, InvalidConstantError, TypeError):
        return False, None
    return (True, value) if isinstance(value, dict) else (False, None)


def _json_object_spans(text: str) -> list[str]:
    """Return balanced top-level brace spans without treating braces in strings as syntax."""
    spans: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and depth:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : i + 1])
                start = None
    return spans


def _loose_json_object(text: str | None) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    strict, value = _strict_json_object(text.strip())
    if strict:
        return value
    for candidate in _json_object_spans(text):
        valid, parsed = _strict_json_object(candidate)
        if valid:
            return parsed
    return None


def score_content(
    content: str | None,
    gold_decision: str,
    arm: str,
    finish_reason: str | None = None,
    brief_max_chars: int | None = None,
    relation_max_chars: int | None = None,
) -> dict[str, Any]:
    """Score only fixed fields; free-form rationale is never used to infer a label.

    Primary accuracy requires an exact JSON object and a valid fixed `decision`
    enum. A well-formed decision can count toward primary accuracy even when
    another field violates the output schema; those violations remain explicit
    in the separate format metrics.
    """
    if gold_decision not in {"ALLOW", "DENY"}:
        raise ValueError("gold_decision must be ALLOW or DENY")
    spec = load_prompt_spec()
    arm_spec = spec["arms"].get(arm)
    if arm_spec is None:
        raise ValueError(f"unknown arm: {arm}")
    expected_keys = set(arm_spec["expected_keys"])
    brief_limit = brief_max_chars if brief_max_chars is not None else int(spec["brief_max_chars"])
    relation_limit = relation_max_chars if relation_max_chars is not None else int(spec["decompose_max_relation_chars"])

    strict_ok, obj = _strict_json_object(content)
    decision = obj.get("decision") if obj else None
    valid_decision = isinstance(decision, str) and decision in {"ALLOW", "DENY"}
    primary_correct = bool(strict_ok and valid_decision and decision == gold_decision)
    semantic_wrong = bool(strict_ok and valid_decision and decision != gold_decision)
    brief = obj.get("brief") if obj else None
    brief_chars = len(brief) if isinstance(brief, str) else None
    brief_ok = isinstance(brief, str) and brief_chars <= brief_limit
    if arm == "decompose":
        relation = obj.get("relation") if obj else None
        relation_chars = len(relation) if isinstance(relation, str) else None
        relation_ok: bool | None = isinstance(relation, str) and relation_chars <= relation_limit
    else:
        relation = obj.get("relation") if obj else None
        relation_chars = len(relation) if isinstance(relation, str) else None
        relation_ok = None

    schema_valid = bool(
        strict_ok
        and obj is not None
        and set(obj) == expected_keys
        and valid_decision
        and brief_ok
        and (arm != "decompose" or relation_ok)
        and all(isinstance(obj.get(key), str) for key in expected_keys if key != "decision")
    )
    loose_obj = _loose_json_object(content)
    loose_decision = loose_obj.get("decision") if loose_obj else None
    loose_valid_decision = isinstance(loose_decision, str) and loose_decision in {"ALLOW", "DENY"}
    loose_correct = bool(loose_valid_decision and loose_decision == gold_decision)
    if not strict_ok or not valid_decision:
        failure_category = "format_or_missing_decision"
    elif decision != gold_decision:
        failure_category = "semantic_wrong"
    elif not schema_valid:
        failure_category = "correct_decision_with_format_violation"
    else:
        failure_category = "correct_and_schema_compliant"

    return {
        "primary_correct": primary_correct,
        "strict_json_ok": strict_ok,
        "schema_valid": schema_valid,
        "decision_field_present": bool(obj is not None and "decision" in obj),
        "decision_enum_valid": valid_decision,
        "decision": decision if valid_decision else None,
        "gold_decision": gold_decision,
        "semantic_wrong": semantic_wrong,
        "format_or_missing_decision": not (strict_ok and valid_decision),
        "format_violation": not schema_valid,
        "brief_chars": brief_chars,
        "brief_within_limit": brief_ok,
        "relation_chars": relation_chars,
        "relation_within_limit": relation_ok,
        "loose_decision": loose_decision if loose_valid_decision else None,
        "loose_correct": loose_correct,
        "loose_parse_used": bool(not strict_ok and loose_valid_decision),
        "failure_category": failure_category,
        "truncated": finish_reason in {"length", "max_tokens"},
    }

