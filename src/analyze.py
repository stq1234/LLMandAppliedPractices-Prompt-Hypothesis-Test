"""Score logged outputs and produce family-clustered analyses and cost records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .prompts import load_prompt_spec
from .runner import DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_LOG, make_tasks, read_jsonl
from .scoring import score_content


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "analysis"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _when(row: dict[str, Any]) -> str:
    return str(row.get("received_at_utc") or "")


def select_successes(attempt_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        if row.get("task_id"):
            grouped[str(row["task_id"])].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for task_id, rows in grouped.items():
        successes = [row for row in rows if row.get("status") == "ok"]
        if successes:
            selected[task_id] = max(successes, key=lambda row: (int(row.get("attempt_no", 0)), _when(row)))
    return selected, grouped


def _usage_values(usage: Any) -> dict[str, int | None]:
    usage = usage if isinstance(usage, dict) else {}
    ptd = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    ctd = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
        "total_tokens": usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None,
        "cached_tokens": ptd.get("cached_tokens") if isinstance(ptd.get("cached_tokens"), int) else None,
        "reasoning_tokens": ctd.get("reasoning_tokens") if isinstance(ctd.get("reasoning_tokens"), int) else None,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def cluster_bootstrap_ci(values: Iterable[float], repetitions: int, seed: int) -> tuple[float | None, float | None, float | None]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return None, None, None
    point = float(array.mean())
    if array.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, array.size, size=(repetitions, array.size))
    means = array[sampled].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return point, float(low), float(high)


def _read_probe_costs(probe_dir: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    probes: list[dict[str, Any]] = []
    if probe_dir.exists():
        for path in sorted(probe_dir.glob("*.json")):
            if path.name == "models.json":
                continue
            try:
                outer = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(outer, dict) or "response" not in outer:
                continue
            response = outer.get("response") if isinstance(outer.get("response"), dict) else {}
            probes.append({
                "source": path.name,
                "status": outer.get("status"),
                "actual_model": response.get("model"),
                "elapsed_seconds": outer.get("elapsed_seconds"),
                "usage": response.get("usage"),
                "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason") if isinstance(response.get("choices"), list) and response.get("choices") else None,
            })
    if probes:
        return probes, "archived protocol-check responses"
    # Fallback is the manually reconciled inventory from the three saved probes.
    fallback = config.get("api", {}).get("known_probe_usage", [])
    if fallback:
        return list(fallback), "config fallback; archive original probe JSON to replace this summary"
    return [{"source": "unitemized-probe-reserve", "status": "known", "usage": {"total_tokens": int(config["api"].get("known_probe_tokens", 0))}}], "config aggregate fallback"


def _usage_sum(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    sums = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    seen = Counter()
    models = Counter()
    for row in rows:
        usage = _usage_values(row.get("usage"))
        for key, value in usage.items():
            if value is not None:
                sums[key] += value
                seen[key] += 1
        model = row.get("actual_model") or row.get("model_id") or row.get("source") or "unknown"
        models[str(model)] += 1
    sums["records_with_usage"] = seen["total_tokens"]
    sums["usage_record_count"] = sum(1 for row in rows if isinstance(row.get("usage"), dict))
    sums["records_by_model"] = dict(sorted(models.items()))
    return sums


def _probe_rows(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "usage": probe.get("usage"),
        "actual_model": probe.get("actual_model"),
        "source": probe.get("source"),
        "model_id": None,
    } for probe in probes]


def _try_cost_calculation(rows: list[dict[str, Any]], rates_path: Path | None) -> dict[str, Any]:
    missing_usage = sum(
        1 for row in rows
        if any(_usage_values(row.get("usage"))[key] is None for key in ("prompt_tokens", "completion_tokens"))
    )
    if rates_path is None:
        return {"computed_cost_rmb": None, "records_missing_input_or_output_usage": missing_usage,
                "basis": "No verified provider invoice or user-supplied endpoint rates; RMB cost is not estimated. Returned usage does not establish the billing of failed requests."}
    rates_obj = json.loads(rates_path.read_text(encoding="utf-8"))
    rates = rates_obj.get("models", rates_obj)
    per_model: dict[str, dict[str, float]] = defaultdict(lambda: {"input_tokens": 0.0, "cached_input_tokens": 0.0, "output_tokens": 0.0, "cost_rmb": 0.0})
    missing: set[str] = set()
    for row in rows:
        usage = _usage_values(row.get("usage"))
        model = str(row.get("model_id") or row.get("actual_model") or "unknown")
        rate = rates.get(model)
        if not isinstance(rate, dict):
            missing.add(model)
            continue
        required = ("input_rmb_per_1m_tokens", "cached_input_rmb_per_1m_tokens", "output_rmb_per_1m_tokens")
        if any(not isinstance(rate.get(key), (int, float)) for key in required):
            missing.add(model)
            continue
        input_tokens = usage["prompt_tokens"] or 0
        cached_tokens = min(input_tokens, usage["cached_tokens"] or 0)
        output_tokens = usage["completion_tokens"] or 0
        uncached_tokens = max(0, input_tokens - cached_tokens)
        cost = (
            uncached_tokens * float(rate["input_rmb_per_1m_tokens"])
            + cached_tokens * float(rate["cached_input_rmb_per_1m_tokens"])
            + output_tokens * float(rate["output_rmb_per_1m_tokens"])
        ) / 1_000_000
        acc = per_model[model]
        acc["input_tokens"] += uncached_tokens
        acc["cached_input_tokens"] += cached_tokens
        acc["output_tokens"] += output_tokens
        acc["cost_rmb"] += cost
    complete = not missing and missing_usage == 0
    return {
        "computed_cost_rmb": sum(v["cost_rmb"] for v in per_model.values()) if complete else None,
        "per_model": dict(per_model),
        "missing_rate_models": sorted(missing),
        "records_missing_input_or_output_usage": missing_usage,
        "basis": f"Rates supplied in {rates_path}; calculated only when all models have rates and every request has input/output usage." if complete else "Total RMB cost omitted because rates or input/output usage are incomplete; per_model contains only the calculable portion.",
    }


def analyze(data_path: Path, config_path: Path, log_path: Path, split: str = "all", output_dir: Path = ANALYSIS_DIR, probe_dir: Path | None = None, prices_path: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt_spec = load_prompt_spec()
    records = read_jsonl(data_path)
    wanted_splits = {"dev", "test"} if split == "all" else {split}
    records = [row for row in records if row.get("split") in wanted_splits]
    attempt_rows = read_jsonl(log_path)
    selected, attempts_by_task = select_successes(attempt_rows)
    response_by_task = {task_id: row for task_id, row in selected.items()}
    scored: list[dict[str, Any]] = []

    for current_split in sorted(wanted_splits):
        split_records = [row for row in records if row.get("split") == current_split]
        if not split_records:
            continue
        tasks = make_tasks(split_records, config, current_split)
        for task in tasks:
            response = response_by_task.get(task["task_id"])
            if response:
                score = score_content(
                    response.get("assistant_content"), task["record"]["gold_decision"], task["arm"], response.get("finish_reason")
                )
            else:
                score = {
                    "primary_correct": None, "strict_json_ok": None, "schema_valid": None,
                    "decision_field_present": None, "decision_enum_valid": None, "decision": None,
                    "gold_decision": task["record"]["gold_decision"], "semantic_wrong": None,
                    "format_or_missing_decision": None, "format_violation": None, "brief_chars": None,
                    "brief_within_limit": None, "relation_chars": None, "relation_within_limit": None,
                    "loose_decision": None, "loose_correct": None, "loose_parse_used": None,
                    "failure_category": "not_scored_no_model_content", "truncated": None,
                }
            usage = _usage_values(response.get("usage") if response else None)
            usage_columns = {
                "prompt_tokens_used": usage["prompt_tokens"],
                "completion_tokens_used": usage["completion_tokens"],
                "total_tokens_value": usage["total_tokens"],
                "cached_tokens_used": usage["cached_tokens"],
                "reasoning_tokens_used": usage["reasoning_tokens"],
            }
            previous = attempts_by_task.get(task["task_id"], [])
            latest_attempt = max(previous, key=lambda item: (int(item.get("attempt_no", 0)), _when(item))) if previous else None
            final_status = "ok" if response else (latest_attempt.get("status") if latest_attempt else "missing")
            if final_status == "protocol_error":
                no_content_reason = "protocol_error"
            elif response is None:
                no_content_reason = "transport_or_run_missing"
            else:
                no_content_reason = None
            scored.append({
                "task_id": task["task_id"], "split": task["split"], "sample_id": task["sample_id"],
                "family_id": task["family_id"], "complexity": task["complexity"], "model_id": task["model_id"],
                "arm": task["arm"], "repeat_id": task["repeat_id"], "task_status": final_status,
                "no_content_reason": no_content_reason,
                "gold_decision": task["record"]["gold_decision"], "assistant_content": response.get("assistant_content") if response else None,
                "actual_model": response.get("actual_model") if response else None,
                "finish_reason": response.get("finish_reason") if response else None,
                "elapsed_seconds": response.get("elapsed_seconds") if response else None,
                **score,
                **usage_columns,
            })

    # Aggregate predictions in two stages: repeats within sample, then examples within family.
    sample_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        if row["task_status"] != "ok":
            continue
        sample_groups[(row["split"], row["model_id"], row["arm"], row["complexity"], row["sample_id"])].append(row)
    sample_accuracy = {key: float(np.mean([int(row["primary_correct"]) for row in rows])) for key, rows in sample_groups.items()}
    family_groups: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    family_ids: dict[tuple[str, str, str, str, str], str] = {}
    for key, rows in sample_groups.items():
        split_name, model, arm, complexity, sample_id = key
        family_id = rows[0]["family_id"]
        fkey = (split_name, model, arm, complexity, family_id)
        family_groups[fkey].append(sample_accuracy[key])
        family_ids[fkey] = family_id
    family_acc = {key: float(np.mean(values)) for key, values in family_groups.items()}

    analysis_cfg = config["analysis"]
    bootstrap_n = int(analysis_cfg["bootstrap_replicates"])
    seed0 = int(analysis_cfg["bootstrap_seed"])
    metrics: list[dict[str, Any]] = []
    metric_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        metric_groups[(row["split"], row["model_id"], row["arm"], row["complexity"])].append(row)
    for idx, (key, rows) in enumerate(sorted(metric_groups.items())):
        split_name, model, arm, complexity = key
        answered = [row for row in rows if row["task_status"] == "ok"]
        fam_values = [acc for (s, m, a, c, _), acc in family_acc.items() if (s, m, a, c) == key]
        acc, acc_low, acc_high = cluster_bootstrap_ci(fam_values, bootstrap_n, seed0 + idx)
        valid_decisions = sum(bool(row["decision_enum_valid"]) for row in answered)
        correct_count = sum(bool(row["primary_correct"]) for row in answered)
        metric = {
            "split": split_name, "model_id": model, "arm": arm, "complexity": complexity,
            "expected_tasks": len(rows), "responses_received": sum(row["task_status"] == "ok" for row in rows),
            "response_coverage": (len(answered) / len(rows)) if rows else None,
            "transport_or_run_missing_tasks": sum(row["no_content_reason"] == "transport_or_run_missing" for row in rows),
            "protocol_error_tasks": sum(row["no_content_reason"] == "protocol_error" for row in rows),
            "families": len(fam_values), "primary_accuracy": acc,
            "primary_accuracy_ci95_low": acc_low, "primary_accuracy_ci95_high": acc_high,
            "correct_decisions": correct_count,
            "valid_decisions": valid_decisions,
            "accuracy_on_valid_decisions": (correct_count / valid_decisions) if valid_decisions else None,
            "semantic_wrong": sum(bool(row["semantic_wrong"]) for row in answered),
            "samples_with_multiple_responses": sum(len(rows_for_sample) > 1 for (s, m, a, c, _), rows_for_sample in sample_groups.items() if (s, m, a, c) == key),
            "samples_with_repeat_disagreement": sum(
                len({bool(row["primary_correct"]) for row in rows_for_sample}) > 1
                for (s, m, a, c, _), rows_for_sample in sample_groups.items() if (s, m, a, c) == key
            ),
            "repeat_disagreement_rate": (
                sum(len({bool(row["primary_correct"]) for row in rows_for_sample}) > 1
                    for (s, m, a, c, _), rows_for_sample in sample_groups.items() if (s, m, a, c) == key)
                / sum(len(rows_for_sample) > 1 for (s, m, a, c, _), rows_for_sample in sample_groups.items() if (s, m, a, c) == key)
                if any(len(rows_for_sample) > 1 for (s, m, a, c, _), rows_for_sample in sample_groups.items() if (s, m, a, c) == key)
                else None
            ),
            "format_or_missing_decision": sum(bool(row["format_or_missing_decision"]) for row in answered),
            "strict_json_rate": float(np.mean([bool(row["strict_json_ok"]) for row in answered])) if answered else None,
            "schema_valid_rate": float(np.mean([bool(row["schema_valid"]) for row in answered])) if answered else None,
            "brief_within_limit_rate": float(np.mean([bool(row["brief_within_limit"]) for row in answered])) if answered else None,
            "relation_within_limit_rate": (float(np.mean([bool(row["relation_within_limit"]) for row in answered])) if arm == "decompose" and answered else None),
            "loose_accuracy_sensitivity": float(np.mean([bool(row["loose_correct"]) for row in answered])) if answered else None,
            "truncated_count": sum(bool(row["truncated"]) for row in answered),
            "p50_elapsed_seconds": _percentile([float(row["elapsed_seconds"]) for row in answered if isinstance(row["elapsed_seconds"], (int, float))], 50),
            "p95_elapsed_seconds": _percentile([float(row["elapsed_seconds"]) for row in answered if isinstance(row["elapsed_seconds"], (int, float))], 95),
            "prompt_tokens_successes": sum(int(row["prompt_tokens_used"] or 0) for row in answered),
            "completion_tokens_successes": sum(int(row["completion_tokens_used"] or 0) for row in answered),
            "total_tokens_successes": sum(int(row["total_tokens_value"] or 0) for row in answered),
            "cached_tokens_successes": sum(int(row["cached_tokens_used"] or 0) for row in answered),
            "reasoning_tokens_successes": sum(int(row["reasoning_tokens_used"] or 0) for row in answered),
        }
        metrics.append(metric)

    # Export family-level values and paired family differences for auditability.
    family_rows = [{
        "split": key[0], "model_id": key[1], "arm": key[2], "complexity": key[3],
        "family_id": key[4], "family_accuracy": value,
    } for key, value in sorted(family_acc.items())]
    contrasts = [tuple(pair) for pair in analysis_cfg["primary_contrasts"]]
    pairwise: list[dict[str, Any]] = []
    family_difference_rows: list[dict[str, Any]] = []
    pair_idx = 0
    for current_split in sorted(wanted_splits):
        models = sorted({row["model_id"] for row in scored if row["split"] == current_split})
        complexities = sorted({row["complexity"] for row in scored if row["split"] == current_split})
        for model in models:
            for complexity in complexities:
                for arm_a, arm_b in contrasts:
                    task_index = {
                        (row["sample_id"], row["repeat_id"], row["arm"]): row
                        for row in scored
                        if row["split"] == current_split and row["model_id"] == model and row["complexity"] == complexity
                    }
                    arm_a_tasks = [
                        row for row in scored
                        if row["split"] == current_split and row["model_id"] == model and row["complexity"] == complexity and row["arm"] == arm_a
                    ]
                    expected_pairs = len(arm_a_tasks)
                    paired_by_family: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
                    for row_a in arm_a_tasks:
                        row_b = task_index.get((row_a["sample_id"], row_a["repeat_id"], arm_b))
                        if row_b is not None and row_a["task_status"] == "ok" and row_b["task_status"] == "ok":
                            paired_by_family[row_a["family_id"]].append((row_a, row_b))
                    family_diffs: list[float] = []
                    paired_response_count = sum(len(values) for values in paired_by_family.values())
                    paired_sample_ids: set[str] = set()
                    for family, pairs in sorted(paired_by_family.items()):
                        a_val = float(np.mean([int(a["primary_correct"]) for a, _ in pairs]))
                        b_val = float(np.mean([int(b["primary_correct"]) for _, b in pairs]))
                        diff = a_val - b_val
                        family_diffs.append(diff)
                        paired_sample_ids.update(a["sample_id"] for a, _ in pairs)
                        family_difference_rows.append({
                            "split": current_split, "model_id": model, "complexity": complexity,
                            "contrast": f"{arm_a}-{arm_b}", "family_id": family,
                            "accuracy_arm_a": a_val, "accuracy_arm_b": b_val, "difference": diff,
                            "complete_sample_repeat_pairs": len(pairs),
                            "samples_with_complete_pair": len({a["sample_id"] for a, _ in pairs}),
                        })
                    point, low, high = cluster_bootstrap_ci(family_diffs, bootstrap_n, seed0 + 100 + pair_idx)
                    pair_idx += 1
                    pairwise.append({
                        "split": current_split, "model_id": model, "complexity": complexity,
                        "contrast": f"{arm_a}-{arm_b}", "families_n": len(family_diffs),
                        "expected_sample_repeat_pairs": expected_pairs,
                        "complete_sample_repeat_pairs": paired_response_count,
                        "paired_response_coverage": (paired_response_count / expected_pairs) if expected_pairs else None,
                        "samples_with_at_least_one_complete_pair": len(paired_sample_ids),
                        "accuracy_difference": point, "ci95_low": low, "ci95_high": high,
                        "family_wins": sum(value > 0 for value in family_diffs),
                        "family_ties": sum(value == 0 for value in family_diffs),
                        "family_losses": sum(value < 0 for value in family_diffs),
                        "leave_one_family_out_min": (min(float(np.mean([value for j, value in enumerate(family_diffs) if j != omit])) for omit in range(len(family_diffs))) if len(family_diffs) > 1 else None),
                        "leave_one_family_out_max": (max(float(np.mean([value for j, value in enumerate(family_diffs) if j != omit])) for omit in range(len(family_diffs))) if len(family_diffs) > 1 else None),
                        "useful_effect_threshold": float(analysis_cfg["useful_effect_threshold"]),
                        "reaches_useful_threshold": bool(point is not None and point >= float(analysis_cfg["useful_effect_threshold"])),
                        "ci_lower_above_zero": bool(low is not None and low > 0),
                        "primary_complex_test_contrast": current_split == "test" and complexity == "complex",
                    })

    model_decisions: dict[str, dict[str, Any]] = {}
    if "test" in wanted_splits:
        for model in sorted({row["model_id"] for row in scored if row["split"] == "test"}):
            complex_rows = [row for row in pairwise if row["split"] == "test" and row["model_id"] == model and row["complexity"] == "complex"]
            simple_rows = [row for row in pairwise if row["split"] == "test" and row["model_id"] == model and row["complexity"] == "simple"]
            complex_pass = len(complex_rows) == 2 and all(
                row["reaches_useful_threshold"] and row["ci_lower_above_zero"] and row["paired_response_coverage"] == 1.0
                for row in complex_rows
            )
            simple_nondecline = len(simple_rows) == 2 and all(
                row["accuracy_difference"] is not None and row["accuracy_difference"] >= 0 and row["paired_response_coverage"] == 1.0
                for row in simple_rows
            )
            model_decisions[model] = {
                "complex_contrasts_meet_criteria": complex_pass,
                "simple_no_observed_decrease_vs_both_baselines": simple_nondecline,
                "model_supports_prespecified_retention_rule": bool(complex_pass and simple_nondecline),
                "interpretation": "supports the prespecified rule for this model" if complex_pass and simple_nondecline else "does not meet the full prespecified rule; interpret each contrast and interval separately",
            }
    cross_model_consistent = bool(model_decisions) and all(v["model_supports_prespecified_retention_rule"] for v in model_decisions.values())

    # Wrong/correct transitions are paired by repeat ID, never inferred from free-form explanations.
    scored_index = {(row["split"], row["model_id"], row["sample_id"], row["repeat_id"], row["arm"]): row for row in scored}
    wrong_cases = [{
        key: row.get(key) for key in ("split", "sample_id", "family_id", "complexity", "model_id", "arm", "repeat_id", "gold_decision", "decision", "failure_category", "task_status", "finish_reason", "assistant_content")
    } for row in scored if row["task_status"] == "ok" and not row["primary_correct"]]
    unanswered_tasks = [{
        key: row.get(key) for key in ("split", "sample_id", "family_id", "complexity", "model_id", "arm", "repeat_id", "task_status", "no_content_reason")
    } for row in scored if row["task_status"] != "ok"]
    transitions: list[dict[str, Any]] = []
    for split_name in sorted(wanted_splits):
        for model in sorted({row["model_id"] for row in scored if row["split"] == split_name}):
            for arm_a, arm_b in contrasts:
                for row in scored:
                    if row["split"] != split_name or row["model_id"] != model or row["arm"] != arm_a:
                        continue
                    baseline = scored_index.get((split_name, model, row["sample_id"], row["repeat_id"], arm_b))
                    if baseline is None or row["task_status"] != "ok" or baseline["task_status"] != "ok":
                        continue
                    a_ok, b_ok = bool(row["primary_correct"]), bool(baseline["primary_correct"])
                    if a_ok == b_ok:
                        continue
                    transitions.append({
                        "split": split_name, "model_id": model, "sample_id": row["sample_id"],
                        "family_id": row["family_id"], "complexity": row["complexity"], "repeat_id": row["repeat_id"],
                        "contrast": f"{arm_a}-{arm_b}", "transition": "wrong_to_right" if a_ok else "right_to_wrong",
                        "arm_a_decision": row["decision"], "arm_b_decision": baseline["decision"], "gold_decision": row["gold_decision"],
                    })

    all_attempts = attempt_rows
    probe_dir = probe_dir or ROOT / "results" / "protocol_checks"
    probes, probe_source = _read_probe_costs(probe_dir, config)
    task_usage_rows = [{
        "usage": row.get("usage"), "actual_model": row.get("actual_model"), "model_id": row.get("model_id"),
    } for row in all_attempts]
    probe_usage_rows = _probe_rows(probes)
    costs: dict[str, Any] = {
        "task_posts": len(all_attempts),
        "probe_posts": len(probes),
        "total_chat_completion_posts": len(all_attempts) + len(probes),
        "post_budget": int(config["api"]["post_budget_including_probes"]),
        "post_budget_remaining": int(config["api"]["post_budget_including_probes"]) - len(all_attempts) - len(probes),
        "inventory_get_requests_separately_reported": 4,
        "probe_cost_source": probe_source,
        "probe_usage": _usage_sum(probe_usage_rows),
        "task_attempt_usage": _usage_sum(task_usage_rows),
        "all_chat_completion_usage": _usage_sum([*probe_usage_rows, *task_usage_rows]),
        "task_attempt_status_counts": dict(Counter(str(row.get("status")) for row in all_attempts)),
        "task_http_status_counts": dict(Counter(str(row.get("http_status")) for row in all_attempts)),
        "rmb_cost": _try_cost_calculation([*probe_usage_rows, *task_usage_rows], prices_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", metrics)
    _write_csv(output_dir / "pairwise.csv", pairwise)
    _write_csv(output_dir / "family_differences.csv", family_difference_rows)
    _write_json(output_dir / "cases.json", {
        "wrong_cases": wrong_cases,
        "unanswered_or_protocol_error_tasks": unanswered_tasks,
        "paired_decision_transitions": transitions,
        "note": "No-content transport and service-protocol failures are NA for model accuracy and excluded from paired comparisons; an actually returned empty string is scored as a model formatting failure. A decision is scored only from the fixed JSON decision field; free-form brief/relation text is never used to infer a label.",
    })
    _write_json(output_dir / "costs.json", costs)
    summary = {
        "analyzed_splits": sorted(wanted_splits),
        "expected_scored_tasks": len(scored),
        "logged_attempt_rows": len(attempt_rows),
        "unique_task_ids_in_log": len(attempts_by_task),
        "successful_unique_task_ids": len(selected),
        "model_response_identifiers": {
            model: sorted({str(row.get("actual_model")) for row in selected.values() if row.get("model_id") == model and row.get("actual_model")})
            for model in sorted({str(row.get("model_id")) for row in selected.values()})
        },
        "strict_prompt_user_character_match": len(prompt_spec["arms"]["decompose"]["extra_instruction"] + "\n" + prompt_spec["arms"]["decompose"]["instruction"]) == len(prompt_spec["arms"]["length_control"]["extra_instruction"] + "\n" + prompt_spec["arms"]["length_control"]["instruction"]),
        "model_prespecified_rule": model_decisions,
        "cross_model_consistent_support": cross_model_consistent,
        "interpretive_limit": "Held-out evaluation on small, model-assisted, independently reviewed synthetic rule families; results are protocol-specific and not a population-level confirmation.",
        "costs_file": str(output_dir / "costs.json"),
    }
    _write_json(output_dir / "summary.json", summary)
    from .plot_results import make_figures
    make_figures(output_dir, ROOT / "figures")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "results" / "protocol_checks")
    parser.add_argument("--prices", type=Path, help="optional verified API rate table; without it no RMB amount is estimated")
    args = parser.parse_args()
    summary = analyze(args.data, args.config, args.log, args.split, args.output_dir, args.probe_dir, args.prices)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

