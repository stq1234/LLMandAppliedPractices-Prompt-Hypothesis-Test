from __future__ import annotations

import itertools
import json
import csv
import tempfile
import unittest
import contextlib
import io
import os
from pathlib import Path
from unittest.mock import patch

from src.analyze import analyze, cluster_bootstrap_ci, select_successes, _try_cost_calculation
from src.data import FAMILIES, make_records
from src.prompts import build_messages, load_prompt_spec
from src.runner import Budget, _retryable_transport, _secret_redact, _success_row, make_tasks, run_split
from src.scoring import score_content


def test_complete_control_prompts_match_and_gold_never_enters_messages():
    spec = load_prompt_spec()
    decompose = spec["arms"]["decompose"]
    control = spec["arms"]["length_control"]
    record = {
        "scenario": "场景甲",
        "rules_text": "规则甲",
        "prohibitions_text": "禁止甲",
        "facts_text": "事实甲",
        "gold_decision": "ALLOW",
        "gold_ast": {"private": True},
        "gold_rationale": "private text",
    }
    decompose_prompt = build_messages(record, "decompose", spec)[1]["content"]
    control_prompt = build_messages(record, "length_control", spec)[1]["content"]
    assert len(decompose_prompt) == len(control_prompt)
    assert "gold_decision" not in decompose_prompt + control_prompt
    assert "private text" not in decompose_prompt + control_prompt
    assert "白色纸张" in control_prompt


def test_task_grid_counts_and_repeat_ids_are_local_identifiers():
    rows = [{
        "sample_id": "D01-V1", "family_id": "D01", "complexity": "simple",
        "split": "dev", "scenario": "s", "rules_text": "r", "prohibitions_text": "p",
        "facts_text": "f", "gold_decision": "ALLOW",
    }]
    config = {
        "models": [{"id": "qwen3.8-flash"}, {"id": "deepseek-v4.1-flash"}],
        "arms": ["direct", "decompose", "length_control"], "dev_repeats": [17],
        "test_repeats": [17, 43], "schedule_seed": 42,
    }
    dev = make_tasks(rows, config, "dev")
    test = make_tasks(rows, config, "test")
    assert len(dev) == 6
    assert len(test) == 12
    assert {task["repeat_id"] for task in test} == {17, 43}
    assert len({task["task_id"] for task in test}) == 12
    assert [t["task_id"] for t in make_tasks(rows, config, "test")] == [t["task_id"] for t in test]


def test_strict_decision_scoring_separates_format_from_semantic_errors():
    correct = score_content('{"decision":"ALLOW","brief":"条件满足"}', "ALLOW", "direct")
    wrong = score_content('{"decision":"DENY","brief":"条件满足"}', "ALLOW", "direct")
    malformed = score_content('{"decision":"ALLOW","brief":"条件满足",}', "ALLOW", "direct")
    missing = score_content('{"brief":"答案是ALLOW"}', "ALLOW", "direct")
    assert correct["primary_correct"] and correct["schema_valid"]
    assert wrong["semantic_wrong"] and not wrong["format_or_missing_decision"]
    assert not malformed["primary_correct"] and malformed["format_or_missing_decision"]
    assert not missing["primary_correct"] and missing["decision"] is None
    assert missing["loose_correct"] is False


def test_loose_json_is_sensitivity_only_and_never_guesses_from_prose():
    fenced = score_content('结果如下：```json\n{"decision":"DENY","brief":"禁止优先"}\n```', "DENY", "direct")
    prose = score_content("判断结果是 ALLOW，因为条件看起来满足。", "ALLOW", "direct")
    duplicate = score_content('{"decision":"ALLOW","decision":"DENY","brief":"x"}', "ALLOW", "direct")
    assert not fenced["strict_json_ok"]
    assert fenced["loose_correct"] and fenced["loose_parse_used"]
    assert not fenced["primary_correct"]
    assert not prose["primary_correct"] and not prose["loose_correct"]
    assert not duplicate["strict_json_ok"] and not duplicate["loose_correct"]
    for constant in ("NaN", "Infinity", "-Infinity"):
        nonstandard_number = score_content(f'{{"decision":"ALLOW","brief":"x","other":{constant}}}', "ALLOW", "direct")
        assert not nonstandard_number["strict_json_ok"]


def test_schema_limits_are_independent_of_fixed_decision_accuracy():
    long_brief = "甲" * 81
    result = score_content(json.dumps({"decision": "ALLOW", "brief": long_brief}, ensure_ascii=False), "ALLOW", "direct")
    decompose = score_content(
        json.dumps({"relation": "关系" * 21, "decision": "ALLOW", "brief": "充分"}, ensure_ascii=False),
        "ALLOW", "decompose",
    )
    assert result["primary_correct"] and not result["brief_within_limit"] and not result["schema_valid"]
    assert decompose["primary_correct"] and not decompose["relation_within_limit"] and not decompose["schema_valid"]


def test_independent_truth_table_matches_all_generated_gold_labels():
    def reference(node, facts):
        # A separate pattern-matching truth-table evaluator, intentionally not
        # calling src.data.eval_ast.
        match node["op"]:
            case "const":
                return node["value"] is True
            case "var":
                return facts[node["name"]] is True
            case "not":
                return not reference(node["arg"], facts)
            case "all":
                return False if any(not reference(arg, facts) for arg in node["args"]) else True
            case "any":
                return True if any(reference(arg, facts) for arg in node["args"]) else False
            case _:
                raise AssertionError(f"unknown operator: {node['op']}")

    records = make_records()
    assert len(records) == 60
    for family in FAMILIES:
        atom_names = list(family["facts"])
        for values in itertools.product((False, True), repeat=len(atom_names)):
            assignment = dict(zip(atom_names, values))
            allow = reference(family["allow"], assignment)
            deny = reference(family["deny"], assignment)
            expected = "DENY" if deny or not allow else "ALLOW"
            assert expected in {"ALLOW", "DENY"}
        for variant, assignment in enumerate(family["cases"], 1):
            record = next(row for row in records if row["sample_id"] == f"{family['id']}-V{variant}")
            allow = reference(family["allow"], assignment)
            deny = reference(family["deny"], assignment)
            assert record["gold_decision"] == ("DENY" if deny or not allow else "ALLOW")
            assert record["gold_eval"]["allow_value"] is allow
            assert record["gold_eval"]["deny_value"] is deny


def test_cluster_bootstrap_is_deterministic_and_paired_success_selection_deduplicates():
    values = [0.1, -0.2, 0.3, 0.0]
    assert cluster_bootstrap_ci(values, 1000, 91) == cluster_bootstrap_ci(values, 1000, 91)
    assert cluster_bootstrap_ci([], 100, 1) == (None, None, None)
    attempts = [
        {"task_id": "t", "attempt_no": 1, "status": "retryable_error", "received_at_utc": "1"},
        {"task_id": "t", "attempt_no": 2, "status": "retryable_error", "received_at_utc": "2"},
        {"task_id": "t", "attempt_no": 3, "status": "failed", "received_at_utc": "3"},
        {"task_id": "t", "attempt_no": 4, "status": "ok", "received_at_utc": "4", "value": "recovered success"},
        {"task_id": "protocol", "attempt_no": 1, "status": "protocol_error", "received_at_utc": "5"},
        {"task_id": "u", "attempt_no": 1, "status": "failed"},
    ]
    selected, grouped = select_successes(attempts)
    assert selected["t"]["value"] == "recovered success"
    assert len(selected) == 1
    assert "protocol" not in selected
    assert "u" not in selected
    assert len(grouped["t"]) == 4


def test_budget_reserves_posts_and_tokens_and_transport_classifier_is_available():
    budget = Budget(post_limit=5, token_limit=100, known_posts=3, known_tokens=20, prior_rows=[])
    assert budget.reserve(30)
    assert not budget.reserve(60)
    budget.finish(30, {"total_tokens": 25})
    assert budget.reserve(50)
    assert not budget.reserve(1)  # all five POST slots are now reserved/used
    assert _secret_redact({"body": "before-secret-after"}, "secret")["body"] == "before-[REDACTED]-after"
    import requests
    assert _retryable_transport(requests.exceptions.ChunkedEncodingError())


def test_empty_model_content_is_scored_but_invalid_chat_envelopes_are_protocol_errors():
    import requests

    task = {
        "task_id": "t", "split": "dev", "sample_id": "D01-V1", "family_id": "D01",
        "complexity": "simple", "model_id": "qwen3.8-flash", "arm": "direct", "repeat_id": 17,
    }
    payload = {"model": task["model_id"], "messages": []}

    def response(body):
        result = requests.Response()
        result.status_code = 200
        result._content = json.dumps(body).encode("utf-8")
        return result

    empty_content = _success_row(
        response({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}),
        payload, task, 1, 0.01, "2026-11-13T00:00:00+00:00", "2026-11-13T00:00:01+00:00",
    )
    invalid_envelope = _success_row(response({"choices": []}), payload, task, 1, 0.01, "t0", "t1")
    assert empty_content["status"] == "ok" and empty_content["assistant_content"] == ""
    assert invalid_envelope["status"] == "protocol_error"
    assert invalid_envelope["assistant_content"] is None
    selected, _ = select_successes([invalid_envelope])
    assert selected == {}


def test_resume_skips_existing_successful_task_ids_without_sending_requests():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        row = {
            "sample_id": "D01-V1", "family_id": "D01", "complexity": "simple", "split": "dev",
            "scenario": "s", "rules_text": "r", "prohibitions_text": "p", "facts_text": "f", "gold_decision": "ALLOW",
        }
        config = {
            "api": {"base_url": "https://invalid.example", "timeout_seconds": 2, "max_retries": 2,
                    "max_consecutive_task_failures": 8, "max_concurrency": 4,
                    "post_budget_including_probes": 700, "token_budget_including_probes": 450000,
                    "known_probe_posts": 0, "known_probe_tokens": 0},
            "models": [{"id": "qwen3.8-flash", "request_extra": {}}],
            "arms": ["direct", "decompose", "length_control"], "temperature": 0.3, "top_p": 1, "max_tokens": 384,
            "dev_repeats": [17], "test_repeats": [17, 43], "schedule_seed": 4,
        }
        config_path, data_path, log_path = root / "config.json", root / "data.jsonl", root / "attempts.jsonl"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        data_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        tasks = make_tasks([row], config, "dev")
        log_path.write_text("".join(json.dumps({"task_id": task["task_id"], "status": "ok", "usage": {"total_tokens": 1}}) + "\n" for task in tasks), encoding="utf-8")
        before = log_path.read_text(encoding="utf-8")
        with patch.dict(os.environ, {"CASDAO_API_KEY": "unit-test-secret"}), \
                patch("requests.Session.post", side_effect=AssertionError("resume must not call the network")), \
                contextlib.redirect_stdout(io.StringIO()):
            status = run_split("dev", config_path, data_path, log_path, resume=True)
        assert status == 0
        assert log_path.read_text(encoding="utf-8") == before


def test_no_response_is_na_and_never_becomes_a_paired_wrong_to_right_transition():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        records = [{
            "sample_id": f"C01-V{i}", "family_id": "C01", "split": "test", "complexity": "complex",
            "scenario": "s", "rules_text": "r", "prohibitions_text": "p", "facts_text": "f",
            "gold_decision": "ALLOW",
        } for i in range(1, 4)]
        config = {
            "api": {"post_budget_including_probes": 700, "known_probe_posts": 0, "known_probe_tokens": 0, "known_probe_usage": []},
            "models": [{"id": "qwen3.8-flash"}], "arms": ["direct", "decompose", "length_control"],
            "dev_repeats": [17], "test_repeats": [17], "schedule_seed": 13,
            "analysis": {"bootstrap_replicates": 100, "bootstrap_seed": 4, "useful_effect_threshold": 0.05,
                         "primary_contrasts": [["decompose", "direct"], ["decompose", "length_control"]]},
        }
        data_path, config_path, log_path = root / "data.jsonl", root / "config.json", root / "attempts.jsonl"
        data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        tasks = make_tasks(records, config, "test")
        log = []
        for task in tasks:
            if task["arm"] == "decompose":
                continue
            content = json.dumps({"decision": "ALLOW", "brief": "许可条件满足"}, ensure_ascii=False)
            log.append({
                **{key: task[key] for key in ("task_id", "split", "sample_id", "family_id", "complexity", "model_id", "arm", "repeat_id")},
                "attempt_no": 1, "status": "ok", "assistant_content": content, "actual_model": task["model_id"],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "finish_reason": "stop", "elapsed_seconds": 0.1,
            })
        log_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in log), encoding="utf-8")
        output = root / "analysis"
        with patch("src.plot_results.make_figures", lambda *_args: None):
            analyze(data_path, config_path, log_path, "test", output, root / "no-probes")
        with (output / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            metrics = list(csv.DictReader(handle))
        missing_arm = next(row for row in metrics if row["arm"] == "decompose")
        assert int(missing_arm["responses_received"]) == 0
        assert missing_arm["primary_accuracy"] == ""
        with (output / "pairwise.csv").open(encoding="utf-8-sig", newline="") as handle:
            pairs = list(csv.DictReader(handle))
        decompose_pairs = [row for row in pairs if row["contrast"].startswith("decompose-")]
        assert all(row["accuracy_difference"] == "" for row in decompose_pairs)
        assert all(int(row["complete_sample_repeat_pairs"]) == 0 for row in decompose_pairs)
        cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
        assert cases["wrong_cases"] == []
        assert cases["paired_decision_transitions"] == []
        assert len(cases["unanswered_or_protocol_error_tasks"]) == 3


def test_complete_rates_do_not_price_unreported_failed_request_usage_as_zero():
    with tempfile.TemporaryDirectory() as temp_dir:
        prices = Path(temp_dir) / "rates.json"
        prices.write_text(json.dumps({"models": {"model": {
            "input_rmb_per_1m_tokens": 1, "cached_input_rmb_per_1m_tokens": 1,
            "output_rmb_per_1m_tokens": 2,
        }}}), encoding="utf-8")
        known = {"model_id": "model", "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        priced = _try_cost_calculation([known], prices)
        assert abs(priced["computed_cost_rmb"] - 0.00012) < 1e-12
        incomplete = _try_cost_calculation([known, {"model_id": "model", "usage": None}], prices)
        assert incomplete["computed_cost_rmb"] is None
        assert incomplete["records_missing_input_or_output_usage"] == 1
        assert incomplete["per_model"]["model"]["cost_rmb"] == priced["computed_cost_rmb"]


def load_tests(loader, tests, pattern):
    """Expose the same offline checks to both pytest and unittest discovery."""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
