"""Resumable, budget-limited runner for the fixed chat-completions protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .prompts import build_messages


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "experiment.json"
DEFAULT_DATA = ROOT / "data" / "dataset.jsonl"
DEFAULT_LOG = ROOT / "results" / "raw" / "attempts.jsonl"
RETRYABLE_HTTP = {429}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def load_dataset(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("split") == split]
    expected = {"sample_id", "family_id", "complexity", "scenario", "rules_text", "prohibitions_text", "facts_text", "gold_decision"}
    for row in rows:
        missing = expected - row.keys()
        if missing:
            raise ValueError(f"dataset record {row.get('sample_id')} lacks fields: {sorted(missing)}")
        if row["gold_decision"] not in {"ALLOW", "DENY"}:
            raise ValueError(f"invalid gold_decision for {row['sample_id']}")
    if not rows:
        raise ValueError(f"no records for split={split!r} in {path}")
    return rows


def _task_id(split: str, sample_id: str, model_id: str, arm: str, repeat_id: int) -> str:
    return f"{split}|{sample_id}|{model_id}|{arm}|r{repeat_id}"


def make_tasks(rows: list[dict[str, Any]], config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    repeats = config["dev_repeats"] if split == "dev" else config["test_repeats"]
    tasks: list[dict[str, Any]] = []
    for record in rows:
        for model in config["models"]:
            for arm in config["arms"]:
                for repeat_id in repeats:
                    task_id = _task_id(split, record["sample_id"], model["id"], arm, repeat_id)
                    tasks.append({
                        "task_id": task_id,
                        "split": split,
                        "sample_id": record["sample_id"],
                        "family_id": record["family_id"],
                        "complexity": record["complexity"],
                        "model_id": model["id"],
                        "arm": arm,
                        "repeat_id": repeat_id,
                        "record": record,
                        "model_spec": model,
                    })
    random.Random(config["schedule_seed"]).shuffle(tasks)
    return tasks


def _secret_redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_secret_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: _secret_redact(item, secret) for key, item in value.items()}
    return value


class JsonlWriter:
    def __init__(self, path: Path, secret: str) -> None:
        self.path = path
        self.secret = secret
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        safe = _secret_redact(row, self.secret)
        encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())


class Budget:
    """Shared POST/token budget; every real POST reserves one unit before sending."""

    def __init__(self, post_limit: int, token_limit: int, known_posts: int, known_tokens: int, prior_rows: list[dict[str, Any]]) -> None:
        self.post_limit = post_limit
        self.token_limit = token_limit
        self.posts = known_posts + len(prior_rows)
        self.tokens = known_tokens + sum(_total_tokens(row.get("usage")) for row in prior_rows)
        self.inflight_estimates = 0
        self.lock = threading.Lock()

    def reserve(self, estimate_tokens: int) -> bool:
        with self.lock:
            if self.posts >= self.post_limit:
                return False
            if self.tokens + self.inflight_estimates + estimate_tokens > self.token_limit:
                return False
            self.posts += 1
            self.inflight_estimates += estimate_tokens
            return True

    def finish(self, estimate_tokens: int, usage: dict[str, Any] | None) -> None:
        with self.lock:
            self.inflight_estimates = max(0, self.inflight_estimates - estimate_tokens)
            self.tokens += _total_tokens(usage)

    def snapshot(self) -> tuple[int, int, int]:
        with self.lock:
            return self.posts, self.tokens, self.inflight_estimates


def _total_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get("total_tokens")
    if isinstance(value, int) and value >= 0:
        return value
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return (prompt if isinstance(prompt, int) else 0) + (completion if isinstance(completion, int) else 0)


def _assistant_content(response_obj: Any) -> str | None:
    try:
        content = response_obj["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = [part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
        return "".join(pieces) if pieces else None
    return None


def _success_row(response: requests.Response, payload: dict[str, Any], task: dict[str, Any], attempt_no: int, elapsed: float, sent_at_utc: str, received_at_utc: str) -> dict[str, Any]:
    try:
        parsed = response.json()
        raw_json: Any = parsed
    except ValueError:
        parsed = None
        raw_json = response.text
    choice = None
    if isinstance(parsed, dict):
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
    content = _assistant_content(parsed)
    protocol_valid = bool(
        choice
        and isinstance(choice.get("message"), dict)
        and content is not None
        and (isinstance(choice["message"].get("content"), str) or isinstance(choice["message"].get("content"), list))
    )
    return {
        **_task_metadata(task),
        "attempt_no": attempt_no,
        "status": "ok" if protocol_valid else "protocol_error",
        "http_status": response.status_code,
        "elapsed_seconds": round(elapsed, 6),
        "sent_at_utc": sent_at_utc,
        "received_at_utc": received_at_utc,
        "request_payload": payload,
        "response_json": raw_json,
        "assistant_content": content if protocol_valid else None,
        "actual_model": parsed.get("model") if isinstance(parsed, dict) else None,
        "usage": parsed.get("usage") if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict) else None,
        "finish_reason": choice.get("finish_reason") if choice else None,
        "error_kind": None if protocol_valid else "missing_or_invalid_chat_completion_content",
    }


def _task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    return {key: task[key] for key in ("task_id", "split", "sample_id", "family_id", "complexity", "model_id", "arm", "repeat_id")}


def _retryable_transport(exc: requests.RequestException) -> bool:
    # All requests transport errors here are transient candidates; retry ceiling is fixed.
    return isinstance(exc, (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError))


def _estimate_tokens(payload: dict[str, Any], max_tokens: int, learned_estimate: int | None) -> int:
    if learned_estimate is not None:
        return learned_estimate
    user_chars = sum(len(m.get("content", "")) for m in payload["messages"] if isinstance(m, dict))
    # A conservative first-call reserve uses character count as a token proxy and a
    # modest completion allowance; observed usage replaces this estimate thereafter.
    return user_chars + min(max_tokens, 80) + 64


def run_one(task: dict[str, Any], config: dict[str, Any], prompt_spec: dict[str, Any], api_key: str, writer: JsonlWriter, budget: Budget, prior_attempts: int, learned: dict[tuple[str, str], list[int]]) -> dict[str, Any]:
    model_spec = task["model_spec"]
    payload = {
        "model": task["model_id"],
        "messages": build_messages(task["record"], task["arm"], prompt_spec),
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        **model_spec.get("request_extra", {}),
    }
    key = (task["model_id"], task["arm"])
    observations = learned.get(key, [])
    learned_estimate = max(observations[-20:]) if observations else None
    estimate = _estimate_tokens(payload, config["max_tokens"], learned_estimate)
    api_cfg = config["api"]
    max_attempts = int(api_cfg["max_retries"]) + 1
    first_attempt = prior_attempts + 1
    if first_attempt > max_attempts:
        return {**_task_metadata(task), "status": "exhausted", "attempts_used": prior_attempts}

    for local_idx in range(max_attempts - prior_attempts):
        attempt_no = first_attempt + local_idx
        if not budget.reserve(estimate):
            return {**_task_metadata(task), "status": "budget_stop", "attempts_used": local_idx}
        start = time.perf_counter()
        sent_at_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        response: requests.Response | None = None
        error_kind: str | None = None
        try:
            with requests.Session() as session:
                session.trust_env = bool(api_cfg.get("trust_env", True))
                connect_timeout = api_cfg.get("connect_timeout_seconds")
                read_timeout = api_cfg.get("read_timeout_seconds")
                if connect_timeout is not None and read_timeout is not None:
                    request_timeout: float | tuple[float, float] = (float(connect_timeout), float(read_timeout))
                else:
                    request_timeout = float(api_cfg["timeout_seconds"])
                response = session.post(
                    api_cfg["base_url"],
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=request_timeout,
                )
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - start
            received_at_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            error_kind = type(exc).__name__
            budget.finish(estimate, None)
            retryable = _retryable_transport(exc)
            row = {
                **_task_metadata(task), "attempt_no": attempt_no, "status": "retryable_error" if retryable and local_idx < max_attempts - prior_attempts - 1 else "failed",
                "http_status": None, "elapsed_seconds": round(elapsed, 6), "sent_at_utc": sent_at_utc,
                "received_at_utc": received_at_utc, "request_payload": payload,
                "response_json": None, "assistant_content": None, "actual_model": None, "usage": None,
                "finish_reason": None, "error_kind": error_kind,
            }
            writer.append(row)
            if retryable and local_idx < max_attempts - prior_attempts - 1:
                time.sleep(min(2 ** local_idx, 4))
                continue
            return {**_task_metadata(task), "status": "failed", "attempts_used": local_idx + 1}

        elapsed = time.perf_counter() - start
        received_at_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        status = response.status_code
        if status == 401 or status == 403:
            budget.finish(estimate, None)
            writer.append({
                **_task_metadata(task), "attempt_no": attempt_no, "status": "auth_error", "http_status": status,
                "elapsed_seconds": round(elapsed, 6), "sent_at_utc": sent_at_utc,
                "received_at_utc": received_at_utc, "request_payload": payload, "response_json": None,
                "assistant_content": None, "actual_model": None, "usage": None, "finish_reason": None,
                "error_kind": "authentication_or_authorization_failed",
            })
            return {**_task_metadata(task), "status": "auth_error", "attempts_used": local_idx + 1}

        if status in RETRYABLE_HTTP or 500 <= status <= 599:
            budget.finish(estimate, None)
            retryable = local_idx < max_attempts - prior_attempts - 1
            try:
                error_body: Any = response.json()
            except ValueError:
                error_body = response.text
            writer.append({
                **_task_metadata(task), "attempt_no": attempt_no, "status": "retryable_http" if retryable else "failed",
                "http_status": status, "elapsed_seconds": round(elapsed, 6), "sent_at_utc": sent_at_utc,
                "received_at_utc": received_at_utc, "request_payload": payload, "response_json": error_body,
                "assistant_content": None, "actual_model": None, "usage": None,
                "finish_reason": None, "error_kind": "retryable_http_status" if retryable else "retry_limit_reached",
            })
            if retryable:
                time.sleep(min(2 ** local_idx, 4))
                continue
            return {**_task_metadata(task), "status": "failed", "attempts_used": local_idx + 1}

        if not 200 <= status < 300:
            budget.finish(estimate, None)
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            writer.append({
                **_task_metadata(task), "attempt_no": attempt_no, "status": "failed", "http_status": status,
                "elapsed_seconds": round(elapsed, 6), "sent_at_utc": sent_at_utc, "received_at_utc": received_at_utc,
                "request_payload": payload, "response_json": error_body,
                "assistant_content": None, "actual_model": None, "usage": None, "finish_reason": None,
                "error_kind": "non_retryable_http_status",
            })
            return {**_task_metadata(task), "status": "failed", "attempts_used": local_idx + 1}

        row = _success_row(response, payload, task, attempt_no, elapsed, sent_at_utc, received_at_utc)
        budget.finish(estimate, row.get("usage"))
        total = _total_tokens(row.get("usage"))
        if total:
            learned.setdefault(key, []).append(total)
        writer.append(row)
        return {**_task_metadata(task), "status": row["status"], "attempts_used": local_idx + 1, "usage": row.get("usage")}

    return {**_task_metadata(task), "status": "failed", "attempts_used": max_attempts - prior_attempts}


def run_split(split: str, config_path: Path, data_path: Path, log_path: Path, resume: bool = False, dry_run: bool = False) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt_spec = json.loads((ROOT / "prompts" / "arms.json").read_text(encoding="utf-8"))
    # Validate the matched-length control at every entry point.
    from .prompts import load_prompt_spec
    prompt_spec = load_prompt_spec()
    records = load_dataset(data_path, split)
    tasks = make_tasks(records, config, split)
    if dry_run:
        print(json.dumps({"split": split, "records": len(records), "tasks": len(tasks), "task_ids": [t["task_id"] for t in tasks[:5]], "models": [m["id"] for m in config["models"]], "arms": config["arms"]}, ensure_ascii=False, indent=2))
        return 0

    api_key = os.environ.get("CASDAO_API_KEY", "")
    if not api_key:
        raise RuntimeError("CASDAO_API_KEY is not set; no request was sent")
    prior_rows = read_jsonl(log_path)
    existing_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in prior_rows:
        existing_by_task.setdefault(row.get("task_id", ""), []).append(row)
    skipped = {task_id for task_id, rows_for_task in existing_by_task.items() if any(row.get("status") == "ok" for row in rows_for_task)} if resume else set()
    pending_tasks = [task for task in tasks if task["task_id"] not in skipped]
    prior_attempt_counts = {task_id: len(rows_for_task) for task_id, rows_for_task in existing_by_task.items()} if resume else {}
    used_posts = len(prior_rows)
    known_posts = int(config["api"].get("known_probe_posts", 0))
    known_tokens = int(config["api"].get("known_probe_tokens", 0))
    budget = Budget(
        int(config["api"]["post_budget_including_probes"]),
        int(config["api"]["token_budget_including_probes"]),
        known_posts,
        known_tokens,
        prior_rows,
    )
    learned: dict[tuple[str, str], list[int]] = {}
    for row in prior_rows:
        total = _total_tokens(row.get("usage"))
        if total:
            learned.setdefault((row.get("model_id", ""), row.get("arm", "")), []).append(total)
    writer = JsonlWriter(log_path, api_key)

    max_workers = int(config["api"]["max_concurrency"])
    max_consecutive = int(config["api"]["max_consecutive_task_failures"])
    queue_idx = 0
    consecutive_failures = 0
    auth_failed = False
    budget_stopped = False
    outcomes: list[dict[str, Any]] = []
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while queue_idx < len(pending_tasks) or futures:
            while queue_idx < len(pending_tasks) and len(futures) < max_workers and consecutive_failures < max_consecutive and not auth_failed and not budget_stopped:
                task = pending_tasks[queue_idx]
                queue_idx += 1
                # Retry only failures with remaining attempt allowance; successful task IDs are skipped.
                future = pool.submit(
                    run_one, task, config, prompt_spec, api_key, writer, budget,
                    prior_attempt_counts.get(task["task_id"], 0), learned,
                )
                futures[future] = task
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: futures[item]["task_id"]):
                task = futures.pop(future)
                outcome = future.result()
                outcomes.append(outcome)
                status = outcome.get("status")
                if status == "ok":
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                if status == "auth_error":
                    auth_failed = True
                if status == "budget_stop":
                    budget_stopped = True
            if auth_failed or budget_stopped or consecutive_failures >= max_consecutive:
                # Already submitted requests finish and are logged; no additional task is sent.
                continue

    status_counts: dict[str, int] = {}
    for row in outcomes:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    posts, tokens, reserved = budget.snapshot()
    print(json.dumps({
        "split": split,
        "scheduled": len(tasks),
        "skipped_successes": len(skipped),
        "attempted_tasks": len(outcomes),
        "task_status": status_counts,
        "consecutive_failures": consecutive_failures,
        "auth_stopped": auth_failed,
        "budget_stopped": budget_stopped,
        "post_budget_used_including_probes": posts,
        "reported_tokens_including_probes": tokens,
        "inflight_token_reserve": reserved,
        "log": str(log_path),
    }, ensure_ascii=False, indent=2))
    return 2 if auth_failed or budget_stopped or consecutive_failures >= max_consecutive else (1 if any(x["status"] not in {"ok"} for x in outcomes) else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--resume", action="store_true", help="skip task IDs with a successful logged response")
    parser.add_argument("--dry-run", action="store_true", help="validate data and show task counts without API calls")
    args = parser.parse_args()
    return run_split(args.split, args.config, args.data, args.log, args.resume, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

