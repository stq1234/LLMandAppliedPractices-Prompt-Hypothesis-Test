"""Generate reproducible figures and their plotted data from analysis CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _selected_split(metrics: list[dict[str, str]]) -> str | None:
    splits = {row.get("split") for row in metrics}
    return "test" if "test" in splits else (sorted(splits)[0] if splits else None)


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _model_label(model_id: str) -> str:
    if model_id.startswith("deepseek"):
        return "DeepSeek"
    if model_id.startswith("qwen"):
        return "Qwen"
    return model_id


def _arm_label(arm: str) -> str:
    return {"direct": "Direct", "decompose": "Decompose", "length_control": "Length control"}.get(arm, arm)


def _fig1(metrics: list[dict[str, str]], output_dir: Path, figures_dir: Path, split: str) -> None:
    arms = ["direct", "decompose", "length_control"]
    colors = {"direct": "#5577A6", "decompose": "#D97941", "length_control": "#54A08A"}
    models = sorted({row["model_id"] for row in metrics if row["split"] == split})
    complexities = [value for value in ("simple", "complex") if any(r["split"] == split and r["complexity"] == value for r in metrics)]
    if not models or not complexities:
        return
    fig, axes = plt.subplots(1, len(complexities), figsize=(max(7.4, 4.0 * len(complexities)), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    plotted: list[dict[str, Any]] = []
    width = 0.22
    group_gap = 0.5
    group_centers = np.arange(len(models)) * (len(arms) * width + group_gap)
    legend_handles = [Patch(facecolor=colors[arm], label=_arm_label(arm)) for arm in arms]
    for ax, complexity in zip(axes, complexities):
        rows = [r for r in metrics if r["split"] == split and r["complexity"] == complexity]
        for model_index, model in enumerate(models):
            xs, heights, lower, upper, labels = [], [], [], [], []
            for arm_index, arm in enumerate(arms):
                row = next((r for r in rows if r["model_id"] == model and r["arm"] == arm), None)
                value = _float(row or {}, "primary_accuracy")
                lo = _float(row or {}, "primary_accuracy_ci95_low")
                hi = _float(row or {}, "primary_accuracy_ci95_high")
                x = group_centers[model_index] + (arm_index - (len(arms) - 1) / 2) * width
                xs.append(x)
                heights.append(value if value is not None else 0.0)
                lower.append(max(0, (value or 0) - (lo if lo is not None else (value or 0))))
                upper.append(max(0, (hi if hi is not None else (value or 0)) - (value or 0)))
                plotted.append({"figure": "Fig1", "split": split, "complexity": complexity, "model_id": model, "arm": arm,
                                "accuracy": value, "ci95_low": lo, "ci95_high": hi,
                                "families": (row or {}).get("families"), "expected_tasks": (row or {}).get("expected_tasks")})
                labels.append(value)
            bars = ax.bar(xs, heights, color=[colors[arm] for arm in arms], width=width * 0.88, edgecolor="white", linewidth=0.5)
            for x, value, low, high, bar in zip(xs, labels, lower, upper, bars):
                if value is None:
                    ax.text(x, 0.025, "NA", ha="center", va="bottom", fontsize=7, color="#555555")
                else:
                    ax.errorbar([x], [value], yerr=[[low], [high]], fmt="none", ecolor="#263238", capsize=3, linewidth=1, zorder=4)
                    label_y = min(value + high + 0.012, 1.115)
                    ax.text(x, label_y, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5, color="#24313A")
        ax.set_xticks(group_centers, [_model_label(model) for model in models])
        ax.set_ylim(0, 1.15)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.set_title(complexity.capitalize())
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel("Primary decision accuracy" if ax is axes[0] else "")
    fig.suptitle(f"Figure 1. Held-out decision accuracy by model and prompt protocol ({split})", y=0.99, fontsize=11)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 0.84])
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "Fig1_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    _write_csv(output_dir / "figure1_plot_data.csv", plotted)


def _fig2(pairwise: list[dict[str, str]], output_dir: Path, figures_dir: Path, split: str) -> None:
    rows = [r for r in pairwise if r["split"] == split and r["complexity"] == "complex"]
    if not rows:
        return
    rows.sort(key=lambda r: (r["model_id"], r["contrast"]))
    y = np.arange(len(rows))
    points = [(_float(r, "accuracy_difference") or 0.0) * 100 for r in rows]
    lows = [None if _float(r, "ci95_low") is None else _float(r, "ci95_low") * 100 for r in rows]
    highs = [None if _float(r, "ci95_high") is None else _float(r, "ci95_high") * 100 for r in rows]
    left = [max(0, point - (low if low is not None else point)) for point, low in zip(points, lows)]
    right = [max(0, (high if high is not None else point) - point) for point, high in zip(points, highs)]
    fig, ax = plt.subplots(figsize=(8.0, max(3.1, 0.7 * len(rows) + 1.2)))
    ax.axvline(0, color="#444444", linewidth=1)
    ax.axvline(5, color="#B23A48", linestyle="--", linewidth=1, label="+5 pp usefulness threshold")
    ax.errorbar(points, y, xerr=[left, right], fmt="o", color="#315A7D", ecolor="#315A7D", capsize=4)
    short_contrast = {"decompose-direct": "Decompose − Direct", "decompose-length_control": "Decompose − Length control"}
    ax.set_yticks(y, [f"{_model_label(r['model_id'])} · {short_contrast.get(r['contrast'], r['contrast'])}" for r in rows])
    for point, ypos in zip(points, y):
        ax.annotate(f"{point:+.1f} pp", (point, ypos), xytext=(5, 7), textcoords="offset points", fontsize=8, color="#263238")
    ax.set_xlabel("Difference in accuracy (pp; paired family-cluster bootstrap 95% CI)", fontsize=9)
    ax.set_title(f"Figure 2. Paired effects on complex held-out cases ({split})")
    if lows and highs:
        finite = [v for v in [*lows, *highs] if v is not None]
        ax.set_xlim(min(-0.5, min(finite, default=0) - 0.5), max(5.6, max(finite, default=0) + 0.6))
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.5, 0.015, "A zero-width interval reflects identical family-level paired differences in this sample; it does not establish equivalence.", ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "Fig2_complex_effects.png", bbox_inches="tight")
    plt.close(fig)
    _write_csv(output_dir / "figure2_plot_data.csv", [
        {"figure": "Fig2", **row, "estimate_pp": point,
         "ci95_low_pp": low, "ci95_high_pp": high}
        for row, point, low, high in zip(rows, points, lows, highs)
    ])


def _fig3(metrics: list[dict[str, str]], output_dir: Path, figures_dir: Path, split: str) -> None:
    rows = [r for r in metrics if r["split"] == split and r["complexity"] == "complex"]
    if not rows:
        return
    rows.sort(key=lambda r: (r["model_id"], ["direct", "decompose", "length_control"].index(r["arm"])))
    colors = [{"direct": "#5577A6", "decompose": "#D97941", "length_control": "#54A08A"}[r["arm"]] for r in rows]
    models = sorted({row["model_id"] for row in rows})
    arms = ["direct", "decompose", "length_control"]
    width = 0.22
    group_gap = 0.5
    group_centers = np.arange(len(models)) * (len(arms) * width + group_gap)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    token_per_task: list[float] = []
    p95: list[float | None] = []
    plot_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for arm_index, arm in enumerate(arms):
            row = next((r for r in rows if r["model_id"] == model and r["arm"] == arm), None)
            if row is None:
                continue
            total = _float(row, "total_tokens_successes")
            responses = _float(row, "responses_received") or 0.0
            per_task = total / responses if total is not None and responses else None
            latency = _float(row, "p95_elapsed_seconds")
            token_per_task.append(per_task if per_task is not None else 0.0)
            p95.append(latency)
            plot_rows.append(row)
    xs: list[float] = []
    for model_index, model in enumerate(models):
        for arm_index, arm in enumerate(arms):
            xs.append(group_centers[model_index] + (arm_index - (len(arms) - 1) / 2) * width)
    axes[0].bar(xs, token_per_task, color=[{"direct": "#5577A6", "decompose": "#D97941", "length_control": "#54A08A"}[r["arm"]] for r in plot_rows], width=width * 0.88)
    axes[0].set_ylabel("Reported API tokens per successful task")
    axes[0].set_title("Token use")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(xs, [v or 0.0 for v in p95], color=[{"direct": "#5577A6", "decompose": "#D97941", "length_control": "#54A08A"}[r["arm"]] for r in plot_rows], width=width * 0.88)
    axes[1].set_ylabel("P95 request latency (s)")
    axes[1].set_title("Latency")
    axes[1].grid(axis="y", alpha=0.2)
    for ax in axes:
        ax.set_xticks(group_centers, [_model_label(model) for model in models])
    handles = [Patch(facecolor={"direct": "#5577A6", "decompose": "#D97941", "length_control": "#54A08A"}[arm], label=_arm_label(arm)) for arm in arms]
    fig.suptitle(f"Figure 3. Resource use on complex held-out cases ({split})", y=0.99, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 0.83])
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "Fig3_resources.png", bbox_inches="tight")
    plt.close(fig)
    _write_csv(output_dir / "figure3_plot_data.csv", [
        {"figure": "Fig3", "split": split, "complexity": row["complexity"], "model_id": row["model_id"],
         "arm": row["arm"], "total_tokens_successes": row["total_tokens_successes"],
         "responses_received": row["responses_received"], "tokens_per_success": tp, "p50_elapsed_seconds": row["p50_elapsed_seconds"],
         "p95_elapsed_seconds": row["p95_elapsed_seconds"]}
        for row, tp in zip(plot_rows, token_per_task)
    ])


def make_figures(analysis_dir: Path, figures_dir: Path) -> None:
    metrics = _read_csv(analysis_dir / "metrics.csv")
    pairwise = _read_csv(analysis_dir / "pairwise.csv")
    split = _selected_split(metrics)
    if split is None:
        return
    _setup_style()
    _fig1(metrics, analysis_dir, figures_dir, split)
    _fig2(pairwise, analysis_dir, figures_dir, split)
    _fig3(metrics, analysis_dir, figures_dir, split)


if __name__ == "__main__":
    from pathlib import Path
    base = Path(__file__).resolve().parents[1]
    make_figures(base / "results" / "analysis", base / "figures")

