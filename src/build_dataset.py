"""Generate the immutable JSONL dataset and its human-review companion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import DATA_DIR, FAMILIES, make_records, write_dataset


def canonical_ast_shape(node: dict[str, Any]) -> str:
    """Return a variable-name-free, commutative tree signature."""
    op = node["op"]
    if op == "var":
        return "var"
    if op == "const":
        return f"const:{bool(node['value'])}"
    if op == "not":
        return f"not({canonical_ast_shape(node['arg'])})"
    children = sorted(canonical_ast_shape(child) for child in node["args"])
    return f"{op}({','.join(children)})"


def ast_variables(node: dict[str, Any]) -> set[str]:
    op = node["op"]
    if op == "var":
        return {node["name"]}
    if op == "const":
        return set()
    if op == "not":
        return ast_variables(node["arg"])
    return set().union(*(ast_variables(child) for child in node["args"]))


def validate_families() -> None:
    if len(FAMILIES) != 20 or len({f["id"] for f in FAMILIES}) != 20:
        raise ValueError("规则家族数量或ID不符合预注册规模")
    records = make_records()
    if len(records) != 60:
        raise ValueError("样本数必须为60")
    if sum(r["split"] == "dev" for r in records) != 12 or sum(r["split"] == "test" for r in records) != 48:
        raise ValueError("开发集/测试集必须分别为12/48例")
    signatures_by_split: dict[str, set[str]] = {"dev": set(), "test": set()}
    signature_owner: dict[str, str] = {}
    for family in FAMILIES:
        fact_keys = set(family["facts"])
        if ast_variables(family["allow"]) | ast_variables(family["deny"]) != fact_keys:
            raise ValueError(f"{family['id']} 存在未使用或未声明的事实变量")
        signature = json.dumps(
            [canonical_ast_shape(family["allow"]), canonical_ast_shape(family["deny"])],
            ensure_ascii=False,
        )
        split = family["split"]
        other = "test" if split == "dev" else "dev"
        if signature in signatures_by_split[other]:
            raise ValueError(f"跨split AST规范结构重复：{signature_owner[signature]} / {family['id']}")
        signatures_by_split[split].add(signature)
        signature_owner[signature] = family["id"]
        family_rows = [r for r in records if r["family_id"] == family["id"]]
        if len(family_rows) != 3 or len({r["gold_decision"] for r in family_rows}) < 2:
            raise ValueError(f"{family['id']} 必须有三个对照赋值并同时包含ALLOW与DENY")
        if any(set(r["facts"]) != fact_keys for r in family_rows):
            raise ValueError(f"{family['id']} 的事实赋值键不一致")
    if sum(f["split"] == "test" and f["complexity"] == "simple" for f in FAMILIES) != 4:
        raise ValueError("测试集必须有4个简单规则家族")
    if sum(f["split"] == "test" and f["complexity"] == "complex" for f in FAMILIES) != 12:
        raise ValueError("测试集必须有12个复杂规则家族")


def pretty_ast(node: dict[str, Any]) -> str:
    op = node["op"]
    if op == "var":
        return node["name"]
    if op == "const":
        return "真" if node["value"] else "假"
    if op == "not":
        return f"非（{pretty_ast(node['arg'])}）"
    sep = " 且 " if op == "all" else " 或 "
    return "（" + sep.join(pretty_ast(arg) for arg in node["args"]) + "）"


def write_review() -> Path:
    validate_families()
    records = make_records()
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(record["family_id"], []).append(record)

    lines = [
        "# 数据与Gold逻辑审查",
        "",
        "本数据由 `src/data.py` 中模型助手辅助编写并经显式复核的20个完整合成规则家族构成。开发集4家族、12例；正式测试集16家族、48例，其中简单家族4个、复杂家族12个。每个家族的3个事实赋值整体位于同一split；所有家族ID互异，开发与测试没有同一家族的换实体复本。下列gold AST和答案只用于审查、评分与分析，不进入模型提示。",
        "",
        "以变量名归一化、对且/或子节点排序的AST树形签名检查跨split重复；当前20个许可/禁止组合签名无跨split重复。该检查针对显式AST结构，不等同于对任意布尔等价式做完备化简。",
        "",
        "标签按 `DENY` 优先规则计算：`deny_if` 为真则拒绝；否则 `allow_when` 为真才允许；其余均拒绝。每例都记录了事实正反值、许可AST值、禁止AST值、决定路径与理由。",
        "",
        "## 按家族检查",
        "",
    ]
    for family in FAMILIES:
        family_records = by_family[family["id"]]
        lines.extend([
            f"### {family['id']} · {family['split']} · {family['complexity']} · {family['scenario']}",
            "",
            f"**规则：** {family['rules']}",
            "",
            f"**禁止：** {family['prohibitions']}",
            "",
            f"**Gold：** allow_when = `{pretty_ast(family['allow'])}`；deny_if = `{pretty_ast(family['deny'])}`。",
            "",
        ])
        for record in family_records:
            fact_values = "；".join(
                f"{record['fact_labels'][key]}={'是' if value else '否'}"
                for key, value in record["facts"].items()
            )
            lines.extend([
                f"- **{record['sample_id']}**：{fact_values}。gold=`{record['gold_decision']}`；"
                f"allow={str(record['gold_eval']['allow_value']).lower()}，"
                f"deny={str(record['gold_eval']['deny_value']).lower()}；"
                f"决策路径：{record['gold_eval']['decision_path']}。",
                f"  - 依据：{record['gold_rationale']}",
            ])
        lines.append("")

    lines.extend([
        "## 统计与检查要点",
        "",
        "| split | 复杂度 | 家族数 | 样本数 | ALLOW | DENY |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for split, complexity in (("dev", "simple"), ("dev", "complex"), ("test", "simple"), ("test", "complex")):
        subset = [r for r in records if r["split"] == split and r["complexity"] == complexity]
        fams = {r["family_id"] for r in subset}
        allow_count = sum(r["gold_decision"] == "ALLOW" for r in subset)
        lines.append(f"| {split} | {complexity} | {len(fams)} | {len(subset)} | {allow_count} | {len(subset)-allow_count} |")
    lines.extend([
        "",
        "开发集含两个6个原子事实的复杂家族（D03、D04）：均跨句表达常规许可、否定/暂停和例外解除；每家族均含许可成立时仍被禁止条款拦截的边界例。测试集复杂题含必要但不充分、否定例外、嵌套例外、条件必要性、恰有一项等结构。",
        "",
        "审阅重点：中文规则是否唯一对应AST；自然语言赋值是否足以判断全部原子事实；许可条件是否为穷尽条件；禁止优先与例外作用范围是否清楚；各家族内三例是否形成有辨识力的反转/边界变化。",
        "",
    ])
    out = DATA_DIR / "data_review.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    validate_families()
    records = write_dataset()
    review = write_review()
    print(f"Wrote {len(records)} examples to {DATA_DIR / 'dataset.jsonl'}")
    print(f"Wrote review document to {review}")
