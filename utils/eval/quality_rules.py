"""
utils/eval/quality_rules.py — 质量规则（E001-E004）

从 chat-log-viewer/src/quality_rules/ 合并而来，提供统一入口:
    evaluate_quality(context) -> List[str]
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass(frozen=True)
class RuleResult:
    code: str
    hit: bool
    note: str = ""


@dataclass
class QualityContext:
    resp: Dict[str, Any] = field(default_factory=dict)
    resp_content: Any = None
    stats: Dict[str, Any] = field(default_factory=dict)


QUALITY_ERRORS: Dict[str, str] = {
    "E001": "乱码(行均字符过少)",
    "E002": "200空响应",
    "E003": "工具调用过少(<3次)",
    "E004": "write成功率低于30%",
}


def _eval_e001(ctx: QualityContext) -> RuleResult:
    return RuleResult(
        code="E001",
        hit=bool(ctx.stats.get("has_garbled")),
        note=QUALITY_ERRORS["E001"],
    )


def _eval_e002(ctx: QualityContext) -> RuleResult:
    status_code = ctx.resp.get("status_code")
    return RuleResult(
        code="E002",
        hit=status_code == 200 and not ctx.resp_content,
        note=QUALITY_ERRORS["E002"],
    )


def _eval_e003(ctx: QualityContext) -> RuleResult:
    tool_use_count = ctx.stats.get("tool_use_count", 0) or 0
    return RuleResult(
        code="E003",
        hit=tool_use_count < 3,
        note=QUALITY_ERRORS["E003"],
    )


def _eval_e004(ctx: QualityContext) -> RuleResult:
    tool_success_detail = ctx.stats.get("tool_success_detail") or {}
    tool_fail_detail = ctx.stats.get("tool_fail_detail") or {}
    write_success = tool_success_detail.get("write", 0) or 0
    write_fail = tool_fail_detail.get("write", 0) or 0
    resolved_calls = write_success + write_fail
    hit = resolved_calls > 0 and (write_success / resolved_calls) < 0.3
    return RuleResult(
        code="E004",
        hit=hit,
        note=QUALITY_ERRORS["E004"],
    )


_RULES = [_eval_e001, _eval_e002, _eval_e003, _eval_e004]


def evaluate_quality(ctx: QualityContext) -> List[str]:
    return [r.code for fn in _RULES for r in [fn(ctx)] if r.hit]


def fmt_quality(error_codes: List[str]) -> tuple:
    if not error_codes:
        return 0, ""
    codes = ",".join(error_codes)
    note = "; ".join(QUALITY_ERRORS[c] for c in error_codes if c in QUALITY_ERRORS)
    return codes, note
