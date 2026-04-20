from .base import QualityContext, QualityRule, RuleResult


class WriteSuccessLt30Rule(QualityRule):
    code = "E004"
    description = "write成功率低于30%"
    depends_on = {"cache"}

    def evaluate(self, context: QualityContext) -> RuleResult:
        tool_success_detail = context.stats.get("tool_success_detail") or context.row.get("tool_success_detail") or {}
        tool_fail_detail = context.stats.get("tool_fail_detail") or context.row.get("tool_fail_detail") or {}
        write_success = tool_success_detail.get("write", 0) or 0
        write_fail = tool_fail_detail.get("write", 0) or 0
        resolved_calls = write_success + write_fail
        hit = resolved_calls > 0 and (write_success / resolved_calls) < 0.3
        return RuleResult(
            code=self.code,
            hit=hit,
            note=self.description,
        )


RULE = WriteSuccessLt30Rule()
