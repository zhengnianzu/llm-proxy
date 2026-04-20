from .base import QualityContext, QualityRule, RuleResult


class ToolUseLt3Rule(QualityRule):
    code = "E003"
    description = "工具调用过少(<3次)"
    depends_on = {"cache"}

    def evaluate(self, context: QualityContext) -> RuleResult:
        tool_use_count = context.stats.get("tool_use_count", context.row.get("tool_use_count", 0)) or 0
        return RuleResult(
            code=self.code,
            hit=tool_use_count < 3,
            note=self.description,
        )


RULE = ToolUseLt3Rule()
