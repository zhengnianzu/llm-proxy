from .base import QualityContext, QualityRule, RuleResult


class GarbledRule(QualityRule):
    code = "E001"
    description = "乱码(行均字符过少)"
    depends_on = {"source_last_snapshot"}

    def evaluate(self, context: QualityContext) -> RuleResult:
        return RuleResult(
            code=self.code,
            hit=bool(context.stats.get("has_garbled") or context.row.get("has_garbled")),
            note=self.description,
        )


RULE = GarbledRule()
