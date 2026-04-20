from .base import QualityContext, QualityRule, RuleResult


class Empty200Rule(QualityRule):
    code = "E002"
    description = "200空响应"
    depends_on = {"source_last_snapshot"}

    def evaluate(self, context: QualityContext) -> RuleResult:
        status_code = context.resp.get("status_code")
        return RuleResult(
            code=self.code,
            hit=status_code == 200 and not context.resp_content,
            note=self.description,
        )


RULE = Empty200Rule()
