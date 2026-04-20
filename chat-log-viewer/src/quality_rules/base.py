from dataclasses import dataclass, field
from typing import Any, Dict, Set


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
    row: Dict[str, Any] = field(default_factory=dict)


class QualityRule:
    code = ""
    description = ""
    depends_on: Set[str] = {"cache"}

    def evaluate(self, context: QualityContext) -> RuleResult:
        raise NotImplementedError
