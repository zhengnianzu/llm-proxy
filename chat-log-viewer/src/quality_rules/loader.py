import importlib
import pkgutil
from functools import lru_cache
from typing import Dict, Iterable, List, Set

from .base import QualityContext, QualityRule


@lru_cache(maxsize=1)
def get_quality_rules() -> List[QualityRule]:
    package_name = __package__
    package = importlib.import_module(package_name)
    rules: List[QualityRule] = []

    for module_info in pkgutil.iter_modules(package.__path__):
        if not module_info.name.startswith("rule_"):
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        rule = getattr(module, "RULE", None)
        if rule is None:
            continue
        if not isinstance(rule, QualityRule):
            raise TypeError(f"{module.__name__}.RULE must be a QualityRule")
        if not rule.code:
            raise ValueError(f"{module.__name__}.RULE missing code")
        rules.append(rule)

    seen: Dict[str, str] = {}
    for rule in rules:
        prev = seen.get(rule.code)
        if prev:
            raise ValueError(f"Duplicate quality rule code: {rule.code} ({prev}, {rule.__class__.__name__})")
        seen[rule.code] = rule.__class__.__name__

    return sorted(rules, key=lambda r: r.code)


def get_quality_error_descriptions() -> Dict[str, str]:
    return {rule.code: rule.description for rule in get_quality_rules()}


def get_quality_rule_map() -> Dict[str, QualityRule]:
    return {rule.code: rule for rule in get_quality_rules()}


def get_rule_dependencies(codes: Iterable[str] = None) -> Dict[str, Set[str]]:
    selected = set(codes or [])
    deps: Dict[str, Set[str]] = {}
    for rule in get_quality_rules():
        if selected and rule.code not in selected:
            continue
        deps[rule.code] = set(rule.depends_on or set())
    return deps


def evaluate_quality_rules(context: QualityContext, codes: Iterable[str] = None) -> List[str]:
    selected = set(codes or [])
    error_codes: List[str] = []
    for rule in get_quality_rules():
        if selected and rule.code not in selected:
            continue
        result = rule.evaluate(context)
        if result.hit:
            error_codes.append(result.code)
    return error_codes
