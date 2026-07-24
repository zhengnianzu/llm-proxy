"""test/test_custom_models.py — 自定义模型映射单元测试"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utils.custom_models as cm
from utils.custom_models import load_custom_models, resolve_custom_model


def _setup():
    """直接注入内存配置，不依赖磁盘 yaml 内容，测试更稳定。"""
    cm._models = {
        "my-opus": {"model": "claude-opus-4-8", "params": {"output_config": {"effort": "high"}}},
    }
    cm._prefixes = {
        "THINKING": {"params": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"},
        }},
    }
    cm._loaded = True


def test_load():
    """真实加载 yaml，确认 prefixes 段能被读到。"""
    load_custom_models()
    assert "THINKING" in cm._prefixes, f"expected THINKING in {list(cm._prefixes.keys())}"
    p = cm._prefixes["THINKING"]
    assert p["params"]["thinking"]["type"] == "adaptive"
    assert p["params"]["output_config"]["effort"] == "high"
    print("[PASS] test_load")


def test_prefix_match_any_model():
    """前缀规则应对任意 <PREFIX><model> 生效，无需逐个配置。"""
    _setup()
    for real in ("claude-opus-4-8", "claude-fable-5", "some-future-model"):
        body = {"model": f"THINKING:{real}", "messages": [{"role": "user", "content": "hi"}]}
        result = resolve_custom_model(body)
        assert result == real, f"got {result} for {real}"
        assert body["model"] == real
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["output_config"] == {"effort": "high"}
        assert body["messages"] == [{"role": "user", "content": "hi"}]
    print("[PASS] test_prefix_match_any_model")


def test_exact_alias_takes_precedence():
    """models 精确别名优先于前缀规则。"""
    _setup()
    body = {"model": "my-opus", "messages": []}
    result = resolve_custom_model(body)
    assert result == "claude-opus-4-8", f"got {result}"
    assert body["model"] == "claude-opus-4-8"
    assert body["output_config"] == {"effort": "high"}
    assert "thinking" not in body, "exact alias should not inject prefix params"
    print("[PASS] test_exact_alias_takes_precedence")


def test_resolve_no_match():
    _setup()
    body = {"model": "claude-sonnet-4-20250514", "messages": []}
    result = resolve_custom_model(body)
    assert result is None
    assert body["model"] == "claude-sonnet-4-20250514"
    assert "thinking" not in body
    print("[PASS] test_resolve_no_match")


def test_prefix_only_no_bare():
    """裸前缀（无真实模型名）不应匹配。"""
    _setup()
    body = {"model": "THINKING:", "messages": []}
    result = resolve_custom_model(body)
    assert result is None, f"bare prefix should not match, got {result}"
    assert body["model"] == "THINKING:"
    print("[PASS] test_prefix_only_no_bare")


def test_unknown_namespace_ignored():
    """带冒号但命名空间未注册的模型名不应被误伤（如上游自带 provider:model）。"""
    _setup()
    body = {"model": "openrouter:anthropic/claude-x", "messages": []}
    result = resolve_custom_model(body)
    assert result is None, f"unknown namespace should not match, got {result}"
    assert body["model"] == "openrouter:anthropic/claude-x"
    assert "thinking" not in body
    print("[PASS] test_unknown_namespace_ignored")


def test_no_overwrite_existing():
    """客户端已显式传入的同名参数不被注入值覆盖。"""
    _setup()
    custom_thinking = {"type": "enabled", "budget_tokens": 5000}
    body = {"model": "THINKING:claude-opus-4-8", "thinking": custom_thinking}
    resolve_custom_model(body)
    assert body["model"] == "claude-opus-4-8"
    assert body["thinking"] == custom_thinking, "should not overwrite client's explicit param"
    assert body["output_config"] == {"effort": "high"}, "should inject missing params"
    print("[PASS] test_no_overwrite_existing")


if __name__ == "__main__":
    test_load()
    test_prefix_match_any_model()
    test_exact_alias_takes_precedence()
    test_resolve_no_match()
    test_prefix_only_no_bare()
    test_unknown_namespace_ignored()
    test_no_overwrite_existing()
    print("\nAll tests passed.")
