"""test/test_custom_models.py — 自定义模型映射单元测试"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.custom_models import load_custom_models, resolve_custom_model, _models


def test_load():
    models = load_custom_models()
    assert "THINKING:claude-opus-4-8" in models, f"expected alias in {list(models.keys())}"
    cfg = models["THINKING:claude-opus-4-8"]
    assert cfg["model"] == "claude-opus-4-8"
    assert cfg["params"]["thinking"]["type"] == "adaptive"
    assert cfg["params"]["output_config"]["effort"] == "high"
    print("[PASS] test_load")


def test_resolve_match():
    load_custom_models()
    body = {"model": "THINKING:claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
    result = resolve_custom_model(body)
    assert result == "claude-opus-4-8", f"got {result}"
    assert body["model"] == "claude-opus-4-8"
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    print("[PASS] test_resolve_match")


def test_resolve_no_match():
    load_custom_models()
    body = {"model": "claude-sonnet-4-20250514", "messages": []}
    result = resolve_custom_model(body)
    assert result is None
    assert body["model"] == "claude-sonnet-4-20250514"
    assert "thinking" not in body
    print("[PASS] test_resolve_no_match")


def test_no_overwrite_existing():
    load_custom_models()
    custom_thinking = {"type": "enabled", "budget_tokens": 5000}
    body = {"model": "THINKING:claude-opus-4-8", "thinking": custom_thinking}
    resolve_custom_model(body)
    assert body["model"] == "claude-opus-4-8"
    assert body["thinking"] == custom_thinking, "should not overwrite client's explicit param"
    assert body["output_config"] == {"effort": "high"}, "should inject missing params"
    print("[PASS] test_no_overwrite_existing")


if __name__ == "__main__":
    test_load()
    test_resolve_match()
    test_resolve_no_match()
    test_no_overwrite_existing()
    print("\nAll tests passed.")
