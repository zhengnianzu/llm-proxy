"""
test/test_hermes_anthropic.py — hermes_traj 双格式（Anthropic SSE + OpenAI SSE）验证

真实 new-api 合并文件（如 jumper-001 源的 260729 叶子）响应侧 ~94% 是
Anthropic SSE（thinking_delta / input_json_delta），请求历史 assistant 消息
是 Anthropic block 形态（thinking/text/tool_use block 列表）；GLM 等模型
则是 OpenAI SSE + OpenAI 形态。本测试覆盖：

  1) assemble_anthropic_stream：thinking/text/tool_use（input_json_delta）
     装配到内部形态，OpenAI 装配器对其返回 None，反之亦然；
  2) 非流式单对象 Anthropic 响应；
  3) assistant_signature：block 形态与归一化形态签名相等（thinking 不参与）；
  4) canonical_message：剥 content 里的 thinking block；
  5) backfill_messages：OpenAI 形态写 reasoning_content、block 形态插 thinking；
  6) process_directory 端到端：重试去重（同签名择优带 reasoning 的）、
     回填、manifest 审计。

运行：.venv/bin/python -m pytest test/test_hermes_anthropic.py -q
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export.hermes_traj import (  # noqa: E402
    ResponseSource,
    assemble_anthropic_stream,
    assemble_streamed_message,
    assistant_signature,
    backfill_messages,
    canonical_message,
    process_directory,
)


def anthropic_sse(thinking, text, tool_use=None, sig="sig-abc-123"):
    """构造一条 Anthropic SSE 响应（thinking 拆两段 delta + signature_delta）。"""
    lines = [
        "event: message_start",
        "data: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {"type": "message", "role": "assistant", "content": [], "stop_reason": None},
            }
        )
        + " ",
        "",
    ]
    idx = 0
    if thinking is not None:
        lines += [
            "event: content_block_start",
            "data: "
            + json.dumps(
                {"type": "content_block_start", "index": idx, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}
            )
            + " ",
        ]
        half = len(thinking) // 2
        for frag in (thinking[:half], thinking[half:]):
            lines += [
                "event: content_block_delta",
                "data: "
                + json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "thinking_delta", "thinking": frag}})
                + " ",
            ]
        if sig:
            lines += [
                "event: content_block_delta",
                "data: "
                + json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "signature_delta", "signature": sig}})
                + " ",
            ]
        lines += ["event: content_block_stop", "data: " + json.dumps({"type": "content_block_stop", "index": idx}) + " "]
        idx += 1
    lines += [
        "event: content_block_start",
        "data: "
        + json.dumps({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
        + " ",
        "event: content_block_delta",
        "data: "
        + json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": text}})
        + " ",
        "event: content_block_stop",
        "data: " + json.dumps({"type": "content_block_stop", "index": idx}) + " ",
    ]
    idx += 1
    if tool_use:
        name, args_json = tool_use
        lines += [
            "event: content_block_start",
            "data: "
            + json.dumps(
                {"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": "toolu_01", "name": name, "input": {}}}
            )
            + " ",
        ]
        half = len(args_json) // 2
        for frag in (args_json[:half], args_json[half:]):
            lines += [
                "event: content_block_delta",
                "data: "
                + json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": frag}})
                + " ",
            ]
        lines += ["event: content_block_stop", "data: " + json.dumps({"type": "content_block_stop", "index": idx}) + " "]
    lines += ["event: message_stop", "data: " + json.dumps({"type": "message_stop"}) + " "]
    return "\n".join(lines)


def merged_file(ts, req_messages, resp_raw):
    return {
        "ts": ts,
        "rid": "r-" + ts,
        "uid": "u1",
        "model": "claude-opus-4-8",
        "api_key": "sk-x",
        "usage": {"token_in": 1, "token_out": 1},
        "req": json.dumps({"model": "claude-opus-4-8", "messages": req_messages}),
        "resp": resp_raw,
        "up_req": json.dumps({"model": "claude-opus-4-8", "messages": req_messages}),
        "up_resp": resp_raw,
    }


def test_anthropic_sse_assembly():
    raw = anthropic_sse("think-about-this", "visible text", ("search", '{"q":"paris"}'))
    msg = assemble_anthropic_stream(raw)
    assert msg["content"] == "visible text"
    assert msg["reasoning_content"] == "think-about-this"
    assert msg["thinking_signature"] == "sig-abc-123"
    assert msg["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"q": "paris"}
    # 交叉识别：Anthropic 装配器不认 OpenAI，反之亦然
    assert assemble_streamed_message(raw) is None
    openai_raw = "data: " + json.dumps({"choices": [{"delta": {"content": "hi", "reasoning_content": "r1"}}]}) + "\n\n"
    assert assemble_anthropic_stream(openai_raw) is None
    assert assemble_streamed_message(openai_raw)["content"] == "hi"
    # 非流式单对象
    nonstream = json.dumps(
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t1", "signature": "s1"},
                {"type": "text", "text": "plain"},
            ],
        }
    )
    ns = assemble_anthropic_stream(nonstream)
    assert ns["reasoning_content"] == "t1" and ns["content"] == "plain" and ns["thinking_signature"] == "s1"


def test_signature_and_canonical_are_reasoning_independent():
    block_msg = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "whatever-reasoning"},
            {"type": "text", "text": "visible text"},
            {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"q": "paris"}},
        ],
    }
    norm_msg = {
        "role": "assistant",
        "content": "visible text",
        "reasoning_content": "think-about-this",
        "thinking_signature": "sig-abc-123",
        "tool_calls": [{"id": "toolu_01", "type": "function", "function": {"name": "search", "arguments": '{"q":"paris"}'}}],
    }
    assert assistant_signature(block_msg) == assistant_signature(norm_msg)
    # canonical 剥 thinking：有无 thinking block 的同一消息序列化相等
    c1 = canonical_message(block_msg)
    c2 = canonical_message(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "visible text"},
                {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"q": "paris"}},
            ],
        }
    )
    assert c1 == c2


def test_backfill_both_shapes():
    registry = {assistant_signature({
        "role": "assistant",
        "content": "visible text",
        "tool_calls": [{"id": "toolu_01", "type": "function", "function": {"name": "search", "arguments": '{"q":"paris"}'}}],
    }): [ResponseSource(0, "think-about-this", "sig-abc-123")]}

    # OpenAI 形态 → 写 reasoning_content 键
    openai_target = {"role": "assistant", "content": "visible text",
                     "tool_calls": [{"id": "toolu_01", "type": "function", "function": {"name": "search", "arguments": '{"q":"paris"}'}}]}
    assert backfill_messages([openai_target], registry) == 1
    assert openai_target["reasoning_content"] == "think-about-this"

    # Anthropic block 形态 → 插 thinking block（在首个非 thinking block 前）
    block_target = {"role": "assistant", "content": [
        {"type": "text", "text": "visible text"},
        {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"q": "paris"}},
    ]}
    assert backfill_messages([block_target], registry) == 1
    assert block_target["content"][0] == {"type": "thinking", "thinking": "think-about-this", "signature": "sig-abc-123"}
    # 已有 reasoning 的不重复回填
    assert backfill_messages([openai_target], registry) == 0
    assert backfill_messages([block_target], registry) == 0


def test_end_to_end_selection_dedup_and_backfill():
    hist_a = [{"role": "user", "content": "hi there"}]
    resp_a = anthropic_sse("thinking-0", "hello")
    resp_retry = anthropic_sse(None, "hello")  # 同一响应但丢了 reasoning
    hist_b = [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        {"role": "user", "content": "what is paris"},
    ]
    resp_b = anthropic_sse("thinking-paris", "Paris lies at 48.85N")

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "leaf"
        src.mkdir()
        out = Path(td) / "out"
        (src / "a.json").write_text(json.dumps(merged_file("2026-07-31_15-00-00_000-1", hist_a, resp_a)))
        (src / "a-retry.json").write_text(json.dumps(merged_file("2026-07-31_15-00-01_000-2", hist_a, resp_retry)))
        (src / "b.json").write_text(json.dumps(merged_file("2026-07-31_15-00-02_000-3", hist_b, resp_b)))

        files, runs, req_fill, up_fill = process_directory(src, out)
        assert files == 2, files  # retry 与 a 同签名 → 择优去重；a + b
        assert runs == 2, runs
        assert req_fill == 1 and up_fill == 1, (req_fill, up_fill)  # b 的 assistant 历史缺 thinking

        names = sorted(p.name for p in out.glob("*.json") if p.name != "_manifest.jsonl")
        assert names == ["a.json", "b.json"], names  # 选带 reasoning 的 a，而非 retry

        b_out = json.loads((out / "b.json").read_text())
        req_decoded = json.loads(b_out["req"])
        asst = [m for m in req_decoded["messages"] if m.get("role") == "assistant"][0]
        assert asst["content"][0] == {"type": "thinking", "thinking": "thinking-0", "signature": "sig-abc-123"}
        assert asst["content"][1] == {"type": "text", "text": "hello"}

        manifest = [json.loads(line) for line in (out / "_manifest.jsonl").read_text().splitlines()]
        assert manifest[0]["type"] == "summary" and manifest[0]["selected_file_count"] == 2
        runs_m = [r for r in manifest if r["type"] == "run"]
        assert len(runs_m) == 2
        backfills = [r["reasoning_backfill"] for r in runs_m]
        assert any("b.json" in e and e["b.json"]["req"] == 1 and e["b.json"]["up_req"] == 1 for e in backfills)


if __name__ == "__main__":
    test_anthropic_sse_assembly()
    test_signature_and_canonical_are_reasoning_independent()
    test_backfill_both_shapes()
    test_end_to_end_selection_dedup_and_backfill()
    print("ALL PASSED")
