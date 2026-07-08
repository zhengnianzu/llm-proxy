#!/usr/bin/env python3
"""
token 统计校准测试脚本

用法: python3 test_token_calibration.py <api_key> [proxy_host]

发 20 次请求（非流式 + 流式各 5 次 × 2 个 provider），
记录每次请求的 usage 原始数据，输出到 CSV 文件。
"""

import json
import sys
import time
import csv
import os
import http.client
from urllib.parse import urlparse

PROXY_HOST = "http://127.0.0.1:4000"

MODEL_ANTHROPIC = "claude-sonnet-4-6"
MODEL_OPENAI = "deepseek-v4-flash"


def _post(url: str, headers: dict, body: dict):
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    data = json.dumps(body).encode()
    conn.request("POST", parsed.path, body=data, headers=headers)
    resp = conn.getresponse()
    return resp, conn


def call_anthropic(api_key: str, prompt: str):
    url = f"{PROXY_HOST}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    body = {"model": MODEL_ANTHROPIC, "max_tokens": 50, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp, conn = _post(url, headers, body)
        data = json.loads(resp.read().decode())
        conn.close()
        usage = data.get("usage", {})
        return {
            "model": MODEL_ANTHROPIC, "mode": "非流式", "ok": resp.status == 200,
            "tok_in": usage.get("input_tokens", 0), "tok_out": usage.get("output_tokens", 0),
            "usage": usage,
        }
    except Exception as e:
        return {"model": MODEL_ANTHROPIC, "mode": "非流式", "ok": False, "tok_in": 0, "tok_out": 0, "usage": {}, "error": str(e)}


def call_anthropic_stream(api_key: str, prompt: str):
    url = f"{PROXY_HOST}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    body = {"model": MODEL_ANTHROPIC, "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp, conn = _post(url, headers, body)
        usage_all = {}
        buf = ""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                u = evt.get("message", {}).get("usage") or evt.get("usage") or {}
                if u:
                    usage_all.update(u)
        conn.close()
        return {
            "model": MODEL_ANTHROPIC, "mode": "流式", "ok": resp.status == 200,
            "tok_in": usage_all.get("input_tokens") or 0,
            "tok_out": usage_all.get("output_tokens") or 0,
            "usage": usage_all,
        }
    except Exception as e:
        return {"model": MODEL_ANTHROPIC, "mode": "流式", "ok": False, "tok_in": 0, "tok_out": 0, "usage": {}, "error": str(e)}


def call_openai(api_key: str, prompt: str):
    url = f"{PROXY_HOST}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": MODEL_OPENAI, "max_tokens": 50, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp, conn = _post(url, headers, body)
        data = json.loads(resp.read().decode())
        conn.close()
        usage = data.get("usage", {})
        return {
            "model": MODEL_OPENAI, "mode": "非流式", "ok": resp.status == 200,
            "tok_in": usage.get("prompt_tokens", 0), "tok_out": usage.get("completion_tokens", 0),
            "usage": usage,
        }
    except Exception as e:
        return {"model": MODEL_OPENAI, "mode": "非流式", "ok": False, "tok_in": 0, "tok_out": 0, "usage": {}, "error": str(e)}


def call_openai_stream(api_key: str, prompt: str):
    url = f"{PROXY_HOST}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": MODEL_OPENAI, "max_tokens": 50, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    try:
        resp, conn = _post(url, headers, body)
        usage_all = {}
        buf = ""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                u = evt.get("usage") or {}
                if u:
                    usage_all.update(u)
        conn.close()
        return {
            "model": MODEL_OPENAI, "mode": "流式", "ok": resp.status == 200,
            "tok_in": usage_all.get("prompt_tokens") or 0,
            "tok_out": usage_all.get("completion_tokens") or 0,
            "usage": usage_all,
        }
    except Exception as e:
        return {"model": MODEL_OPENAI, "mode": "流式", "ok": False, "tok_in": 0, "tok_out": 0, "usage": {}, "error": str(e)}


def format_usage(usage: dict) -> str:
    if not usage:
        return ""
    parts = []
    for k, v in sorted(usage.items()):
        if k == "total_tokens":
            continue
        if isinstance(v, dict):
            sub = ", ".join(f"{sk}={sv}" for sk, sv in sorted(v.items()))
            parts.append(f"{k}={{{sub}}}")
        elif v:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else ""


def write_csv(results, filepath):
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "测试模型", "输出形式", "输入token", "输出token", "usage"])
        for i, r in enumerate(results, 1):
            writer.writerow([
                i, r["model"], r["mode"], r["tok_in"], r["tok_out"],
                format_usage(r["usage"]),
            ])
        total_in = sum(r["tok_in"] for r in results)
        total_out = sum(r["tok_out"] for r in results)
        writer.writerow(["", "合计", "", total_in, total_out, ""])


def main():
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} <api_key> [proxy_host]")
        sys.exit(1)

    api_key = sys.argv[1]
    global PROXY_HOST
    if len(sys.argv) > 2:
        PROXY_HOST = sys.argv[2].rstrip("/")

    print(f"Proxy: {PROXY_HOST}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print()

    prompts = [
        "Say hello in one word.",
        "What is 2+2? Answer with just the number.",
        "Name one color.",
        "Say yes or no: is the sky blue?",
        "What is the capital of France? One word.",
    ]

    calls = [
        (call_anthropic, "Anthropic 非流式"),
        (call_anthropic_stream, "Anthropic 流式"),
        (call_openai, "OpenAI 非流式"),
        (call_openai_stream, "OpenAI 流式"),
    ]

    results = []
    for fn, label in calls:
        print(f"=== {label} (5 次) ===")
        for i, p in enumerate(prompts):
            r = fn(api_key, p)
            status = "OK" if r["ok"] else "ERR"
            print(f"  [{i+1}] {status} tok_in={r['tok_in']}, tok_out={r['tok_out']}")
            results.append(r)
            time.sleep(0.5)
        print()

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"calibration_{api_key[-4:]}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    write_csv(results, csv_path)
    print(f"测试结果已写入: {csv_path}")


if __name__ == "__main__":
    main()
