"""
utils/connection_test.py — 连通性测试

支持两种方式：
  - 通过代理端口测试（默认）
  - --noproxy 直接测试上游渠道
支持两种协议：
  - anthropic: POST /v1/messages
  - openai:    POST /v1/chat/completions
"""

import json
import sys
import time

import httpx


DEFAULT_MODEL = "claude-sonnet-4-6"
TEST_PROMPT = "Say hi in one sentence."


def _anthropic_payload(model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
    }


def _openai_payload(model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
    }


def _extract_anthropic_text(data: dict) -> str:
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return str(data)


def _extract_openai_text(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        return str(data)
    msg = choices[0].get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if content and reasoning:
        return f"[reasoning] {reasoning}\n[content] {content}"
    if reasoning:
        return f"[reasoning] {reasoning}"
    return content or str(data)


def run_test(
    base_url: str,
    method: str,
    model: str,
    api_key: str = "",
    timeout: int = 30,
) -> dict:
    """执行单次连通测试，返回结果字典。"""
    method = method.lower()

    if method == "anthropic":
        url = f"{base_url.rstrip('/')}/v1/messages"
        payload = _anthropic_payload(model)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key or "test",
            "anthropic-version": "2023-06-01",
        }
    else:
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = _openai_payload(model)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'test'}",
        }

    result = {
        "url": url,
        "method": method,
        "model": model,
        "api_key": (api_key[:4] + "..." + api_key[-4:]) if api_key and len(api_key) > 10 else api_key or "(none)",
    }

    t0 = time.time()
    try:
        with httpx.Client(verify=False, timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
        elapsed = time.time() - t0
        result["status_code"] = resp.status_code
        result["elapsed_ms"] = int(elapsed * 1000)

        if resp.status_code == 200:
            data = resp.json()
            if method == "anthropic":
                result["reply"] = _extract_anthropic_text(data)
                result["model_used"] = data.get("model", "")
                usage = data.get("usage", {})
                result["tokens"] = f"in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')}"
            else:
                result["reply"] = _extract_openai_text(data)
                result["model_used"] = data.get("model", "")
                usage = data.get("usage", {})
                result["tokens"] = f"in={usage.get('prompt_tokens', '?')} out={usage.get('completion_tokens', '?')}"
            result["ok"] = True
        else:
            try:
                result["error"] = resp.text[:300]
            except Exception:
                result["error"] = f"HTTP {resp.status_code}"
            result["ok"] = False
    except Exception as e:
        elapsed = time.time() - t0
        result["elapsed_ms"] = int(elapsed * 1000)
        result["error"] = str(e)
        result["ok"] = False

    return result


def print_result(r: dict) -> None:
    status = "OK" if r.get("ok") else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"  [{status}] {r['method'].upper()} -> {r['url']}")
    print(f"  Model: {r['model']}  Key: {r['api_key']}")
    if "status_code" in r:
        print(f"  Status: {r['status_code']}  Time: {r['elapsed_ms']}ms")
    if r.get("ok"):
        print(f"  Model used: {r.get('model_used', '?')}")
        print(f"  Tokens: {r.get('tokens', '?')}")
        print(f"  Reply: {(r.get('reply') or '')[:200]}")
    else:
        print(f"  Error: {r.get('error', 'unknown')}")
    print(f"{'=' * 60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM proxy connection test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to test (default: {DEFAULT_MODEL})")
    parser.add_argument("--method", default="anthropic", choices=["anthropic", "openai"], help="API method (default: anthropic)")
    parser.add_argument("--host", default="127.0.0.1", help="Proxy host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=4000, help="Proxy port (default: 4000)")
    parser.add_argument("--key", default="", help="API key for the request")
    parser.add_argument("--noproxy", action="store_true", help="Test upstream directly (reads UPSTREAM_URL and UPSTREAM_API_KEY from env)")
    parser.add_argument("--upstream-url", default="", help="Upstream URL (used with --noproxy)")
    parser.add_argument("--upstream-key", default="", help="Upstream API key (used with --noproxy)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    args = parser.parse_args()

    import os
    if args.noproxy:
        base_url = args.upstream_url or os.getenv("UPSTREAM_URL", "")
        api_key = args.upstream_key or os.getenv("UPSTREAM_API_KEY", "").strip().strip('"').split(",")[0].strip()
        if not base_url:
            print("[connect] --noproxy requires UPSTREAM_URL env or --upstream-url", file=sys.stderr)
            sys.exit(1)
        # 上游是 OpenAI 兼容的
        base_url = base_url.rstrip("/").removesuffix("/v1")
        print(f"[connect] direct upstream: {base_url}")
    else:
        base_url = f"http://{args.host}:{args.port}"
        api_key = args.key
        print(f"[connect] via proxy: {base_url}")

    print(f"[connect] method={args.method} model={args.model}")

    result = run_test(base_url, args.method, args.model, api_key, args.timeout)
    print_result(result)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
