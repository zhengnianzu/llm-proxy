#!/usr/bin/env python3
"""把 CoT（reasoning content）回填到 newapi 会话文件里，供 /export/view 浏览时查看。

背景
----
某些模型（如 gemini-4-argon）的响应里不直接带思维链，而是留一个引用：

    "cot_task_id": "20261119T025957Z-d2a3a5522c569cc2dfe9ae3e1d350222",
    "cot_url": "/v1/cot/20261119T025957Z-d2a3a5522c569cc2dfe9ae3e1d350222",
    "cot_status": "queued", ...

真正的 reasoning 内容要去 CoT 服务取：

    GET https://api.loopsforge.ai/cot-api/v1/cot/<cot_task_id>
    Authorization: Bearer <token>

本脚本把取回的 content 写进响应里 assistant 消息的 reasoning_content 字段。
前端 static/js/chat-render.js 会把 reasoning_content 渲染成可折叠的 thinking 块
（见 chat-render.js:484 与 blocks 里的 {"type":"thinking"} 分支），所以写进去就能
直接在浏览页展开查看。

写回格式
--------
原文件的 resp 可能是两种形态，分别处理：
  * 流式 SSE（以 "data: " 开头的多行）：在首个带 choices 的 chunk 之后插入一个
    {"choices":[{"index":0,"delta":{"reasoning_content":"..."}}]}，解析器
    parse_openai_response_content 会逐 chunk 累加 delta.reasoning_content。
  * 非流式整块 JSON：直接写 choices[0].message.reasoning_content。

用法
----
  # 预览（默认，不写入）。--session 支持文件名或 ISO 时间（与页面 URL 的 session 参数一致）
  python fill_cot_reasoning.py --key sk-xxx --date 261119 --leaf 26111911 \
      --session 2026-11-19T11:00:52.797+08:00

  # 实际写入
  python fill_cot_reasoning.py ... --write

  # token 建议走环境变量，避免留在 shell history 里
  export COT_API_TOKEN=sk-...
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

DETAILS_ROOT = (
    "/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/raw/harness/"
    "proxy-004/data/newapi/logs/details"
)
COT_API_BASE = "https://api.loopsforge.ai/cot-api/v1/cot/"
# 默认 token 从环境变量取；不建议写死在命令行（会进 shell history）。
DEFAULT_TOKEN = os.environ.get("COT_API_TOKEN", "")

_COT_ID_RE = re.compile(r'"cot_task_id"\s*:\s*"([^"]+)"')
_COT_STATUS_RE = re.compile(r'"cot_status"\s*:\s*"([^"]+)"')


# ────────────────────────────── CoT API ──────────────────────────────

def fetch_cot(task_id: str, token: str, timeout: float = 30.0,
              retries: int = 2) -> Dict[str, Any]:
    """取一个 CoT 任务。返回 {"ok":bool, "content":str, "raw":dict, "error":str}。"""
    url = COT_API_BASE + task_id
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    last_err = ""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", "replace")
            data = json.loads(body)
        except urllib.error.HTTPError as e:
            # 404 是「任务不存在/已过期」，重试无意义
            body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
            try:
                j = json.loads(body)
            except Exception:
                j = {}
            return {"ok": False, "content": "", "raw": j,
                    "error": j.get("error") or f"HTTP {e.code}"}
        except Exception as e:  # noqa: BLE001 — 网络抖动可重试
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return {"ok": False, "content": "", "raw": {}, "error": last_err}

        if not isinstance(data, dict):
            return {"ok": False, "content": "", "raw": {}, "error": "响应不是 JSON 对象"}
        if data.get("error"):
            return {"ok": False, "content": "", "raw": data, "error": str(data["error"])}
        content = _extract_content(data)
        if not content:
            return {"ok": False, "content": "", "raw": data,
                    "error": f"响应里没有 content（status={data.get('status')}）"}
        return {"ok": True, "content": content, "raw": data, "error": ""}
    return {"ok": False, "content": "", "raw": {}, "error": last_err}


def _extract_content(data: Dict[str, Any]) -> str:
    """从 CoT 响应里取出正文。兼容几种常见形态。"""
    c = data.get("content")
    if isinstance(c, str) and c.strip():
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or "")
        joined = "".join(parts)
        if joined.strip():
            return joined
    if isinstance(c, dict):
        for k in ("text", "content", "reasoning", "reasoning_content"):
            v = c.get(k)
            if isinstance(v, str) and v.strip():
                return v
    # 兜底：data.content / reasoning / reasoning_content
    for k in ("reasoning", "reasoning_content", "text"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v
    inner = data.get("data")
    if isinstance(inner, dict):
        return _extract_content(inner)
    return ""


# ────────────────────────── 文件写入（注入 reasoning）──────────────────────────

def inject_reasoning(resp: str, reasoning: str) -> Optional[str]:
    """把 reasoning 注入 resp。返回新的 resp 文本；无改动返回 None。

    流式：在首个带 choices 的 chunk 之后插入一个新的 delta chunk。
      解析器（utils/message_common.parse_openai_response_content）逐 chunk 累加
      delta.reasoning_content，插入的顺序不影响最终拼接结果。
    非流式：写到 choices[0].message.reasoning_content。
    """
    if not isinstance(resp, str) or not resp.strip():
        return None
    if not reasoning:
        return None
    # 已经注入过就不要再插一个 delta —— 注入是加数据，重复调用会叠加。
    if '"reasoning_content"' in resp:
        return None

    stripped = resp.lstrip()
    # ── 非流式整块 JSON ──
    if not stripped.startswith("data:"):
        try:
            obj = json.loads(resp)
        except json.JSONDecodeError:
            return None
        choices = obj.get("choices") if isinstance(obj, dict) else None
        if not choices:
            return None
        msg = (choices[0] or {}).get("message")
        if not isinstance(msg, dict):
            return None
        if msg.get("reasoning_content"):
            return None            # 已有内容，不覆盖
        msg["reasoning_content"] = reasoning
        return json.dumps(obj, ensure_ascii=False)

    # ── 流式 SSE ──
    lines = resp.split("\n")
    pulse = json.dumps(
        {"choices": [{"index": 0, "delta": {"reasoning_content": reasoning}}]},
        ensure_ascii=False,
    )
    new_lines: List[str] = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if inserted or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("choices"):
            # 已有 reasoning_content 就别重复注入
            for c in obj.get("choices") or []:
                d = (c or {}).get("delta") or {}
                if d.get("reasoning_content"):
                    return None
            new_lines.append("data: " + pulse)
            inserted = True
    if not inserted:
        return None
    return "\n".join(new_lines)


# ────────────────────────────── 取文件列表 ──────────────────────────────

def resolve_session_files(details_root: str, date_dir: str, leaf: str,
                          key: str, session: str = "") -> List[str]:
    """按 date/leaf + api_key 取会话文件。

    session 为空时返回该 key 在该叶子的全部文件；给了 session 则：
      * 形如 2026-11-19T11:00:52.797+08:00 → 用 ts 前缀匹配
      * 形如 xxx.json → 直接定位该文件
    """
    leaf_dir = os.path.join(details_root, date_dir, leaf)
    db = os.path.join(leaf_dir, "index.db")
    if not os.path.exists(db):
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT req_file, ts FROM requests WHERE api_key=? ORDER BY ts ASC", (key,)
        ).fetchall()
    finally:
        conn.close()

    out: List[str] = []
    for r in rows:
        fn = (r["req_file"] or "").split("/")[-1]
        if not fn:
            continue
        if session:
            if session.endswith(".json"):
                if fn != session:
                    continue
            else:
                # ISO 时间与文件 ts 比较：按前 19 位（到秒）比对
                if (r["ts"] or "")[:19] != session[:19]:
                    continue
        out.append(os.path.join(leaf_dir, fn))
    return out


# ────────────────────────────── 主流程 ──────────────────────────────

def process_file(path: str, token: str, write: bool, force: bool) -> tuple:
    """返回 (state, detail)。state ∈ {'fixed','skip','error'}。"""
    name = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return ("error", f"{name}: 读取失败 {type(e).__name__}: {e}")
    if not isinstance(data, dict):
        return ("error", f"{name}: 顶层不是对象")

    resp = data.get("resp", "")
    if isinstance(resp, str) and '"reasoning_content"' in resp:
        if not force:
            return ("skip", f"{name}: 已有 reasoning_content")

    ids = _COT_ID_RE.findall(resp if isinstance(resp, str) else "")
    if not ids:
        return ("skip", f"{name}: 无 cot_task_id")
    task_id = ids[0]

    got = fetch_cot(task_id, token)
    if not got["ok"]:
        return ("error", f"{name}: CoT 取回失败 [{task_id}] {got['error']}")

    new_resp = inject_reasoning(resp, got["content"])
    if new_resp is None:
        return ("skip", f"{name}: 注入失败（resp 形态未识别或已有内容）")

    statuses = _COT_STATUS_RE.findall(resp)
    status = statuses[0] if statuses else ""
    preview = got["content"][:60].replace("\n", " ")

    if write:
        data["resp"] = new_resp
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    return ("fixed", f"{name}: +{len(got['content'])} 字 (cot_status={status}) 「{preview}…」")


def main() -> int:
    ap = argparse.ArgumentParser(description="把 CoT reasoning 回填到会话文件")
    ap.add_argument("--key", required=True, help="api_key（页面 URL 的 key 参数）")
    ap.add_argument("--date", required=True, help="日期目录，如 261119")
    ap.add_argument("--leaf", required=True, help="小时叶子，如 26111911")
    ap.add_argument("--session", default="", help="session 时间或文件名；空=该叶子全部文件")
    ap.add_argument("--write", action="store_true", help="实际写入（默认仅预览）")
    ap.add_argument("--force", action="store_true", help="已有 reasoning_content 也覆盖")
    ap.add_argument("--token", default=DEFAULT_TOKEN, help="CoT API token（默认取 $COT_API_TOKEN）")
    ap.add_argument("--root", default=DETAILS_ROOT, help="details 根目录")
    args = ap.parse_args()

    if not args.token:
        print("错误：缺少 CoT API token。用 --token 或 export COT_API_TOKEN=...", file=sys.stderr)
        return 2

    files = resolve_session_files(args.root, args.date, args.leaf, args.key, args.session)
    if not files:
        sess = args.session or "*"
        print(f"没有匹配到文件（date={args.date} leaf={args.leaf} session={sess}）")
        return 1

    print("=== WRITE 模式 ===" if args.write else "=== DRY RUN（不写入）===")
    print(f"文件数: {len(files)}\n")

    n_fixed = n_skip = n_err = 0
    for p in files:
        state, detail = process_file(p, args.token, args.write, args.force)
        if state == "fixed":
            n_fixed += 1
            print(f"  ✓ {detail}")
        elif state == "skip":
            n_skip += 1
            print(f"  - {detail}")
        else:
            n_err += 1
            print(f"  ✗ {detail}")

    print(f"\n合计: 可回填/已回填 {n_fixed}  跳过 {n_skip}  失败 {n_err}")
    if not args.write:
        print("（dry-run，未写入；加 --write 实际执行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
