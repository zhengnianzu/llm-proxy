"""
/logs/* 路由：列表、聚合、单文件读取（Anthropic + OpenAI）
在 app.py 里调用 register_log_routes(app) 即可挂载。
"""

import glob
import json
import os
import re as _re
from collections import OrderedDict

from fastapi import FastAPI
from fastapi.responses import JSONResponse

LOGS_ANTHROPIC = "logs_anthropic"
LOGS_OPENAI = "logs_openai"


def register_log_routes(app: FastAPI) -> None:

    # ------------------------------------------------------------------ #
    #  Anthropic                                                           #
    # ------------------------------------------------------------------ #

    @app.get("/logs/anthropic/list")
    def logs_anthropic_list(min_messages: int = 10):
        result = []
        pattern = os.path.join(LOGS_ANTHROPIC, "*-req.json")
        for path in sorted(glob.glob(pattern), reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                messages = data.get("messages")
                if not isinstance(messages, list) or len(messages) <= min_messages:
                    continue
                result.append({
                    "filename": os.path.basename(path),
                    "message_count": len(messages),
                    "model": data.get("model", ""),
                })
            except Exception:
                continue
        return JSONResponse(result)

    @app.get("/logs/anthropic/file")
    def logs_anthropic_file(filename: str):
        if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = os.path.join(LOGS_ANTHROPIC, filename)
        if not os.path.isfile(path):
            return JSONResponse({"error": "file not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(data)

    @app.get("/logs/anthropic/aggregate")
    def logs_anthropic_aggregate(min_messages: int = 1):
        ts_pat = _re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+)-req\.json$")

        def extract_res_content(res_path: str):
            if not os.path.isfile(res_path):
                return None
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict) and "json" in d:
                    j = d["json"]
                    if isinstance(j, dict) and j.get("role") == "assistant":
                        return j.get("content")
                if isinstance(d, list):
                    for chunk in reversed(d):
                        msg = chunk.get("message", {})
                        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content") is not None:
                            return msg["content"]
            except Exception:
                pass
            return None

        def get_q1_full(msgs):
            if not msgs:
                return ""
            c = msgs[0].get("content", "")
            if isinstance(c, list):
                c = "|".join(b.get("text") or b.get("id") or str(b)[:200] for b in c if isinstance(b, dict))
            return str(c)

        req_files = sorted(glob.glob(os.path.join(LOGS_ANTHROPIC, "*-req.json")))
        entries = []
        for path in req_files:
            m = ts_pat.search(os.path.basename(path))
            if not m:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            msgs = data.get("messages")
            if not isinstance(msgs, list) or len(msgs) < min_messages:
                continue
            entries.append({
                "ts": m.group(1),
                "path": path,
                "res_path": path.replace("-req.json", "-res.json"),
                "messages": msgs,
                "model": data.get("model", ""),
            })

        if not entries:
            return JSONResponse([])

        chains_map: OrderedDict = OrderedDict()
        for entry in entries:
            key = get_q1_full(entry["messages"])
            if key not in chains_map:
                chains_map[key] = {"chain_id": len(chains_map), "messages": entry["messages"],
                                   "entries": [entry], "model": entry["model"]}
            else:
                chain = chains_map[key]
                if len(entry["messages"]) > len(chain["messages"]):
                    chain["messages"] = entry["messages"]
                chain["entries"].append(entry)

        result = []
        for chain in chains_map.values():
            best = max(chain["entries"], key=lambda e: len(e["messages"]))
            res_content = extract_res_content(best["res_path"])
            full_messages = list(chain["messages"])
            if res_content is not None:
                full_messages.append({
                    "role": "assistant", "content": res_content,
                    "_from_res": True, "_source_file": os.path.basename(best["res_path"]),
                })
            first_ts, last_ts = chain["entries"][0]["ts"], chain["entries"][-1]["ts"]
            result.append({
                "chain_id": chain["chain_id"],
                "first_time": first_ts.replace("_", " ", 1).replace("_", ".").replace("-", ":"),
                "last_time": last_ts.replace("_", " ", 1).replace("_", ".").replace("-", ":"),
                "file_count": len(chain["entries"]),
                "message_count": len(full_messages),
                "model": chain["model"],
                "messages": full_messages,
            })

        result.sort(key=lambda x: x["first_time"], reverse=True)
        return JSONResponse(result)

    # ------------------------------------------------------------------ #
    #  OpenAI                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/logs/openai/list")
    def logs_openai_list(min_messages: int = 10):
        result = []
        pattern = os.path.join(LOGS_OPENAI, "*-req.json")
        for path in sorted(glob.glob(pattern), reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                messages = data.get("messages")
                if not isinstance(messages, list) or len(messages) <= min_messages:
                    continue
                result.append({
                    "filename": os.path.basename(path),
                    "message_count": len(messages),
                    "model": data.get("model", ""),
                })
            except Exception:
                continue
        return JSONResponse(result)

    @app.get("/logs/openai/file")
    def logs_openai_file(filename: str):
        if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = os.path.join(LOGS_OPENAI, filename)
        if not os.path.isfile(path):
            return JSONResponse({"error": "file not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(data)

    @app.get("/logs/openai/aggregate")
    def logs_openai_aggregate(min_messages: int = 1):
        ts_pat = _re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+)-req\.json$")

        def get_oai_chain_key(msgs):
            for m in msgs:
                if m.get("role") == "system":
                    match = _re.search(r"# Origin_query\s*\n+(.*?)(\n---|\Z)", str(m.get("content", "")), _re.DOTALL)
                    if match:
                        return "oq:" + match.group(1).strip()
            for m in msgs:
                if m.get("role") == "user":
                    c = m.get("content", "")
                    if isinstance(c, list):
                        c = "|".join(b.get("text", "") for b in c if isinstance(b, dict))
                    return "u:" + str(c)
            return str(msgs[0].get("content", ""))

        def extract_oai_res_content(res_path):
            if not os.path.isfile(res_path):
                return None
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                msg = d.get("json", {}).get("choices", [{}])[0].get("message")
                if msg and msg.get("role") == "assistant":
                    return msg
            except Exception:
                pass
            return None

        req_files = sorted(glob.glob(os.path.join(LOGS_OPENAI, "*-req.json")))
        entries = []
        for path in req_files:
            m = ts_pat.search(os.path.basename(path))
            if not m:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            msgs = data.get("messages")
            if not isinstance(msgs, list) or len(msgs) < min_messages:
                continue
            entries.append({
                "ts": m.group(1),
                "path": path,
                "res_path": path.replace("-req.json", "-res.json"),
                "messages": msgs,
                "model": data.get("model", ""),
            })

        if not entries:
            return JSONResponse([])

        chains_map: OrderedDict = OrderedDict()
        for entry in entries:
            key = get_oai_chain_key(entry["messages"])
            if key not in chains_map:
                chains_map[key] = {"chain_id": len(chains_map), "chain_key": key,
                                   "messages": entry["messages"], "entries": [entry], "model": entry["model"]}
            else:
                chain = chains_map[key]
                if len(entry["messages"]) > len(chain["messages"]):
                    chain["messages"] = entry["messages"]
                chain["entries"].append(entry)

        result = []
        for chain in chains_map.values():
            best = max(chain["entries"], key=lambda e: len(e["messages"]))
            res_msg = extract_oai_res_content(best["res_path"])
            full_messages = list(chain["messages"])
            if res_msg is not None:
                full_messages.append({**res_msg, "_from_res": True,
                                      "_source_file": os.path.basename(best["res_path"])})
            first_ts, last_ts = chain["entries"][0]["ts"], chain["entries"][-1]["ts"]
            result.append({
                "chain_id": chain["chain_id"],
                "first_time": first_ts.replace("_", " ", 1).replace("_", ".").replace("-", ":"),
                "last_time": last_ts.replace("_", " ", 1).replace("_", ".").replace("-", ":"),
                "file_count": len(chain["entries"]),
                "message_count": len(full_messages),
                "model": chain["model"],
                "messages": full_messages,
                "chain_key": chain["chain_key"],
            })

        result.sort(key=lambda x: x["first_time"], reverse=True)
        return JSONResponse(result)
