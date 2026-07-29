"""
utils/newapi_consumer.py — new-api 合并文件 → 会话聚合

把 new-api 叶子目录（{天}/{小时}）的合并文件消费成会话，写入 session_cache.db，
复用现有历史预览 / 导出链路。会话归并逻辑对齐 log_routes._process_req_row：
  lookup_key = api_key || q1_hash；latest_file 取消息最多/带响应的那次；
  user_turns 回落 ≤1 视作新会话，另起 ##session_N 后缀。

两个入口：
  - aggregate_leaf(leaf): 纯函数，不碰 DB，返回 sessions 列表（供多进程 worker）
  - consume_leaf(leaf):   增量消费（用 index_progress byte_offset），逐条写 DB（供实时 refresh）
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.newapi_format import parse_combined_file, compute_session_fields

_INDEX_NAME = "index.jsonl"


def _resolve_combined_path(leaf: Path, req_file: str) -> Optional[Path]:
    """index 行的 req_file 定位到合并文件。

    new-api 的 req_file 形如 '26072717/2026-..-1048577.json'（相对天目录），
    也可能是纯文件名。优先在叶子目录内按 basename 找。
    """
    raw = (req_file or "").strip()
    if not raw:
        return None
    name = os.path.basename(raw)
    cand = leaf / name
    if cand.is_file():
        return cand
    # 退回：相对叶子父目录
    cand2 = (leaf.parent / raw)
    if cand2.is_file():
        return cand2
    return None


def _new_session(sess_key: str, api_key: str, model: str, ts: str,
                 fields: Dict[str, Any], filename: str) -> Dict[str, Any]:
    return {
        "_key": sess_key,
        "api_key": api_key,
        "q1": fields.get("q1_preview") or fields.get("chain_key", "")[:200],
        "models": [model] if model else [],
        "latest_file": filename,
        "msg_count": fields.get("msg_count", 0),
        "first_ts": ts,
        "last_ts": ts,
        "_best_req_count": fields.get("msg_count", 0),
        "_max_real_turns": fields.get("user_turns", 0),
        "trace_list": [],
    }


def _iter_index_rows(leaf: Path, start_offset: int = 0):
    """从 index.jsonl 的 start_offset 处读取，yield (entry, new_offset_after_line)。"""
    index_file = leaf / _INDEX_NAME
    if not index_file.is_file():
        return
    try:
        with index_file.open("rb") as f:
            if start_offset > 0:
                f.seek(start_offset)
            data = f.read()
            end_offset = f.tell()
    except OSError:
        return
    for raw in data.split(b"\n"):
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("_meta"):
            continue
        yield entry, end_offset


def _process_entry(entry: dict, leaf: Path,
                   sessions: Dict[str, Dict[str, Any]],
                   chain_map: Dict[str, str]) -> None:
    """处理一条 index 行，就地更新 sessions / chain_map（内存聚合）。"""
    req_file = entry.get("req_file", "")
    path = _resolve_combined_path(leaf, req_file)
    if path is None:
        return
    parsed = parse_combined_file(path)
    if parsed is None:
        return

    messages = parsed["messages"]
    fields = compute_session_fields(messages)
    api_key = parsed["api_key"]
    model = parsed["model"]
    ts = parsed["ts"] or os.path.basename(str(path))
    filename = os.path.basename(str(path))
    success = parsed["success"]

    q1_hash = fields["q1_hash"]
    msg_count = fields["msg_count"]
    user_turns = fields["user_turns"]
    has_res = parsed["assistant_content"] is not None
    full_msg_count = msg_count + (1 if has_res else 0)

    lookup_key = f"{api_key}||{q1_hash}"
    sess_key = chain_map.get(lookup_key)
    session = sessions.get(sess_key) if sess_key else None

    # 新会话检测：user_turns 回落 ≤1 且低于已记录的峰值 → 另起会话
    if session is not None and user_turns <= 1 and \
       user_turns < session.get("_max_real_turns", 1):
        suffix = 1
        new_lookup = f"{lookup_key}##session_{suffix}"
        while new_lookup in chain_map:
            suffix += 1
            new_lookup = f"{lookup_key}##session_{suffix}"
        lookup_key = new_lookup
        session = None
        sess_key = None

    trace_entry = {
        "filename": filename,
        "model": model,
        "msg_count": full_msg_count,
        "ts": ts,
    }
    if not success:
        trace_entry["success"] = False

    if session is None:
        sess_key = ts
        # 防止同一叶子内 ts 撞键
        while sess_key in sessions:
            sess_key = sess_key + "_x"
        session = _new_session(sess_key, api_key, model, ts, fields, filename)
        session["_lookup_key"] = lookup_key  # 供回填后重建 chain_index
        session["trace_list"].append(trace_entry)
        sessions[sess_key] = session
        chain_map[lookup_key] = sess_key
    else:
        models = session.get("models", [])
        if model and model not in models:
            models.append(model)
        session["_max_real_turns"] = max(user_turns, session.get("_max_real_turns", 0))
        best = session.get("_best_req_count", 0)
        if msg_count > best or (msg_count == best and has_res):
            session["latest_file"] = filename
            session["msg_count"] = full_msg_count
            session["_best_req_count"] = msg_count
        session["last_ts"] = ts
        session["trace_list"].append(trace_entry)


def aggregate_leaf(leaf_dir: str) -> List[Dict[str, Any]]:
    """纯函数：聚合一个叶子目录的全部合并文件，返回 sessions 列表。不碰 DB。

    供多进程 worker 调用。返回结构与 session_store.bulk_insert 期望一致。
    """
    leaf = Path(leaf_dir)
    sessions: Dict[str, Dict[str, Any]] = {}
    chain_map: Dict[str, str] = {}
    for entry, _off in _iter_index_rows(leaf, 0):
        _process_entry(entry, leaf, sessions, chain_map)
    return list(sessions.values())


def leaf_end_offset(leaf_dir: str) -> int:
    """返回 index.jsonl 当前字节大小（回填后记为消费进度）。"""
    index_file = Path(leaf_dir) / _INDEX_NAME
    try:
        return index_file.stat().st_size
    except OSError:
        return 0


def consume_leaf(leaf_dir: str, force: bool = False) -> int:
    """增量消费一个叶子目录，写入 session_cache.db。返回新处理的行数。

    用 index_progress 的 byte_offset 只读新增行。force=True 时清空重建。
    供 log_routes / export_sync 的实时 refresh 调用（单进程）。
    """
    import utils.session_store as _ss

    leaf = Path(leaf_dir)
    root_dir = str(leaf)

    if force:
        _ss.delete_root(root_dir)
        start_offset = 0
    else:
        start_offset, _lc = _ss.get_progress(root_dir)

    # 重建内存 chain_map（增量归并需要已有会话的 lookup→session_key 映射）
    chain_map = _ss.get_all_chain_index(root_dir) if not force else {}
    # 内存 sessions 缓存：只装本次可能更新的会话，按需从 DB 拉
    sessions: Dict[str, Dict[str, Any]] = {}

    processed = 0
    new_offset = start_offset
    for entry, off in _iter_index_rows(leaf, start_offset):
        new_offset = off
        _process_entry_db(entry, leaf, root_dir, chain_map, sessions, _ss)
        processed += 1

    # 落盘 sessions（create/update）+ traces
    for sk, s in sessions.items():
        _flush_session(root_dir, sk, s, _ss)

    if processed or force:
        _, prev_lc = _ss.get_progress(root_dir)
        _ss.set_progress(root_dir, leaf_end_offset(leaf_dir), (0 if force else prev_lc) + processed)
    return processed


def _process_entry_db(entry, leaf, root_dir, chain_map, sessions, _ss):
    """增量路径：与 _process_entry 同逻辑，但 session 从 DB 拉、trace 直接入库。"""
    req_file = entry.get("req_file", "")
    path = _resolve_combined_path(leaf, req_file)
    if path is None:
        return
    parsed = parse_combined_file(path)
    if parsed is None:
        return

    fields = compute_session_fields(parsed["messages"])
    api_key = parsed["api_key"]
    model = parsed["model"]
    ts = parsed["ts"] or os.path.basename(str(path))
    filename = os.path.basename(str(path))
    has_res = parsed["assistant_content"] is not None
    full_msg_count = fields["msg_count"] + (1 if has_res else 0)
    user_turns = fields["user_turns"]

    lookup_key = f"{api_key}||{fields['q1_hash']}"
    sess_key = chain_map.get(lookup_key)
    session = sessions.get(sess_key) if sess_key else (
        _ss.get_session(sess_key) if sess_key else None)

    if session is not None and user_turns <= 1 and \
       user_turns < session.get("_max_real_turns", 1):
        suffix = 1
        new_lookup = f"{lookup_key}##session_{suffix}"
        while new_lookup in chain_map:
            suffix += 1
            new_lookup = f"{lookup_key}##session_{suffix}"
        lookup_key = new_lookup
        session = None
        sess_key = None

    trace_entry = {"filename": filename, "model": model, "msg_count": full_msg_count, "ts": ts}
    if not parsed["success"]:
        trace_entry["success"] = False

    if session is None:
        sess_key = ts
        while _ss.get_session(sess_key) is not None or sess_key in sessions:
            sess_key = sess_key + "_x"
        session = _new_session(sess_key, api_key, model, ts, fields, filename)
        session["_dirty_new"] = True
        session.setdefault("_pending_traces", []).append(trace_entry)
        sessions[sess_key] = session
        chain_map[lookup_key] = sess_key
        _ss.set_chain_index(root_dir, lookup_key, sess_key)
    else:
        models = session.get("models", [])
        if model and model not in models:
            models.append(model)
        session["models"] = models
        session["_max_real_turns"] = max(user_turns, session.get("_max_real_turns", 0))
        best = session.get("_best_req_count", 0)
        if fields["msg_count"] > best or (fields["msg_count"] == best and has_res):
            session["latest_file"] = filename
            session["msg_count"] = full_msg_count
            session["_best_req_count"] = fields["msg_count"]
        session["last_ts"] = ts
        session.setdefault("_pending_traces", []).append(trace_entry)
        sessions[sess_key] = session


def _flush_session(root_dir, sess_key, s, _ss):
    if s.get("_dirty_new"):
        _ss.create_session(root_dir, sess_key, s)
    else:
        _ss.update_session(sess_key, s)
    for tr in s.get("_pending_traces", []):
        _ss.append_trace(root_dir, sess_key, tr)