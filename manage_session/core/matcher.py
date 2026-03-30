"""
core/matcher.py — 匹配引擎

精确匹配：task.query == index.q1（文本精确相等）。
不做模糊匹配，不同批次天然为 0，符合预期。
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MatchResult:
    task_id: str
    index_id: str
    task_hash: str
    index_hash: str

    matched: list = field(default_factory=list)
    unmatched_tasks: list = field(default_factory=list)
    unmatched_indexes: list = field(default_factory=list)

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ── computed ──────────────────────────────────────────────
    @property
    def matched_count(self):
        return len(self.matched)

    @property
    def unmatched_task_count(self):
        return len(self.unmatched_tasks)

    @property
    def unmatched_index_count(self):
        return len(self.unmatched_indexes)

    def to_dict(self):
        return {
            "pair_key": f"{self.task_hash[:8]}__{self.index_hash[:8]}",
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "index_id": self.index_id,
            "index_hash": self.index_hash,
            "matched_count": self.matched_count,
            "unmatched_task_count": self.unmatched_task_count,
            "unmatched_index_count": self.unmatched_index_count,
            "duplicate_query_count": 0,
            "matches": self.matched,
            "unmatched_tasks": self.unmatched_tasks,
            "unmatched_indexes": self.unmatched_indexes,
            "generated_at": self.generated_at,
        }


def match(task_queries, index_q1s, task_id, index_id, task_hash, index_hash):
    """
    精确匹配 task.query == index.q1。

    task_queries: {query_text: task_record}
    index_q1s:    {q1_text: index_entry}

    matched 中每条记录:
    {
      "query": ...,
      "task_row_data": {...},    # task 里的完整记录
      "index_entry": {...},      # index 里的完整条目，附加 _task_* 元数据
    }
    """
    result = MatchResult(
        task_id=task_id,
        index_id=index_id,
        task_hash=task_hash,
        index_hash=index_hash,
    )

    matched_queries = set()

    for query, task_rec in task_queries.items():
        if query in index_q1s:
            idx_entry = dict(index_q1s[query])
            # 把 task 的元数据挂载到 index 条目上
            idx_entry["_task_id"]          = task_id
            idx_entry["_batch_date"]       = task_rec.get("date")
            idx_entry["_topic"]            = task_rec.get("topic")
            idx_entry["_env_name"]         = task_rec.get("env_name")
            idx_entry["_profile_name"]     = task_rec.get("profile_name")
            idx_entry["_required_skills"]  = task_rec.get("required_skills", [])
            idx_entry["_required_files"]   = task_rec.get("required_files", [])

            result.matched.append({
                "query": query,
                "task_row_data": task_rec,
                "index_entry": idx_entry,
            })
            matched_queries.add(query)
        else:
            result.unmatched_tasks.append(task_rec)

    for q1, idx_entry in index_q1s.items():
        if q1 not in matched_queries:
            result.unmatched_indexes.append(idx_entry)

    return result
