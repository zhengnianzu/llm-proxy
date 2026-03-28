"""
report.py — 基于 report/ 目录动态展示各 key 的分析报告汇总

用法:
    python report.py <report_dir> [--port 8080] [--host 0.0.0.0]

report_dir 结构示意:
    report/
        key1/
            session_report.html   ← 子报告链接目标
            session_report.xlsx   ← 用于读取统计数据
        key2/
            session_report.html
            session_report.xlsx
        overview.html             ← 自动生成的全量合并报告

主界面列出所有 key，显示：
  - session 数量
  - 任务完成率（成功率）
  - 工具成功率
  - 链接到 report/key/session_report.html
  - 链接到 report/overview.html（全量合并报告）

每次请求主页时：
  1. 扫描 report_dir 下所有含 xlsx 的子目录
  2. 若目录集合较上次发生变化，重新合并生成 overview.html
  3. 渲染主页 overview.html.j2

依赖:
    pip install fastapi uvicorn[standard]
"""

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from jinja2 import Environment, FileSystemLoader
import uvicorn


# ---------------------------------------------------------------------------
# 动态导入同目录模块
# ---------------------------------------------------------------------------

def _import_module(name: str) -> object:
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 从 xlsx 提取关键指标
# ---------------------------------------------------------------------------

def load_key_stats(key_dir: Path, mr, az) -> Optional[Dict]:
    """读取 key_dir 下的 session_report.xlsx，返回汇总指标字典。"""
    xlsx_path = key_dir / "session_report.xlsx"
    if not xlsx_path.is_file():
        return None

    sessions = mr.load_xlsx(xlsx_path)
    if not sessions:
        return None

    stats = az.compute_stats(sessions)
    total = stats["total"]
    ok_count = sum(1 for s in sessions if s["completed"] == 0)
    success_rate = ok_count / total * 100 if total else 0

    total_tr = stats["total_tr"]
    tool_succ = stats["total_succ"]
    tool_rate = tool_succ / total_tr * 100 if total_tr else None

    return {
        "key":          key_dir.name,
        "total":        total,
        "ok_count":     ok_count,
        "success_rate": round(success_rate, 1),
        "tool_rate":    round(tool_rate, 1) if tool_rate is not None else None,
        "has_html":     (key_dir / "session_report.html").is_file(),
    }


def scan_report_dir(report_dir: Path, mr, az) -> List[Dict]:
    """扫描 report_dir 下所有含 xlsx 的子目录，返回排序后的统计列表。"""
    results = []
    for sub in sorted(report_dir.iterdir()):
        if not sub.is_dir():
            continue
        stats = load_key_stats(sub, mr, az)
        if stats is not None:
            results.append(stats)
    return results


# ---------------------------------------------------------------------------
# 合并生成 overview.html
# ---------------------------------------------------------------------------

def build_overview_html(report_dir: Path, keys: List[Dict], mr, az) -> None:
    """用所有 key 的 xlsx 合并生成 report_dir/overview.html。"""
    all_sessions = []
    for k in keys:
        xlsx_path = report_dir / k["key"] / "session_report.xlsx"
        sessions = mr.load_xlsx(xlsx_path)
        for s in sessions:
            s["_source"] = k["key"]
        all_sessions.extend(sessions)

    if not all_sessions:
        return

    stats = az.compute_stats(all_sessions)
    ctx = az.build_context(all_sessions, stats)
    ctx["generated_at"] += f"  (合并自 {len(keys)} 个 key)"
    ctx["total_sources"] = len(keys)

    az.render_report("report.html.j2", ctx, report_dir / "overview.html")


# ---------------------------------------------------------------------------
# HTML 渲染
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _render_index(report_dir: Path, keys: List[Dict], has_overview: bool) -> str:
    total_sessions = sum(k["total"] for k in keys)
    avg_sr = (sum(k["success_rate"] for k in keys) / len(keys)) if keys else 0
    tool_rates = [k["tool_rate"] for k in keys if k["tool_rate"] is not None]
    avg_tr = f"{sum(tool_rates)/len(tool_rates):.1f}%" if tool_rates else "N/A"

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("overview.html.j2").render(
        report_dir=str(report_dir),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        key_count=len(keys),
        total_sessions=total_sessions,
        avg_success_rate=f"{avg_sr:.1f}",
        avg_tool_rate=avg_tr,
        keys=keys,
        has_overview=has_overview,
    )


# ---------------------------------------------------------------------------
# FastAPI 应用工厂
# ---------------------------------------------------------------------------

def create_app(report_dir: Path) -> FastAPI:
    print("[info] 导入依赖模块 ...", file=sys.stderr)
    mr = _import_module("merge_reports")
    az = _import_module("analyze_sessions")

    # 记录上次生成 overview.html 时的 key 集合，用于变更检测
    _state: Dict = {"last_keys": None}

    def _refresh_overview_if_needed(keys: List[Dict]) -> None:
        key_set = frozenset(k["key"] for k in keys)
        if key_set != _state["last_keys"]:
            print(f"[info] key 集合变化，重新生成 overview.html ({len(keys)} 个 key) ...",
                  file=sys.stderr)
            build_overview_html(report_dir, keys, mr, az)
            _state["last_keys"] = key_set

    app = FastAPI(title="报告总览")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        keys = scan_report_dir(report_dir, mr, az)
        _refresh_overview_if_needed(keys)
        has_overview = (report_dir / "overview.html").is_file()
        return _render_index(report_dir, keys, has_overview)

    @app.get("/files/{file_path:path}")
    async def serve_file(file_path: str):
        """提供 report_dir 下的静态文件（安全限制在 report_dir 内）。"""
        target = (report_dir / file_path).resolve()
        try:
            target.relative_to(report_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(target)

    return app


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="为 report/ 目录下各 key 的分析报告提供 Web 总览",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python report.py ./report\n"
            "  python report.py /data/reports --port 9000 --host 127.0.0.1"
        ),
    )
    parser.add_argument("report_dir", metavar="REPORT_DIR",
                        help="包含各 key 子目录的 report 根目录")
    parser.add_argument("--port", "-p", type=int, default=8080,
                        help="监听端口（默认 8080）")
    parser.add_argument("--host", default="0.0.0.0",
                        help="监听地址（默认 0.0.0.0）")
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    if not report_dir.is_dir():
        print(f"[error] 目录不存在: {report_dir}", file=sys.stderr)
        sys.exit(1)

    app = create_app(report_dir)
    print(f"[info] 服务启动: http://{args.host}:{args.port}/  (report_dir={report_dir})",
          file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
