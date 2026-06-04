"""
utils/export_session_index.py — 导出 session_index.jsonl 并可选同步到 OBS

推荐通过 sess export 调用（自动读取 meta），本脚本用于手动指定目录。

用法:
    # 从 app-meta.json 读取 previous_logs_dir 进行 export
    python -m utils.export_session_index --meta logs/app-meta-port4001.json

    # 指定目录
    python -m utils.export_session_index --logs-dir logs_all/env-xunxing-zyKA/26060120

    # sync 模式
    python -m utils.export_session_index --logs-dir logs_all/env-xunxing-zyKA/26060120 \\
        --sync --obs-dst obs://bucket/prefix/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from utils.export_sync import export_session_index, sync_session_index

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _resolve_logs_dir(args: argparse.Namespace) -> str:
    if args.logs_dir:
        p = Path(args.logs_dir).resolve()
        if not p.is_dir():
            sys.exit(f"[error] 目录不存在: {p}")
        return str(p)

    if args.meta:
        meta_path = Path(args.meta)
        if not meta_path.is_file():
            sys.exit(f"[error] meta 文件不存在: {meta_path}")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            sys.exit(f"[error] 读取 meta 文件失败: {e}")
        prev = meta.get("previous_logs_dir")
        if not prev:
            sys.exit("[error] meta 文件中 previous_logs_dir 为空 (可能尚未 restart)")
        p = Path(prev)
        if not p.is_absolute():
            p = meta_path.parent.parent / prev
        p = p.resolve()
        if not p.is_dir():
            sys.exit(f"[error] previous_logs_dir 目录不存在: {p}")
        return str(p)

    sys.exit("[error] 必须指定 --logs-dir 或 --meta")


def main():
    parser = argparse.ArgumentParser(
        description="导出 session_index.jsonl 并可选同步到 OBS"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--logs-dir",
        help="指定单个时间戳目录 (如 logs_all/env-xxx/26060120)",
    )
    group.add_argument(
        "--meta",
        help="从 app-meta.json 读取 previous_logs_dir",
    )
    parser.add_argument("--force", action="store_true", help="强制重新生成，忽略 mtime 检查")
    parser.add_argument("--sync", action="store_true", help="启用上传模式")
    parser.add_argument("--obs-dst", help="OBS 目标路径前缀 (如 obs://bucket/prefix/)")
    parser.add_argument("--upload-script", default=None, help="自定义上传脚本")
    parser.add_argument("--workers", type=int, default=4, help="上传并发数 (默认 4)")
    args = parser.parse_args()

    if args.sync and not args.obs_dst:
        parser.error("--sync 模式需要指定 --obs-dst")

    logs_dir = _resolve_logs_dir(args)
    print(f"[info] logs_dir: {logs_dir}")

    result = export_session_index(logs_dir, force=args.force)
    if result["skipped"]:
        print(f"[info] 跳过 (无变更). 共 {result['total_sessions']} 条 session, 平均 {result['avg_msg_count']} 轮 msg")
    else:
        print(f"[done] 共 {result['total_sessions']} 条 session, 平均 {result['avg_msg_count']} 轮 msg")

    if args.sync:
        print(f"[info] 开始同步到 {args.obs_dst}")
        sync_result = sync_session_index(
            logs_dir,
            obs_dst=args.obs_dst,
            workers=args.workers,
            upload_script=args.upload_script,
        )
        print(f"[done] 上传 {sync_result['uploaded']} 成功, {sync_result['failed']} 失败, {sync_result['skipped']} 跳过")


if __name__ == "__main__":
    main()
