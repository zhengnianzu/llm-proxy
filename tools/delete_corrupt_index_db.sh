#!/usr/bin/env bash
# delete_corrupt_index_db.sh — 删除 proxy-004 源里 6 个已确认 malformed 的 index.db
#
# 这 6 个 index.db 读 requests 表即抛 "database disk image is malformed"，
# 是「构建索引」在 needs_build 阶段整根崩溃的元凶。删掉后重「构建索引」即可，
# 只动 index.db 及其 -wal/-shm/-journal 兄弟文件，绝不碰 index.jsonl / 原始 .json。
#
# 默认 dry-run，只打印不删；加 --apply 才真正删除。
set -euo pipefail

ROOT="/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/raw/harness/proxy-004/data/newapi/logs/details"

LEAVES=(
  "details/260728/26072806"
  "details/260728/26072813"
  "details/260728/26072815"
  "details/260729/26072905"
  "details/260730/26073011"
  "details/260730/26073012"
)

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

echo "root  = $ROOT"
echo "mode  = $([[ $APPLY -eq 1 ]] && echo 'APPLY (真删)' || echo 'DRY-RUN (只打印)')"
echo "------------------------------------------------------------"

removed=0
for leaf in "${LEAVES[@]}"; do
  base="$ROOT/$leaf/index.db"
  for f in "$base" "$base-wal" "$base-shm" "$base-journal"; do
    if [[ -e "$f" ]]; then
      if [[ $APPLY -eq 1 ]]; then
        rm -f -- "$f"
        echo "[DEL]       $f"
        removed=$((removed+1))
      else
        echo "[WOULD-DEL] $f"
      fi
    fi
  done
done

echo "------------------------------------------------------------"
if [[ $APPLY -eq 1 ]]; then
  echo "完成：删除 $removed 个文件（6 个坏叶子）。现在到数据管理页对该源重新「构建索引」。"
else
  echo "Dry-run 结束。确认无误后加 --apply 重跑即可删除。"
fi
