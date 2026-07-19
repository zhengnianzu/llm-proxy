#!/bin/bash
# obs_upload.sh — wrapper around obsutil cp
# Usage: obs_upload.sh <local_path> <obs_path> [jobs]
# Example: obs_upload.sh test/ obs://rl-agentdata/zhengnianzu/test/ 8

OBSUTIL_BIN="$(dirname "$0")/obsutil/obsutil"

if [ ! -x "$OBSUTIL_BIN" ]; then
    echo "ERROR: obsutil not found at $OBSUTIL_BIN" >&2
    exit 1
fi

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <local_path> <obs_path> [jobs]" >&2
    echo "Example: $0 test/ obs://rl-agentdata/zhengnianzu/test/ 8" >&2
    exit 1
fi

LOCAL="$1"
OBS="$2"
JOBS="${3:-8}"

exec "$OBSUTIL_BIN" cp "$LOCAL" "$OBS" -f -r -j "$JOBS"
