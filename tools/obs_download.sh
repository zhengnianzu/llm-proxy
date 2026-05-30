#!/bin/bash
# obs_download.sh — wrapper around obsutil cp
# Usage: obs_download.sh <local_path> <obs_path>
# Example: obs_download.sh test/ obs://rl-agentdata/zhengnianzu/test/

OBSUTIL_BIN="$(dirname "$0")/obsutil/obsutil"

if [ ! -x "$OBSUTIL_BIN" ]; then
    echo "ERROR: obsutil not found at $OBSUTIL_BIN" >&2
    exit 1
fi

if [ "$#" -lt 2 ]; then
    echo "Usage: $0  <obs_path> <local_path>" >&2
    echo "Example: $0 obs://rl-agentdata/zhengnianzu/test/ test/ " >&2
    exit 1
fi

OBS="$1"
LOCAL="$2"


exec "$OBSUTIL_BIN" cp "$OBS" "$LOCAL" -f -r
