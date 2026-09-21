#!/bin/sh
# Retain the sleep assertion for the entire staged worker lifetime.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
if [ "$(uname -s)" != Darwin ]; then
    echo 'r1-restore.sh requires macOS caffeinate' >&2
    exit 2
fi
exec /usr/bin/caffeinate -is uv run --offline --python 3.10 --extra mps python -m \
    pulsefield_model.research.r1_restore.hydra "$@"
