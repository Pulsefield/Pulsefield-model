#!/bin/sh
# Keep the assertion attached to the worker; display sleep remains available.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
if [ "$(uname -s)" != Darwin ]; then
    echo 'vacation-training.sh requires macOS caffeinate' >&2
    exit 2
fi
exec /usr/bin/caffeinate -is uv run --offline --python 3.10 --extra mps python -m \
    pulsefield_model.research.vacation_training.hydra "$@"
