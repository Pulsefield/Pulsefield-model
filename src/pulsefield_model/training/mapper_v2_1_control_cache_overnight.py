from __future__ import annotations

import argparse
import signal
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from pulsefield_model.training.overnight import hydra_override
from pulsefield_model.training.overnight import load_hydra_training_config
from pulsefield_model.training.overnight import reject_deprecated_overnight_trainer_args
from pulsefield_model.training.overnight import sleep_until_stop_or_timeout
from pulsefield_model.training.overnight import terminate_process_group


DEFAULT_MAPPER_PRESET = "v2_1_sparse_d384_l4_phase_b"
DEFAULT_UV_COMMAND = "uv run --extra mps python -m pulsefield_model.training.mapper_training_hydra"
DEFAULT_CONTROL_TEACHER_PRECOMPUTE_BATCH_SIZE = 24
DEFAULT_MAX_RUNS = 16000


def run_supervisor(args: argparse.Namespace, trainer_args: Sequence[str]) -> int:
    base_overrides = [str(arg) for arg in trainer_args]
    base_config = load_hydra_training_config(str(args.mapper_preset), base_overrides)
    output_dir = Path(args.output_dir or base_config.get("output_dir", "artifacts/runs/stage2_mapper_v2_1/overnight"))
    output_dir.mkdir(parents=True, exist_ok=True)
    supervisor_dir = output_dir / "overnight_control_cache_supervisor"
    log_dir = Path(args.log_dir) if args.log_dir is not None else supervisor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    child_overrides = [
        f"training/mapper={args.mapper_preset}",
        *base_overrides,
        hydra_override("output.output_dir", output_dir),
        "data.precompute_control_teacher_cache_only=true",
        hydra_override("data.control_teacher_precompute_batch_size", int(args.control_teacher_precompute_batch_size)),
    ]

    stop_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = int(signum)
            print(f"overnight_cache_stop_requested signal={signum}", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def should_stop() -> bool:
        return stop_signal is not None

    run_count = 0
    consecutive_failures = 0
    while True:
        if should_stop():
            return 128 + int(stop_signal or signal.SIGTERM)
        if args.max_runs and run_count >= args.max_runs:
            print(f"overnight_cache_max_runs_reached runs={run_count}", flush=True)
            return 0

        run_count += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"attempt_{run_count:04d}_{stamp}.log"
        command = [*shlex.split(args.uv_command), *child_overrides]
        if args.dry_run:
            print("overnight_cache_dry_run " + shlex.join(command), flush=True)
            return 0

        print(
            "overnight_cache_start "
            f"run={run_count} mapper_preset={args.mapper_preset} log={log_path}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            while process.poll() is None:
                if not sleep_until_stop_or_timeout(float(args.poll_seconds), should_stop):
                    terminate_process_group(process, timeout_s=float(args.terminate_timeout_seconds))
                    return 128 + int(stop_signal or signal.SIGTERM)
            return_code = int(process.returncode or 0)

        if return_code == 0:
            print(f"overnight_cache_complete run={run_count} log={log_path}", flush=True)
            return 0

        consecutive_failures += 1
        print(
            "overnight_cache_child_failed "
            f"run={run_count} returncode={return_code} failures={consecutive_failures} "
            f"log={log_path}",
            flush=True,
        )
        if consecutive_failures >= int(args.max_consecutive_failures):
            return return_code
        sleep_until_stop_or_timeout(float(args.restart_delay_seconds), should_stop)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 2 mapper v2.1 control-teacher cache precompute in retryable child processes. "
            "The cache writer skips existing entries, so reruns are restart-safe unless overwrite is requested."
        )
    )
    parser.add_argument("--mapper-preset", default=DEFAULT_MAPPER_PRESET)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--terminate-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=20.0)
    parser.add_argument("--max-runs", type=int, default=DEFAULT_MAX_RUNS, help="0 means retry until success")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument(
        "--control-teacher-precompute-batch-size",
        type=int,
        default=DEFAULT_CONTROL_TEACHER_PRECOMPUTE_BATCH_SIZE,
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--uv-command",
        default=DEFAULT_UV_COMMAND,
        help="Command prefix used to launch the mapper v2.1 cache-only precompute.",
    )
    args, trainer_args = parser.parse_known_args(argv)
    reject_deprecated_overnight_trainer_args(
        trainer_args,
        entrypoint="the mapper v2.1 control-cache overnight supervisor trainer args",
    )
    return args, trainer_args


def main(argv: Sequence[str] | None = None) -> None:
    args, trainer_args = parse_args(argv)
    raise SystemExit(run_supervisor(args, trainer_args))


if __name__ == "__main__":
    main(sys.argv[1:])
