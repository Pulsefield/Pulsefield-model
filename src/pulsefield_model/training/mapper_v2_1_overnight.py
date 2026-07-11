from __future__ import annotations

import argparse
import signal
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from pulsefield_model.training.overnight import checkpoint_saved_after
from pulsefield_model.training.overnight import existing_resume_checkpoint
from pulsefield_model.training.overnight import hydra_override
from pulsefield_model.training.overnight import load_hydra_training_config
from pulsefield_model.training.overnight import next_saved_step_target
from pulsefield_model.training.overnight import read_progress
from pulsefield_model.training.overnight import reject_deprecated_overnight_trainer_args
from pulsefield_model.training.overnight import sleep_until_stop_or_timeout
from pulsefield_model.training.overnight import terminate_process_group


DEFAULT_MAPPER_PRESET = "v2_1_sparse_d384_l4_phase_b"
DEFAULT_UV_COMMAND = "uv run --extra mps python -m pulsefield_model.training.mapper_training_hydra"


def run_supervisor(args: argparse.Namespace, trainer_args: Sequence[str]) -> int:
    base_overrides = [str(arg) for arg in trainer_args]
    base_config = load_hydra_training_config(str(args.mapper_preset), base_overrides)
    output_dir = Path(args.output_dir or base_config.get("output_dir", "artifacts/runs/stage2_mapper_v2_1/overnight"))
    max_steps = int(args.max_steps or base_config.get("max_steps", 5000))
    save_every = int(args.save_every or base_config.get("save_every") or base_config.get("eval_every", 100))
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")
    steps_per_process = int(args.steps_per_process or save_every)
    if steps_per_process <= 0:
        raise ValueError(f"steps_per_process must be positive, got {steps_per_process}")

    output_dir.mkdir(parents=True, exist_ok=True)
    supervisor_dir = output_dir / "overnight_supervisor"
    log_dir = Path(args.log_dir) if args.log_dir is not None else supervisor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"
    stop_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = int(signum)
            print(f"overnight_stop_requested signal={signum}", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def should_stop() -> bool:
        return stop_signal is not None

    run_count = 0
    consecutive_failures = 0
    while True:
        if should_stop():
            return 128 + int(stop_signal or signal.SIGTERM)
        progress = read_progress(report_path, max_steps=max_steps)
        if progress.is_complete and args.stop_when_complete:
            print(f"overnight_complete step={progress.completed_steps}/{max_steps}", flush=True)
            return 0
        if args.max_runs and run_count >= args.max_runs:
            print(f"overnight_max_runs_reached runs={run_count} step={progress.completed_steps}/{max_steps}", flush=True)
            return 0

        resume_checkpoint = existing_resume_checkpoint(base_config, output_dir)
        child_overrides = [
            f"training/mapper={args.mapper_preset}",
            *base_overrides,
            hydra_override("output.output_dir", output_dir),
            hydra_override("run.max_steps", max_steps),
            hydra_override("run.save_every", save_every),
            hydra_override("output.resume_from", resume_checkpoint),
        ]
        target_step = next_saved_step_target(
            completed_steps=progress.completed_steps,
            max_steps=max_steps,
            save_every=save_every,
            steps_per_process=steps_per_process,
        )
        previous_mtime_ns = checkpoint_path.stat().st_mtime_ns if checkpoint_path.is_file() else None
        run_count += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"attempt_{run_count:04d}_{stamp}.log"
        command = [*shlex.split(args.uv_command), *child_overrides]
        if args.dry_run:
            print("overnight_dry_run " + shlex.join(command), flush=True)
            return 0

        print(
            "overnight_start "
            f"run={run_count} step={progress.completed_steps}/{max_steps} "
            f"target_step={target_step} resume_from={resume_checkpoint} log={log_path}",
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
            saved = False
            while process.poll() is None:
                if not sleep_until_stop_or_timeout(float(args.poll_seconds), should_stop):
                    terminate_process_group(process, timeout_s=float(args.terminate_timeout_seconds))
                    return 128 + int(stop_signal or signal.SIGTERM)
                current = read_progress(report_path, max_steps=max_steps)
                saved = checkpoint_saved_after(
                    output_dir=output_dir,
                    checkpoint_path=checkpoint_path,
                    completed_steps=current.completed_steps,
                    target_step=target_step,
                    previous_mtime_ns=previous_mtime_ns,
                )
                if saved:
                    print(
                        "overnight_saved "
                        f"run={run_count} step={current.completed_steps}/{max_steps} "
                        f"terminating_child_pid={process.pid}",
                        flush=True,
                    )
                    sleep_until_stop_or_timeout(float(args.post_save_grace_seconds), should_stop)
                    terminate_process_group(process, timeout_s=float(args.terminate_timeout_seconds))
                    break

            return_code = process.poll()

        current = read_progress(report_path, max_steps=max_steps)
        if not saved:
            saved = checkpoint_saved_after(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                completed_steps=current.completed_steps,
                target_step=target_step,
                previous_mtime_ns=previous_mtime_ns,
            )
        if saved:
            consecutive_failures = 0
            print(f"overnight_restart_ready run={run_count} step={current.completed_steps}/{max_steps}", flush=True)
            if current.is_complete and args.stop_when_complete:
                print(f"overnight_complete step={current.completed_steps}/{max_steps}", flush=True)
                return 0
            sleep_until_stop_or_timeout(float(args.restart_delay_seconds), should_stop)
            continue

        consecutive_failures += 1
        print(
            "overnight_child_failed "
            f"run={run_count} returncode={return_code} failures={consecutive_failures} "
            f"log={log_path}",
            flush=True,
        )
        if consecutive_failures >= int(args.max_consecutive_failures):
            return int(return_code or 1)
        sleep_until_stop_or_timeout(float(args.restart_delay_seconds), should_stop)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 2 mapper v2.1 training in disposable child processes, "
            "restarting each child after a durable checkpoint save."
        )
    )
    parser.add_argument("--mapper-preset", default=DEFAULT_MAPPER_PRESET)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--steps-per-process", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--post-save-grace-seconds", type=float, default=3.0)
    parser.add_argument("--terminate-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=20.0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run until max_steps is complete")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-when-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit when report.json says the configured max_steps run is complete.",
    )
    parser.add_argument(
        "--uv-command",
        default=DEFAULT_UV_COMMAND,
        help="Command prefix used to launch the mapper v2.1 trainer.",
    )
    args, trainer_args = parser.parse_known_args(argv)
    reject_deprecated_overnight_trainer_args(
        trainer_args,
        entrypoint="the mapper v2.1 overnight supervisor trainer args",
    )
    return args, trainer_args


def main(argv: Sequence[str] | None = None) -> None:
    args, trainer_args = parse_args(argv)
    raise SystemExit(run_supervisor(args, trainer_args))


if __name__ == "__main__":
    main(sys.argv[1:])
