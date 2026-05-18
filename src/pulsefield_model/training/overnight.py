from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


@dataclass(frozen=True)
class RunProgress:
    completed_steps: int
    is_complete: bool


def load_config(config_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"overnight config must be a YAML mapping: {config_path}")
    return dict(loaded)


def read_progress(report_path: Path, *, max_steps: int) -> RunProgress:
    if not report_path.is_file():
        return RunProgress(completed_steps=0, is_complete=False)
    try:
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return RunProgress(completed_steps=0, is_complete=False)
    if not isinstance(loaded, Mapping):
        return RunProgress(completed_steps=0, is_complete=False)
    raw_completed = loaded.get("completed_steps", 0)
    completed_steps = int(raw_completed) if isinstance(raw_completed, int) and raw_completed >= 0 else 0
    is_complete = bool(loaded.get("is_complete", False)) or completed_steps >= max_steps
    return RunProgress(completed_steps=completed_steps, is_complete=is_complete)


def archive_checkpoint_path(output_dir: Path, completed_steps: int) -> Path:
    return output_dir / "checkpoints" / f"checkpoint_step_{completed_steps:06d}.pt"


def existing_resume_checkpoint(config: Mapping[str, Any], output_dir: Path) -> Path | None:
    latest_checkpoint = output_dir / "checkpoint.pt"
    if latest_checkpoint.is_file():
        return latest_checkpoint
    configured_resume = config.get("resume_from")
    if isinstance(configured_resume, str) and configured_resume:
        resume_path = Path(configured_resume)
        if resume_path.is_file():
            return resume_path
    return None


def next_saved_step_target(*, completed_steps: int, max_steps: int, save_every: int, steps_per_process: int) -> int:
    if completed_steps >= max_steps:
        return max_steps
    threshold = min(max_steps, completed_steps + steps_per_process)
    if threshold == max_steps:
        return max_steps
    if threshold <= 1 and completed_steps < 1:
        return 1
    next_boundary = ((threshold + save_every - 1) // save_every) * save_every
    return min(max_steps, max(next_boundary, completed_steps + 1))


def write_child_config(
    *,
    base_config: Mapping[str, Any],
    output_dir: Path,
    config_path: Path,
    resume_checkpoint: Path | None,
) -> None:
    child_config = dict(base_config)
    child_config["output_dir"] = output_dir.as_posix()
    if resume_checkpoint is None:
        child_config.pop("resume_from", None)
    else:
        child_config["resume_from"] = resume_checkpoint.as_posix()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(child_config, sort_keys=False), encoding="utf-8")


def checkpoint_saved_after(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    completed_steps: int,
    target_step: int,
    previous_mtime_ns: int | None,
) -> bool:
    if completed_steps < target_step or not checkpoint_path.is_file():
        return False
    archive_path = archive_checkpoint_path(output_dir, completed_steps)
    if archive_path.is_file():
        return True
    if previous_mtime_ns is None:
        return True
    return checkpoint_path.stat().st_mtime_ns > previous_mtime_ns


def terminate_process_group(process: Any, *, timeout_s: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def sleep_until_stop_or_timeout(seconds: float, stop_requested: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + max(float(seconds), 0.0)
    while time.monotonic() < deadline:
        if stop_requested():
            return False
        time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))
    return not stop_requested()
