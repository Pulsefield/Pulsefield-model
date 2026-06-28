from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hydra import main as hydra_main
from omegaconf import DictConfig

from pulsefield_model.training.cli_legacy import reject_deprecated_legacy_training_flag
from pulsefield_model.training.hydra_config import (
    compose_training_experiment_config,
    mapper_training_kind,
    training_experiment_config_to_legacy_dict,
    validate_training_experiment_config,
    write_training_experiment_config_artifacts,
)
from pulsefield_model.training.mapper_common import precompute_mapper_tuple_phase_b_control_teacher_cache


MapperRunner = Callable[..., Any]

_CONFIG_PATH = "../configs/hydra"
_KWARG_ALIASES = {
    "device": "device_name",
    "model": "model_config_overrides",
    "control_model": "control_model_config_overrides",
    "loss": "loss_config_overrides",
}
_PATH_KWARGS = {
    "dataset_root",
    "index_path",
    "eval_index_path",
    "control_v3_timeseries_path",
    "output_dir",
    "init_from_control_checkpoint",
    "init_from_mapper_checkpoint",
    "resume_from",
    "mapper_record_cache_path",
    "control_teacher_cache_dir",
}
_HYDRA_META_FLAGS_WITHOUT_JOB_CONFIG = frozenset(("--hydra-help", "--info", "--version", "-i"))
_HYDRA_SHORT_FLAGS = frozenset(("-c", "-cd", "-cn", "-cp", "-h", "-i", "-m", "-p", "-r", "-sc"))
_HYDRA_VALUE_FLAGS = {
    "--cfg": "--cfg/-c",
    "-c": "--cfg/-c",
    "--package": "--package/-p",
    "-p": "--package/-p",
    "--config-path": "--config-path/-cp",
    "-cp": "--config-path/-cp",
    "--config-name": "--config-name/-cn",
    "-cn": "--config-name/-cn",
    "--config-dir": "--config-dir/-cd",
    "-cd": "--config-dir/-cd",
    "--shell-completion": "--shell-completion/-sc",
    "-sc": "--shell-completion/-sc",
    "--experimental-rerun": "--experimental-rerun",
}


@hydra_main(version_base="1.3", config_path=_CONFIG_PATH, config_name="mapper_training")
def _hydra_main(config: DictConfig) -> None:
    run_mapper_training_config(config)


def main(argv: Sequence[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    with _patched_argv([sys.argv[0], *_normalize_training_hydra_argv(args)]):
        _hydra_main()


def run_mapper_preset_cli(
    argv: Sequence[str] | None,
    *,
    mapper_preset: str,
    v2_runner: MapperRunner | None = None,
    v21_runner: MapperRunner | None = None,
    precompute_runner: MapperRunner | None = None,
) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    normalized_args = _normalize_training_hydra_argv(args)
    _reject_mapper_group_override(normalized_args, mapper_preset=mapper_preset)
    preset_override = f"training/mapper={mapper_preset}"
    if _requires_hydra_cli(normalized_args):
        hydra_args = list(normalized_args)
        if _should_apply_mapper_preset_for_hydra_cli(normalized_args):
            hydra_args.append(preset_override)
        main(hydra_args)
        return
    overrides = [preset_override, *normalized_args]
    run_mapper_training_from_overrides(
        overrides,
        v2_runner=v2_runner,
        v21_runner=v21_runner,
        precompute_runner=precompute_runner,
    )


def run_mapper_training_from_overrides(
    overrides: Sequence[str],
    *,
    v2_runner: MapperRunner | None = None,
    v21_runner: MapperRunner | None = None,
    precompute_runner: MapperRunner | None = None,
) -> None:
    run_mapper_training_config(
        compose_training_experiment_config(overrides=overrides),
        v2_runner=v2_runner,
        v21_runner=v21_runner,
        precompute_runner=precompute_runner,
    )


def run_mapper_training_config(
    config: Any,
    *,
    v2_runner: MapperRunner | None = None,
    v21_runner: MapperRunner | None = None,
    precompute_runner: MapperRunner | None = None,
) -> None:
    resolved = validate_training_experiment_config(config)
    legacy_config = training_experiment_config_to_legacy_dict(config)
    output_dir = Path(str(legacy_config["output_dir"]))
    write_training_experiment_config_artifacts(config, output_dir=output_dir)

    mapper_kind = mapper_training_kind(config)
    if bool(resolved.dry_run):
        print(
            "mapper_training_hydra_dry_run "
            f"mapper={mapper_kind} output_dir={output_dir.as_posix()} "
            f"legacy_config={(output_dir / 'legacy_run_config.yaml').as_posix()}",
            flush=True,
        )
        return

    if bool(legacy_config.get("precompute_control_teacher_cache_only", False)):
        _run_control_teacher_cache_precompute(legacy_config, runner=precompute_runner)
        return

    runner = _training_runner(mapper_kind, v2_runner=v2_runner, v21_runner=v21_runner)
    result = runner(**_call_kwargs(runner, legacy_config))
    print(
        "mapper_training_hydra_done "
        f"steps={result.completed_steps} final_loss={result.final_loss:.6f} "
        f"report={result.report_path.as_posix()} checkpoint={result.checkpoint_path.as_posix()}",
        flush=True,
    )


def _run_control_teacher_cache_precompute(config: dict[str, Any], *, runner: MapperRunner | None = None) -> None:
    runner = precompute_mapper_tuple_phase_b_control_teacher_cache if runner is None else runner
    result = runner(**_call_kwargs(runner, config))
    for report in result.reports:
        print(
            "control_teacher_cache_report "
            f"split={report['split']} total={report['total_entries']} "
            f"computed={report['computed_entries']} skipped={report['skipped_entries']} "
            f"elapsed_s={float(report['elapsed_s']):.1f}",
            flush=True,
        )


def _training_runner(
    mapper_kind: str,
    *,
    v2_runner: MapperRunner | None,
    v21_runner: MapperRunner | None,
) -> MapperRunner:
    if mapper_kind == "v2":
        if v2_runner is not None:
            return v2_runner
        from pulsefield_model.training.mapper_v2 import run_mapper_v2_phase_b_training

        return run_mapper_v2_phase_b_training
    if mapper_kind == "v2_1":
        if v21_runner is not None:
            return v21_runner
        from pulsefield_model.training.mapper_v2_1 import run_mapper_v2_1_phase_b_training

        return run_mapper_v2_1_phase_b_training
    raise ValueError(f"unknown mapper training kind: {mapper_kind!r}")


def _call_kwargs(runner: MapperRunner, config: dict[str, Any]) -> dict[str, Any]:
    accepted = set(inspect.signature(runner).parameters)
    kwargs: dict[str, Any] = {}
    for source_key, value in config.items():
        target_key = _KWARG_ALIASES.get(source_key, source_key)
        if target_key not in accepted or value is None:
            continue
        kwargs[target_key] = _coerce_call_value(target_key, value)
    return kwargs


def _coerce_call_value(key: str, value: Any) -> Any:
    if key in _PATH_KWARGS:
        return Path(str(value))
    if key.endswith("_config_overrides"):
        return dict(value)
    return value


def _normalize_training_hydra_argv(argv: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    args = list(argv)
    index = 0
    while index < len(args):
        token = args[index]
        reject_deprecated_legacy_training_flag(args, index)
        _reject_missing_hydra_flag_value(args, index)
        if token == "--dry-run":
            normalized.append("dry_run=true")
        elif token == "--config-dir":
            normalized.extend(["--config-path", args[index + 1]])
            index += 1
        else:
            normalized.append(token)
        index += 1
    return normalized


def _requires_hydra_cli(argv: Sequence[str]) -> bool:
    return any(token.startswith("--") or _flag_name(token) in _HYDRA_SHORT_FLAGS for token in argv)


def _should_apply_mapper_preset_for_hydra_cli(argv: Sequence[str]) -> bool:
    return not any(_flag_name(token) in _HYDRA_META_FLAGS_WITHOUT_JOB_CONFIG for token in argv)


def _flag_name(token: str) -> str:
    return token.split("=", 1)[0]


def _reject_missing_hydra_flag_value(argv: Sequence[str], index: int) -> None:
    token = argv[index]
    display = _HYDRA_VALUE_FLAGS.get(_flag_name(token))
    if display is None:
        return
    if "=" in token:
        if token.split("=", 1)[1]:
            return
        _raise_hydra_cli_usage(f"argument {display}: expected one argument")
    next_index = index + 1
    if next_index >= len(argv) or argv[next_index].startswith("-"):
        _raise_hydra_cli_usage(f"argument {display}: expected one argument")


def _raise_hydra_cli_usage(message: str) -> None:
    print("usage: mapper training Hydra entrypoint [Hydra flags] [overrides ...]", file=sys.stderr)
    print(f"mapper training Hydra entrypoint: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _reject_mapper_group_override(argv: Sequence[str], *, mapper_preset: str) -> None:
    for token in argv:
        if token.startswith("training/mapper=") or token.startswith("+training/mapper="):
            _raise_hydra_cli_usage(
                f"mapper preset aliases are fixed to training/mapper={mapper_preset}; "
                "use pulsefield_model.training.mapper_training_hydra to select another mapper group",
            )


@contextmanager
def _patched_argv(argv: Sequence[str]) -> Any:
    original = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
