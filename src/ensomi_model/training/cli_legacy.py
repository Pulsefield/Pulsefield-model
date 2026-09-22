from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence


DEPRECATED_LEGACY_TRAINING_FLAG_REPLACEMENTS = {
    "--batch-size": "run.batch_size=<value>",
    "--config": "training/mapper=<preset> or --config-path/--config-name for Hydra configs",
    "--control-teacher-cache-dir": "data.control_teacher_cache_dir=<path>",
    "--control-teacher-cache-overwrite": "data.control_teacher_cache_overwrite=true",
    "--control-teacher-precompute-batch-size": "data.control_teacher_precompute_batch_size=<value>",
    "--control-v3-timeseries-path": "data.control_v3_timeseries_path=<path>",
    "--dataset-progress": "runtime.dataset_progress=true",
    "--dataset-root": "data.dataset_root=<path>",
    "--device": "runtime.device=<device>",
    "--eval-every": "run.eval_every=<value>",
    "--eval-fraction": "data.eval_fraction=<value>",
    "--eval-index-path": "data.eval_index_path=<path>",
    "--eval-size": "data.eval_size=<value>",
    "--final-train-eval-size": "data.final_train_eval_size=<value>",
    "--include-full-song-context": "data.include_full_song_context=true",
    "--index-path": "data.index_path=<path>",
    "--init-from-control": "output.init_from_control_checkpoint=<path>",
    "--init-from-control-checkpoint": "output.init_from_control_checkpoint=<path>",
    "--init-from-mapper": "output.init_from_mapper_checkpoint=<path>",
    "--init-from-mapper-checkpoint": "output.init_from_mapper_checkpoint=<path>",
    "--learning-rate": "run.learning_rate=<value>",
    "--length-bucket-size-multiplier": "data.length_bucket_size_multiplier=<value>",
    "--length-bucketed-batches": "data.length_bucketed_batches=true",
    "--log-every": "run.log_every=<value>",
    "--mapper-record-cache-path": "data.mapper_record_cache_path=<path>",
    "--max-cached-maps": "runtime.max_cached_maps=<value>",
    "--max-steps": "run.max_steps=<value>",
    "--mps-cleanup-every": "run.mps_cleanup_every=<value>",
    "--no-dataset-progress": "runtime.dataset_progress=false",
    "--no-include-full-song-context": "data.include_full_song_context=false",
    "--no-length-bucketed-batches": "data.length_bucketed_batches=false",
    "--no-skip-first-eval-pass": "run.skip_first_eval_pass=false",
    "--num-workers": "runtime.num_workers=<value>",
    "--output-dir": "output.output_dir=<path>",
    "--precompute-control-teacher-cache": "data.precompute_control_teacher_cache=true",
    "--precompute-control-teacher-cache-only": "data.precompute_control_teacher_cache_only=true",
    "--require-control-teacher-cache": "data.require_control_teacher_cache=true",
    "--resume-from": "output.resume_from=<path>",
    "--run-name": "run.run_name=<name>",
    "--save-every": "run.save_every=<value>",
    "--seed": "runtime.seed=<value>",
    "--skip-first-eval-pass": "run.skip_first_eval_pass=true",
    "--weight-decay": "run.weight_decay=<value>",
}


def reject_deprecated_legacy_training_flags(
    argv: Sequence[str],
    *,
    entrypoint: str = "the Hydra training entrypoints",
    argument_name: str = "legacy training flag",
    replacement_overrides: Mapping[str, str] | None = None,
) -> None:
    for index in range(len(argv)):
        reject_deprecated_legacy_training_flag(
            argv,
            index,
            entrypoint=entrypoint,
            argument_name=argument_name,
            replacement_overrides=replacement_overrides,
        )


def reject_deprecated_legacy_training_flag(
    argv: Sequence[str],
    index: int,
    *,
    entrypoint: str = "the Hydra training entrypoints",
    argument_name: str = "legacy training flag",
    replacement_overrides: Mapping[str, str] | None = None,
) -> None:
    token = argv[index]
    flag = legacy_flag_name(token)
    replacements = replacement_overrides or {}
    replacement = replacements.get(flag, DEPRECATED_LEGACY_TRAINING_FLAG_REPLACEMENTS.get(flag))
    if replacement is None:
        return
    value = legacy_flag_value(argv, index, token)
    if value is not None:
        replacement = replacement.replace("<value>", value)
        replacement = replacement.replace("<path>", value)
        replacement = replacement.replace("<device>", value)
        replacement = replacement.replace("<name>", value)
    print(
        f"Deprecated {argument_name} {flag!r} is not supported by {entrypoint}. "
        f"Use {replacement!r} instead.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def legacy_flag_name(token: str) -> str:
    return token.split("=", 1)[0]


def legacy_flag_value(argv: Sequence[str], index: int, token: str) -> str | None:
    if "=" in token:
        return token.split("=", 1)[1]
    next_index = index + 1
    if next_index < len(argv) and not argv[next_index].startswith("-"):
        return argv[next_index]
    return None
