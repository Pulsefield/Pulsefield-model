"""Local sequence-training run over explicitly selected, verified train sources.

The selected corpus uses the M0 in-memory source owner. Disk-backed loading,
cache budgets and durable restart are separate runtime work; final weights are
an initialization artifact, not a resumable checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..scoped_style_modeling.dataset import ContractError, canonical_json, digest
from .data import admit_source
from .model import CausalBackbone
from .training import SequenceTrainer
from .training_config import TrainExperimentConfig
from .windows import WindowSampler


def load_training_sources(config: TrainExperimentConfig):
    """Verify the pinned existing split before reading selected source payloads.

    Held-out payloads are never opened. Group leakage is checked across the
    entire manifest, including identities outside the selected train subset.
    """
    if not config.split_sha256 or not config.source_sha256:
        raise ContractError("Training requires split_sha256 and an explicit nonempty source_sha256 selection")
    manifest = json.loads(Path(config.split_manifest).read_text())
    actual = digest(canonical_json({key: value for key, value in manifest.items() if key != "sha256"}).encode())
    if actual != manifest.get("sha256") or actual != config.split_sha256:
        raise ContractError("Split manifest differs from its pinned SHA-256 identity")
    assignments = manifest["sources"]
    groups = {}
    for assignment in assignments.values():
        group, split = assignment["group_id"], assignment["split"]
        if split not in ("train", "validation", "test") or groups.setdefault(group, split) != split:
            raise ContractError("Split manifest has invalid or conflicting song-group assignments")
    for sha in config.source_sha256:
        if sha not in assignments or assignments[sha]["split"] != "train":
            raise ContractError("Selected source must belong to the pinned train split")
    return tuple(admit_source((Path(config.source_dir) / (sha + ".osu")).read_bytes(), sha,
                              group_id=assignments[sha]["group_id"], split=assignments[sha]["split"])
                 for sha in config.source_sha256)


def run_training(config: TrainExperimentConfig, *, resolved_yaml: str) -> dict:
    """Write configuration, draw paths, update metrics and final FP32 weights.

    The output directory must be absent or empty. Failed runs propagate the
    error; they do not publish final weights or substitute another device/data.
    """
    config.validate()
    if not config.output_dir:
        raise ContractError("Training requires an output_dir")
    output = Path(config.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ContractError("Training output_dir must be absent or empty")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ContractError("Requested MPS device is unavailable")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ContractError("Requested CUDA device is unavailable")
    sources = load_training_sources(config)
    sampler = WindowSampler(sources, config.windows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved.yaml").write_text(resolved_yaml)
    (output / "run-config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "population.json").write_text(json.dumps({
        "split_sha256": config.split_sha256, "groups": len(sampler.groups),
        "eligible_sources": [asdict(chart.source.identity) for chart in sampler.charts.values()],
        "excluded_sources": sampler.excluded,
    }, indent=2) + "\n")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(config.model_seed)
        model = CausalBackbone(config.model).to(config.device)
        trainer = SequenceTrainer(model, config.training, config.objective)
        prefill_rows = target_rows = 0
        with (output / "windows.jsonl").open("w") as draw_log, (output / "updates.jsonl").open("w") as update_log:
            for _ in range(config.updates):
                report = trainer.update(tuple(sampler.draw() for _ in range(config.training.effective_batch_size)))
                for index, window in enumerate(report.pop("windows")):
                    draw_log.write(json.dumps({"update": report["update"], "window": index,
                                               "sampling_seed": config.windows.seed, **window}, allow_nan=False) + "\n")
                prefill_rows += report["prefill_rows"]
                target_rows += report["target_rows"]
                report.update(cumulative_prefill_rows=prefill_rows, cumulative_target_rows=target_rows)
                update_log.write(json.dumps(report, allow_nan=False) + "\n")
                draw_log.flush()
                update_log.flush()
        torch.save({"format": "oracle-time-continuation/weights-v1", "model_config": asdict(config.model),
                    "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "updates": trainer.updates}, output / "weights.pt")
        return {"output_dir": str(output.resolve()), "updates": trainer.updates,
                "prefill_rows": prefill_rows, "target_rows": target_rows, "last_update": report}
    finally:
        torch.set_num_threads(previous_threads)
