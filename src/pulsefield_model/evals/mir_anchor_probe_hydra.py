from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from pulsefield_model.features.mir_backbone import MIRBackboneConfig


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "hydra"
_REGISTERED = False
_DEFAULT_TEACHER = MIRBackboneConfig()


@dataclass
class DataSection:
    index_path: str = "artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet"
    dataset_root: str = "dataset"
    manifest_path: str = "artifacts/evals/mir_anchor_probe_750_150_150_3seed/manifest.parquet"
    train_audio_count: int = 750
    validation_audio_count: int = 150
    test_audio_count: int = 150
    target_difficulty: float = 4.0


@dataclass
class RiskSection:
    controls_per_case: int = 16
    max_gap_ms: int = 2_000
    support_half_width_ms: int = 10


@dataclass
class TeacherSection:
    sample_rate: int = _DEFAULT_TEACHER.sample_rate
    mel_bins: int = _DEFAULT_TEACHER.mel_bins
    mel_hop_ms: int = _DEFAULT_TEACHER.mel_hop_ms
    mel_window_ms: int = _DEFAULT_TEACHER.mel_window_ms
    fmin_hz: float = _DEFAULT_TEACHER.fmin_hz
    fmax_hz: float = _DEFAULT_TEACHER.fmax_hz
    log_mel_floor: float = _DEFAULT_TEACHER.log_mel_floor
    novelty_band_edges_hz: list[float] = field(
        default_factory=lambda: list(_DEFAULT_TEACHER.novelty_band_edges_hz),
    )
    novelty_diff_frames: int = _DEFAULT_TEACHER.novelty_diff_frames
    tempogram_hop_ms: int = _DEFAULT_TEACHER.tempogram_hop_ms
    novelty_local_average_ms: int = _DEFAULT_TEACHER.novelty_local_average_ms
    novelty_clip: float = _DEFAULT_TEACHER.novelty_clip
    tempogram_window_seconds: list[float] = field(
        default_factory=lambda: list(_DEFAULT_TEACHER.tempogram_window_seconds),
    )
    tempo_min_bpm: float = _DEFAULT_TEACHER.tempo_min_bpm
    tempo_max_bpm: float = _DEFAULT_TEACHER.tempo_max_bpm
    tempo_bins: int = _DEFAULT_TEACHER.tempo_bins


@dataclass
class ProbeSection:
    history_rows: int = 32
    history_lookback_ms: int = 8_000
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_choice_sets_per_song: int = 256
    train_choice_sets_per_batch: int = 32
    eval_choice_sets_per_batch: int = 32
    encoder_chunk_frames: int = 8_192
    encoder_max_fast_frames: int = 147_456
    gradient_clip_norm: float = 1.0
    time_scale_ms: int = 2_000
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    device: str = "auto"
    circular_shift_fraction: float = 1.0 / 3.0


@dataclass
class OutputSection:
    dir: str = "artifacts/evals/mir_anchor_probe_750_150_150_3seed"


@dataclass
class MIRAnchorExperimentConfig:
    stage: str = "prepare"
    seed: int = 1_337
    data: DataSection = field(default_factory=DataSection)
    risk: RiskSection = field(default_factory=RiskSection)
    teacher: TeacherSection = field(default_factory=TeacherSection)
    probe: ProbeSection = field(default_factory=ProbeSection)
    output: OutputSection = field(default_factory=OutputSection)


def register_mir_anchor_probe_config() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    ConfigStore.instance().store(name="mir_anchor_probe_schema", node=MIRAnchorExperimentConfig)
    _REGISTERED = True


register_mir_anchor_probe_config()


def compose_mir_anchor_probe_config(
    overrides: Sequence[str] = (),
    *,
    config_dir: Path | None = None,
) -> MIRAnchorExperimentConfig:
    resolved_dir = CONFIGS_DIR if config_dir is None else Path(config_dir).resolve()
    with initialize_config_dir(version_base="1.3", config_dir=resolved_dir.as_posix()):
        config = compose(config_name="mir_anchor_probe", overrides=list(overrides))
    return mir_anchor_probe_config_from_hydra(config)


def mir_anchor_probe_config_from_hydra(config: DictConfig | MIRAnchorExperimentConfig) -> MIRAnchorExperimentConfig:
    source = config if OmegaConf.is_config(config) else OmegaConf.structured(config)
    resolved = OmegaConf.to_container(source, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("MIR anchor experiment config must resolve to a mapping")
    _reject_unknown_config_keys(resolved)
    merged = OmegaConf.merge(OmegaConf.structured(MIRAnchorExperimentConfig), source)
    typed = OmegaConf.to_object(merged)
    if not isinstance(typed, MIRAnchorExperimentConfig):
        raise TypeError("MIR anchor experiment config did not project to its typed schema")
    _validate_config(typed)
    return typed


def run_mir_anchor_stage(config: MIRAnchorExperimentConfig) -> object:
    """Project the typed process config into one research-stage call."""

    _validate_config(config)
    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(config), output_dir / f"resolved_{config.stage}.yaml")

    from pulsefield_model.evals import mir_anchor_probe

    teacher = teacher_config_from_section(config.teacher)

    if config.stage == "prepare":
        split_counts = {
            "train": config.data.train_audio_count,
            "validation": config.data.validation_audio_count,
            "test": config.data.test_audio_count,
        }
        return mir_anchor_probe.prepare_mir_anchor_manifest(
            index_path=config.data.index_path,
            dataset_root=config.data.dataset_root,
            output_path=config.data.manifest_path,
            audio_count=sum(split_counts.values()),
            split_counts=split_counts,
            seed=config.seed,
            target_difficulty=config.data.target_difficulty,
            controls_per_case=config.risk.controls_per_case,
            support_half_width_ms=config.risk.support_half_width_ms,
            max_gap_ms=config.risk.max_gap_ms,
        )
    if config.stage == "extract":
        return mir_anchor_probe.extract_mir_anchor_features(
            manifest_path=config.data.manifest_path,
            output_dir=output_dir / "features",
            config=teacher,
        )
    reports = []
    for seed in config.probe.seeds:
        run_config = probe_run_config_from_sections(config, seed=seed)
        reports.append(
            mir_anchor_probe.run_mir_anchor_probe(
                manifest_path=config.data.manifest_path,
                feature_dir=output_dir / "features",
                output_dir=output_dir / "runs" / f"seed_{seed}",
                run_config=run_config,
                feature_config=teacher,
            ),
        )
    inference = mir_anchor_probe.aggregate_mir_probe_runs(
        reports,
        output_path=output_dir / "multi_seed_inference.json",
        split="test",
        bootstrap_seed=config.seed,
    )
    report_path = output_dir / "multi_seed_report.json"
    report_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "seed": report.seed,
                        "parameter_count": report.parameter_count,
                        "best_epoch": report.best_epoch,
                        "best_validation_nll": report.best_validation_nll,
                        "effective_audio_counts": report.effective_audio_counts,
                        "effective_case_counts": report.effective_case_counts,
                        "summary_path": str(report.summary_path),
                    }
                    for report in reports
                ],
                "inference": inference,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(reports)


def teacher_config_from_section(section: TeacherSection) -> MIRBackboneConfig:
    return MIRBackboneConfig(
        sample_rate=section.sample_rate,
        mel_bins=section.mel_bins,
        mel_hop_ms=section.mel_hop_ms,
        mel_window_ms=section.mel_window_ms,
        fmin_hz=section.fmin_hz,
        fmax_hz=section.fmax_hz,
        log_mel_floor=section.log_mel_floor,
        novelty_band_edges_hz=tuple(section.novelty_band_edges_hz),
        novelty_diff_frames=section.novelty_diff_frames,
        tempogram_hop_ms=section.tempogram_hop_ms,
        novelty_local_average_ms=section.novelty_local_average_ms,
        novelty_clip=section.novelty_clip,
        tempogram_window_seconds=tuple(section.tempogram_window_seconds),
        tempo_min_bpm=section.tempo_min_bpm,
        tempo_max_bpm=section.tempo_max_bpm,
        tempo_bins=section.tempo_bins,
    )


def probe_run_config_from_sections(config: MIRAnchorExperimentConfig, *, seed: int):
    """Project Hydra sections through the runtime config's sole validator."""

    from pulsefield_model.evals.mir_anchor_probe import MIRProbeRunConfig

    return MIRProbeRunConfig(
        epochs=config.probe.epochs,
        learning_rate=config.probe.learning_rate,
        weight_decay=config.probe.weight_decay,
        grad_clip_norm=config.probe.gradient_clip_norm,
        max_choice_sets_per_song=config.probe.max_choice_sets_per_song,
        train_choice_sets_per_batch=config.probe.train_choice_sets_per_batch,
        eval_choice_sets_per_batch=config.probe.eval_choice_sets_per_batch,
        encoder_chunk_frames=config.probe.encoder_chunk_frames,
        encoder_max_fast_frames=config.probe.encoder_max_fast_frames,
        controls_per_case=config.risk.controls_per_case,
        support_half_width_ms=config.risk.support_half_width_ms,
        max_gap_ms=config.risk.max_gap_ms,
        history_rows=config.probe.history_rows,
        history_lookback_ms=config.probe.history_lookback_ms,
        time_scale_ms=config.probe.time_scale_ms,
        choice_seed=config.seed,
        seed=seed,
        device=config.probe.device,
        circular_shift_fraction=config.probe.circular_shift_fraction,
    )


def _validate_config(config: MIRAnchorExperimentConfig) -> None:
    if config.stage not in {"prepare", "extract", "probe"}:
        raise ValueError("stage must be one of: prepare, extract, probe")
    for name in ("train_audio_count", "validation_audio_count", "test_audio_count"):
        _positive_int(getattr(config.data, name), f"data.{name}")
    _nonnegative_int(config.seed, "seed")
    for name in ("index_path", "dataset_root", "manifest_path"):
        if not str(getattr(config.data, name)):
            raise ValueError(f"data.{name} must be non-empty")
    if not str(config.output.dir):
        raise ValueError("output.dir must be non-empty")
    if len(config.teacher.tempogram_window_seconds) != 1:
        raise ValueError("teacher.tempogram_window_seconds must contain exactly one scale")
    if not config.probe.seeds or any(isinstance(seed, bool) or seed < 0 for seed in config.probe.seeds):
        raise ValueError("probe.seeds must contain non-negative integers")
    if len(set(config.probe.seeds)) != len(config.probe.seeds):
        raise ValueError("probe.seeds must be unique")
    teacher_config_from_section(config.teacher)
    for seed in config.probe.seeds:
        probe_run_config_from_sections(config, seed=seed)


def _reject_unknown_config_keys(config: Mapping[object, object]) -> None:
    _reject_unknown_section(config, MIRAnchorExperimentConfig, "config")
    section_types = {
        "data": DataSection,
        "risk": RiskSection,
        "teacher": TeacherSection,
        "probe": ProbeSection,
        "output": OutputSection,
    }
    for name, section_type in section_types.items():
        section = config.get(name)
        if not isinstance(section, Mapping):
            raise TypeError(f"{name} must be a mapping")
        _reject_unknown_section(section, section_type, name)


def _reject_unknown_section(config: Mapping[object, object], schema: type[object], name: str) -> None:
    allowed = {item.name for item in fields(schema)}
    unknown = sorted(str(key) for key in config if key not in allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@hydra_main(version_base="1.3", config_path="../configs/hydra", config_name="mir_anchor_probe")
def _hydra_main(config: DictConfig) -> None:
    run_mir_anchor_stage(mir_anchor_probe_config_from_hydra(config))


def main(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    with _patched_argv([sys.argv[0], *args]):
        _hydra_main()
    return 0


@contextmanager
def _patched_argv(argv: Sequence[str]):
    original = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
