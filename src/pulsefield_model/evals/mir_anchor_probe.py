from __future__ import annotations

import json
import math
import unicodedata
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from pulsefield_model.data.beatmap_index import load_index
from pulsefield_model.evals.mir_anchor_data import (
    CANDIDATE_FEATURE_DIM,
    HISTORY_FEATURE_DIM,
    ManiaAnchorRow,
    build_anchor_episodes,
    build_candidate_chart_features,
    build_episode_histories,
    build_same_gap_choice_sets,
    collapse_hit_objects_to_rows,
)
from pulsefield_model.evals.mir_anchor_metrics import choice_metrics
from pulsefield_model.evals.mir_anchor_metrics import exact_group_shapley
from pulsefield_model.evals.mir_anchor_metrics import paired_cluster_bootstrap
from pulsefield_model.evals.mir_anchor_metrics import paired_sign_flip_pvalue
from pulsefield_model.features.mir_backbone import (
    MIRBackboneConfig,
    MIRProbeFeatures,
    build_mir_backbone,
    mir_probe_features,
)
from pulsefield_model.osu_core.hitobjects import ManiaHitObject, parse_mania_hit_objects


IndexLoader: TypeAlias = Callable[[Path], pd.DataFrame]
HitObjectLoader: TypeAlias = Callable[[Path], Sequence[ManiaHitObject]]
AudioLoader: TypeAlias = Callable[[Path, int], object]
FeatureExtractor: TypeAlias = Callable[[object, int, MIRBackboneConfig], MIRProbeFeatures]

_REQUIRED_INDEX_COLUMNS = frozenset(("shard", "audio_path", "beatmap_path", "difficulty"))
_MANIFEST_COLUMNS = (
    "audio_id",
    "audio_group",
    "split",
    "shard",
    "audio_path",
    "beatmap_path",
    "beatmap_set_id",
    "audio_lead_in",
    "title",
    "artist",
    "creator",
    "version",
    "difficulty",
    "row_count",
    "episode_count",
    "structural_choice_count",
)
_FEATURE_SCHEMA = "mir_anchor_v1"
_CACHE_KEYS = frozenset(
    (
        "fast_frame_centers_s",
        "slow_frame_centers_s",
        "A",
        "N",
        "T",
        "P",
        "A_valid",
        "N_valid",
        "T_valid",
        "P_valid",
        "audio_duration_ms",
        "config_json",
        "feature_schema",
        "source_audio_path",
    )
)


@dataclass(frozen=True)
class MIRExtractionReport:
    audio_count: int
    written_count: int
    skipped_count: int


@dataclass(frozen=True)
class MIRProbeRunConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    max_choice_sets_per_song: int = 256
    train_choice_sets_per_batch: int = 32
    eval_choice_sets_per_batch: int = 32
    encoder_chunk_frames: int = 8_192
    encoder_max_fast_frames: int = 147_456
    controls_per_case: int = 16
    support_half_width_ms: int = 10
    max_gap_ms: int = 2_000
    history_rows: int = 32
    history_lookback_ms: int = 8_000
    time_scale_ms: int = 2_000
    choice_seed: int = 1_337
    seed: int = 0
    device: str = "auto"
    circular_shift_fraction: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "max_choice_sets_per_song",
            "train_choice_sets_per_batch",
            "eval_choice_sets_per_batch",
            "encoder_chunk_frames",
            "encoder_max_fast_frames",
            "controls_per_case",
            "max_gap_ms",
            "history_rows",
            "history_lookback_ms",
            "time_scale_ms",
        ):
            _validate_positive_integer(getattr(self, name), name)
        _validate_nonnegative_integer(self.support_half_width_ms, "support_half_width_ms")
        _validate_nonnegative_integer(self.choice_seed, "choice_seed")
        _validate_nonnegative_integer(self.seed, "seed")
        if self.support_half_width_ms >= self.max_gap_ms:
            raise ValueError("support_half_width_ms must be smaller than max_gap_ms.")
        for name in ("learning_rate", "grad_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative and finite.")
        if not math.isfinite(self.circular_shift_fraction) or not 0.0 < self.circular_shift_fraction < 1.0:
            raise ValueError("circular_shift_fraction must lie within (0,1).")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, mps, cuda.")


@dataclass(frozen=True)
class MIRProbeRunReport:
    seed: int
    parameter_count: int
    best_epoch: int
    best_validation_nll: float
    effective_audio_counts: Mapping[str, int]
    effective_case_counts: Mapping[str, int]
    state_path: Path
    per_audio_path: Path
    summary_path: Path


@dataclass(frozen=True)
class _ProbeSong:
    audio_id: str
    audio_group: str
    split: str
    rows: tuple[ManiaAnchorRow, ...]
    episode_indices: np.ndarray
    candidate_center_times_ms: np.ndarray
    episode_count: int
    eligible_choice_count: int
    audio_duration_ms: float
    feature_path: Path
    source_audio_path: Path
    source_beatmap_path: Path
    feature_config: MIRBackboneConfig


@dataclass(frozen=True)
class _SongFeatureArrays:
    features: Mapping[str, np.ndarray]
    clocks_ms: Mapping[str, np.ndarray]
    valid: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _SongScores:
    observed: Mapping[str, Any]
    shifted: Mapping[str, tuple[Any, Any]]


@dataclass(frozen=True)
class _EncodedGroups:
    sequences: Mapping[str, Any]
    lengths: Mapping[str, int]
    valid: Mapping[str, Any]
    clocks_ms: Mapping[str, np.ndarray]


PROBE_COALITIONS: tuple[tuple[str, ...], ...] = (
    (),
    ("A",),
    ("A", "N"),
    ("A", "P"),
    ("A", "T"),
    ("A", "N", "P"),
    ("A", "N", "T"),
    ("A", "P", "T"),
    ("A", "N", "P", "T"),
)


def prepare_mir_anchor_manifest(
    *,
    index_path: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    audio_count: int,
    seed: int,
    split_counts: Mapping[str, int] | None = None,
    target_difficulty: float = 4.0,
    controls_per_case: int = 16,
    support_half_width_ms: int = 10,
    max_gap_ms: int = 2_000,
    index_loader: IndexLoader = load_index,
    hitobject_loader: HitObjectLoader = parse_mania_hit_objects,
) -> pd.DataFrame:
    """Select one chart per audio and write the deterministic pilot manifest."""

    _validate_positive_integer(audio_count, "audio_count")
    _validate_nonnegative_integer(seed, "seed")
    if not math.isfinite(target_difficulty):
        raise ValueError("target_difficulty must be finite.")

    index = index_loader(Path(index_path))
    missing_columns = sorted(_REQUIRED_INDEX_COLUMNS.difference(index.columns))
    if missing_columns:
        raise ValueError(f"Beatmap index is missing required column(s): {missing_columns}.")

    root = Path(dataset_root).resolve()
    candidates_by_audio: dict[str, list[dict[str, object]]] = {}
    for source_row in index.to_dict(orient="records"):
        row = dict(source_row)
        difficulty = float(row["difficulty"])
        if not math.isfinite(difficulty):
            raise ValueError(f"Non-finite difficulty for beatmap {row['beatmap_path']!r}.")
        shard = str(row["shard"])
        audio_path = (root / shard / str(row["audio_path"])).resolve()
        beatmap_path = (root / shard / str(row["beatmap_path"])).resolve()
        row["difficulty"] = difficulty
        row["resolved_audio_path"] = str(audio_path)
        row["resolved_beatmap_path"] = str(beatmap_path)
        candidates_by_audio.setdefault(str(audio_path), []).append(row)

    if audio_count > len(candidates_by_audio):
        raise ValueError(
            f"Requested {audio_count} audio files, but the index contains only "
            f"{len(candidates_by_audio)} unique resolved audio paths."
        )

    selected_by_audio = {
        audio_path: min(
            candidates,
            key=lambda row: (
                abs(float(row["difficulty"]) - target_difficulty),
                str(row["resolved_beatmap_path"]),
            ),
        )
        for audio_path, candidates in candidates_by_audio.items()
    }
    selected_by_group: dict[str, dict[str, object]] = {}
    for audio_path, selected in selected_by_audio.items():
        audio_group = _metadata_audio_group(selected, fallback_audio_path=audio_path)
        candidate = dict(selected)
        candidate["audio_group"] = audio_group
        previous = selected_by_group.get(audio_group)
        if previous is None or (
            abs(float(candidate["difficulty"]) - target_difficulty),
            str(candidate["resolved_beatmap_path"]),
            str(candidate["resolved_audio_path"]),
        ) < (
            abs(float(previous["difficulty"]) - target_difficulty),
            str(previous["resolved_beatmap_path"]),
            str(previous["resolved_audio_path"]),
        ):
            selected_by_group[audio_group] = candidate
    if audio_count > len(selected_by_group):
        raise ValueError(
            f"Requested {audio_count} audio files, but only {len(selected_by_group)} "
            "normalized artist/title groups are available."
        )

    all_audio_groups = sorted(selected_by_group)
    rng = np.random.default_rng(seed)
    sampled_positions = rng.choice(len(all_audio_groups), size=audio_count, replace=False)
    selected_audio_groups = sorted(all_audio_groups[int(position)] for position in sampled_positions)
    split_by_audio_group = _audio_split(
        selected_audio_groups,
        rng=rng,
        split_counts=split_counts,
    )

    records: list[dict[str, object]] = []
    for position, audio_group in enumerate(selected_audio_groups):
        audio_id = f"audio_{position:05d}"
        selected = selected_by_group[audio_group]
        audio_path = str(selected["resolved_audio_path"])
        beatmap_path = Path(str(selected["resolved_beatmap_path"]))
        rows = collapse_hit_objects_to_rows(hitobject_loader(beatmap_path))
        episodes = build_anchor_episodes(rows, map_id=audio_id)
        choice_sets = build_same_gap_choice_sets(
            episodes,
            controls_per_case=controls_per_case,
            max_gap_ms=max_gap_ms,
            support_half_width_ms=support_half_width_ms,
            seed=seed + position,
        )
        records.append(
            {
                "audio_id": audio_id,
                "audio_group": audio_group,
                "split": split_by_audio_group[audio_group],
                "shard": str(selected["shard"]),
                "audio_path": audio_path,
                "beatmap_path": str(beatmap_path),
                "beatmap_set_id": _optional_text(selected, "beatmap_set_id"),
                "audio_lead_in": _optional_integer(selected, "audio_lead_in"),
                "title": _optional_text(selected, "title"),
                "artist": _optional_text(selected, "artist"),
                "creator": _optional_text(selected, "creator"),
                "version": _optional_text(selected, "version"),
                "difficulty": float(selected["difficulty"]),
                "row_count": len(rows),
                "episode_count": len(episodes),
                "structural_choice_count": len(choice_sets),
            }
        )

    manifest = pd.DataFrame.from_records(records, columns=_MANIFEST_COLUMNS)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(destination, index=False)
    return manifest


def extract_mir_anchor_features(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    config: MIRBackboneConfig = MIRBackboneConfig(),
    manifest_loader: IndexLoader = pd.read_parquet,
    audio_loader: AudioLoader | None = None,
    feature_extractor: FeatureExtractor | None = None,
) -> MIRExtractionReport:
    """Extract one uncompressed, independently resumable feature file per audio."""

    manifest = manifest_loader(Path(manifest_path))
    required_columns = frozenset(("audio_id", "audio_path"))
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing required column(s): {missing_columns}.")
    if manifest["audio_id"].duplicated().any():
        raise ValueError("Manifest audio_id values must be unique.")

    load_audio = audio_loader or _load_audio
    extract_features = feature_extractor or _extract_features
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written_count = 0
    skipped_count = 0

    ordered_manifest = manifest.sort_values("audio_id", kind="stable")
    for row in ordered_manifest.to_dict(orient="records"):
        audio_id = _safe_audio_id(row["audio_id"])
        source_audio_path = Path(str(row["audio_path"])).resolve()
        output_path = destination / f"{audio_id}.npz"
        if _is_valid_feature_file(
            output_path,
            config=config,
            source_audio_path=source_audio_path,
        ):
            skipped_count += 1
            continue

        waveform = load_audio(source_audio_path, config.sample_rate)
        features = extract_features(waveform, config.sample_rate, config)
        waveform_array = np.asarray(waveform)
        if waveform_array.ndim != 1 or waveform_array.size == 0:
            raise ValueError("Loaded audio must be a non-empty one-dimensional waveform.")
        payload = _feature_payload(
            features,
            config=config,
            audio_duration_ms=waveform_array.size * 1_000.0 / config.sample_rate,
            source_audio_path=source_audio_path,
        )
        _validate_feature_payload(
            payload,
            config=config,
            source_audio_path=source_audio_path,
        )
        with output_path.open("wb") as handle:
            np.savez(handle, **payload)
        written_count += 1

    return MIRExtractionReport(
        audio_count=len(manifest),
        written_count=written_count,
        skipped_count=skipped_count,
    )


def run_mir_anchor_probe(
    *,
    manifest_path: str | Path,
    feature_dir: str | Path,
    output_dir: str | Path,
    run_config: MIRProbeRunConfig | None = None,
    model_config: object | None = None,
    feature_config: MIRBackboneConfig | None = None,
    manifest_loader: IndexLoader = pd.read_parquet,
    hitobject_loader: HitObjectLoader = parse_mania_hit_objects,
) -> MIRProbeRunReport:
    """Fit and evaluate the matched conditional-choice probe.

    Each optimizer step is one song. All nine declared coalitions contribute
    equally to the step objective, so every reported additive head is trained.
    Evaluation controls are rebuilt from fixed per-song seeds.
    """

    import torch

    from pulsefield_model.evals.mir_anchor_model import MirAnchorProbe, MirAnchorProbeConfig

    run_config = run_config or MIRProbeRunConfig()
    torch.manual_seed(run_config.seed)
    rng = np.random.default_rng(run_config.seed)
    manifest = manifest_loader(Path(manifest_path))
    songs = _load_probe_songs(
        manifest,
        feature_dir=Path(feature_dir),
        run_config=run_config,
        feature_config=feature_config,
        model_config=model_config,
        hitobject_loader=hitobject_loader,
    )
    train_songs = tuple(song for song in songs if song.split == "train")
    validation_songs = tuple(song for song in songs if song.split == "validation")
    test_songs = tuple(song for song in songs if song.split == "test")
    if not train_songs:
        raise ValueError("Probe manifest has no eligible training songs.")
    if not validation_songs:
        raise ValueError("Probe manifest has no eligible validation songs.")
    if not test_songs:
        raise ValueError("Probe manifest has no eligible test songs.")

    feature_dimensions = _feature_dimensions(songs[0].feature_config)
    if model_config is None:
        resolved_model_config = MirAnchorProbeConfig(
            acoustic_dim=feature_dimensions["A"],
            novelty_dim=feature_dimensions["N"],
            tempogram_dim=feature_dimensions["T"],
            pulse_dim=feature_dimensions["P"],
        )
    elif isinstance(model_config, MirAnchorProbeConfig):
        resolved_model_config = model_config
    else:
        raise TypeError("model_config must be a MirAnchorProbeConfig.")
    configured_dimensions = {
        "A": resolved_model_config.acoustic_dim,
        "N": resolved_model_config.novelty_dim,
        "T": resolved_model_config.tempogram_dim,
        "P": resolved_model_config.pulse_dim,
    }
    dimension_mismatch = {
        group: {"cache": feature_dimensions[group], "model": configured_dimensions[group]}
        for group in configured_dimensions
        if configured_dimensions[group] != feature_dimensions[group]
    }
    if dimension_mismatch:
        raise ValueError(f"Model and cached feature dimensions differ: {dimension_mismatch}.")
    if resolved_model_config.candidate_dim != CANDIDATE_FEATURE_DIM:
        raise ValueError("Model candidate_dim does not match the causal chart feature width.")
    if resolved_model_config.history_dim != HISTORY_FEATURE_DIM:
        raise ValueError("Model history_dim does not match the causal history feature width.")

    device = _resolve_torch_device(torch, run_config.device)
    model = MirAnchorProbe(resolved_model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    best_epoch = -1
    best_validation_nll = math.inf
    best_state: dict[str, object] | None = None
    training_history: list[dict[str, float | int]] = []

    for epoch in range(run_config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for song_position in rng.permutation(len(train_songs)):
            song = train_songs[int(song_position)]
            choice_indexes = _sample_choice_indexes(
                song.candidate_center_times_ms.shape[0],
                maximum=run_config.max_choice_sets_per_song,
                rng=rng,
            )
            song_loss = _train_probe_song(
                model,
                song,
                choice_indexes,
                optimizer=optimizer,
                run_config=run_config,
                device=device,
            )
            if song_loss is None:
                continue
            epoch_losses.append(song_loss)
        if not epoch_losses:
            raise ValueError("No training choice set has valid support across all feature groups.")

        validation_nll = _validation_nll(
            model,
            validation_songs,
            run_config=run_config,
            device=device,
        )
        training_history.append(
            {
                "epoch": epoch,
                "training_nll": float(np.mean(epoch_losses)),
                "validation_nll": validation_nll,
            }
        )
        if validation_nll < best_validation_nll:
            best_epoch = epoch
            best_validation_nll = validation_nll
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    if best_state is None:
        raise ValueError("No validation choice set has valid support across all feature groups.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    per_audio, evaluation_summary = _evaluate_probe(
        model,
        songs,
        run_config=run_config,
        device=device,
    )
    effective_audio_counts, effective_case_counts = _effective_counts(per_audio)
    coverage = _coverage_summary(per_audio)
    parameter_count = model.parameter_count()

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "probe_state.pt"
    per_audio_path = destination / "per_audio.parquet"
    summary_path = destination / "summary.json"
    torch.save(
        {
            "model_state": best_state,
            "model_config": asdict(resolved_model_config),
            "run_config": asdict(run_config),
            "feature_config": asdict(songs[0].feature_config),
            "best_epoch": best_epoch,
            "best_validation_nll": best_validation_nll,
            "parameter_count": parameter_count,
        },
        state_path,
    )
    per_audio.to_parquet(per_audio_path, index=False)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation_nll,
        "seed": run_config.seed,
        "run_config": asdict(run_config),
        "model_config": asdict(resolved_model_config),
        "feature_config": asdict(songs[0].feature_config),
        "data_sources": {
            song.audio_id: {
                "audio_path": song.source_audio_path.as_posix(),
                "beatmap_path": song.source_beatmap_path.as_posix(),
                "audio_group": song.audio_group,
            }
            for song in songs
        },
        "parameter_count": parameter_count,
        "effective_audio_counts": effective_audio_counts,
        "effective_case_counts": effective_case_counts,
        "coverage": coverage,
        "training_history": training_history,
        "manifest_audio_count": len(manifest),
        "eligible_audio_count": len(songs),
        "coalitions": [_coalition_name(coalition) for coalition in PROBE_COALITIONS],
        "evaluation": evaluation_summary,
        "scientific_scope": {
            "estimand": (
                "incremental conditional anchor accessibility within a jointly trained "
                "multi-coalition probe"
            ),
            "split_unit": (
                "normalized artist/title identity with resolved-path fallback"
            ),
            "duplicate_limit": (
                "retitled or otherwise metadata-divergent copied/transcoded audio is not detected"
            ),
            "feature_claim": (
                "N/T/P are deterministic transforms of A, so gains measure useful inductive "
                "structure under this model and budget, not unique information or causality"
            ),
            "shift_interpretation": (
                "test-time alignment sensitivity, not an independently trained matched null"
            ),
        },
        "alignment_sensitivity": {
            "within_song_mir_circular_shift": ["N", "T", "P", "NTP"],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MIRProbeRunReport(
        seed=run_config.seed,
        parameter_count=parameter_count,
        best_epoch=best_epoch,
        best_validation_nll=best_validation_nll,
        effective_audio_counts=effective_audio_counts,
        effective_case_counts=effective_case_counts,
        state_path=state_path,
        per_audio_path=per_audio_path,
        summary_path=summary_path,
    )


def erode_valid_mask(valid: object, *, radius_frames: int) -> np.ndarray:
    """Require an encoder's complete centered receptive field to be valid."""

    _validate_nonnegative_integer(radius_frames, "radius_frames")
    mask = np.asarray(valid)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("valid must be a one-dimensional boolean mask.")
    if radius_frames == 0:
        return mask.copy()
    result = np.zeros_like(mask)
    if mask.size < 2 * radius_frames + 1:
        return result
    invalid_prefix = np.concatenate(([0], np.cumsum(~mask, dtype=np.int64)))
    centers = np.arange(radius_frames, mask.size - radius_frames)
    invalid_count = invalid_prefix[centers + radius_frames + 1] - invalid_prefix[centers - radius_frames]
    result[centers] = invalid_count == 0
    return result


def _load_probe_songs(
    manifest: pd.DataFrame,
    *,
    feature_dir: Path,
    run_config: MIRProbeRunConfig,
    feature_config: MIRBackboneConfig | None = None,
    model_config: object | None = None,
    hitobject_loader: HitObjectLoader,
) -> tuple[_ProbeSong, ...]:
    from pulsefield_model.evals.mir_anchor_model import MirAnchorProbeConfig

    required = frozenset(("audio_id", "audio_group", "split", "audio_path", "beatmap_path"))
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required probe column(s): {missing}.")
    if manifest["audio_id"].duplicated().any():
        raise ValueError("Manifest audio_id values must be unique.")

    songs: list[_ProbeSong] = []
    encoder_ineligible_audio_ids: list[str] = []
    ordered = manifest.sort_values("audio_id", kind="stable")
    inferred_model_config: MirAnchorProbeConfig | None = None
    expected_feature_config: MIRBackboneConfig | None = None
    for position, source_row in enumerate(ordered.to_dict(orient="records")):
        audio_id = _safe_audio_id(source_row["audio_id"])
        audio_group = str(source_row["audio_group"])
        if not audio_group:
            raise ValueError(f"Manifest audio_group must be non-empty for {audio_id}.")
        split = str(source_row["split"])
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown manifest split {split!r} for {audio_id}.")
        source_audio_path = Path(str(source_row["audio_path"])).resolve()
        source_beatmap_path = Path(str(source_row["beatmap_path"])).resolve()
        feature_path = feature_dir / f"{audio_id}.npz"
        cached = _load_probe_feature_file(
            feature_path,
            config=feature_config,
            source_audio_path=source_audio_path,
        )
        feature_dimensions = {group: int(np.asarray(cached[group]).shape[1]) for group in ("A", "N", "T", "P")}
        cached_feature_config = _config_from_json(str(np.asarray(cached["config_json"]).item()))
        if expected_feature_config is None:
            expected_feature_config = cached_feature_config
        elif cached_feature_config != expected_feature_config:
            raise ValueError("Cached teacher configs differ across probe songs.")
        if model_config is None and inferred_model_config is None:
            inferred_model_config = MirAnchorProbeConfig(
                acoustic_dim=feature_dimensions["A"],
                novelty_dim=feature_dimensions["N"],
                tempogram_dim=feature_dimensions["T"],
                pulse_dim=feature_dimensions["P"],
            )
        choice_model_config = inferred_model_config if model_config is None else model_config
        if not isinstance(choice_model_config, MirAnchorProbeConfig):
            raise TypeError("model_config must be a MirAnchorProbeConfig.")
        configured_dimensions = {
            "A": choice_model_config.acoustic_dim,
            "N": choice_model_config.novelty_dim,
            "T": choice_model_config.tempogram_dim,
            "P": choice_model_config.pulse_dim,
        }
        if configured_dimensions != feature_dimensions:
            raise ValueError(
                f"Model and cached feature dimensions differ for {audio_id}: "
                f"cache={feature_dimensions}, model={configured_dimensions}.",
            )
        if _encoder_limit_violations(
            cached,
            run_config=run_config,
            model_config=choice_model_config,
            feature_config=cached_feature_config,
        ):
            encoder_ineligible_audio_ids.append(audio_id)
            continue
        duration_ms = float(cached["audio_duration_ms"])
        support_valid = _common_support_valid_by_ms(
            cached,
            model_config=choice_model_config,
            audio_duration_ms=duration_ms,
            support_half_width_ms=run_config.support_half_width_ms,
        )
        rows = collapse_hit_objects_to_rows(hitobject_loader(source_beatmap_path))
        episodes = build_anchor_episodes(rows, map_id=audio_id)
        choice_sets = build_same_gap_choice_sets(
            episodes,
            controls_per_case=run_config.controls_per_case,
            max_gap_ms=run_config.max_gap_ms,
            support_half_width_ms=run_config.support_half_width_ms,
            seed=run_config.choice_seed + position,
            candidate_time_is_valid=lambda time_ms: (
                0 <= time_ms < support_valid.size and bool(support_valid[time_ms])
            ),
        )
        if not choice_sets:
            continue
        episode_indices = np.asarray(
            [choice.candidate_episode_indices for choice in choice_sets],
            dtype=np.int64,
        )
        candidate_center_times_ms = np.asarray(
            [choice.candidate_center_times_ms for choice in choice_sets],
            dtype=np.int64,
        )
        songs.append(
            _ProbeSong(
                audio_id=audio_id,
                audio_group=audio_group,
                split=split,
                rows=rows,
                episode_indices=episode_indices,
                candidate_center_times_ms=candidate_center_times_ms,
                episode_count=len(episodes),
                eligible_choice_count=len(choice_sets),
                audio_duration_ms=duration_ms,
                feature_path=feature_path,
                source_audio_path=source_audio_path,
                source_beatmap_path=source_beatmap_path,
                feature_config=cached_feature_config,
            )
        )
    if encoder_ineligible_audio_ids:
        warnings.warn(
            f"Skipping {len(encoder_ineligible_audio_ids)} probe song(s) whose cached features "
            f"exceed encoder_max_fast_frames={run_config.encoder_max_fast_frames}: "
            f"{', '.join(encoder_ineligible_audio_ids)}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return tuple(songs)


def _resolve_torch_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_song_feature_arrays(song: _ProbeSong) -> _SongFeatureArrays:
    cached = _load_probe_feature_file(
        song.feature_path,
        config=song.feature_config,
        source_audio_path=song.source_audio_path,
    )
    fast_clock_ms = np.asarray(cached["fast_frame_centers_s"], dtype=np.float64) * 1_000.0
    slow_clock_ms = np.asarray(cached["slow_frame_centers_s"], dtype=np.float64) * 1_000.0
    return _SongFeatureArrays(
        features={group: np.asarray(cached[group], dtype=np.float32) for group in ("A", "N", "T", "P")},
        clocks_ms={"A": fast_clock_ms, "N": fast_clock_ms, "T": slow_clock_ms, "P": fast_clock_ms},
        valid={group: np.asarray(cached[f"{group}_valid"], dtype=np.bool_) for group in ("A", "N", "T", "P")},
    )


def _load_probe_feature_file(
    path: Path,
    *,
    config: MIRBackboneConfig | None = None,
    source_audio_path: Path | None = None,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MIR probe feature file: {path}")
    with np.load(path, allow_pickle=False) as stored:
        missing = sorted(_CACHE_KEYS.difference(stored.files))
        if missing:
            raise ValueError(f"MIR probe feature file {path} is missing key(s): {missing}.")
        payload = {key: np.array(stored[key], copy=True) for key in _CACHE_KEYS}
    resolved_config = config or _config_from_json(str(np.asarray(payload["config_json"]).item()))
    try:
        _validate_feature_payload(
            payload,
            config=resolved_config,
            source_audio_path=source_audio_path,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MIR probe feature file {path}: {exc}") from exc
    return payload


def _common_support_valid_by_ms(
    cached: Mapping[str, np.ndarray],
    *,
    model_config: object,
    audio_duration_ms: float,
    support_half_width_ms: int,
) -> np.ndarray:
    """Precompute which integer-ms support centers are valid in every group."""

    from pulsefield_model.evals.mir_anchor_model import MirAnchorProbeConfig

    if not isinstance(model_config, MirAnchorProbeConfig):
        raise TypeError("model_config must be a MirAnchorProbeConfig.")
    fast_clock = np.asarray(cached["fast_frame_centers_s"], dtype=np.float64) * 1_000.0
    slow_clock = np.asarray(cached["slow_frame_centers_s"], dtype=np.float64) * 1_000.0
    clocks = {"A": fast_clock, "N": fast_clock, "T": slow_clock, "P": fast_clock}
    radii = {
        "A": model_config.acoustic_radius_frames,
        "N": model_config.high_rate_radius_frames,
        "T": model_config.tempogram_radius_frames,
        "P": model_config.high_rate_radius_frames,
    }
    integer_times = np.arange(math.ceil(audio_duration_ms) + 1, dtype=np.float64)
    common = np.ones(integer_times.size, dtype=np.bool_)
    for group in ("A", "N", "T", "P"):
        valid = np.asarray(cached[f"{group}_valid"], dtype=np.bool_)
        clock = clocks[group]
        if group == "A":
            valid = erode_valid_mask(
                valid,
                radius_frames=model_config.acoustic_pool_kernel // 2,
            )[:: model_config.acoustic_stride]
            clock = clock[:: model_config.acoustic_stride]
        valid = erode_valid_mask(valid, radius_frames=radii[group])
        common &= _interpolation_valid_at_times(clock, valid, integer_times)

    support_valid = np.zeros_like(common)
    width = 2 * support_half_width_ms + 1
    if common.size < width:
        return support_valid
    invalid_prefix = np.concatenate(([0], np.cumsum(~common, dtype=np.int64)))
    centers = np.arange(support_half_width_ms, common.size - support_half_width_ms)
    invalid_count = (
        invalid_prefix[centers + support_half_width_ms + 1]
        - invalid_prefix[centers - support_half_width_ms]
    )
    support_valid[centers] = invalid_count == 0
    return support_valid


def _interpolation_valid_at_times(
    clock_ms: np.ndarray,
    frame_valid: np.ndarray,
    times_ms: np.ndarray,
) -> np.ndarray:
    origin_ms, hop_ms = _uniform_clock(clock_ms, group="validity")
    positions = (times_ms - origin_ms) / hop_ms
    lower = np.floor(positions).astype(np.int64)
    exact = np.isclose(positions, lower, rtol=0.0, atol=1e-6)
    inside = (positions >= 0.0) & (positions <= frame_valid.size - 1)
    lower_safe = np.clip(lower, 0, frame_valid.size - 1)
    upper_safe = np.minimum(lower_safe + 1, frame_valid.size - 1)
    return inside & frame_valid[lower_safe] & (frame_valid[upper_safe] | exact)


def _sample_choice_indexes(
    choice_count: int,
    *,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if choice_count <= maximum:
        return np.arange(choice_count, dtype=np.int64)
    return np.sort(rng.choice(choice_count, size=maximum, replace=False)).astype(np.int64)


def _train_probe_song(
    model: Any,
    song: _ProbeSong,
    choice_indexes: np.ndarray,
    *,
    optimizer: Any,
    run_config: MIRProbeRunConfig,
    device: Any,
) -> float | None:
    """Run one bounded-memory optimizer step and release its graph on return."""

    import torch

    if choice_indexes.size == 0:
        return None
    optimizer.zero_grad(set_to_none=True)
    try:
        encoded = _encode_audio_groups(model, song, run_config=run_config, device=device)
        result = _backward_probe_choice_batches(
            model,
            song,
            encoded,
            choice_indexes,
            batch_size=run_config.train_choice_sets_per_batch,
            run_config=run_config,
            device=device,
        )
        if result is None:
            return None
        detached_loss_sum, valid_count = result
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(float(valid_count))
        torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.grad_clip_norm)
        optimizer.step()
        return float(detached_loss_sum.cpu()) / valid_count
    finally:
        optimizer.zero_grad(set_to_none=True)


def _backward_probe_choice_batches(
    model: Any,
    song: _ProbeSong,
    encoded: _EncodedGroups,
    choice_indexes: np.ndarray,
    *,
    batch_size: int,
    run_config: MIRProbeRunConfig,
    device: Any,
) -> tuple[Any, int] | None:
    """Accumulate bounded head batches, then traverse the encoder graph once."""

    import torch

    detached_encoded = _EncodedGroups(
        sequences={
            group: sequence.detach().requires_grad_(True)
            for group, sequence in encoded.sequences.items()
        },
        lengths=encoded.lengths,
        valid=encoded.valid,
        clocks_ms=encoded.clocks_ms,
    )
    detached_loss_sums = []
    valid_count = 0
    for first in range(0, choice_indexes.size, batch_size):
        last = min(first + batch_size, choice_indexes.size)
        result = _backward_probe_choice_batch(
            model,
            song,
            detached_encoded,
            choice_indexes[first:last],
            device=device,
            run_config=run_config,
        )
        if result is None:
            raise ValueError(
                "A training choice set lost feature support after manifest validation.",
            )
        detached_loss_sum, batch_valid_count = result
        if batch_valid_count != last - first:
            raise ValueError(
                "A training choice set lost feature support after manifest validation.",
            )
        detached_loss_sums.append(detached_loss_sum)
        valid_count += batch_valid_count

    if valid_count == 0:
        return None
    encoder_outputs = []
    encoder_gradients = []
    for group, output in encoded.sequences.items():
        gradient = detached_encoded.sequences[group].grad
        if gradient is None:
            raise RuntimeError(f"Feature group {group} did not contribute to the probe objective.")
        encoder_outputs.append(output)
        encoder_gradients.append(gradient)
    torch.autograd.backward(encoder_outputs, encoder_gradients)
    return torch.stack(detached_loss_sums).sum(), valid_count


def _backward_probe_choice_batch(
    model: Any,
    song: _ProbeSong,
    encoded: _EncodedGroups,
    choice_indexes: np.ndarray,
    *,
    device: Any,
    run_config: MIRProbeRunConfig,
) -> tuple[Any, int] | None:
    """Backpropagate an unnormalized choice sum for one bounded head batch."""

    import torch

    scores = _score_song_batch(
        model,
        song,
        encoded,
        choice_indexes,
        device=device,
        support_half_width_ms=run_config.support_half_width_ms,
        time_scale_ms=run_config.time_scale_ms,
        history_rows=run_config.history_rows,
        history_lookback_ms=run_config.history_lookback_ms,
        include_shift=False,
        circular_shift_fraction=run_config.circular_shift_fraction,
    )
    if scores is None:
        return None
    coalition_losses = [
        _support_choice_nll(values, case_index=0)
        for values in scores.observed.values()
    ]
    batch_valid_count = int(next(iter(scores.observed.values())).shape[0])
    loss = torch.stack(coalition_losses).mean()
    loss_sum = loss * batch_valid_count
    loss_sum.backward()
    return loss_sum.detach(), batch_valid_count


def _encode_audio_groups(
    model: Any,
    song: _ProbeSong,
    *,
    run_config: MIRProbeRunConfig,
    device: Any,
) -> _EncodedGroups:
    import torch
    from torch.nn import functional as torch_functional

    radii = {
        "A": model.config.acoustic_radius_frames,
        "N": model.config.high_rate_radius_frames,
        "T": model.config.tempogram_radius_frames,
        "P": model.config.high_rate_radius_frames,
    }
    sequences: dict[str, Any] = {}
    lengths: dict[str, int] = {}
    valid: dict[str, Any] = {}
    clocks_ms: dict[str, np.ndarray] = {}
    arrays = _load_song_feature_arrays(song)
    for group in ("A", "N", "T", "P"):
        source_features = torch.from_numpy(
            np.ascontiguousarray(arrays.features[group], dtype=np.float32),
        ).unsqueeze(0)
        source_valid = arrays.valid[group]
        source_clock = arrays.clocks_ms[group]
        prepooled = False
        if group == "A":
            source_features = torch_functional.avg_pool1d(
                source_features.transpose(1, 2),
                kernel_size=model.config.acoustic_pool_kernel,
                stride=model.config.acoustic_stride,
                padding=model.config.acoustic_pool_kernel // 2,
                count_include_pad=False,
            ).transpose(1, 2)
            prepooled = True
            source_valid = erode_valid_mask(
                source_valid,
                radius_frames=model.config.acoustic_pool_kernel // 2,
            )[:: model.config.acoustic_stride]
            source_clock = source_clock[:: model.config.acoustic_stride]

        actual_length = int(source_features.shape[1])
        bank_capacity = _encoder_bank_capacity(
            group,
            actual_length=actual_length,
            audio_id=song.audio_id,
            run_config=run_config,
            model_config=model.config,
            feature_config=song.feature_config,
        )
        feature_bank = source_features.new_zeros(
            (1, bank_capacity, source_features.shape[-1]),
        )
        feature_bank[:, :actual_length] = source_features
        encoded_bank, _ = model.encode_group_chunked(
            group,
            feature_bank.to(device),
            lengths=[actual_length],
            chunk_size=run_config.encoder_chunk_frames,
            bank_capacity=bank_capacity,
            prepooled=prepooled,
        )
        sequences[group] = encoded_bank
        lengths[group] = actual_length
        eroded = erode_valid_mask(source_valid, radius_frames=radii[group])
        if actual_length != source_clock.size:
            raise ValueError(f"Encoded {group} sequence does not match its projected clock.")
        valid[group] = eroded
        clocks_ms[group] = source_clock
    return _EncodedGroups(
        sequences=sequences,
        lengths=lengths,
        valid=valid,
        clocks_ms=clocks_ms,
    )


def _encoder_bank_capacity(
    group: str,
    *,
    actual_length: int,
    audio_id: str,
    run_config: MIRProbeRunConfig,
    model_config: Any,
    feature_config: MIRBackboneConfig,
) -> int:
    """Choose one of a bounded set of power-of-two chunk banks."""

    maximum = _encoder_group_frame_limit(
        group,
        run_config=run_config,
        model_config=model_config,
        feature_config=feature_config,
    )
    if actual_length > maximum:
        raise ValueError(
            f"Feature group {group} for {audio_id} has {actual_length} frames, "
            f"exceeding its configured limit {maximum}; increase encoder_max_fast_frames.",
        )
    required_chunks = math.ceil(actual_length / run_config.encoder_chunk_frames)
    maximum_chunks = math.ceil(maximum / run_config.encoder_chunk_frames)
    bucket_chunks = 1 << (required_chunks - 1).bit_length()
    return min(bucket_chunks, maximum_chunks) * run_config.encoder_chunk_frames


def _encoder_limit_violations(
    cached: Mapping[str, np.ndarray],
    *,
    run_config: MIRProbeRunConfig,
    model_config: Any,
    feature_config: MIRBackboneConfig,
) -> dict[str, tuple[int, int]]:
    """Return encoded group lengths that exceed the configured fast-frame budget."""

    cached_lengths = {
        group: int(np.asarray(cached[group]).shape[0])
        for group in ("A", "N", "T", "P")
    }
    cached_lengths["A"] = math.ceil(cached_lengths["A"] / model_config.acoustic_stride)
    violations = {}
    for group, actual_length in cached_lengths.items():
        maximum = _encoder_group_frame_limit(
            group,
            run_config=run_config,
            model_config=model_config,
            feature_config=feature_config,
        )
        if actual_length > maximum:
            violations[group] = (actual_length, maximum)
    return violations


def _encoder_group_frame_limit(
    group: str,
    *,
    run_config: MIRProbeRunConfig,
    model_config: Any,
    feature_config: MIRBackboneConfig,
) -> int:
    maximum = run_config.encoder_max_fast_frames
    if group == "A":
        return math.ceil(maximum / model_config.acoustic_stride)
    if group == "T":
        slow_stride = feature_config.tempogram_hop_ms // feature_config.mel_hop_ms
        return math.ceil(maximum / slow_stride)
    return maximum


def _score_song_batch(
    model: Any,
    song: _ProbeSong,
    encoded: _EncodedGroups,
    choice_indexes: np.ndarray,
    *,
    device: Any,
    support_half_width_ms: int,
    time_scale_ms: int,
    history_rows: int,
    history_lookback_ms: int,
    include_shift: bool,
    circular_shift_fraction: float,
) -> _SongScores | None:
    import torch

    from pulsefield_model.evals.mir_anchor_model import triangular_support_log_scores

    if choice_indexes.size == 0:
        return None
    episode_indices = song.episode_indices[choice_indexes]
    flat_episode_indices = episode_indices.reshape(-1)
    unique_episode_indices, inverse = np.unique(flat_episode_indices, return_inverse=True)
    unique_histories, unique_padding = build_episode_histories(
        song.rows,
        episode_indices=unique_episode_indices,
        history_rows=history_rows,
        lookback_ms=history_lookback_ms,
        gap_scale_ms=time_scale_ms,
    )
    history_indexes = inverse.reshape(episode_indices.shape)
    histories = torch.as_tensor(
        unique_histories[history_indexes],
        dtype=torch.float32,
        device=device,
    )
    padding = torch.as_tensor(unique_padding[history_indexes], dtype=torch.bool, device=device)
    strata, alternatives, history_rows, history_dim = histories.shape
    history_state = model.encode_history(
        histories.reshape(strata * alternatives, history_rows, history_dim),
        padding.reshape(strata * alternatives, history_rows),
    ).reshape(strata, alternatives, -1)
    offsets = np.arange(-support_half_width_ms, support_half_width_ms + 1, dtype=np.int64)
    query_times_ms = song.candidate_center_times_ms[choice_indexes, :, None] + offsets
    candidate_features = torch.as_tensor(
        build_candidate_chart_features(
            song.rows,
            episode_indices[..., None],
            query_times_ms,
            song_duration_ms=song.audio_duration_ms,
            time_scale_ms=time_scale_ms,
        ),
        dtype=torch.float32,
        device=device,
    )
    gathered, gathered_valid = _gather_audio_embeddings(
        song,
        encoded,
        query_times_ms,
        device=device,
    )
    common_valid = _common_stratum_valid(gathered_valid)
    if not bool(torch.any(common_valid)):
        return None

    valid_history_state = history_state[common_valid]
    valid_candidate_features = candidate_features[common_valid]
    valid_embeddings = {
        group: values[common_valid]
        for group, values in gathered.items()
    }
    observed: dict[str, Any] = {}
    for coalition in PROBE_COALITIONS:
        point_scores = model(
            history_state=valid_history_state,
            candidate_features=valid_candidate_features,
            embeddings=valid_embeddings,
            coalition=coalition,
        )
        observed[_coalition_name(coalition)] = triangular_support_log_scores(
            point_scores,
            half_width_ms=support_half_width_ms,
        )

    shift_results: dict[str, tuple[Any, Any]] = {}
    if include_shift:
        shifted_centers = np.mod(
            song.candidate_center_times_ms[choice_indexes]
            + song.audio_duration_ms * circular_shift_fraction,
            song.audio_duration_ms,
        )
        shifted_times = shifted_centers[:, :, None] + offsets
        shifted, shifted_valid = _gather_audio_embeddings(
            song,
            encoded,
            shifted_times,
            device=device,
        )
        for control_name, shifted_groups in (
            ("N", frozenset(("N",))),
            ("T", frozenset(("T",))),
            ("P", frozenset(("P",))),
            ("NTP", frozenset(("N", "T", "P"))),
        ):
            mixed = {
                group: shifted[group] if group in shifted_groups else gathered[group]
                for group in ("A", "N", "T", "P")
            }
            mixed_valid = {
                group: shifted_valid[group] if group in shifted_groups else gathered_valid[group]
                for group in ("A", "N", "T", "P")
            }
            paired_valid = common_valid & _common_stratum_valid(mixed_valid)
            if not bool(torch.any(paired_valid)):
                continue
            full_coalition = PROBE_COALITIONS[-1]
            reference_points = model(
                history_state=history_state[paired_valid],
                candidate_features=candidate_features[paired_valid],
                embeddings={group: values[paired_valid] for group, values in gathered.items()},
                coalition=full_coalition,
            )
            shifted_points = model(
                history_state=history_state[paired_valid],
                candidate_features=candidate_features[paired_valid],
                embeddings={group: values[paired_valid] for group, values in mixed.items()},
                coalition=full_coalition,
            )
            shift_results[control_name] = (
                triangular_support_log_scores(
                    reference_points,
                    half_width_ms=support_half_width_ms,
                ),
                triangular_support_log_scores(
                    shifted_points,
                    half_width_ms=support_half_width_ms,
                ),
            )
    return _SongScores(
        observed=observed,
        shifted=shift_results,
    )


def _gather_audio_embeddings(
    song: _ProbeSong,
    encoded: _EncodedGroups,
    query_times_ms: np.ndarray,
    *,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from pulsefield_model.evals.mir_anchor_model import interpolate_encoded_sequence

    queries = torch.as_tensor(query_times_ms, dtype=torch.float32, device=device)
    gathered: dict[str, Any] = {}
    gathered_valid: dict[str, Any] = {}
    for group in ("A", "N", "T", "P"):
        origin_ms, hop_ms = _uniform_clock(encoded.clocks_ms[group], group=group)
        values, _ = interpolate_encoded_sequence(
            encoded.sequences[group],
            queries,
            frame_origin_ms=origin_ms,
            frame_hop_ms=hop_ms,
        )
        if encoded.lengths[group] != encoded.clocks_ms[group].size:
            raise ValueError(f"Encoded {group} length does not match its projected clock.")
        valid = torch.as_tensor(
            _interpolation_valid_at_times(
                encoded.clocks_ms[group],
                encoded.valid[group],
                query_times_ms,
            ),
            dtype=torch.bool,
            device=device,
        )
        gathered[group] = values.masked_fill(~valid.unsqueeze(-1), 0.0)
        gathered_valid[group] = valid
    return gathered, gathered_valid


def _uniform_clock(clock_ms: np.ndarray, *, group: str) -> tuple[float, float]:
    values = np.asarray(clock_ms, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"Feature clock for {group} must contain at least two finite frames.")
    differences = np.diff(values)
    hop_ms = float(np.median(differences))
    if hop_ms <= 0.0 or not np.allclose(differences, hop_ms, rtol=0.0, atol=1e-5):
        raise ValueError(f"Feature clock for {group} must be strictly uniform.")
    return float(values[0]), hop_ms


def _common_stratum_valid(valid_by_group: Mapping[str, Any]) -> Any:
    import torch

    group_valid = [valid.reshape(valid.shape[0], -1).all(dim=1) for valid in valid_by_group.values()]
    return torch.stack(group_valid, dim=0).all(dim=0)


def _support_choice_nll(log_supports: Any, *, case_index: int) -> Any:
    import torch

    return (torch.logsumexp(log_supports, dim=-1) - log_supports[..., case_index]).mean()


def _validation_nll(
    model: Any,
    songs: Sequence[_ProbeSong],
    *,
    run_config: MIRProbeRunConfig,
    device: Any,
) -> float:
    model.eval()
    audio_losses: list[float] = []
    for song in songs:
        scores = _collect_song_scores(
            model,
            song,
            run_config=run_config,
            device=device,
            include_shift=False,
        )
        if scores is None:
            continue
        audio_losses.append(float(np.mean([choice_metrics(values).nll for values in scores.observed.values()])))
    if not audio_losses:
        return math.inf
    return float(np.mean(audio_losses))


def _collect_song_scores(
    model: Any,
    song: _ProbeSong,
    *,
    run_config: MIRProbeRunConfig,
    device: Any,
    include_shift: bool,
) -> _SongScores | None:
    import torch

    observed_parts: dict[str, list[np.ndarray]] = {
        _coalition_name(coalition): [] for coalition in PROBE_COALITIONS
    }
    shifted_parts: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        name: ([], []) for name in ("N", "T", "P", "NTP")
    }
    with torch.no_grad():
        encoded = _encode_audio_groups(model, song, run_config=run_config, device=device)
        for first in range(0, song.candidate_center_times_ms.shape[0], run_config.eval_choice_sets_per_batch):
            indexes = np.arange(
                first,
                min(first + run_config.eval_choice_sets_per_batch, song.candidate_center_times_ms.shape[0]),
                dtype=np.int64,
            )
            scores = _score_song_batch(
                model,
                song,
                encoded,
                indexes,
                device=device,
                support_half_width_ms=run_config.support_half_width_ms,
                time_scale_ms=run_config.time_scale_ms,
                history_rows=run_config.history_rows,
                history_lookback_ms=run_config.history_lookback_ms,
                include_shift=include_shift,
                circular_shift_fraction=run_config.circular_shift_fraction,
            )
            if scores is None:
                continue
            for name, values in scores.observed.items():
                observed_parts[name].append(values.detach().cpu().numpy())
            for name, (reference, shifted) in scores.shifted.items():
                shifted_parts[name][0].append(reference.detach().cpu().numpy())
                shifted_parts[name][1].append(shifted.detach().cpu().numpy())
    if not any(observed_parts.values()):
        return None
    observed = {
        name: np.concatenate(parts, axis=0)
        for name, parts in observed_parts.items()
    }
    return _SongScores(
        observed=observed,
        shifted={
            name: (np.concatenate(reference, axis=0), np.concatenate(shifted, axis=0))
            for name, (reference, shifted) in shifted_parts.items()
            if reference
        },
    )


def _evaluate_probe(
    model: Any,
    songs: Sequence[_ProbeSong],
    *,
    run_config: MIRProbeRunConfig,
    device: Any,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for song in songs:
        scores = _collect_song_scores(
            model,
            song,
            run_config=run_config,
            device=device,
            include_shift=True,
        )
        if scores is None:
            continue
        for coalition, values in scores.observed.items():
            rows.append(_metric_row(song, coalition=coalition, condition="observed", values=values))
        full_name = _coalition_name(PROBE_COALITIONS[-1])
        for name, (reference, shifted) in scores.shifted.items():
            for condition, values in (
                (f"{name}_shift_reference", reference),
                (f"{name}_circular_shift", shifted),
            ):
                rows.append(_metric_row(song, coalition=full_name, condition=condition, values=values))

    if not rows:
        raise ValueError("No evaluation choice set has valid support across all feature groups.")
    frame = pd.DataFrame.from_records(rows)
    summary: dict[str, object] = {}
    for scope in ("all", "train", "validation", "test"):
        scoped = frame if scope == "all" else frame[frame["split"] == scope]
        observed: dict[str, object] = {}
        for coalition in (_coalition_name(value) for value in PROBE_COALITIONS):
            selected = scoped[(scoped["condition"] == "observed") & (scoped["coalition"] == coalition)]
            if not selected.empty:
                observed[coalition] = _macro_audio_metric_summary(selected)
        negative: dict[str, object] = {}
        for condition in sorted(set(scoped["condition"]) - {"observed"}):
            selected = scoped[scoped["condition"] == condition]
            if not selected.empty:
                negative[condition] = _macro_audio_metric_summary(selected)
        if observed:
            summary[scope] = {
                "observed": observed,
                "within_song_circular_shift": negative,
                "paired_effects": _paired_effect_summaries(scoped),
                "mir_shapley": _conditional_mir_shapley(scoped),
            }
    return frame, summary


def _macro_audio_metric_summary(rows: pd.DataFrame) -> dict[str, float | int]:
    metric_columns = (
        "conditional_nll",
        "top1",
        "mean_reciprocal_rank",
        "pairwise_concordance",
        "mean_case_probability",
    )
    return {
        "audio_count": int(rows["audio_id"].nunique()),
        "choice_count": int(rows["choice_count"].sum()),
        **{name: float(rows[name].mean()) for name in metric_columns},
    }


def _paired_effect_summaries(rows: pd.DataFrame) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for name, comparison in _effect_comparisons().items():
        paired = _per_audio_nll_delta(rows, *comparison)
        if paired.empty:
            continue
        values = paired["value"].to_numpy(dtype=np.float64)
        summaries[name] = {
            "audio_count": int(values.size),
            "mean_nll_reduction": float(np.mean(values)),
        }
    return summaries


def _effect_comparisons() -> dict[str, tuple[str, str, str, str]]:
    full_name = _coalition_name(PROBE_COALITIONS[-1])
    return {
        "audio_over_history": ("observed", "H", "observed", "H+A"),
        "mir_over_acoustic": ("observed", "H+A", "observed", full_name),
        "full_over_history": ("observed", "H", "observed", full_name),
        "N_alignment_over_shift": ("N_circular_shift", full_name, "N_shift_reference", full_name),
        "T_alignment_over_shift": ("T_circular_shift", full_name, "T_shift_reference", full_name),
        "P_alignment_over_shift": ("P_circular_shift", full_name, "P_shift_reference", full_name),
        "mir_alignment_over_joint_shift": (
            "NTP_circular_shift",
            full_name,
            "NTP_shift_reference",
            full_name,
        ),
    }


def _conditional_mir_shapley(rows: pd.DataFrame) -> dict[str, object]:
    per_audio = _per_audio_mir_shapley(rows)
    report: dict[str, object] = {}
    for group, group_rows in per_audio.groupby("metric"):
        array = group_rows["value"].to_numpy(dtype=np.float64)
        report[str(group)] = {
            "audio_count": int(array.size),
            "mean_nll_reduction": float(np.mean(array)),
        }
    return report


def _per_audio_mir_shapley(rows: pd.DataFrame) -> pd.DataFrame:
    observed = rows[rows["condition"] == "observed"]
    coalition_names = {
        frozenset(group for group in coalition if group != "A"): _coalition_name(coalition)
        for coalition in PROBE_COALITIONS
        if "A" in coalition
    }
    records: list[dict[str, object]] = []
    for audio_id, audio_rows in observed.groupby("audio_id"):
        losses = dict(zip(audio_rows["coalition"], audio_rows["conditional_nll"]))
        if not all(name in losses for name in coalition_names.values()):
            continue
        values = {coalition: -float(losses[name]) for coalition, name in coalition_names.items()}
        for group, contribution in exact_group_shapley(values).items():
            records.append({"audio_id": audio_id, "metric": group, "value": contribution})
    return pd.DataFrame.from_records(records, columns=("audio_id", "metric", "value"))


def aggregate_mir_probe_runs(
    reports: Sequence[MIRProbeRunReport],
    *,
    output_path: str | Path,
    split: str = "test",
    bootstrap_seed: int = 1_337,
) -> dict[str, object]:
    """Aggregate predeclared seeds before audio-cluster inference."""

    if not reports:
        raise ValueError("reports must contain at least one completed seed")
    records: list[pd.DataFrame] = []
    reference_metadata: dict[str, object] | None = None
    reference_choices: list[dict[str, object]] | None = None
    for report in reports:
        summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
        run_metadata = dict(summary["run_config"])
        summary_seed = run_metadata.pop("seed")
        if summary_seed != report.seed or summary.get("seed") != report.seed:
            raise ValueError("Run report seed does not match its persisted summary metadata")
        metadata = {
            "run_config_without_seed": run_metadata,
            "model_config": summary["model_config"],
            "feature_config": summary["feature_config"],
            "data_sources": summary["data_sources"],
        }
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError("Cannot aggregate runs with different data or experiment configs")

        frame = pd.read_parquet(report.per_audio_path)
        scoped = frame[frame["split"] == split]
        choice_columns = (
            "audio_id",
            "split",
            "condition",
            "coalition",
            "episode_count",
            "eligible_choice_count",
            "choice_count",
        )
        choices = (
            scoped.loc[:, choice_columns]
            .sort_values(list(choice_columns[:-1]), kind="stable")
            .to_dict(orient="records")
        )
        if reference_choices is None:
            reference_choices = choices
        elif choices != reference_choices:
            raise ValueError("Cannot aggregate runs evaluated on different per-audio choice sets")
        for metric, comparison in _effect_comparisons().items():
            paired = _per_audio_nll_delta(scoped, *comparison)
            if not paired.empty:
                paired["metric"] = metric
                paired["seed"] = report.seed
                records.append(paired)
        shapley = _per_audio_mir_shapley(scoped)
        if not shapley.empty:
            shapley = shapley.copy()
            shapley["metric"] = "shapley_" + shapley["metric"].astype(str)
            shapley["seed"] = report.seed
            records.append(shapley)
    if not records:
        raise ValueError(f"No per-audio effects were available for split {split!r}")
    seed_effects = pd.concat(records, ignore_index=True)
    expected_seed_count = len(reports)
    counts = seed_effects.groupby(["audio_id", "metric"])["seed"].nunique()
    if not counts.eq(expected_seed_count).all():
        raise ValueError("Every aggregated audio effect must be present for every predeclared seed")
    per_audio = seed_effects.groupby(["audio_id", "metric"], as_index=False)["value"].mean()

    metrics: dict[str, object] = {}
    primary_metric = "mir_over_acoustic"
    for metric, metric_rows in per_audio.groupby("metric"):
        values = metric_rows["value"].to_numpy(dtype=np.float64)
        interval = paired_cluster_bootstrap(values, samples=10_000, seed=bootstrap_seed)
        seed_means = seed_effects[seed_effects["metric"] == metric].groupby("seed")["value"].mean()
        result = {
            "audio_count": int(values.size),
            "mean_nll_reduction": float(np.mean(values)),
            "bootstrap_95": [interval.lower, interval.upper],
            "between_seed_sd": float(seed_means.std(ddof=1)) if seed_means.size > 1 else 0.0,
            "inference_role": "primary" if metric == primary_metric else "exploratory",
        }
        if metric == primary_metric:
            result["one_sided_sign_flip_p"] = paired_sign_flip_pvalue(
                values,
                samples=10_000,
                seed=bootstrap_seed,
            )
        metrics[str(metric)] = result
    expected_metrics = set(_effect_comparisons()) | {"shapley_N", "shapley_T", "shapley_P"}
    required_metrics = {
        "audio_over_history",
        "mir_over_acoustic",
        "full_over_history",
        "shapley_N",
        "shapley_T",
        "shapley_P",
    }
    missing_required = sorted(required_metrics.difference(metrics))
    if missing_required:
        raise ValueError(f"Required aggregate metrics are unavailable: {missing_required}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    per_audio_path = destination.with_name(f"{destination.stem}_per_audio.parquet")
    per_audio.to_parquet(per_audio_path, index=False)
    result = {
        "split": split,
        "seeds": [report.seed for report in reports],
        "aggregation": "mean each audio over every predeclared seed, then infer over audio clusters",
        "seed_selection": False,
        "primary_metric": primary_metric,
        "multiplicity_policy": (
            "Only the primary metric carries a confirmatory p-value; all other intervals are exploratory."
        ),
        "unavailable_metrics": sorted(expected_metrics.difference(metrics)),
        "per_audio_path": str(per_audio_path),
        "metrics": metrics,
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _per_audio_nll_delta(
    rows: pd.DataFrame,
    baseline_condition: str,
    baseline_coalition: str,
    intervention_condition: str,
    intervention_coalition: str,
) -> pd.DataFrame:
    baseline = rows[
        (rows["condition"] == baseline_condition) & (rows["coalition"] == baseline_coalition)
    ][["audio_id", "conditional_nll"]].rename(columns={"conditional_nll": "baseline"})
    intervention = rows[
        (rows["condition"] == intervention_condition) & (rows["coalition"] == intervention_coalition)
    ][["audio_id", "conditional_nll"]].rename(columns={"conditional_nll": "intervention"})
    paired = baseline.merge(intervention, on="audio_id", validate="one_to_one")
    if paired.empty:
        return pd.DataFrame(columns=("audio_id", "value"))
    return pd.DataFrame(
        {
            "audio_id": paired["audio_id"],
            "value": paired["baseline"] - paired["intervention"],
        },
    )


def _metric_row(
    song: _ProbeSong,
    *,
    coalition: str,
    condition: str,
    values: np.ndarray,
) -> dict[str, object]:
    return {
        "audio_id": song.audio_id,
        "split": song.split,
        "coalition": coalition,
        "condition": condition,
        "episode_count": song.episode_count,
        "eligible_choice_count": song.eligible_choice_count,
        **_metric_summary(values),
    }


def _metric_summary(values: np.ndarray) -> dict[str, float | int]:
    metrics = choice_metrics(values)
    return {
        "choice_count": int(values.shape[0]),
        "conditional_nll": metrics.nll,
        "top1": metrics.top1,
        "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
        "pairwise_concordance": metrics.pairwise_concordance,
        "mean_case_probability": metrics.mean_case_probability,
    }


def _effective_counts(per_audio: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    baseline = per_audio[
        (per_audio["condition"] == "observed")
        & (per_audio["coalition"] == _coalition_name(PROBE_COALITIONS[0]))
    ]
    audio_counts = {split: 0 for split in ("train", "validation", "test")}
    case_counts = {split: 0 for split in ("train", "validation", "test")}
    for split, group in baseline.groupby("split"):
        audio_counts[str(split)] = int(group["audio_id"].nunique())
        case_counts[str(split)] = int(group["choice_count"].sum())
    return audio_counts, case_counts


def _coverage_summary(per_audio: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    baseline = per_audio[
        (per_audio["condition"] == "observed")
        & (per_audio["coalition"] == _coalition_name(PROBE_COALITIONS[0]))
    ]
    report: dict[str, dict[str, float | int]] = {}
    for split, rows in baseline.groupby("split"):
        episode_count = int(rows["episode_count"].sum())
        eligible_count = int(rows["eligible_choice_count"].sum())
        evaluated_count = int(rows["choice_count"].sum())
        report[str(split)] = {
            "episode_count": episode_count,
            "eligible_choice_count": eligible_count,
            "evaluated_choice_count": evaluated_count,
            "eligible_fraction_of_episodes": (
                eligible_count / episode_count if episode_count else 0.0
            ),
        }
    return report


def _coalition_name(coalition: Sequence[str]) -> str:
    return "H" if not coalition else "H+" + "+".join(coalition)


def _audio_split(
    audio_paths: Sequence[str],
    *,
    rng: np.random.Generator,
    split_counts: Mapping[str, int] | None = None,
) -> dict[str, str]:
    count = len(audio_paths)
    if split_counts is None:
        train_count = math.floor(0.70 * count)
        validation_count = math.floor(0.15 * count)
        test_count = count - train_count - validation_count
    else:
        expected = {"train", "validation", "test"}
        supplied = set(split_counts)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"split_counts keys differ: missing={missing}, extra={extra}")
        for name in sorted(expected):
            _validate_nonnegative_integer(split_counts[name], f"split_counts[{name!r}]")
        train_count = split_counts["train"]
        validation_count = split_counts["validation"]
        test_count = split_counts["test"]
        if train_count + validation_count + test_count != count:
            raise ValueError("split_counts must sum to audio_count")
    labels = (
        ["train"] * train_count
        + ["validation"] * validation_count
        + ["test"] * test_count
    )
    permutation = rng.permutation(count)
    return {audio_paths[int(position)]: labels[index] for index, position in enumerate(permutation)}


def _metadata_audio_group(row: Mapping[str, object], *, fallback_audio_path: str) -> str:
    values: list[str] = []
    for name in ("artist", "title"):
        if name not in row or pd.isna(row[name]):
            values = []
            break
        normalized = " ".join(
            unicodedata.normalize("NFKC", str(row[name])).casefold().split()
        )
        if not normalized:
            values = []
            break
        values.append(normalized)
    if values:
        return "metadata:" + json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return f"path:{fallback_audio_path}"


def _load_audio(path: Path, sample_rate: int) -> object:
    from pulsefield_model.features.audio import load_audio_file

    return load_audio_file(path, sample_rate=sample_rate)


def _extract_features(waveform: object, sample_rate: int, config: MIRBackboneConfig) -> MIRProbeFeatures:
    backbone = build_mir_backbone(waveform, sample_rate=sample_rate, config=config)
    return mir_probe_features(backbone, config=config)


def _feature_payload(
    features: MIRProbeFeatures,
    *,
    config: MIRBackboneConfig,
    audio_duration_ms: float,
    source_audio_path: Path,
) -> dict[str, np.ndarray]:
    return {
        "fast_frame_centers_s": np.asarray(features.fast_frame_centers_s, dtype=np.float64),
        "slow_frame_centers_s": np.asarray(features.slow_frame_centers_s, dtype=np.float64),
        "A": np.asarray(features.acoustic, dtype=np.float32),
        "N": np.asarray(features.novelty, dtype=np.float32),
        "T": np.asarray(features.tempogram, dtype=np.float32),
        "P": np.asarray(features.pulse, dtype=np.float32),
        "A_valid": np.asarray(features.acoustic_valid, dtype=np.bool_),
        "N_valid": np.asarray(features.novelty_valid, dtype=np.bool_),
        "T_valid": np.asarray(features.tempogram_valid, dtype=np.bool_),
        "P_valid": np.asarray(features.pulse_valid, dtype=np.bool_),
        "audio_duration_ms": np.asarray(audio_duration_ms, dtype=np.float64),
        "config_json": np.asarray(_config_json(config)),
        "feature_schema": np.asarray(_FEATURE_SCHEMA),
        "source_audio_path": np.asarray(source_audio_path.resolve().as_posix()),
    }


def _is_valid_feature_file(
    path: Path,
    *,
    config: MIRBackboneConfig,
    source_audio_path: Path,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as stored:
            if not _CACHE_KEYS.issubset(stored.files):
                return False
            _validate_feature_payload(
                {key: stored[key] for key in _CACHE_KEYS},
                config=config,
                source_audio_path=source_audio_path,
            )
    except (OSError, ValueError, TypeError, EOFError):
        return False
    return True


def _validate_feature_payload(
    payload: Mapping[str, np.ndarray],
    *,
    config: MIRBackboneConfig,
    source_audio_path: Path | None = None,
) -> None:
    schema = np.asarray(payload["feature_schema"])
    if schema.shape != () or str(schema.item()) != _FEATURE_SCHEMA:
        raise ValueError("Stored MIR feature schema does not match this implementation.")
    stored_audio_path = np.asarray(payload["source_audio_path"])
    if stored_audio_path.shape != ():
        raise ValueError("source_audio_path must be a scalar string.")
    if source_audio_path is not None and str(stored_audio_path.item()) != source_audio_path.resolve().as_posix():
        raise ValueError("Stored MIR features belong to a different source audio path.")
    stored_config = np.asarray(payload["config_json"])
    if stored_config.shape != () or str(stored_config.item()) != _config_json(config):
        raise ValueError("Stored MIR feature configuration does not match the requested configuration.")
    duration = np.asarray(payload["audio_duration_ms"])
    if duration.shape != () or not math.isfinite(float(duration)) or float(duration) <= 0.0:
        raise ValueError("audio_duration_ms must be a positive finite scalar.")

    fast_centers = np.asarray(payload["fast_frame_centers_s"])
    slow_centers = np.asarray(payload["slow_frame_centers_s"])
    expected_dimensions = _feature_dimensions(config)
    for name, expected_dimension in expected_dimensions.items():
        values = np.asarray(payload[name])
        if values.ndim != 2 or values.shape[1] != expected_dimension:
            raise ValueError(f"{name} must have shape (frames, {expected_dimension}), got {values.shape}.")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite floating-point values.")

        expected_frames = slow_centers.size if name == "T" else fast_centers.size
        if values.shape[0] != expected_frames:
            raise ValueError(f"{name} frame count does not match its feature clock.")
        valid = np.asarray(payload[f"{name}_valid"])
        if valid.dtype != np.bool_ or valid.shape != (expected_frames,):
            raise ValueError(f"{name}_valid must be a boolean vector matching {name}.")

    for name, centers in (
        ("fast_frame_centers_s", fast_centers),
        ("slow_frame_centers_s", slow_centers),
    ):
        if centers.ndim != 1 or not np.issubdtype(centers.dtype, np.floating):
            raise ValueError(f"{name} must be a floating-point vector.")
        if not np.all(np.isfinite(centers)) or np.any(np.diff(centers) <= 0.0):
            raise ValueError(f"{name} must be finite and strictly increasing.")
    for name, centers, expected_hop in (
        ("fast_frame_centers_s", fast_centers, config.mel_hop_seconds),
        ("slow_frame_centers_s", slow_centers, config.tempogram_hop_ms / 1_000.0),
    ):
        if centers.size and not math.isclose(float(centers[0]), 0.0, abs_tol=1e-9):
            raise ValueError(f"{name} must begin at audio time zero.")
        if centers.size > 1 and not np.allclose(
            np.diff(centers),
            expected_hop,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"{name} does not use the configured frame hop.")


def _safe_audio_id(value: object) -> str:
    audio_id = str(value)
    if not audio_id or audio_id in {".", ".."} or Path(audio_id).name != audio_id:
        raise ValueError(f"Unsafe manifest audio_id: {audio_id!r}.")
    return audio_id


def _optional_text(row: Mapping[str, object], name: str) -> str | None:
    if name not in row or pd.isna(row[name]):
        return None
    return str(row[name])


def _optional_integer(row: Mapping[str, object], name: str) -> int | None:
    if name not in row or pd.isna(row[name]):
        return None
    numeric = float(row[name])
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer when present.")
    return int(numeric)


def _config_json(config: MIRBackboneConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _config_from_json(serialized: str) -> MIRBackboneConfig:
    try:
        values = json.loads(serialized)
        if not isinstance(values, dict):
            raise TypeError("teacher config must be a JSON object")
        values["novelty_band_edges_hz"] = tuple(values["novelty_band_edges_hz"])
        values["tempogram_window_seconds"] = tuple(values["tempogram_window_seconds"])
        return MIRBackboneConfig(**values)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Stored MIR teacher configuration is invalid.") from exc


def _feature_dimensions(config: MIRBackboneConfig) -> dict[str, int]:
    return {
        "A": config.mel_bins,
        "N": 5,
        "T": config.tempo_bins + 26,
        "P": 6,
    }


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
