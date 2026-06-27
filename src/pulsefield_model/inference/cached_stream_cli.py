from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pulsefield_model.data.control_windows import normalize_difficulty
from pulsefield_model.events.canonical import CanonicalTimepoint, LaneAction as CanonicalLaneAction
from pulsefield_model.inference.audio_probe import audio_length_ms_from_file
from pulsefield_model.inference.model_runtime import ModelRuntimeConfig, load_model_runtime, release_torch_cache
from pulsefield_model.inference.osu_export import OsuExportMetadata, format_osu_export
from pulsefield_model.inference.session_runtime import SessionRuntime, SessionRuntimeConfig
from pulsefield_model.inference.stream_windows import DecoderWindow, decoder_windows_until_audio_end
from pulsefield_model.models.mapper.shared.tokenizer import MAPPER_WRITE_MS
from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    TIMING_CANONICALIZATION_CHOICES,
    TIMING_CANONICALIZATION_NONE,
)
from pulsefield_model.timing.providers.beatthis import DEFAULT_BEATTHIS_DEVICE


DEFAULT_INDEX_PATH = Path("artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet")
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_OUTPUT_DIR = Path("artifacts/inference/mapper_v2_1_cached_stream_random_diff4")
T = TypeVar("T")


@dataclass(frozen=True)
class CandidateMap:
    row_index: int
    shard: str
    beatmap_id: int | None
    beatmap_set_id: int | None
    beatmap_path: Path
    audio_path: Path
    audio_filename: str
    title: str
    artist: str
    creator: str
    version: str
    hp_drain_rate: float
    overall_difficulty: float
    frame_count: int
    duration_s: float


def run_cached_stream_sample(argv: Sequence[str] | None = None) -> int:
    from pulsefield_model.inference.stream_with_cache import (
        DEFAULT_CONTROL_CHECKPOINT_PATH,
        DEFAULT_MAPPER_CHECKPOINT_PATH,
        DEFAULT_TIME_SHIFT_LENGTH_PENALTY,
        StreamWithCache,
        StreamWithCacheConfig,
    )

    args = _parse_args(
        argv,
        default_mapper_checkpoint_path=DEFAULT_MAPPER_CHECKPOINT_PATH,
        default_control_checkpoint_path=DEFAULT_CONTROL_CHECKPOINT_PATH,
        default_time_shift_length_penalty=DEFAULT_TIME_SHIFT_LENGTH_PENALTY,
    )
    difficulty = float(args.difficulty)
    normalize_difficulty(difficulty)

    seed = int(args.seed if args.seed is not None else time.time_ns() % (2**32))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_maps(
        index_path=Path(args.index_path),
        dataset_root=Path(args.dataset_root),
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
    )
    if len(candidates) < int(args.count):
        raise ValueError(f"not enough candidate maps after filtering: {len(candidates)} < {args.count}")
    sampled = sample_candidates(candidates, count=int(args.count), seed=seed)

    print(
        "inference_progress "
        f"status=sampled seed={seed} count={len(sampled)} pool={len(candidates)} "
        f"difficulty={difficulty:.3f} max_duration_s={args.max_duration_s}",
        flush=True,
    )
    for index, candidate in enumerate(sampled, start=1):
        print(
            "selected_map "
            f"index={index}/{len(sampled)} beatmap_id={candidate.beatmap_id} "
            f"duration_s={candidate.duration_s:.2f} audio={candidate.audio_path.as_posix()} "
            f"title={json.dumps(candidate.title, ensure_ascii=False)}",
            flush=True,
        )

    device = str(args.device)
    runtime = run_with_heartbeat(
        "runtime_load",
        lambda: load_model_runtime(
            ModelRuntimeConfig(
                mapper_checkpoint_path=Path(args.mapper_checkpoint_path),
                control_checkpoint_path=Path(args.control_checkpoint_path),
                mapper_profile=args.mapper_profile,
                device=device,
                beatthis_device=args.beatthis_device,
                beatthis_float16=bool(args.beatthis_float16),
                eager_load_beatthis=bool(args.eager_load_beatthis),
            ),
        ),
        interval_s=float(args.progress_interval_s),
    )
    config = StreamWithCacheConfig(
        mapper_checkpoint_path=Path(args.mapper_checkpoint_path),
        control_checkpoint_path=Path(args.control_checkpoint_path),
        mapper_profile=args.mapper_profile,
        device=device,
        beatthis_device=args.beatthis_device,
        beatthis_float16=bool(args.beatthis_float16),
        eager_load_beatthis=bool(args.eager_load_beatthis),
        default_difficulty=difficulty,
        max_control_batch_size=int(args.control_batch_size),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=args.top_p,
        use_incremental_mapper_decode=bool(args.use_incremental_mapper_decode),
        time_shift_length_penalty_alpha=float(args.time_shift_length_penalty_alpha),
        seed=args.generation_seed,
        token_send_interval_s=0.0,
        canonicalization=args.canonicalization,
    )
    stream = StreamWithCache(config)
    stream.model_runtime = runtime
    stream.models_ready = True

    reports = [
        run_candidate(
            stream=stream,
            candidate=candidate,
            index=index,
            total=len(sampled),
            difficulty=difficulty,
            output_dir=output_dir,
            device=device,
            control_batch_size=int(args.control_batch_size),
            precompute_full_control=bool(args.precompute_full_control),
            progress_interval_s=float(args.progress_interval_s),
        )
        for index, candidate in enumerate(sampled, start=1)
    ]

    manifest_path = output_dir / f"manifest_seed{seed}_diff{_difficulty_slug(difficulty)}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "difficulty": difficulty,
                "mapper_checkpoint_path": Path(args.mapper_checkpoint_path).as_posix(),
                "control_checkpoint_path": Path(args.control_checkpoint_path).as_posix(),
                "mapper_profile": args.mapper_profile,
                "device": device,
                "canonicalization": args.canonicalization,
                "count": len(reports),
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"inference_progress status=done manifest={manifest_path.as_posix()}", flush=True)
    return 0


def run_candidate(
    *,
    stream: Any,
    candidate: CandidateMap,
    index: int,
    total: int,
    difficulty: float,
    output_dir: Path,
    device: str,
    control_batch_size: int,
    precompute_full_control: bool,
    progress_interval_s: float,
) -> dict[str, Any]:
    from pulsefield_model.inference.stream_with_cache import _grid_fitter_config_for_canonicalization

    session_id = f"offline-{index}-{candidate.beatmap_id or candidate.row_index}"
    audio_length_ms = audio_length_ms_from_file(candidate.audio_path)
    audio_length_source = "file"
    if audio_length_ms is None:
        audio_length_ms = max(1, int(round(candidate.frame_count * 20.0)))
        audio_length_source = "frame_count"

    print(
        "map_progress "
        f"index={index}/{total} status=start beatmap_id={candidate.beatmap_id} "
        f"duration_s={audio_length_ms / 1000.0:.2f} audio_length_source={audio_length_source}",
        flush=True,
    )

    assert stream.model_runtime is not None
    session_runtime = SessionRuntime(
        session_id=session_id,
        model_runtime=stream.model_runtime,
        config=SessionRuntimeConfig(
            device=device,
            default_normalized_difficulty=normalize_difficulty(difficulty),
            max_control_batch_size=control_batch_size,
            grid_fitter_config=_grid_fitter_config_for_canonicalization(stream.config.canonicalization),
        ),
    )
    stream._session_runtimes[session_id] = session_runtime
    stream._last_context_token_by_session.pop(session_id, None)
    stream._last_carry_state_by_session.pop(session_id, None)

    try:
        audio_cache = run_with_heartbeat(
            f"map{index}_prepare_audio",
            lambda: session_runtime.prepare_audio(candidate.audio_path, audio_length_ms=audio_length_ms, start_ms=0),
            interval_s=progress_interval_s,
        )
        print(
            "map_progress "
            f"index={index}/{total} status=audio_ready source_frames={audio_cache.source_frame_count} "
            f"padded_frames={audio_cache.padded_frame_count}",
            flush=True,
        )

        if precompute_full_control:
            full_control = run_with_heartbeat(
                f"map{index}_full_control",
                lambda: session_runtime.prepare_full_control(max_batch_size=control_batch_size),
                interval_s=progress_interval_s,
            )
            print(
                "map_progress "
                f"index={index}/{total} status=control_ready windows={len(full_control.start_ms_values)} "
                f"batch_size={full_control.max_batch_size}",
                flush=True,
            )

        windows = decoder_windows_until_audio_end(
            DecoderWindow(start_ms=0, end_ms=MAPPER_WRITE_MS),
            audio_length_ms=audio_length_ms,
            config=stream.config,
        )
        generated_windows = []
        window_reports = []
        started_at = time.monotonic()
        for window_index, window in enumerate(windows, start=1):
            print(
                "window_progress "
                f"map={index}/{total} window={window_index}/{len(windows)} status=start "
                f"start_ms={window.start_ms} end_ms={window.end_ms}",
                flush=True,
            )
            window_started_at = time.monotonic()
            generated = run_with_heartbeat(
                f"map{index}_window{window_index:03d}",
                lambda window=window: stream._generate_window(
                    session_id,
                    session_runtime,
                    window,
                    audio_length_ms,
                ),
                interval_s=progress_interval_s,
            )
            elapsed_s = time.monotonic() - window_started_at
            generated_windows.append(generated)
            event_count = sum(1 for token_id in generated.tokens if stream._vocab().is_event_token(token_id))
            window_report = {
                "window_index": window_index,
                "start_ms": int(window.start_ms),
                "end_ms": int(window.end_ms),
                "tokens": len(generated.tokens),
                "events": event_count,
                "completed": bool(generated.completed),
                "dead_end": bool(generated.dead_end),
                "max_tokens_exceeded": bool(generated.max_tokens_exceeded),
                "terminal_ms": int(generated.terminal_state.current_ms),
                "elapsed_s": elapsed_s,
            }
            window_reports.append(window_report)
            print(
                "window_progress "
                f"map={index}/{total} window={window_index}/{len(windows)} status=done "
                f"tokens={len(generated.tokens)} events={event_count} completed={int(generated.completed)} "
                f"dead_end={int(generated.dead_end)} max_tokens_exceeded={int(generated.max_tokens_exceeded)} "
                f"terminal_ms={generated.terminal_state.current_ms} elapsed_s={elapsed_s:.2f}",
                flush=True,
            )

        timepoints = generated_windows_to_timepoints(generated_windows, stream._vocab())
        output_path = output_dir / output_filename(candidate, index=index, difficulty=difficulty)
        metadata = OsuExportMetadata(
            audio_filename=relative_audio_filename(candidate.audio_path, output_path),
            title=candidate.title,
            artist=candidate.artist,
            creator="Mapperatorinator",
            version=f"Mapper V2 cached stream diff {difficulty:.2f} / {candidate.version}",
            difficulty=difficulty,
            hp_drain_rate=candidate.hp_drain_rate,
            overall_difficulty=candidate.overall_difficulty,
        )
        timing_grid = session_runtime.audio_cache.timing_grid if session_runtime.audio_cache is not None else None
        output_path.write_text(
            format_osu_export(timepoints=timepoints, metadata=metadata, timing_grid=timing_grid),
            encoding="utf-8",
        )

        total_elapsed_s = time.monotonic() - started_at
        completed_windows = sum(1 for item in window_reports if item["completed"])
        report = {
            "status": "ok",
            "beatmap_id": candidate.beatmap_id,
            "beatmap_set_id": candidate.beatmap_set_id,
            "title": candidate.title,
            "artist": candidate.artist,
            "source_version": candidate.version,
            "audio_path": candidate.audio_path.as_posix(),
            "output_path": output_path.as_posix(),
            "audio_length_ms": audio_length_ms,
            "audio_length_source": audio_length_source,
            "window_count": len(windows),
            "completed_windows": completed_windows,
            "timepoint_count": len(timepoints),
            "elapsed_s": total_elapsed_s,
            "window_reports": window_reports,
        }
        print(
            "map_progress "
            f"index={index}/{total} status=done output={output_path.as_posix()} "
            f"windows={completed_windows}/{len(windows)} timepoints={len(timepoints)} elapsed_s={total_elapsed_s:.2f}",
            flush=True,
        )
        return report
    except Exception as exc:
        error_path = output_dir / f"FAILED_{index:02d}_{candidate.beatmap_id or candidate.row_index}.json"
        report = {
            "status": "error",
            "beatmap_id": candidate.beatmap_id,
            "beatmap_set_id": candidate.beatmap_set_id,
            "title": candidate.title,
            "artist": candidate.artist,
            "source_version": candidate.version,
            "audio_path": candidate.audio_path.as_posix(),
            "error": str(exc),
        }
        error_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "map_progress "
            f"index={index}/{total} status=error beatmap_id={candidate.beatmap_id} "
            f"error={json.dumps(str(exc), ensure_ascii=False)} details={error_path.as_posix()}",
            flush=True,
        )
        return report
    finally:
        stream._session_runtimes.pop(session_id, None)
        stream._last_context_token_by_session.pop(session_id, None)
        stream._last_carry_state_by_session.pop(session_id, None)
        session_runtime.reset_audio_cache()
        release_torch_cache(device)


def load_candidate_maps(
    *,
    index_path: Path,
    dataset_root: Path,
    min_duration_s: float | None,
    max_duration_s: float | None,
) -> list[CandidateMap]:
    import pandas as pd

    frame = pd.read_parquet(index_path)
    required = {
        "shard",
        "beatmap_path",
        "audio_path",
        "audio_filename",
        "title",
        "artist",
        "creator",
        "version",
        "beatmap_id",
        "beatmap_set_id",
        "hp_drain_rate",
        "overall_difficulty",
        "frame_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"index is missing required columns: {missing}")

    unique = frame.drop_duplicates(["beatmap_path"]).reset_index(drop=True)
    candidates: list[CandidateMap] = []
    for row_index, row in unique.iterrows():
        frame_count = _optional_int(row["frame_count"])
        if frame_count is None or frame_count <= 0:
            continue
        duration_s = float(frame_count) * 0.02
        if min_duration_s is not None and duration_s < float(min_duration_s):
            continue
        if max_duration_s is not None and duration_s > float(max_duration_s):
            continue
        shard = str(row["shard"])
        audio_path = resolve_dataset_path(dataset_root, shard, row["audio_path"])
        beatmap_path = resolve_dataset_path(dataset_root, shard, row["beatmap_path"])
        if not audio_path.is_file() or not beatmap_path.is_file():
            continue
        candidates.append(
            CandidateMap(
                row_index=int(row_index),
                shard=shard,
                beatmap_id=_optional_int(row["beatmap_id"]),
                beatmap_set_id=_optional_int(row["beatmap_set_id"]),
                beatmap_path=beatmap_path,
                audio_path=audio_path,
                audio_filename=str(row["audio_filename"]),
                title=_clean_text(row["title"], "Unknown Title"),
                artist=_clean_text(row["artist"], "Unknown Artist"),
                creator=_clean_text(row["creator"], "Unknown Creator"),
                version=_clean_text(row["version"], "Generated"),
                hp_drain_rate=_finite_float(row["hp_drain_rate"], default=5.0),
                overall_difficulty=_finite_float(row["overall_difficulty"], default=5.0),
                frame_count=frame_count,
                duration_s=duration_s,
            ),
        )
    return candidates


def sample_candidates(candidates: Sequence[CandidateMap], *, count: int, seed: int) -> list[CandidateMap]:
    return random.Random(seed).sample(list(candidates), k=count)


def generated_windows_to_timepoints(generated_windows: Sequence[Any], vocab: Any) -> list[CanonicalTimepoint]:
    timepoints: list[CanonicalTimepoint] = []
    for generated in generated_windows:
        for token_id, state_before in zip(generated.tokens, generated.states_before, strict=True):
            token_id = int(token_id)
            if not vocab.is_event_token(token_id):
                continue
            lane_actions = tuple(CanonicalLaneAction(action.value) for action in vocab.decode_event(token_id))
            timepoints.append(CanonicalTimepoint(time_ms=int(state_before.current_ms), lane_actions=lane_actions))
    return sorted(timepoints, key=lambda item: item.time_ms)


def run_with_heartbeat(label: str, fn: Callable[[], T], *, interval_s: float) -> T:
    interval_s = max(1.0, float(interval_s))
    started_at = time.monotonic()
    print(f"progress label={label} status=started", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        while True:
            try:
                result = future.result(timeout=interval_s)
            except concurrent.futures.TimeoutError:
                print(
                    f"progress label={label} status=running elapsed_s={time.monotonic() - started_at:.1f}",
                    flush=True,
                )
                continue
            elapsed_s = time.monotonic() - started_at
            print(f"progress label={label} status=done elapsed_s={elapsed_s:.1f}", flush=True)
            return result


def resolve_dataset_path(dataset_root: Path, shard: str, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    shard_path = dataset_root / shard / path
    if shard_path.exists():
        return shard_path
    return dataset_root / path


def output_filename(candidate: CandidateMap, *, index: int, difficulty: float) -> str:
    identity = candidate.beatmap_id or candidate.beatmap_set_id or candidate.row_index
    return f"{index:02d}_{identity}_mapper_cached_diff{_difficulty_slug(difficulty)}.osu"


def relative_audio_filename(audio_path: Path, output_path: Path) -> str:
    return os.path.relpath(audio_path, start=output_path.parent).replace(os.sep, "/")


def main(argv: Sequence[str] | None = None) -> int:
    return run_cached_stream_sample(argv)


def _parse_args(
    argv: Sequence[str] | None,
    *,
    default_mapper_checkpoint_path: Path,
    default_control_checkpoint_path: Path,
    default_time_shift_length_penalty: float,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mapper V2 cached full-song inference on indexed maps.")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--difficulty", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mapper-checkpoint-path", type=Path, default=default_mapper_checkpoint_path)
    parser.add_argument("--control-checkpoint-path", type=Path, default=default_control_checkpoint_path)
    parser.add_argument("--mapper-profile", choices=("auto", "v2_tuple", "v2_1_sparse"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--beatthis-device", default=DEFAULT_BEATTHIS_DEVICE)
    parser.add_argument("--beatthis-float16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eager-load-beatthis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--canonicalization",
        nargs="?",
        const=TIMING_CANONICALIZATION_BPM_80_160,
        default=TIMING_CANONICALIZATION_NONE,
        choices=TIMING_CANONICALIZATION_CHOICES,
        help="Fold fitted timing BPMs into [80, 160); pass 'none' to leave timing unchanged.",
    )
    parser.add_argument("--min-duration-s", type=float, default=45.0)
    parser.add_argument("--max-duration-s", type=float, default=120.0)
    parser.add_argument("--control-batch-size", type=int, default=4)
    parser.add_argument("--precompute-full-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument("--use-incremental-mapper-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-shift-length-penalty-alpha", type=float, default=default_time_shift_length_penalty)
    parser.add_argument("--progress-interval-s", type=float, default=15.0)
    args = parser.parse_args(argv)
    if int(args.count) <= 0:
        raise ValueError("--count must be positive")
    if int(args.control_batch_size) <= 0:
        raise ValueError("--control-batch-size must be positive")
    if int(args.max_tokens) <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.top_p is not None and not 0.0 < float(args.top_p) <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.min_duration_s is not None and float(args.min_duration_s) <= 0:
        raise ValueError("--min-duration-s must be positive")
    if args.max_duration_s is not None and float(args.max_duration_s) <= 0:
        raise ValueError("--max-duration-s must be positive")
    return args


def _difficulty_slug(difficulty: float) -> str:
    return f"{float(difficulty):.2f}".replace(".", "p")


def _clean_text(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    if not text or text.lower() == "nan":
        return default
    return text


def _optional_int(value: object) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: object, *, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(resolved):
        return float(default)
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
