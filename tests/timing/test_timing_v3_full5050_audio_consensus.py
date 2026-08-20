from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from pulsefield_model.features.mel import pack_full_song_mel_20ms
from pulsefield_model.timing.evaluation.full5050_audio_consensus import (
    BEATNET_VIEW_NAME,
    BEATTHIS_VIEW_NAME,
    FULL5050_AUDIO_CONSENSUS_PLAN_SCHEMA,
    FULL5050_BEATNET_EVENTS_SCHEMA,
    RAW_FLUX_VIEW_NAME,
    AudioConsensusScreenConfig,
    Full5050AudioConsensusPipeline,
    Full5050AudioConsensusRunnerConfig,
    MissingBeatNetEventsError,
    beat_times_to_impulse_salience,
    evaluate_view_stability,
    load_precomputed_beatnet_event_store,
    main,
    run_full5050_audio_consensus,
    run_full5050_audio_consensus_row,
    screen_audio_consensus,
    screen_natural_change_consensus,
    select_constant_sources_from_results,
    select_natural_change_sources_from_results,
)
from pulsefield_model.timing.evaluation.full5050_shadow_runner import Full5050LocatorRow
from pulsefield_model.timing.providers.beatthis_cache import BeatThisFramePredictionCacheConfig
from pulsefield_model.timing.schema import FrameTimingPrediction


def test_screen_accepts_same_octave_family_across_three_correlated_views() -> None:
    config = _screen_config()
    duration_seconds = 12.0
    beatthis = evaluate_view_stability(
        BEATTHIS_VIEW_NAME,
        _periodic_salience(duration_seconds, frame_rate_hz=100.0, bpm=120.0),
        frame_rate_hz=100.0,
        config=config,
    )
    raw = evaluate_view_stability(
        RAW_FLUX_VIEW_NAME,
        _periodic_salience(duration_seconds, frame_rate_hz=100.0, bpm=60.0),
        frame_rate_hz=100.0,
        config=config,
    )
    beatnet = evaluate_view_stability(
        BEATNET_VIEW_NAME,
        _periodic_salience(duration_seconds, frame_rate_hz=100.0, bpm=240.0),
        frame_rate_hz=100.0,
        config=config,
    )

    consensus = screen_audio_consensus(
        {
            BEATTHIS_VIEW_NAME: beatthis,
            RAW_FLUX_VIEW_NAME: raw,
            BEATNET_VIEW_NAME: beatnet,
        },
        config=config,
    )

    assert consensus.accepted_constant_source is True
    assert consensus.reason is None
    assert consensus.global_octave_family_bpm == pytest.approx(120.0, abs=1.0)


def test_constant_consensus_allows_different_tracker_tactus_when_each_view_is_stable() -> None:
    config = _screen_config()
    duration_seconds = 12.0
    beatthis = evaluate_view_stability(
        BEATTHIS_VIEW_NAME,
        _periodic_salience(duration_seconds, frame_rate_hz=100.0, bpm=100.0),
        frame_rate_hz=100.0,
        config=config,
    )
    raw = evaluate_view_stability(
        RAW_FLUX_VIEW_NAME,
        _periodic_salience(duration_seconds, frame_rate_hz=100.0, bpm=200.0),
        frame_rate_hz=100.0,
        config=config,
    )
    beatnet = evaluate_view_stability(
        BEATNET_VIEW_NAME,
        _periodic_salience(duration_seconds, frame_rate_hz=100.0, bpm=50.0),
        frame_rate_hz=100.0,
        config=config,
    )

    consensus = screen_audio_consensus(
        {
            BEATTHIS_VIEW_NAME: beatthis,
            RAW_FLUX_VIEW_NAME: raw,
            BEATNET_VIEW_NAME: beatnet,
        },
        config=config,
    )

    assert beatthis.global_estimate.direct_bpm == pytest.approx(100.0)
    assert raw.global_estimate.direct_bpm == pytest.approx(200.0)
    assert beatnet.global_estimate.direct_bpm == pytest.approx(50.0)
    assert beatthis.stable is True
    assert raw.stable is True
    assert beatnet.stable is True
    assert consensus.accepted_constant_source is True
    assert consensus.global_octave_family_bpm == pytest.approx(100.0, abs=1.0)


def test_view_rejects_persistent_tempo_family_change() -> None:
    config = _screen_config()
    salience = np.concatenate(
        [
            _periodic_salience(6.0, frame_rate_hz=100.0, bpm=120.0),
            _periodic_salience(6.0, frame_rate_hz=100.0, bpm=150.0),
        ],
    )

    view = evaluate_view_stability(
        BEATTHIS_VIEW_NAME,
        salience,
        frame_rate_hz=100.0,
        config=config,
    )

    assert view.stable is False
    assert view.reason == "persistent_family_change"
    assert view.max_persistent_mismatch_run >= config.persistent_run_windows


def test_three_views_direct_octave_jump_is_natural_change_and_not_constant() -> None:
    config = _screen_config()
    views = {
        BEATTHIS_VIEW_NAME: evaluate_view_stability(
            BEATTHIS_VIEW_NAME,
            _change_salience(first_bpm=100.0, second_bpm=200.0),
            frame_rate_hz=100.0,
            config=config,
        ),
        RAW_FLUX_VIEW_NAME: evaluate_view_stability(
            RAW_FLUX_VIEW_NAME,
            _change_salience(first_bpm=100.0, second_bpm=200.0),
            frame_rate_hz=100.0,
            config=config,
        ),
        BEATNET_VIEW_NAME: evaluate_view_stability(
            BEATNET_VIEW_NAME,
            _change_salience(first_bpm=100.0, second_bpm=200.0),
            frame_rate_hz=100.0,
            config=config,
        ),
    }

    constant = screen_audio_consensus(views, config=config)
    natural_change = screen_natural_change_consensus(views, config=config)

    assert all(view.stable is False for view in views.values())
    assert all(view.reason == "persistent_family_change" for view in views.values())
    assert all(view.persistent_change is not None for view in views.values())
    assert constant.accepted_constant_source is False
    assert constant.reason == "beatthis_probabilities:persistent_family_change"
    assert natural_change.accepted_natural_change_source is True
    assert natural_change.signed_ratio_octaves == pytest.approx(1.0)
    assert natural_change.family_ratio == pytest.approx(2.0)


def test_natural_change_consensus_accepts_intersecting_boundaries_and_matching_ratio() -> None:
    config = _screen_config()
    beatthis = evaluate_view_stability(
        BEATTHIS_VIEW_NAME,
        _change_salience(first_bpm=120.0, second_bpm=150.0),
        frame_rate_hz=100.0,
        config=config,
    )
    raw = evaluate_view_stability(
        RAW_FLUX_VIEW_NAME,
        _change_salience(first_bpm=60.0, second_bpm=75.0),
        frame_rate_hz=100.0,
        config=config,
    )
    beatnet = evaluate_view_stability(
        BEATNET_VIEW_NAME,
        _change_salience(first_bpm=80.0, second_bpm=100.0),
        frame_rate_hz=100.0,
        config=config,
    )

    consensus = screen_natural_change_consensus(
        {
            BEATTHIS_VIEW_NAME: beatthis,
            RAW_FLUX_VIEW_NAME: raw,
            BEATNET_VIEW_NAME: beatnet,
        },
        config=config,
    )

    assert consensus.accepted_natural_change_source is True
    assert consensus.reason is None
    assert consensus.family_ratio == pytest.approx(1.25, rel=0.05)
    assert consensus.boundary_start_seconds is not None
    assert consensus.boundary_end_seconds is not None
    assert consensus.boundary_start_seconds <= consensus.boundary_end_seconds


def test_plan_only_projects_locator_scope_and_does_not_write_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "labels.jsonl"
    output = tmp_path / "results.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "resolved_audio_path": "/audio/a.mp3",
                "source": {
                    "cache_audio_key": "cache-a",
                    "cache_duration_seconds": 12.0,
                    "cache_status": "valid",
                },
                "label": {"must_not_be_read": True},
                "maps": [{"redline": "not ground truth"}],
                "metadata": {"title": "ignored", "artist": "ignored"},
                "metadata_bpm_evidence": {"bpm": 999},
                "representative_redline_grid": {"bpm": 999},
            },
        ],
    )

    exit_code = main(
        [
            "--labels-path",
            str(manifest),
            "--output-jsonl",
            str(output),
            "--expected-row-count",
            "1",
            "--min-duration-seconds",
            "12",
            "--max-duration-seconds",
            "20",
        ],
    )

    assert exit_code == 0
    assert not output.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == FULL5050_AUDIO_CONSENSUS_PLAN_SCHEMA
    assert payload["action"] == "plan_only"
    assert payload["total_rows"] == 1
    assert payload["notes"]["input_scope"] == "locator_only"
    text = json.dumps(payload, sort_keys=True).lower()
    assert "999" not in text
    assert "title" not in text
    assert "artist" not in text
    assert "sha" not in text


def test_runner_row_records_audio_only_views_and_no_hash_or_mapper_payload() -> None:
    config = _screen_config()
    row = Full5050LocatorRow(
        row_index=7,
        resolved_audio_path=Path("/audio/a.mp3"),
        beatthis_audio_cache_key="cache-a",
        duration_seconds=12.0,
        input_status="valid",
    )

    result = run_full5050_audio_consensus_row(
        row,
        pipeline=_fake_pipeline(config),
        config=config,
    )

    assert result["status"] == "completed"
    assert result["consensus"]["accepted_constant_source"] is True
    assert result["natural_change_consensus"]["accepted_natural_change_source"] is False
    assert set(result["views"]) == {BEATTHIS_VIEW_NAME, RAW_FLUX_VIEW_NAME, BEATNET_VIEW_NAME}
    assert result["beatnet"]["source"] == "BeatNet model=3 mode=offline inference_model=DBN"
    payload_text = json.dumps(result, sort_keys=True).lower()
    assert "sha" not in payload_text
    assert "fingerprint" not in payload_text
    assert "redline" not in payload_text
    assert "label" not in payload_text
    assert "title" not in payload_text
    assert "artist" not in payload_text


def test_full5050_audio_consensus_run_resumes_final_rows(tmp_path: Path) -> None:
    config = _screen_config()
    manifest = tmp_path / "labels.jsonl"
    output = tmp_path / "results.jsonl"
    beatnet_events = tmp_path / "beatnet-events.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row("/audio/a.mp3", "cache-a"),
            _manifest_row("/audio/b.mp3", "cache-b"),
        ],
    )
    _write_beatnet_events(beatnet_events, [0, 1])
    output.write_text(json.dumps(_accepted_existing_result_row(0, "/audio/a.mp3")) + "\n", encoding="utf-8")
    runner_config = Full5050AudioConsensusRunnerConfig(
        labels_path=manifest,
        output_jsonl=output,
        beatnet_events_jsonl=beatnet_events,
        expected_row_count=2,
        screen=config,
    )

    summary = run_full5050_audio_consensus(
        runner_config,
        pipeline=_fake_pipeline(config),
    )

    assert summary["total_rows"] == 2
    assert summary["resumed_rows"] == 1
    assert summary["attempted_rows"] == 1
    assert summary["completed_rows"] == 2
    assert summary["accepted_rows"] == 2
    assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_full5050_audio_consensus_run_requires_precomputed_beatnet_events(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.jsonl"
    output = tmp_path / "results.jsonl"
    _write_manifest(manifest, [_manifest_row("/audio/a.mp3", "cache-a")])
    runner_config = Full5050AudioConsensusRunnerConfig(
        labels_path=manifest,
        output_jsonl=output,
        beatnet_events_jsonl=None,
        expected_row_count=1,
        screen=_screen_config(),
    )

    with pytest.raises(MissingBeatNetEventsError, match="precomputed BeatNet events"):
        run_full5050_audio_consensus(runner_config, pipeline=_fake_pipeline(_screen_config()))


def test_full5050_audio_consensus_run_rejects_missing_beatnet_events_file(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.jsonl"
    output = tmp_path / "results.jsonl"
    missing_events = tmp_path / "missing-events.jsonl"
    _write_manifest(manifest, [_manifest_row("/audio/a.mp3", "cache-a")])
    runner_config = Full5050AudioConsensusRunnerConfig(
        labels_path=manifest,
        output_jsonl=output,
        beatnet_events_jsonl=missing_events,
        expected_row_count=1,
        screen=_screen_config(),
    )

    with pytest.raises(FileNotFoundError, match="BeatNet events JSONL not found"):
        run_full5050_audio_consensus(runner_config, pipeline=_fake_pipeline(_screen_config()))


def test_precomputed_beatnet_event_store_rejects_incomplete_or_empty_completed_rows(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "labels.jsonl"
    missing_events = tmp_path / "missing-events.jsonl"
    empty_events = tmp_path / "empty-events.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row("/audio/a.mp3", "cache-a"),
            _manifest_row("/audio/b.mp3", "cache-b"),
        ],
    )
    rows = (
        Full5050LocatorRow(0, Path("/audio/a.mp3"), "cache-a", 12.0, "valid"),
        Full5050LocatorRow(1, Path("/audio/b.mp3"), "cache-b", 12.0, "valid"),
    )
    _write_beatnet_events(missing_events, [0])
    with pytest.raises(ValueError, match="incomplete"):
        load_precomputed_beatnet_event_store(
            missing_events,
            locator_rows=rows,
            expected_row_count=2,
        )

    empty_events.write_text(
        json.dumps(
            {
                "schema": FULL5050_BEATNET_EVENTS_SCHEMA,
                "row_index": 0,
                "row_id": "full5050:0",
                "resolved_audio_path": "/audio/a.mp3",
                "duration_seconds": 12.0,
                "status": "completed",
                "beat_times_seconds": [],
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(_beatnet_event_row(1), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty beat times"):
        load_precomputed_beatnet_event_store(
            empty_events,
            locator_rows=rows,
            expected_row_count=2,
        )


def test_precomputed_beatnet_explicit_error_becomes_row_failure(tmp_path: Path) -> None:
    config = _screen_config()
    row = Full5050LocatorRow(0, Path("/audio/a.mp3"), "cache-a", 12.0, "valid")
    events = tmp_path / "beatnet-error.jsonl"
    events.write_text(json.dumps(_beatnet_event_error_row(0), sort_keys=True) + "\n", encoding="utf-8")
    store = load_precomputed_beatnet_event_store(events, locator_rows=(row,), expected_row_count=1)
    pipeline = _fake_pipeline(config)
    pipeline.beatnet_extractor = store

    result = run_full5050_audio_consensus_row(row, pipeline=pipeline, config=config)

    assert result["status"] == "failed"
    assert result["reason"] == "beatnet_failed"
    assert result["error"]["type"] == "MissingBeatNetEventsError"


def test_selector_kills_when_fewer_than_required_unique_sources(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "selected.jsonl"
    _write_result_rows(results, count=3, duplicate_everything=False)

    summary = select_constant_sources_from_results(
        results,
        output_jsonl=output,
        train_count=2,
        holdout_count=2,
    )

    assert summary["status"] == "kill"
    assert summary["reason"] == "fewer_than_required_constant_sources"
    assert summary["required_sources"] == 4
    assert summary["selected_sources"] == 3
    assert not output.exists()


def test_selector_writes_top_unique_sources_only_after_threshold(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "selected.jsonl"
    _write_result_rows(results, count=6, duplicate_everything=False)
    with results.open("a", encoding="utf-8") as handle:
        duplicate = _accepted_result_row(99, "/audio/source-0.mp3", confidence=9.0)
        handle.write(json.dumps(duplicate, sort_keys=True) + "\n")

    summary = select_constant_sources_from_results(
        results,
        output_jsonl=output,
        train_count=2,
        holdout_count=2,
    )

    assert summary["status"] == "selected"
    assert summary["selected_sources"] == 4
    selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(selected) == 4
    assert selected[0]["row_index"] == 99
    assert len({row["source_key"] for row in selected}) == 4
    assert [row["split"] for row in selected] == [
        "train",
        "train",
        "untouched_holdout",
        "untouched_holdout",
    ]


def test_natural_change_selector_reports_shortfall_without_writing(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "selected-change.jsonl"
    rows = [_accepted_change_result_row(index, f"/audio/change-{index}.mp3") for index in range(3)]
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = select_natural_change_sources_from_results(
        results,
        output_jsonl=output,
        train_count=2,
        holdout_count=2,
    )

    assert summary["status"] == "report_only"
    assert summary["selected_sources"] == 3
    assert not output.exists()


def test_natural_change_selector_writes_source_disjoint_train_holdout(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "selected-change.jsonl"
    rows = [_accepted_change_result_row(index, f"/audio/change-{index}.mp3") for index in range(6)]
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = select_natural_change_sources_from_results(
        results,
        output_jsonl=output,
        train_count=2,
        holdout_count=2,
    )

    assert summary["status"] == "selected"
    selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(selected) == 4
    assert len({row["source_key"] for row in selected}) == 4
    assert [row["split"] for row in selected] == [
        "train",
        "train",
        "untouched_holdout",
        "untouched_holdout",
    ]


def _screen_config() -> AudioConsensusScreenConfig:
    return AudioConsensusScreenConfig(
        min_duration_seconds=12.0,
        max_duration_seconds=20.0,
        window_seconds=4.0,
        hop_seconds=2.0,
        min_complete_windows=5,
        global_min_confidence=0.03,
        window_min_confidence=0.02,
        min_confident_window_ratio=0.80,
        persistent_run_windows=2,
    )


def _periodic_salience(
    duration_seconds: float,
    *,
    frame_rate_hz: float,
    bpm: float,
) -> np.ndarray:
    frame_count = int(round(duration_seconds * frame_rate_hz))
    signal = np.zeros(frame_count, dtype=np.float32)
    period_seconds = 60.0 / bpm
    times = np.arange(0.25, duration_seconds, period_seconds, dtype=np.float64)
    indices = np.rint(times * frame_rate_hz).astype(np.int64)
    indices = indices[(indices >= 0) & (indices < frame_count)]
    signal[indices] = 1.0
    return signal


def _change_salience(*, first_bpm: float, second_bpm: float) -> np.ndarray:
    return np.concatenate(
        [
            _periodic_salience(6.0, frame_rate_hz=100.0, bpm=first_bpm),
            _periodic_salience(6.0, frame_rate_hz=100.0, bpm=second_bpm),
        ],
    )


def _log_mel_for_salience(salience: np.ndarray) -> np.ndarray:
    mel = np.zeros((salience.shape[0], 80), dtype=np.float32)
    mel[salience > 0.0] = 5.0
    return mel


def _prediction(config: AudioConsensusScreenConfig, *, cache_key: str = "cache-a") -> FrameTimingPrediction:
    del config
    beat_prob = _periodic_salience(12.0, frame_rate_hz=100.0, bpm=120.0)
    return FrameTimingPrediction(
        provider="beat-this",
        checkpoint_path="final0",
        source_path=f"/cache/{cache_key}",
        beat_prob=beat_prob,
        downbeat_prob=np.zeros_like(beat_prob),
        frame_rate_hz=100.0,
    )


def _fake_pipeline(config: AudioConsensusScreenConfig) -> Full5050AudioConsensusPipeline:
    tick = count().__next__
    beatnet_times = np.flatnonzero(
        _periodic_salience(12.0, frame_rate_hz=100.0, bpm=120.0),
    ).astype(np.float64) / 100.0
    raw_mel = _log_mel_for_salience(
        _periodic_salience(12.0, frame_rate_hz=100.0, bpm=120.0),
    )
    packed = pack_full_song_mel_20ms(raw_mel)
    return Full5050AudioConsensusPipeline(
        beatnet_extractor=_FakeBeatNetExtractor(beatnet_times),
        beatthis_cache_config=BeatThisFramePredictionCacheConfig(),
        beatthis_cache_loader=lambda key, _cache_config: _prediction(config, cache_key=key),
        mel_loader=lambda _path, *, audio_cache_key=None: packed,
        clock=tick,
    )


class _FakeBeatNetExtractor:
    def __init__(self, beat_times: np.ndarray) -> None:
        self.beat_times = beat_times

    def extract_beat_times(self, audio_path: str | Path) -> list[float]:
        del audio_path
        return self.beat_times.tolist()


def _manifest_row(audio_path: str, cache_key: str) -> dict[str, object]:
    return {
        "resolved_audio_path": audio_path,
        "source": {
            "cache_audio_key": cache_key,
            "cache_duration_seconds": 12.0,
            "cache_status": "valid",
        },
        "label": {"ignored": True},
        "maps": [{"ignored": True}],
        "representative_redline_grid": {"ignored": True},
    }


def _beatnet_event_row(row_index: int) -> dict[str, object]:
    return {
        "schema": FULL5050_BEATNET_EVENTS_SCHEMA,
        "row_index": row_index,
        "row_id": f"full5050:{row_index}",
        "resolved_audio_path": f"/audio/{'a' if row_index == 0 else 'b'}.mp3",
        "duration_seconds": 12.0,
        "status": "completed",
        "beat_times_seconds": (
            np.flatnonzero(
            _periodic_salience(12.0, frame_rate_hz=100.0, bpm=120.0),
            ).astype(np.float64)
            / 100.0
        ).tolist(),
    }


def _beatnet_event_error_row(row_index: int) -> dict[str, object]:
    return {
        "schema": FULL5050_BEATNET_EVENTS_SCHEMA,
        "row_index": row_index,
        "row_id": f"full5050:{row_index}",
        "resolved_audio_path": f"/audio/{'a' if row_index == 0 else 'b'}.mp3",
        "duration_seconds": 12.0,
        "status": "failed",
        "beat_times_seconds": [],
        "error": {
            "type": "BeatNetRuntimeError",
            "message": "fixture failure",
        },
    }


def _write_beatnet_events(path: Path, row_indexes: list[int]) -> None:
    path.write_text(
        "".join(json.dumps(_beatnet_event_row(index), sort_keys=True) + "\n" for index in row_indexes),
        encoding="utf-8",
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _accepted_result_row(row_index: int, source: str, *, confidence: float) -> dict[str, object]:
    return {
        "schema": "test",
        "row_index": row_index,
        "resolved_audio_path": source,
        "duration_seconds": 120.0,
        "consensus": {
            "accepted_constant_source": True,
            "confidence_score": confidence,
            "confidence_floor": confidence,
            "global_octave_family_bpm": 120.0,
            "max_cross_view_family_distance_octaves": 0.0,
        },
    }


def _accepted_existing_result_row(row_index: int, source: str) -> dict[str, object]:
    return {
        "schema": "test",
        "row_index": row_index,
        "resolved_audio_path": source,
        "duration_seconds": 12.0,
        "status": "completed",
        "consensus": {
            "accepted_constant_source": True,
            "confidence_score": 1.0,
            "confidence_floor": 1.0,
            "global_octave_family_bpm": 120.0,
            "max_cross_view_family_distance_octaves": 0.0,
        },
        "natural_change_consensus": {
            "accepted_natural_change_source": False,
            "confidence_score": 0.0,
        },
    }


def _accepted_change_result_row(row_index: int, source: str) -> dict[str, object]:
    confidence = float(row_index + 1)
    return {
        "schema": "test",
        "row_index": row_index,
        "resolved_audio_path": source,
        "duration_seconds": 120.0,
        "natural_change_consensus": {
            "accepted_natural_change_source": True,
            "boundary_start_seconds": 48.0,
            "boundary_end_seconds": 54.0,
            "signed_ratio_octaves": 0.321928,
            "family_ratio": 1.25,
            "confidence_score": confidence,
            "confidence_floor": confidence,
            "max_cross_view_ratio_distance_octaves": 0.0,
        },
    }


def _write_result_rows(path: Path, *, count: int, duplicate_everything: bool) -> None:
    rows = []
    for index in range(count):
        source_index = 0 if duplicate_everything else index
        rows.append(
            _accepted_result_row(
                index,
                f"/audio/source-{source_index}.mp3",
                confidence=float(index + 1),
            ),
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
