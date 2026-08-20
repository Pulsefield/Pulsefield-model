from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from pulsefield_model.features.mel import stage2_log_mel_cache_path
from pulsefield_model.timing.evaluation.real_audio_pilot import (
    _coverage_start_ms,
    main,
    projection_cj_candidates,
    run_real_audio_pilot,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    PhaseContinuousTimingCurve,
)


def _sha(character: str) -> str:
    return character * 64


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _curve(
    *,
    fingerprint: str,
    origin_time_ms: float = 0.0,
    source: str = "hook",
) -> Any:
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=origin_time_ms,
        sections=(ConstantTempoSection(0, 24, 120.0),),
    )

    class Candidate:
        start_beat = curve.start_beat
        end_beat = curve.end_beat
        fingerprint_sha256 = fingerprint

        def time_at_beat(self, beat: float) -> float:
            return curve.time_at_beat(beat)

    candidate = Candidate()
    candidate.grid = curve
    candidate.source = source
    return candidate


def _fixture_files(tmp_path: Path, *, stratum: str = "stable") -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    audio_path = repo_root / "dataset" / "0" / "1" / "audio.ogg"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"not-decoded-because-mel-exists")
    cache_key = "cache-key"
    weak_oracle_path = repo_root / "dataset" / "0" / "1" / "map.osu"
    weak_oracle_path.write_text("evaluation fixture", encoding="utf-8")
    pilot = tmp_path / "pilot.jsonl"
    projection = tmp_path / "projection.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    output = tmp_path / "output.jsonl"
    summary = tmp_path / "summary.json"
    label_stratum = {
        "stable": "stable",
        "jump": "jump_candidate",
        "ramp": "ramp_candidate",
    }[stratum]
    evidence_class = {
        "stable": "stable",
        "jump": "jump_candidate",
        "ramp": "ramp_candidate",
    }[stratum]
    _write_jsonl(
        pilot,
        [
            {
                "audio_group_index": 1,
                "audio_group_key": audio_path.as_posix(),
                "resolved_audio_path": audio_path.as_posix(),
                "pilot_stratum": stratum,
                "label": {"stratum": label_stratum, "confidence": "medium", "ambiguous": False},
                "source": {"cache_audio_key": cache_key, "cache_duration_seconds": 12.0},
                "maps": [{"beatmap_path": weak_oracle_path.as_posix()}],
                "representative_redline_grid": {
                    "beatmap_path": weak_oracle_path.as_posix(),
                    "evidence_class": evidence_class,
                    "agreement_rate": 1.0,
                },
            }
        ],
    )
    accepted_grid = {
        "schema": "pulsefield_model.timing_v3_grid_v1",
        "version": 1,
        "origin_beat": 0,
        "origin_time_ms": 0.0,
        "coverage_start_ms": 0.0,
        "coverage_end_ms": 12000.0,
        "sections": [{"start_beat": 0, "end_beat": 24, "bpm": 120.0}],
    }
    _write_jsonl(
        projection,
        [
            {
                "identity": {"cache_audio_key": cache_key},
                "cache": {"coverage_start_ms": 0.0, "coverage_end_ms": 12000.0},
                "variants": {
                    "CJ0": {
                        "status": "accepted",
                        "grid": accepted_grid,
                        "grid_fingerprint": _sha("a"),
                    },
                    "CJ1": {
                        "status": "accepted",
                        "grid": accepted_grid,
                        "grid_fingerprint": _sha("a"),
                    },
                    "CJ2": {"status": "not_run"},
                    "CJ3": {"status": "not_run"},
                },
            }
        ],
    )
    _write_jsonl(
        baseline,
        [
            {
                "audio_key": cache_key,
                "audio_path": audio_path.as_posix(),
                "beatmap_paths": [weak_oracle_path.as_posix()],
                "comparisons": [{"oracle": "must-not-reach-inference"}],
                "fit": {
                    "predicted_segments": [
                        {"offset_ms": 0.0, "beat_length_ms": 500.0, "meter": 4}
                    ]
                },
            }
        ],
    )
    mel_path = stage2_log_mel_cache_path(
        audio_path,
        audio_cache_key=audio_path.relative_to(repo_root).as_posix(),
    )
    mel_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_path, np.zeros((1200, 80), dtype=np.float32))
    return {
        "repo_root": repo_root,
        "pilot": pilot,
        "projection": projection,
        "baseline": baseline,
        "output": output,
        "summary": summary,
        "weak_oracle": weak_oracle_path,
    }


def test_projection_candidates_deduplicate_accepted_grid_fingerprints(tmp_path: Path) -> None:
    paths = _fixture_files(tmp_path)
    pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
    projection = json.loads(paths["projection"].read_text(encoding="utf-8"))

    candidates = projection_cj_candidates(pilot, projection, None)

    assert len(candidates) == 1
    assert candidates[0].source == "CJ0"
    assert candidates[0].fingerprint_sha256 == _sha("a")


def test_projection_candidates_expose_realistic_common_integer_beat_domain() -> None:
    def grid(start: int, end: int, bpm: float) -> dict[str, Any]:
        first_time = 1000.0 + start * 60_000.0 / bpm
        end_time = 1000.0 + end * 60_000.0 / bpm
        return {
            "schema": "pulsefield_model.timing_v3_grid_v1",
            "version": 1,
            "origin_beat": 0,
            "origin_time_ms": 1000.0,
            "coverage_start_ms": first_time,
            "coverage_end_ms": end_time,
            "sections": [{"start_beat": start, "end_beat": end, "bpm": bpm}],
        }

    projection = {
        "variants": {
            "CJ0": {
                "status": "accepted",
                "grid": grid(-3, 40, 120.0),
                "grid_fingerprint": _sha("a"),
            },
            "CJ1": {
                "status": "accepted",
                "grid": grid(-1, 30, 240.0),
                "grid_fingerprint": _sha("b"),
            },
            "CJ2": {
                "status": "accepted",
                "grid": grid(-2, 38, 180.0),
                "grid_fingerprint": _sha("c"),
            },
        }
    }

    candidates = projection_cj_candidates({}, projection, None)

    assert len(candidates) == 3
    assert {(candidate.start_beat, candidate.end_beat) for candidate in candidates} == {
        (-1, 30)
    }
    assert candidates[0].time_at_beat(0) == pytest.approx(1000.0)


def test_projection_candidates_reject_common_domain_shorter_than_scorer_window() -> None:
    def grid(start: int, end: int, origin_beat: int) -> dict[str, Any]:
        origin_time = origin_beat * 500.0
        return {
            "schema": "pulsefield_model.timing_v3_grid_v1",
            "version": 1,
            "origin_beat": origin_beat,
            "origin_time_ms": origin_time,
            "coverage_start_ms": start * 500.0,
            "coverage_end_ms": end * 500.0,
            "sections": [{"start_beat": start, "end_beat": end, "bpm": 120.0}],
        }

    projection = {
        "variants": {
            "CJ0": {
                "status": "accepted",
                "grid": grid(-1, 16, 0),
                "grid_fingerprint": _sha("a"),
            },
            "CJ1": {
                "status": "accepted",
                "grid": grid(1, 17, 1),
                "grid_fingerprint": _sha("b"),
            },
        }
    }

    with pytest.raises(ValueError, match="fewer than 16 integer beats"):
        projection_cj_candidates({}, projection, None)


def test_evaluation_coverage_respects_grid_mathematical_domain() -> None:
    grid = {
        "schema": "pulsefield_model.timing_v3_grid_v1",
        "version": 1,
        "origin_beat": 0,
        "origin_time_ms": 962.0,
        "coverage_start_ms": 962.0,
        "coverage_end_ms": 5000.0,
        "sections": [{"start_beat": 1, "end_beat": 20, "bpm": 120.0}],
    }
    # Make origin valid while retaining a positive first derived boundary.
    grid["origin_beat"] = 1
    candidate = {
        "variants": {
            "CJ0": {
                "status": "accepted",
                "grid": grid,
                "grid_fingerprint": _sha("d"),
            }
        }
    }
    (wrapped,) = projection_cj_candidates({}, candidate, None)

    assert wrapped.grid.start_time_ms == pytest.approx(962.0)
    assert _coverage_start_ms(wrapped.grid, {}) == pytest.approx(962.0)


def test_projection_candidates_use_common_scoring_domain_across_aliases(
    tmp_path: Path,
) -> None:
    paths = _fixture_files(tmp_path)
    pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
    projection = json.loads(paths["projection"].read_text(encoding="utf-8"))
    faster = dict(projection["variants"]["CJ0"]["grid"])
    faster["sections"] = [{"start_beat": -2, "end_beat": 48, "bpm": 240.0}]
    projection["variants"]["CJ1"] = {
        "status": "accepted",
        "grid": faster,
        "grid_fingerprint": _sha("d"),
    }

    candidates = projection_cj_candidates(pilot, projection, None)

    assert len(candidates) == 2
    assert {(candidate.start_beat, candidate.end_beat) for candidate in candidates} == {
        (0, 24)
    }
    # The wrappers retain the underlying curve's exact time mapping.
    assert candidates[1].time_at_beat(1) == pytest.approx(250.0)


def test_runner_freezes_selection_before_loading_weak_oracle_and_sanitizes_hook_inputs(
    tmp_path: Path,
) -> None:
    paths = _fixture_files(tmp_path)
    events: list[tuple[str, str | None]] = []

    def generator(
        pilot_row: Mapping[str, Any],
        projection_row: Mapping[str, Any],
        baseline_row: Mapping[str, Any] | None,
    ) -> list[Any]:
        assert "representative_redline_grid" not in pilot_row
        assert "maps" not in pilot_row
        assert baseline_row is not None
        assert "beatmap_paths" not in baseline_row
        assert "comparisons" not in baseline_row
        assert projection_row["identity"]["cache_audio_key"] == "cache-key"
        events.append(("generate", None))
        return [_curve(fingerprint=_sha("b"))]

    def weak_oracle_loader(path: Path) -> FittedTimingGrid:
        # The row file is written after evaluation, so the proof carried across
        # this boundary is the immutable fingerprint supplied to the payload.
        assert not paths["output"].exists()
        events.append(("weak_oracle", path.as_posix()))
        return FittedTimingGrid((TimingSegment(0.0, 500.0),))

    summary = run_real_audio_pilot(
        pilot_jsonl_path=paths["pilot"],
        projection_jsonl_path=paths["projection"],
        baseline_jsonl_path=paths["baseline"],
        output_jsonl_path=paths["output"],
        summary_json_path=paths["summary"],
        repo_root=paths["repo_root"],
        candidate_generator=generator,
        weak_oracle_loader=weak_oracle_loader,
    )

    assert [event[0] for event in events] == ["generate", "weak_oracle"]
    row = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert row["inference"]["selected_fingerprint_sha256"] == _sha("b")
    assert row["inference"]["frozen_before_weak_oracle_load"] is True
    assert row["inference"]["mel_source"] == "existing_10ms_mel_cache"
    frozen = row["inference"]["frozen_payload_sha256"]
    assert row["weak_oracle_evaluation"]["frozen_inference_sha256"] == frozen
    assert row["weak_oracle_evaluation"]["selected_metrics"]["weak_oracle_constant_exact_hit"] is True
    assert row["weak_oracle_evaluation"]["weak_oracle_ramp_accuracy"] is None
    assert summary["weak_oracle_exact_accuracy"]["constant"]["rate"] == pytest.approx(1.0)
    assert summary["weak_oracle_exact_accuracy"]["ramp"] is None


def test_ramp_like_row_always_reports_null_accuracy(tmp_path: Path) -> None:
    paths = _fixture_files(tmp_path, stratum="ramp")

    summary = run_real_audio_pilot(
        pilot_jsonl_path=paths["pilot"],
        projection_jsonl_path=paths["projection"],
        baseline_jsonl_path=paths["baseline"],
        output_jsonl_path=paths["output"],
        summary_json_path=paths["summary"],
        repo_root=paths["repo_root"],
        candidate_generator=lambda *_: [_curve(fingerprint=_sha("c"))],
        weak_oracle_loader=lambda _: FittedTimingGrid((TimingSegment(0.0, 500.0),)),
    )

    row = json.loads(paths["output"].read_text(encoding="utf-8"))
    evaluation = row["weak_oracle_evaluation"]
    assert evaluation["weak_oracle_class"] == "ramp_like"
    assert evaluation["weak_oracle_ramp_accuracy"] is None
    assert evaluation["selected_metrics"]["weak_oracle_ramp_accuracy"] is None
    assert summary["weak_oracle_exact_accuracy"]["ramp"] is None


def test_cli_runs_default_projection_candidates(tmp_path: Path) -> None:
    paths = _fixture_files(tmp_path)

    exit_code = main(
        [
            "--pilot-jsonl",
            str(paths["pilot"]),
            "--projection-jsonl",
            str(paths["projection"]),
            "--baseline-jsonl",
            str(paths["baseline"]),
            "--output-jsonl",
            str(paths["output"]),
            "--summary-json",
            str(paths["summary"]),
            "--repo-root",
            str(paths["repo_root"]),
        ]
    )

    assert exit_code == 0
    assert paths["output"].exists()
    assert paths["summary"].exists()
    row = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert row["inference"]["candidate_sources"] == ["CJ0"]
    # Fixture `.osu` is deliberately invalid; evaluation failure is isolated
    # and does not retroactively invalidate the frozen inference result.
    assert row["ok"] is True
    assert row["weak_oracle_evaluation"]["available"] is False
    assert row["weak_oracle_evaluation"]["unavailable_reason"] == "weak_oracle_evaluation_error"
