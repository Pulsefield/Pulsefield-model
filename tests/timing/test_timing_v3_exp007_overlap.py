from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from pulsefield_model.timing.v3 import local_frontier as lf
from pulsefield_model.timing.v3.local_frontier import (
    MAX_OVERLAP_RECORDS_PER_AUDIO,
    MAX_OVERLAP_RESIDUAL_PAIRS,
    MAX_OVERLAP_TRACE_REAL_BOUNDARIES,
    MAX_OVERLAP_TRACE_BEATS,
    REASON_DIAGNOSTICS_INTEGRITY_FAILURE,
    LocalFrontierConfig,
    LocalFrontierObjectiveVariant,
    LocalFrontierScheduleArm,
    fit_local_frontier_boundary_pair_transition,
    fit_local_frontier_boundary_pair_transition_bounded,
)


_MATRIX_HELPER_PATH = Path(__file__).with_name(
    "test_timing_v3_boundary_pair_transition_matrix.py"
)
_MATRIX_SPEC = importlib.util.spec_from_file_location(
    "_exp007_matrix_fixture_helpers",
    _MATRIX_HELPER_PATH,
)
assert _MATRIX_SPEC is not None and _MATRIX_SPEC.loader is not None
_MATRIX = importlib.util.module_from_spec(_MATRIX_SPEC)
sys.modules[_MATRIX_SPEC.name] = _MATRIX
_MATRIX_SPEC.loader.exec_module(_MATRIX)


def _context_for_case(case_id: str = "clean_constant_120") -> lf._LocalScoreContext:  # noqa: SLF001
    case = _MATRIX._case_by_id(case_id)  # noqa: SLF001
    fixture = _MATRIX._fixture_for_case(case)  # noqa: SLF001
    restricted = lf.restrict_timing_prediction(fixture.prediction)
    return lf._LocalScoreContext(  # noqa: SLF001
        restricted,
        fixture.candidate_set,
        LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4,
        lf._LocalFrontierDiagnosticsMode.BOUNDED,  # noqa: SLF001
    )


def _path(
    *,
    closed_sections: tuple[lf._ClosedSection, ...] = (),  # noqa: SLF001
    selected_boundaries: tuple[lf._SelectedBoundary, ...] = (),  # noqa: SLF001
    open_start_beat: int = 0,
    open_start_time_ms: float = 0.0,
    current_bpm: float = 120.0,
    replay_key: tuple[Any, ...] = ("path",),
) -> lf._Path:  # noqa: SLF001
    return lf._Path(  # noqa: SLF001
        origin_time_ms=0.0,
        serialized_first_start_beat=0,
        open_start_beat=open_start_beat,
        open_start_time_ms=open_start_time_ms,
        current_bpm=current_bpm,
        previous_bpm=None,
        global_downbeat_phase=None,
        closed_sections=closed_sections,
        duration_objective_numerator=0.0,
        transition_objective=0.0,
        real_section_count=len(closed_sections) + 1,
        alias_switch_count=0,
        max_boundary_displacement_ms=0.0,
        replay_key=replay_key,
        selected_boundaries=selected_boundaries,
        last_boundary_anchor_id=(
            -1 if not selected_boundaries else selected_boundaries[-1].anchor_id
        ),
    )


def _lineage(
    *,
    provisional_trace: lf._OverlapTrace,  # noqa: SLF001
    previous_core_end_ms: float = 0.0,
    previous_lookahead_end_ms: float = 10_000.0,
    prior_block_index: int = 0,
    prior_export_ordinal: int = 0,
) -> lf._OverlapLineage:  # noqa: SLF001
    token = (
        lf.LOOKAHEAD_OVERLAP_RECORD_CONTRACT_VERSION,
        prior_block_index,
        prior_export_ordinal,
        "f" * 64,
        "e" * 64,
    )
    return lf._OverlapLineage(  # noqa: SLF001
        record_contract_version=lf.LOOKAHEAD_OVERLAP_RECORD_CONTRACT_VERSION,
        prior_block_index=prior_block_index,
        prior_export_ordinal=prior_export_ordinal,
        future_equivalence_sha256="f" * 64,
        committed_replay_sha256="e" * 64,
        lineage_sha256=lf._json_fingerprint(token),  # noqa: SLF001
        previous_core_end_ms=previous_core_end_ms,
        previous_lookahead_end_ms=previous_lookahead_end_ms,
        provisional_trace=provisional_trace,
    )


def _stable_sha(payload: Any) -> str:
    return lf._json_fingerprint(payload)  # noqa: SLF001


@pytest.mark.parametrize(
    ("case", "schedule_arm"),
    tuple(
        (case, schedule_arm)
        for case in _MATRIX.MATRIX_CASES
        for schedule_arm in _MATRIX.SCHEDULE_ARMS
    ),
)
def test_exp007_bounded_matches_full_search_grid_and_base_diagnostics_for_matrix(
    case: Any,
    schedule_arm: LocalFrontierScheduleArm,
) -> None:
    fixture = _MATRIX._fixture_for_case(case)  # noqa: SLF001
    config = LocalFrontierConfig(schedule_arm=schedule_arm)

    full = fit_local_frontier_boundary_pair_transition(
        fixture.prediction,
        config=config,
        candidate_set=fixture.candidate_set,
    )
    bounded = fit_local_frontier_boundary_pair_transition_bounded(
        fixture.prediction,
        config=config,
        candidate_set=fixture.candidate_set,
    )

    assert bounded.fit_result.reason == full.reason
    assert (None if bounded.grid is None else bounded.grid.to_dict()) == (
        None if full.grid is None else full.grid.to_dict()
    )
    assert bounded.fit_result.diagnostics == full.diagnostics
    assert bounded.fit_result.objective_diagnostics is None

    full_objective = full.objective_diagnostics
    assert full_objective is not None
    diagnostics = bounded.diagnostics
    assert diagnostics.objective_variant == full_objective.objective_variant
    assert diagnostics.candidate_fingerprint == full_objective.candidate_fingerprint
    assert diagnostics.transition_cache_size == full_objective.transition_cache_size
    assert diagnostics.actual_scored_edge_count == len(full_objective.actual_scored_edges)
    assert diagnostics.selected_terminal_objective == (
        full_objective.selected_terminal_objective
    )
    assert diagnostics.runner_up_terminal_objective == (
        full_objective.runner_up_terminal_objective
    )
    assert diagnostics.selected_runner_up_margin == (
        full_objective.selected_runner_up_margin
    )
    assert diagnostics.block_resource_records == full_objective.block_resource_records
    assert diagnostics.class_coverage_records == full_objective.class_coverage_records


def test_exp007_constant_overlap_has_zero_residuals_and_no_final_cut_record() -> None:
    case = _MATRIX._case_by_id("clean_constant_120")  # noqa: SLF001
    fixture = _MATRIX._fixture_for_case(case)  # noqa: SLF001
    config = LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30)

    bounded = fit_local_frontier_boundary_pair_transition_bounded(
        fixture.prediction,
        config=config,
        candidate_set=fixture.candidate_set,
    )

    overlap = bounded.diagnostics.overlap
    assert overlap.record_count == sum(
        bounded.fit_result.diagnostics.frontier_widths[:-1]
    )
    assert overlap.record_count > 0
    assert overlap.available_record_count == overlap.record_count
    assert overlap.unavailable_record_count == 0
    assert overlap.comparable_beat_count > 0
    assert overlap.p90_ms == 0.0
    assert overlap.p90_beats == 0.0
    assert overlap.residual_vector_sha256 is not None
    assert all(
        record.next_block_index < bounded.fit_result.diagnostics.block_count
        for record in overlap.records
    )
    json.dumps(
        asdict(bounded.diagnostics),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert bounded.diagnostics.overlap.records
    assert bounded.diagnostics.overlap.records_sha256 == _stable_sha(
        tuple(asdict(record) for record in bounded.diagnostics.overlap.records)
    )


def test_exp007_bounded_and_full_canonical_reason_grid_base_match_for_matrix() -> None:
    payloads = []
    for case in _MATRIX.MATRIX_CASES:
        fixture = _MATRIX._fixture_for_case(case)  # noqa: SLF001
        for schedule_arm in _MATRIX.SCHEDULE_ARMS:
            config = LocalFrontierConfig(schedule_arm=schedule_arm)
            full = fit_local_frontier_boundary_pair_transition(
                fixture.prediction,
                config=config,
                candidate_set=fixture.candidate_set,
            )
            bounded = fit_local_frontier_boundary_pair_transition_bounded(
                fixture.prediction,
                config=config,
                candidate_set=fixture.candidate_set,
            )
            full_payload = {
                "reason": full.reason,
                "grid": None if full.grid is None else full.grid.to_dict(),
                "base": asdict(full.diagnostics),
            }
            bounded_payload = {
                "reason": bounded.fit_result.reason,
                "grid": None if bounded.grid is None else bounded.grid.to_dict(),
                "base": asdict(bounded.fit_result.diagnostics),
            }
            assert json.dumps(
                bounded_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ) == json.dumps(
                full_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            payloads.append(
                {
                    "case": case.case_id,
                    "arm": schedule_arm.value,
                    "sha": _stable_sha(bounded_payload),
                }
            )
    assert len(payloads) == _MATRIX.EXPECTED_MATRIX_ARM_COUNT


def test_exp007_trace_payload_is_exact_beats_and_boundaries_with_right_bpm() -> None:
    context = _context_for_case()
    boundary = lf._SelectedBoundary(  # noqa: SLF001
        anchor_id=3,
        source_time_ms=1000.0,
        boundary_beat=2,
        boundary_time_ms=1000.0,
    )
    path = _path(
        closed_sections=(
            lf._ClosedSection(start_beat=0, end_beat=2, bpm=120.0),  # noqa: SLF001
        ),
        selected_boundaries=(boundary,),
        open_start_beat=2,
        open_start_time_ms=1000.0,
        current_bpm=150.0,
    )

    trace = lf._overlap_trace_for_path(  # noqa: SLF001
        path,
        context=context,
        start_ms=0.0,
        end_ms=2200.0,
    )

    assert trace.beats == (
        (0, float.hex(0.0), float.hex(120.0)),
        (1, float.hex(500.0), float.hex(120.0)),
        (2, float.hex(1000.0), float.hex(150.0)),
        (3, float.hex(1400.0), float.hex(150.0)),
        (4, float.hex(1800.0), float.hex(150.0)),
    )
    assert trace.boundaries == (
        (
            3,
            float.hex(1000.0),
            2,
            float.hex(1000.0),
            float.hex(120.0),
            float.hex(150.0),
        ),
    )
    assert trace.sha256 == _stable_sha(
        {
            "beats": trace.beats,
            "boundaries": trace.boundaries,
        }
    )


def test_exp007_comparison_domain_hash_binds_exact_k_intersection() -> None:
    context = _context_for_case()
    provisional_path = _path(
        open_start_beat=0,
        open_start_time_ms=0.0,
        current_bpm=120.0,
        replay_key=("provisional",),
    )
    descendant_path = _path(
        open_start_beat=4,
        open_start_time_ms=2000.0,
        current_bpm=120.0,
        replay_key=("descendant",),
    )
    provisional_trace = lf._overlap_trace_for_path(  # noqa: SLF001
        provisional_path,
        context=context,
        start_ms=0.0,
        end_ms=10_000.0,
    )

    record, residuals = lf._lookahead_overlap_record(  # noqa: SLF001
        lineage=_lineage(provisional_trace=provisional_trace),
        descendant=descendant_path,
        context=context,
        next_block_index=1,
        next_core_end_ms=10_000.0,
    )

    common_beats = tuple(beat for beat, _, _ in residuals)
    assert common_beats == tuple(range(4, 20))
    assert record.comparison_domain_sha256 == _stable_sha(
        [float.hex(0.0), float.hex(10_000.0), list(common_beats)]
    )
    assert lf._comparison_domain_sha256(  # noqa: SLF001
        comparison_start_ms=0.0,
        comparison_end_ms=10_000.0,
        common_beats=tuple(range(20)),
    ) != record.comparison_domain_sha256


def test_exp007_residuals_use_smaller_period_divisor_and_linear_p90() -> None:
    provisional = lf._OverlapTrace(  # noqa: SLF001
        beats=tuple((beat, float.hex(float(beat * 500)), float.hex(120.0)) for beat in range(8)),
        boundaries=(),
        sha256="a" * 64,
    )
    recomputed = lf._OverlapTrace(  # noqa: SLF001
        beats=tuple(
            (beat, float.hex(float(beat * 400 + beat)), float.hex(150.0))
            for beat in range(8)
        ),
        boundaries=(),
        sha256="b" * 64,
    )

    residuals = lf._overlap_residuals(  # noqa: SLF001
        provisional=provisional,
        recomputed=recomputed,
        comparison_start_ms=0.0,
        comparison_end_ms=10_000.0,
    )

    assert residuals[1][1] == pytest.approx(99.0)
    assert residuals[1][2] == pytest.approx(99.0 / 400.0)
    assert lf._p90_linear([float(index) for index in range(10)]) == pytest.approx(8.1)  # noqa: SLF001


def test_exp007_available_requires_8_comparable_pairs() -> None:
    context = _context_for_case()
    short_path = _path(current_bpm=120.0, replay_key=("short",))
    short_trace = lf._overlap_trace_for_path(  # noqa: SLF001
        short_path,
        context=context,
        start_ms=0.0,
        end_ms=3500.0,
    )

    short_record, short_residuals = lf._lookahead_overlap_record(  # noqa: SLF001
        lineage=_lineage(
            provisional_trace=short_trace,
            previous_lookahead_end_ms=3500.0,
        ),
        descendant=short_path,
        context=context,
        next_block_index=1,
        next_core_end_ms=3500.0,
    )
    assert len(short_residuals) == 0
    assert short_record.comparable_beat_count == 7
    assert short_record.unavailable_reason == lf.UNAVAILABLE_FEWER_THAN_8_COMPARABLE_BEATS
    assert short_record.p90_ms is None

    enough_trace = lf._overlap_trace_for_path(  # noqa: SLF001
        short_path,
        context=context,
        start_ms=0.0,
        end_ms=4000.0,
    )
    enough_record, enough_residuals = lf._lookahead_overlap_record(  # noqa: SLF001
        lineage=_lineage(
            provisional_trace=enough_trace,
            previous_lookahead_end_ms=4000.0,
        ),
        descendant=short_path,
        context=context,
        next_block_index=1,
        next_core_end_ms=4000.0,
    )
    assert len(enough_residuals) == 8
    assert enough_record.comparable_beat_count == 8
    assert enough_record.unavailable_reason is None
    assert enough_record.p90_ms == 0.0


def test_exp007_unavailable_reasons_and_empty_domain_hashes() -> None:
    context = _context_for_case()
    path = _path()
    trace = lf._overlap_trace_for_path(  # noqa: SLF001
        path,
        context=context,
        start_ms=0.0,
        end_ms=10_000.0,
    )

    empty_record, _ = lf._lookahead_overlap_record(  # noqa: SLF001
        lineage=_lineage(
            provisional_trace=trace,
            previous_core_end_ms=10_000.0,
            previous_lookahead_end_ms=10_000.0,
        ),
        descendant=path,
        context=context,
        next_block_index=1,
        next_core_end_ms=10_000.0,
    )
    assert empty_record.unavailable_reason == lf.UNAVAILABLE_EMPTY_COMMON_TIME_DOMAIN
    assert empty_record.comparison_domain_sha256 == _stable_sha(
        [float.hex(10_000.0), float.hex(10_000.0), []]
    )

    pruned_record, _ = lf._lookahead_overlap_record(  # noqa: SLF001
        lineage=_lineage(provisional_trace=trace),
        descendant=None,
        context=context,
        next_block_index=1,
        next_core_end_ms=10_000.0,
    )
    assert pruned_record.unavailable_reason == lf.UNAVAILABLE_LINEAGE_NOT_RETAINED_AT_NEXT_CUT
    assert pruned_record.recomputed_trace_sha256 is None
    assert pruned_record.residual_vector_sha256 is None


def test_exp007_public_diagnostics_failure_classification_is_stable() -> None:
    exc = lf._DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)  # noqa: SLF001
    classification = lf.classify_local_frontier_exception(exc)

    assert classification is not None
    assert classification.reason == REASON_DIAGNOSTICS_INTEGRITY_FAILURE
    assert classification.stage == "diagnostics"
    assert str(exc) == REASON_DIAGNOSTICS_INTEGRITY_FAILURE


def test_exp007_caps_are_limit_inclusive_and_limit_plus_one_closed() -> None:
    lf._ensure_overlap_record_capacity(MAX_OVERLAP_RECORDS_PER_AUDIO - 1, 1)  # noqa: SLF001
    with pytest.raises(lf._DiagnosticsIntegrityFailure):  # noqa: SLF001
        lf._ensure_overlap_record_capacity(MAX_OVERLAP_RECORDS_PER_AUDIO, 1)  # noqa: SLF001

    lf._ensure_overlap_residual_capacity(MAX_OVERLAP_RESIDUAL_PAIRS - 1, 1)  # noqa: SLF001
    with pytest.raises(lf._DiagnosticsIntegrityFailure):  # noqa: SLF001
        lf._ensure_overlap_residual_capacity(MAX_OVERLAP_RESIDUAL_PAIRS, 1)  # noqa: SLF001

    context = _context_for_case()
    max_beats_path = _path(current_bpm=1000.0)
    max_trace = lf._overlap_trace_for_path(  # noqa: SLF001
        max_beats_path,
        context=context,
        start_ms=0.0,
        end_ms=MAX_OVERLAP_TRACE_BEATS * 60.0,
    )
    assert len(max_trace.beats) == MAX_OVERLAP_TRACE_BEATS
    with pytest.raises(lf._DiagnosticsIntegrityFailure):  # noqa: SLF001
        lf._overlap_trace_for_path(  # noqa: SLF001
            max_beats_path,
            context=context,
            start_ms=0.0,
            end_ms=(MAX_OVERLAP_TRACE_BEATS + 1) * 60.0,
        )

    boundary_path = _path(
        closed_sections=tuple(
            lf._ClosedSection(  # noqa: SLF001
                start_beat=index,
                end_beat=index + 1,
                bpm=120.0,
            )
            for index in range(MAX_OVERLAP_TRACE_REAL_BOUNDARIES)
        ),
        selected_boundaries=tuple(
            lf._SelectedBoundary(  # noqa: SLF001
                anchor_id=index,
                source_time_ms=float((index + 1) * 500),
                boundary_beat=index + 1,
                boundary_time_ms=float((index + 1) * 500),
            )
            for index in range(MAX_OVERLAP_TRACE_REAL_BOUNDARIES)
        ),
        open_start_beat=MAX_OVERLAP_TRACE_REAL_BOUNDARIES,
        open_start_time_ms=float(MAX_OVERLAP_TRACE_REAL_BOUNDARIES * 500),
        current_bpm=120.0,
    )
    boundary_trace = lf._overlap_trace_for_path(  # noqa: SLF001
        boundary_path,
        context=context,
        start_ms=0.0,
        end_ms=float((MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 2) * 500),
    )
    assert len(boundary_trace.boundaries) == MAX_OVERLAP_TRACE_REAL_BOUNDARIES

    too_many_boundaries = replace(
        boundary_path,
        closed_sections=boundary_path.closed_sections
        + (
            lf._ClosedSection(  # noqa: SLF001
                start_beat=MAX_OVERLAP_TRACE_REAL_BOUNDARIES,
                end_beat=MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 1,
                bpm=120.0,
            ),
        ),
        selected_boundaries=boundary_path.selected_boundaries
        + (
            lf._SelectedBoundary(  # noqa: SLF001
                anchor_id=MAX_OVERLAP_TRACE_REAL_BOUNDARIES,
                source_time_ms=float((MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 1) * 500),
                boundary_beat=MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 1,
                boundary_time_ms=float((MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 1) * 500),
            ),
        ),
        open_start_beat=MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 1,
        open_start_time_ms=float((MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 1) * 500),
        last_boundary_anchor_id=MAX_OVERLAP_TRACE_REAL_BOUNDARIES,
        real_section_count=MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 2,
    )
    with pytest.raises(lf._DiagnosticsIntegrityFailure):  # noqa: SLF001
        lf._overlap_trace_for_path(  # noqa: SLF001
            too_many_boundaries,
            context=context,
            start_ms=0.0,
            end_ms=float((MAX_OVERLAP_TRACE_REAL_BOUNDARIES + 3) * 500),
        )


def test_exp007_record_cap_preflight_skips_record_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context_for_case()
    path = _path()
    trace = lf._overlap_trace_for_path(  # noqa: SLF001
        path,
        context=context,
        start_ms=0.0,
        end_ms=4000.0,
    )
    lineage = _lineage(
        provisional_trace=trace,
        previous_lookahead_end_ms=4000.0,
    )
    context.pending_overlap_lineages = (lineage,)
    context.overlap_records = [
        lf.LookaheadOverlapRecord(
            previous_block_index=0,
            next_block_index=1,
            previous_export_ordinal=0,
            lineage_sha256=lineage.lineage_sha256,
            comparison_start_ms=0.0,
            comparison_end_ms=4000.0,
            provisional_trace_sha256=trace.sha256,
            recomputed_trace_sha256=trace.sha256,
            comparison_domain_sha256="d" * 64,
            comparable_beat_count=8,
            residual_vector_sha256="r" * 64,
            p90_ms=0.0,
            p90_beats=0.0,
            unavailable_reason=None,
        )
    ] * MAX_OVERLAP_RECORDS_PER_AUDIO
    record_builder_called = False

    def _raise_record_builder(*args: Any, **kwargs: Any) -> Any:
        nonlocal record_builder_called
        record_builder_called = True
        raise AssertionError("record builder should not run after record cap")

    monkeypatch.setattr(lf, "_lookahead_overlap_record", _raise_record_builder)

    with pytest.raises(
        lf._DiagnosticsIntegrityFailure,  # noqa: SLF001
        match=REASON_DIAGNOSTICS_INTEGRITY_FAILURE,
    ):
        lf._record_pending_overlap_at_export(  # noqa: SLF001
            context=context,
            exported_paths=(),
            next_block_index=1,
            next_core_end_ms=4000.0,
        )
    assert not record_builder_called


def test_exp007_residual_cap_prevents_plus_one_aggregate_append() -> None:
    context = _context_for_case()
    path = _path()
    trace = lf._overlap_trace_for_path(  # noqa: SLF001
        path,
        context=context,
        start_ms=0.0,
        end_ms=4000.0,
    )
    lineage = _lineage(
        provisional_trace=trace,
        previous_lookahead_end_ms=4000.0,
    )
    context.pending_overlap_lineages = (lineage,)
    append_called = False

    class _NoAppendResiduals:
        count = MAX_OVERLAP_RESIDUAL_PAIRS - 7

        def extend_tagged(self, **kwargs: Any) -> None:
            nonlocal append_called
            append_called = True
            raise AssertionError("aggregate append should not run after residual cap")

    context.overlap_residuals = _NoAppendResiduals()  # type: ignore[assignment]

    with pytest.raises(
        lf._DiagnosticsIntegrityFailure,  # noqa: SLF001
        match=REASON_DIAGNOSTICS_INTEGRITY_FAILURE,
    ):
        lf._record_pending_overlap_at_export(  # noqa: SLF001
            context=context,
            exported_paths=(replace(path, overlap_lineage=lineage),),
            next_block_index=1,
            next_core_end_ms=4000.0,
        )
    assert not append_called
    assert context.overlap_records == []


def test_exp007_audio_residual_hash_streams_without_payload_list() -> None:
    packed = lf._PackedOverlapResiduals()  # noqa: SLF001
    residuals = tuple((beat, float(beat), float(beat) / 10.0) for beat in range(3))

    packed.extend_tagged(
        prior_block_index=5,
        prior_export_ordinal=7,
        residuals=residuals,
    )

    assert packed.count == 3
    assert packed.residual_vector_sha256() == _stable_sha(
        tuple(
            (
                5,
                7,
                beat,
                float.hex(residual_ms),
                float.hex(residual_beats),
            )
            for beat, residual_ms, residual_beats in residuals
        )
    )
    assert not hasattr(packed, "payloads")
    packed.release()
    assert packed.count == 0


def test_exp007_bounded_does_not_call_full_ledger_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _MATRIX._case_by_id("original_two_jump_120_150_100")  # noqa: SLF001
    fixture = _MATRIX._fixture_for_case(case)  # noqa: SLF001

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("full ledger builder should not run in bounded mode")

    full = fit_local_frontier_boundary_pair_transition(
        fixture.prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=fixture.candidate_set,
    )
    full_objective = full.objective_diagnostics
    assert full.ok
    assert full_objective is not None
    assert any(
        ledger.transition_entries
        for ledger in full_objective.terminal_path_ledgers
    )

    monkeypatch.setattr(lf, "BoundaryPairTransitionLedgerEntry", _raise)
    monkeypatch.setattr(lf, "_build_objective_diagnostics", _raise)
    monkeypatch.setattr(lf, "_terminal_objective_ledger", _raise)
    monkeypatch.setattr(lf, "_transition_component_cache_records", _raise)
    monkeypatch.setattr(lf, "_actual_scored_edges_with_membership", _raise)
    monkeypatch.setattr(lf, "_terminal_occurrence_records", _raise)
    monkeypatch.setattr(lf, "_provisional_occurrence_records", _raise)

    bounded = fit_local_frontier_boundary_pair_transition_bounded(
        fixture.prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=fixture.candidate_set,
    )

    assert bounded.ok
    assert bounded.fit_result.objective_diagnostics is None
    assert bounded.diagnostics.actual_scored_edge_count > 0


def test_exp007_private_lineage_is_excluded_from_path_equality_future_and_order() -> None:
    case = _MATRIX._case_by_id("clean_constant_120")  # noqa: SLF001
    fixture = _MATRIX._fixture_for_case(case)  # noqa: SLF001
    restricted = lf.restrict_timing_prediction(fixture.prediction)
    context = lf._LocalScoreContext(  # noqa: SLF001
        restricted,
        fixture.candidate_set,
        LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4,
        lf._LocalFrontierDiagnosticsMode.BOUNDED,  # noqa: SLF001
    )
    base_path = lf._bootstrap_paths(  # noqa: SLF001
        fixture.candidate_set,
        context=context,
        phases=(None,),
    )[0]
    state = lf._state_from_path(  # noqa: SLF001
        base_path,
        cut_time_ms=30_000.0,
        context=context,
    )
    trace = lf._overlap_trace_for_path(  # noqa: SLF001
        base_path,
        context=context,
        start_ms=30_000.0,
        end_ms=40_000.0,
    )
    lineage = lf._make_overlap_lineage(  # noqa: SLF001
        state=state,
        path=base_path,
        block_index=0,
        export_ordinal=0,
        core_end_ms=30_000.0,
        lookahead_end_ms=40_000.0,
        provisional_trace=trace,
    )
    with_lineage = replace(base_path, overlap_lineage=lineage)

    assert with_lineage == base_path
    assert lf._path_future_key(with_lineage) == lf._path_future_key(base_path)  # noqa: SLF001
    assert lf._path_order_key(with_lineage, context) == lf._path_order_key(  # noqa: SLF001
        base_path,
        context,
    )


def test_exp007_overlap_trace_cap_fails_closed() -> None:
    case = _MATRIX._case_by_id("clean_constant_120")  # noqa: SLF001
    fixture = _MATRIX._fixture_for_case(case)  # noqa: SLF001
    restricted = lf.restrict_timing_prediction(fixture.prediction)
    context = lf._LocalScoreContext(  # noqa: SLF001
        restricted,
        fixture.candidate_set,
        LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4,
        lf._LocalFrontierDiagnosticsMode.BOUNDED,  # noqa: SLF001
    )
    too_dense = lf._Path(  # noqa: SLF001
        origin_time_ms=0.0,
        serialized_first_start_beat=0,
        open_start_beat=0,
        open_start_time_ms=0.0,
        current_bpm=1000.0,
        previous_bpm=None,
        global_downbeat_phase=None,
        closed_sections=(),
        duration_objective_numerator=0.0,
        transition_objective=0.0,
        real_section_count=1,
        alias_switch_count=0,
        max_boundary_displacement_ms=0.0,
        replay_key=("too_dense",),
        selected_boundaries=(),
        last_boundary_anchor_id=-1,
    )

    with pytest.raises(
        lf._DiagnosticsIntegrityFailure,  # noqa: SLF001
        match=REASON_DIAGNOSTICS_INTEGRITY_FAILURE,
    ):
        lf._overlap_trace_for_path(  # noqa: SLF001
            too_dense,
            context=context,
            start_ms=0.0,
            end_ms=(MAX_OVERLAP_TRACE_BEATS + 1) * 60.0,
        )
