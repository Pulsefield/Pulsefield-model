from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from pulsefield_model.timing.v3 import local_frontier as lf
from pulsefield_model.timing.v3.local_frontier import (
    BOUNDARY_PAIR_ALIAS_MULTIPLIERS,
    EXP006_PAIR_SPARSITY_FLOOR,
    REASON_NO_ORIGIN_CANDIDATE,
    BoundaryPairTransitionComponents,
    LocalFrontierConfig,
    LocalFrontierObjectiveVariant,
    LocalFrontierScheduleArm,
    alias_aware_boundary_period_distance,
    boundary_pair_transition_components,
    fit_local_frontier_boundary_pair_transition,
    fit_local_frontier_constant_jump,
    fit_local_frontier_objective_variant,
    replay_terminal_objective,
    replay_terminal_objective_with_scrambled_pairs,
    scrambled_boundary_period_pairs,
)


# Reuse source-owned synthetic builders without duplicating their candidate
# contract.  They are loaded as helpers, not collected a second time.
_HELPER_PATH = Path(__file__).with_name("test_timing_v3_local_frontier.py")
_SPEC = importlib.util.spec_from_file_location("_exp005_fixture_helpers", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def test_alias_distance_and_pair_formula_follow_frozen_binary64_order() -> None:
    assert BOUNDARY_PAIR_ALIAS_MULTIPLIERS == (
        0.25,
        1.0 / 3.0,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
    )
    assert alias_aware_boundary_period_distance(120.0, 120.0) == 0.0
    assert alias_aware_boundary_period_distance(240.0, 120.0) == 0.0
    mismatch = alias_aware_boundary_period_distance(100.0, 150.0)
    assert mismatch == abs(math.log2(100.0 / 75.0))

    component = boundary_pair_transition_components(
        boundary_anchor_id=7,
        source_candidate_time_ms=12_000.0,
        left_period_ms=500.0,
        right_period_ms=400.0,
        left_bpm=120.0,
        right_bpm=150.0,
        boundary_support_cost=0.0,
        transition_normalizer=1.2,
        objective_variant=(
            LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
        ),
    )
    assert component.pair_cost == 0.0
    assert component.sparsity_floor == EXP006_PAIR_SPARSITY_FLOOR
    assert component.change_cost == EXP006_PAIR_SPARSITY_FLOOR
    assert component.raw_transition_cost >= 0.045
    assert component.raw_transition_cost == pytest.approx(
        0.18 * 0.25
        + 0.12 * component.alias_switch_cost
        + 0.10 * abs(math.log2(150.0 / 120.0))
    )
    assert component.normalized_increment == component.raw_transition_cost / 1.2


def test_exp005_default_is_pinned_while_explicit_exp005_has_a_ledger() -> None:
    prediction, candidates = _original_kill_fixture()
    default = fit_local_frontier_constant_jump(prediction, candidate_set=candidates)
    explicit = fit_local_frontier_objective_variant(
        prediction,
        objective_variant=LocalFrontierObjectiveVariant.EXP005_CONSTANT_CHANGE,
        candidate_set=candidates,
    )

    assert _bpms(default) == (120.0, 100.0)
    assert _bpms(explicit) == (120.0, 100.0)
    assert default.grid is not None and explicit.grid is not None
    assert default.grid.to_dict() == explicit.grid.to_dict()
    assert default.objective_diagnostics is None
    assert default.diagnostics == explicit.diagnostics
    assert default.diagnostics.candidate_fingerprint == explicit.diagnostics.candidate_fingerprint
    assert default.diagnostics.replay_fingerprint == explicit.diagnostics.replay_fingerprint
    assert explicit.objective_diagnostics is not None
    assert explicit.objective_diagnostics.objective_variant == "exp005_constant_change"


@pytest.mark.parametrize("schedule", tuple(LocalFrontierScheduleArm))
def test_original_kill_fixture_recovers_exact_pair_conditioned_grid_and_ledgers(
    schedule: LocalFrontierScheduleArm,
) -> None:
    prediction, candidates = _original_kill_fixture()
    baseline = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=schedule),
        candidate_set=candidates,
    )
    result = fit_local_frontier_boundary_pair_transition(
        prediction,
        config=LocalFrontierConfig(schedule_arm=schedule),
        candidate_set=candidates,
    )

    assert _bpms(baseline) == (120.0, 100.0)
    assert _bpms(result) == (120.0, 150.0, 100.0)
    assert result.grid is not None
    assert result.grid.section_end_time_ms(0) == pytest.approx(12_000.0)
    assert result.grid.section_end_time_ms(1) == pytest.approx(36_000.0)
    assert baseline.diagnostics.candidate_fingerprint == result.diagnostics.candidate_fingerprint

    diagnostics = result.objective_diagnostics
    assert diagnostics is not None
    assert diagnostics.objective_variant == "pair_conditioned_change_floor_1_4"
    assert diagnostics.sparsity_floor == 0.25
    assert diagnostics.selected_runner_up_margin is not None
    assert diagnostics.selected_runner_up_margin > 0.0
    assert diagnostics.transition_cache_size > 0
    assert diagnostics.provisional_path_ledgers

    correct = _ledger_for_bpms(diagnostics.terminal_path_ledgers, (120.0, 150.0, 100.0))
    shortcut = _ledger_for_bpms(diagnostics.terminal_path_ledgers, (120.0, 100.0))
    for ledger in (correct, shortcut):
        reconstructed_transition = sum(
            entry.components.normalized_increment for entry in ledger.transition_entries
        )
        assert reconstructed_transition == pytest.approx(
            ledger.recorded_transition_objective,
            abs=max(1e-12, 8.0 * math.ulp(ledger.recorded_terminal_objective)),
        )
        reconstructed = (
            ledger.normalized_duration_objective
            + reconstructed_transition
            + ledger.normalized_tail_prior
        )
        assert reconstructed == pytest.approx(
            ledger.recorded_terminal_objective,
            abs=max(1e-12, 8.0 * math.ulp(ledger.recorded_terminal_objective)),
        )
        assert all(
            entry.components.raw_transition_cost >= 0.045
            for entry in ledger.transition_entries
        )

    exp005_correct = replay_terminal_objective(correct, sparsity_floor=None)
    exp005_shortcut = replay_terminal_objective(shortcut, sparsity_floor=None)
    half_correct = replay_terminal_objective(correct, sparsity_floor=0.5)
    half_shortcut = replay_terminal_objective(shortcut, sparsity_floor=0.5)
    quarter_correct = replay_terminal_objective(correct, sparsity_floor=0.25)
    quarter_shortcut = replay_terminal_objective(shortcut, sparsity_floor=0.25)
    scrambled_correct = replay_terminal_objective_with_scrambled_pairs(
        correct,
        boundary_candidates=candidates.boundary_candidates,
    )
    scrambled_shortcut = replay_terminal_objective_with_scrambled_pairs(
        shortcut,
        boundary_candidates=candidates.boundary_candidates,
    )

    assert exp005_correct.terminal_objective > exp005_shortcut.terminal_objective
    assert half_correct.terminal_objective > half_shortcut.terminal_objective
    assert quarter_correct.terminal_objective < quarter_shortcut.terminal_objective
    assert not (
        scrambled_correct.terminal_objective
        < scrambled_shortcut.terminal_objective
    )

    mapping = scrambled_boundary_period_pairs(candidates.boundary_candidates)
    assert mapping == {
        0: (400.0, 600.0),
        1: (500.0, 400.0),
    }


def test_transition_component_cache_identity_includes_variant_periods_and_bpm_pair() -> None:
    prediction, candidates = _original_kill_fixture()
    result = fit_local_frontier_boundary_pair_transition(prediction, candidate_set=candidates)
    diagnostics = result.objective_diagnostics
    assert diagnostics is not None

    all_components = tuple(
        entry.components
        for ledger in diagnostics.terminal_path_ledgers
        for entry in ledger.transition_entries
    )
    assert all_components
    identities = {
        (
            component.objective_variant,
            component.boundary_anchor_id,
            float.hex(component.left_period_ms),
            float.hex(component.right_period_ms),
            float.hex(component.left_bpm),
            float.hex(component.right_bpm),
        )
        for component in all_components
    }
    assert len(identities) > 2
    assert all(
        component.objective_variant == "pair_conditioned_change_floor_1_4"
        for component in all_components
    )


def test_exp006_objective_diagnostics_are_deterministic_and_strict_json_finite() -> None:
    prediction, candidates = _original_kill_fixture()
    first = fit_local_frontier_boundary_pair_transition(prediction, candidate_set=candidates)
    second = fit_local_frontier_boundary_pair_transition(prediction, candidate_set=candidates)

    assert first.objective_diagnostics is not None
    assert second.objective_diagnostics is not None
    assert first.objective_diagnostics == second.objective_diagnostics
    assert (
        first.objective_diagnostics.deterministic_fingerprint
        == second.objective_diagnostics.deterministic_fingerprint
    )
    json.dumps(
        asdict(first.objective_diagnostics),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    # The path-scoped key prevents two lookahead alternatives sharing one
    # anchor from being represented as one aggregate transition record.
    provisional = first.objective_diagnostics.provisional_path_ledgers
    assert len({(item.block_index, item.committed_replay_key) for item in provisional}) == len(
        provisional
    )
    assert first.objective_diagnostics.terminal_path_occurrence_records
    assert first.objective_diagnostics.provisional_path_occurrence_records


def test_actual_scored_edge_membership_is_occurrence_specific_for_duplicate_replay_edges() -> None:
    prediction, candidates = _original_kill_fixture()
    result = fit_local_frontier_boundary_pair_transition(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=candidates,
    )
    diagnostics = result.objective_diagnostics
    assert diagnostics is not None
    edges = tuple(diagnostics.actual_scored_edges)
    assert edges
    actual_by_order = {edge.edge_order: edge for edge in edges}
    assert set(actual_by_order) == set(range(len(edges)))

    terminal_orders: set[int] = set()
    selected_orders: set[int] = set()
    assert len(diagnostics.terminal_path_occurrence_records) == len(
        diagnostics.terminal_path_ledgers
    )
    for record, ledger in zip(
        diagnostics.terminal_path_occurrence_records,
        diagnostics.terminal_path_ledgers,
        strict=True,
    ):
        assert record.selection_rank == ledger.selection_rank
        assert record.selected == ledger.selected
        assert record.replay_key == ledger.replay_key
        assert len(record.edge_orders) == len(ledger.transition_entries)
        assert tuple(
            _scored_edge_ledger_payload(actual_by_order[edge_order])
            for edge_order in record.edge_orders
        ) == tuple(
            _transition_ledger_entry_payload(entry)
            for entry in ledger.transition_entries
        )
        terminal_orders.update(record.edge_orders)
        if record.selected:
            selected_orders.update(record.edge_orders)

    provisional_orders: set[int] = set()
    assert len(diagnostics.provisional_path_occurrence_records) == len(
        diagnostics.provisional_path_ledgers
    )
    for index, (record, ledger) in enumerate(
        zip(
            diagnostics.provisional_path_occurrence_records,
            diagnostics.provisional_path_ledgers,
            strict=True,
        )
    ):
        assert record.record_index == index
        assert record.block_index == ledger.block_index
        assert record.committed_replay_key == ledger.committed_replay_key
        assert record.ranked_lookahead_replay_key == ledger.ranked_lookahead_replay_key
        assert len(record.edge_orders) == len(ledger.transition_entries)
        assert tuple(
            _scored_edge_ledger_payload(actual_by_order[edge_order])
            for edge_order in record.edge_orders
        ) == tuple(
            _transition_ledger_entry_payload(entry)
            for entry in ledger.transition_entries
        )
        provisional_orders.update(record.edge_orders)

    assert {
        edge.edge_order for edge in edges if edge.retained_terminal_path
    } == terminal_orders
    assert {
        edge.edge_order for edge in edges if edge.retained_provisional_path
    } == provisional_orders
    assert {
        edge.edge_order for edge in edges if edge.selected_traceback_path
    } == selected_orders

    by_identity: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for edge in edges:
        by_identity[(edge.predecessor_replay_key, edge.successor_replay_key)].append(
            edge
        )
    duplicate_second_jump_groups = tuple(
        sorted(group, key=lambda edge: edge.edge_order)
        for group in by_identity.values()
        if len(group) > 1
        and group[0].boundary_anchor_id == 1
        and any(edge.selected_traceback_path for edge in group)
    )
    assert duplicate_second_jump_groups

    group = duplicate_second_jump_groups[0]
    assert {(edge.stage, edge.block_index) for edge in group} == {
        ("lookahead", 0),
        ("core", 1),
    }
    selected = tuple(edge for edge in group if edge.selected_traceback_path)
    retained = tuple(edge for edge in group if edge.retained_terminal_path)
    assert len(selected) == 1
    assert len(retained) == 1
    assert selected[0].edge_order == retained[0].edge_order
    assert selected[0].stage == "core"
    assert selected[0].block_index == 1
    assert selected_orders & {edge.edge_order for edge in group} == {
        selected[0].edge_order
    }

    selected_identity_set = {
        (entry.predecessor_replay_key, entry.successor_replay_key)
        for ledger in diagnostics.terminal_path_ledgers
        if ledger.selected
        for entry in ledger.transition_entries
    }
    identity_selected_orders = {
        edge.edge_order
        for edge in edges
        if (edge.predecessor_replay_key, edge.successor_replay_key)
        in selected_identity_set
    }
    assert selected_orders <= identity_selected_orders
    assert identity_selected_orders != selected_orders


def test_frontier_class_coverage_keys_sort_by_numeric_alias_then_phase_sentinel() -> None:
    states = (
        _HELPERS._frontier_state(  # noqa: SLF001
            alias_family=100.0,
            global_downbeat_phase=2,
            deterministic_replay_key=("100", 2),
        ),
        _HELPERS._frontier_state(  # noqa: SLF001
            alias_family=60.0,
            global_downbeat_phase=1,
            deterministic_replay_key=("60", 1),
        ),
        _HELPERS._frontier_state(  # noqa: SLF001
            alias_family=100.0,
            global_downbeat_phase=None,
            deterministic_replay_key=("100", None),
        ),
        _HELPERS._frontier_state(  # noqa: SLF001
            alias_family=60.0,
            global_downbeat_phase=None,
            deterministic_replay_key=("60", None),
        ),
    )

    assert lf._sorted_unique_frontier_class_keys(states) == (  # noqa: SLF001
        (60.0, None),
        (60.0, 1),
        (100.0, None),
        (100.0, 2),
    )


def test_exp006_failure_suppresses_incomplete_measurement_v2_objective_ledger() -> None:
    prediction = _HELPERS._constant_prediction(  # noqa: SLF001
        duration_ms=20_000.0,
        bpm=120.0,
        downbeat_phase=None,
    )
    candidates = _HELPERS._candidate_set_for_prediction(  # noqa: SLF001
        prediction,
        tempo_candidates=(
            _HELPERS.TempoCandidate(bpm=120.0, source="test", score=1.0),
        ),
        origin_candidates=(),
        boundary_candidates=(),
    )

    result = fit_local_frontier_boundary_pair_transition(
        prediction,
        candidate_set=candidates,
    )

    assert not result.ok
    assert result.reason == REASON_NO_ORIGIN_CANDIDATE
    assert result.diagnostics.fallback_reason == REASON_NO_ORIGIN_CANDIDATE
    assert result.objective_diagnostics is None


def test_exp006_schema_failure_after_scoring_suppresses_incomplete_measurement_v2_objective_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, candidates = _original_kill_fixture()

    def _raise_schema_failure(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("forced schema failure")

    monkeypatch.setattr(lf, "TimingV3Grid", _raise_schema_failure)

    result = fit_local_frontier_boundary_pair_transition(
        prediction,
        candidate_set=candidates,
    )

    assert not result.ok
    assert result.reason == lf.REASON_SCHEMA_CONSTRUCTION_FAILED
    assert result.diagnostics.fallback_stage == "schema"
    assert result.objective_diagnostics is None


def _original_kill_fixture() -> tuple[Any, Any]:
    prediction = _HELPERS._two_jump_prediction(  # noqa: SLF001
        first_bpm=120.0,
        second_bpm=150.0,
        third_bpm=100.0,
        first_boundary_ms=12_000.0,
        second_boundary_ms=36_000.0,
        duration_ms=72_000.0,
        include_downbeats=False,
    )
    candidates = _HELPERS._candidate_set_for_prediction(  # noqa: SLF001
        prediction,
        tempo_candidates=(
            _HELPERS.TempoCandidate(bpm=120.0, source="test", score=1.0),
            _HELPERS.TempoCandidate(bpm=150.0, source="test", score=0.9),
            _HELPERS.TempoCandidate(bpm=100.0, source="test", score=0.8),
        ),
        origin_candidates=(
            _HELPERS.OriginCandidate(
                anchor_id=0,
                time_ms=0.0,
                bpm=120.0,
                score=1.0,
            ),
        ),
        boundary_candidates=(
            _HELPERS._boundary_candidate(  # noqa: SLF001
                anchor_id=0,
                time_ms=12_000.0,
                left_period_ms=500.0,
                right_period_ms=400.0,
            ),
            _HELPERS._boundary_candidate(  # noqa: SLF001
                anchor_id=1,
                time_ms=36_000.0,
                left_period_ms=400.0,
                right_period_ms=600.0,
            ),
        ),
    )
    return prediction, candidates


def _bpms(result: Any) -> tuple[float, ...]:
    assert result.ok and result.grid is not None
    return tuple(section.bpm for section in result.grid.sections)


def _ledger_for_bpms(ledgers: Any, bpms: tuple[float, ...]) -> Any:
    for ledger in ledgers:
        if ledger.bpm_sequence == bpms:
            return ledger
    raise AssertionError(f"terminal ledger for {bpms!r} not retained")


def _transition_ledger_entry_payload(entry: Any) -> tuple[Any, ...]:
    return (
        entry.predecessor_replay_key,
        entry.successor_replay_key,
        entry.boundary_beat,
        float.hex(entry.boundary_time_ms),
        asdict(entry.components),
    )


def _scored_edge_ledger_payload(edge: Any) -> tuple[Any, ...]:
    return (
        edge.predecessor_replay_key,
        edge.successor_replay_key,
        edge.boundary_beat,
        float.hex(edge.boundary_time_ms),
        asdict(edge.components),
    )
