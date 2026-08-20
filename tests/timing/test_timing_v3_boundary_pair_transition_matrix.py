from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import pytest

from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3 import global_constant_jump as exp004
from pulsefield_model.timing.v3.global_constant_jump import (
    BOUNDARY_CANDIDATE_SCORE_VERSION,
    CANDIDATE_CONTRACT_VERSION,
    GLOBAL_CONSTANT_JUMP_CONSTANTS,
    GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
    PULSE_CORRELATION_VERSION,
    BoundaryCandidate,
    GlobalConstantJumpCandidateDiagnostics,
    GlobalConstantJumpCandidateSet,
    OriginCandidate,
    TempoCandidate,
    materialize_global_constant_jump_peaks,
)
from pulsefield_model.timing.v3.local_frontier import (
    BOUNDARY_PAIR_ALIAS_MULTIPLIERS,
    BOUNDARY_PAIR_TRANSITION_CONTRACT_VERSION,
    EXP006_PAIR_SPARSITY_FLOOR,
    BoundaryPairTransitionComponents,
    LocalFrontierConfig,
    LocalFrontierObjectiveVariant,
    LocalFrontierResult,
    LocalFrontierScheduleArm,
    TerminalObjectiveLedger,
    alias_aware_boundary_period_distance,
    boundary_pair_transition_components,
    fit_local_frontier_boundary_pair_transition,
    fit_local_frontier_constant_jump,
    fit_local_frontier_objective_variant,
    replay_terminal_objective,
)
from pulsefield_model.timing.v3.schema import TimingV3Grid


FRAME_RATE_HZ = 50.0
DURATION_MS = 72_000.0
NOISY_CONSTANT_SEED = 6006
NOISY_CONSTANT_STD = 0.03

SCHEDULE_ARMS = (
    LocalFrontierScheduleArm.S30,
    LocalFrontierScheduleArm.S60,
    LocalFrontierScheduleArm.S90,
    LocalFrontierScheduleArm.S64,
)
EXPECTED_MATRIX_ARM_COUNT = 44


@dataclass(frozen=True)
class SegmentSpec:
    start_ms: float
    end_ms: float
    bpm: float


@dataclass(frozen=True)
class BoundarySpec:
    time_ms: float
    left_period_ms: float
    right_period_ms: float


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    segments: tuple[SegmentSpec, ...]
    tempo_bpms: tuple[float, ...]
    boundary_specs: tuple[BoundarySpec, ...]
    expected_sections: tuple[tuple[int, int, float], ...]
    baseline_expected_sections: tuple[tuple[int, int, float], ...] | None = None
    expected_alias_switch_count: int | None = None
    noise_seed: int | None = None
    noise_std: float = 0.0
    require_false_transition_floor: bool = False


@dataclass(frozen=True)
class Fixture:
    case: MatrixCase
    prediction: FrameTimingPrediction
    candidate_set: GlobalConstantJumpCandidateSet


def _period_ms(bpm: float) -> float:
    return 60000.0 / bpm


ONE_SECTION_120 = ((0, 144, 120.0),)

MATRIX_CASES = (
    MatrixCase(
        case_id="clean_constant_120",
        segments=(SegmentSpec(0.0, DURATION_MS, 120.0),),
        tempo_bpms=(120.0,),
        boundary_specs=(),
        expected_sections=ONE_SECTION_120,
        baseline_expected_sections=ONE_SECTION_120,
    ),
    MatrixCase(
        case_id="noisy_constant_120_seed_6006",
        segments=(SegmentSpec(0.0, DURATION_MS, 120.0),),
        tempo_bpms=(120.0,),
        boundary_specs=(),
        expected_sections=ONE_SECTION_120,
        baseline_expected_sections=ONE_SECTION_120,
        noise_seed=NOISY_CONSTANT_SEED,
        noise_std=NOISY_CONSTANT_STD,
    ),
    MatrixCase(
        case_id="single_jump_120_150_exact_periods",
        segments=(
            SegmentSpec(0.0, 36_000.0, 120.0),
            SegmentSpec(36_000.0, DURATION_MS, 150.0),
        ),
        tempo_bpms=(120.0, 150.0),
        boundary_specs=(
            BoundarySpec(36_000.0, _period_ms(120.0), _period_ms(150.0)),
        ),
        expected_sections=((0, 72, 120.0), (72, 162, 150.0)),
    ),
    MatrixCase(
        case_id="single_jump_120_150_left_plus_2_right_minus_2",
        segments=(
            SegmentSpec(0.0, 36_000.0, 120.0),
            SegmentSpec(36_000.0, DURATION_MS, 150.0),
        ),
        tempo_bpms=(120.0, 150.0),
        boundary_specs=(
            BoundarySpec(
                36_000.0,
                _period_ms(120.0) * 1.02,
                _period_ms(150.0) * 0.98,
            ),
        ),
        expected_sections=((0, 72, 120.0), (72, 162, 150.0)),
    ),
    MatrixCase(
        case_id="single_jump_120_150_left_minus_2_right_plus_2",
        segments=(
            SegmentSpec(0.0, 36_000.0, 120.0),
            SegmentSpec(36_000.0, DURATION_MS, 150.0),
        ),
        tempo_bpms=(120.0, 150.0),
        boundary_specs=(
            BoundarySpec(
                36_000.0,
                _period_ms(120.0) * 0.98,
                _period_ms(150.0) * 1.02,
            ),
        ),
        expected_sections=((0, 72, 120.0), (72, 162, 150.0)),
    ),
    MatrixCase(
        case_id="original_two_jump_120_150_100",
        segments=(
            SegmentSpec(0.0, 12_000.0, 120.0),
            SegmentSpec(12_000.0, 36_000.0, 150.0),
            SegmentSpec(36_000.0, DURATION_MS, 100.0),
        ),
        tempo_bpms=(120.0, 150.0, 100.0),
        boundary_specs=(
            BoundarySpec(12_000.0, _period_ms(120.0), _period_ms(150.0)),
            BoundarySpec(36_000.0, _period_ms(150.0), _period_ms(100.0)),
        ),
        expected_sections=((0, 24, 120.0), (24, 84, 150.0), (84, 144, 100.0)),
        baseline_expected_sections=((0, 24, 120.0), (24, 124, 100.0)),
    ),
    MatrixCase(
        case_id="direction_control_100_150_120",
        segments=(
            SegmentSpec(0.0, 12_000.0, 100.0),
            SegmentSpec(12_000.0, 36_000.0, 150.0),
            SegmentSpec(36_000.0, DURATION_MS, 120.0),
        ),
        tempo_bpms=(100.0, 150.0, 120.0),
        boundary_specs=(
            BoundarySpec(12_000.0, _period_ms(100.0), _period_ms(150.0)),
            BoundarySpec(36_000.0, _period_ms(150.0), _period_ms(120.0)),
        ),
        expected_sections=((0, 20, 100.0), (20, 80, 150.0), (80, 152, 120.0)),
    ),
    MatrixCase(
        case_id="short_false_island_constant_120",
        segments=(SegmentSpec(0.0, DURATION_MS, 120.0),),
        tempo_bpms=(120.0, 150.0),
        boundary_specs=(
            BoundarySpec(24_000.0, _period_ms(120.0), _period_ms(150.0)),
            BoundarySpec(32_400.0, _period_ms(150.0), _period_ms(120.0)),
        ),
        expected_sections=ONE_SECTION_120,
    ),
    MatrixCase(
        case_id="alias_trap_constant_120",
        segments=(SegmentSpec(0.0, DURATION_MS, 120.0),),
        tempo_bpms=(60.0, 120.0, 240.0),
        boundary_specs=(
            BoundarySpec(24_000.0, _period_ms(120.0), _period_ms(240.0)),
            BoundarySpec(32_500.0, _period_ms(240.0), _period_ms(120.0)),
        ),
        expected_sections=ONE_SECTION_120,
        expected_alias_switch_count=0,
    ),
    MatrixCase(
        case_id="dense_compatible_islands_constant_120",
        segments=(SegmentSpec(0.0, DURATION_MS, 120.0),),
        tempo_bpms=(120.0, 150.0),
        boundary_specs=(
            BoundarySpec(12_000.0, _period_ms(120.0), _period_ms(150.0)),
            BoundarySpec(20_400.0, _period_ms(150.0), _period_ms(120.0)),
            BoundarySpec(28_800.0, _period_ms(120.0), _period_ms(150.0)),
            BoundarySpec(37_200.0, _period_ms(150.0), _period_ms(120.0)),
        ),
        expected_sections=ONE_SECTION_120,
        require_false_transition_floor=True,
    ),
    MatrixCase(
        case_id="dense_alias_islands_constant_120",
        segments=(SegmentSpec(0.0, DURATION_MS, 120.0),),
        tempo_bpms=(60.0, 120.0, 240.0),
        boundary_specs=(
            BoundarySpec(12_000.0, _period_ms(120.0), _period_ms(240.0)),
            BoundarySpec(20_400.0, _period_ms(240.0), _period_ms(120.0)),
            BoundarySpec(28_800.0, _period_ms(120.0), _period_ms(240.0)),
            BoundarySpec(37_200.0, _period_ms(240.0), _period_ms(120.0)),
        ),
        expected_sections=ONE_SECTION_120,
        expected_alias_switch_count=0,
        require_false_transition_floor=True,
    ),
)

EXPECTED_INPUT_SIGNAL_SHA256_BY_CASE = {
    "clean_constant_120": "6ddcd88393d65184a947ba9c4e14136e28360760f9b18369ee43d6f6f7f19fce",
    "noisy_constant_120_seed_6006": "9eaa1ad6e61b9ade070bb83e3d59d32fdd012ca54f7bcb4a433e5ee18cb989d0",
    "single_jump_120_150_exact_periods": "f216c394cd759c625f3332a77c2354132b188854a7c5b462528af0d39d26dfdf",
    "single_jump_120_150_left_plus_2_right_minus_2": "f216c394cd759c625f3332a77c2354132b188854a7c5b462528af0d39d26dfdf",
    "single_jump_120_150_left_minus_2_right_plus_2": "f216c394cd759c625f3332a77c2354132b188854a7c5b462528af0d39d26dfdf",
    "original_two_jump_120_150_100": "21c66bff2e4285838feea04ad4474355150bf9f0299e6520235a92c0a1adead9",
    "direction_control_100_150_120": "432bf04d7668633643595360bc146739d219536611d5c965cb00c009c7be49db",
    "short_false_island_constant_120": "fd16a9094b6423c1efa2ed33b42a4e6f50812dc8e3b9bb76ede5cd6eb6775bf2",
    "alias_trap_constant_120": "6ddcd88393d65184a947ba9c4e14136e28360760f9b18369ee43d6f6f7f19fce",
    "dense_compatible_islands_constant_120": "3d8a9f9e1481e73a70c80afbc414050bc6b8dd425c7e697072314cff3e5a2744",
    "dense_alias_islands_constant_120": "3d8a9f9e1481e73a70c80afbc414050bc6b8dd425c7e697072314cff3e5a2744",
}

EXPECTED_CANDIDATE_FINGERPRINT_BY_CASE = {
    "clean_constant_120": "c96941fd943af57242e55796c3ddc320f853187c5c04f5f1d3a0fb5b8b89590d",
    "noisy_constant_120_seed_6006": "674a0f06085814f1d6c63d57f08b332491de08a0b58ff674f35fe59b54b45e99",
    "single_jump_120_150_exact_periods": "37503f6fb03d56919b6b1a42b1f40cdab44f60b75efbe4e835398b4f00609cab",
    "single_jump_120_150_left_plus_2_right_minus_2": "6e57905435626d6d2bb7253981c84fd3b0379d7fad724b76f9293e746ab43d3b",
    "single_jump_120_150_left_minus_2_right_plus_2": "dd68b8e573db9c9e8423f87f903620fce2f1db19635765f460e7f3014a7907d8",
    "original_two_jump_120_150_100": "208b7c10842ea933a89807cf446a4aa2c5ce3da7a6639a2edad52dac71682644",
    "direction_control_100_150_120": "ce9b6d5732e3b2c288785d2b6042859ffd9f0b303549ddd78786d6d161d1217a",
    "short_false_island_constant_120": "1044366c4a7d07177a799ee2693b413af44203863c476b2c0a2c3ed6cd110f4a",
    "alias_trap_constant_120": "9f09057669f2aa0688ca01237dd498a32e652ada82d60d2d74aec201dfd73212",
    "dense_compatible_islands_constant_120": "ff6c5820d58a02381bfa38f8d17b7514fda3ce13526b9fc4da9275ad756a9684",
    "dense_alias_islands_constant_120": "58621f5118ec80c2f6aa39470ca5e8e0f953fb5e13ed9c45f6e767c3548ac1a5",
}

EXPECTED_MATRIX_DEFINITION_SHA256 = (
    "d9545723b82aa385d4dceb4051b8dd6cc2791208cc7624c546b870652c816aa2"
)
EXPECTED_DEFAULT_EXP005_LEGACY_AGGREGATE_SHA256 = (
    "a7401b73d9ed9dca9b3f7b65ae599e6715d9a9df2bf621cc063d1f54e88b57c6"
)
EXPECTED_EXP006_BEHAVIOR_AGGREGATE_SHA256 = (
    "f72574d4af07d6b2b13699276ba9a60d68c915819459851edca7217780b26af8"
)
EXPECTED_EXPLICIT_EXP005_LEDGER_AGGREGATE_SHA256 = (
    "3911d91c787e92af8b4e9554a20186b7f178f1a66e9d040f4c86f1963806a127"
)
EXPECTED_TRANSITION_CACHE_AGGREGATE_SHA256 = (
    "abf249fdff92c60712b2d05e02f736768079958e5fa7353a86a921e31f66e7a9"
)
EXPECTED_SCORED_EDGE_ACCESS_AGGREGATE_SHA256 = (
    "e1f8135afbe114aabeeb7162694fa84f8740c84242eb4313e6de44ca3b27e438"
)
EXPECTED_ORIGINAL_KILL_DEFAULT_EXP005_FULL_PAYLOAD_SHA256 = {
    LocalFrontierScheduleArm.S30: (
        "95fe5e5be4ffdc846b3f120d86011d8a24545e78ffe7601d4ed127f9c2c51d2a"
    ),
    LocalFrontierScheduleArm.S60: (
        "5151384b5289cc1cb28baf1b679a3b21108bd3018f72141ecc864cde050c0eff"
    ),
    LocalFrontierScheduleArm.S90: (
        "1e9f6f89522940e23cb0a237a393318d90f73afee788922dc4020ea54b7e9a7b"
    ),
    LocalFrontierScheduleArm.S64: (
        "483259df753406b3410296374e5b3c3485ee5556fd432aaf6d32f92c4e95b229"
    ),
}
EXPECTED_DENSE_PUBLIC_TRACE_BY_CASE_AND_ARM = {
    "dense_compatible_islands_constant_120": {
        LocalFrontierScheduleArm.S30: (1352, 299, {0, 1, 2, 3}, {0, 1, 3}),
        LocalFrontierScheduleArm.S60: (1170, 234, {0, 1, 2, 3}, {0, 1, 3}),
        LocalFrontierScheduleArm.S90: (1170, 234, {0, 1, 2, 3}, {0, 1, 3}),
        LocalFrontierScheduleArm.S64: (1170, 234, {0, 1, 2, 3}, {0, 1, 3}),
    },
    "dense_alias_islands_constant_120": {
        LocalFrontierScheduleArm.S30: (11, 11, {0}, {0}),
        LocalFrontierScheduleArm.S60: (11, 11, {0}, {0}),
        LocalFrontierScheduleArm.S90: (11, 11, {0}, {0}),
        LocalFrontierScheduleArm.S64: (11, 11, {0}, {0}),
    },
}
OBJECTIVE_V1_BEHAVIOR_FIELD_NAMES = {
    "objective_variant",
    "sparsity_floor",
    "alias_multipliers",
    "candidate_fingerprint",
    "transition_cache_size",
    "terminal_path_ledgers",
    "provisional_path_ledgers",
    "selected_terminal_objective",
    "runner_up_terminal_objective",
    "selected_runner_up_margin",
}
OBJECTIVE_NON_BEHAVIOR_FIELD_NAMES = {
    "contract_version",
    "deterministic_fingerprint",
}
OBJECTIVE_V2_MEASUREMENT_FIELD_NAMES = {
    "transition_component_cache_entries",
    "actual_scored_edges",
    "block_resource_records",
    "class_coverage_records",
    "terminal_path_occurrence_records",
    "provisional_path_occurrence_records",
    "selected_traceback_edge_orders",
}
MAX_MATRIX_ARM_RUNTIME_SECONDS = 10.0


def _matrix_params() -> tuple[tuple[MatrixCase, LocalFrontierScheduleArm], ...]:
    return tuple((case, schedule) for case in MATRIX_CASES for schedule in SCHEDULE_ARMS)


def test_exp006_matrix_defines_exact_frozen_44_arms() -> None:
    assert tuple(case.case_id for case in MATRIX_CASES) == (
        "clean_constant_120",
        "noisy_constant_120_seed_6006",
        "single_jump_120_150_exact_periods",
        "single_jump_120_150_left_plus_2_right_minus_2",
        "single_jump_120_150_left_minus_2_right_plus_2",
        "original_two_jump_120_150_100",
        "direction_control_100_150_120",
        "short_false_island_constant_120",
        "alias_trap_constant_120",
        "dense_compatible_islands_constant_120",
        "dense_alias_islands_constant_120",
    )
    assert SCHEDULE_ARMS == (
        LocalFrontierScheduleArm.S30,
        LocalFrontierScheduleArm.S60,
        LocalFrontierScheduleArm.S90,
        LocalFrontierScheduleArm.S64,
    )
    assert len(MATRIX_CASES) * len(SCHEDULE_ARMS) == EXPECTED_MATRIX_ARM_COUNT
    assert len(_matrix_params()) == EXPECTED_MATRIX_ARM_COUNT
    assert _matrix_definition_fingerprint() == EXPECTED_MATRIX_DEFINITION_SHA256


def test_pair_conditioned_component_formula_is_frozen() -> None:
    assert BOUNDARY_PAIR_ALIAS_MULTIPLIERS == GLOBAL_CONSTANT_JUMP_CONSTANTS.alias_multipliers
    assert alias_aware_boundary_period_distance(120.0, 120.0) == pytest.approx(0.0)
    assert alias_aware_boundary_period_distance(240.0, 120.0) == pytest.approx(0.0)
    assert alias_aware_boundary_period_distance(60.0, 120.0) == pytest.approx(0.0)
    assert alias_aware_boundary_period_distance(100.0, 150.0) == pytest.approx(
        abs(math.log2(100.0 / (0.5 * 150.0)))
    )

    exp006_exact = boundary_pair_transition_components(
        boundary_anchor_id=0,
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
    _assert_component_matches_manual_formula(exp006_exact)
    assert exp006_exact.pair_cost == pytest.approx(0.0)
    assert exp006_exact.change_cost == pytest.approx(EXP006_PAIR_SPARSITY_FLOOR)
    assert exp006_exact.raw_transition_cost == pytest.approx(
        0.18 * 0.25
        + 0.12 * 1.0
        + 0.10 * abs(math.log2(150.0 / 120.0))
    )

    exp006_shortcut = boundary_pair_transition_components(
        boundary_anchor_id=0,
        source_candidate_time_ms=12_000.0,
        left_period_ms=500.0,
        right_period_ms=400.0,
        left_bpm=120.0,
        right_bpm=100.0,
        boundary_support_cost=0.0,
        transition_normalizer=1.2,
        objective_variant=(
            LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
        ),
    )
    _assert_component_matches_manual_formula(exp006_shortcut)
    assert exp006_shortcut.alias_distance_left == pytest.approx(0.0)
    assert exp006_shortcut.alias_distance_right == pytest.approx(
        abs(math.log2(100.0 / (0.5 * 150.0)))
    )
    assert exp006_shortcut.change_cost == pytest.approx(
        0.25 + 0.75 * exp006_shortcut.pair_cost
    )

    exp005 = boundary_pair_transition_components(
        boundary_anchor_id=0,
        source_candidate_time_ms=12_000.0,
        left_period_ms=500.0,
        right_period_ms=400.0,
        left_bpm=120.0,
        right_bpm=100.0,
        boundary_support_cost=0.0,
        transition_normalizer=1.2,
        objective_variant=LocalFrontierObjectiveVariant.EXP005_CONSTANT_CHANGE,
    )
    _assert_component_matches_manual_formula(exp005)
    assert exp005.sparsity_floor is None
    assert exp005.change_cost == pytest.approx(1.0)
    assert exp005.raw_transition_cost == pytest.approx(
        0.18 + 0.12 * 1.0 + 0.10 * abs(math.log2(100.0 / 120.0))
    )


@pytest.mark.parametrize(("case", "schedule_arm"), _matrix_params())
def test_exp006_boundary_pair_transition_matrix(
    case: MatrixCase,
    schedule_arm: LocalFrontierScheduleArm,
    record_property: Any,
) -> None:
    started_at = time.perf_counter()
    fixture = _fixture_for_case(case)
    _assert_fixture_fingerprints(fixture)

    config = LocalFrontierConfig(schedule_arm=schedule_arm)
    baseline = fit_local_frontier_constant_jump(
        fixture.prediction,
        config=config,
        candidate_set=fixture.candidate_set,
    )
    exp006_first = fit_local_frontier_boundary_pair_transition(
        fixture.prediction,
        config=config,
        candidate_set=fixture.candidate_set,
    )
    exp006_second = fit_local_frontier_boundary_pair_transition(
        fixture.prediction,
        config=config,
        candidate_set=fixture.candidate_set,
    )

    _assert_ok(baseline)
    _assert_expected_grid(exp006_first, case.expected_sections)
    if case.baseline_expected_sections is not None:
        _assert_expected_grid(baseline, case.baseline_expected_sections)

    _assert_local_frontier_diagnostics_guard(baseline, fixture, config)
    _assert_local_frontier_diagnostics_guard(exp006_first, fixture, config)
    _assert_local_frontier_diagnostics_guard(exp006_second, fixture, config)
    _assert_candidate_fingerprint_reused(fixture, baseline, exp006_first, exp006_second)
    _assert_exp006_objective_diagnostics(exp006_first, fixture)
    _assert_exp006_measurement_v2_records(exp006_first, fixture, config)
    _assert_transition_ledgers_reconstruct(exp006_first)
    _assert_two_run_determinism(exp006_first, exp006_second)

    if case.expected_alias_switch_count == 0:
        assert _selected_terminal_ledger(exp006_first).transition_entries == ()
    if case.case_id in {
        "single_jump_120_150_left_plus_2_right_minus_2",
        "single_jump_120_150_left_minus_2_right_plus_2",
    }:
        _assert_perturbed_single_jump_margin(exp006_first)
    if case.require_false_transition_floor:
        _assert_dense_anchor_floor_coverage(exp006_first, fixture, schedule_arm)
    if case.case_id == "original_two_jump_120_150_100":
        _assert_original_kill_counterfactual_order(fixture, schedule_arm, exp006_first)
    runtime_seconds = time.perf_counter() - started_at
    record_property("case_id", case.case_id)
    record_property("schedule_arm", schedule_arm.value)
    record_property("runtime_seconds", runtime_seconds)
    assert runtime_seconds < MAX_MATRIX_ARM_RUNTIME_SECONDS


def test_measurement_repair_public_oracle_hashes_match_frozen_card() -> None:
    rows = _matrix_oracle_rows()
    default_aggregate = _stable_json_sha256(
        [
            {"case": row["case"], "arm": row["arm"], "sha": row["default_sha"]}
            for row in rows
        ]
    )
    exp006_behavior_aggregate = _stable_json_sha256(
        [
            {
                "case": row["case"],
                "arm": row["arm"],
                "sha": row["exp006_behavior_sha"],
            }
            for row in rows
        ]
    )
    explicit_exp005_aggregate = _stable_json_sha256(
        [
            {
                "case": row["case"],
                "arm": row["arm"],
                "sha": row["explicit_exp005_sha"],
            }
            for row in rows
        ]
    )
    cache_aggregate = _stable_json_sha256(
        [
            {
                "case": row["case"],
                "arm": row["arm"],
                "sha": row["cache_sha"],
                "n": row["cache_count"],
            }
            for row in rows
        ]
    )
    access_aggregate = _stable_json_sha256(
        [
            {
                "case": row["case"],
                "arm": row["arm"],
                "count": row["access_count"],
                "sha": row["access_sha"],
                "unique_count": row["access_unique_count"],
                "unique_sha": row["access_unique_sha"],
            }
            for row in rows
        ]
    )

    assert default_aggregate == EXPECTED_DEFAULT_EXP005_LEGACY_AGGREGATE_SHA256
    assert exp006_behavior_aggregate == EXPECTED_EXP006_BEHAVIOR_AGGREGATE_SHA256
    assert explicit_exp005_aggregate == EXPECTED_EXPLICIT_EXP005_LEDGER_AGGREGATE_SHA256
    assert cache_aggregate == EXPECTED_TRANSITION_CACHE_AGGREGATE_SHA256
    assert access_aggregate == EXPECTED_SCORED_EDGE_ACCESS_AGGREGATE_SHA256


def test_original_kill_default_exp005_full_payload_hashes_stay_frozen() -> None:
    case = _case_by_id("original_two_jump_120_150_100")
    fixture = _fixture_for_case(case)
    for schedule_arm, expected_sha in (
        EXPECTED_ORIGINAL_KILL_DEFAULT_EXP005_FULL_PAYLOAD_SHA256.items()
    ):
        result = fit_local_frontier_constant_jump(
            fixture.prediction,
            config=LocalFrontierConfig(schedule_arm=schedule_arm),
            candidate_set=fixture.candidate_set,
        )
        payload = {
            "reason": result.reason,
            "grid": None if result.grid is None else result.grid.to_dict(),
            "diagnostics": asdict(result.diagnostics),
            "objective": result.objective_diagnostics,
        }
        assert _stable_json_sha256(payload) == expected_sha


def _assert_original_kill_counterfactual_order(
    fixture: Fixture,
    schedule_arm: LocalFrontierScheduleArm,
    exp006_result: LocalFrontierResult,
) -> None:
    baseline = fit_local_frontier_objective_variant(
        fixture.prediction,
        objective_variant=LocalFrontierObjectiveVariant.EXP005_CONSTANT_CHANGE,
        config=LocalFrontierConfig(schedule_arm=schedule_arm),
        candidate_set=fixture.candidate_set,
    )
    _assert_expected_grid(baseline, fixture.case.baseline_expected_sections or ())
    _assert_exp005_objective_stays_legacy_v1(baseline)
    _assert_transition_ledgers_reconstruct(baseline)

    correct = _find_terminal_ledger(
        exp006_result,
        fixture.case.expected_sections,
    )
    shortcut = _find_terminal_ledger(
        exp006_result,
        fixture.case.baseline_expected_sections or (),
    )

    exp005_correct = replay_terminal_objective(correct, sparsity_floor=None)
    exp005_shortcut = replay_terminal_objective(shortcut, sparsity_floor=None)
    rho_half_correct = replay_terminal_objective(correct, sparsity_floor=0.5)
    rho_half_shortcut = replay_terminal_objective(shortcut, sparsity_floor=0.5)
    rho_quarter_correct = replay_terminal_objective(
        correct,
        sparsity_floor=EXP006_PAIR_SPARSITY_FLOOR,
    )
    rho_quarter_shortcut = replay_terminal_objective(
        shortcut,
        sparsity_floor=EXP006_PAIR_SPARSITY_FLOOR,
    )
    scrambled_mapping = _explicit_anchor_swap_period_mapping(fixture)
    scrambled_correct = replay_terminal_objective(
        correct,
        sparsity_floor=EXP006_PAIR_SPARSITY_FLOOR,
        boundary_period_pairs_by_anchor=scrambled_mapping,
    )
    scrambled_shortcut = replay_terminal_objective(
        shortcut,
        sparsity_floor=EXP006_PAIR_SPARSITY_FLOOR,
        boundary_period_pairs_by_anchor=scrambled_mapping,
    )

    assert exp005_shortcut.terminal_objective < exp005_correct.terminal_objective
    assert rho_half_shortcut.terminal_objective < rho_half_correct.terminal_objective
    assert rho_quarter_correct.terminal_objective < rho_quarter_shortcut.terminal_objective
    assert scrambled_correct.terminal_objective >= scrambled_shortcut.terminal_objective

    objective = exp006_result.objective_diagnostics
    assert objective is not None
    assert objective.selected_runner_up_margin is not None
    assert objective.selected_runner_up_margin > 0.0


def _explicit_anchor_swap_period_mapping(
    fixture: Fixture,
) -> dict[int, tuple[float, float]]:
    assert len(fixture.candidate_set.boundary_candidates) == 2
    left = fixture.candidate_set.boundary_candidates[0]
    right = fixture.candidate_set.boundary_candidates[1]
    mapping = {
        left.anchor_id: (right.left_period_ms, right.right_period_ms),
        right.anchor_id: (left.left_period_ms, left.right_period_ms),
    }
    assert set(mapping) == {left.anchor_id, right.anchor_id}
    assert mapping[left.anchor_id] == (right.left_period_ms, right.right_period_ms)
    assert mapping[right.anchor_id] == (left.left_period_ms, left.right_period_ms)
    assert mapping[left.anchor_id] != (left.left_period_ms, left.right_period_ms)
    assert mapping[right.anchor_id] != (right.left_period_ms, right.right_period_ms)
    return mapping


@lru_cache(maxsize=None)
def _fixture_for_case(case: MatrixCase) -> Fixture:
    prediction = _piecewise_prediction(case)
    for boundary in case.boundary_specs:
        _write_pulse(
            prediction.beat_prob,
            time_ms=boundary.time_ms,
            frame_rate_hz=prediction.frame_rate_hz,
        )
    candidate_set = _candidate_set_for_case(prediction, case)
    return Fixture(case=case, prediction=prediction, candidate_set=candidate_set)


def _case_by_id(case_id: str) -> MatrixCase:
    for case in MATRIX_CASES:
        if case.case_id == case_id:
            return case
    raise AssertionError(f"unknown matrix case {case_id!r}")


def _piecewise_prediction(case: MatrixCase) -> FrameTimingPrediction:
    frame_count = int(round(DURATION_MS * FRAME_RATE_HZ / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    beat_index = 0

    for segment_index, segment in enumerate(case.segments):
        period_ms = _period_ms(segment.bpm)
        time_ms = segment.start_ms if segment_index == 0 else segment.start_ms + period_ms
        while time_ms <= segment.end_ms + 1e-9 and time_ms < DURATION_MS:
            _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=FRAME_RATE_HZ)
            beat_index += 1
            time_ms += period_ms

    if case.noise_seed is not None:
        rng = np.random.default_rng(case.noise_seed)
        beat_prob = np.clip(
            beat_prob + rng.normal(0.0, case.noise_std, size=frame_count).astype(np.float32),
            0.0,
            1.0,
        ).astype(np.float32)
        downbeat_prob = np.clip(
            downbeat_prob + rng.normal(0.0, case.noise_std, size=frame_count).astype(np.float32),
            0.0,
            1.0,
        ).astype(np.float32)

    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path=None,
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=FRAME_RATE_HZ,
    )


def _write_pulse(signal: np.ndarray, *, time_ms: float, frame_rate_hz: float) -> None:
    frame = int(round(time_ms * frame_rate_hz / 1000.0))
    if 0 <= frame < signal.shape[0]:
        signal[frame] = max(signal[frame], np.float32(1.0))
    if 0 <= frame - 1 < signal.shape[0]:
        signal[frame - 1] = max(signal[frame - 1], np.float32(0.1))
    if 0 <= frame + 1 < signal.shape[0]:
        signal[frame + 1] = max(signal[frame + 1], np.float32(0.1))


def _candidate_set_for_case(
    prediction: FrameTimingPrediction,
    case: MatrixCase,
) -> GlobalConstantJumpCandidateSet:
    beat_peaks = materialize_global_constant_jump_peaks(
        prediction.beat_prob,
        frame_rate_hz=prediction.frame_rate_hz,
    )
    downbeat_peaks = materialize_global_constant_jump_peaks(
        prediction.downbeat_prob,
        frame_rate_hz=prediction.frame_rate_hz,
    )
    boundary_candidates = _boundary_candidates_for_case(case, beat_peaks)
    _assert_strict_boundary_contract(boundary_candidates, beat_peaks)

    tempo_candidates = tuple(
        TempoCandidate(bpm=bpm, source="exp006-matrix", score=1.0 - index * 0.01)
        for index, bpm in enumerate(case.tempo_bpms)
    )
    origin_candidates = (
        OriginCandidate(
            anchor_id=0,
            time_ms=0.0,
            bpm=case.segments[0].bpm,
            score=1.0,
        ),
    )
    beat_signal = np.asarray(prediction.beat_prob, dtype=np.float64)
    downbeat_signal = np.asarray(prediction.downbeat_prob, dtype=np.float64)
    input_signal_sha256 = exp004._input_signal_sha256(beat_signal, downbeat_signal)  # noqa: SLF001
    candidate_fingerprint = exp004._candidate_fingerprint(  # noqa: SLF001
        tempo_candidates=tempo_candidates,
        origin_candidates=origin_candidates,
        boundary_candidates=boundary_candidates,
        beat_peaks=tuple(beat_peaks),
        downbeat_peaks=tuple(downbeat_peaks),
        input_signal_sha256=input_signal_sha256,
    )
    (
        frame_count,
        frame_rate_hz,
        coverage_start_ms,
        coverage_end_ms,
        min_period_frames,
        max_period_frames,
    ) = exp004._prediction_geometry(  # noqa: SLF001
        beat_signal,
        prediction.frame_rate_hz,
        GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )
    diagnostics = GlobalConstantJumpCandidateDiagnostics(
        candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
        constants_json_sha256=GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        pulse_correlation_version=PULSE_CORRELATION_VERSION,
        boundary_candidate_score_version=BOUNDARY_CANDIDATE_SCORE_VERSION,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        min_period_frames=min_period_frames,
        max_period_frames=max_period_frames,
        beat_peak_count=len(beat_peaks),
        downbeat_peak_count=len(downbeat_peaks),
        tempo_candidate_count=len(tempo_candidates),
        origin_candidate_count=len(origin_candidates),
        boundary_candidate_count=len(boundary_candidates),
        input_signal_sha256=input_signal_sha256,
        candidate_fingerprint=candidate_fingerprint,
    )
    return GlobalConstantJumpCandidateSet(
        beat_peaks=tuple(beat_peaks),
        downbeat_peaks=tuple(downbeat_peaks),
        tempo_candidates=tempo_candidates,
        origin_candidates=origin_candidates,
        boundary_candidates=boundary_candidates,
        diagnostics=diagnostics,
    )


def _boundary_candidates_for_case(
    case: MatrixCase,
    beat_peaks: tuple[Any, ...],
) -> tuple[BoundaryCandidate, ...]:
    boundaries = []
    for anchor_id, spec in enumerate(case.boundary_specs):
        peak_index, peak = _exact_peak_at(beat_peaks, spec.time_ms)
        boundaries.append(
            BoundaryCandidate(
                anchor_id=anchor_id,
                time_ms=spec.time_ms,
                source_peak_index=peak_index,
                source_peak_time_ms=peak.time_ms,
                source_peak_confidence=peak.confidence,
                rank_score=1.0 - anchor_id * 0.01,
                evidence_mode="ordinary",
                left_period_ms=spec.left_period_ms,
                right_period_ms=spec.right_period_ms,
                ordinary_score=1.0,
                super_score=None,
                downbeat_bonus=0.0,
                nearest_downbeat_distance_ms=None,
            )
        )
    return tuple(boundaries)


def _exact_peak_at(peaks: Sequence[Any], time_ms: float) -> tuple[int, Any]:
    for peak_index, peak in enumerate(peaks):
        if peak.time_ms == time_ms:
            return peak_index, peak
    raise AssertionError(f"no exact materialized source peak at {time_ms!r} ms")


def _assert_strict_boundary_contract(
    boundary_candidates: tuple[BoundaryCandidate, ...],
    beat_peaks: tuple[Any, ...],
) -> None:
    previous_time_ms = -math.inf
    for index, boundary in enumerate(boundary_candidates):
        assert boundary.anchor_id == index
        assert boundary.time_ms > previous_time_ms
        if previous_time_ms > -math.inf:
            assert (
                boundary.time_ms - previous_time_ms
                > GLOBAL_CONSTANT_JUMP_CONSTANTS.boundary_merge_ms
            )
        assert boundary.source_peak_time_ms == boundary.time_ms
        assert beat_peaks[boundary.source_peak_index].time_ms == boundary.time_ms
        assert boundary.left_period_ms > 0.0
        assert boundary.right_period_ms > 0.0
        previous_time_ms = boundary.time_ms


def _assert_fixture_fingerprints(fixture: Fixture) -> None:
    case_id = fixture.case.case_id
    diagnostics = fixture.candidate_set.diagnostics
    assert diagnostics.input_signal_sha256 == EXPECTED_INPUT_SIGNAL_SHA256_BY_CASE[case_id]
    assert (
        diagnostics.candidate_fingerprint
        == EXPECTED_CANDIDATE_FINGERPRINT_BY_CASE[case_id]
    )
    _assert_sha256(diagnostics.input_signal_sha256)
    _assert_sha256(diagnostics.candidate_fingerprint)


def _assert_ok(result: LocalFrontierResult) -> TimingV3Grid:
    assert isinstance(result, LocalFrontierResult)
    assert result.ok
    assert result.reason is None
    assert result.grid is not None
    assert result.diagnostics.fallback_reason is None
    return result.grid


def _assert_expected_grid(
    result: LocalFrontierResult,
    expected_sections: tuple[tuple[int, int, float], ...],
) -> None:
    grid = _assert_ok(result)
    assert grid.origin_beat == 0
    assert grid.origin_time_ms == pytest.approx(0.0)
    assert grid.coverage_start_ms == pytest.approx(0.0)
    assert grid.coverage_end_ms == pytest.approx(DURATION_MS)
    assert _grid_signature(grid) == tuple(
        (start, end, float.hex(bpm)) for start, end, bpm in expected_sections
    )
    assert grid.boundary_times_ms == pytest.approx(
        _expected_boundary_times_ms(expected_sections),
        rel=0.0,
        abs=1e-6,
    )
    assert result.diagnostics.selected_section_count == len(expected_sections)
    assert result.diagnostics.grid_fingerprint is not None
    _assert_sha256(result.diagnostics.grid_fingerprint)
    _assert_sha256(result.diagnostics.replay_fingerprint)
    _assert_sha256(result.diagnostics.deterministic_projection_sha256)


def _grid_signature(grid: TimingV3Grid) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (section.start_beat, section.end_beat, float.hex(section.bpm))
        for section in grid.sections
    )


def _expected_boundary_times_ms(
    expected_sections: tuple[tuple[int, int, float], ...],
) -> tuple[float, ...]:
    times = [0.0]
    current = 0.0
    for start_beat, end_beat, bpm in expected_sections:
        current += (end_beat - start_beat) * _period_ms(bpm)
        times.append(float(current))
    return tuple(times)


def _assert_candidate_fingerprint_reused(
    fixture: Fixture,
    *results: LocalFrontierResult,
) -> None:
    expected = fixture.candidate_set.diagnostics.candidate_fingerprint
    for result in results:
        assert result.diagnostics.candidate_fingerprint == expected
        if result.objective_diagnostics is not None:
            assert result.objective_diagnostics.candidate_fingerprint == expected


def _assert_exp006_objective_diagnostics(
    result: LocalFrontierResult,
    fixture: Fixture,
) -> None:
    objective = _require_exp006_objective_v2(result)
    assert objective.contract_version == BOUNDARY_PAIR_TRANSITION_CONTRACT_VERSION
    assert objective.objective_variant == (
        LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4.value
    )
    assert objective.sparsity_floor == pytest.approx(EXP006_PAIR_SPARSITY_FLOOR)
    assert objective.alias_multipliers == BOUNDARY_PAIR_ALIAS_MULTIPLIERS
    assert objective.candidate_fingerprint == fixture.candidate_set.diagnostics.candidate_fingerprint
    assert objective.terminal_path_ledgers
    assert objective.terminal_path_ledgers[0].selected
    assert objective.selected_terminal_objective == pytest.approx(
        objective.terminal_path_ledgers[0].recorded_terminal_objective
    )
    if objective.runner_up_terminal_objective is not None:
        assert objective.selected_runner_up_margin == pytest.approx(
            objective.runner_up_terminal_objective
            - objective.selected_terminal_objective
        )
        assert objective.selected_runner_up_margin > 0.0
    _assert_sha256(objective.deterministic_fingerprint)


def _assert_exp006_measurement_v2_records(
    result: LocalFrontierResult,
    fixture: Fixture,
    config: LocalFrontierConfig,
) -> None:
    objective = _require_exp006_objective_v2(result)
    diagnostics = result.diagnostics
    cache_entries = tuple(objective.transition_component_cache_entries)
    actual_edges = tuple(objective.actual_scored_edges)
    resource_records = tuple(objective.block_resource_records)
    class_records = tuple(objective.class_coverage_records)

    assert len(cache_entries) == objective.transition_cache_size
    cache_keys = tuple(entry.component_cache_key for entry in cache_entries)
    assert cache_keys == tuple(sorted(cache_keys))
    assert len(set(cache_keys)) == len(cache_keys)
    cache_by_key = {entry.component_cache_key: entry for entry in cache_entries}
    for entry in cache_entries:
        _assert_component_matches_manual_formula(entry.components)
        assert entry.component_cache_key == _component_cache_identity(entry.components)

    seen_edge_orders = []
    for edge in actual_edges:
        seen_edge_orders.append(edge.edge_order)
        assert edge.objective_variant == (
            LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4.value
        )
        assert edge.stage in {"core", "lookahead"}
        assert 0 <= edge.block_index < diagnostics.block_count
        assert 0 <= edge.boundary_anchor_id < len(fixture.candidate_set.boundary_candidates)
        assert edge.snapped_beat == edge.boundary_beat
        assert edge.snapped_time_ms == pytest.approx(edge.boundary_time_ms)
        assert edge.component_cache_key in cache_by_key
        assert edge.component_cache_key == _component_cache_identity(edge.components)
        assert asdict(edge.components) == asdict(
            cache_by_key[edge.component_cache_key].components
        )
        assert edge.normalized_increment == pytest.approx(
            edge.components.normalized_increment
        )
    assert seen_edge_orders == list(range(len(actual_edges)))
    _assert_occurrence_membership_flags_match_public_ledgers(objective, actual_edges)

    assert len(resource_records) == diagnostics.block_count
    assert len(class_records) == diagnostics.block_count
    previous_row_score_miss_after = 0
    previous_cache_after = 0
    previous_scored_edge_after = 0
    for block, resource, class_record in zip(
        diagnostics.block_diagnostics,
        resource_records,
        class_records,
        strict=True,
    ):
        assert resource.block_index == block.block_index
        assert resource.core_start_ms == pytest.approx(block.core_start_ms)
        assert resource.core_end_ms == pytest.approx(block.core_end_ms)
        assert resource.lookahead_end_ms == pytest.approx(block.lookahead_end_ms)
        assert resource.dominant_committed_path_count <= resource.raw_committed_path_count
        assert resource.lookahead_call_count == resource.dominant_committed_path_count
        assert resource.lookahead_successor_count >= 0
        assert resource.pre_export_state_count >= resource.exported_state_count
        assert resource.exported_state_count == block.frontier_state_count
        assert resource.block_score_miss_count_before == 0
        assert resource.block_score_miss_count_before <= resource.block_score_miss_count_after
        assert resource.row_score_miss_count_before == previous_row_score_miss_after
        assert resource.row_score_miss_count_before <= resource.row_score_miss_count_after
        assert resource.transition_component_cache_count_before == previous_cache_after
        assert (
            resource.transition_component_cache_count_before
            <= resource.transition_component_cache_count_after
        )
        assert resource.scored_edge_count_before == previous_scored_edge_after
        assert resource.scored_edge_count_before <= resource.scored_edge_count_after
        assert resource.scored_edge_count_after <= len(actual_edges)
        assert resource.transition_component_cache_count_after <= len(cache_entries)
        assert resource.dominance_pruned_state_count == block.dominance_pruned_state_count
        assert resource.width_pruned_state_count == block.width_pruned_state_count
        assert resource.exported_frontier_width_cap == config.exported_frontier_width
        assert resource.local_beam_width_cap == config.local_beam_width
        assert (
            resource.max_boundary_candidates_per_block_cap
            == config.max_boundary_candidates_per_block
        )
        assert (
            resource.max_tempo_candidates_per_block_cap
            == config.max_tempo_candidates_per_block
        )
        assert resource.max_blocks_cap == config.max_blocks
        assert resource.max_sections_cap == config.max_sections
        assert (
            resource.max_section_score_misses_per_block_cap
            == config.max_section_score_misses_per_block
        )
        assert (
            resource.max_section_score_misses_per_audio_cap
            == config.max_section_score_misses_per_audio
        )
        _assert_class_coverage_record(class_record, resource)
        previous_row_score_miss_after = resource.row_score_miss_count_after
        previous_cache_after = resource.transition_component_cache_count_after
        previous_scored_edge_after = resource.scored_edge_count_after
    if resource_records:
        assert resource_records[-1].scored_edge_count_after == len(actual_edges)
        assert (
            resource_records[-1].transition_component_cache_count_after
            == len(cache_entries)
        )


def _assert_class_coverage_record(
    class_record: Any,
    resource: Any,
) -> None:
    assert class_record.block_index == resource.block_index
    assert class_record.cut_time_ms == pytest.approx(resource.core_end_ms)
    assert class_record.input_state_count == resource.pre_export_state_count
    assert class_record.final_state_count == resource.exported_state_count
    assert class_record.post_future_equivalence_state_count <= class_record.input_state_count
    assert class_record.reserved_state_count <= class_record.post_future_equivalence_state_count
    assert class_record.final_state_count <= max(
        class_record.post_future_equivalence_state_count,
        resource.exported_frontier_width_cap,
    )
    for key_field, count_field in (
        ("input_unique_class_keys", "input_state_count"),
        ("post_future_equivalence_unique_class_keys", "post_future_equivalence_state_count"),
        ("reserved_unique_class_keys", "reserved_state_count"),
        ("final_unique_class_keys", "final_state_count"),
    ):
        keys = tuple(getattr(class_record, key_field))
        assert keys == tuple(sorted(keys, key=_class_key_sort_key))
        assert len(set(keys)) == len(keys)
        assert len(keys) <= getattr(class_record, count_field)


def _class_key_sort_key(value: tuple[float, int | None]) -> tuple[float, int]:
    return (float(value[0]), -1 if value[1] is None else int(value[1]))


def _assert_local_frontier_diagnostics_guard(
    result: LocalFrontierResult,
    fixture: Fixture,
    config: LocalFrontierConfig,
) -> None:
    _assert_config_is_frozen(config)
    grid = _assert_ok(result)
    diagnostics = result.diagnostics
    candidate_diagnostics = fixture.candidate_set.diagnostics

    assert result.reason is None
    assert diagnostics.fallback_reason is None
    assert diagnostics.fallback_stage is None
    assert diagnostics.schedule_arm == config.schedule_arm.value
    assert diagnostics.coverage_start_ms == pytest.approx(0.0)
    assert diagnostics.coverage_end_ms == pytest.approx(DURATION_MS)
    assert diagnostics.frame_count == fixture.prediction.frame_count
    assert diagnostics.frame_rate_hz == pytest.approx(FRAME_RATE_HZ)
    assert diagnostics.input_signal_sha256 == candidate_diagnostics.input_signal_sha256
    assert diagnostics.candidate_fingerprint == candidate_diagnostics.candidate_fingerprint
    assert diagnostics.origin_candidate_count == len(fixture.candidate_set.origin_candidates)
    assert diagnostics.boundary_candidate_count == len(fixture.candidate_set.boundary_candidates)
    assert diagnostics.tempo_candidate_count == len(fixture.candidate_set.tempo_candidates)
    assert diagnostics.origin_candidate_count <= GLOBAL_CONSTANT_JUMP_CONSTANTS.max_origin_candidates
    assert (
        diagnostics.boundary_candidate_count
        <= GLOBAL_CONSTANT_JUMP_CONSTANTS.max_interior_boundary_candidates
    )
    assert (
        diagnostics.tempo_candidate_count
        <= GLOBAL_CONSTANT_JUMP_CONSTANTS.max_tempo_candidates_retained
    )
    assert diagnostics.bootstrap_state_count == len(diagnostics.bootstrap_replay_keys)
    assert set(diagnostics.bootstrap_downbeat_phases) <= {None, 0, 1, 2, 3}
    assert diagnostics.selected_section_count == len(grid.sections)
    assert diagnostics.final_global_rescore_count == 0
    assert diagnostics.block_count == len(diagnostics.block_diagnostics)
    assert diagnostics.block_count == len(diagnostics.frontier_widths)
    assert diagnostics.block_count > 0
    assert diagnostics.local_bucket_width_max <= config.local_beam_width

    previous_core_end_ms = diagnostics.coverage_start_ms
    max_block_bucket_width = 0
    for block_index, block in enumerate(diagnostics.block_diagnostics):
        assert block.block_index == block_index
        assert block.cut_time_ms == pytest.approx(block.core_start_ms)
        assert block.core_start_ms == pytest.approx(previous_core_end_ms)
        assert block.core_start_ms < block.core_end_ms
        assert block.core_end_ms <= block.lookahead_end_ms
        assert block.lookahead_end_ms <= diagnostics.coverage_end_ms
        assert block.frontier_state_count == diagnostics.frontier_widths[block_index]
        assert 0 < block.frontier_state_count <= config.exported_frontier_width
        assert 0 <= block.boundary_candidate_count <= config.max_boundary_candidates_per_block
        assert 0 < block.tempo_candidate_count <= config.max_tempo_candidates_per_block
        assert 0 <= block.bucket_width_max <= config.local_beam_width
        assert block.frontier_state_count <= max(block.bucket_width_max, config.exported_frontier_width)
        assert block.dominance_pruned_state_count >= 0
        assert block.width_pruned_state_count >= 0
        if block.width_pruned_state_count > 0:
            assert block.frontier_state_count == config.exported_frontier_width
        _assert_sha256(block.lookahead_trace_fingerprint)
        _assert_sha256(block.exported_frontier_fingerprint)
        max_block_bucket_width = max(max_block_bucket_width, block.bucket_width_max)
        previous_core_end_ms = block.core_end_ms
    assert previous_core_end_ms == pytest.approx(diagnostics.coverage_end_ms)
    assert diagnostics.local_bucket_width_max >= max_block_bucket_width

    for record in diagnostics.boundary_ownership_records:
        assert 0 <= record.boundary_anchor_id < len(fixture.candidate_set.boundary_candidates)
        assert record.retained_frontier_charge_path_count >= 0
        assert record.selected_traceback_objective_charge_count in (0, 1)
        assert record.objective_charge_count == record.selected_traceback_objective_charge_count
        assert record.owner_block_index >= -1
        assert set(record.provisional_block_indexes) <= set(range(diagnostics.block_count))
    for record in diagnostics.lookahead_recompute_records:
        assert 0 <= record.boundary_anchor_id < len(fixture.candidate_set.boundary_candidates)
        assert 0 <= record.previous_block_index < diagnostics.block_count
        assert 0 <= record.next_block_index < diagnostics.block_count
        assert record.previous_block_index < record.next_block_index
        assert record.retained_frontier_charge_path_count >= 0
        assert record.selected_traceback_objective_charge_count in (0, 1)
        assert record.objective_charge_count == record.selected_traceback_objective_charge_count
        _assert_sha256(record.provisional_trace_fingerprint)
        _assert_sha256(record.recomputed_trace_fingerprint)


def _assert_config_is_frozen(config: LocalFrontierConfig) -> None:
    assert config.exported_frontier_width == 16
    assert config.local_beam_width == 64
    assert config.max_boundary_candidates_per_block == 32
    assert config.max_tempo_candidates_per_block == 64
    assert config.max_blocks == 192
    assert config.max_sections == 20
    assert config.max_section_score_misses_per_block == 30_000
    assert config.max_section_score_misses_per_audio == 500_000


def _assert_transition_ledgers_reconstruct(result: LocalFrontierResult) -> None:
    objective = result.objective_diagnostics
    assert objective is not None
    for ledger in objective.terminal_path_ledgers:
        for entry in ledger.transition_entries:
            _assert_component_matches_manual_formula(entry.components)
            _assert_public_component_helper_matches(entry.components)
        replay = replay_terminal_objective(
            ledger,
            sparsity_floor=objective.sparsity_floor,
        )
        _assert_objective_close(
            replay.transition_objective,
            ledger.recorded_transition_objective,
        )
        _assert_objective_close(
            replay.terminal_objective,
            ledger.recorded_terminal_objective,
        )
        _assert_objective_close(
            ledger.reconstructed_terminal_objective,
            ledger.recorded_terminal_objective,
        )


def _assert_public_component_helper_matches(
    component: BoundaryPairTransitionComponents,
) -> None:
    variant = LocalFrontierObjectiveVariant(component.objective_variant)
    rebuilt = boundary_pair_transition_components(
        boundary_anchor_id=component.boundary_anchor_id,
        source_candidate_time_ms=component.source_candidate_time_ms,
        left_period_ms=component.left_period_ms,
        right_period_ms=component.right_period_ms,
        left_bpm=component.left_bpm,
        right_bpm=component.right_bpm,
        boundary_support_cost=component.boundary_support_cost,
        transition_normalizer=component.transition_normalizer,
        objective_variant=variant,
    )
    assert asdict(rebuilt) == asdict(component)


def _require_exp006_objective_v2(result: LocalFrontierResult) -> Any:
    objective = result.objective_diagnostics
    assert objective is not None
    present_fields = set(asdict(objective))
    assert OBJECTIVE_V2_MEASUREMENT_FIELD_NAMES <= present_fields
    assert (
        present_fields
        - OBJECTIVE_NON_BEHAVIOR_FIELD_NAMES
        - OBJECTIVE_V2_MEASUREMENT_FIELD_NAMES
    ) == OBJECTIVE_V1_BEHAVIOR_FIELD_NAMES
    return objective


def _assert_exp005_objective_stays_legacy_v1(result: LocalFrontierResult) -> None:
    objective = result.objective_diagnostics
    assert objective is not None
    fields = set(asdict(objective))
    assert not (fields & OBJECTIVE_V2_MEASUREMENT_FIELD_NAMES)
    assert (
        fields - OBJECTIVE_NON_BEHAVIOR_FIELD_NAMES
    ) == OBJECTIVE_V1_BEHAVIOR_FIELD_NAMES
    assert objective.contract_version.endswith("-v1")
    assert objective.objective_variant == LocalFrontierObjectiveVariant.EXP005_CONSTANT_CHANGE.value
    assert objective.sparsity_floor is None


def _component_cache_identity(
    component: BoundaryPairTransitionComponents,
) -> tuple[Any, ...]:
    return (
        component.objective_variant,
        component.boundary_anchor_id,
        float.hex(component.source_candidate_time_ms),
        float.hex(component.left_period_ms),
        float.hex(component.right_period_ms),
        float.hex(component.left_bpm),
        float.hex(component.right_bpm),
    )


def _assert_occurrence_membership_flags_match_public_ledgers(
    objective: Any,
    actual_edges: tuple[Any, ...],
) -> None:
    edge_by_order = {edge.edge_order: edge for edge in actual_edges}
    assert len(edge_by_order) == len(actual_edges)
    assert set(edge_by_order) == set(range(len(actual_edges)))

    terminal_records = tuple(objective.terminal_path_occurrence_records)
    provisional_records = tuple(objective.provisional_path_occurrence_records)
    assert len(terminal_records) == len(objective.terminal_path_ledgers)
    assert len(provisional_records) == len(objective.provisional_path_ledgers)

    terminal_occurrence_orders: set[int] = set()
    selected_occurrence_orders: set[int] = set()
    for index, (record, ledger) in enumerate(
        zip(terminal_records, objective.terminal_path_ledgers, strict=True)
    ):
        assert record.selection_rank == index
        assert record.selection_rank == ledger.selection_rank
        assert record.replay_key == ledger.replay_key
        assert record.selected == ledger.selected
        assert len(set(record.edge_orders)) == len(record.edge_orders)
        assert set(record.edge_orders) <= set(edge_by_order)
        assert tuple(
            _scored_edge_ledger_payload(edge_by_order[edge_order])
            for edge_order in record.edge_orders
        ) == tuple(_ledger_entry_payload(entry) for entry in ledger.transition_entries)
        terminal_occurrence_orders.update(record.edge_orders)
        if record.selected:
            selected_occurrence_orders.update(record.edge_orders)

    selected_ledgers = tuple(
        ledger for ledger in objective.terminal_path_ledgers if ledger.selected
    )
    assert len(selected_ledgers) == 1
    selected_ledger = selected_ledgers[0]
    selected_records = tuple(record for record in terminal_records if record.selected)
    assert len(selected_records) == 1
    assert tuple(selected_records[0].edge_orders) == tuple(
        objective.selected_traceback_edge_orders
    )

    if selected_ledger.transition_entries:
        assert selected_occurrence_orders
    else:
        assert selected_occurrence_orders == set()
    assert selected_occurrence_orders <= terminal_occurrence_orders
    assert len(selected_occurrence_orders) == len(selected_ledger.transition_entries)

    provisional_occurrence_orders: set[int] = set()
    for index, (record, ledger) in enumerate(
        zip(provisional_records, objective.provisional_path_ledgers, strict=True)
    ):
        assert record.record_index == index
        assert record.block_index == ledger.block_index
        assert record.committed_replay_key == ledger.committed_replay_key
        assert record.ranked_lookahead_replay_key == ledger.ranked_lookahead_replay_key
        assert len(set(record.edge_orders)) == len(record.edge_orders)
        assert set(record.edge_orders) <= set(edge_by_order)
        assert tuple(
            _scored_edge_ledger_payload(edge_by_order[edge_order])
            for edge_order in record.edge_orders
        ) == tuple(_ledger_entry_payload(entry) for entry in ledger.transition_entries)
        provisional_occurrence_orders.update(record.edge_orders)

    for edge in actual_edges:
        assert edge.retained_terminal_path == (
            edge.edge_order in terminal_occurrence_orders
        )
        assert edge.retained_provisional_path == (
            edge.edge_order in provisional_occurrence_orders
        )
        assert edge.selected_traceback_path == (
            edge.edge_order in selected_occurrence_orders
        )

    # Negative regression guard: duplicate logical transitions must not be
    # selected solely by replay-key identity.  The source now marks concrete
    # edge_order occurrences; identity is only used here to prove the verifier
    # would catch the old over-selection failure.
    selected_identity_set = {
        _ledger_entry_identity(entry) for entry in selected_ledger.transition_entries
    }
    identity_selected_orders = {
        edge.edge_order
        for edge in actual_edges
        if _scored_edge_identity(edge) in selected_identity_set
    }
    assert selected_occurrence_orders <= identity_selected_orders
    if len(identity_selected_orders) != len(selected_occurrence_orders):
        assert identity_selected_orders != selected_occurrence_orders


def _ledger_entry_identity(entry: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (entry.predecessor_replay_key, entry.successor_replay_key)


def _scored_edge_identity(edge: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (edge.predecessor_replay_key, edge.successor_replay_key)


def _ledger_entry_payload(entry: Any) -> str:
    return _stable_json_sort_key(
        {
            "predecessor_replay_key": entry.predecessor_replay_key,
            "successor_replay_key": entry.successor_replay_key,
            "boundary_beat": entry.boundary_beat,
            "boundary_time_ms": entry.boundary_time_ms,
            "components": asdict(entry.components),
        }
    )


def _scored_edge_ledger_payload(edge: Any) -> str:
    return _stable_json_sort_key(
        {
            "predecessor_replay_key": edge.predecessor_replay_key,
            "successor_replay_key": edge.successor_replay_key,
            "boundary_beat": edge.boundary_beat,
            "boundary_time_ms": edge.boundary_time_ms,
            "components": asdict(edge.components),
        }
    )


def _scored_edge_access(edge: Any) -> tuple[Any, ...]:
    return (
        edge.objective_variant,
        edge.boundary_anchor_id,
        float.hex(edge.source_candidate_time_ms),
        float.hex(edge.left_period_ms),
        float.hex(edge.right_period_ms),
        float.hex(edge.left_bpm),
        float.hex(edge.right_bpm),
    )


@lru_cache(maxsize=None)
def _matrix_oracle_rows() -> tuple[dict[str, Any], ...]:
    rows = []
    for case in MATRIX_CASES:
        fixture = _fixture_for_case(case)
        for schedule_arm in SCHEDULE_ARMS:
            config = LocalFrontierConfig(schedule_arm=schedule_arm)
            default = fit_local_frontier_constant_jump(
                fixture.prediction,
                config=config,
                candidate_set=fixture.candidate_set,
            )
            exp006 = fit_local_frontier_boundary_pair_transition(
                fixture.prediction,
                config=config,
                candidate_set=fixture.candidate_set,
            )
            explicit_exp005 = fit_local_frontier_objective_variant(
                fixture.prediction,
                objective_variant=LocalFrontierObjectiveVariant.EXP005_CONSTANT_CHANGE,
                config=config,
                candidate_set=fixture.candidate_set,
            )
            _assert_expected_grid(exp006, case.expected_sections)
            _assert_local_frontier_diagnostics_guard(default, fixture, config)
            _assert_local_frontier_diagnostics_guard(exp006, fixture, config)
            _assert_local_frontier_diagnostics_guard(explicit_exp005, fixture, config)
            _assert_exp005_objective_stays_legacy_v1(explicit_exp005)
            objective = _require_exp006_objective_v2(exp006)
            cache_components = sorted(
                (asdict(record.components) for record in objective.transition_component_cache_entries),
                key=_stable_json_sort_key,
            )
            accesses = tuple(
                _scored_edge_access(edge) for edge in objective.actual_scored_edges
            )
            unique_accesses = tuple(sorted(set(accesses)))
            rows.append(
                {
                    "case": case.case_id,
                    "arm": schedule_arm.value,
                    "default_sha": _stable_json_sha256(_legacy_result_payload(default)),
                    "exp006_behavior_sha": _stable_json_sha256(
                        _exp006_behavior_projection(exp006)
                    ),
                    "explicit_exp005_sha": _stable_json_sha256(
                        _exp005_behavior_projection(explicit_exp005)
                    ),
                    "cache_sha": _stable_json_sha256(cache_components),
                    "cache_count": len(cache_components),
                    "access_count": len(accesses),
                    "access_sha": _stable_json_sha256(accesses),
                    "access_unique_count": len(unique_accesses),
                    "access_unique_sha": _stable_json_sha256(unique_accesses),
                }
            )
    return tuple(rows)


def _legacy_result_payload(result: LocalFrontierResult) -> dict[str, Any]:
    return {
        "reason": result.reason,
        "grid": None if result.grid is None else result.grid.to_dict(),
        "base": asdict(result.diagnostics),
        "objective": (
            None
            if result.objective_diagnostics is None
            else asdict(result.objective_diagnostics)
        ),
    }


def _exp006_behavior_projection(result: LocalFrontierResult) -> dict[str, Any]:
    objective = _require_exp006_objective_v2(result)
    objective_payload = asdict(objective)
    legacy_fields = {
        key: value
        for key, value in objective_payload.items()
        if key in OBJECTIVE_V1_BEHAVIOR_FIELD_NAMES
    }
    assert set(legacy_fields) == OBJECTIVE_V1_BEHAVIOR_FIELD_NAMES
    return {
        "reason": result.reason,
        "grid": None if result.grid is None else result.grid.to_dict(),
        "base": asdict(result.diagnostics),
        "objective_legacy_fields": legacy_fields,
    }


def _exp005_behavior_projection(result: LocalFrontierResult) -> dict[str, Any]:
    objective = result.objective_diagnostics
    assert objective is not None
    objective_payload = asdict(objective)
    assert not (set(objective_payload) & OBJECTIVE_V2_MEASUREMENT_FIELD_NAMES)
    legacy_fields = {
        key: value
        for key, value in objective_payload.items()
        if key in OBJECTIVE_V1_BEHAVIOR_FIELD_NAMES
    }
    assert set(legacy_fields) == OBJECTIVE_V1_BEHAVIOR_FIELD_NAMES
    return {
        "reason": result.reason,
        "grid": None if result.grid is None else result.grid.to_dict(),
        "base": asdict(result.diagnostics),
        "objective_legacy_fields": legacy_fields,
    }


def _stable_json_sort_key(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _assert_component_matches_manual_formula(
    component: BoundaryPairTransitionComponents,
) -> None:
    qhat_left = 60000.0 / component.left_period_ms
    qhat_right = 60000.0 / component.right_period_ms
    left_min, left_distance = _manual_alias_distance_parts(
        component.left_bpm,
        qhat_left,
    )
    right_min, right_distance = _manual_alias_distance_parts(
        component.right_bpm,
        qhat_right,
    )
    pair_cost = min(1.0, left_distance + right_distance)
    if component.sparsity_floor is None:
        change_cost = 1.0
    else:
        change_cost = component.sparsity_floor + (
            1.0 - component.sparsity_floor
        ) * pair_cost
    alias_switch = _manual_alias_switch_cost(component.left_bpm, component.right_bpm)
    jump_size = min(1.0, abs(math.log2(component.right_bpm / component.left_bpm)))
    if component.sparsity_floor is None:
        raw_transition = (
            0.18
            + 0.12 * alias_switch
            + 0.10 * jump_size
            + 0.15 * component.boundary_support_cost
        )
    else:
        raw_transition = (
            0.18 * change_cost
            + 0.12 * alias_switch
            + 0.10 * jump_size
            + 0.15 * component.boundary_support_cost
        )

    assert component.qhat_left_bpm == pytest.approx(qhat_left)
    assert component.qhat_right_bpm == pytest.approx(qhat_right)
    assert component.alias_min_left_unclipped == pytest.approx(left_min)
    assert component.alias_min_right_unclipped == pytest.approx(right_min)
    assert component.alias_distance_left == pytest.approx(left_distance)
    assert component.alias_distance_right == pytest.approx(right_distance)
    assert component.pair_cost == pytest.approx(pair_cost)
    assert component.change_cost == pytest.approx(change_cost)
    assert component.alias_switch_cost == pytest.approx(alias_switch)
    assert component.jump_size_cost == pytest.approx(jump_size)
    assert component.raw_transition_cost == pytest.approx(raw_transition)
    assert component.normalized_increment == pytest.approx(
        raw_transition / component.transition_normalizer
    )


def _manual_alias_distance_parts(q_bpm: float, qhat_bpm: float) -> tuple[float, float]:
    distances = tuple(
        abs(math.log2(q_bpm / (multiplier * qhat_bpm)))
        for multiplier in BOUNDARY_PAIR_ALIAS_MULTIPLIERS
    )
    minimum = min(distances)
    return float(minimum), float(min(1.0, minimum))


def _manual_alias_switch_cost(left_bpm: float, right_bpm: float) -> float:
    if _relative_close(left_bpm, right_bpm, tolerance=0.005):
        return 0.0
    if any(
        _relative_close(left_bpm * multiplier, right_bpm, tolerance=0.005)
        for multiplier in GLOBAL_CONSTANT_JUMP_CONSTANTS.alias_multipliers
    ):
        return 0.5
    return 1.0


def _relative_close(left: float, right: float, *, tolerance: float) -> bool:
    return abs(left - right) / max(abs(right), 1e-12) <= tolerance


def _same_float(left: float, right: float, *, abs_tol: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=abs_tol)


def _assert_perturbed_single_jump_margin(result: LocalFrontierResult) -> None:
    objective = result.objective_diagnostics
    assert objective is not None
    selected = _selected_terminal_ledger(result)
    assert selected.bpm_sequence == (120.0, 150.0)
    persistence = tuple(
        ledger
        for ledger in objective.terminal_path_ledgers
        if ledger.bpm_sequence == (120.0,)
    )
    assert persistence
    best_persistence = min(
        persistence,
        key=lambda ledger: ledger.recorded_terminal_objective,
    )
    same_boundary_wrong_right = tuple(
        ledger
        for ledger in objective.terminal_path_ledgers
        if len(ledger.bpm_sequence) == 2
        and ledger.bpm_sequence[0] == 120.0
        and float.hex(ledger.bpm_sequence[1]) != float.hex(150.0)
        and len(ledger.transition_entries) == 1
        and ledger.transition_entries[0].components.boundary_anchor_id == 0
    )
    assert same_boundary_wrong_right
    best_wrong_right = min(
        same_boundary_wrong_right,
        key=lambda ledger: ledger.recorded_terminal_objective,
    )
    assert selected.recorded_terminal_objective < best_persistence.recorded_terminal_objective
    assert selected.recorded_terminal_objective < best_wrong_right.recorded_terminal_objective


def _assert_dense_anchor_floor_coverage(
    result: LocalFrontierResult,
    fixture: Fixture,
    schedule_arm: LocalFrontierScheduleArm,
) -> None:
    objective = _require_exp006_objective_v2(result)
    entries = tuple(objective.actual_scored_edges)
    assert entries
    expected_count, expected_unique_count, expected_anchors, expected_zero_floor = (
        EXPECTED_DENSE_PUBLIC_TRACE_BY_CASE_AND_ARM[fixture.case.case_id][schedule_arm]
    )
    accesses = tuple(_scored_edge_access(entry) for entry in entries)
    assert len(accesses) == expected_count
    assert len(set(accesses)) == expected_unique_count
    assert {entry.boundary_anchor_id for entry in entries} == expected_anchors

    for entry in entries:
        component = entry.components
        assert component.sparsity_floor == pytest.approx(EXP006_PAIR_SPARSITY_FLOOR)
        assert component.change_cost >= EXP006_PAIR_SPARSITY_FLOOR
        assert 0.18 * component.change_cost >= 0.045
    actual_zero_floor = {
        entry.boundary_anchor_id
        for entry in entries
        if _same_float(entry.components.pair_cost, 0.0)
        and _same_float(entry.components.change_cost, EXP006_PAIR_SPARSITY_FLOOR)
        and _same_float(0.18 * entry.components.change_cost, 0.045)
    }
    assert actual_zero_floor == expected_zero_floor


def _all_reported_transition_entries(result: LocalFrontierResult) -> tuple[Any, ...]:
    objective = _require_exp006_objective_v2(result)
    return tuple(objective.actual_scored_edges)


def _selected_terminal_ledger(result: LocalFrontierResult) -> TerminalObjectiveLedger:
    objective = result.objective_diagnostics
    assert objective is not None
    selected = tuple(ledger for ledger in objective.terminal_path_ledgers if ledger.selected)
    assert len(selected) == 1
    return selected[0]


def _find_terminal_ledger(
    result: LocalFrontierResult,
    expected_sections: tuple[tuple[int, int, float], ...],
) -> TerminalObjectiveLedger:
    expected = tuple((start, end, float.hex(bpm)) for start, end, bpm in expected_sections)
    objective = result.objective_diagnostics
    assert objective is not None
    for ledger in objective.terminal_path_ledgers:
        if ledger.section_signature == expected:
            return ledger
    raise AssertionError(f"terminal ledger not retained for section signature {expected!r}")


def _assert_two_run_determinism(
    first: LocalFrontierResult,
    second: LocalFrontierResult,
) -> None:
    assert first.grid is not None
    assert second.grid is not None
    assert first.grid.to_dict() == second.grid.to_dict()
    assert first.diagnostics.replay_fingerprint == second.diagnostics.replay_fingerprint
    assert first.diagnostics.grid_fingerprint == second.diagnostics.grid_fingerprint
    assert first.diagnostics.deterministic_projection_sha256 == (
        second.diagnostics.deterministic_projection_sha256
    )
    assert first.objective_diagnostics is not None
    assert second.objective_diagnostics is not None
    assert asdict(first.objective_diagnostics) == asdict(second.objective_diagnostics)
    _assert_result_json_finite(first)
    _assert_result_json_finite(second)


def _assert_result_json_finite(result: LocalFrontierResult) -> None:
    payload = {
        "reason": result.reason,
        "diagnostics": _jsonable(result.diagnostics),
        "objective_diagnostics": _jsonable(result.objective_diagnostics),
        "grid": None if result.grid is None else result.grid.to_dict(),
    }
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_jsonable"):
        return value.to_jsonable()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _assert_objective_close(actual: float, expected: float) -> None:
    assert actual == pytest.approx(expected, rel=0.0, abs=_objective_abs_tol(expected))


def _objective_abs_tol(value: float) -> float:
    return max(1e-12, 8.0 * math.ulp(float(value)))


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    assert value == value.lower()
    assert all(character in "0123456789abcdef" for character in value)


def _matrix_definition_fingerprint() -> str:
    payload = {
        "cases": tuple(_case_payload(case) for case in MATRIX_CASES),
        "schedules": tuple(schedule.value for schedule in SCHEDULE_ARMS),
        "arm_count": EXPECTED_MATRIX_ARM_COUNT,
    }
    return _stable_json_sha256(payload)


def _case_payload(case: MatrixCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "segments": tuple(asdict(segment) for segment in case.segments),
        "tempo_bpms": case.tempo_bpms,
        "boundary_specs": tuple(asdict(boundary) for boundary in case.boundary_specs),
        "expected_sections": case.expected_sections,
        "baseline_expected_sections": case.baseline_expected_sections,
        "expected_alias_switch_count": case.expected_alias_switch_count,
        "noise_seed": case.noise_seed,
        "noise_std": case.noise_std,
        "require_false_transition_floor": case.require_false_transition_floor,
    }


def _stable_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
