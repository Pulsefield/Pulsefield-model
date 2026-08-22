from __future__ import annotations

import math

import numpy as np
import pytest

from pulsefield_model.evals.mir_anchor_data import (
    build_anchor_episodes,
    build_candidate_chart_features,
    build_episode_histories,
    build_same_gap_choice_sets,
    collapse_hit_objects_to_rows,
    triangular_log_support_scores,
    triangular_support_choice_nll,
    triangular_support_weights,
)
from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind


def test_collapse_hit_objects_uses_exact_onsets_and_preserves_chords() -> None:
    hold = ManiaHitObject(1000.0, 1200.0, 2, ManiaHitObjectKind.HOLD)
    tap = ManiaHitObject(1000.0, 1000.0, 0, ManiaHitObjectKind.TAP)
    one_ms_later = ManiaHitObject(1001.0, 1001.0, 1, ManiaHitObjectKind.TAP)

    rows = collapse_hit_objects_to_rows((one_ms_later, hold, tap))

    assert tuple(row.time_ms for row in rows) == (1000, 1001)
    assert rows[0].hitobjects == (tap, hold)
    assert rows[1].hitobjects == (one_ms_later,)


def test_collapse_hit_objects_rejects_fractional_millisecond_onsets() -> None:
    hitobject = ManiaHitObject(1000.5, 1000.5, 0, ManiaHitObjectKind.TAP)

    with pytest.raises(ValueError, match="integer-millisecond"):
        collapse_hit_objects_to_rows((hitobject,))


def test_collapse_rejects_overlapping_same_lane_objects_but_allows_release_rehit() -> None:
    hold = ManiaHitObject(100.0, 500.0, 1, ManiaHitObjectKind.HOLD)

    with pytest.raises(ValueError, match="before its previous hold closes"):
        collapse_hit_objects_to_rows((hold, ManiaHitObject(300.0, 300.0, 1, ManiaHitObjectKind.TAP)))

    rows = collapse_hit_objects_to_rows(
        (hold, ManiaHitObject(500.0, 500.0, 1, ManiaHitObjectKind.TAP)),
    )
    assert tuple(row.time_ms for row in rows) == (100, 500)


def test_build_anchor_episodes_preserves_one_millisecond_gap() -> None:
    rows = collapse_hit_objects_to_rows(tuple(_tap(time_ms) for time_ms in (1000, 1001, 1006)))

    episodes = build_anchor_episodes(rows, map_id="map-a")

    assert tuple(episode.gap_ms for episode in episodes) == (1, 5)
    assert episodes[0].origin_time_ms == 1000
    assert episodes[0].target_time_ms == 1001


def test_episode_history_includes_current_row_but_no_future_materialization() -> None:
    rows = collapse_hit_objects_to_rows(
        (
            _tap(100),
            ManiaHitObject(200.0, 350.0, 2, ManiaHitObjectKind.HOLD),
            _tap(300),
        ),
    )

    histories, padding = build_episode_histories(rows, history_rows=3, lookback_ms=1_000)

    assert histories.shape == (2, 3, 10)
    assert padding.tolist() == [[False, True, True], [False, False, True]]
    assert histories[0, 0, 2] == 1.0  # origin row lane 0
    assert histories[0, :, 4].sum() == 0.0  # future lane 2 is absent
    assert histories[1, 1, 4] == 1.0
    assert histories[1, 1, 8] == 1.0  # lane-2 hold start
    assert histories[1, 1, 1] == 0.0  # current origin age


def test_candidate_features_expose_open_age_but_not_future_hold_remaining_time() -> None:
    rows = collapse_hit_objects_to_rows(
        (
            ManiaHitObject(100.0, 500.0, 2, ManiaHitObjectKind.HOLD),
            _tap(200),
            _tap(600),
        ),
    )

    features = build_candidate_chart_features(
        rows,
        np.asarray([[0, 0], [1, 1]]),
        np.asarray([[300.0, 500.0], [300.0, 550.0]]),
        song_duration_ms=1_000.0,
    )

    assert features.shape == (2, 2, 6)
    assert features[0, 0, 4] > 1.0  # lane 2 is open and carries elapsed age
    assert features[0, 1, 4] == 0.0  # release at 500 ms is playable
    assert features[1, 0, 4] == features[0, 0, 4]
    assert features[1, 1, 4] == 0.0

    later_release_rows = collapse_hit_objects_to_rows(
        (
            ManiaHitObject(100.0, 900.0, 2, ManiaHitObjectKind.HOLD),
            _tap(200),
            _tap(950),
        ),
    )
    same_query = build_candidate_chart_features(
        later_release_rows,
        np.asarray([[0]]),
        np.asarray([[300.0]]),
        song_duration_ms=1_000.0,
    )
    assert same_query[0, 0, 4] == features[0, 0, 4]


def test_same_gap_controls_do_not_cross_maps() -> None:
    map_a = _episodes_from_gaps((20, 30), map_id="map-a")
    map_b = _episodes_from_gaps((50, 60), map_id="map-b")

    choice_sets = build_same_gap_choice_sets(
        (*map_a, *map_b),
        controls_per_case=1,
        support_half_width_ms=10,
        seed=0,
    )

    assert not any(choice.map_id == "map-a" and choice.case_episode_index == 0 for choice in choice_sets)


def test_same_gap_rejects_incomplete_episode_chain() -> None:
    complete = _episodes_from_gaps((20, 5, 50, 60))

    with pytest.raises(ValueError, match="complete zero-based chain"):
        build_same_gap_choice_sets(
            (complete[0], *complete[2:]),
            controls_per_case=1,
            support_half_width_ms=10,
            seed=0,
        )


def test_same_gap_controls_survive_the_entire_symmetric_support() -> None:
    episodes = _episodes_from_gaps((20, 31, 30, 41))

    choice_sets = build_same_gap_choice_sets(
        episodes,
        controls_per_case=2,
        support_half_width_ms=10,
        seed=7,
    )
    first_case = next(choice for choice in choice_sets if choice.case_episode_index == 0)

    assert first_case.control_episode_indices == (1, 3)
    assert first_case.candidate_episode_indices == (0, 1, 3)
    assert first_case.candidate_center_times_ms == (20, 40, 101)
    assert 2 not in first_case.control_episode_indices  # gap 30 reaches its row at +10 ms


def test_same_gap_sampling_is_fixed_size_without_replacement_and_seeded() -> None:
    episodes = _episodes_from_gaps((20, 31, 32, 33, 34, 35))
    kwargs = {
        "controls_per_case": 3,
        "support_half_width_ms": 10,
        "seed": 123,
    }

    first = build_same_gap_choice_sets(episodes, **kwargs)
    second = build_same_gap_choice_sets(episodes, **kwargs)
    first_case = next(choice for choice in first if choice.case_episode_index == 0)

    assert first == second
    assert len(first_case.control_episode_indices) == 3
    assert len(set(first_case.control_episode_indices)) == 3


def test_same_gap_samples_only_after_candidate_support_filtering() -> None:
    episodes = _episodes_from_gaps((20, 31, 32, 33, 34))
    invalid_times = {40, 71}

    choice_sets = build_same_gap_choice_sets(
        episodes,
        controls_per_case=2,
        support_half_width_ms=10,
        seed=2,
        candidate_time_is_valid=lambda time_ms: time_ms not in invalid_times,
    )

    first_case = next(choice for choice in choice_sets if choice.case_episode_index == 0)
    assert all(time_ms not in invalid_times for time_ms in first_case.candidate_center_times_ms)


def test_same_gap_excludes_cases_beyond_the_preregistered_horizon() -> None:
    episodes = _episodes_from_gaps((2_001, 3_001, 4_001))

    choice_sets = build_same_gap_choice_sets(
        episodes,
        controls_per_case=1,
        max_gap_ms=2_000,
        support_half_width_ms=10,
        seed=4,
    )

    assert choice_sets == ()


def test_same_gap_excludes_case_whose_support_reaches_the_following_row() -> None:
    episodes = _episodes_from_gaps((20, 10, 50, 60))

    choice_sets = build_same_gap_choice_sets(
        episodes,
        controls_per_case=1,
        support_half_width_ms=10,
        seed=0,
    )

    assert 0 not in {choice.case_episode_index for choice in choice_sets}


def test_same_gap_case_origin_boundary_is_strict() -> None:
    episodes = _episodes_from_gaps((10, 11, 30, 40))

    choice_sets = build_same_gap_choice_sets(
        episodes,
        controls_per_case=1,
        support_half_width_ms=10,
        seed=0,
    )
    case_indices = {choice.case_episode_index for choice in choice_sets}

    assert 0 not in case_indices
    assert 1 in case_indices


@pytest.mark.parametrize("controls_per_case", [True, 1.5])
def test_same_gap_requires_positive_integer_control_count(controls_per_case: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_same_gap_choice_sets(
            _episodes_from_gaps((20, 40)),
            controls_per_case=controls_per_case,  # type: ignore[arg-type]
            support_half_width_ms=10,
            seed=0,
        )


def test_triangular_support_weights_are_positive_and_normalized() -> None:
    weights = triangular_support_weights(1)

    np.testing.assert_array_equal(weights, np.array([0.25, 0.5, 0.25]))
    assert float(weights.sum()) == pytest.approx(1.0)
    np.testing.assert_array_equal(triangular_support_weights(0), np.array([1.0]))


def test_triangular_log_support_integrates_mass_before_choice_loss() -> None:
    scores = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    support_scores = triangular_log_support_scores(scores, half_width_ms=1)
    expected_case_score = math.log(0.25 + 0.5 * math.exp(2.0) + 0.25)
    expected_nll = np.logaddexp(expected_case_score, 0.0) - expected_case_score

    np.testing.assert_allclose(support_scores, np.array([expected_case_score, 0.0]), atol=1e-15)
    assert triangular_support_choice_nll(scores, half_width_ms=1) == pytest.approx(expected_nll)


def test_triangular_choice_loss_is_stable_and_invariant_to_common_offset() -> None:
    scores = np.array([[1000.0, 1001.0, 1000.0], [999.0, 999.0, 999.0]])

    loss = triangular_support_choice_nll(scores, half_width_ms=1)
    shifted_loss = triangular_support_choice_nll(scores - 10_000.0, half_width_ms=1)

    case_support = math.log(0.25 * math.exp(1000.0 - 1001.0) + 0.5 + 0.25 * math.exp(1000.0 - 1001.0))
    expected_loss = np.logaddexp(case_support, 999.0 - 1001.0) - case_support

    assert loss == pytest.approx(expected_loss)
    assert shifted_loss == pytest.approx(loss)


def test_triangular_choice_loss_supports_nonzero_case_index() -> None:
    scores = np.array([[0.0], [1.0]])

    assert triangular_support_choice_nll(scores, half_width_ms=0, case_index=1) == pytest.approx(
        math.log1p(math.exp(-1.0))
    )


def _episodes_from_gaps(gaps_ms: tuple[int, ...], *, map_id: str = "map"):
    times = [0]
    for gap_ms in gaps_ms:
        times.append(times[-1] + gap_ms)
    rows = collapse_hit_objects_to_rows(tuple(_tap(time_ms) for time_ms in times))
    return build_anchor_episodes(rows, map_id=map_id)


def _tap(time_ms: int) -> ManiaHitObject:
    return ManiaHitObject(float(time_ms), float(time_ms), 0, ManiaHitObjectKind.TAP)
