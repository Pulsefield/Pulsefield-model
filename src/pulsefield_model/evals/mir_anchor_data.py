from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind


HISTORY_FEATURE_DIM = 10
CANDIDATE_FEATURE_DIM = 6


@dataclass(frozen=True)
class ManiaAnchorRow:
    """Hit-object onsets that share one exact integer-millisecond timestamp."""

    time_ms: int
    hitobjects: tuple[ManiaHitObject, ...]


@dataclass(frozen=True)
class AnchorEpisode:
    """Interval from one materialized row onset to the next row onset."""

    map_id: str
    episode_index: int
    origin_row_index: int
    target_row_index: int
    origin_time_ms: int
    target_time_ms: int
    gap_ms: int


@dataclass(frozen=True)
class SameGapChoiceSet:
    """One case and uniformly sampled episodes surviving the same elapsed gap."""

    map_id: str
    case_episode_index: int
    control_episode_indices: tuple[int, ...]
    elapsed_ms: int
    candidate_center_times_ms: tuple[int, ...]
    support_half_width_ms: int

    @property
    def candidate_episode_indices(self) -> tuple[int, ...]:
        return (self.case_episode_index, *self.control_episode_indices)


def collapse_hit_objects_to_rows(hitobjects: Sequence[ManiaHitObject]) -> tuple[ManiaAnchorRow, ...]:
    """Sort and collapse exact simultaneous onsets without quantizing timestamps."""

    grouped: dict[int, list[ManiaHitObject]] = {}
    for hitobject in hitobjects:
        time_ms = _exact_integer_ms(hitobject.start_time_ms)
        grouped.setdefault(time_ms, []).append(hitobject)

    rows: list[ManiaAnchorRow] = []
    for time_ms in sorted(grouped):
        row_hitobjects = tuple(
            sorted(
                grouped[time_ms],
                key=lambda item: (item.lane, item.end_time_ms, item.kind.value),
            )
        )
        rows.append(ManiaAnchorRow(time_ms=time_ms, hitobjects=row_hitobjects))
    _validate_playable_lane_sequence(rows)
    return tuple(rows)


def build_anchor_episodes(
    rows: Sequence[ManiaAnchorRow],
    *,
    map_id: str,
) -> tuple[AnchorEpisode, ...]:
    """Build consecutive raw-millisecond onset episodes from sorted rows."""

    if not map_id:
        raise ValueError("map_id must be non-empty.")
    for previous, current in zip(rows, rows[1:]):
        if current.time_ms <= previous.time_ms:
            raise ValueError("Anchor rows must have strictly increasing timestamps.")

    return tuple(
        AnchorEpisode(
            map_id=map_id,
            episode_index=index,
            origin_row_index=index,
            target_row_index=index + 1,
            origin_time_ms=origin.time_ms,
            target_time_ms=target.time_ms,
            gap_ms=target.time_ms - origin.time_ms,
        )
        for index, (origin, target) in enumerate(zip(rows, rows[1:]))
    )


def build_episode_histories(
    rows: Sequence[ManiaAnchorRow],
    *,
    episode_indices: ArrayLike | None = None,
    history_rows: int = 32,
    lookback_ms: int = 8_000,
    gap_scale_ms: int = 2_000,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Encode causal row history once for every possible next-row episode.

    Tokens are ordered oldest to newest and contain normalized previous gap,
    age from the current origin, four onset-lane flags, and four hold-start
    flags. The current origin row is included; no hold end or future row is
    observed.
    """

    for name, value in (
        ("history_rows", history_rows),
        ("lookback_ms", lookback_ms),
        ("gap_scale_ms", gap_scale_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for previous, current in zip(rows, rows[1:]):
        if current.time_ms <= previous.time_ms:
            raise ValueError("Anchor rows must have strictly increasing timestamps.")

    episode_count = max(0, len(rows) - 1)
    if episode_indices is None:
        selected_indices = np.arange(episode_count, dtype=np.int64)
    else:
        raw_indices = np.asarray(episode_indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("episode_indices must be a one-dimensional integer array.")
        selected_indices = raw_indices.astype(np.int64, copy=False)
        if np.any(selected_indices < 0) or np.any(selected_indices >= episode_count):
            raise ValueError("episode_indices contains an out-of-range episode.")
    histories = np.zeros((selected_indices.size, history_rows, HISTORY_FEATURE_DIM), dtype=np.float32)
    padding = np.ones((selected_indices.size, history_rows), dtype=np.bool_)
    gap_denominator = math.log1p(gap_scale_ms)

    for output_index, episode_index in enumerate(selected_indices):
        origin_time_ms = rows[episode_index].time_ms
        first = episode_index
        while first > 0 and origin_time_ms - rows[first - 1].time_ms <= lookback_ms:
            first -= 1
        selected = range(max(first, episode_index - history_rows + 1), episode_index + 1)
        tokens: list[NDArray[np.float32]] = []
        for row_index in selected:
            row = rows[row_index]
            previous_gap_ms = 0 if row_index == 0 else row.time_ms - rows[row_index - 1].time_ms
            token = np.zeros(HISTORY_FEATURE_DIM, dtype=np.float32)
            token[0] = math.log1p(previous_gap_ms) / gap_denominator
            token[1] = (origin_time_ms - row.time_ms) / lookback_ms
            for hitobject in row.hitobjects:
                if not 0 <= hitobject.lane < 4:
                    raise ValueError(f"Expected a 4K lane in [0,3], got {hitobject.lane}.")
                token[2 + hitobject.lane] = 1.0
                if hitobject.kind is ManiaHitObjectKind.HOLD:
                    token[6 + hitobject.lane] = 1.0
            tokens.append(token)
        token_count = len(tokens)
        histories[output_index, :token_count] = np.asarray(tokens)
        padding[output_index, :token_count] = False
    return histories, padding


def build_candidate_chart_features(
    rows: Sequence[ManiaAnchorRow],
    episode_indices: ArrayLike,
    query_times_ms: ArrayLike,
    *,
    song_duration_ms: float,
    time_scale_ms: int = 2_000,
) -> NDArray[np.float32]:
    """Return causal query covariates, including occupancy of the four lanes.

    The six channels are log-scaled elapsed time, relative song position, and
    four causal lane-state values.  A lane is zero after its hold-close event;
    while open it is one plus the normalized log age of the hold.  This makes
    occupancy and age observable without exposing a future close timestamp.
    """

    if not math.isfinite(song_duration_ms) or song_duration_ms <= 0.0:
        raise ValueError("song_duration_ms must be positive and finite.")
    if isinstance(time_scale_ms, bool) or not isinstance(time_scale_ms, int) or time_scale_ms <= 0:
        raise ValueError("time_scale_ms must be a positive integer.")
    raw_episode_indices = np.asarray(episode_indices)
    if not np.issubdtype(raw_episode_indices.dtype, np.integer):
        raise ValueError("episode_indices must contain integers.")
    query_times = np.asarray(query_times_ms, dtype=np.float64)
    indexes, queries = np.broadcast_arrays(raw_episode_indices.astype(np.int64), query_times)
    if not np.all(np.isfinite(queries)):
        raise ValueError("query_times_ms must be finite.")
    episode_count = max(0, len(rows) - 1)
    if np.any(indexes < 0) or np.any(indexes >= episode_count):
        raise ValueError("episode_indices contains an out-of-range episode.")

    hold_starts = np.zeros((episode_count, 4), dtype=np.float64)
    hold_ends = np.zeros((episode_count, 4), dtype=np.float64)
    running_hold_starts = np.zeros(4, dtype=np.float64)
    running_hold_ends = np.zeros(4, dtype=np.float64)
    for row_index in range(episode_count):
        row = rows[row_index]
        for hitobject in row.hitobjects:
            if not 0 <= hitobject.lane < 4:
                raise ValueError(f"Expected a 4K lane in [0,3], got {hitobject.lane}.")
            if hitobject.kind is ManiaHitObjectKind.HOLD:
                end_time_ms = _exact_integer_ms(hitobject.end_time_ms)
                if end_time_ms < row.time_ms:
                    raise ValueError("Hold end must not precede its onset.")
                running_hold_starts[hitobject.lane] = row.time_ms
                running_hold_ends[hitobject.lane] = end_time_ms
        hold_starts[row_index] = running_hold_starts
        hold_ends[row_index] = running_hold_ends

    origins = np.asarray([rows[int(index)].time_ms for index in indexes.flat]).reshape(indexes.shape)
    elapsed = queries - origins
    if np.any(elapsed < 0.0):
        raise ValueError("Every query must be at or after its episode origin.")
    denominator = math.log1p(time_scale_ms)
    open_lanes = hold_ends[indexes] > queries[..., None]
    open_age = np.maximum(queries[..., None] - hold_starts[indexes], 0.0)
    causal_lane_state = np.where(
        open_lanes,
        1.0 + np.log1p(open_age) / denominator,
        0.0,
    )
    features = np.concatenate(
        (
            (np.log1p(elapsed) / denominator)[..., None],
            (queries / song_duration_ms)[..., None],
            causal_lane_state,
        ),
        axis=-1,
    ).astype(np.float32)
    if features.shape[-1] != CANDIDATE_FEATURE_DIM:
        raise AssertionError("candidate feature contract changed unexpectedly")
    return features


def build_same_gap_choice_sets(
    episodes: Sequence[AnchorEpisode],
    *,
    controls_per_case: int,
    max_gap_ms: int = 2_000,
    support_half_width_ms: int,
    seed: int,
    candidate_time_is_valid: Callable[[int], bool] | None = None,
) -> tuple[SameGapChoiceSet, ...]:
    """Sample fixed-size within-map controls whose next row is beyond the support."""

    if isinstance(controls_per_case, bool) or not isinstance(controls_per_case, int) or controls_per_case <= 0:
        raise ValueError("controls_per_case must be a positive integer.")
    if isinstance(max_gap_ms, bool) or not isinstance(max_gap_ms, int) or max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be a positive integer.")
    _validate_half_width(support_half_width_ms)

    rng = np.random.default_rng(seed)
    choice_sets: list[SameGapChoiceSet] = []
    episodes_by_map: dict[str, list[AnchorEpisode]] = {}
    for episode in episodes:
        episodes_by_map.setdefault(episode.map_id, []).append(episode)

    for map_id in sorted(episodes_by_map):
        map_episodes = tuple(sorted(episodes_by_map[map_id], key=_episode_index))
        _validate_episode_chain(map_id, map_episodes)
        next_episode_by_origin = {episode.origin_row_index: episode for episode in map_episodes}

        for case in map_episodes:
            if case.gap_ms <= support_half_width_ms or case.gap_ms > max_gap_ms:
                continue
            if candidate_time_is_valid is not None and not candidate_time_is_valid(case.target_time_ms):
                continue

            following = next_episode_by_origin.get(case.target_row_index)
            if following is not None and following.gap_ms <= support_half_width_ms:
                continue

            control_pool = [
                control
                for control in map_episodes
                if control.episode_index != case.episode_index
                and control.gap_ms > case.gap_ms + support_half_width_ms
                and (
                    candidate_time_is_valid is None
                    or candidate_time_is_valid(control.origin_time_ms + case.gap_ms)
                )
            ]
            if len(control_pool) < controls_per_case:
                continue

            sampled_positions = rng.choice(len(control_pool), size=controls_per_case, replace=False)
            controls = tuple(
                sorted((control_pool[int(position)] for position in sampled_positions), key=_episode_index)
            )
            choice_sets.append(
                SameGapChoiceSet(
                    map_id=map_id,
                    case_episode_index=case.episode_index,
                    control_episode_indices=tuple(control.episode_index for control in controls),
                    elapsed_ms=case.gap_ms,
                    candidate_center_times_ms=(
                        case.target_time_ms,
                        *(control.origin_time_ms + case.gap_ms for control in controls),
                    ),
                    support_half_width_ms=support_half_width_ms,
                )
            )

    return tuple(choice_sets)


def triangular_support_weights(half_width_ms: int) -> NDArray[np.float64]:
    """Return normalized positive triangular weights at integer-ms offsets."""

    _validate_half_width(half_width_ms)
    offsets = np.arange(-half_width_ms, half_width_ms + 1, dtype=np.int64)
    numerators = half_width_ms + 1 - np.abs(offsets)
    denominator = float((half_width_ms + 1) ** 2)
    return numerators.astype(np.float64) / denominator


def triangular_log_support_scores(
    point_log_scores: ArrayLike,
    *,
    half_width_ms: int,
) -> NDArray[np.float64]:
    """Integrate candidate point log-scores over triangular local support."""

    weights = triangular_support_weights(half_width_ms)
    scores = np.asarray(point_log_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("point_log_scores must have shape (candidate, support_offset).")
    if scores.shape[0] == 0:
        raise ValueError("point_log_scores must contain at least one candidate.")
    if scores.shape[1] != weights.size:
        raise ValueError(
            f"Expected {weights.size} support offsets for half_width_ms={half_width_ms}, "
            f"got {scores.shape[1]}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("point_log_scores must be finite.")

    return np.logaddexp.reduce(scores + np.log(weights)[None, :], axis=1)


def triangular_support_choice_nll(
    point_log_scores: ArrayLike,
    *,
    half_width_ms: int,
    case_index: int = 0,
) -> float:
    """Conditional-choice negative log-likelihood after support integration."""

    support_scores = triangular_log_support_scores(
        point_log_scores,
        half_width_ms=half_width_ms,
    )
    if not 0 <= case_index < support_scores.size:
        raise ValueError("case_index is outside the candidate axis.")
    return float(np.logaddexp.reduce(support_scores) - support_scores[case_index])


def _exact_integer_ms(value: float) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Hit-object onset must be a finite integer-millisecond timestamp, got {value!r}.")
    return int(numeric)


def _validate_half_width(half_width_ms: int) -> None:
    if isinstance(half_width_ms, bool) or not isinstance(half_width_ms, int) or half_width_ms < 0:
        raise ValueError("support half-width must be a non-negative integer number of milliseconds.")


def _validate_playable_lane_sequence(rows: Sequence[ManiaAnchorRow]) -> None:
    active_until_ms = [-1, -1, -1, -1]
    for row in rows:
        lanes_in_row: set[int] = set()
        for hitobject in row.hitobjects:
            lane = hitobject.lane
            if not 0 <= lane < 4:
                raise ValueError(f"Expected a 4K lane in [0,3], got {lane}.")
            if lane in lanes_in_row:
                raise ValueError("A materialized onset row cannot contain two objects in one lane.")
            lanes_in_row.add(lane)
            if row.time_ms < active_until_ms[lane]:
                raise ValueError("A lane onset cannot occur before its previous hold closes.")
            if hitobject.kind is ManiaHitObjectKind.HOLD:
                end_time_ms = _exact_integer_ms(hitobject.end_time_ms)
                if end_time_ms < row.time_ms:
                    raise ValueError("Hold end must not precede its onset.")
                active_until_ms[lane] = end_time_ms


def _validate_episode_chain(map_id: str, episodes: Sequence[AnchorEpisode]) -> None:
    for index, episode in enumerate(episodes):
        if episode.map_id != map_id:
            raise ValueError("Every episode in a map group must have the same map_id.")
        if episode.episode_index != index:
            raise ValueError(f"Episodes for map {map_id!r} must form a complete zero-based chain.")
        if episode.origin_row_index != index or episode.target_row_index != index + 1:
            raise ValueError(f"Episode row indices for map {map_id!r} must be consecutive.")
        if episode.gap_ms <= 0 or episode.gap_ms != episode.target_time_ms - episode.origin_time_ms:
            raise ValueError(f"Episode gap for map {map_id!r} is inconsistent with its timestamps.")
        if index > 0 and episodes[index - 1].target_time_ms != episode.origin_time_ms:
            raise ValueError(f"Episode timestamps for map {map_id!r} do not form a complete chain.")


def _episode_index(episode: AnchorEpisode) -> int:
    return episode.episode_index
