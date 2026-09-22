from dataclasses import asdict, replace
import json
import subprocess
import sys

import pytest

from ensomi_model.research.oracle_time_continuation.data import admit_source
from ensomi_model.research.oracle_time_continuation.engine import ContinuationState, prefill
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow, TimeSkeleton
from ensomi_model.research.scoped_style_modeling.dataset import ContractError, digest
from .conftest import admit, source_bytes


def test_seed_counts_objects_and_keeps_complete_chords_and_release_rows(chord_source):
    source = chord_source
    seed = source.minimum_seed()
    assert seed.eligible
    assert seed.seed_note_count == 33
    assert seed.seed_row_count == 10
    assert seed.seed_elapsed_ms == 800
    assert source.targets[1] == CompleteRow(50, (3, 0, 0, 0))
    state = source.prefix_state()
    assert state.next_index == 10
    assert state.replay.note_count == 33
    assert state.replay.open_ln_start_ms == (800,) * 4
    assert state.query().time_ms == 900
    assert state.query().history is state.replay
    assert source.identity.group_id == "song:fixture"


def test_later_prefix_replays_from_true_beginning(chord_source):
    source = chord_source
    state = source.prefix_state(12)
    assert state == prefill(source.skeleton, source.targets[:12])
    assert state.replay.first_time_ms == 0
    assert state.replay.note_count == 33
    assert state.replay.open_ln_start_ms == (None, None, 800, 800)
    for invalid in (0, 9, len(source.targets), True):
        with pytest.raises(ContractError, match="full 30-note seed"):
            source.prefix_state(invalid)


@pytest.mark.parametrize("notes,reason", [(0, "fewer-than-30-notes"), (29, "fewer-than-30-notes"),
                                         (30, "no-target-suffix")])
def test_short_charts_are_explicitly_ineligible(notes, reason):
    source = admit([(0, index * 10, index * 10) for index in range(notes)])
    seed = source.minimum_seed()
    assert seed.ineligible_reason == reason
    assert seed.seed_note_count == seed.seed_row_count == notes
    assert not seed.eligible
    with pytest.raises(ContractError, match=reason):
        source.prefix_state()


def test_exactly_thirty_notes_can_leave_a_release_only_suffix():
    objects = [(0, index * 10, index * 10) for index in range(29)] + [(0, 290, 10_000)]
    source = admit(objects)
    assert source.minimum_seed().seed_note_count == 30
    state = source.prefix_state()
    assert state.query().is_terminal
    assert (3, 0, 0, 0) in state.query().legal_actions
    assert state.commit(source.targets[-1]).replay.note_count == 30


@pytest.mark.parametrize("times", [(0, 0), (1, 0), (-1, 0), (0, float("inf")), (float("nan"),), (True,)])
def test_skeleton_rejects_duplicates_bad_order_and_invalid_times(times):
    with pytest.raises(ContractError):
        TimeSkeleton(times)


@pytest.mark.parametrize("actions", [(0, 0, 0, 0), (1, None, 0, 0), (1, 0, 0),
                                   (1, 0, 0, 4), (True, 0, 0, 0), None])
def test_unknown_actions_and_padding_cannot_be_materialized(actions):
    with pytest.raises(ContractError):
        CompleteRow(0, actions)


def test_input_mutability_and_target_skeleton_mismatch(chord_source):
    actions, times = [1, 0, 0, 0], [0, 10]
    row, skeleton = CompleteRow(0, actions), TimeSkeleton(times)
    actions[0], times[1] = 2, 20
    assert row.actions == (1, 0, 0, 0) and skeleton.times_ms == (0, 10)
    with pytest.raises(ContractError, match="exactly cover"):
        replace(chord_source, targets=chord_source.targets[:-1])


def test_current_and_future_actions_endpoints_and_identities_are_isolated():
    a = admit([(0, 0, 1000), (1, 10, 10), (2, 20, 20), (1, 1000, 1000), (1, 2000, 2000)])
    b = admit([(0, 0, 2000), (1, 10, 10), (3, 20, 20), (1, 1000, 1000), (1, 2000, 2000)])
    b = replace(b, identity=replace(b.identity, group_id="another-song", split="test"))
    assert a.skeleton == b.skeleton
    assert a.targets != b.targets and a.identity != b.identity
    sa, sb = prefill(a.skeleton, a.targets[:2]), prefill(b.skeleton, b.targets[:2])
    assert sa.query() == sb.query()
    assert sa.query(16) == sb.query(16)
    assert sa.query().clocks == sb.query().clocks
    assert sa.query().legal_actions == sb.query().legal_actions
    payload = asdict(sa.query())
    assert set(payload) == {"time_ms", "is_terminal", "history", "future_offsets_ms"}
    assert set(payload["history"]) == {"row_count", "note_count", "first_time_ms", "last_row",
                                      "open_ln_start_ms", "last_lane_attack_ms", "last_lane_release_ms", "is_complete"}
    assert payload['future_offsets_ms'] == ()
    assert not any(word in json.dumps(payload) for word in ("sha256", "group", "end_ms", "remaining", "duration", "target"))
    sa, sb = sa.commit(a.targets[2]), sb.commit(b.targets[2])
    assert sa.query() != sb.query()
    assert sa.replay.last_lane_attack_ms != sb.replay.last_lane_attack_ms


def test_future_time_changes_do_not_affect_current_query_except_true_terminal():
    a = ContinuationState(TimeSkeleton((0, 10, 100)))
    b = ContinuationState(TimeSkeleton((0, 10, 1000, 2000)))
    assert a.query() == b.query()
    a, b = a.commit(CompleteRow(0, (2, 0, 0, 0))), b.commit(CompleteRow(0, (2, 0, 0, 0)))
    assert a.query() == b.query()
    terminal = prefill(TimeSkeleton((0, 10)), (CompleteRow(0, (2, 0, 0, 0)),))
    assert terminal.query().is_terminal and not a.query().is_terminal
    assert terminal.query().history == a.query().history


def test_bounded_time_context_comes_from_full_skeleton_and_uses_float64_differences():
    start = 1e12
    times = tuple(start + i * .125 for i in range(40))
    state = prefill(TimeSkeleton(times), (CompleteRow(times[0], (2, 0, 0, 0)),))
    query = state.query(16)
    assert query.future_offsets_ms == tuple(.125 * i for i in range(1, 17))
    assert state.query().future_offsets_ms == ()
    assert state.query(1).future_offsets_ms == (.125,)
    terminal = prefill(TimeSkeleton(times[:2]), (CompleteRow(times[0], (2, 0, 0, 0)),))
    assert terminal.query(16).is_terminal and terminal.query(16).future_offsets_ms == ()
    for invalid in (-1, 17, True, .5):
        with pytest.raises(ContractError, match='time_lookahead_rows'):
            state.query(invalid)
    for invalid in ((0.,), (2., 1.), (float('inf'),), [1.]):
        with pytest.raises(ContractError):
            replace(query, future_offsets_ms=invalid)
    with pytest.raises(ContractError, match='nonterminal'):
        replace(terminal.query(), future_offsets_ms=(1.,))


@pytest.mark.parametrize("objects", [((0, 0, 100), (0, 100, 100)), ((0, 0, 100), (0, 100, 200)),
                                     ((0, 0, 100), (0, 50, 50)), ((0, 0, 0), (0, 0, 0)), ((0, -1, -1),)])
def test_raw_admission_rejects_incompatible_sources_without_retiming(objects):
    with pytest.raises(ContractError):
        admit(objects)


def test_raw_identity_and_mode_are_verified():
    data = source_bytes([(0, 0, 0)])
    with pytest.raises(ContractError, match="SHA-256"):
        admit_source(data, "0" * 64, group_id="song:one", split="train")
    bad = data.replace(b"Mode:3", b"Mode:0")
    with pytest.raises(ContractError, match="Mode:3"):
        admit_source(bad, digest(bad), group_id="song:one", split="train")
    with pytest.raises(ContractError, match="song group"):
        admit_source(data, digest(data), group_id="", split="train")


def test_package_import_does_not_load_legacy_or_model_feature_paths():
    script = """
import sys
from ensomi_model.research.oracle_time_continuation import data, engine
for module in sys.modules:
    assert not module.startswith(('torch', 'ensomi_model.models', 'ensomi_model.timing',
                                  'ensomi_model.inference', 'ensomi_model.training')), module
    assert module not in ('ensomi_model.research.source_action_modeling.observation',
                          'ensomi_model.research.source_action_modeling.tensors',
                          'ensomi_model.research.source_action_modeling.local_representation')
"""
    subprocess.run([sys.executable, "-c", script], check=True)
