from dataclasses import asdict, fields, replace

import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.dataset import ContractError, Interval, NoteRef, digest
from pulsefield_model.research.scoped_style_modeling.replay import parse_source, prepare_chart
from pulsefield_model.research.source_action_modeling.observation import EventBlock, observe, visible_states
from pulsefield_model.research.source_action_modeling.tensors import collate, observation_relations, row_token
from .conftest import example, fixture_chart


def test_hidden_changes_leave_all_encoder_inputs_and_topology_identical():
    a, b = example(), example(changed=True)
    chart = fixture_chart()
    original = asdict(chart)
    observe(chart, EventBlock(1, 4))
    assert a.targets != b.targets
    assert a.observation == b.observation
    assert observation_relations(a.observation) == observation_relations(b.observation)
    x, y = collate([a]), collate([b])
    for field in fields(x.observation):
        assert torch.equal(getattr(x.observation, field.name), getattr(y.observation, field.name)), field.name
    for field in fields(x.queries):
        assert torch.equal(getattr(x.queries, field.name), getattr(y.queries, field.name)), field.name
    assert original == asdict(chart)


def test_unknown_occupation_and_actions_are_not_absence_and_post_block_is_not_replayed():
    item = example()
    obs = item.observation
    before, after, entry = visible_states(obs)
    hidden = obs.target_indices
    assert entry == (True, True, False, False)
    assert before[hidden[0]] == entry
    assert after[hidden[0]] == (None,) * 4
    assert before[hidden[-1] + 1] == (None,) * 4
    batch = collate([item])
    assert batch.observation.lanes[0, hidden[0], :, :, 3].count_nonzero() == 0
    assert batch.observation.lanes[0, 0, :, :, 3].all()
    unknown = collate([replace(item, observation=replace(obs, entering_occupancy=(None,) * 4))])
    assert unknown.observation.lanes[0, 0, :, :, 5].count_nonzero() == 0
    assert batch.observation.lanes[0, 0, :, :, 5].all()


def test_known_zero_time_survives_and_immediate_successors_do_not_cross_unknown():
    batch = collate([example()])
    # Boundary at t=0 precedes a visible LN head in the inner left lane at t=0.
    assert batch.observation.lanes[0, 0, 0, 1, 10:].tolist() == [0, 1]
    obs = example().observation
    first = next(i for i, row in enumerate(obs.rows) if row.time_ms == 0 and row.phase == "source")
    assert batch.observation.lanes[0, first, 0, 1, 10:].tolist() == [0, 0]
    lo, hi = obs.target_indices[0], obs.target_indices[-1]
    for q, n, kinds in observation_relations(obs):
        if min(q // 2, n // 2) < lo and max(q // 2, n // 2) > hi:
            assert not {kind for kind, *_ in kinds} & {"recurrence", "attack_1", "attack_2", "ln_identity"}


def test_release_only_rows_close_head_coincidences_and_boundary_at_same_time():
    data = b"osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:4\n[HitObjects]\n64,192,0,128,0,100:0:0:0:0:\n64,192,100,128,0,200:0:0:0:0:\n"
    source = parse_source(data, digest(data))
    chart = prepare_chart(source.objects, Interval(100, 250), Interval(0, 300))
    item = observe(chart, EventBlock(1, 2), entering_occupancy=(False,) * 4)
    assert item.targets == ((6, 0, 0, 0), (4, 0, 0, 0))
    assert [item.observation.rows[i].time_ms for i in item.observation.target_indices] == [100, 200]
    assert item.attack_group_span == 1
    assert all(item.observation.rows[i].phase == "source" for i in item.observation.target_indices)
    marker = next(r for r in item.observation.rows if r.time_ms == 100 and r.phase == "boundary")
    assert marker.actions == (0,) * 4
    assert source.objects[0].end_ms == 100


def test_entering_hold_closing_at_context_start_and_exiting_hold():
    chart = prepare_chart((NoteRef(1, 0, "long", -100, 0), NoteRef(2, 1, "long", 100, 1000)),
                          Interval(0, 200), Interval(0, 200))
    item = observe(chart, EventBlock(0, 2), entering_occupancy=(True, False, False, False))
    assert item.targets == ((4, 0, 0, 0), (0, 2, 0, 0))
    assert [r.time_ms for r in item.observation.rows if r.phase == "source"] == [0, 100]
    assert not any(k == "ln_identity" for _, _, kinds in observation_relations(item.observation) for k, *_ in kinds)
    before, after, _ = visible_states(item.observation)
    assert before[0][0] is True and after[-1] == (None,) * 4


def test_visible_ln_identity_preserves_both_endpoints_before_an_unknown_interval():
    obs = example(start=5, size=1).observation
    indices = {r.time_ms: i for i, r in enumerate(obs.rows) if r.phase == "source"}
    relations = {(q, n): kinds for q, n, kinds in observation_relations(obs)}
    for head, close in ((0, 100), (100, 300)):
        assert ("ln_identity", 1, 1) in relations[2 * indices[head], 2 * indices[close]]


def test_analysis_metadata_never_enters_tensors_and_bad_blocks_fail():
    a = example()
    x, y = collate([a]), collate([replace(a, attack_group_span=999)])
    assert torch.equal(x.observation.lanes, y.observation.lanes)
    for block in (EventBlock(99, 4), EventBlock(0, 64)):
        with pytest.raises(ContractError, match="exceeds"):
            observe(fixture_chart(), block)
    with pytest.raises(ContractError, match="positive"):
        EventBlock(0, 0)
    with pytest.raises(ContractError, match="source-event"):
        replace(a.observation, target_indices=(0,))
    with pytest.raises(ContractError, match="Invalid joint"):
        row_token((3, 0, 0, 0))
