from dataclasses import asdict, replace
import gzip

import pytest

from ensomi_model.research.scoped_style_modeling.dataset import ContractError, Interval, NoteRef, canonical_json, digest
from ensomi_model.research.scoped_style_modeling.prepare import _chart_payload, load_chart
from ensomi_model.research.scoped_style_modeling.relations import prepare_relations
from ensomi_model.research.scoped_style_modeling.replay import (
    evidence_masks, hand_mask, mirror_objects, parse_source, prepare_chart, selected_objects, time_feature,
)


def note(line, lane, start, end=None):
    return NoteRef(line, lane, "normal" if end is None else "long", start, start if end is None else end)


def fixture():
    # An entering hold outlives Q; another closes at a. A real row at a follows
    # the marker, and a head at b remains context without entering aggregation.
    objects = (note(10, 0, 0, 900), note(11, 1, 20, 100), note(12, 2, 100),
               note(13, 3, 180, 250), note(14, 2, 400), note(15, 3, 480))
    return objects, prepare_chart(objects, Interval(100, 400), Interval(50, 500))


def row_index(prepared, t, phase="source"):
    return next(i for i, row in enumerate(prepared.inputs.rows) if (row.time_ms, row.phase) == (t, phase))


def test_boundary_identity_full_tails_and_roundtrip():
    objects, prepared = fixture()
    assert prepared.decisions[0].candidates == (objects[0], None, None, None)
    first = prepared.inputs.rows[prepared.decisions[0].encoder_index]
    assert first.phase == "boundary"
    assert first.lanes[1].occupied_before and first.lanes[1].before_remaining_ms == 0
    source = prepared.inputs.rows[row_index(prepared, 100)]
    assert source.lanes[1].ln_close and not source.lanes[1].occupied_after
    assert source.lanes[2].tap
    assert prepared.decisions[1].encoder_index == row_index(prepared, 100)
    targets = (objects[0], objects[2], objects[3])
    masks = evidence_masks(prepared, targets)
    assert selected_objects(prepared, masks) == tuple(sorted(targets))
    assert targets[0].end_ms == 900
    assert not first.lanes[0].before_head_visible
    assert not first.lanes[0].before_close_visible
    assert prepared.inputs.rows[prepared.inputs.section_indices[-1]].time_ms == 400
    assert row_index(prepared, 400) not in prepared.inputs.section_indices
    assert prepared.inputs.readout_elapsed_ms[-1] == 150
    assert prepared.inputs.rows[row_index(prepared, 100, "boundary")].lanes[2].next_attack_ms == 0
    assert source.lanes[2].previous_attack_ms is None
    assert source.lanes[2].next_attack_ms == 300


@pytest.mark.parametrize("bad", [note(11, 1, 20, 100), note(14, 2, 400), note(10, 0, 0, 899), note(999, 2, 100)])
def test_bad_reference_never_becomes_missing(bad):
    _, prepared = fixture()
    with pytest.raises(ContractError, match="identity/membership"):
        evidence_masks(prepared, (bad,))


def test_missing_empty_release_only_and_invalid_masks():
    _, prepared = fixture()
    assert evidence_masks(prepared, None) is None
    empty = evidence_masks(prepared, ())
    assert empty == (0,) * len(prepared.decisions)
    assert selected_objects(prepared, empty) == ()
    release = next(d for d in prepared.decisions if prepared.inputs.rows[d.encoder_index].time_ms == 250)
    assert release.valid_masks == (0,)
    assert len(prepared.decisions[0].valid_masks) == 2
    with pytest.raises(ContractError, match="Impossible"):
        selected_objects(prepared, (15, *empty[1:]))
    with pytest.raises(ContractError, match="count"):
        selected_objects(prepared, ())
    with pytest.raises(ContractError):
        evidence_masks(prepared, (prepared.visible_objects[0],) * 2)


def test_selection_cannot_change_source_or_graph_and_skips_preserve_ab():
    objects = tuple(note(i+1, i % 2 * 3, i*100) for i in range(6))
    prepared = prepare_chart(objects, Interval(0, 550), Interval(0, 600))
    before = asdict(prepared)
    edges = prepare_relations(prepared)
    evidence_masks(prepared, objects[::2])
    evidence_masks(prepared, ())
    assert asdict(prepared) == before
    assert prepare_relations(prepared) == edges
    a, b = row_index(prepared, 0), row_index(prepared, 200)
    relation = next(e for e in edges if (e.query, e.neighbor) == (2*a, 2*b))
    assert {r.kind for r in relation.relations} >= {"attack_succession_2", "same_lane_recurrence"}
    assert next(r for r in relation.relations if r.kind == "same_lane_recurrence").intervening_attack_rows == 1


def test_empty_section_and_strict_context_limits():
    objects = (note(1, 0, 0, 1000), note(2, 1, 20), note(3, 1, 900))
    prepared = prepare_chart(objects, Interval(200, 400), Interval(100, 500))
    assert prepared.inputs.event_indices == ()
    assert len(prepared.inputs.section_indices) == 2
    assert prepared.inputs.readout_elapsed_ms == (0, 200)
    assert len(prepared.decisions) == 1
    assert prepared.visible_objects == (objects[0],)
    for row in prepared.inputs.rows:
        assert row.lanes[0].occupied_before and row.lanes[0].occupied_after
        assert row.lanes[1].previous_attack_ms is None
        assert row.lanes[1].next_attack_ms is None
    assert not any(r.kind in ("ln_identity", "occupied_role") for e in prepare_relations(prepared) for r in e.relations)


def test_context_start_close_and_context_end_head():
    objects = (note(1, 0, 0, 100), note(2, 0, 500))
    prepared = prepare_chart(objects, Interval(100, 500), Interval(100, 500))
    assert prepared.visible_objects == (objects[0],)
    assert prepared.decisions[0].eligible_mask == 0
    assert prepared.inputs.rows[0].lanes[0].occupied_before
    assert prepared.inputs.rows[1].lanes[0].ln_close
    assert not prepared.inputs.rows[-1].lanes[0].occupied_before
    assert len(prepared.inputs.event_indices) == 1


def test_relations_keep_multiple_role_pairs_and_real_ln_endpoints():
    objects = (note(1, 0, 0, 300), note(2, 1, 0, 300), note(3, 2, 100), note(4, 3, 100), note(5, 2, 200))
    prepared = prepare_chart(objects, Interval(0, 400), Interval(0, 400))
    edges = prepare_relations(prepared)
    assert len({(e.query, e.neighbor) for e in edges}) == len(edges)
    assert sum(e.query == e.neighbor for e in edges) == len(prepared.inputs.rows)*2
    a, b = row_index(prepared, 100), row_index(prepared, 0)
    edge = next(e for e in edges if (e.query, e.neighbor) == (2*a+1, 2*b))
    assert {(r.query_role, r.neighbor_role) for r in edge.relations if r.kind == "occupied_role"} == {(0, 2), (0, 3), (1, 2), (1, 3)}
    end = row_index(prepared, 300)
    edge = next(e for e in edges if (e.query, e.neighbor) == (2*b, 2*end))
    assert {(r.query_role, r.neighbor_role) for r in edge.relations if r.kind == "ln_identity"} == {(0, 0), (1, 1)}
    assert all(r.duration_ms == 300 for r in edge.relations if r.kind == "ln_identity")
    assert any(r.kind == "event_succession" for e in edges if e.query//2 == end for r in e.relations)


def test_mirror_graph_is_equivariant_in_query_relative_coordinates():
    objects, prepared = fixture()
    mirrored = prepare_chart(mirror_objects(objects), prepared.inputs.scope, prepared.inputs.context)
    original_edges = prepare_relations(prepared)
    mirrored_edges = {(e.query, e.neighbor): e for e in prepare_relations(mirrored)}
    for edge in original_edges:
        assert replace(edge, query=edge.query ^ 1, neighbor=edge.neighbor ^ 1) == mirrored_edges[edge.query ^ 1, edge.neighbor ^ 1]
    assert mirrored.inputs.section_indices == prepared.inputs.section_indices
    for row, other in zip(prepared.inputs.rows, mirrored.inputs.rows):
        assert row.lanes == other.lanes[::-1]
    masks = evidence_masks(prepared, objects[:1])
    mirror_masks = evidence_masks(mirrored, mirror_objects(objects[:1]))
    for a, b in zip(masks, mirror_masks):
        assert hand_mask(a, 0) == hand_mask(b, 1)
        assert hand_mask(a, 1) == hand_mask(b, 0)


def osu(lines):
    return ("osu file format v14\r\n\r\n[General]\r\nMode:3\r\n[Difficulty]\r\nCircleSize:4\r\n[HitObjects]\r\n" + "\r\n".join(lines) + "\r\n").encode()


def test_hash_before_identity_and_physical_lines():
    data = osu(["64,192,100.5,128,0,200.25:0:0:0:0:", "448,192,200.25,1,0,0:0:0:0:"])
    chart = parse_source(data, digest(data))
    assert chart.objects == (note(8, 0, 100.5, 200.25), note(9, 3, 200.25))
    with pytest.raises(ContractError, match="SHA-256"):
        parse_source(data + b"\n", digest(data))
    # Unicode/control separators inside metadata are not physical .osu lines.
    data = data.replace(b"[General]", b"// metadata\x0bseparator\r\n[General]")
    assert parse_source(data, digest(data)).objects[0].source_line == 9


def test_boundary_does_not_replace_original_event_adjacency():
    objects = (note(1, 0, 100), note(2, 1, 300))
    prepared = prepare_chart(objects, Interval(200, 400), Interval(0, 500))
    a, b = row_index(prepared, 100), row_index(prepared, 300)
    edge = next(e for e in prepare_relations(prepared) if (e.query, e.neighbor) == (2*a, 2*b))
    assert "event_succession" in {r.kind for r in edge.relations}


@pytest.mark.parametrize("lines", [
    ["64,192,100,128,0,100:0:0:0:0:"],
    ["64,192,100,128,0,200:0:0:0:0:", "64,192,150,1,0,0:0:0:0:"],
    ["64,192,100,1,0,0:0:0:0:", "64,192,100,1,0,0:0:0:0:"],
    ["64,192,100,2,0,0:0:0:0:"],
])
def test_unrepresentable_source_fails_without_repair(lines):
    data = osu(lines)
    with pytest.raises(ContractError):
        parse_source(data, digest(data))


def test_close_head_coincidence_is_explicit_and_never_retimed():
    data = osu(["64,192,100,128,0,200:0:0:0:0:", "64,192,200,128,0,300:0:0:0:0:"])
    chart = parse_source(data, digest(data))
    assert chart.row_incompatibilities == ((8, 9),)
    prepared = prepare_chart(chart.objects, Interval(100, 400), Interval(0, 500))
    facts = prepared.inputs.rows[row_index(prepared, 200)].lanes[0]
    assert facts.ln_start and facts.ln_close and facts.occupied_before and facts.occupied_after
    assert facts.before_remaining_ms == 0 and facts.after_age_ms == 0


def test_prepared_artifact_round_trip(tmp_path):
    _, prepared = fixture()
    graph = prepare_relations(prepared)
    path = tmp_path/'chart.json.gz'
    payload = canonical_json(_chart_payload(prepared, graph))
    with gzip.open(path, 'wt') as stream:
        stream.write(payload)
    assert load_chart(path, expected_sha256=digest(payload.encode())) == (prepared, graph)
    with pytest.raises(ContractError, match="SHA-256"):
        load_chart(path, expected_sha256="changed")


def test_time_features_keep_zero_and_unavailability_distinct():
    assert time_feature(None) == (0, False)
    assert time_feature(0) == (0, True)
    assert time_feature(-1000)[0] == -time_feature(1000)[0]
