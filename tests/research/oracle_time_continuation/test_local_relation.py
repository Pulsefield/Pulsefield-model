import torch

from pulsefield_model.research.oracle_time_continuation.config import BackboneConfig
from pulsefield_model.research.oracle_time_continuation.local import LocalEncoder, LocalState
from pulsefield_model.research.oracle_time_continuation.relation import RelationEncoder, RelationState
from pulsefield_model.research.oracle_time_continuation.schema import CompleteRow


def test_local_kernel_uses_time_and_ordered_endpoint_actions_on_each_edge():
    torch.manual_seed(71)
    encoder = LocalEncoder(8)
    raw = torch.randn(2, 8)
    first = CompleteRow(0, (1, 0, 0, 0))
    state = encoder.commit(LocalState(), first, 1, raw, None)
    fast = encoder.commit(state, CompleteRow(10, (0, 1, 0, 0)), 2, raw, 10)
    slow = encoder.commit(state, CompleteRow(10000, (0, 1, 0, 0)), 2, raw, 10000)
    roles = encoder.commit(state, CompleteRow(10, (1, 0, 0, 0)), 2, raw, 10)
    assert not torch.allclose(fast.latest[0].value, slow.latest[0].value)
    assert not torch.allclose(fast.latest[0].value, roles.latest[0].value)
    # Zeroing the conditional modulation removes the endpoint-action effect
    # with identical raw values; actions therefore affect the kernel itself.
    with torch.no_grad():
        for layer in encoder.layers:
            for gate in layer.gates:
                for parameter in gate.parameters():
                    parameter.zero_()
    fast = encoder.commit(state, CompleteRow(10, (0, 1, 0, 0)), 2, raw, 10)
    roles = encoder.commit(state, CompleteRow(10, (1, 0, 0, 0)), 2, raw, 10)
    torch.testing.assert_close(fast.latest[0].value, roles.latest[0].value, atol=0, rtol=0)


def test_relation_chord_is_one_node_with_all_lane_tags_and_causal_links():
    encoder = RelationEncoder(BackboneConfig(hidden=8, heads=2))
    state = encoder.commit(RelationState(), CompleteRow(0, (2, 1, 1, 2)), 1, torch.randn(2, 8))
    assert state.visible_ids == (1,)
    assert state.attacks == ((1,),) * 4
    assert state.active_heads == (1, None, None, 1)
    previous = state.nodes[0].payload.clone()
    state = encoder.commit(state, CompleteRow(5000, (3, 1, 1, 0)), 2, torch.randn(2, 8))
    second = state.nodes[-1]
    assert state.visible_ids == (1, 2)
    assert second.lane_predecessors == (1,) * 4
    assert second.hand_predecessors == (1, 1)
    assert second.closed_heads == (1, None, None, None)
    assert second.closed_durations_ms == (5000, None, None, None)
    assert state.active_heads == (None, None, None, 1)
    torch.testing.assert_close(state.nodes[0].payload, previous, atol=0, rtol=0)


def test_relation_indices_are_bounded_and_pinned_payload_does_not_follow_predecessor_chains():
    config = BackboneConfig(hidden=8, heads=2)
    encoder = RelationEncoder(config)
    state = encoder.commit(RelationState(), CompleteRow(0, (2, 0, 0, 0)), 1, torch.randn(2, 8))
    for index in range(2, 250):
        actions = (0, 2 if index % 2 == 0 else 3, 1, 1)
        state = encoder.commit(state, CompleteRow(index, actions), index, torch.randn(2, 8))
        assert len(state.nodes) == len(set(state.visible_ids)) <= 68
        assert all(len(indices) <= 12 for indices in state.attacks)
        assert all(len(indices) <= 4 for indices in state.releases)
        assert state.active_heads[0] == 1
        assert 1 in state.visible_ids
    assert state.nodes[0].row.time_ms == 0
    assert all(previous is None or isinstance(previous, int)
               for node in state.nodes for previous in (*node.lane_predecessors, *node.closed_heads))
