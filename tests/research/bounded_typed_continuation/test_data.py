from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import Arm, Schedule
from ensomi_model.research.bounded_typed_continuation.data import SourceChart, SourceInterval, batch_likelihood, prepare_batch
from ensomi_model.research.bounded_typed_continuation.features import query_features
from ensomi_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from ensomi_model.research.oracle_time_continuation.data import SourceIdentity
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE, SOURCE_FORMAT
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


IDENTITY = SourceIdentity('a' * 64, 'b' * 64, 'synthetic-group', 'train')


def chart(actions, *, times=None, minimum_seed_notes=1):
    rows = np.zeros(len(actions), dtype=ROW_DTYPE)
    rows['time'] = np.arange(len(actions)) * 100. if times is None else times
    rows['actions'] = actions
    return SourceChart(IDENTITY, rows, minimum_seed_notes=minimum_seed_notes)


def mixed_chart():
    return chart([(2, 2, 0, 0), (0, 0, 1, 0), (3, 0, 0, 0), (1, 0, 2, 0),
                  (0, 3, 0, 0), (2, 0, 0, 1), (0, 0, 3, 0), (3, 0, 0, 0), (1, 0, 0, 0)])


@pytest.mark.parametrize('arm', list(Arm))
def test_indexed_exact_states_match_full_physical_replay_and_plan_visibility(arm):
    source = mixed_chart()
    state = Schedule.from_seed(arm, source.view(arm).timing, [source.row(0)],
                               None if arm == Arm.R0 else {0: 2, 1: 4})
    for i in range(1, len(source.rows)):
        assert source.state(arm, i) == state
        ends = {c: int(end) for c, end in enumerate(source.endpoints[i]) if end >= 0} if arm == Arm.O1 else None
        state, _ = state.advance(source.row(i).actions, ends)
    assert source.state(arm, len(source.rows)) == state


def test_r1_source_future_endpoint_labels_do_not_enter_history_or_query():
    a = chart([(1, 0, 0, 0), (2, 2, 0, 0), (0, 0, 1, 0), (3, 0, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0)])
    b = chart([(1, 0, 0, 0), (2, 2, 0, 0), (0, 0, 1, 0), (0, 3, 0, 0), (3, 0, 0, 0), (1, 0, 0, 0)])
    for arm in (Arm.R0, Arm.R1):
        np.testing.assert_array_equal(a.content(arm, 0, 3), b.content(arm, 0, 3))
        np.testing.assert_array_equal(query_features([a.state(arm, 3)], a.view(arm)),
                                      query_features([b.state(arm, 3)], b.view(arm)))
    assert not np.array_equal(a.content(Arm.O1, 0, 3), b.content(Arm.O1, 0, 3))
    # The current O1 endpoint label still cannot change its own head decision.
    model = BoundedModel(ModelConfig(Arm.O1, hidden=8, levels=2, coupling_rank=2))
    encoded = []
    for source in (a, b):
        batch = prepare_batch([SourceInterval(source, 0, 1)], Arm.O1, model.temporal.config.receptive_tokens)
        raw, valid = torch.from_numpy(batch.raw), torch.from_numpy(batch.valid)
        before = model.temporal.before(model.temporal(raw, valid), valid, truncated_start=torch.from_numpy(batch.truncated))
        hands = model.readout(before[batch.batch_indices, batch.positions], torch.from_numpy(batch.query_features))
        encoded.append(model.decision_log_probs(hands, batch.states))
    torch.testing.assert_close(encoded[0], encoded[1], atol=0, rtol=0)


def test_shared_onset_intervals_partition_the_complete_suffix_including_releases():
    source = mixed_chart()
    parts = [SourceInterval(source, i, 1) for i in range(len(source.onsets))]
    assert parts[0].start == source.seed_rows and parts[-1].stop == len(source.rows)
    assert all(a.stop == b.start for a, b in zip(parts, parts[1:]))
    for arm in Arm:
        batch = prepare_batch(parts, arm, 7)
        assert batch.source_onsets == len(source.onsets)
        assert batch.physical_rows == len(source.rows) - source.seed_rows
        assert len(batch.states) == (len(source.onsets) if arm == Arm.O1 else batch.physical_rows)
    first = prepare_batch([parts[0]], Arm.O1, 7)
    assert first.physical_rows == 2 and len(first.states) == 1


@pytest.mark.parametrize('arm', list(Arm))
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_batched_bounded_source_likelihood_has_finite_gradients_and_actual_onset_denominator(arm, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    model = BoundedModel(ModelConfig(arm, hidden=8, levels=2, coupling_rank=2)).to(device)
    source = mixed_chart()
    items = [SourceInterval(source, 0, 2), SourceInterval(source, 2, 2)]
    prepared = prepare_batch(items, arm, model.temporal.config.receptive_tokens)
    loss, factors = batch_likelihood(model, prepared, candidate_budget=3)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, factors.sum() / 4)
    loss.backward()
    assert model.temporal.input.weight.grad.norm() > 0 and model.joint.unary.weight.grad.norm() > 0


@pytest.mark.parametrize('arm,consequence', [(Arm.O1, 'none'), (Arm.R1, 'frontier')])
def test_crop_matches_full_bos_scoring_with_an_ancient_held_plan(arm, consequence):
    actions = [(2, 0, 0, 0)] + [(0, 1, 0, 0)] * 50 + [(3, 0, 0, 0), (1, 0, 0, 0)]
    source = chart(actions)
    model = BoundedModel(ModelConfig(arm, hidden=8, levels=3, coupling_rank=2, row_consequence=consequence)).double()
    if model.row_consequence is not None:
        torch.nn.init.normal_(model.row_consequence.output.weight, std=.05)
    item = SourceInterval(source, 32, 8)
    cropped = prepare_batch([item], arm, model.temporal.config.receptive_tokens)
    complete = prepare_batch([item], arm, 100)
    assert cropped.truncated[0] and not complete.truncated[0]
    assert cropped.states[0].replay.open_ln_start_ms[0] == 0.
    assert cropped.states[0].known_ends[0] == 51
    a, _ = batch_likelihood(model, cropped, candidate_budget=3)
    gradients_a = torch.autograd.grad(a, tuple(model.parameters()), allow_unused=True)
    b, _ = batch_likelihood(model, complete, candidate_budget=3)
    gradients_b = torch.autograd.grad(b, tuple(model.parameters()), allow_unused=True)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    for x, y in zip(gradients_a, gradients_b):
        if x is None:
            assert y is None
        else:
            torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)


def test_source_owner_rejects_lane_overlap_and_a_corrupt_admitted_cache(tmp_path):
    with pytest.raises(ContractError, match='occupancy'):
        chart([(2, 0, 0, 0), (1, 0, 0, 0), (3, 0, 0, 0), (0, 1, 0, 0)])
    source = chart([(1, 0, 0, 0)] * 31, minimum_seed_notes=30)
    raw = source.rows.tobytes()
    (tmp_path / 'rows.bin').write_bytes(raw)
    metadata = dict(format=SOURCE_FORMAT, identity=asdict(IDENTITY), row_count=31,
                    rows_sha256=hashlib.sha256(raw).hexdigest(), seed=dict(seed_row_count=30, ineligible_reason=None))
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    restored = SourceChart.from_cache(tmp_path, IDENTITY)
    assert restored.seed_rows == 30 and len(restored.onsets) == 1
    (tmp_path / 'rows.bin').write_bytes(raw[:-1] + b'\1')
    with pytest.raises(ContractError, match='digest'):
        SourceChart.from_cache(tmp_path, IDENTITY)
