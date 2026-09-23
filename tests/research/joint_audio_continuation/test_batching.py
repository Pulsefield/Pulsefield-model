from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.bounded_typed_continuation.data import SourceChart
from ensomi_model.research.joint_audio_continuation.batching import (
    batch_losses, collate, interpolate_audio, score_batch,
)
from ensomi_model.research.joint_audio_continuation.data import JointChart, query
from ensomi_model.research.joint_audio_continuation.model import JointAudioModel, JointModelConfig
from ensomi_model.research.joint_audio_continuation.state import exact_features
from ensomi_model.research.oracle_time_continuation.data import SourceIdentity
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


def chart(times, actions, duration_ms=1280):
    rows = np.zeros(len(times), ROW_DTYPE)
    rows['time'], rows['actions'] = times, actions
    identity = SourceIdentity('a' * 64, 'b' * 64, 'group', 'train')
    source = SourceChart(identity, rows, minimum_seed_notes=1)
    mel = np.random.default_rng(32).normal(size=((duration_ms + 9) // 10, 128)).astype(np.float32)
    return JointChart(asdict(identity), source, mel, duration_ms)


def config():
    return JointModelConfig(hidden=12, audio_width=8, audio_levels=2, history_levels=2,
                            expansion=2, coupling_rank=3, routing_hidden=20, release_hidden=24)


def test_absolute_bins_native_boundaries_targets_and_true_terminal_only():
    source = chart([0, 17, 28, 42], [(2, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (3, 0, 0, 0)], 42)
    queries = [query(source, 17, horizon_ms=25), query(source, -1, horizon_ms=1),
               query(source, 28, horizon_ms=1), query(source, 28, horizon_ms=14)]
    batch = collate(queries, [source] * len(queries), config())
    inputs, targets = batch.inputs, batch.targets
    assert inputs.raw.shape[:2] == (4, 7)
    assert inputs.timing_times_ms[0].tolist() == [19, 29, 39, 49]
    assert inputs.timing_valid[0].nonzero().flatten().tolist() == list(range(8, 33))
    assert inputs.timing_forced[0].nonzero().flatten().tolist() == [32]
    assert inputs.timing_valid[1].nonzero().flatten().tolist() == [0]
    assert inputs.timing_valid[2].nonzero().flatten().tolist() == [9]
    assert not inputs.timing_forced[2].any()
    assert targets.event_index.tolist() == [18, 0, -1, 22]
    assert targets.row_mask.tolist() == [True, True, False, True]
    assert targets.row_index.tolist() == [ROW_ACTIONS.index((0, 0, 0, 1)), ROW_ACTIONS.index((2, 0, 0, 0)),
                                         -1, ROW_ACTIONS.index((3, 0, 0, 0))]
    assert inputs.row_times_ms.tolist() == [28, 0, 28, 42]
    assert not inputs.row_legal[-1, ROW_ACTIONS.index((0, 1, 0, 0))]
    assert not inputs.row_legal[-1, ROW_ACTIONS.index((3, 2, 0, 0))]
    np.testing.assert_array_equal(inputs.row_exact.numpy(), exact_features([q.replay for q in queries],
                                                                        inputs.row_times_ms.tolist()))
    assert inputs.history_valid.sum(-1).tolist() == [2, 0, 3, 3]


def test_audio_interpolation_clock_clamping_and_gradients():
    encoded = torch.arange(5, dtype=torch.float64).reshape(1, 5, 1).requires_grad_()
    times = torch.tensor([[-100, 0, 20, 25, 45, 90]])
    values = interpolate_audio(encoded, times)
    torch.testing.assert_close(values[..., 0], torch.tensor([[0., 0., 0., .5, 2.5, 4.]], dtype=torch.float64))
    values.sum().backward()
    torch.testing.assert_close(encoded.grad.flatten(), torch.tensor([3.5, .5, .5, .5, 1.], dtype=torch.float64))
    virtual_crop = torch.cat((torch.zeros(1, 3, 1), encoded.detach(), torch.zeros(1, 2, 1)), 1)
    torch.testing.assert_close(values, interpolate_audio(virtual_crop, times, torch.tensor([-3]), torch.tensor([5])))
    torch.testing.assert_close(interpolate_audio(encoded, torch.tensor([25])), values[:, 3])
    with pytest.raises(ContractError, match='neighbors'):
        interpolate_audio(encoded[:, 1:3], torch.tensor([[20, 55]]), torch.tensor([1]), torch.tensor([5]))


def test_collated_crop_scores_match_complete_audio_at_both_song_edges():
    torch.manual_seed(84)
    model = JointAudioModel(config())
    source = chart([10, 430, 940, 1200], [(1, 0, 0, 0), (0, 2, 0, 0), (0, 3, 1, 0), (0, 0, 0, 1)])
    queries = [query(source, -1, horizon_ms=30), query(source, 430, horizon_ms=360),
               query(source, 940, horizon_ms=340), query(source, 1200, horizon_ms=80)]
    batch = collate(queries, [source] * len(queries), config())
    inputs = batch.inputs
    assert inputs.mel_starts[0] < 0
    assert not inputs.mel_valid[-1, -1]
    with torch.no_grad():
        model.set_audio_normalization(torch.linspace(-2., 2., 128), torch.linspace(.3, 2., 128))
        for block in model.audio_blocks:
            block.norm.bias.fill_(.4)
        scores = score_batch(model, inputs)
        full = model.encode_audio(torch.from_numpy(source.mel)[None].repeat(len(queries), 1, 1))
        history = model.encode_history(inputs.raw, inputs.history_valid, inputs.truncated)
        audio = interpolate_audio(full, inputs.timing_times_ms)
        expected_timing = model.timing_logits(audio, history, inputs.timing_exact).flatten(1)
        torch.testing.assert_close(scores.timing_logits, expected_timing, atol=2e-6, rtol=2e-6)
        rows = model.row_log_probs(interpolate_audio(full, inputs.row_times_ms), history, inputs.row_exact,
                                    inputs.row_legal, inputs.occupancy)
        torch.testing.assert_close(scores.row_log_probs, rows, atol=3e-6, rtol=3e-6)


def test_horizon_partition_preserves_shared_bin_logits_and_exact_queries():
    source = chart([0, 1000], [(2, 0, 0, 0), (3, 1, 0, 0)])
    early, later = query(source, 0, horizon_ms=127), query(source, 127, horizon_ms=200)
    full = query(source, 0, horizon_ms=327)
    model = JointAudioModel(config())
    with torch.no_grad():
        a = collate([early], [source], config())
        b = collate([later], [source], config())
        entire = collate([full], [source], config())
        aa, bb, all_scores = (score_batch(model, batch.inputs).timing_logits.reshape(-1, 10)
                              for batch in (a, b, entire))
        torch.testing.assert_close(aa, all_scores[:len(aa)], atol=2e-6, rtol=2e-6)
        first = int(b.inputs.timing_times_ms[0, 0] // 10)
        torch.testing.assert_close(bb, all_scores[first:first + len(bb)], atol=2e-6, rtol=2e-6)
        assert not a.inputs.timing_forced.any() and not b.inputs.timing_forced.any()


def test_censored_losses_are_zero_for_rows_and_joint_loss_trains_both_heads():
    source = chart([0, 30, 50], [(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0)], 50)
    queries = [query(source, 0, horizon_ms=10), query(source, 40, horizon_ms=10)]
    batch = collate(queries, [source, source], config())
    model = JointAudioModel(config())
    scores = score_batch(model, batch.inputs)
    losses = batch_losses(scores, batch)
    assert losses.row[0] == 0 and losses.row[1] > 0
    assert torch.isfinite(losses.total).all()
    losses.total.mean().backward()
    for parameter in (model.audio_input.weight, model.temporal.input.weight, model.timing[-1].weight,
                      model.joint.unary.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_target_time_does_not_change_timing_predictors_or_audio_crop():
    source = chart([0, 30, 50], [(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0)], 50)
    q = query(source, 0, horizon_ms=40)
    moved = replace(q, target_time_ms=39, target_actions=(0, 0, 1, 0))
    a, b = (collate([value], [source], config()) for value in (q, moved))
    for name in ('mel', 'mel_valid', 'mel_starts', 'mel_frame_counts', 'raw', 'history_valid', 'truncated',
                 'timing_times_ms', 'timing_exact', 'timing_valid', 'timing_forced', 'occupancy'):
        torch.testing.assert_close(getattr(a.inputs, name), getattr(b.inputs, name), atol=0, rtol=0)
    assert a.targets.event_index.item() != b.targets.event_index.item()


def test_empty_terminal_interval_and_truncated_history_preserve_masks():
    source = chart(list(range(10)), [(1, 0, 0, 0)] * 10, 10)
    q = query(source, 10, history_limit=7)
    batch = collate([q], [source], config())
    assert batch.inputs.truncated.tolist() == [True]
    assert batch.inputs.history_valid.sum() == 7
    assert not batch.inputs.timing_valid.any() and not batch.targets.row_mask.any()
    model = JointAudioModel(config())
    losses = batch_losses(score_batch(model, batch.inputs), batch)
    assert losses.total.item() == 0.
    with pytest.raises(ContractError, match='envelope'):
        collate([query(source, 10, history_limit=10)], [source], config())
