from dataclasses import asdict, replace
import math

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.bounded_typed_continuation.data import SourceChart
from ensomi_model.research.bounded_typed_continuation.features import TIME_DIM, time_features
from ensomi_model.research.joint_audio_continuation.data import JointChart
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.joint_audio_continuation.model import JointModelConfig
from ensomi_model.research.oracle_time_continuation.data import SourceIdentity
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE
from ensomi_model.research.planned_audio_continuation.features import (
    HeadPreview, LNProjection, consequences, preview_after, release_masks, row_support,
)
from ensomi_model.research.planned_audio_continuation.generation import rollout
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, interval_losses, score_interval
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel, PlannedModelConfig
from ensomi_model.research.planned_audio_continuation.release import conditioned_release_logits
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


def chart(times, actions, duration_ms):
    rows = np.zeros(len(times), ROW_DTYPE)
    rows['time'], rows['actions'] = times, actions
    identity = SourceIdentity('a' * 64, 'b' * 64, 'group', 'train')
    source = SourceChart(identity, rows, minimum_seed_notes=1)
    mel = np.random.default_rng(32).normal(size=((duration_ms + 9) // 10, 128)).astype(np.float32)
    return JointChart(asdict(identity), source, mel, duration_ms)


def config():
    row = JointModelConfig(hidden=12, audio_width=8, audio_levels=2, history_levels=2,
                           expansion=2, coupling_rank=3, routing_hidden=20, release_hidden=24)
    return PlannedModelConfig(**asdict(row), context_width=12, context_heads=2, context_layers=1,
                              head_hidden=8, head_levels=2, skeleton_hidden=8, skeleton_levels=2, lookahead=2)


def source():
    return chart([0, 4, 5, 10], [(2, 2, 2, 2), (3, 0, 0, 0), (1, 0, 0, 0), (0, 3, 3, 3)], 10)


@pytest.mark.parametrize('bounded', [False, True])
def test_joint_law_counts_head_survival_and_release_deadline_atoms_independently(bounded):
    model = PlannedAudioModel(replace(config(), bounded_head=bounded))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        (model.head_base if bounded else model.timing[-1]).bias.fill_(math.log(.2 / .8))
        model.release_clock[-1].bias.fill_(math.log(.3 / .7))
    c = source()
    coarse = model.encode_coarse(torch.from_numpy(c.mel)[None])
    batch = collate_interval(IntervalExample(c, 0, 20), model.config)
    h, r, row, joint = interval_losses(score_interval(model, batch.inputs, coarse), batch)
    assert batch.head_event.sum() == 2 and batch.release_event.sum() == 2
    assert batch.inputs.base.timing_valid.sum() == 11
    assert batch.inputs.release_valid.sum() == 9
    assert batch.inputs.release_forced.sum() == 2
    expected = torch.tensor([-2 * math.log(.2) - 9 * math.log(.8), -7 * math.log(.7),
                             math.log(80) + math.log(15) + math.log(16)])
    torch.testing.assert_close(torch.stack((h, r, row)), expected)
    torch.testing.assert_close(joint, expected.sum())
    joint.backward()
    assert torch.isfinite(model.timing[-1].bias.grad).all()
    assert torch.isfinite(model.release_clock[-1].bias.grad).all()


def test_support_preserves_native_millisecond_spacing_and_required_head_feasibility():
    # One-ms spacing remains representable; the only removed choices would
    # occupy every lane without a possible intervening release clock.
    state = ExactReplayState()
    next_head = HeadPreview((1, 7), False)
    support = row_support([state], [0], [True], [next_head], 10)[0]
    assert not support[ROW_ACTIONS.index((2, 2, 2, 2))]
    assert support[ROW_ACTIONS.index((2, 2, 2, 1))]
    held = commit(state, CompleteRow(0, (2, 2, 2, 2)))
    projection = LNProjection(held.open_ln_start_ms, 0)
    valid, forced = release_masks([projection], np.arange(1, 8)[None], [HeadPreview((5, 7), False)], 10)
    assert valid.tolist() == [[True, True, True, True, False, False, False]]
    assert forced.tolist() == [[False, False, False, True, False, False, False]]
    with pytest.raises(ContractError, match='no release clock'):
        release_masks([projection], np.array([[1]]), [next_head], 10)
    with pytest.raises(ContractError, match='next head'):
        release_masks([projection], np.array([[1]]), [HeadPreview((), False)], 10)
    # A scheduler crop at 3 is not the deadline at 4.
    _, forced = release_masks([projection], np.array([[1, 2, 3]]), [HeadPreview((5,), True)], 10)
    assert not forced.any()


def test_candidate_features_match_immediate_replay_and_distinguish_tap_from_hold():
    state = commit(ExactReplayState(), CompleteRow(10, (2, 1, 0, 0)))
    state = commit(state, CompleteRow(20, (0, 0, 1, 0)))
    now, next_h = 30, 41
    preview = HeadPreview((next_h, 90), False)
    local, timing = consequences([state], [now], [preview], 100)
    for row in ROW_ACTIONS:
        if not row_support([state], [now], [any(a in (1, 2) for a in row)], [preview], 100)[0, ROW_ACTIONS.index(row)]:
            continue
        after = commit(state, CompleteRow(now, row))
        clocks = after.clocks_at(next_h)
        for lane, action in enumerate(row):
            assert local[0, lane, action, 4] == after.occupancy[lane]
            expected = [clocks.lane_attack_ms[lane], clocks.lane_release_ms[lane], clocks.ln_age_ms[lane],
                        next_h - (now + 1) if after.occupancy[lane] else None]
            np.testing.assert_array_equal(local[0, lane, action, 5 + 3 * TIME_DIM:],
                                          time_features(expected).reshape(-1))
    assert local[0, 1, 1, 4] == 0 and local[0, 1, 2, 4] == 1
    np.testing.assert_array_equal(timing[0, :TIME_DIM], time_features(1))


def test_tap_materialization_cannot_change_either_skeleton_distribution():
    a = chart([0, 7, 9, 30], [(1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 1), (0, 3, 0, 0)], 50)
    b = chart([0, 7, 9, 30], [(0, 0, 0, 1), (0, 2, 0, 0), (1, 0, 0, 0), (0, 3, 0, 0)], 50)
    model = PlannedAudioModel(config())
    coarse = model.encode_coarse(torch.from_numpy(a.mel)[None])
    ba, bb = [collate_interval(IntervalExample(c, 0, 100), model.config) for c in (a, b)]
    sa, sb = [score_interval(model, x.inputs, coarse) for x in (ba, bb)]
    torch.testing.assert_close(sa.head, sb.head, rtol=0, atol=0)
    torch.testing.assert_close(sa.release, sb.release, rtol=0, atol=0)
    assert not torch.equal(ba.inputs.base.raw, bb.inputs.base.raw)
    valid = torch.isfinite(sa.row) & torch.isfinite(sb.row)
    assert float((sa.row[valid] - sb.row[valid]).abs().max().detach()) > 1e-5
    # Future actual tails never enter an earlier query or candidate score.
    shifted = chart([0, 7, 9, 40], [(1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 1), (0, 3, 0, 0)], 50)
    x, y = [collate_interval(IntervalExample(c, 0, 20), model.config) for c in (a, shifted)]
    sx, sy = [score_interval(model, q.inputs, coarse) for q in (x, y)]
    for first, second in zip(vars(sx).values(), vars(sy).values()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize('width', [7, 17, 51])
def test_partitioned_training_has_same_full_audio_joint_probability(width):
    c = chart([0, 4, 5, 10, 17, 19, 40], [(2, 2, 2, 2), (3, 0, 0, 0), (1, 0, 0, 0),
        (0, 3, 3, 3), (0, 2, 0, 1), (1, 0, 0, 0), (0, 3, 0, 0)], 50)
    model = PlannedAudioModel(config())
    coarse = model.encode_coarse(torch.from_numpy(c.mel)[None])
    full = collate_interval(IntervalExample(c, 0, 51), model.config)
    expected = torch.stack(interval_losses(score_interval(model, full.inputs, coarse), full))
    actual = expected * 0
    for index in range(IntervalExample(c, 0, width).count):
        batch = collate_interval(IntervalExample(c, index, width), model.config)
        actual = actual + torch.stack(interval_losses(score_interval(model, batch.inputs, coarse), batch))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)


def test_row_audio_preview_and_consequence_receive_joint_gradients():
    model = PlannedAudioModel(config())
    torch.nn.init.normal_(model.row_consequence.output.weight, std=.02)
    c = source()
    coarse = model.encode_coarse(torch.from_numpy(c.mel)[None])
    batch = collate_interval(IntervalExample(c, 0, 20), model.config)
    loss = interval_losses(score_interval(model, batch.inputs, coarse), batch)
    loss[2].backward()
    for parameter in (model.audio_residual.weight, model.context_condition.weight, model.preview_condition.weight,
                      model.row_consequence.output.weight, model.row_consequence.lanes[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_mirrored_charts_have_the_same_skeleton_and_mirrored_row_probabilities():
    torch.manual_seed(27)
    model = PlannedAudioModel(config()).eval()
    torch.nn.init.normal_(model.row_consequence.output.weight, std=.05)
    original = source()
    mirrored = chart(original.source.rows['time'], original.source.rows['actions'][:, ::-1], original.duration_ms)
    coarse = model.encode_coarse(torch.from_numpy(original.mel)[None])
    scores = []
    for c in (original, mirrored):
        batch = collate_interval(IntervalExample(c, 0, 20), model.config)
        scores.append(score_interval(model, batch.inputs, coarse))
    reverse = [ROW_ACTIONS.index(row[::-1]) for row in ROW_ACTIONS]
    torch.testing.assert_close(scores[0].head, scores[1].head, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(scores[0].release, scores[1].release, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(scores[0].row, scores[1].row[:, reverse], atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('conditioned', [False, True])
def test_mps_joint_distribution_and_gradients_match_cpu(bounded, conditioned):
    torch.manual_seed(52)
    settings = replace(config(), bounded_head=bounded, condition_full_holds=conditioned)
    cpu = PlannedAudioModel(settings)
    torch.nn.init.normal_(cpu.row_consequence.output.weight, std=.05)
    mps = PlannedAudioModel(settings).to('mps')
    mps.load_state_dict(cpu.state_dict())
    c = source()
    losses, gradients = [], []
    for device, model in (('cpu', cpu), ('mps', mps)):
        batch = collate_interval(IntervalExample(c, 0, 20), model.config, device)
        coarse = model.encode_coarse(torch.from_numpy(c.mel)[None].to(device))
        loss = torch.stack(interval_losses(score_interval(model, batch.inputs, coarse), batch))
        loss[-1].backward()
        losses.append(loss.detach().cpu())
        gradients.append({n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None})
    torch.testing.assert_close(losses[0], losses[1], atol=2e-4, rtol=1e-5)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('conditioned', [False, True])
def test_cached_native_scores_match_teacher_scores_and_chunk_partition_preserves_draws(bounded, conditioned):
    torch.manual_seed(113)
    model = PlannedAudioModel(replace(config(), bounded_head=bounded, condition_full_holds=conditioned)).eval()
    with torch.no_grad():
        (model.head_base if bounded else model.timing[-1]).bias.fill_(math.log(.05 / .95))
        model.release_clock[-1].bias.fill_(math.log(.08 / .92))
        model.row_consequence.output.weight.normal_(std=.05)
    mel = np.random.default_rng(32).normal(size=(15, 128)).astype(np.float32)
    if conditioned:
        row_base = model.planned_row_log_probs

        def favor_full_holds(*args):
            scores = row_base(*args)
            bias = torch.zeros(256)
            bias[ROW_ACTIONS.index((2, 2, 2, 2))] = 30
            return (scores + bias).log_softmax(-1)

        model.planned_row_log_probs = favor_full_holds
    original_head, original_release, original_row = model.head_logits, model.release_logits, model.planned_row_log_probs
    head_records, release_records, row_records = {}, {}, []

    def tracked(function, storage):
        def run(audio, history, clocks):
            result = function(audio, history, clocks)
            for key, value in zip(clocks.detach().cpu().numpy(), result.detach().cpu()):
                storage[key.tobytes()] = value
            return result
        return run

    model.head_logits = tracked(original_head, head_records)
    model.release_logits = tracked(original_release, release_records)

    def rows(*args):
        result = original_row(*args)
        row_records.append(result.detach().cpu())
        return result

    model.planned_row_log_probs = rows
    a = rollout(model, mel, 150, seed=17, chunk_ms=23, head_chunk_ms=31, max_seconds=20)
    model.head_logits, model.release_logits, model.planned_row_log_probs = original_head, original_release, original_row
    assert a.completed and a.rows and not any(a.metrics['open_lanes'])
    if conditioned:
        assert a.metrics['conditioned_release_waits'] > 0
    b = rollout(model, mel, 150, seed=17, chunk_ms=17, head_chunk_ms=7, max_seconds=20)
    assert a.rows == b.rows
    generated = chart([r.time_ms for r in a.rows], [r.actions for r in a.rows], 150)
    batch = collate_interval(IntervalExample(generated, 0, 200), model.config)
    with torch.no_grad():
        coarse = model.encode_coarse(torch.from_numpy(mel)[None])
        scores = score_interval(model, batch.inputs, coarse)
    torch.testing.assert_close(scores.row[:len(a.rows)], torch.cat(row_records), atol=2e-5, rtol=2e-6)
    head_clocks_array = batch.inputs.head_clock.cpu().numpy()
    for i in torch.where(batch.inputs.base.timing_valid.any(-1))[0]:
        torch.testing.assert_close(scores.head[i], head_records[head_clocks_array[i].tobytes()], atol=2e-5, rtol=2e-6)
    release_clocks_array = batch.inputs.release_clock.cpu().numpy()
    ordinary = batch.inputs.release_valid.clone().flatten()
    for wait in batch.inputs.release_waits:
        ordinary[wait.destinations] = False
        native_raw = torch.stack([release_records[t.tobytes()] for t in wait.clocks.cpu().numpy()])
        expected = conditioned_release_logits(native_raw.flatten()[wait.native_indices])[wait.offsets]
        torch.testing.assert_close(scores.release.flatten()[wait.destinations], expected, atol=2e-5, rtol=2e-6)
    ordinary = ordinary.reshape_as(batch.inputs.release_valid)
    for i in torch.where(batch.inputs.release_valid.any(-1))[0]:
        mask = ordinary[i]
        if mask.any():
            torch.testing.assert_close(scores.release[i][mask], release_records[release_clocks_array[i].tobytes()][mask],
                                       atol=2e-5, rtol=2e-6)
    if conditioned:
        model.config = replace(model.config, condition_full_holds=False)
        baseline = rollout(model, mel, 150, seed=17, max_seconds=20)
        assert baseline.completed
        heads = lambda rows: [r.time_ms for r in rows if any(a in (1, 2) for a in r.actions)]
        assert heads(a.rows) == heads(baseline.rows)
