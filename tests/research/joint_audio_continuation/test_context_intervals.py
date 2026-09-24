from dataclasses import asdict, replace
import math

import numpy as np
import pytest
import torch

from ensomi_model.research.joint_audio_continuation.batching import collate, score_batch, batch_losses, interpolate_audio
from ensomi_model.research.joint_audio_continuation.context_model import ContextAudioModel, ContextModelConfig
from ensomi_model.research.joint_audio_continuation.context_evaluation import score_context_queries
from ensomi_model.research.joint_audio_continuation.data import query
from ensomi_model.research.joint_audio_continuation.intervals import (
    IntervalExample, IntervalScores, collate_interval, score_interval, interval_losses,
)
from ensomi_model.research.joint_audio_continuation.model import JointAudioModel
from ensomi_model.research.joint_audio_continuation.state import exact_features
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_batching import chart, config


def context_config(**kwargs):
    return ContextModelConfig(**asdict(config()), context_width=12, context_heads=2, **kwargs)


def test_local_fused_has_identical_parameters_and_scores_to_existing_model():
    torch.manual_seed(5)
    old = JointAudioModel(config())
    torch.manual_seed(5)
    new = ContextAudioModel(context_config())
    assert set(old.state_dict()) == set(new.state_dict())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    source = chart([0, 17, 400], [(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0)])
    q = query(source, 17, history_limit=7, horizon_ms=500)
    batch = collate([q], [source], config())
    a, b = score_batch(old, batch.inputs), score_batch(new, batch.inputs)
    torch.testing.assert_close(a.timing_logits, b.timing_logits, rtol=0, atol=0)
    torch.testing.assert_close(a.row_log_probs, b.row_log_probs, rtol=0, atol=0)


def test_shared_coarse_initialization_does_not_depend_on_timing_factor():
    torch.manual_seed(16)
    a = ContextAudioModel(context_config(global_audio=True))
    torch.manual_seed(16)
    b = ContextAudioModel(context_config(global_audio=True, bounded_timing=True))
    for name, value in a.context.state_dict().items():
        torch.testing.assert_close(value, b.context.state_dict()[name], rtol=0, atol=0)


def test_full_context_padding_crop_clocks_and_joint_gradient():
    torch.manual_seed(18)
    model = ContextAudioModel(context_config(global_audio=True))
    source = chart([0, 417, 901], [(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0)], 1321)
    mel = torch.from_numpy(source.mel)[None]
    valid = torch.ones(mel.shape[:2], dtype=torch.bool)
    coarse = model.encode_coarse(mel, valid)
    padded = torch.nn.functional.pad(mel, (0, 0, 0, 101))
    masked = torch.nn.functional.pad(valid, (0, 101))
    padded_coarse = model.encode_coarse(padded, masked)
    assert coarse[1].tolist() == [3]
    torch.testing.assert_close(coarse[0], padded_coarse[0][:, :3], atol=2e-6, rtol=2e-6)
    full = model.encode_audio(mel)
    for index in range(3):
        batch = collate_interval(IntervalExample(source, index, 500), model.config)
        x = batch.inputs
        crop = model.encode_crop(x.mel, x.mel_valid, x.mel_start, x.frame_count, coarse)
        torch.testing.assert_close(interpolate_audio(crop, x.timing_times[None], x.mel_start, x.frame_count),
                                   interpolate_audio(full, x.timing_times[None]), atol=3e-6, rtol=3e-6)
    with pytest.raises(ContractError, match='full-song'):
        model.encode_crop(x.mel, x.mel_valid, x.mel_start, x.frame_count)
    # Zero output projections deliberately delay coarse gradients initially;
    # open them to verify the complete supervised path reaches audio weights.
    with torch.no_grad():
        model.context_condition.weight.normal_(std=.02)
        model.context_timing.weight.normal_(std=.02)
    batch = collate_interval(IntervalExample(source, 0, 500), model.config)
    loss = interval_losses(score_interval(model, batch.inputs, coarse), batch)[2]
    loss.backward()
    assert model.context.reduce.weight.grad.abs().sum() > 0
    assert model.context.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.audio_input.weight.grad.abs().sum() > 0


def test_hold_only_base_and_bounded_decay_keep_physical_obligations():
    model = ContextAudioModel(context_config(bounded_timing=True))
    free = commit(ExactReplayState(), CompleteRow(1000, (1, 0, 0, 0)))
    other = commit(commit(ExactReplayState(), CompleteRow(0, (0, 1, 0, 0))), CompleteRow(2000, (0, 0, 1, 0)))
    held = commit(ExactReplayState(), CompleteRow(1000, (2, 0, 0, 0)))
    exact = torch.from_numpy(exact_features([ExactReplayState(), free, other, held], [5000] * 4))[:, None]
    audio = torch.randn(1, 1, model.config.audio_width).expand(4, -1, -1)
    history = torch.randn(4, 2, model.config.hidden)
    base, residual, gate = model.timing_parts(audio, history, exact)
    torch.testing.assert_close(base[0], base[1], rtol=0, atol=0)
    torch.testing.assert_close(base[1], base[2], rtol=0, atol=0)
    torch.testing.assert_close(gate[:, 0], torch.tensor([0., math.exp(-4), math.exp(-3), 1.]))
    assert bool((residual.abs() <= 4 * gate[..., None]).all())
    changed = history * 100
    torch.testing.assert_close(model.timing_parts(audio, changed, exact)[0], base, rtol=0, atol=0)


@pytest.mark.parametrize('width', [333, 500, 1281])
def test_interval_likelihood_matches_independent_single_queries_across_events_and_holds(width):
    source = chart([0, 17, 38, 90, 101, 199, 321, 417, 502, 701, 900, 1000],
        [(2, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
         (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 1, 0, 0),
         (0, 0, 1, 0), (0, 0, 0, 1), (0, 1, 0, 0), (3, 0, 0, 0)], 1280)
    model = ContextAudioModel(context_config())
    for index in range(IntervalExample(source, 0, width).count):
        example = IntervalExample(source, index, width)
        batch = collate_interval(example, model.config)
        actual = interval_losses(score_interval(model, batch.inputs), batch)
        cursor = example.start_ms - 1
        expected = torch.zeros(2)
        while cursor < example.end_ms - 1:
            q = query(source, cursor, history_limit=7, horizon_ms=example.end_ms - 1 - cursor)
            end = q.target_time_ms if q.target_time_ms is not None else q.horizon_end_ms
            q = query(source, cursor, history_limit=7, horizon_ms=end - cursor)
            b = collate([q], [source], model.config)
            losses = batch_losses(score_batch(model, b.inputs), b)
            expected += torch.stack((losses.timing.sum(), losses.row.sum()))
            cursor = end
        torch.testing.assert_close(torch.stack(actual[:2]), expected, atol=2e-5, rtol=2e-6)


def test_uniform_partition_weights_include_empty_and_short_final_intervals():
    source = chart([0, 17], [(1, 0, 0, 0), (0, 1, 0, 0)], 1020)
    weighted, milliseconds = [], []
    for index in range(3):
        ex = IntervalExample(source, index, 500)
        batch = collate_interval(ex, context_config())
        logits = torch.full(batch.inputs.timing_valid.shape, math.log(.2 / .8))
        rows = torch.full((len(batch.targets.row_index), 256), -math.log(4))
        loss = interval_losses(IntervalScores(logits, rows), batch)[2]
        weighted.append(loss * ex.weight_per_second)
        milliseconds.append(int(batch.inputs.timing_valid.sum()))
        if index:
            assert not batch.targets.row_index.numel()
    assert milliseconds == [500, 500, 21]
    analytic = (-2 * math.log(.2) - 1019 * math.log(.8) + 2 * math.log(4)) * 1000 / 1021
    torch.testing.assert_close(torch.stack(weighted).mean(), torch.tensor(analytic))


def test_interval_causal_indices_exclude_current_and_future_target_content():
    source = chart([10, 17, 40], [(1, 0, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0)], 80)
    model = ContextAudioModel(context_config())
    batch = collate_interval(IntervalExample(source, 0, 100), model.config)
    assert batch.inputs.row_history[:3].tolist() == [-1, 0, 1]
    assert batch.inputs.history_valid.sum() == 3
    assert batch.inputs.row_times.shape == (64,) and batch.targets.row_index.shape == (3,)
    original = score_interval(model, batch.inputs)
    raw = batch.inputs.raw.clone()
    raw[:, 1:] = torch.randn_like(raw[:, 1:]) * 20
    modified = score_interval(model, replace(batch.inputs, raw=raw))
    torch.testing.assert_close(original.row_log_probs[:2], modified.row_log_probs[:2], rtol=0, atol=0)
    early = batch.inputs.timing_history < 1
    torch.testing.assert_close(original.timing_logits[early], modified.timing_logits[early], rtol=0, atol=0)


def test_global_query_evaluation_requires_full_song_and_matches_native_cache():
    source = chart([0, 300, 700], [(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0)], 900)
    model = ContextAudioModel(context_config(global_audio=True, bounded_timing=True))
    with torch.no_grad():
        model.context_base.weight.normal_(std=.1)
    q = query(source, 300, history_limit=7, horizon_ms=250)
    batch = collate([q], [source], model.config)
    with pytest.raises(ContractError, match='full-song'):
        score_batch(model, batch.inputs)
    coarse = model.encode_coarse(torch.from_numpy(source.mel)[None])
    scores = score_context_queries(model, batch.inputs, coarse)
    full = model.encode_audio(torch.from_numpy(source.mel)[None])
    x = batch.inputs
    history = model.encode_history(x.raw, x.history_valid, x.truncated)
    expected = model.timing_logits(interpolate_audio(full, x.timing_times_ms), history, x.timing_exact).flatten(1)
    torch.testing.assert_close(scores.timing_logits, expected, atol=2e-6, rtol=2e-6)
