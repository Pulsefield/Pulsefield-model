from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.joint_audio_continuation.data import digest, frontend_identity
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation import training
from ensomi_model.research.planned_audio_continuation.counts import (
    COUNT_MARKS, ROW_COUNTS, RowCountModel, count_state,
)
from ensomi_model.research.planned_audio_continuation.generation import HeadPlanner, load_model, rollout
from ensomi_model.research.planned_audio_continuation.hydra import compose_config
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, interval_losses, score_interval
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel
from ensomi_model.research.planned_audio_continuation.profiles import build_profile_bank
from ensomi_model.research.planned_audio_continuation.session import ContinuationSession
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_distribution import chart, config, source
from .test_profiles import corpus


def settings(**kwargs):
    return replace(config(), history_levels=4, row_factorization='count_layout', **kwargs)


def mark_mass(model, rows):
    return rows.exp() @ model.row_counts.members.T.to(rows.dtype)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_exact_count_law_ignores_group_multiplicity_and_has_finite_inactive_gradients(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    head = RowCountModel(3, 2, 2).to(device)
    assert len(COUNT_MARKS) == 35 and len(ROW_COUNTS) == 256
    single, double = COUNT_MARKS.index((1, 0, 0)), COUNT_MARKS.index((2, 0, 0))
    legal = (head.members[single] | head.members[double])[None]
    scores = torch.zeros((1, 256), device=device, requires_grad=True)
    logits = torch.zeros((1, 35), device=device, requires_grad=True)
    rows = head.compose(scores, logits, legal)
    torch.testing.assert_close(rows.exp()[0, head.members[single]], torch.full((4,), .5 / 4, device=device))
    torch.testing.assert_close(rows.exp()[0, head.members[double]], torch.full((6,), .5 / 6, device=device))
    assert not rows.exp()[~legal].count_nonzero()
    (-rows[0, ROW_ACTIONS.index((1, 0, 0, 0))]).backward()
    assert torch.isfinite(scores.grad).all() and torch.isfinite(logits.grad).all()
    assert not scores.grad[0, ~head.members[single]].count_nonzero()
    assert logits.grad[0, single].item() == pytest.approx(-.5)
    assert logits.grad[0, double].item() == pytest.approx(.5)
    # R1 can redistribute layouts within a group, but its total group bias
    # cannot override the explicitly learned count marginal.
    torch.manual_seed(71)
    scores = torch.randn(3, 256, device=device)
    logits = torch.randn(3, 35, device=device)
    legal = torch.rand(3, 256, device=device) > .3
    legal[0, head.members[single]] = True
    original = head.compose(scores, logits, legal)
    shifted = head.compose(scores + torch.randn(3, 35, device=device)[:, head.row_mark], logits, legal)
    torch.testing.assert_close(original, shifted, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(original.exp().sum(-1), torch.ones(3, device=device))
    actual = original.exp() @ head.members.T.to(original.dtype)
    active = (legal[:, None] & head.members).any(-1)
    torch.testing.assert_close(actual, logits.masked_fill(~active, -torch.inf).softmax(-1))
    group = torch.where(legal[0] & head.members[single])[0]
    a, b = group[:2]
    torch.testing.assert_close(original[0, a] - original[0, b], scores[0, a] - scores[0, b])


def test_count_inputs_exclude_layout_totals_and_future_tails_but_read_active_hold_ages():
    a = chart([0, 7, 9, 30], [(1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 1), (0, 3, 0, 0)], 50)
    b = chart([0, 7, 9, 30], [(0, 0, 0, 1), (0, 2, 0, 0), (1, 0, 0, 0), (0, 3, 0, 0)], 50)
    model = PlannedAudioModel(settings()).eval()
    coarse = model.encode_coarse(torch.from_numpy(a.mel)[None])
    batches = [collate_interval(IntervalExample(c, 0, 100), model.config) for c in (a, b)]
    torch.testing.assert_close(batches[0].inputs.count_raw, batches[1].inputs.count_raw, rtol=0, atol=0)
    torch.testing.assert_close(batches[0].inputs.count_clock, batches[1].inputs.count_clock, rtol=0, atol=0)
    scores = [score_interval(model, q.inputs, coarse) for q in batches]
    torch.testing.assert_close(mark_mass(model, scores[0].row), mark_mass(model, scores[1].row), atol=2e-6, rtol=2e-6)
    finite = torch.isfinite(scores[0].row) & torch.isfinite(scores[1].row)
    assert (scores[0].row[finite] - scores[1].row[finite]).abs().max() > 1e-5
    shifted = chart([0, 7, 9, 40], a.source.rows['actions'], 50)
    early = [collate_interval(IntervalExample(c, 0, 20), model.config) for c in (a, shifted)]
    left, right = [score_interval(model, q.inputs, coarse) for q in early]
    for field in ('head', 'release', 'row'):
        torch.testing.assert_close(getattr(left, field), getattr(right, field), atol=0, rtol=0)
    state_a = count_state([[0, None, 10, None]], [11], [20])
    state_b = count_state([[None, 10, None, 0]], [11], [20])
    np.testing.assert_array_equal(state_a, state_b)
    changed_age = count_state([[1, None, 10, None]], [11], [20])
    audio = torch.zeros(1, model.config.conditioned_audio_width)
    history = model.row_counts.temporal.read(model.row_counts.temporal.empty_cache())[None]
    preview = batches[0].inputs.row_preview[:1]
    first = model.row_counts.logits(audio, history, preview, torch.from_numpy(state_a))
    second = model.row_counts.logits(audio, history, preview, torch.from_numpy(changed_age))
    assert (first - second).abs().max() > 1e-6


def test_mirror_and_interval_partitions_preserve_the_complete_count_layout_law():
    original = source()
    reverse = chart(original.source.rows['time'], original.source.rows['actions'][:, ::-1], original.duration_ms)
    model = PlannedAudioModel(settings()).eval()
    coarse = model.encode_coarse(torch.from_numpy(original.mel)[None])
    batches = [collate_interval(IntervalExample(c, 0, 20), model.config) for c in (original, reverse)]
    scores = [score_interval(model, q.inputs, coarse) for q in batches]
    mirror = [ROW_ACTIONS.index(row[::-1]) for row in ROW_ACTIONS]
    torch.testing.assert_close(scores[0].row, scores[1].row[:, mirror], atol=3e-6, rtol=3e-6)
    expected = torch.stack(interval_losses(scores[0], batches[0]))
    parts = []
    for i in range(IntervalExample(original, 0, 3).count):
        batch = collate_interval(IntervalExample(original, i, 3), model.config)
        parts.append(torch.stack(interval_losses(score_interval(model, batch.inputs, coarse), batch)))
    torch.testing.assert_close(sum(parts), expected, atol=3e-5, rtol=3e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_count_model_cpu_mps_joint_losses_and_gradients_agree():
    cpu = PlannedAudioModel(settings(bounded_head=True, condition_full_holds=True))
    mps = PlannedAudioModel(cpu.config).to('mps')
    mps.load_state_dict(cpu.state_dict())
    values, gradients = [], []
    for device, model in (('cpu', cpu), ('mps', mps)):
        c = source()
        batch = collate_interval(IntervalExample(c, 0, 20), model.config, device)
        coarse = model.encode_coarse(torch.as_tensor(c.mel, device=device)[None])
        losses = torch.stack(interval_losses(score_interval(model, batch.inputs, coarse), batch))
        losses[-1].backward()
        values.append(losses.detach().cpu())
        gradients.append({n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None})
    torch.testing.assert_close(values[0], values[1], atol=3e-4, rtol=2e-5)
    assert gradients[0]['row_counts.temporal.input.weight'].abs().sum() > 0
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        assert torch.isfinite(gradients[0][name]).all() and torch.isfinite(gradients[1][name]).all()
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=3e-4, rtol=3e-4)


def test_cached_count_history_matches_teacher_and_forks_keep_owned_future():
    torch.manual_seed(77)
    model = PlannedAudioModel(settings(bounded_head=True, condition_full_holds=True)).eval()
    mel = np.random.default_rng(23).normal(size=(40, 128)).astype(np.float32)
    heads = list(range(0, 381, 10))
    original, recorded = model.planned_row_log_probs, []

    def observe(*args, **kwargs):
        value = original(*args, **kwargs)
        recorded.append(value.detach().clone())
        return value

    model.planned_row_log_probs = observe
    generated = rollout(model, mel, 400, seed=17, head_times=heads, chunk_ms=11)
    model.planned_row_log_probs = original
    assert generated.completed and len(generated.rows) > 31
    alternative = rollout(model, mel, 400, seed=17, head_times=heads, chunk_ms=31)
    assert alternative.rows == generated.rows
    c = chart([r.time_ms for r in generated.rows], [r.actions for r in generated.rows], 400)
    c = replace(c, mel=mel)
    batch = collate_interval(IntervalExample(c, 0, 401), model.config)
    with torch.no_grad():
        teacher = score_interval(model, batch.inputs, model.encode_coarse(torch.from_numpy(mel)[None]))
    assert len(batch.row_index) == len(generated.rows)
    torch.testing.assert_close(torch.cat(recorded), teacher.row[:len(batch.row_index)], atol=2e-5, rtol=2e-5)
    session = ContinuationSession(model, mel, 400, seed=17, planner_factory=HeadPlanner, head_times=heads)
    session.step()
    before = model.row_counts.temporal.read(session.count_cache).clone()
    fork = session.fork()
    while len(fork.rows) == len(session.rows):
        fork.step()
    torch.testing.assert_close(model.row_counts.temporal.read(session.count_cache), before, atol=0, rtol=0)
    assert len(fork.rows) > len(session.rows)


def test_hydra_warm_start_and_actual_training_freeze_head_audio_and_restore_count_endpoint(tmp_path, monkeypatch):
    charts = corpus()
    bank = build_profile_bank(charts, 2)
    source_model = PlannedAudioModel(replace(settings(bounded_head=True, condition_full_holds=True),
                                             row_factorization='flat', profile_count=2))
    source_model.configure_profiles(bank)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}\n')
    norm = dict(mean=[0.] * 128, std=[1.] * 128, frontend=frontend_identity(), frame_count=40,
                audio_sha256=sorted({c.entry['audio_sha256'] for c in charts if c.split == 'train'}))
    norm_file, bank_file, path = tmp_path / 'normalization.json', tmp_path / 'bank.json', tmp_path / 'initial.pt'
    norm_file.write_text(json.dumps(norm))
    bank_file.write_text(json.dumps(bank))
    source_config = asdict(source_model.config)
    source_config.pop('row_factorization')  # Old checkpoints receive the flat default.
    torch.save(dict(format='joint-audio/planned-profile-v1', model_config=source_config,
        model=source_model.state_dict(), source_revision='f'*40, manifest_sha256=digest(manifest), config={}), path)
    cfg = compose_config(['row_factorization=count_layout', 'train_scope=materializer',
        f'initial_checkpoint_file={path}', f'initial_checkpoint_sha256={digest(path)}'])
    assert cfg.row_factorization == 'count_layout' and cfg.train_scope == 'materializer'
    for override in ('row_factorization=unused', 'train_scope=unused', 'train_scope=materializer'):
        with pytest.raises(ValueError):
            compose_config([override])
    cfg = replace(cfg, root=str(tmp_path), manifest_sha256=digest(manifest),
        normalization_file=str(norm_file), normalization_sha256=digest(norm_file),
        profile_bank_file=str(bank_file), profile_bank_sha256=digest(bank_file),
        plan_updates=2, updates=2, songs_per_update=1, intervals_per_song=1,
        interval_ms=201, validation_every=2, device='cpu', max_seconds=60,
        bounded_head=True, condition_full_holds=True)
    target = PlannedAudioModel(replace(source_model.config, row_factorization='count_layout'))
    target.configure_profiles(bank)
    receipt = training.initialize_from_planned(target, cfg, norm)
    assert set(receipt['new']) == {n for n in target.state_dict() if n.startswith('row_counts.')}
    for name, value in source_model.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)
    scope = training.configure_train_scope(target, 'materializer')
    groups = training.optimizer_groups(target, cfg)
    names = {n for group in groups for n in group['param_names']}
    assert not names.intersection(scope['frozen_parameters'])
    assert 'row_counts.readout.0.weight' in names and 'row_consequence.output.weight' in names
    assert {'head_base.weight', 'profile_condition.weight', 'audio_input.weight'}.issubset(scope['frozen_parameters'])
    monkeypatch.setattr(training, 'revision', lambda: 'e'*40)
    monkeypatch.setattr(training, 'load_corpus', lambda root: (charts, norm))
    monkeypatch.setattr(training, '_resource_stop', lambda root: None)

    def small_model(settings):
        return PlannedAudioModel(replace(source_model.config, row_factorization=settings.row_factorization))

    monkeypatch.setattr(training, 'PlannedAudioModel', small_model)
    result = training.train(cfg)
    assert result['status'] == 'completed' and result['parameter_update_l2']['row_counts'] > 0
    directory = tmp_path / 'planned-training' / cfg.run_name
    restored, metadata = load_model(directory / 'last.pt', result['checkpoint_sha256'])
    assert restored.config.row_factorization == 'count_layout'
    assert metadata['config']['train_scope'] == 'materializer'
    for name in result['parameter_scope']['frozen_parameters']:
        torch.testing.assert_close(restored.state_dict()[name], source_model.state_dict()[name], atol=0, rtol=0)
    generated = rollout(restored, charts[0].mel, charts[0].duration_ms, seed=17, arrangement_profile=0)
    assert generated.completed
    with pytest.raises(ContractError, match='31 rows'):
        PlannedAudioModel(replace(config(), row_factorization='count_layout'))
