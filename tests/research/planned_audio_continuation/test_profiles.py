from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from ensomi_model.research.joint_audio_continuation.batching import interpolate_audio
from ensomi_model.research.joint_audio_continuation.data import digest, frontend_identity
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation import training
from ensomi_model.research.planned_audio_continuation.config import PlannedTrainConfig
from ensomi_model.research.planned_audio_continuation.generation import load_model, rollout
from ensomi_model.research.planned_audio_continuation.hydra import compose_config
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, interval_losses, score_interval
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel, PlannedModelConfig
from ensomi_model.research.planned_audio_continuation.profiles import (
    arrangement_values, build_profile_bank, chart_assignment, normalized,
)
from .test_distribution import chart, config


def corpus():
    specifications = [
        ([0, 20, 40, 60], [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]),
        ([0, 30, 60, 90], [(2,0,0,0), (0,1,0,0), (3,0,1,0), (1,1,0,0)]),
        ([0, 50, 100, 150], [(2,2,0,0), (3,3,0,0), (0,0,1,1), (1,0,0,1)]),
        ([0, 50, 90, 110], [(1,1,1,1), (2,2,2,2), (3,3,3,3), (1,0,0,0)]),
    ]
    charts = []
    for i, (times, actions) in enumerate(specifications):
        c = chart(times, actions, 200)
        group = str(i // 2)
        charts.append(replace(c, entry=dict(c.entry, source_sha256=f'{i:064x}', group_id=group,
                                            audio_sha256=f'{i // 2 + 20:064x}')))
    val = replace(charts[-1], entry=dict(charts[-1].entry, split='validation', source_sha256='f'*64,
                                        group_id='validation', audio_sha256='e'*64))
    return charts + [val]


def profiled():
    bank = build_profile_bank(corpus(), 2)
    model = PlannedAudioModel(replace(config(), bounded_head=True, condition_full_holds=True, profile_count=2))
    model.configure_profiles(bank)
    return model, bank


def test_joint_representatives_use_train_only_and_retain_deterministic_chart_assignments():
    charts = corpus()
    bank = build_profile_bank(charts, 2)
    assert bank == build_profile_bank(charts[::-1], 2)
    corrupted_val = replace(charts[-1], duration_ms=1000000)
    assert bank == build_profile_bank(charts[:-1] + [corrupted_val], 2)
    assert all(c['source_sha256'] != 'f'*64 for c in bank['charts'])
    raw = [arrangement_values(c.source.rows['actions'], c.duration_ms).tolist() for c in charts[:-1]]
    assert all(p in raw for p in bank['profiles'])
    assert len(set(bank['medoid_source_sha256'])) == 2 and all(p > 0 for p in bank['masses'])
    assert sum(bank['masses']) == pytest.approx(1.)
    for c, record in zip(charts[:-1], bank['charts']):
        assert chart_assignment(c, bank) == record['profile_index']
        assert record['weight'] == .25
    np.testing.assert_allclose(np.average(normalized(raw, bank), axis=0), 0., atol=1e-14)


def test_zero_condition_preserves_all_generator_scores_samples_and_single_audio_encoding():
    base = PlannedAudioModel(replace(config(), bounded_head=True, condition_full_holds=True))
    with torch.no_grad():
        base.head_base.bias.fill_(-2.)
    model, bank = profiled()
    model.load_state_dict(base.state_dict(), strict=False)
    c = corpus()[1]
    batch = collate_interval(IntervalExample(c, 0, 201), base.config)
    full = torch.from_numpy(c.mel)[None]
    expected = score_interval(base, batch.inputs, base.encode_coarse(full))
    for k in range(2):
        actual = score_interval(model, batch.inputs, model.encode_coarse(full), profile_index=k)
        for field in ('head', 'release', 'row'):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    baseline = rollout(base, c.mel, c.duration_ms, seed=17)
    calls = []
    handle = model.context.register_forward_pre_hook(lambda *args: calls.append(True))
    for k in (0, 1, None):
        result = rollout(model, c.mel, c.duration_ms, seed=17, arrangement_profile=k)
        assert result.rows == baseline.rows and result.completed
        assert result.metrics['arrangement_selection'] == ('audio_prior' if k is None else 'requested')
    handle.remove()
    assert len(calls) == 3
    real = PlannedAudioModel(PlannedModelConfig(bounded_head=True, profile_count=16))
    assert sum(p.numel() for p in real.profile_condition.parameters()) + sum(p.numel() for p in real.profile_prior.parameters()) == 2736


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_shared_condition_has_three_factor_gradients_and_full_crop_native_agreement(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    torch.manual_seed(87)
    model, bank = profiled()
    with torch.no_grad():
        model.profile_condition.weight.normal_(std=.05)
        model.audio_residual.weight.normal_(std=.03)
        model.context_condition.weight.normal_(std=.03)
    model = model.to(device)
    c = corpus()[1]
    full = torch.as_tensor(c.mel, device=device)[None]
    coarse = model.encode_coarse(full)
    batch = collate_interval(IntervalExample(c, 0, 201), model.config, device)
    scores = [score_interval(model, batch.inputs, coarse, profile_index=k) for k in (0, 1)]
    for field in ('head', 'release', 'row'):
        a, b = getattr(scores[0], field), getattr(scores[1], field)
        finite = torch.isfinite(a) & torch.isfinite(b)
        assert float((a[finite] - b[finite]).abs().max().detach().cpu()) > 1e-7
    for loss in interval_losses(scores[0], batch)[:3]:
        grad = torch.autograd.grad(loss, model.profile_condition.weight, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and float(grad.abs().sum().cpu()) > 0
    x = batch.inputs.base
    crop = model.condition_audio(model.encode_crop(x.mel, x.mel_valid, x.mel_start, x.frame_count, coarse), 1)
    encoded, _ = model.encode_generation(full, seed=1, code=1)
    torch.testing.assert_close(interpolate_audio(crop, x.timing_times[None], x.mel_start, x.frame_count),
                               interpolate_audio(encoded, x.timing_times[None]), rtol=2e-5, atol=2e-5)
    # The prior's pooling cannot see artificial padded-token contents.
    values, counts = coarse
    padded = torch.cat((values, torch.randn_like(values) * 100), 1)
    torch.testing.assert_close(model.profile_log_probs(coarse), model.profile_log_probs((padded, counts)))
    with torch.no_grad():
        model.profile_prior.weight.normal_(std=.03)
    padded = padded.detach().requires_grad_()
    prior_loss = -model.profile_log_probs((padded, counts))[0, 1]
    prior_grad = torch.autograd.grad(prior_loss, padded)[0]
    assert torch.isfinite(prior_grad).all() and float(prior_grad[:, :values.shape[1]].abs().sum().cpu()) > 0
    assert not prior_grad[:, values.shape[1]:].count_nonzero()


def test_prior_is_one_chart_factor_not_reweighted_by_interval_count():
    losses = tuple(torch.tensor(v) for v in (2., 3., 4., 9.))
    logits = torch.tensor([[.2, -.7]], requires_grad=True)
    prior = logits.log_softmax(-1)
    once = training._weighted_losses(losses, .1, prior, 1, 10000, 1)
    duplicated = sum(training._weighted_losses(losses, .05, prior, 1, 10000, 2) for _ in range(2))
    torch.testing.assert_close(once, duplicated)
    torch.testing.assert_close(once[4], -prior[0, 1] * (1000 / 10001))
    torch.testing.assert_close(torch.autograd.grad(once[-1], logits, retain_graph=True)[0],
                               torch.autograd.grad(duplicated[-1], logits)[0])


def test_profile_runner_warm_start_and_checkpoint_need_no_profile_file_at_inference(tmp_path, monkeypatch):
    charts = corpus()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}\n')
    norm = dict(mean=[.7]*128, std=[1.3]*128, frontend=frontend_identity(),
                frame_count=40, audio_sha256=sorted({c.entry['audio_sha256'] for c in charts if c.split == 'train'}))
    norm_file = tmp_path / 'normalization.json'
    norm_file.write_text(json.dumps(norm))
    bank = build_profile_bank(charts, 2)
    bank_file = tmp_path / 'profiles.json'
    bank_file.write_text(json.dumps(bank))
    base = PlannedAudioModel(replace(config(), bounded_head=True, condition_full_holds=True))
    base.set_audio_normalization(torch.tensor(norm['mean']), torch.tensor(norm['std']))
    initial = tmp_path / 'initial.pt'
    torch.save(dict(format='joint-audio/planned-v1', model_config=asdict(base.config), model=base.state_dict(),
        source_revision='d'*40, manifest_sha256=digest(manifest), config={}), initial)
    cfg = PlannedTrainConfig(root=str(tmp_path), manifest_sha256=digest(manifest),
        normalization_file=str(norm_file), normalization_sha256=digest(norm_file), device='cpu',
        initial_checkpoint_file=str(initial), initial_checkpoint_sha256=digest(initial),
        profile_bank_file=str(bank_file), profile_bank_sha256=digest(bank_file),
        bounded_head=True, condition_full_holds=True, updates=2, plan_updates=2,
        songs_per_update=1, intervals_per_song=2, interval_ms=80, validation_every=2, max_seconds=30.)
    monkeypatch.setattr(training, 'revision', lambda: 'a'*40)
    monkeypatch.setattr(training, '_resource_stop', lambda _: None)
    monkeypatch.setattr(training, 'load_corpus', lambda _: (charts, norm))
    monkeypatch.setattr(training, 'PlannedAudioModel', lambda cfg: PlannedAudioModel(replace(
        config(), bounded_head=cfg.bounded_head, condition_full_holds=cfg.condition_full_holds,
        profile_count=cfg.profile_count)))
    result = training.train(cfg)
    assert result['updates'] == 2 and result['status'] == 'completed'
    assert result['parameter_update_l2']['profile_condition'] > 0
    assert result['parameter_update_l2']['profile_prior'] > 0
    run = tmp_path / 'planned-training' / cfg.run_name
    checkpoint = run / 'last.pt'
    model, metadata = load_model(checkpoint, result['checkpoint_sha256'])
    bank_file.unlink()
    assert metadata['config']['profile_bank_sha256'] == cfg.profile_bank_sha256
    assert torch.allclose(model.profile_values, torch.tensor(bank['profiles']))
    native = rollout(model, charts[0].mel, 200, seed=23, arrangement_profile=1)
    assert native.completed and native.metrics['arrangement_profile'] == 1
    evaluation = json.loads((run / 'evaluation-2.json').read_text())
    assert evaluation['arrangement_condition'] == 'reference_profile'
    assert all(r['arrangement_profile'] is not None for r in evaluation['records'])
    values = evaluation['population']
    assert values['joint_nll_per_second'] == pytest.approx(values['conditional_nll_per_second'] + values['profile_nll_per_second'], rel=1e-6)
    projected = compose_config(['initial_checkpoint_file=initial.pt', 'initial_checkpoint_sha256='+'a'*64,
        'profile_bank_file=profiles.json', 'profile_bank_sha256='+'b'*64])
    assert projected.initial_checkpoint_file == 'initial.pt' and projected.profile_bank_file == 'profiles.json'
    with pytest.raises(ValueError, match='pinned planned'):
        compose_config(['profile_bank_file=profiles.json', 'profile_bank_sha256='+'b'*64])
