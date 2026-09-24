from dataclasses import replace
import copy
from importlib.resources import files
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ensomi_model.research.joint_audio_continuation.config import JointConfig
from ensomi_model.research.joint_audio_continuation.hydra import compose_config
from ensomi_model.research.joint_audio_continuation.training import (
    backward_logical, draw_examples, make_batch, selected_groups, training_normalization,
)
from ensomi_model.research.joint_audio_continuation.batching import batch_losses, score_batch
from ensomi_model.research.joint_audio_continuation.data import digest, frontend_identity, query
from ensomi_model.research.joint_audio_continuation.model import JointAudioModel, JointModelConfig
from ensomi_model.research.joint_audio_continuation.sampling import LogicalExample
from .test_data import chart


def test_packaged_configuration_projects_and_rejects_unknown_or_unpinned_inputs():
    assert files('ensomi_model.configs.hydra').joinpath('joint_audio.yaml').is_file()
    config = compose_config(['mode=train', 'batch_size=3', 'fixed_train_queries=16', 'train_groups=0', 'cpu_threads=2'])
    assert (config.mode, config.batch_size, config.fixed_train_queries, config.train_groups, config.cpu_threads) == ('train', 3, 16, 0, 2)
    with pytest.raises(ValueError, match='Unknown'):
        compose_config(['+unused=1'])
    with pytest.raises(ValueError, match='pinned'):
        compose_config(['mode=generate'])
    with pytest.raises(ValueError, match='511'):
        compose_config(['history_rows=512'])
    with pytest.raises(ValueError, match='Coverage pass'):
        compose_config(['coverage_pass=true'])
    with pytest.raises(ValueError, match='random queries'):
        compose_config(['full_wait_supervision=true', 'fixed_train_queries=16'])
    expanded = compose_config(['full_wait_supervision=true', 'coverage_pass=true'])
    assert expanded.full_wait_supervision and expanded.coverage_pass
    prior = compose_config(['mode=generate', 'checkpoint_file=model.pt',
                            'checkpoint_sha256=' + '0' * 64, 'head_spacing_ms=27'])
    assert prior.head_spacing_ms == 27
    with pytest.raises(ValueError, match='decoder setting'):
        compose_config(['mode=train', 'head_spacing_ms=27'])
    pinned = compose_config(['mode=train', 'normalization_file=stats.json',
                             'normalization_sha256=' + 'a' * 64])
    assert pinned.normalization_file == 'stats.json' and pinned.normalization_sha256 == 'a' * 64
    with pytest.raises(ValueError, match='normalization override'):
        compose_config(['mode=train', 'normalization_file=stats.json'])
    with pytest.raises(ValueError, match='train mode'):
        compose_config(['normalization_file=stats.json', 'normalization_sha256=' + 'a' * 64])
    paired = compose_config(['source_aliases_file=aliases.json', 'source_aliases_sha256=' + 'b' * 64])
    assert paired.source_aliases_file == 'aliases.json' and paired.source_aliases_sha256 == 'b' * 64
    for overrides in (['source_aliases_file=aliases.json'],
                      ['source_aliases_file=aliases.json', 'source_aliases_sha256=' + 'z' * 64],
                      ['mode=train', 'source_aliases_file=aliases.json', 'source_aliases_sha256=' + 'b' * 64]):
        with pytest.raises(ValueError, match='Source aliases require prepare mode'):
            compose_config(overrides)


def test_frozen_normalization_uses_training_subset_and_reaches_model_buffers(tmp_path):
    current = dict(mean=[1.] * 128, std=[2.] * 128)
    frozen = dict(mean=[3.] * 128, std=[4.] * 128, audio_sha256=['a' * 64],
                  frame_count=100, frontend=frontend_identity())
    path = tmp_path / 'frozen.json'
    path.write_text(json.dumps(frozen))
    charts = [SimpleNamespace(split=split, entry=dict(audio_sha256=sha * 64))
              for split, sha in [('train', 'a'), ('train', 'b'), ('validation', 'c')]]
    cfg = JointConfig(mode='train', normalization_file=str(path), normalization_sha256=digest(path))
    selected, identity = training_normalization(cfg, charts, current)
    assert selected == frozen and identity['normalization_override']
    assert identity['normalization_sha256'] == digest(path)
    model = JointAudioModel(JointModelConfig())
    model.set_audio_normalization(torch.tensor(selected['mean']), torch.tensor(selected['std']))
    torch.testing.assert_close(model.audio_mean, torch.full((128,), 3.))
    torch.testing.assert_close(model.audio_std, torch.full((128,), 4.))
    # Adding a held-out identity to the pinned metadata is still forbidden.
    frozen['audio_sha256'].append('c' * 64)
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='pinned SHA'):
        training_normalization(cfg, charts, current)
    with pytest.raises(ValueError, match='TRAIN audio subset'):
        training_normalization(replace(cfg, normalization_sha256=digest(path)), charts, current)


def test_default_normalization_keeps_corpus_statistics(tmp_path):
    statistics = dict(mean=[0.] * 128, std=[1.] * 128)
    path = tmp_path / 'normalization.json'
    path.write_text(json.dumps(statistics))
    chosen, identity = training_normalization(JointConfig(root=str(tmp_path)), [], statistics)
    assert chosen is statistics and not identity['normalization_override']
    assert identity['normalization_sha256'] == digest(path)


def test_train_sampling_preserves_alternatives_and_excludes_validation():
    source = chart([(2, 0, 0, 0), (3, 0, 0, 0), (0, 1, 0, 0)], [0, 7000, 9000])
    base = replace(source, entry=dict(source.entry, selection='base', stratum=[0, 1]))
    alt = replace(base, entry=dict(base.entry, source_sha256='c' * 64, selection='alternative'))
    other = replace(base, entry=dict(base.entry, group_id='other', source_sha256='d'*64, stratum=[1, 0]))
    val = replace(base, entry=dict(base.entry, group_id='val', split='validation'))
    groups = selected_groups([val, alt, other, base], 1)
    assert len(groups) == 1 and len(groups[0]) == 2
    assert {c.entry['source_sha256'] for c in groups[0]} == {'a'*64, 'c'*64}
    examples = draw_examples(groups, np.random.default_rng(1), JointConfig(), 200)
    assert {c.entry['source_sha256'] for c, q in examples} == {'a'*64, 'c'*64}
    assert all(c.split == 'train' and not q.replay.is_complete for c, q in examples)
    assert any(q.cursor_ms == -1 for c, q in examples)
    assert any(q.censored and q.replay.occupancy[0] for c, q in examples)
    assert any(q.censored and q.source_index == 3 for c, q in examples)


@pytest.mark.parametrize('microbatch_size', [1, 2, 4])
def test_microbatch_gradients_normalize_per_logical_wait_not_window_count(microbatch_size):
    torch.manual_seed(917)
    source = chart([(2, 0, 0, 0), (3, 0, 0, 0), (0, 1, 0, 0)], [0, 90, 110], duration_ms=120)
    examples = [LogicalExample(source, tuple(query(source, cursor, horizon_ms=30) for cursor in (0, 30, 60))),
                LogicalExample(source, (query(source, 90, horizon_ms=20),))]
    settings = JointModelConfig(hidden=8, audio_width=8, audio_levels=1, history_levels=2,
                                expansion=2, coupling_rank=2, routing_hidden=12, release_hidden=12)
    model = JointAudioModel(settings)
    reference = copy.deepcopy(model)
    flat = [(e.chart, q) for e in examples for q in e.queries]
    batch = make_batch(flat, settings, 'cpu')
    terms = batch_losses(score_batch(reference, batch.inputs), batch)
    expected = terms.total.sum() / len(examples)
    expected.backward()
    observed, windows = backward_logical(model, examples, JointConfig(device='cpu', batch_size=microbatch_size))
    assert windows == 4
    assert observed[0] == pytest.approx(expected.item(), rel=1e-6)
    for a, b in zip(model.parameters(), reference.parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-5)
