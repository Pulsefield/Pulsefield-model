from dataclasses import replace
from importlib.resources import files

import numpy as np
import pytest

from ensomi_model.research.joint_audio_continuation.config import JointConfig
from ensomi_model.research.joint_audio_continuation.hydra import compose_config
from ensomi_model.research.joint_audio_continuation.training import draw_examples, selected_groups
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
