from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation import train_run, generate_run
from ensomi_model.research.bounded_typed_continuation.corpus import read_plan
from ensomi_model.research.bounded_typed_continuation.model import BoundedModel
from ensomi_model.research.r1_restore import run
from ensomi_model.research.r1_restore.config import RestoreConfig, STAGES, stage_training
from ensomi_model.research.r1_restore.harvest import collect_pool, condition_for, preference_markers
from ensomi_model.research.r1_restore.hydra import compose_config
from ensomi_model.research.r1_restore.preparation import transition_witness, select_sources
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from ensomi_model.research.vacation_training.control import Control, publish_json
from bounded_typed_continuation.test_data import chart
from bounded_typed_continuation.test_recovery import pool_fixture
from vacation_training.test_queue import configuration


def test_profile_stages_add_only_the_historical_modules():
    cfg = compose_config(['mode=status', 'output_dir=unused'])
    counts = []
    for stage in STAGES:
        settings = stage_training(cfg, stage)
        settings.validate()
        with torch.device('meta'):
            model = BoundedModel(settings.model)
        counts.append(sum(p.numel() for p in model.parameters()))
    assert counts == [2281104, 2330384, 2777232, 2917008, 3056784, 3084432]
    assert [stage_training(cfg, s).trainable for s in STAGES] == ['all', 'all', 'all', 'routing', 'release', 'consequence']
    assert stage_training(cfg, 'response').source_kl_weight == 1
    with pytest.raises((ContractError, ValueError)):
        compose_config(['mode=status', 'output_dir=unused', '+unused=1'])


def test_reconstructed_detectors_find_repetition_holds_and_response_without_changing_support():
    source = chart([(1, 0, 0, 0)] + [tuple(int(c == i % 4) for c in range(4)) for i in range(1, 65)])
    condition = condition_for(source)
    repeated = [[1, 0, 0, 0]] * 64
    markers = preference_markers(source, condition, repeated, 'routing')
    assert markers and markers[0]['index'] == 33
    assert all(m['core_mask'] == 1 and m['blocking_mask'] == 0 for m in markers)
    assert all(b['head_number'] - a['head_number'] >= 8 for a, b in zip(markers, markers[1:]))
    actions = [(1, 0, 0, 0), (0, 2, 0, 0)] + [tuple(int(c == i % 4) for c in range(4)) for i in range(2, 21)]
    # Keep the source legal while its attack groups vary; its final candidate releases an LN.
    actions = [a if i < 2 or i == 21 else tuple(1 if c == (0, 2, 3)[i % 3] else 0 for c in range(4))
               for i, a in enumerate(actions)] + [(0, 3, 0, 0)]
    held_source = chart(actions)
    held_condition = condition_for(held_source)
    native = [[1, 2, 2, 2]] + [[1, 0, 0, 0]] * 19 + [[0, 3, 3, 3]]
    held = preference_markers(held_source, held_condition, native, 'release')
    assert held and all(m['core_mask'] == 1 and m['blocking_mask'] == 14 for m in held)
    response_source = chart([(1, 0, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0),
                             (0, 0, 1, 0), (0, 0, 0, 2), (0, 0, 0, 3)],
                            times=np.array([0., 50., 100., 117., 154., 200.]))
    response = preference_markers(response_source, condition_for(response_source),
        [[2, 2, 2, 0], [0, 0, 3, 0], [0, 0, 2, 0], [0, 0, 0, 1], [3, 3, 3, 0]], 'response')
    assert response and all(m['kind'] == 'response' and m['source_short_heads'] == 0 for m in response)
    with pytest.raises(ContractError, match='every candidate'):
        preference_markers(response_source, condition_for(response_source), [], 'response')


def test_transition_selector_rejects_a_long_empty_break_as_the_only_sparsity():
    times = np.cumsum([0.] + [700.] * 12 + [100.] * 12)
    assert transition_witness(times)['ratio'] == 7
    times = np.cumsum([0., 7300.] + [100.] * 23)
    assert transition_witness(times) is None
    with pytest.raises(ContractError, match='too few'):
        select_sources([])


def test_source_selection_retains_fixed_strata_distinct_groups_and_release_exclusions():
    census = [dict(source_sha256=f'{i:064x}', group_id=f'g{i}', onsets=1600, duration_ms=200000.,
                   stars=2.2 + i % 4, ln_fraction=.2 if i % 8 >= 4 else 0.,
                   transition=dict(ratio=6. + i / 100, count=i + 1)) for i in range(128)]
    selected = select_sources(census)
    assert select_sources(list(reversed(census))) == selected
    assert len(selected['routing']) == len(selected['response']) == 32
    assert not set(selected['excluded_ordinary_release_groups']) & {r['group_id'] for r in selected['release']}
    assert len({r['group_id'] for r in selected['response']}) == 32
    for lower in (4, 5):
        for rich in (False, True):
            assert sum(lower <= r['stars'] < lower + 1 and (r['ln_fraction'] >= .1) == rich
                       for r in selected['response']) == 8


def test_real_native_collection_preserves_complete_decisions_and_stops_on_insufficient_data(tmp_path, monkeypatch):
    queue = configuration(tmp_path)
    for module in (train_run, generate_run):
        monkeypatch.setattr(module, 'source_revision', lambda: 'e' * 40)
    training = queue.teacher.training
    trained = train_run.run_training(training)
    plan = read_plan(training.plan_file, training.plan_sha256)
    parent = dict(checkpoint_file=str(Path(training.output_dir) / 'checkpoint.pt'),
        checkpoint_sha256=trained['checkpoint_sha256'], source_revision='e' * 40, plan_sha256=training.plan_sha256)
    sha, pin = next(iter(plan['sources'].items()))
    selected = [dict(source_sha256=sha, group_id=pin['identity']['group_id'])]
    cfg = RestoreConfig(output_dir=str(tmp_path / 'collector'), training=training,
        require_ac_power=False, check_thermal=False, disk_reserve_bytes=1024**2,
        resources=replace(queue.resources, disk_reserve_bytes=1024**2))
    root = Path(cfg.output_dir)
    root.mkdir()
    control = Control(cfg, root, time.time(), time.time() + 60)
    state = {}
    (root / 'PAUSE').touch()
    assert collect_pool(cfg, 'routing', parent, plan, selected, root / 'native', state, lambda: None, control)['status'] == 'paused'
    (root / 'PAUSE').unlink()
    result = collect_pool(cfg, 'routing', parent, plan, selected, root / 'native', state, lambda: None, control)
    assert result['status'] == 'insufficient-native-failures'
    manifest = json.loads(Path(result['path']).read_text())
    payload = json.loads((Path(result['path']).parent / manifest['runs'][0]['path']).read_text())
    assert len(payload['actions']) == pin['rows'] - pin['seed_rows']
    assert payload['policy_checkpoint_sha256'] == parent['checkpoint_sha256']
    assert state['generation']['cases'][sha]['status'] == 'completed'
    # The saved complete chart is checked and reused; no additional generation segment is created.
    collect_pool(cfg, 'routing', parent, plan, selected, root / 'native', state, lambda: None, control)
    assert len(state['generation']['cases'][sha]['segments']) == 1
    path = Path(result['path']).parent / manifest['runs'][0]['path']
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ContractError, match='payload changed'):
        collect_pool(cfg, 'routing', parent, plan, selected, root / 'native', state, lambda: None, control)


def fixture(tmp_path, monkeypatch):
    queue = configuration(tmp_path)
    for module in (run, train_run, generate_run):
        monkeypatch.setattr(module, 'source_revision', lambda: 'e' * 40)
    cfg = RestoreConfig(output_dir=str(tmp_path / 'restore'), base_plan_file=queue.teacher.base_plan_file,
        base_plan_sha256=queue.teacher.base_plan_sha256, evaluation_file=queue.teacher.evaluation_file,
        evaluation_sha256=queue.teacher.evaluation_sha256, catalog_file=str(tmp_path / 'catalog.json'), catalog_root=str(tmp_path),
        require_ac_power=False, check_thermal=False, disk_reserve_bytes=1024**2,
        resources=replace(queue.resources, disk_reserve_bytes=1024**2),
        training=replace(queue.teacher.training, plan_file='', plan_sha256='',
                         output_dir='artifacts/bounded-typed-continuation/corpus'),
        milestones=dict(base=[37, 103], seed=[137], memory=[167], routing=[197], release=[227], response=[257]))
    def prepare(config, plan, root, check):
        selected = [dict(source_sha256=sha, group_id=p['identity']['group_id']) for sha, p in plan['sources'].items()]
        a = publish_json(root / 'census.json', dict(sources=selected))
        b = publish_json(root / 'selection.json', dict(selections={s: selected for s in ('routing', 'release', 'response')}))
        return dict(census_sha256=a, selection_sha256=b)
    monkeypatch.setattr(run, 'prepare_census', prepare)
    parents = []
    def collect(config, kind, parent, plan, selection, output, state, save, control):
        # Synthetic preference fixtures isolate queue ordering and real fork/optimizer execution.
        # Detector and collector admission rules have separate tests; no production threshold is lowered.
        parents.append((kind, parent['onset_exposures']))
        output.mkdir(parents=True, exist_ok=True)
        settings = replace(config.training, plan_file=parent['plan_file'], plan_sha256=parent['plan_sha256'])
        path, digest, _, _ = pool_fixture(settings, output)
        return dict(status='ready', path=str(path), sha256=digest)
    monkeypatch.setattr(run, 'collect_pool', collect)
    return cfg, parents


def test_six_stage_queue_pauses_recovers_and_preserves_frozen_parent_states(tmp_path, monkeypatch):
    cfg, parents = fixture(tmp_path, monkeypatch)
    assert run.run_restore(cfg)['status'] == 'preflight_passed'
    root = Path(cfg.output_dir)
    (root / 'PAUSE').touch()
    paused = run.run_restore(replace(cfg, mode='run'))
    assert paused['status'] == 'paused' and paused['stages']['base']['segments'][0]['result']['durable_onset_exposures'] == 0
    (root / 'PAUSE').unlink()
    original_collector = run.collect_pool
    first_harvest = True
    def pause_before_harvest(*args):
        nonlocal first_harvest
        if first_harvest:
            first_harvest = False
            return dict(status='paused', reason='pause_file')
        return original_collector(*args)
    monkeypatch.setattr(run, 'collect_pool', pause_before_harvest)
    between = run.run_restore(replace(cfg, mode='resume'))
    assert between['status'] == 'paused' and between['stages']['memory']['status'] == 'completed'
    result = run.run_restore(replace(cfg, mode='resume'))
    assert result['status'] == 'completed' and result['checkpoint']['onset_exposures'] == 257
    assert parents == [('routing', 167), ('release', 197), ('response', 227)]
    assert all(v['status'] == 'completed' for v in result['stages'].values())
    for stage in ('routing', 'release', 'response'):
        assert result['stages'][stage]['parent_audit']['inherited_adam_unchanged']
    response = result['stages']['response']['segments'][-1]['result']
    assert response['metrics']['source_kl_sum'] > 0 and response['metrics']['recovery_queries'] > 0
    assert run.run_restore(replace(cfg, mode='resume')) == result
    assert len(parents) == 3
    with pytest.raises(ContractError, match='configuration'):
        run.run_restore(replace(cfg, mode='resume', max_seconds=cfg.max_seconds + 1))


def test_insufficient_pool_stops_before_any_routing_training_and_is_not_retried(tmp_path, monkeypatch):
    cfg, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(run, 'collect_pool', lambda *args: dict(status='insufficient-native-failures', queries=0, groups=0))
    result = run.run_restore(replace(cfg, mode='run'))
    assert result['status'] == 'needs_review' and result['active_stage'] == 'routing'
    assert not (Path(cfg.output_dir) / 'routing/training').exists()
    assert run.run_restore(replace(cfg, mode='resume')) == result
