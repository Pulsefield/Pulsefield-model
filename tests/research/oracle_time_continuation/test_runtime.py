from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from pulsefield_model.research.oracle_time_continuation.config import BackboneConfig
from pulsefield_model.research.oracle_time_continuation.decoding import DecodeSamplingPolicy, policy_probabilities, sample_row
from pulsefield_model.research.oracle_time_continuation.export import verify_rows
from pulsefield_model.research.oracle_time_continuation.generation import Rollout
from pulsefield_model.research.oracle_time_continuation.model import CausalBackbone, JointRowDistribution, WEIGHTS_FORMAT
from pulsefield_model.research.oracle_time_continuation.runtime import ResourceConfig, ResourceLimit, atomic_checkpoint, validate_envelope
from pulsefield_model.research.oracle_time_continuation.storage import SourceCacheConfig, SourceStore
from pulsefield_model.research.oracle_time_continuation.train_hydra import compose_config
from pulsefield_model.research.oracle_time_continuation.train_run import run_training
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError, digest
from .conftest import admit, source_bytes
from .test_train_hydra import write_inputs


def test_playback_header_preserves_timing_but_replaces_difficulty_identity(tmp_path):
    from pulsefield_model.research.oracle_time_continuation.export import presentation_header
    path = tmp_path / 'source.osu'
    path.write_text('osu file format v14\n[General]\nAudioFilename: song.mp3\nMode:3\n'
                    '[Metadata]\nTitle:Example\nVersion:Hard\nBeatmapID:42\nBeatmapSetID:77\n'
                    '[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n'
                    '[TimingPoints]\n1000,400,4,2,0,100,1,0\n1200,-50,4,2,0,100,0,0\n'
                    '[HitObjects]\n64,192,1000,128,0,1200:0:0:0:0:\n')
    header = presentation_header(path).decode()
    assert 'AudioFilename:song.mp3' in header and 'OverallDifficulty:8' in header
    assert '1000,400,4,2,0,100,1,0' in header and '1200,-50,4,2,0,100,0,0' in header
    assert 'BeatmapID:0\n' in header and 'BeatmapSetID:-1\n' in header
    assert 'Version:Hard (oracle continuation)' in header and header.endswith('[HitObjects]\n')
    assert '64,192' not in header


def model(**kwargs):
    torch.manual_seed(7)
    return CausalBackbone(BackboneConfig(hidden=8, heads=2, recent=4, coarse_group=2, coarse_capacity=2,
                                         coupling_rank=4, **kwargs)).eval()


def test_export_database_cap_stops_insertion_and_keeps_previous_publication(tmp_path, monkeypatch):
    from pulsefield_model.research.oracle_time_continuation import export
    import sqlite3

    rows, destination = tmp_path / 'rows.jsonl', tmp_path / 'generated.osu'
    rows.write_text(''.join(json.dumps(dict(event_id=i, time_ms=i, actions=[1, 1, 1, 1])) + '\n'
                            for i in range(500)))
    destination.write_bytes(b'previous publication')
    inserts = []
    original = sqlite3.connect

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith('INSERT'):
                inserts.append(sql)
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(export.sqlite3, 'connect', lambda path: original(path, factory=ObservedConnection))
    with pytest.raises(ResourceLimit, match='SQLite page or disk limit'):
        export.export_osu(rows, destination, range(500), replace(ResourceConfig(), output_max_bytes=8192))
    assert 0 < len(inserts) < 2000  # The database may not finish growing before the cap is checked.
    assert destination.read_bytes() == b'previous publication'
    assert not (tmp_path / '.generated.osu.staging').exists()


def test_export_header_budget_is_checked_before_writing_any_oversized_output(tmp_path):
    from pulsefield_model.research.oracle_time_continuation.export import export_osu

    rows, destination = tmp_path / 'rows.jsonl', tmp_path / 'generated.osu'
    rows.write_text(json.dumps(dict(event_id=0, time_ms=0, actions=[1, 0, 0, 0])) + '\n')
    destination.write_bytes(b'previous publication')
    with pytest.raises(ResourceLimit, match='Export exceeds output_max_bytes'):
        export_osu(rows, destination, [0], replace(ResourceConfig(), output_max_bytes=16384), header=b'x' * 16385)
    assert destination.read_bytes() == b'previous publication'
    assert not (tmp_path / '.generated.osu.staging').exists()


def test_export_streams_start_order_without_an_additional_disk_sort(tmp_path, monkeypatch):
    from pulsefield_model.research.oracle_time_continuation import export
    import sqlite3

    rows, destination = tmp_path / 'rows.jsonl', tmp_path / 'generated.osu'
    actions = [(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0)]
    rows.write_text(''.join(json.dumps(dict(event_id=i, time_ms=i, actions=row)) + '\n'
                            for i, row in enumerate(actions)))
    original, plans = sqlite3.connect, []

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == 'SELECT time,lane,end FROM notes ORDER BY time,lane':
                plans.extend(super().execute('EXPLAIN QUERY PLAN ' + sql).fetchall())
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(export.sqlite3, 'connect', lambda path: original(path, factory=ObservedConnection))
    export.export_osu(rows, destination, range(3))
    assert plans and all('TEMP B-TREE' not in row[-1] for row in plans)
    assert destination.read_text().split('[HitObjects]\n')[1].splitlines() == [
        '64,192,0,128,0,2:0:0:0:0:', '192,192,1,1,0,0:0:0:0:']


def test_resource_journal_limits_apply_before_the_complete_record(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pulsefield_model.research.oracle_time_continuation import runtime

    path = tmp_path / 'resources.jsonl'
    path.write_text('previous\n')
    with path.open('a') as log:
        with pytest.raises(ResourceLimit, match='output_max_bytes'):
            runtime.ResourceGuard('cpu', replace(ResourceConfig(), output_max_bytes=32), log)
    assert path.read_text() == 'previous\n'
    monkeypatch.setattr(runtime.shutil, 'disk_usage', lambda path: SimpleNamespace(free=512 * 1024**2))
    with path.open('a') as log:
        with pytest.raises(ResourceLimit, match='disk reserve'):
            runtime.ResourceGuard('cpu', ResourceConfig(), log)
    assert path.read_text() == 'previous\n'


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_checkpoint_reowns_online_carry_without_retained_attention_banks(tmp_path):
    import gc
    from pulsefield_model.research.oracle_time_continuation.checkpoint import pack_state
    from pulsefield_model.research.oracle_time_continuation.runtime import tensor_bytes

    gc.collect()
    torch.mps.synchronize()
    baseline = torch.mps.current_allocated_memory()
    source = admit([(i % 4, i * 100, i * 100) for i in range(300)])
    network = CausalBackbone().eval().to('mps')
    resources = replace(ResourceConfig(), check_every_rows=64)
    with Rollout(network, source.skeleton, tmp_path / 'owned', resources=resources) as run:
        run.prefill(source.targets[:30])
        run.advance(98)
        payload = tensor_bytes(pack_state(run.state))
        torch.mps.synchronize()
        allocated = torch.mps.current_allocated_memory() - baseline
        parameters = sum(p.numel() * p.element_size() for p in network.parameters())
        assert allocated < parameters + payload + 2 * 1024**2
    del run, network
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()


def test_sampling_preserves_cutoff_ties_and_raw_support():
    legal = torch.zeros(256, dtype=torch.bool)
    legal[1:5] = True
    log_probs = torch.full((256,), -torch.inf)
    log_probs[1:5] = torch.log(torch.tensor([.4, .2, .2, .2]))
    distribution = JointRowDistribution(log_probs, legal)
    raw, policy, keep = policy_probabilities(distribution, DecodeSamplingPolicy(top_p=.5))
    assert keep.sum() == 4
    torch.testing.assert_close(policy, raw.exp(), rtol=1e-6, atol=1e-7)
    _, policy, keep = policy_probabilities(distribution, DecodeSamplingPolicy(top_p=.3))
    assert keep.sum() == 1 and policy[1] == 1
    rng = torch.Generator().manual_seed(17)
    _, raw_policy, _ = policy_probabilities(distribution, DecodeSamplingPolicy(top_p=1))
    seen = {sample_row(distribution, 3, DecodeSamplingPolicy(top_p=1), rng)[0].actions for _ in range(100)}
    assert len(seen) == 4
    for kwargs in ({'temperature': 0}, {'top_p': 0}, {'top_p': 1.01}, {'beta': .1}, {'temperature': float('nan')}):
        with pytest.raises(ContractError):
            DecodeSamplingPolicy(**kwargs)
    distribution.log_probs[1] = float('nan')
    with pytest.raises(ContractError, match='finite'):
        policy_probabilities(distribution, DecodeSamplingPolicy())


def test_generation_checkpoint_cadence_is_independent_of_checks_and_preserves_output(tmp_path):
    source = admit([(i % 4, i * 100, i * 100) for i in range(100)])
    network = model()
    for interval in (4, 17):
        directory = tmp_path / str(interval)
        with Rollout(network, source.skeleton, directory, checkpoint_every_rows=interval,
                     resources=replace(ResourceConfig(), check_every_rows=8)) as run:
            run.prefill(iter(source.targets[:30]))
            run.advance()
            run.finish()
    assert (tmp_path/'4/rows.jsonl').read_bytes() == (tmp_path/'17/rows.jsonl').read_bytes()
    assert (tmp_path/'4/generated.osu').read_bytes() == (tmp_path/'17/generated.osu').read_bytes()
    logs = [[json.loads(line) for line in (tmp_path/str(n)/'resources.jsonl').read_text().splitlines()]
            for n in (4, 17)]
    assert [r['row'] for r in logs[0] if r['phase']=='decode'] == [r['row'] for r in logs[1] if r['phase']=='decode']
    assert sum(r['phase']=='checkpoint' for r in logs[1]) < sum(r['phase']=='checkpoint' for r in logs[0])
    with pytest.raises(ContractError, match='identity mismatch'):
        Rollout(network, source.skeleton, tmp_path/'17', checkpoint_every_rows=4, resume=True)
    for value in (0, True, 8193):
        with pytest.raises(ContractError, match='checkpoint_every_rows'):
            Rollout(network, source.skeleton, tmp_path/'invalid', checkpoint_every_rows=value)


def test_source_lru_bounds_and_oversize_staging(tmp_path):
    data = source_bytes([(i % 4, i * 100, i * 100) for i in range(90)])
    sha = digest(data)
    source_path = tmp_path / 'chart.osu'
    source_path.write_bytes(data)
    for budget in (100, 10000):
        store = SourceStore(tmp_path / str(budget), SourceCacheConfig(max_sources=1, max_bytes=budget, staging_rows=7))
        source = store.admit(source_path, sha, group_id='song:a', split='train')
        expected = admit([(i % 4, i * 100, i * 100) for i in range(90)])
        assert tuple(source.targets) == expected.targets
        assert source.minimum_seed() == expected.minimum_seed()
        assert tuple(source.skeleton.times_ms) == expected.skeleton.times_ms
        assert store.owned_bytes <= budget and store.metrics()['staging_bytes'] <= 7 * 12
        assert not isinstance(source.targets, tuple)
        store.clear()
        assert store.owned_bytes == 0
    limited = SourceStore(tmp_path / 'limited', SourceCacheConfig(source_max_bytes=10))
    with pytest.raises(ContractError, match='byte limit'):
        limited.admit(source_path, sha, group_id='song:a', split='train')


def test_source_lru_evicts_and_reloads_without_pinned_buffers(tmp_path):
    store = SourceStore(tmp_path / 'cache', SourceCacheConfig(max_sources=1, max_bytes=10000))
    sources = []
    for shift in range(2):
        data = source_bytes([(i % 4, i * 100 + shift, i * 100 + shift) for i in range(40)])
        path = tmp_path / f'{shift}.osu'
        path.write_bytes(data)
        sources.append(store.admit(path, digest(data), group_id=f'song:{shift}', split='train'))
    for source in (*sources, sources[0]):
        assert source.targets[0].time_ms >= 0
        assert len(store.cache) == 1
    assert store.evictions == 2


def test_checkpoint_compacts_views_and_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.pt'
    config = replace(ResourceConfig(), checkpoint_max_bytes=100_000)
    value = torch.zeros(1_000_000)[123:133]
    report = atomic_checkpoint(path, {'value': value}, config)
    assert report['checkpoint_bytes'] < 10000 and report['tensor_bytes'] == 40
    original = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('interrupted write')
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'save', fail)
        with pytest.raises(OSError):
            atomic_checkpoint(path, {'value': value}, config)
    assert path.read_bytes() == original
    assert {p.name for p in tmp_path.iterdir()} == {'checkpoint.pt', '.checkpoint.pt.lock'}
    with pytest.raises(ResourceLimit):
        atomic_checkpoint(path, {'value': torch.zeros(100_000)}, config)
    assert path.read_bytes() == original


def test_sigkill_during_real_checkpoint_write_preserves_publication_and_bounds_retry_staging(tmp_path):
    import signal
    import subprocess
    import sys

    path = tmp_path / 'checkpoint.pt'
    config = replace(ResourceConfig(), checkpoint_max_bytes=1024**2)
    atomic_checkpoint(path, {'value': torch.tensor([7.])}, config)
    original = path.read_bytes()
    script = '''
from dataclasses import replace
from pathlib import Path
import os, signal, sys
import torch
from pulsefield_model.research.oracle_time_continuation import runtime
write = runtime.LimitedWriter.write
def interrupt(self, data):
    count = write(self, data)
    if self.stream.tell() > 65536:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        os.kill(os.getpid(), signal.SIGKILL)
    return count
runtime.LimitedWriter.write = interrupt
runtime.atomic_checkpoint(Path(sys.argv[1]), {'value': torch.ones(40000)},
                          replace(runtime.ResourceConfig(), checkpoint_max_bytes=1024**2))
'''
    for _ in range(3):
        child = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, timeout=30)
        assert child.returncode == -signal.SIGKILL, child.stderr.decode()
        assert path.read_bytes() == original
        assert {p.name for p in tmp_path.iterdir()} == {
            'checkpoint.pt', '.checkpoint.pt.lock', '.checkpoint.pt.staging'}
        assert sum(p.stat().st_size for p in (tmp_path / '.checkpoint.pt.staging').iterdir()) < 1024**2
    atomic_checkpoint(path, {'value': torch.tensor([9.])}, config)
    assert torch.load(path, weights_only=True)['value'].item() == 9
    assert not (tmp_path / '.checkpoint.pt.staging').exists()


def test_publication_lock_does_not_remove_an_active_writer_staging(tmp_path):
    from pulsefield_model.research.oracle_time_continuation.publication import staging_directory

    path = tmp_path / 'checkpoint.pt'
    with staging_directory(path) as first:
        marker = first / 'live'
        marker.write_bytes(b'current writer')
        with pytest.raises(ContractError, match='Another writer'):
            with staging_directory(path):
                pytest.fail('Concurrent publication must not take the active staging directory')
        assert marker.read_bytes() == b'current writer'


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='requires Apple MPS'))])
def test_rollout_seed_only_resume_output_alignment_and_osu_roundtrip(tmp_path, device):
    source = admit([(0, 0, 15000)] + [(1 + i % 3, i * 100, i * 100) for i in range(1, 150)])
    prefix = source.targets[:source.minimum_seed().seed_row_count]
    network = model(time_lookahead_rows=16, clock_readout_hidden=12).to(device)
    torch.nn.init.normal_(network.timing.projection[-1].weight, std=.1)
    torch.nn.init.normal_(network.head.clock_readout.projection[-1].weight, std=.1)
    policy = DecodeSamplingPolicy(top_p=.95)
    resources = replace(ResourceConfig(), check_every_rows=32)
    with Rollout(network, source.skeleton, tmp_path / 'reference', policy, resources=resources) as run:
        assert run.guard.device == device
        run.prefill(iter(prefix))
        run.advance()
        expected = run.finish()
    with Rollout(network, source.skeleton, tmp_path / 'resumed', policy, resources=resources) as run:
        run.prefill(iter(prefix))
        run.advance(40)
        checkpoint = (tmp_path / 'resumed/checkpoint.pt').read_bytes()
        run.advance(9)
    # Emulate a crash after output persistence but before the new checkpoint publication.
    (tmp_path / 'resumed/checkpoint.pt').write_bytes(checkpoint)
    with Rollout(network, source.skeleton, tmp_path / 'resumed', policy, resources=resources, resume=True) as run:
        run.advance()
        report = run.finish()
    assert report == expected
    assert (tmp_path / 'resumed/rows.jsonl').read_bytes() == (tmp_path / 'reference/rows.jsonl').read_bytes()
    output = (tmp_path / 'resumed/generated.osu').read_bytes()
    from pulsefield_model.research.oracle_time_continuation.data import admit_source
    recovered = admit_source(output, digest(output), group_id='generated', split='validation')
    rows = [json.loads(line) for line in (tmp_path / 'resumed/rows.jsonl').read_text().splitlines()]
    assert [r.actions for r in recovered.targets] == [tuple(r['actions']) for r in rows]
    assert recovered.skeleton == source.skeleton
    assert report['terminal_forced_closes'] <= 4
    with pytest.raises(ContractError, match='identity mismatch'):
        Rollout(network, source.skeleton, tmp_path / 'resumed', DecodeSamplingPolicy(top_p=1), resume=True)
    journal = tmp_path / 'resumed/rows.jsonl'
    journal.write_bytes(b'x' + journal.read_bytes()[1:])
    with pytest.raises(ContractError, match='digest differs'):
        Rollout(network, source.skeleton, tmp_path / 'resumed', policy, resume=True)


@pytest.mark.parametrize('gap_probability', [0., .5])
def test_training_resume_replays_draws_optimizer_and_weights(tmp_path, gap_probability):
    sha, path, manifest = write_inputs(tmp_path)
    cfg = compose_config([f'source_dir={tmp_path / "sources"}', f'split_manifest={path}',
                          f'split_sha256={manifest["sha256"]}', f'source_sha256=[{sha}]',
                          f'cache_dir={tmp_path / "cache"}', f'output_dir={tmp_path / "reference"}',
                          'device=cpu', 'updates=3', 'model.hidden=8', 'model.heads=2', 'model.recent=4',
                          'model.time_lookahead_rows=16',
                          f'windows.gap_sampling_probability={gap_probability}',
                          'windows.gap_thresholds_ms=[100,200,400]', 'windows.windows_per_chart=2',
                          'model.coarse_group=2', 'model.coarse_capacity=2', 'model.coupling_rank=4'])
    run_training(cfg, resolved_yaml='reference')
    cfg.output_dir = str(tmp_path / 'resume')
    cfg.updates = 1
    run_training(cfg, resolved_yaml='resume')
    with (tmp_path / 'resume/windows.jsonl').open('ab') as stream:
        stream.write(b'{incomplete update\n')
    cfg.resume = True
    cfg.updates = 3
    run_training(cfg, resolved_yaml='resume')
    a = torch.load(tmp_path / 'reference/weights.pt', weights_only=True)
    b = torch.load(tmp_path / 'resume/weights.pt', weights_only=True)
    for key in a['model_state_dict']:
        torch.testing.assert_close(a['model_state_dict'][key], b['model_state_dict'][key], atol=0, rtol=0)
    def draws(directory):
        return [{key: value for key, value in record.items() if key != 'prefill_seconds'} for record in
                map(json.loads, (directory / 'windows.jsonl').read_text().splitlines())]
    assert draws(tmp_path / 'reference') == draws(tmp_path / 'resume')


def test_pinned_weight_initialization_starts_fresh_and_resume_owns_the_parameters(tmp_path, monkeypatch):
    from pulsefield_model.research.oracle_time_continuation import train_run
    from pulsefield_model.research.oracle_time_continuation.storage import file_digest

    sha, path, manifest = write_inputs(tmp_path)
    cfg = compose_config([f'source_dir={tmp_path / "sources"}', f'split_manifest={path}',
                          f'split_sha256={manifest["sha256"]}', f'source_sha256=[{sha}]',
                          f'output_dir={tmp_path / "initial"}', f'cache_dir={tmp_path / "cache"}',
                          'model.hidden=8', 'model.heads=2', 'model.recent=4',
                          'model.coarse_group=2', 'model.coarse_capacity=2', 'model.coupling_rank=4',
                          'updates=2'])
    run_training(cfg, resolved_yaml='initial')
    initial = tmp_path / 'initial/weights.pt'
    expected = torch.load(initial, weights_only=True)
    cfg.initial_weights = str(initial)
    cfg.initial_weights_sha256 = file_digest(initial)
    cfg.output_dir, cfg.updates = str(tmp_path / 'warm'), 1
    original = train_run.SequenceTrainer.update
    starts = []

    def observe(self, windows):
        starts.append((self.updates, bool(self.optimizer.state)))
        if self.updates == 0:
            for name, value in self.engine.model.state_dict().items():
                torch.testing.assert_close(value, expected['model_state_dict'][name], rtol=0, atol=0)
        return original(self, windows)

    monkeypatch.setattr(train_run.SequenceTrainer, 'update', observe)
    result = run_training(cfg, resolved_yaml='warm')
    assert starts == [(0, False)] and result['updates'] == 1
    assert result['initialization'] == dict(weights_sha256=cfg.initial_weights_sha256, source_updates=2)
    weights = torch.load(tmp_path / 'warm/weights.pt', weights_only=True)
    assert weights['initialization'] == result['initialization']
    cfg.output_dir = str(tmp_path / 'wrong-digest')
    pinned = cfg.initial_weights_sha256
    cfg.initial_weights_sha256 = '0' * 64
    with pytest.raises(ContractError, match='pinned SHA'):
        run_training(cfg, resolved_yaml='bad')
    assert starts == [(0, False)]
    cfg.initial_weights_sha256 = pinned
    initial.unlink()
    cfg.output_dir, cfg.resume, cfg.updates = str(tmp_path / 'warm'), True, 2
    resumed = run_training(cfg, resolved_yaml='resume')
    assert starts == [(0, False), (1, True)]
    assert resumed['initialization'] == result['initialization']


def test_periodic_training_checkpoint_replays_completed_unsaved_updates_and_warmup(tmp_path, monkeypatch):
    from pulsefield_model.research.oracle_time_continuation import train_run

    sha, path, manifest = write_inputs(tmp_path)
    cfg = compose_config([f'source_dir={tmp_path / "sources"}', f'split_manifest={path}',
                          f'split_sha256={manifest["sha256"]}', f'source_sha256=[{sha}]',
                          f'output_dir={tmp_path / "interrupted"}', f'cache_dir={tmp_path / "cache"}',
                          'model.hidden=8', 'model.heads=2', 'model.recent=4',
                          'model.coarse_group=2', 'model.coarse_capacity=2', 'model.coupling_rank=4',
                          'updates=4', 'checkpoint_every_updates=2', 'training.warmup_updates=4',
                          'model.time_lookahead_rows=16', 'training.timing_learning_rate=0.0003',
                          'model.clock_readout_hidden=12', 'training.clock_readout_learning_rate=0.0008'])
    original = train_run.SequenceTrainer.update

    def interrupt(self, windows):
        if self.updates == 3:
            raise ResourceLimit('test interruption after an unsaved completed update')
        return original(self, windows)

    with monkeypatch.context() as patch:
        patch.setattr(train_run.SequenceTrainer, 'update', interrupt)
        with pytest.raises(ResourceLimit, match='unsaved'):
            run_training(cfg, resolved_yaml='interrupted')
    checkpoint = torch.load(tmp_path / 'interrupted/checkpoint.pt', weights_only=True)
    assert checkpoint['updates'] == 2
    assert len((tmp_path / 'interrupted/updates.jsonl').read_text().splitlines()) == 3
    cfg.resume = True
    run_training(cfg, resolved_yaml='resume')
    cfg.resume, cfg.output_dir = False, str(tmp_path / 'reference')
    run_training(cfg, resolved_yaml='reference')
    a = torch.load(tmp_path / 'interrupted/weights.pt', weights_only=True)
    b = torch.load(tmp_path / 'reference/weights.pt', weights_only=True)
    for key, value in a['model_state_dict'].items():
        torch.testing.assert_close(value, b['model_state_dict'][key], atol=0, rtol=0)
    draws = lambda folder: [json.loads(line) for line in (folder / 'windows.jsonl').read_text().splitlines()]
    for left, right in zip(draws(tmp_path / 'interrupted'), draws(tmp_path / 'reference')):
        left.pop('prefill_seconds'); right.pop('prefill_seconds')
        assert left == right
    updates = [json.loads(line) for line in (tmp_path / 'interrupted/updates.jsonl').read_text().splitlines()]
    assert [r['learning_rate'] for r in updates] == pytest.approx([.000025, .00005, .000075, .0001])
    assert [r['learning_rates']['timing'] for r in updates] == pytest.approx([.000075, .00015, .000225, .0003])
    assert [r['learning_rates']['clock_readout'] for r in updates] == pytest.approx([.0002, .0004, .0006, .0008])


def test_reject_unvalidated_model_envelope():
    with pytest.raises(ContractError, match='envelope'):
        validate_envelope(BackboneConfig(hidden=256))
    with pytest.raises(ContractError, match='envelope'):
        validate_envelope(BackboneConfig(), 3)


def test_training_rejects_insufficient_adamw_checkpoint_budget_before_first_update(tmp_path, monkeypatch):
    from pulsefield_model.research.oracle_time_continuation import train_run

    sha, path, manifest = write_inputs(tmp_path)
    cfg = compose_config([f'source_dir={tmp_path / "sources"}', f'split_manifest={path}',
                          f'split_sha256={manifest["sha256"]}', f'source_sha256=[{sha}]',
                          f'output_dir={tmp_path / "run"}', f'cache_dir={tmp_path / "cache"}',
                          'model.hidden=8', 'model.heads=2', 'resources.checkpoint_max_bytes=1'])
    def no_optimizer(*args, **kwargs):
        pytest.fail('Insufficient checkpoint capacity must be detected before constructing the optimizer')
    monkeypatch.setattr(train_run, 'SequenceTrainer', no_optimizer)
    with pytest.raises(ResourceLimit, match='AdamW moments'):
        run_training(cfg, resolved_yaml='small-budget')
    assert not (tmp_path / 'run/checkpoint.pt').exists()


def test_disk_stream_admission_matches_canonical_unsorted_crlf_source_and_rejects_conflicts(tmp_path):
    from pulsefield_model.research.oracle_time_continuation.data import admit_source
    objects = [(i % 4, i * 100, i * 100 + (50 if i % 3 == 0 else 0)) for i in range(60)]
    data = b'\xef\xbb\xbf' + source_bytes(list(reversed(objects))).replace(b'\n', b'\r\n')
    path = tmp_path / 'unsorted.osu'
    path.write_bytes(data)
    source = SourceStore(tmp_path / 'cache').admit(path, digest(data), group_id='song:test', split='train')
    expected = admit_source(data, digest(data), group_id='song:test', split='train')
    assert source.identity == expected.identity
    assert source.minimum_seed() == expected.minimum_seed()
    assert tuple(source.targets) == expected.targets
    for bad in ([(0, 0, 100), (0, 100, 100)], [(0, 10, 200), (0, 100, 100)], [(0, -1, -1)]):
        raw = source_bytes(bad)
        path.write_bytes(raw)
        with pytest.raises(ContractError):
            SourceStore(tmp_path / 'bad').admit(path, digest(raw), group_id='song:test', split='train')


def test_generation_hydra_and_full_cli_runner_consumption(tmp_path):
    from pulsefield_model.research.oracle_time_continuation.generate_hydra import compose_config as generate_config
    from pulsefield_model.research.oracle_time_continuation.generate_run import run_generation
    large = generate_config(config_name='oracle_time_generate_mac_large')
    assert large.resources.checkpoint_max_bytes == 1024**3
    assert large.resources.driver_limit_bytes == 6 * 1024**3
    assert large.checkpoint_every_rows == 1024
    sha, path, manifest = write_inputs(tmp_path)
    network = model()
    weights = tmp_path / 'weights.pt'
    from dataclasses import asdict
    torch.save(dict(format=WEIGHTS_FORMAT, model_config=asdict(network.config),
                    model_state_dict=network.state_dict()), weights)
    cfg = generate_config([f'source_dir={tmp_path / "sources"}', f'split_manifest={path}',
                           f'split_sha256={manifest["sha256"]}', f'source_sha256={sha}',
                           f'cache_dir={tmp_path / "cache"}', f'output_dir={tmp_path / "run"}',
                           f'weights={weights}', 'device=cpu', 'split=train', 'seed=91',
                           'decode.top_p=1', 'resources.check_every_rows=4', 'checkpoint_every_rows=13'])
    result = run_generation(cfg, resolved_yaml='resolved: yes\n')
    assert result['generated_rows'] > 0
    assert (tmp_path / 'run/resolved.yaml').read_text() == 'resolved: yes\n'
    metadata = json.loads((tmp_path / 'run/generation-config.json').read_text())
    assert metadata['policy']['top_p'] == 1 and metadata['seed'] == 91
    assert metadata['checkpoint_every_rows'] == 13
    cfg.resume = True
    assert run_generation(cfg, resolved_yaml='resume')['generated_rows'] == result['generated_rows']
    for override in ('+decode.unused=true', '+cache.unused=1', '+resources.unused=3', '+unexpected=4'):
        with pytest.raises(ContractError, match='Unknown'):
            generate_config([override])


def test_evaluation_scores_same_rows_as_reference_without_gradients(chord_source):
    from pulsefield_model.research.oracle_time_continuation.evaluation import evaluate_windows
    from pulsefield_model.research.oracle_time_continuation.engine import ContinuationEngine
    from pulsefield_model.research.oracle_time_continuation.windows import WindowSampler
    from pulsefield_model.research.oracle_time_continuation.model import row_index
    window = WindowSampler((chord_source,)).window(chord_source.identity.source_sha256,
                                                 chord_source.minimum_seed().seed_row_count, 2)
    network = model()
    result = evaluate_windows(network, (window,))
    with torch.no_grad():
        engine = ContinuationEngine(network)
        state = engine.prefill(chord_source.skeleton, chord_source.targets[:window.start], inference=True)
        expected = 0
        for row in chord_source.targets[window.start:window.stop]:
            expected -= engine.predict(state).log_probs[row_index(row.actions)].item()
            state = engine.commit(state, row)
    assert result['sequence_nll'] == pytest.approx(expected, abs=1e-5)
    assert all(p.grad is None for p in network.parameters())


def test_time_encoding_preserves_missing_zero_order_and_controls_long_duration_scale():
    from pulsefield_model.research.oracle_time_continuation.features import clock_features
    values = clock_features([None, 0., 1., 1000., 100_000., 4_000_000., 1e9], torch.zeros(1))
    assert torch.isfinite(values).all()
    assert values[0].abs().sum() == 0 and values[1, -1] == 1
    assert torch.all(values[2:, 0] < 1) and torch.all(values[2:, 0] > 0)
    assert torch.all(values[2:, 1][1:] > values[2:, 1][:-1])
    assert values[-1].norm() < 20
    assert values[5].norm() / values[4].norm() < 2


def test_bad_state_recovery_never_truncates_valid_output_tail(tmp_path):
    source = admit([(i % 4, i * 100, i * 100) for i in range(70)])
    network = model()
    output = tmp_path / 'run'
    with Rollout(network, source.skeleton, output) as run:
        run.prefill(source.targets[:30])
        run.advance(4)
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
    with (output / 'rows.jsonl').open('ab') as stream:
        stream.write(b'uncommitted tail')
    original = (output / 'rows.jsonl').read_bytes()
    checkpoint['next_index'] += 1
    torch.save(checkpoint, output / 'checkpoint.pt')
    with pytest.raises(ContractError, match='next event'):
        Rollout(network, source.skeleton, output, resume=True)
    assert (output / 'rows.jsonl').read_bytes() == original


def test_guard_failure_preserves_durable_update_and_restarts_same_draw(tmp_path, monkeypatch):
    from pulsefield_model.research.oracle_time_continuation import train_run
    sha, path, manifest = write_inputs(tmp_path)
    cfg = compose_config([f'source_dir={tmp_path / "sources"}', f'split_manifest={path}',
                          f'split_sha256={manifest["sha256"]}', f'source_sha256=[{sha}]',
                          f'cache_dir={tmp_path / "cache"}', f'output_dir={tmp_path / "interrupted"}',
                          'device=cpu', 'updates=2', 'model.hidden=8', 'model.heads=2', 'model.recent=4',
                          'model.coarse_group=2', 'model.coarse_capacity=2', 'model.coupling_rank=4'])
    actual = train_run.ResourceGuard.check
    def fail(self, phase, **kwargs):
        if phase == 'forward':
            raise ResourceLimit('injected memory guard')
        return actual(self, phase, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(train_run.ResourceGuard, 'check', fail)
        with pytest.raises(ResourceLimit):
            run_training(cfg, resolved_yaml='interrupted')
    checkpoint = torch.load(tmp_path / 'interrupted/checkpoint.pt', weights_only=True)
    assert checkpoint['updates'] == 0
    cfg.resume = True
    run_training(cfg, resolved_yaml='resumed')
    cfg.output_dir = str(tmp_path / 'reference')
    cfg.resume = False
    run_training(cfg, resolved_yaml='reference')
    a = torch.load(tmp_path / 'interrupted/weights.pt', weights_only=True)
    b = torch.load(tmp_path / 'reference/weights.pt', weights_only=True)
    for key in a['model_state_dict']:
        torch.testing.assert_close(a['model_state_dict'][key], b['model_state_dict'][key], atol=0, rtol=0)
