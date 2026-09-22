from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from ensomi_model.research.scoped_style_modeling.config import ModelConfig
from ensomi_model.research.scoped_style_modeling.dataset import CONCEPTS, ContractError, canonical_json, digest
from ensomi_model.research.scoped_style_modeling.model import initialize_model
from ensomi_model.research.scoped_style_modeling.probe_config import ProbeConfig
from ensomi_model.research.scoped_style_modeling.probe_data import (ProbeCorpus, TARGET_POLICY, TENSOR_VERSION,
    confidence, input_identity, select_targets, sampled_cells, support)
from ensomi_model.research.scoped_style_modeling.probe_model import initialize_probe
from ensomi_model.research.scoped_style_modeling.probe_hydra import compose_config
from ensomi_model.research.scoped_style_modeling.probe_metrics import (prediction_rows, weighted_ranking,
    joint_readouts, localized_inventory, paired_changes)
from ensomi_model.research.scoped_style_modeling.probe_cache import build_cache, BackboneCache
from ensomi_model.research.scoped_style_modeling.probes import run_probe, prepare_batch, forward, feasible_updates
from ensomi_model.research.scoped_style_modeling.temporal import (parse_redlines, temporal_sidecars, TimeCoordinates)
from ensomi_model.research.scoped_style_modeling.tensors import collate
from ensomi_model.research.scoped_style_modeling.train import write_json
from test_model import fixture, DEVICES
from test_training import corpus_fixture


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def record(layer='human', concept='tech', confidence_value='high', **changes):
    return {'layer': layer, 'cell_id': 'cell-'+concept, 'record_ids': ['effective'],
            'source_sha256': 'source', 'scope': {'start_ms': 100, 'end_ms': 400},
            'context': {'start_ms': 50, 'end_ms': 500}, 'playback_rate': 1.0,
            'concept': concept, 'assessment': 'supporting', 'group_id': 'g', 'split': 'train',
            'chart_key': 'chart', 'provenance_json': canonical_json([{'record_id': 'effective',
              'human_confidence': confidence_value, 'human_revision': {'human_confidence': 'high'}}]), **changes}


def test_policy_precedence_confidence_and_no_unresolved_fallback():
    rows = [record('machine', 'tech'), record(confidence_value='low'),
            record('machine', CONCEPTS[0]), record(concept=CONCEPTS[0], confidence_value=None),
            record('machine', CONCEPTS[1]), record('human', CONCEPTS[2]), record('machine', CONCEPTS[4])]
    chosen, decisions = select_targets(rows, [{'layer': 'human', 'cell_id': 'cell-'+CONCEPTS[1], 'kind': 'unsupervised-cell'}])
    assert chosen == [3, 5, 6]
    assert confidence(rows[1]) == 'low'
    rows[1] = record(confidence_value='high')
    assert select_targets(rows, [])[0] == [1, 3, 4, 5, 6]
    assert decisions[4]['reason'] == 'unresolved-or-conflicting-human'
    changed_context = {**rows[3], 'context': {'start_ms': 0, 'end_ms': 1000}}
    assert select_targets([rows[2], changed_context], [])[0] == [1]
    assert input_identity(changed_context) != input_identity(rows[3])
    with pytest.raises(ContractError, match='1x'):
        select_targets([{**rows[1], 'playback_rate': 1.5}], [])


def test_redlines_ignore_sv_and_ratio_availability_and_rate():
    data = b'[TimingPoints]\n0,500,4,1,0,80,1,0\n100,-50,4,1,0,80,0,0\n200,250,4,1,0,80,1,0\n'
    points = parse_redlines(data, digest(data))
    assert points == [(0, 500), (200, 250)]
    item = fixture()
    a = TimeCoordinates(item.chart, points, 1, (1, 2))
    b = TimeCoordinates(item.chart, points, 2, (1, 2))
    assert a.at(100)[0] is None  # The attack at 20 ms is outside review context.
    assert a.at(250)[3] == 250
    x, y = a.describe(-100, 250), b.describe(-100, 250)
    assert x[0] != y[0] and x[2:] == y[2:]
    assert a.describe(None, 250) == [0, False]*4
    assert a.describe(0, 250)[1::2] == [True]*4
    assert a.row(item.chart.inputs.rows[0])[9] is False
    assert a.at(-1)[3] is None
    with pytest.raises(ContractError, match='SHA-256'):
        parse_redlines(data, 'wrong')
    bad = b'[TimingPoints]\n0,0,4,1,0,80,1,0'
    with pytest.raises(ContractError, match='positive'):
        parse_redlines(bad, digest(bad))
    future = data+b'\n10000,-500,4,1,0,80,1,0'
    assert parse_redlines(future, digest(future), last_query_ms=9999) == points


def model_batch(examples, device='cpu'):
    chart = collate(examples).chart.to(device)
    side = temporal_sidecars(examples, [[(0, 500), (200, 250)]]*len(examples), [1]*len(examples), [2, 8], [1, 2, 4]).to(device)
    return chart, side


@pytest.mark.parametrize('name', ['C0', 'CT', 'CM', 'CTM'])
@pytest.mark.parametrize('device', DEVICES)
def test_probe_padding_empty_and_mirror(name, device):
    config = ProbeConfig()
    model = initialize_probe(config, name, 17).to(device).eval()
    outputs = []
    for examples in ([fixture()], [fixture(), fixture(long=True)], [fixture(mirror=True)], [fixture(empty=True)]):
        chart, side = model_batch(examples, device)
        encoded = model.encode(chart, side)
        result = model.read(encoded, chart, side, torch.zeros(len(examples), dtype=torch.long, device=device),
                            torch.arange(len(examples), device=device), inspect=True)
        logits, inspection = result
        assert torch.isfinite(logits).all()
        outputs.append(logits[0])
        if inspection is not None:
            assert not inspection['weights'][~inspection['candidate_mask']].count_nonzero()
            if not chart.section_events.any():
                assert inspection['weights'].count_nonzero() == 0
    torch.testing.assert_close(outputs[0], outputs[1], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(outputs[0], outputs[2], atol=3e-6, rtol=3e-5)


def test_initialization_all_shared_parameters_match_and_rng_is_preserved():
    rng = torch.random.get_rng_state().clone()
    models = {n: initialize_probe(ProbeConfig(), n, 17) for n in ('C0', 'CT', 'CM', 'CTM')}
    assert torch.equal(rng, torch.random.get_rng_state())
    for a, b in (('C0', 'CT'), ('C0', 'CM'), ('CT', 'CTM'), ('CM', 'CTM')):
        x, y = models[a].state_dict(), models[b].state_dict()
        for key in x.keys() & y.keys():
            assert torch.equal(x[key], y[key]), (a, b, key)
    chart, side = model_batch([fixture()])
    original = initialize_model(ModelConfig(), 17, auxiliary=False).eval()
    probe = models['C0'].eval()
    torch.testing.assert_close(original.assessment(chart, torch.tensor([0])),
                              probe.read(probe.encode(chart, side), chart, side, torch.tensor([0]), torch.tensor([0])), rtol=0, atol=0)


def test_local_gradient_all_scales_and_target_isolation():
    model = initialize_probe(ProbeConfig(), 'CTM', 17).eval()
    chart, side = model_batch([fixture()])
    encoded = model.encode(chart, side)
    encoded[1].retain_grad()
    logits, inspection = model.read(encoded, chart, side, torch.tensor([2]), torch.tensor([0]), inspect=True)
    logits.sum().backward()
    assert all(encoded[1].grad[:, :, s].abs().sum() > 0 for s in range(4))
    assert all(block.weight.grad.abs().sum() > 0 for block in model.composition.blocks)
    changed, changed_side = model_batch([replace(fixture(), assessment=2, masks=None)])
    torch.testing.assert_close(logits, model.read(model.encode(changed, changed_side), changed, changed_side,
                                                 torch.tensor([2]), torch.tensor([0])))


def test_ranking_ties_group_weights_and_missed_strength():
    rows = prediction_rows([record(assessment='prominent'), record(assessment='absent', group_id='h')], [[8, 0, 1], [8, 0, 1]])
    rank = weighted_ranking(rows)
    assert rank['auroc'] == .5 and rank['average_precision'] == .5
    duplicated = rows+[dict(rows[0])]*5
    assert weighted_ranking(duplicated)['average_precision'] == pytest.approx(.5)
    assert rows[0]['presence_predicted'] == 0 and rows[0]['strength_nll'] is not None
    assert weighted_ranking(rows, strength=True)['auroc'] is None
    assert weighted_ranking(rows[:1])['average_precision'] is None


def test_same_input_pairs_and_local_gold_inventory():
    rows = prediction_rows([record(concept=CONCEPTS[0]), record(concept=CONCEPTS[1])], [[0, 1, 2], [3, 0, 0]])
    joint = joint_readouts(rows)
    assert joint['cases'][0]['reference_pair'] == [1, 1]
    assert joint['cases'][0]['predicted_pair'] == [1, 0]
    assert joint['combinations']['1/1']['joint_errors'] == 1
    shifted = {**rows[1], 'input_id': 'different-context'}
    assert joint_readouts([rows[0], shifted])['cases'] == []
    gold = localized_inventory(None, [record(concept=CONCEPTS[2])])
    assert len(gold['short_section_cases_under_2s']) == 1
    assert gold['explicitly_localized_cells'] == 0
    with pytest.raises(ContractError, match='identities'):
        paired_changes(rows, [rows[0], {**rows[1], 'input_id': 'other'}])


def probe_fixture(tmp_path):
    root = corpus_fixture(tmp_path)
    original = json.loads((root/'summary.json').read_text())
    rows = [json.loads(x) for x in (root/'assessment-cohort.jsonl').read_text().splitlines()]
    for row in rows:
        row.update(scope={'start_ms': 100, 'end_ms': 400}, context={'start_ms': 50, 'end_ms': 500}, playback_rate=1.0)
        row['provenance_json'] = canonical_json([{'record_id': row['record_ids'][0], 'human_confidence': 'high'}])
    human_tech = {**rows[3], 'layer': 'human', 'record_ids': ['human-tech-train']}
    human_tech['provenance_json'] = canonical_json([{'record_id': 'human-tech-train', 'human_confidence': 'high'}])
    rows.append(human_tech)
    data = ''.join(canonical_json(r)+'\n' for r in rows).encode()
    (root/'assessment-cohort.jsonl').write_bytes(data)
    chosen, decisions = select_targets(rows, [])
    write_json(root/'target-policy.json', {'selected_indices': chosen, 'decisions': decisions, 'adapter_issues': [],
                                          'support': support(rows, chosen)})
    write_json(root/'source-timing.json', {r['source_sha256']: [[0, 500]] for r in rows if r['split'] != 'test'})
    summary = {**original, 'schema_version': 2, 'tensor_version': TENSOR_VERSION, 'target_policy': TARGET_POLICY,
               'cohort_sha256': digest(data), 'parent_cohort_sha256': digest(data),
               'source_timing_sha256': digest((root/'source-timing.json').read_bytes()),
               'target_policy_sha256': digest((root/'target-policy.json').read_bytes())}
    write_json(root/'summary.json', summary)
    config = ProbeConfig(probe_prepared_dir=str(root), output_dir=str(tmp_path/'run'), device='cpu',
                         max_updates=2, throughput_updates=1, evaluation_every=1, batch_size=3,
                         cache_dir=str(tmp_path/'cache'), checkpoint=str(tmp_path/'backbone.pt'))
    torch.save({'model': initialize_model(config.model, 17, auxiliary=False).state_dict(), 'config': asdict(config.model),
                'cohort_sha256': summary['parent_cohort_sha256'], 'split_sha256': summary['split_sha256'],
                'beta': 0, 'seed': 17}, config.checkpoint)
    return config, ProbeCorpus(root)


def test_deduplicated_inputs_loss_multiplicity_and_sampler(tmp_path):
    config, corpus = probe_fixture(tmp_path)
    indices = [corpus.train_indices[0]]*3+[corpus.train_indices[1]]
    batch = prepare_batch(config, corpus, indices, None)
    assert len(batch['chart'].lengths) == 1
    assert batch['gather'].tolist() == [0, 0, 0, 1]
    model = initialize_probe(config, 'C0', 17).eval()
    logits = forward(model, batch)
    torch.testing.assert_close(logits[0], logits[2])
    loss = torch.nn.functional.cross_entropy(logits, batch['labels'], reduction='none')
    assert loss.mean() == (3*loss[0]+loss[3])/4
    sampled = sampled_cells(corpus.records, corpus.train_indices, 17, 100)
    assert sampled == sampled_cells(corpus.records, corpus.train_indices, 17, 100)
    assert any(corpus.records[i]['layer'] == 'human' for i in sampled)
    with pytest.raises(ContractError, match='test'):
        corpus.example(corpus.indices('human', 'test')[0])


def test_disk_cache_exact_states_and_stale_rejection(tmp_path):
    config, corpus = probe_fixture(tmp_path)
    result = build_cache(config, corpus)
    cache = BackboneCache(config, corpus)
    row = corpus.records[corpus.train_indices[0]]
    states = cache.batch([row])
    chart, _, _, _ = corpus.unique_batch([corpus.train_indices[0]])
    torch.testing.assert_close(states, cache.encoder(chart))
    assert states.device.type == 'cpu' and result['bytes'] > 0
    manifest = json.loads((Path(config.cache_dir)/'manifest.json').read_text())
    manifest['checkpoint_sha256'] = 'changed'
    write_json(Path(config.cache_dir)/'manifest.json', manifest)
    with pytest.raises(ContractError, match='identity'):
        BackboneCache(config, corpus)


@pytest.mark.parametrize('stage,names', [('readout', ['R0', 'R1']), ('pilot', ['C0', 'CT', 'CM', 'CTM'])])
def test_comparison_end_to_end_final_selected_and_frozen_backbone(tmp_path, monkeypatch, stage, names):
    from ensomi_model.research.scoped_style_modeling import probe_metrics, probes
    monkeypatch.setattr(probe_metrics, 'plot_scores', lambda *args: None)
    monkeypatch.setattr(probes, 'plot_curves', lambda *args: None)
    config, corpus = probe_fixture(tmp_path)
    config.stage = stage
    if stage == 'readout':
        build_cache(config, corpus)
    result = run_probe(config, resolved_yaml='stage: '+stage+'\n')
    assert result['common_updates'] == 1
    assert result['test_evaluated'] is False
    assert set(result['arms']) == set(names)
    for name in names:
        folder = Path(config.output_dir)/name
        initial = torch.load(folder/'initial.pt', weights_only=True)['model']
        final = torch.load(folder/'final.pt', weights_only=True)['model']
        assert any(not torch.equal(initial[k], final[k]) for k in initial if k.startswith('assessor.'))
        if stage == 'readout':
            assert all(torch.equal(initial[k], final[k]) for k in initial if k.startswith('encoder.'))
        assert (folder/'best/primary-human/scores.json').exists()
        assert (folder/'final/primary-human/section-errors.csv').exists()
    with pytest.raises(FileExistsError):
        run_probe(config)


def test_budget_and_hydra_limits_and_resources():
    assert feasible_updates(1000, 100, {'a': 10}, {'a': 2}, {'a': 5}, 900) == 430
    assert feasible_updates(1000, 100, {'a': 901}, {'a': 2}, {'a': 5}, 900) == 0
    assert compose_config() == ProbeConfig()
    assert compose_config(config_name='scoped_style_probe_b').stage == 'readout'
    assert compose_config(config_name='scoped_style_pilot_c').arm_budget_seconds == 1800
    for override in ('+ignored=1', '+model.ignored=1', 'pace_radii=[0,8]', 'dilations=[]', 'max_updates=0'):
        with pytest.raises(Exception):
            compose_config([override])
    with pytest.raises(ContractError, match='caps'):
        compose_config(['max_updates=1001'], config_name='scoped_style_probe_b')
