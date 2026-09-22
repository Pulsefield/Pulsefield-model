"""Saved-score audit, frozen-readout comparison and end-to-end 2x2 pilot.

All training arms share sampled cells and completed-update barriers. Checkpoint
selection uses only the primary human validation view; test graphs are blocked.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import monotonic

import torch
from torch.nn import functional as F

from .corpus import PreparedCorpus
from .dataset import ASSESSMENTS, CONCEPTS, ContractError, canonical_json, digest
from .metrics import assessment_report
from .probe_cache import BackboneCache, build_cache
from .probe_data import ProbeCorpus, input_identity, prepare_probes, sampled_cells
from .probe_metrics import (localized_inventory, paired_changes, prediction_rows, write_report)
from .probe_model import initialize_probe
from .temporal import temporal_sidecars
from .train import memory_snapshot, seed_update, source_record, synchronize, write_json


def audit(config, output):
    corpus = PreparedCorpus(Path(config.prepared_dir))
    lookup = {(r['layer'], r['cell_id']): r for r in corpus.records if r['split'] == 'validation'}
    inventory = localized_inventory(config.localized_trill_path, [r for r in corpus.records if r['split'] in ('train', 'validation')])
    write_json(output/'localized-trill.json', inventory)
    results, predictions, identities = {}, {}, {}
    for arm in ('style-only', 'style+evidence'):
        directory = Path(config.seed_run_dir)/arm
        checkpoint_sha = digest((directory/'best.pt').read_bytes())
        checkpoint = torch.load(directory/'best.pt', weights_only=True, map_location='cpu')
        if checkpoint['cohort_sha256'] != corpus.summary['cohort_sha256'] or checkpoint['split_sha256'] != corpus.summary['split_sha256']:
            raise ContractError('Saved logits checkpoint differs from the supplied cohort/split')
        if checkpoint['seed'] != 17 or bool(checkpoint['beta']) != (arm == 'style+evidence'):
            raise ContractError('Probe A requires the paired seed 17 arms')
        results[arm], predictions[arm] = {}, {}
        for layer in ('human', 'machine'):
            path = directory/f'{layer}-validation.json'
            saved = json.loads(path.read_text())['predictions']
            expected = {key for key in lookup if key[0] == layer}
            keys = [(r['layer'], r['cell_id']) for r in saved]
            if len(keys) != len(set(keys)) or set(keys) != expected:
                raise ContractError('Saved prediction cells do not exactly match validation')
            records = []
            for row, key in zip(saved, keys):
                record = lookup[key]
                for field in ('source_sha256', 'group_id', 'concept', 'assessment', 'record_ids', 'provenance_json'):
                    if row[field] != record[field]:
                        raise ContractError(f'Saved prediction {field} differs from cohort')
                records.append(record)
            rows = prediction_rows(records, [r['logits'] for r in saved])
            for row in rows:
                row['checkpoint_sha256'] = checkpoint_sha
            predictions[arm][layer] = rows
            results[arm][layer] = write_report(output/arm/layer, rows)
            identities[f'{arm}/{layer}'] = {'checkpoint_sha256': checkpoint_sha, 'prediction_sha256': digest(path.read_bytes())}
        primary = [r for r in predictions[arm]['human'] if r['concept'] != 'tech' or r['human_confidence'] == 'high']
        results[arm]['primary'] = write_report(output/arm/'primary-human', primary)
    changes = {layer: paired_changes(predictions['style-only'][layer], predictions['style+evidence'][layer])
               for layer in ('human', 'machine')}
    summary = {'stage': 'A', 'cohort_sha256': corpus.summary['cohort_sha256'], 'identities': identities,
               'arms': results, 'paired_changes': changes, 'localized_trill': inventory,
               'forward_passes': 0, 'test_evaluated': False}
    write_json(output/'summary.json', summary)
    return {'stage': 'A', 'output_dir': str(output), 'forward_passes': 0,
            'human_cells': len(predictions['style-only']['human']),
            'joint_inputs': len(results['style-only']['human']['joint_stream_jack']['cases']),
            'high_confidence_trill_cells': inventory['high_confidence_cells'],
            'short_gold_trill_sections': len(inventory['short_section_cases_under_2s']),
            'explicitly_localized_cells': inventory['explicitly_localized_cells'], 'test_evaluated': False}


def prepare_batch(config, corpus, indices, cache):
    chart, examples, unique, inverse = corpus.unique_batch(indices)
    records = [corpus.records[i] for i in unique]
    side = temporal_sidecars(examples, [corpus.timing[r['source_sha256']] for r in records],
                            [r['playback_rate'] for r in records], config.pace_radii, config.dilations)
    frozen = cache.batch(records).to(config.device) if cache else None
    queries, query_ids, positions = [], [], {}
    for index, input_index in zip(indices, inverse.tolist()):
        key = (input_index, CONCEPTS.index(corpus.records[index]['concept']))
        if key not in positions:
            positions[key] = len(queries)
            queries.append(key)
        query_ids.append(positions[key])
    return {'chart': chart.to(config.device), 'side': side.to(config.device), 'frozen': frozen,
            'input_indices': torch.tensor([q[0] for q in queries], device=config.device),
            'concepts': torch.tensor([q[1] for q in queries], device=config.device),
            'gather': torch.tensor(query_ids, device=config.device),
            'labels': torch.tensor([ASSESSMENTS.index(corpus.records[i]['assessment']) for i in indices], device=config.device)}


def forward(model, batch):
    encoded = model.encode(batch['chart'], batch['side'], batch['frozen'])
    return model.read(encoded, batch['chart'], batch['side'], batch['concepts'], batch['input_indices'])[batch['gather']]


def train_step(model, optimizer, batch, config, step):
    model.train()
    seed_update(config.seed, step)
    optimizer.zero_grad(set_to_none=True)
    logits = forward(model, batch)
    # The gather retains every sampled occurrence; unavailable readouts never
    # enter the denominator and complete inputs receive no additional targets.
    loss = F.cross_entropy(logits, batch['labels'])
    if not torch.isfinite(loss):
        raise ContractError('Nonfinite probe training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.gradient_cap,
                                         error_if_nonfinite=True)
    optimizer.step()
    return {'nll': loss.item(), 'gradient_norm': norm.item()}


@torch.no_grad()
def evaluate(model, config, corpus, indices, cache):
    model.eval()
    predictions = []
    for start in range(0, len(indices), config.batch_size):
        selected = indices[start:start+config.batch_size]
        batch = prepare_batch(config, corpus, selected, cache)
        predictions.extend(prediction_rows([corpus.records[i] for i in selected], forward(model, batch).cpu()))
    return predictions


def optimizer_for(model, config):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=config.learning_rate, weight_decay=config.weight_decay)


def checkpoint(arm, name, config, corpus, step):
    return {'model': {k: v.detach().cpu().clone() for k, v in arm['model'].state_dict().items()},
            'config': asdict(config), 'arm': name, 'updates': step, 'seed': config.seed,
            'cohort_sha256': corpus.summary['cohort_sha256'], 'split_sha256': corpus.summary['split_sha256'],
            'target_policy_sha256': corpus.summary['target_policy_sha256'],
            'selection': 'primary human group-macro three-class NLL; earliest tie', 'beta': 0}


def feasible_updates(maximum, cadence, seconds, step_seconds, evaluation_seconds, budget):
    """Reserve one terminal evaluation as well as the fixed-cadence evaluations."""
    for updates in range(maximum, -1, -1):
        if all(seconds[n]+updates*step_seconds[n]+(math.ceil(updates/cadence)+1)*evaluation_seconds[n] <= budget
               for n in seconds):
            return updates
    return 0


def compare(config, output, preparation_started):
    corpus = ProbeCorpus(Path(config.probe_prepared_dir))
    names = ('R0', 'R1') if config.stage == 'readout' else ('C0', 'CT', 'CM', 'CTM')
    cache = BackboneCache(config, corpus) if config.stage == 'readout' else None
    inventory = localized_inventory(config.localized_trill_path, [r for r in corpus.records if r['split'] in ('train', 'validation')])
    write_json(output/'localized-trill.json', inventory)
    arms = {}
    for name in names:
        model = initialize_probe(config, name, config.seed)
        if cache:
            model.encoder.load_state_dict(cache.encoder.state_dict())
            model.encoder.requires_grad_(False)
            model.encoder.eval()
        model.to(config.device)
        directory = output/name
        directory.mkdir()
        arms[name] = {'model': model, 'optimizer': optimizer_for(model, config), 'seconds': 0.0,
                      'best': float('inf'), 'best_update': None, 'directory': directory,
                      'training_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'parameters': sum(p.numel() for p in model.parameters()), 'peak_memory': {}}
    initial = {name: checkpoint(arm, name, config, corpus, 0) for name, arm in arms.items()}
    shared_seconds = monotonic()-preparation_started
    for arm in arms.values():
        arm['seconds'] = shared_seconds
    write_json(output/'cohort.json', {'cohort_sha256': corpus.summary['cohort_sha256'],
               'split_sha256': corpus.summary['split_sha256'], 'target_policy_sha256': corpus.summary['target_policy_sha256'],
               'selected_support': corpus.targets['support'], 'cache': cache.manifest if cache else None})
    # Save exact starting states, making copied common parameters inspectable.
    for name in names:
        torch.save(initial[name], arms[name]['directory']/'initial.pt')
    stream = sampled_cells(corpus.records, corpus.train_indices, config.seed, config.max_updates*config.batch_size)
    write_json(output/'sampled-cells.json', [{'index': i, 'layer': corpus.records[i]['layer'],
               'cell_id': corpus.records[i]['cell_id']} for i in stream])
    shared_seconds = monotonic()-preparation_started
    for arm in arms.values():
        arm['seconds'] = shared_seconds
    history, evaluation_seconds, last_primary = [], {}, {}

    def measure(step, *, select):
        for name, arm in arms.items():
            started = monotonic()
            rows = evaluate(arm['model'], config, corpus, corpus.primary_indices, cache)
            report = assessment_report(rows)
            value = report['macro_nll']
            if value is None or not math.isfinite(value):
                raise ContractError('Primary checkpoint metric must be finite with all concepts represented')
            if select and value < arm['best']:
                arm['best'], arm['best_update'] = value, step
                torch.save(checkpoint(arm, name, config, corpus, step), arm['directory']/'best.pt')
            synchronize(config.device)
            elapsed = monotonic()-started
            arm['seconds'] += elapsed
            evaluation_seconds[name] = max(evaluation_seconds.get(name, 0), elapsed)
            last_primary[name] = rows
            history.append({'arm': name, 'updates': step, 'primary_macro_nll': value,
                            'metrics': report, 'charged_seconds': arm['seconds'], 'selection_eligible': select})
        write_json(output/'validation-history.json', history)

    measure(0, select=True)
    step_seconds = {name: 0.0 for name in names}
    warmup = 0
    # Disposable throughput updates consume the same budget and are subtracted
    # from the 1000-update compute cap. Their parameters/optimizer state are reset.
    preflight_count = min(config.throughput_updates, max(0, config.max_updates-1))
    for step in range(preflight_count):
        if any(a['seconds'] >= config.arm_budget_seconds for a in arms.values()):
            break
        started = monotonic()
        batch = prepare_batch(config, corpus, stream[step*config.batch_size:(step+1)*config.batch_size], cache)
        synchronize(config.device)
        preparation = monotonic()-started
        for name, arm in arms.items():
            started = monotonic()
            train_step(arm['model'], arm['optimizer'], batch, config, step)
            synchronize(config.device)
            elapsed = preparation+monotonic()-started
            arm['seconds'] += elapsed
            step_seconds[name] = max(step_seconds[name], elapsed)
        warmup += 1
    for name, arm in arms.items():
        started = monotonic()
        arm['model'].load_state_dict(initial[name]['model'])
        arm['optimizer'] = optimizer_for(arm['model'], config)
        synchronize(config.device)
        arm['seconds'] += monotonic()-started
    del initial
    diagnostic_cells = 2*(len(corpus.train_indices)+len(corpus.indices('human', 'validation'))+
                          len(corpus.indices('machine', 'validation')))
    diagnostic_reserve = {name: evaluation_seconds[name]*diagnostic_cells/len(corpus.primary_indices)+20
                          for name in names}
    planned = feasible_updates(config.max_updates-warmup, config.evaluation_every,
                               {n: a['seconds']+diagnostic_reserve[n] for n, a in arms.items()}, step_seconds,
                               evaluation_seconds, config.arm_budget_seconds)
    write_json(output/'throughput.json', {'discarded_updates_per_arm': warmup, 'step_seconds': step_seconds,
               'primary_evaluation_seconds': evaluation_seconds, 'feasible_common_updates': planned,
               'estimated_final_diagnostics_seconds': diagnostic_reserve,
               'budget_seconds': config.arm_budget_seconds, 'shared_preparation_seconds': shared_seconds})
    print(f'{config.stage}: {planned} planned common updates after {warmup} disposable throughput updates', flush=True)
    completed, last_evaluated = 0, 0
    reason = 'feasible_common_updates'
    for step in range(planned):
        if any(a['seconds']+diagnostic_reserve[n] >= config.arm_budget_seconds for n, a in arms.items()):
            reason = 'wall_clock'
            break
        selected = stream[step*config.batch_size:(step+1)*config.batch_size]
        started = monotonic()
        batch = prepare_batch(config, corpus, selected, cache)
        synchronize(config.device)
        preparation = monotonic()-started
        logs = {}
        for name, arm in arms.items():
            started = monotonic()
            logs[name] = train_step(arm['model'], arm['optimizer'], batch, config, step)
            sample = memory_snapshot(config.device)
            for key, value in sample.items():
                arm['peak_memory'][key] = max(arm['peak_memory'].get(key, 0), value)
            arm['seconds'] += preparation+monotonic()-started
            logs[name]['charged_seconds'] = arm['seconds']
        completed = step+1
        with (output/'updates.jsonl').open('a') as file:
            file.write(canonical_json({'common_update': completed, 'arms': logs})+'\n')
        if completed % config.evaluation_every == 0 or completed == planned:
            measure(completed, select=True)
            last_evaluated = completed
            print(f'Common update {completed}/{planned}: '+', '.join(f'{n} NLL {arms[n]["best"]:.4f}' for n in names), flush=True)
    if completed != last_evaluated:
        measure(completed, select=False)
    summaries, all_results = {}, {}
    for name, arm in arms.items():
        started = monotonic()
        torch.save(checkpoint(arm, name, config, corpus, completed), arm['directory']/'final.pt')
        all_results[name] = {}
        for point in ('final', 'best'):
            saved = torch.load(arm['directory']/f'{point}.pt', weights_only=True, map_location='cpu')
            arm['model'].load_state_dict(saved['model'])
            point_dir = arm['directory']/point
            point_dir.mkdir()
            point_results = {}
            checkpoint_sha = digest((arm['directory']/f'{point}.pt').read_bytes())
            for layer in ('human', 'machine'):
                rows = evaluate(arm['model'], config, corpus, corpus.indices(layer, 'validation'), cache)
                for row in rows:
                    row['checkpoint_sha256'] = checkpoint_sha
                point_results[layer] = rows
                write_report(point_dir/layer, rows)
            primary_ids = {corpus.records[i]['cell_id'] for i in corpus.primary_indices}
            primary = [r for r in point_results['human'] if r['cell_id'] in primary_ids]
            write_report(point_dir/'primary-human', primary)
            point_results['primary'] = primary
            fit = evaluate(arm['model'], config, corpus, corpus.train_indices, cache)
            for row in fit:
                row['checkpoint_sha256'] = checkpoint_sha
            for layer in ('human', 'machine'):
                write_report(point_dir/f'training-{layer}', [r for r in fit if r['layer'] == layer], plots=False)
            inspect_positions(arm['model'], config, corpus, cache, point_dir, inventory)
            all_results[name][point] = point_results
        diagnostics = monotonic()-started
        arm['seconds'] += diagnostics
        summaries[name] = {'updates': completed, 'draws': completed*config.batch_size,
                           'discarded_throughput_updates': warmup, 'best_update': arm['best_update'],
                           'best_primary_macro_nll': arm['best'], 'charged_seconds': arm['seconds'],
                           'budget_overshoot_seconds': max(0, arm['seconds']-config.arm_budget_seconds),
                           'diagnostics_seconds': diagnostics, 'training_parameters': arm['training_parameters'],
                           'parameters': arm['parameters'], 'sampled_peak_memory': arm['peak_memory']}
    comparisons = {point: {name: paired_changes(all_results[names[0]][point]['primary'], all_results[name][point]['primary'])
                           for name in names[1:]} for point in ('final', 'best')}
    interaction = None
    if config.stage == 'pilot':
        interaction = {}
        for concept in CONCEPTS:
            losses = {n: assessment_report(all_results[n]['final']['primary'])['concepts'][concept]['nll'] for n in names}
            interaction[concept] = (losses['CM']-losses['CTM'])-(losses['C0']-losses['CT'])
    plot_curves(output, history)
    summary = {'stage': config.stage, 'common_updates': completed, 'planned_common_updates': planned,
               'stop_reason': reason, 'arms': summaries, 'paired_changes': comparisons,
               'final_interaction_by_concept': interaction, 'seed_uncertainty': 'unavailable: one pilot seed',
               'localized_trill': inventory, 'test_evaluated': False, 'convergence': 'bounded pilot; not established',
               'memory_accounting': 'Process/device-wide sampled maxima; MPS peaks between samples can be missed.'}
    write_json(output/'summary.json', summary)
    return {'stage': config.stage, 'output_dir': str(output), 'common_updates': completed, 'arms': summaries,
            'test_evaluated': False}


@torch.no_grad()
def inspect_positions(model, config, corpus, cache, directory, inventory):
    if model.composition is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted = {r['cell_id'] for r in inventory['intervals']+inventory['short_section_cases_under_2s']}
    indices = [i for i, r in enumerate(corpus.records) if r['split'] in ('train', 'validation') and
               r['layer'] == 'human' and r['concept'] == 'trill-organization' and
               (r['cell_id'] in wanted or (r['split'] == 'validation' and r['assessment'] != 'absent'))]
    model.eval()
    destination = directory/'position-scales'
    destination.mkdir()
    for index in indices:
        row = corpus.records[index]
        batch = prepare_batch(config, corpus, [index], cache)
        encoded = model.encode(batch['chart'], batch['side'], batch['frozen'])
        _, inspection = model.read(encoded, batch['chart'], batch['side'], batch['concepts'], batch['input_indices'], inspect=True)
        example = corpus.example(index)
        times = [r.time_ms for r in example.chart.inputs.rows]
        radii, radius = [0], 0
        for dilation in config.dilations:
            radius += dilation
            radii.append(radius)
        spans = [[[times[max(0, i-r)], times[min(len(times)-1, i+r)]] for r in radii] for i in range(len(times))]
        responses = inspection['responses'][0].cpu().tolist()
        write_json(destination/f'{row["cell_id"]}.json', {'cell': row, 'source_times_ms': times,
                   'composition_spans_ms': spans, 'responses': responses,
                   'candidate_mask': inspection['candidate_mask'][0].cpu().tolist(),
                   'attention_weights': inspection['weights'][0].cpu().tolist(),
                   'interpretation': 'Internal responses over contextual states; not local probabilities or causal attribution.'})
        fig, axes = plt.subplots(len(radii), 1, figsize=(9, 2*len(radii)), sharex=True, constrained_layout=True)
        for s, ax in enumerate(axes):
            ax.plot(times, [v[s] for v in responses], linewidth=.8)
            ax.axvline(row['scope']['start_ms'], color='black', linestyle='--')
            ax.axvline(row['scope']['end_ms'], color='black', linestyle='--')
            mask = inspection['candidate_mask'][0].cpu()
            candidates = [i for i, value in enumerate(mask) if value]
            if candidates:
                best = max(candidates, key=lambda i: responses[i][s])
                ax.axvspan(*spans[best][s], color='#2677b9', alpha=.15, label='Strongest anchor composition span')
            for interval in inventory['intervals']:
                if interval['cell_id'] == row['cell_id']:
                    ax.axvspan(interval['start_ms'], interval['end_ms'], color='#da8b21', alpha=.2)
            ax.set_ylabel(f'Scale {s} response')
        axes[-1].set_xlabel('Source milliseconds; dashed lines delimit target section')
        fig.savefig(destination/f'{row["cell_id"]}.png', dpi=110)
        plt.close(fig)


def plot_curves(output, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    updates = [json.loads(x) for x in (output/'updates.jsonl').read_text().splitlines()] if (output/'updates.jsonl').exists() else []
    for name in sorted({r['arm'] for r in history}):
        rows = [r for r in history if r['arm'] == name]
        axes[0].plot([r['updates'] for r in rows], [r['primary_macro_nll'] for r in rows], label=name)
        axes[1].plot([r['common_update'] for r in updates], [r['arms'][name]['nll'] for r in updates], label=name, alpha=.7)
    axes[0].set(xlabel='Common updates', ylabel='Primary human macro NLL')
    axes[1].set(xlabel='Common updates', ylabel='Sampled training NLL')
    axes[0].legend()
    fig.savefig(output/'curves.png', dpi=140)
    plt.close(fig)


def run_probe(config, *, resolved_yaml=None):
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    if config.stage == 'prepare':
        return prepare_probes(config)
    if config.stage != 'audit':
        if config.device == 'mps' and not torch.backends.mps.is_available():
            raise ContractError('Requested MPS device is unavailable')
        if config.device == 'cuda' and not torch.cuda.is_available():
            raise ContractError('Requested CUDA device is unavailable')
    if config.stage == 'cache':
        return build_cache(config, ProbeCorpus(Path(config.probe_prepared_dir)))
    started = monotonic()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', asdict(config))
    if resolved_yaml is not None:
        (output/'resolved-hydra.yaml').write_text(resolved_yaml)
    write_json(output/'source.json', source_record(output))
    try:
        return audit(config, output) if config.stage == 'audit' else compare(config, output, started)
    except Exception as exc:
        write_json(output/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
