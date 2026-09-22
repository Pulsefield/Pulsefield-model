"""Group-weighted ranking, same-input readouts, confidence and error inspection."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import ASSESSMENTS, CONCEPTS, ContractError, canonical_json
from .metrics import assessment_report, record_metrics, group_mean
from .probe_data import confidence, input_identity


def prediction_rows(records, logits):
    values = record_metrics(logits, [ASSESSMENTS.index(r['assessment']) for r in records])
    result = []
    for row, scores, metrics in zip(records, logits, values):
        scores = torch.as_tensor(scores, dtype=torch.float64)
        probabilities = scores.softmax(-1).tolist()
        result.append({**row, **metrics, 'logits': scores.tolist(), 'probabilities': probabilities,
                       'presence_score': sum(probabilities[1:]),
                       'strength_score': scores[1:].softmax(-1)[1].item(),
                       'human_confidence': confidence(row) if row['layer'] == 'human' else None,
                       'presence_error': ('false-negative' if row['assessment'] != 'absent' else 'false-positive')
                           if metrics['presence_predicted'] != int(row['assessment'] != 'absent') else None,
                       'class_error': metrics['predicted'] != ASSESSMENTS.index(row['assessment']),
                       'input_id': input_identity(row),
                       'section_seconds': (row['scope']['end_ms']-row['scope']['start_ms'])/1000/row['playback_rate']})
    return result


def weighted_ranking(rows, *, strength=False):
    members = [r for r in rows if not strength or r['assessment'] != 'absent']
    counts = Counter(r['group_id'] for r in members)
    weights = np.array([1/counts[r['group_id']] for r in members])
    labels = np.array([int(r['assessment'] == 'prominent' if strength else r['assessment'] != 'absent') for r in members])
    scores = np.array([r['strength_score' if strength else 'presence_score'] for r in members])
    support = {'cells': len(members), 'groups': len(counts),
               'positive_cells': int(labels.sum()), 'negative_cells': int(len(labels)-labels.sum())}
    if len(set(labels.tolist())) != 2:
        return {**support, 'auroc': None, 'average_precision': None, 'curve': []}
    positive, negative = weights[labels == 1].sum(), weights[labels == 0].sum()
    tp = fp = auc = ap = prev_tpr = prev_fpr = 0.0
    curve = []
    for score in sorted(set(scores.tolist()), reverse=True):
        tied = scores == score
        old_tp = tp
        tp += weights[tied & (labels == 1)].sum()
        fp += weights[tied & (labels == 0)].sum()
        tpr, fpr = tp/positive, fp/negative
        precision = tp/(tp+fp)
        auc += (fpr-prev_fpr)*(tpr+prev_tpr)/2
        ap += (tp-old_tp)/positive*precision
        curve.append({'threshold': score, 'recall': tpr, 'false_positive_rate': fpr, 'precision': precision})
        prev_tpr, prev_fpr = tpr, fpr
    return {**support, 'auroc': float(auc), 'average_precision': float(ap), 'curve': curve}


def joint_readouts(rows):
    inputs = defaultdict(dict)
    for row in rows:
        if row['concept'] in CONCEPTS[:2]:
            if row['concept'] in inputs[row['input_id']]:
                raise ContractError('Duplicate same-input concept output')
            inputs[row['input_id']][row['concept']] = row
    cases = []
    for identity, values in sorted(inputs.items()):
        if set(values) != set(CONCEPTS[:2]):
            continue
        jack, stream = (values[c] for c in CONCEPTS[:2])
        reference = [int(r['assessment'] != 'absent') for r in (jack, stream)]
        predicted = [r['presence_predicted'] for r in (jack, stream)]
        cases.append({'input_id': identity, 'group_id': jack['group_id'], 'reference_pair': reference,
                      'predicted_pair': predicted, 'joint_correct': int(reference == predicted),
                      'class_correct': [int(r['predicted'] == ASSESSMENTS.index(r['assessment'])) for r in (jack, stream)],
                      'jack': jack, 'stream': stream})
    combinations = {}
    for a in (0, 1):
        for b in (0, 1):
            subset = [r for r in cases if r['reference_pair'] == [a, b]]
            combinations[f'{a}/{b}'] = {'inputs': len(subset), 'groups': len({r['group_id'] for r in subset}),
                                        'joint_accuracy': group_mean(subset, 'joint_correct'),
                                        'joint_errors': sum(not r['joint_correct'] for r in subset)}
    return {'cases': cases, 'combinations': combinations, 'group_joint_accuracy': group_mean(cases, 'joint_correct')}


def views(rows):
    """One labeled layer per call; absent classes retain unavailable metrics."""
    report = assessment_report(rows)
    report['ranking'] = {c: {'presence': weighted_ranking([r for r in rows if r['concept'] == c]),
                             'strength': weighted_ranking([r for r in rows if r['concept'] == c], strength=True)} for c in CONCEPTS}
    report['joint_stream_jack'] = joint_readouts(rows)
    report['confidence'] = {str(conf): assessment_report([r for r in rows if r['human_confidence'] == conf])
                            for conf in (None, 'low', 'high')}
    report['section_length'] = {label: assessment_report([r for r in rows if lower <= r['section_seconds'] < upper])
                               for label, lower, upper in (('under-2s', 0, 2), ('2-5s', 2, 5),
                                                          ('5-10s', 5, 10), ('10s-plus', 10, float('inf')))}
    return report


def localized_inventory(path, rows):
    gold = [r for r in rows if r['layer'] == 'human' and r['concept'] == 'trill-organization'
            and confidence(r) == 'high']
    section_cases = [{k: r[k] for k in ('cell_id', 'record_ids', 'source_sha256', 'scope', 'context',
                                       'playback_rate', 'assessment', 'split')}
                     for r in gold if r['assessment'] != 'absent']
    section_cases.sort(key=lambda r: r['scope']['end_ms']-r['scope']['start_ms'])
    gold_inventory = {'high_confidence_cells': len(gold), 'positive_section_cases': section_cases,
                      'short_section_cases_under_2s': [r for r in section_cases
                         if (r['scope']['end_ms']-r['scope']['start_ms'])/r['playback_rate'] < 2000],
                      'section_case_interpretation': 'Human-labeled section scopes; not internal episode boundaries in a larger labeled section.'}
    if path is None:
        return {**gold_inventory, 'status': 'section-judgments-only', 'explicitly_localized_cells': 0, 'intervals': [],
                'reason': 'Section judgments and evidence selections do not define exhaustive local intervals.'}
    entries = json.loads(Path(path).read_text())
    lookup = {(r['layer'], r['cell_id']): r for r in rows}
    for entry in entries:
        row = lookup.get(('human', entry['cell_id']))
        if row is None or row['concept'] != 'trill-organization' or row['assessment'] == 'absent':
            raise ContractError('Localized intervals require a positive human Trill train/validation cell')
        if any(entry[k] != row[k] for k in ('source_sha256', 'scope', 'context', 'playback_rate')):
            raise ContractError('Localized interval input identity differs from its human judgment')
        if not row['scope']['start_ms'] <= entry['start_ms'] < entry['end_ms'] <= row['scope']['end_ms']:
            raise ContractError('Localized interval must lie within its labeled section')
        if not entry.get('human_reference'):
            raise ContractError('Localized intervals require an explicit human_reference')
    return {**gold_inventory, 'status': 'provided', 'explicitly_localized_cells': len({r['cell_id'] for r in entries}), 'intervals': entries}


def paired_changes(left, right):
    a = {(r['layer'], r['cell_id']): r for r in left}
    b = {(r['layer'], r['cell_id']): r for r in right}
    if a.keys() != b.keys():
        raise ContractError('Compared prediction populations differ')
    changes = []
    for key, x in a.items():
        y = b[key]
        if any(x[k] != y[k] for k in ('input_id', 'assessment', 'group_id', 'record_ids')):
            raise ContractError('Compared prediction identities/targets differ')
        changes.append({'layer': key[0], 'cell_id': key[1], 'concept': x['concept'], 'group_id': x['group_id'],
                        'nll_gain': x['nll']-y['nll'], 'presence_before': x['presence_predicted'],
                        'presence_after': y['presence_predicted']})
    return {'per_concept_nll_gain': {c: group_mean([r for r in changes if r['concept'] == c], 'nll_gain') for c in CONCEPTS},
            'cells': changes}


def write_report(directory, rows, *, plots=True):
    from .train import write_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = views(rows)
    write_json(directory/'scores.json', {'metrics': report, 'predictions': rows})
    fields = ('layer', 'cell_id', 'record_ids', 'source_sha256', 'group_id', 'input_id', 'chart_key', 'checkpoint_sha256',
              'scope', 'context', 'playback_rate', 'concept', 'assessment', 'human_confidence',
              'section_seconds', 'probabilities', 'presence_score', 'strength_score', 'predicted',
              'presence_predicted', 'strength_predicted', 'presence_error', 'class_error',
              'nll', 'presence_nll', 'strength_nll', 'provenance_json')
    with (directory/'section-errors.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical_json(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fields})
    if plots:
        plot_scores(directory, rows, report)
    return report


def plot_scores(directory, rows, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(15, 5), constrained_layout=True)
    for j, concept in enumerate(CONCEPTS):
        members = [r for r in rows if r['concept'] == concept]
        for label, color in zip(ASSESSMENTS, ('#657786', '#da8b21', '#2677b9')):
            selected = [r for r in members if r['assessment'] == label]
            for k, score in enumerate(('presence_score', 'strength_score')):
                data = [r[score] for r in selected if k == 0 or label != 'absent']
                axes[k, j].scatter(data, np.arange(len(data)), label=label, s=16, color=color)
                axes[k, j].axvline(.5, color='black', linewidth=.5)
                axes[k, j].set_xlim(0, 1)
                axes[k, j].set_xlabel('Presence' if k == 0 else 'Strength | reference positive')
        axes[0, j].set_title(concept.replace('-organization', ''))
    axes[0, 0].legend(fontsize=7)
    fig.savefig(directory/'scores.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for case in report['joint_stream_jack']['cases']:
        x, y = case['jack']['presence_score'], case['stream']['presence_score']
        ax.scatter(x, y, marker='o' if case['joint_correct'] else 'x')
        ax.annotate('/'.join(map(str, case['reference_pair'])), (x, y), fontsize=7)
    ax.axhline(.5, color='gray', linewidth=.7)
    ax.axvline(.5, color='gray', linewidth=.7)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Jack presence score', ylabel='Stream presence score',
           title='Same-input readouts; labels are reference Jack/Stream')
    fig.savefig(directory/'joint-readouts.png', dpi=140)
    plt.close(fig)
