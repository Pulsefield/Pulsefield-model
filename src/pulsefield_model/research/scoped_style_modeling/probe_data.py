"""Versioned human-priority targets and exact-input reuse for development probes."""
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import sys
from time import monotonic

from .corpus import PreparedCorpus
from .dataset import (CONCEPTS, ContractError, canonical_json, checked_bytes, digest, load_snapshot,
                      adapt_records)
from .temporal import parse_redlines

TARGET_POLICY = 'human-priority-high-tech-v1'
TENSOR_VERSION = 'scoped-style-probes-v2'


def confidence(row):
    """Use the context-owning effective observation, never its revision ancestry."""
    provenance = json.loads(row['provenance_json'])
    representative = row['record_ids'][0]
    chosen = [p for p in provenance if p['record_id'] == representative]
    if len(chosen) != 1:
        raise ContractError(f'{row["cell_id"]}: missing or duplicate representative provenance')
    value = chosen[0].get('human_confidence')
    if value not in (None, 'low', 'high'):
        raise ContractError(f'{row["cell_id"]}: unsupported human confidence {value}')
    return value


def exact_cell(row):
    return (row['source_sha256'], row['scope']['start_ms'], row['scope']['end_ms'],
            row['concept'], row['playback_rate'])


def input_identity(row):
    return digest(canonical_json([row['source_sha256'], row['scope'], row['context'], row['playback_rate']]).encode())


def select_targets(records, issues):
    """Block fallback at unresolved human cells even when no human row survives."""
    blocked_ids = {i['cell_id'] for i in issues if i.get('layer') == 'human' and
                   i['kind'] in ('unsupervised-cell', 'conflicting-cell')}
    humans = {exact_cell(r) for r in records if r['layer'] == 'human'}
    selected, decisions = [], []
    for index, row in enumerate(records):
        if row['split'] not in ('train', 'validation'):
            continue
        if row['playback_rate'] != 1:
            raise ContractError('The frozen adapter admits only 1x; do not transfer labels across rates')
        reason = None
        if row['cell_id'] in blocked_ids:
            reason = 'unresolved-or-conflicting-human'
        elif row['concept'] == 'tech' and (row['layer'] != 'human' or confidence(row) != 'high'):
            reason = 'tech-requires-effective-high-human'
        elif row['layer'] == 'machine' and exact_cell(row) in humans:
            reason = 'human-precedence'
        if reason is None:
            selected.append(index)
        decisions.append({'index': index, 'layer': row['layer'], 'cell_id': row['cell_id'],
                          'selected': reason is None, 'reason': reason})
    keys = [exact_cell(records[i]) for i in selected]
    if len(keys) != len(set(keys)):
        raise ContractError('Target selection produced duplicate exact cells')
    return selected, decisions


def support(records, indices):
    result = []
    for split in ('train', 'validation'):
        for concept in CONCEPTS:
            rows = [records[i] for i in indices if records[i]['split'] == split and records[i]['concept'] == concept]
            result.append({'split': split, 'concept': concept, 'cells': len(rows),
                           'groups': len({r['group_id'] for r in rows}),
                           'classes': dict(Counter(r['assessment'] for r in rows)),
                           'layers': dict(Counter(r['layer'] for r in rows))})
    return result


def prepare_probes(config):
    """Create a fresh preparation with verified redline sidecars and unchanged splits.

    Publication exclusions must be limited to sources outside the published
    cohort. A new exclusion category requires an identity-aware adapter first.
    Only training and validation graphs and source files are opened.
    """
    from .train import write_json
    started = monotonic()
    parent = PreparedCorpus(Path(config.prepared_dir))
    preparation = json.loads((parent.root/'config.json').read_text())
    manifest, sources, tables = load_snapshot(Path(preparation['dataset_dir']))
    _, issues = adapt_records(tables, sources)
    exclusions = manifest['exclusions']['human']
    unsupported = {k: v for k, v in exclusions.items() if v and k != 'source-excluded'}
    if unsupported:
        raise ContractError(f'Human publication exclusions need exact-cell blocking: {unsupported}')
    selected, decisions = select_targets(parent.records, issues)
    last_queries = {}
    for row in parent.records:
        if row['split'] in ('train', 'validation'):
            sha = row['source_sha256']
            last_queries[sha] = max(last_queries.get(sha, -float('inf')), row['context']['end_ms'])
    # Timing after every declared query cannot affect these inputs. Validate all
    # required source timing before creating the output directory.
    timing = {sha: parse_redlines(checked_bytes(Path(preparation['source_cache'])/f'{sha}.osu', sha),
                                 sha, last_query_ms=end) for sha, end in last_queries.items()}
    output = Path(config.probe_prepared_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'contexts').mkdir()
    for name in ('assessment-cohort.jsonl', 'split-manifest.json', 'config.json'):
        shutil.copyfile(parent.root/name, output/name)
    contexts = {}
    for row in parent.records:
        if row['split'] not in ('train', 'validation'):
            continue
        key = row['chart_key']
        if key not in contexts:
            # Verify every copied graph once, including diagnostic-only inputs.
            parent.chart(key, row['chart_sha256'])
            shutil.copyfile(parent.root/'contexts'/f'{key}.json.gz', output/'contexts'/f'{key}.json.gz')
            contexts[key] = row['chart_sha256']
            if len(contexts) % 100 == 0:
                print(f'Verified {len(contexts)} train/validation contexts', flush=True)
    write_json(output/'source-timing.json', timing)
    target = {'policy': TARGET_POLICY, 'selected_indices': selected, 'decisions': decisions,
              'support': support(parent.records, selected), 'publication_human_exclusions': exclusions,
              'adapter_issues': issues, 'test_selected': False}
    write_json(output/'target-policy.json', target)
    summary = {**parent.summary, 'schema_version': 2, 'tensor_version': TENSOR_VERSION, 'command': sys.argv,
               'parent_cohort_sha256': parent.summary['cohort_sha256'],
               'source_timing_sha256': digest((output/'source-timing.json').read_bytes()),
               'target_policy_sha256': digest((output/'target-policy.json').read_bytes()),
               'target_policy': TARGET_POLICY, 'prepared_splits': ['train', 'validation'],
               'timing_source_count': len(timing), 'runtime_seconds': monotonic()-started}
    write_json(output/'summary.json', summary)
    write_json(output/'probe-config.json', vars(config) | {'model': vars(config.model)})
    return {'output_dir': str(output), 'timing_sources': len(timing), 'contexts': len(contexts),
            'selected_support': target['support'], 'runtime_seconds': summary['runtime_seconds']}


class ProbeCorpus(PreparedCorpus):
    def __init__(self, root):
        super().__init__(root)
        if self.summary.get('tensor_version') != TENSOR_VERSION or self.summary.get('target_policy') != TARGET_POLICY:
            raise ContractError('Run stage=prepare to create the versioned probe preparation')
        self.timing = json.loads(checked_bytes(self.root/'source-timing.json', self.summary['source_timing_sha256']))
        self.targets = json.loads(checked_bytes(self.root/'target-policy.json', self.summary['target_policy_sha256']))
        selected, decisions = select_targets(self.records, self.targets['adapter_issues'])
        if selected != self.targets['selected_indices'] or decisions != self.targets['decisions']:
            raise ContractError('Selected targets differ from the recorded policy')
        self.train_indices = [i for i in selected if self.records[i]['split'] == 'train']
        self.primary_indices = [i for i in selected if self.records[i]['split'] == 'validation' and self.records[i]['layer'] == 'human']
        for indices, name in ((self.train_indices, 'training'), (self.primary_indices, 'primary validation')):
            if {self.records[i]['concept'] for i in indices} != set(CONCEPTS):
                raise ContractError(f'{name} requires represented targets for all five concepts')

    def example(self, index):
        if self.records[index]['split'] not in ('train', 'validation'):
            raise ContractError('Probe code cannot open test charts')
        return super().example(index)

    def unique_batch(self, indices):
        """Encode distinct inputs once and return each sampled cell's gather index."""
        import torch
        from .tensors import collate
        unique, inverse, positions = [], [], {}
        for index in indices:
            identity = input_identity(self.records[index])
            if identity not in positions:
                positions[identity] = len(unique)
                unique.append(index)
            inverse.append(positions[identity])
        # Chart tensor construction receives harmless placeholders; actual targets
        # stay in the sampled-cell list and are never supplied to feature code.
        examples = [replace(self.example(i), concept=0, assessment=0, masks=None) for i in unique]
        return collate(examples).chart, examples, unique, torch.tensor(inverse)


def sampled_cells(records, indices, seed, count):
    groups = defaultdict(lambda: defaultdict(list))
    for i in indices:
        row = records[i]
        if row['split'] != 'train':
            raise ContractError('Sampling may only consume training targets')
        groups[row['concept']][row['group_id']].append(i)
    if set(groups) != set(CONCEPTS):
        raise ContractError('Sampling requires all five concepts')
    ids = {c: sorted(groups[c]) for c in CONCEPTS}
    rng = random.Random(f'scoped-style-probe:{seed}')
    return [rng.choice(groups[c][rng.choice(ids[c])]) for c in (rng.choice(CONCEPTS) for _ in range(count))]
