"""Seed-only free-running generation with atomic state/output recovery."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
from time import perf_counter

import torch

from ..scoped_style_modeling.dataset import ContractError
from .checkpoint import pack_state, restore_state, runtime_identity
from .decoding import DecodeSamplingPolicy, sample_row
from .engine import ContinuationEngine
from .export import export_osu, verify_rows
from .runtime import (ResourceConfig, ResourceGuard, align_boundary, atomic_checkpoint, log_boundary,
                      tensor_bytes, validate_envelope, verify_boundary, write_record)
from .training import memory_coverage
from .quality import generated_metrics


def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def skeleton_digest(skeleton):
    h = hashlib.sha256()
    for time in skeleton.times_ms:
        h.update(struct.pack('<d', time))
    return h.hexdigest()


class Rollout:
    """Own one output journal and bounded online state; no target suffix is accepted.

    Resume requires the identical model, runtime, skeleton and decode policy.
    Durable prefix bytes are verified before trailing uncommitted output is
    truncated. Checkpoints contain the RNG and all local/relation/temporal facts,
    but neither the source suffix nor a copy of the entire skeleton.
    """
    def __init__(self, model, skeleton, output: Path, policy=DecodeSamplingPolicy(), *, seed=17,
                 resources=ResourceConfig(), resume=False, export_header=None, source_identity=None,
                 checkpoint_every_rows=512):
        validate_envelope(model.config)
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ContractError('Generation currently requires FP32 model parameters')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ContractError('Decode seed must be an integer in [0,2**63)')
        if type(checkpoint_every_rows) is not int or not 1 <= checkpoint_every_rows <= 8192:
            raise ContractError('checkpoint_every_rows must be an integer in [1,8192]')
        self.model, self.skeleton, self.output = model.eval(), skeleton, Path(output)
        self.policy, self.resources = policy, resources
        self.checkpoint_every_rows = checkpoint_every_rows
        self.export_header = export_header
        self.engine = ContinuationEngine(model)
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.device = str(next(model.parameters()).device)
        self.metadata = dict(format='oracle-time/rollout-v1', model_sha256=model_digest(model),
                             model_config=asdict(model.config), skeleton_sha256=skeleton_digest(skeleton),
                             source_identity=source_identity,
                             export_header_sha256=None if export_header is None else hashlib.sha256(export_header).hexdigest(),
                             policy=asdict(policy), seed=seed, checkpoint_every_rows=checkpoint_every_rows,
                             runtime=runtime_identity(self.device))
        if not resume and self.output.exists() and any(self.output.iterdir()):
            raise ContractError('Generation output_dir must be absent or empty')
        self.output.mkdir(parents=True, exist_ok=True)
        self.resource_log = (self.output / 'resources.jsonl').open('a')
        try:
            self.guard = ResourceGuard(self.device, resources, self.resource_log)
            if resume:
                checkpoint_path = self.output / 'checkpoint.pt'
                if checkpoint_path.stat().st_size > resources.checkpoint_max_bytes:
                    raise ContractError('Checkpoint exceeds configured load size limit')
                payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if payload['metadata'] != self.metadata:
                    raise ContractError('Resume model/runtime/skeleton/decode identity mismatch')
                verify_boundary(self.output / 'rows.jsonl', payload['output'])
                self.state = restore_state(payload['state'], skeleton, model)
                if payload['next_index'] != self.state.execution.next_index:
                    raise ContractError('Resume next event differs from state position')
                self.rng.set_state(payload['rng'])
                self.stats = payload['stats']
                report = verify_rows(self.output / 'rows.jsonl', skeleton.times_ms, require_complete=False,
                                     limit_bytes=payload['output']['bytes'])
                if (report['rows'] != self.state.execution.next_index or
                        tuple(report['open_heads']) != self.state.execution.replay.open_ln_start_ms):
                    raise ContractError('Resume output rows differ from exact state')
                align_boundary(self.output / 'rows.jsonl', payload['output'])
                self.rows = (self.output / 'rows.jsonl').open('ab')
            else:
                (self.output / 'generation-config.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
                self.rows = (self.output / 'rows.jsonl').open('wb')
                self.state = None
                self.stats = dict(seed_rows=0, generated_rows=0, total_ln_closes=0, generated_ln_closes=0,
                                  terminal_forced_closes=0, terminal_pre_open_ages_ms=[],
                                  ln_duration_sum_ms=0., ln_duration_max_ms=0., press_histogram=[0] * 5)
        except BaseException:
            self.resource_log.close()
            raise

    @torch.no_grad()
    def prefill(self, prefix):
        if self.state is not None:
            raise ContractError('Cannot prefill an initialized rollout')
        state = self.engine.start(self.skeleton, inference=True)
        for row in prefix:
            previous = state
            state = self.engine.commit(state, row)
            write_record(self.rows, dict(event_id=previous.execution.next_index, time_ms=row.time_ms,
                                         actions=row.actions, seed=True), self.resources)
            self.stats['seed_rows'] += 1
            self.stats['total_ln_closes'] += row.actions.count(3)
            if state.execution.next_index % self.resources.check_every_rows == 0:
                self.guard.check('prefill', row=state.execution.next_index)
        if state.execution.replay.note_count < 30 or state.execution.finished:
            raise ContractError('Generation requires at least 30 seed notes and a nonempty suffix')
        self.state = state
        del state, previous
        self.guard.check('prefill-complete', row=self.state.execution.next_index)
        self.save()

    @torch.no_grad()
    def advance(self, rows: int | None = None):
        if self.state is None:
            raise ContractError('Prefill the complete seed before generation')
        if rows is not None and (type(rows) is not int or rows < 0):
            raise ContractError('Generation row budget must be nonnegative')
        stop = len(self.skeleton.times_ms) if rows is None else min(len(self.skeleton.times_ms),
                                                                  self.state.execution.next_index + rows)
        started = perf_counter()
        first = self.state.execution.next_index
        while self.state.execution.next_index < stop:
            state = self.state
            query = state.execution.query()
            row, sampling = sample_row(self.engine.predict(state), query.time_ms, self.policy, self.rng)
            following = self.engine.commit(state, row)
            ages = [None if t is None else row.time_ms - t for t in state.execution.replay.open_ln_start_ms]
            closed = [ages[lane] for lane, action in enumerate(row.actions) if action == 3]
            forced = sum(query.history.occupancy) if query.is_terminal else 0
            record = dict(event_id=state.execution.next_index, time_ms=row.time_ms, actions=row.actions, seed=False,
                          terminal=query.is_terminal, pre_open_ages_ms=ages, forced_closes=forced, **sampling)
            write_record(self.rows, record, self.resources)
            self.state = following
            self.stats['generated_rows'] += 1
            self.stats['generated_ln_closes'] += len(closed)
            self.stats['total_ln_closes'] += len(closed)
            self.stats['ln_duration_sum_ms'] += sum(closed)
            self.stats['ln_duration_max_ms'] = max([self.stats['ln_duration_max_ms'], *closed])
            self.stats['press_histogram'][sum(a in (1, 2) for a in row.actions)] += 1
            if query.is_terminal:
                self.stats['terminal_forced_closes'] = forced
                self.stats['terminal_pre_open_ages_ms'] = ages
            # Do not keep the pre-commit carry alive while checkpointing the
            # replacement state or releasing backend-owned storage.
            del state, following
            if self.state.execution.next_index % self.resources.check_every_rows == 0:
                self.guard.check('decode', row=self.state.execution.next_index,
                                 rows_per_second=(self.state.execution.next_index - first) / (perf_counter() - started),
                                 **memory_coverage(self.state))
                if torch.device(self.device).type == 'mps':
                    self.state = self.state.detached()
                    self.guard.release_idle_cache('decode-boundary', row=self.state.execution.next_index)
            if self.state.execution.next_index % self.checkpoint_every_rows == 0:
                self.save()
        self.save()
        return dict(rows=self.state.execution.next_index - first, seconds=perf_counter() - started,
                    finished=self.state.execution.finished)

    def save(self):
        if self.state is None:
            raise ContractError('Cannot checkpoint uninitialized generation')
        if torch.device(self.device).type == 'mps':
            # Online MPS operations can retain backing allocations beyond the
            # tensors' reported storage. Re-own the bounded carry at a no-grad
            # durable boundary to release the observed backend retention.
            self.state = self.state.detached()
        payload = dict(metadata=self.metadata, state=pack_state(self.state), rng=self.rng.get_state(),
                       next_index=self.state.execution.next_index, output=log_boundary(self.rows), stats=self.stats)
        size = atomic_checkpoint(self.output / 'checkpoint.pt', payload, self.resources)
        self.guard.release_idle_cache('checkpoint', row=self.state.execution.next_index, **size)
        return size

    def finish(self):
        if not self.state.execution.finished:
            raise ContractError('Generation must reach the real skeleton terminal before export')
        verification = export_osu(self.output / 'rows.jsonl', self.output / 'generated.osu',
                                  self.skeleton.times_ms, self.resources, header=self.export_header)
        report = dict(**self.stats, verification=verification,
                      organization=generated_metrics(self.output / 'rows.jsonl'),
                      terminal_close_fraction=self.stats['terminal_forced_closes'] /
                                              max(1, self.stats['generated_ln_closes']),
                      coverage=memory_coverage(self.state), state_tensor_bytes=tensor_bytes(pack_state(self.state)))
        (self.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        return report

    def close(self):
        self.rows.close()
        self.resource_log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
