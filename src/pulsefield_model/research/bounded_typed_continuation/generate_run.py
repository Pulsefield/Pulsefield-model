"""Durable native generation from portable conditions and pinned model weights.

Each invocation owns a fresh output directory. Recovery copies only verified
durable log prefixes; a failed parent's uncommitted tail remains untouched.
"""
from collections import deque
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import torch

from ..oracle_time_continuation.data import source_rows
from ..oracle_time_continuation.export import export_osu, presentation_header
from ..oracle_time_continuation.runtime import (
    ResourceGuard, ResourceLimit, atomic_checkpoint, log_boundary, verify_boundary, write_record,
)
from ..oracle_time_continuation.schema import CompleteRow
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from ..source_action_modeling.actions import parse_source
from .condition import GenerationCondition, exact_fields, pinned_bytes
from .contract import Arm
from .generate_config import GenerateConfig
from .generation import RawEvent, Rollout
from .memory import footprint_bytes
from .model import BoundedModel, ModelConfig
from .smoke_run import save_json, source_revision
from .verification import verify_complete

FORMAT = 'bounded-typed/generation-run-v1'
MODEL_FORMATS = ('bounded-typed/corpus-training-v1', 'bounded-typed/learning-check-v1')
IDENTITY_FIELDS = ('checkpoint_sha256', 'condition_sha256', 'presentation_sha256',
                   'device', 'cpu_threads', 'seed', 'candidate_budget', 'score_endpoints')


def _load(path, digest, limit):
    return torch.load(io.BytesIO(pinned_bytes(path, digest, limit)), map_location='cpu', weights_only=True)


def _model(config):
    payload = _load(config.checkpoint_file, config.checkpoint_sha256, config.resources.checkpoint_max_bytes)
    if payload.get('format') not in MODEL_FORMATS:
        raise ContractError('Generation requires a bounded corpus-training or learning-check checkpoint')
    settings = dict(payload['config']['model'])
    settings['arm'] = Arm(settings['arm'])
    model_config = ModelConfig(**settings)
    if (model_config.hidden > 128 or model_config.levels > 8 or model_config.expansion > 4 or
            model_config.coupling_rank > 16):
        raise ContractError('Checkpoint model exceeds the bounded generation envelope')
    model = BoundedModel(model_config)
    model.load_state_dict(payload['model'], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ContractError('Checkpoint contains nonfinite model parameters')
    provenance = dict(format=payload['format'], source_revision=payload['source_revision'],
                      model_config=asdict(model_config))
    del payload
    return model.to(config.device).eval(), provenance


def _records(path, size=None):
    with Path(path).open('rb') as stream:
        while size is None or stream.tell() < size:
            remaining = 1024 ** 2 if size is None else min(1024 ** 2, size - stream.tell())
            line = stream.readline(remaining)
            if not line:
                if size is not None:
                    raise ContractError('Generation journal is shorter than its durable boundary')
                break
            if not line.endswith(b'\n'):
                raise ContractError('Generation journal needs complete lines smaller than 1 MiB')
            yield json.loads(line)


def _row_record(record, index, candidate, row):
    exact_fields(record, ('event_id', 'candidate_index', 'time_ms', 'actions'), 'Generation row')
    if (type(record['event_id']) is not int or type(record['candidate_index']) is not int or
            record['event_id'] != index or record['candidate_index'] != candidate or
            CompleteRow(record['time_ms'], tuple(record['actions'])) != row):
        raise ContractError('Generation row journal differs from its seed or decision journal')


def _endpoints(value):
    if not isinstance(value, dict) or any(key not in ('0', '1', '2', '3') for key in value):
        raise ContractError('Endpoint journal keys must be zero-based lane strings')
    return {int(lane): end for lane, end in value.items()}


def _verify_parent(model, condition, restored, parent, payload):
    """Tie the snapshot's exact state and bounded history to its complete logs."""
    initial = Rollout.from_seed(model, condition.timing, condition.seed_rows, condition.crossing)
    if restored.seed_history != initial.seed_history:
        raise ContractError('Persistent seed differs from the external generation condition')
    state = initial.state
    history = deque(initial.history, maxlen=model.temporal.config.receptive_tokens)
    rows = iter(_records(parent / 'rows.jsonl', payload['rows']['bytes']))
    count = 0
    for index, row in enumerate(condition.seed_rows):
        _row_record(next(rows, None), index, index, row)
        count += 1
    for record in _records(parent / 'decisions.jsonl', payload['decisions']['bytes']):
        exact_fields(record, ('candidate_index', 'row', 'endpoints', 'row_or_head_log_prob',
                              'endpoint_log_prob'), 'Generation decision')
        if type(record['candidate_index']) is not int or record['candidate_index'] != state.index:
            raise ContractError('Decision journal candidate cursor is discontinuous')
        value = record['row']
        if value is not None:
            exact_fields(value, ('time_ms', 'actions'), 'Decision row')
        chosen = None if value is None else CompleteRow(value['time_ms'], tuple(value['actions']))
        following, row = state.advance(None if chosen is None else chosen.actions, _endpoints(record['endpoints']))
        if row != chosen:
            raise ContractError('Decision journal row differs from its candidate time')
        if row is not None:
            _row_record(next(rows, None), count, state.index, row)
            previous = None if state.replay.last_row is None else state.replay.last_row.time_ms
            ends = tuple(state.timing.times_ms[end] if action == 2 and end is not None else None
                         for action, end in zip(row.actions, following.known_ends))
            history.append(RawEvent(row, previous, ends))
            count += 1
        state = following
    if next(rows, None) is not None or state != restored.state or list(history) != list(restored.history):
        raise ContractError('Durable generation logs and rollout snapshot disagree')


def _resume(config, model, condition, revision, identity, header_sha):
    if config.resume_from is None:
        return Rollout.from_seed(model, condition.timing, condition.seed_rows, condition.crossing), \
            torch.Generator().manual_seed(config.seed), None
    path = Path(config.resume_from)
    payload = _load(path, config.resume_sha256, config.resources.checkpoint_max_bytes)
    if (payload.get('format') != FORMAT or payload.get('source_revision') != revision or
            payload.get('identity') != identity or payload.get('header_sha256') != header_sha):
        raise ContractError('Generation resume source, input identities or sampling execution differs')
    for name in ('rows', 'decisions'):
        boundary = payload[name]
        if (type(boundary['bytes']) is not int or not 0 <= boundary['bytes'] <= config.resources.output_max_bytes):
            raise ContractError('Parent journal boundary exceeds its supported output envelope')
        verify_boundary(path.parent / f'{name}.jsonl', boundary)
    restored, rng = Rollout.restore(model, condition.timing, payload['rollout'])
    if restored.seed_rows != len(condition.seed_rows):
        raise ContractError('Resume seed count differs from the external condition')
    _verify_parent(model, condition, restored, path.parent, payload)
    return restored, rng, payload


def _copy_prefix(path, destination, boundary):
    remaining = boundary['bytes']
    with Path(path).open('rb') as source:
        while remaining:
            block = source.read(min(1024 ** 2, remaining))
            if not block:
                raise ContractError('Parent journal changed while copying its durable prefix')
            destination.write(block)
            remaining -= len(block)
    if log_boundary(destination) != boundary:
        raise ContractError('Copied parent journal does not match its durable identity')


def _complete_output(output, condition, config, header):
    rows = []
    previous = -1
    for index, record in enumerate(_records(output / 'rows.jsonl')):
        candidate = record['candidate_index']
        if (type(candidate) is not int or not previous < candidate < len(condition.timing.times_ms) or
                record['time_ms'] != condition.timing.times_ms[candidate]):
            raise ContractError('Output row does not belong to its timing candidate')
        row = CompleteRow(record['time_ms'], tuple(record['actions']))
        _row_record(record, index, candidate, row)
        rows.append(row)
        previous = candidate
    plans = {(record['candidate_index'], lane): end
             for record in _records(output / 'decisions.jsonl')
             for lane, end in _endpoints(record['endpoints']).items()}
    mechanics = verify_complete(rows, condition.timing, condition.arm, condition.seed_rows, condition.crossing, plans)
    osu = output / 'generated.osu'
    export_osu(output / 'rows.jsonl', osu, [row.time_ms for row in rows], config.resources, header=header)
    osu_sha = file_digest(osu)
    parsed = parse_source(pinned_bytes(osu, osu_sha, config.resources.output_max_bytes), osu_sha)
    if source_rows(parsed.objects) != tuple(rows):
        raise ContractError('Exported osu! objects do not round-trip to the generated rows')
    return dict(mechanical=mechanics, reparse_pass=True, osu_file=str(osu), osu_sha256=osu_sha,
                rows_sha256=file_digest(output / 'rows.jsonl'), decisions_sha256=file_digest(output / 'decisions.jsonl'))


@torch.no_grad()
def run_generation(config: GenerateConfig, *, resolved_yaml=''):
    """Generate or resume into a fresh directory from digest-pinned inputs.

    Return a completed report only after independent verification and exact
    osu! reparse. A cursor/time pause returns a durable checkpoint without an
    osu! export. Failures propagate and leave the last complete checkpoint;
    recovery copies verified journal prefixes without modifying their parent.
    The source checkout must be clean and committed. CPU thread settings are
    restored on return or failure; exact recovery requires the same execution
    environment as the parent run.
    """
    config.validate()
    if any(not getattr(config, key) for key in ('checkpoint_file', 'checkpoint_sha256',
                                               'condition_file', 'condition_sha256', 'output_dir')):
        raise ContractError('Generation requires pinned checkpoint/condition files and a fresh output_dir')
    revision = source_revision()
    raw_condition = pinned_bytes(config.condition_file, config.condition_sha256)
    condition = GenerationCondition.from_payload(json.loads(raw_condition))
    header = None
    if config.presentation_source:
        pinned_bytes(config.presentation_source, config.presentation_sha256)
        header = presentation_header(Path(config.presentation_source))
        if file_digest(Path(config.presentation_source)) != config.presentation_sha256:
            raise ContractError('Presentation source changed while reading its header')
    header_sha = None if header is None else hashlib.sha256(header).hexdigest()
    identity = {key: getattr(config, key) for key in IDENTITY_FIELDS}
    output = Path(config.output_dir).resolve()
    if output.exists():
        raise ContractError('Generation needs a fresh output_dir, including when resuming')
    previous_threads = torch.get_num_threads()
    began = time.monotonic()
    durable_index = checkpoint_sha = None
    owns_output = False
    try:
        torch.set_num_threads(config.cpu_threads)
        guard = ResourceGuard(config.device, config.resources)
        model, provenance = _model(config)
        guard.check('model-loaded')
        if model.config.arm != condition.arm:
            raise ContractError('Checkpoint task arm differs from the external condition')
        rollout, rng, parent = _resume(config, model, condition, revision, identity, header_sha)
        if config.stop_after_candidate is not None and not rollout.state.index <= config.stop_after_candidate <= len(condition.timing.times_ms):
            raise ContractError('Stopping cursor must lie between the restored cursor and the timing end')
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        save_json(output / 'run-config.json', asdict(config))
        (output / 'resolved.yaml').write_text(resolved_yaml)
        (output / 'condition.json').write_bytes(raw_condition)
        if header is not None:
            (output / 'presentation-header.osu').write_bytes(header)
        save_json(output / 'environment.json', dict(source_revision=revision, python=platform.python_version(),
                  torch=torch.__version__, device=config.device, cpu_threads=config.cpu_threads, model=provenance))
        with (output / 'resources.jsonl').open('x') as resources, \
                (output / 'rows.jsonl').open('xb') as row_log, (output / 'decisions.jsonl').open('xb') as decision_log:
            guard.log = resources

            def check(phase, *, release_cache=False):
                footprint = footprint_bytes()
                if footprint is not None and footprint > config.footprint_limit_bytes:
                    raise ResourceLimit('Generation exceeded its physical-footprint limit')
                if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > config.resources.output_max_bytes:
                    raise ResourceLimit('Generation output exceeded its total byte limit')
                operation = guard.release_idle_cache if release_cache else guard.check
                operation(phase, candidate_index=rollout.state.index, footprint_bytes=footprint)

            def checkpoint():
                nonlocal durable_index, checkpoint_sha
                value = dict(format=FORMAT, source_revision=revision, identity=identity, header_sha256=header_sha,
                             parent_sha256=config.resume_sha256 or None, rollout=rollout.snapshot(rng),
                             rows=log_boundary(row_log), decisions=log_boundary(decision_log))
                atomic_checkpoint(output / 'checkpoint.pt', value, config.resources)
                durable_index = rollout.state.index
                checkpoint_sha = file_digest(output / 'checkpoint.pt')
                check('checkpoint', release_cache=True)

            if parent is None:
                for i, row in enumerate(condition.seed_rows):
                    write_record(row_log, dict(event_id=i, candidate_index=i, **asdict(row)), config.resources)
            else:
                directory = Path(config.resume_from).parent
                _copy_prefix(directory / 'rows.jsonl', row_log, parent['rows'])
                _copy_prefix(directory / 'decisions.jsonl', decision_log, parent['decisions'])
            checkpoint()
            reason = None
            while not rollout.state.finished:
                if rollout.state.index == config.stop_after_candidate:
                    reason = 'requested_cursor'
                    break
                if time.monotonic() - began >= config.max_seconds:
                    reason = 'time_limit'
                    break
                step = rollout.step(rng, candidate_budget=config.candidate_budget, score_endpoints=config.score_endpoints)
                write_record(decision_log, asdict(step), config.resources)
                if step.row is not None:
                    write_record(row_log, dict(event_id=rollout.state.replay.row_count - 1,
                                              candidate_index=step.candidate_index, **asdict(step.row)), config.resources)
                if rollout.state.index % 128 == 0:
                    check('generation')
                if rollout.state.index % config.checkpoint_every_candidates == 0:
                    checkpoint()
            checkpoint()
        guard.log = None
        result = dict(status='completed' if rollout.state.finished else 'paused', pause_reason=reason,
                      source_revision=revision, model_checkpoint_sha256=config.checkpoint_sha256,
                      condition_sha256=config.condition_sha256, arm=condition.arm.value,
                      candidate_index=rollout.state.index, candidates=len(condition.timing.times_ms),
                      rows=rollout.state.replay.row_count, output_dir=str(output),
                      checkpoint_path=str(output / 'checkpoint.pt'), checkpoint_sha256=checkpoint_sha,
                      parent_checkpoint_sha256=config.resume_sha256 or None)
        if rollout.state.finished:
            result.update(_complete_output(output, condition, config, header))
            footprint = footprint_bytes()
            if footprint is not None and footprint > config.footprint_limit_bytes:
                raise ResourceLimit('Output verification exceeded the physical-footprint limit')
            with (output / 'resources.jsonl').open('a') as resources:
                guard.log = resources
                guard.check('output-verified', footprint_bytes=footprint)
            guard.log = None
            if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > config.resources.output_max_bytes:
                raise ResourceLimit('Verified generation output exceeded its total byte limit')
        result['seconds'] = time.monotonic() - began
        save_json(output / 'result.json', result)
        return result
    except BaseException as error:
        if owns_output:
            save_json(output / 'failure.json', dict(status='stopped', exception=type(error).__name__, reason=str(error),
                      durable_candidate_index=durable_index, checkpoint_sha256=checkpoint_sha,
                      seconds=time.monotonic() - began))
        raise
    finally:
        torch.set_num_threads(previous_threads)
