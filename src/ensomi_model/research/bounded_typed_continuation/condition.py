"""Portable external timing and seed conditions, independent of corpus labels."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..oracle_time_continuation.data import source_rows
from ..oracle_time_continuation.schema import CompleteRow
from ..scoped_style_modeling.dataset import ContractError
from ..source_action_modeling.actions import parse_source
from .contract import Arm, Schedule, Timing

CONDITION_FORMAT = 'bounded-typed/condition-v1'
MAX_INPUT_BYTES = 64 * 1024 ** 2
MAX_CANDIDATES = 250000


def pinned_bytes(path, expected, limit=MAX_INPUT_BYTES):
    path = Path(path)
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ContractError(f'{path}: exceeds the {limit}-byte input limit')
    if hashlib.sha256(data).hexdigest() != expected:
        raise ContractError(f'{path}: input SHA-256 differs from the declared digest')
    return data


def exact_fields(value, names, owner):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ContractError(f'{owner} requires exactly these fields: {", ".join(names)}')


@dataclass(frozen=True)
class GenerationCondition:
    arm: Arm
    timing: Timing
    seed_rows: tuple[CompleteRow, ...]
    crossing_ends: tuple[int | None, ...]

    def __post_init__(self):
        if len(self.timing.times_ms) > MAX_CANDIDATES:
            raise ContractError(f'Generation supports at most {MAX_CANDIDATES} candidates')
        if (not isinstance(self.seed_rows, tuple) or not self.seed_rows or
                len(self.seed_rows) >= len(self.timing.times_ms)):
            raise ContractError('Generation needs a nonempty complete seed and a remaining candidate suffix')
        if (not isinstance(self.crossing_ends, tuple) or len(self.crossing_ends) != 4 or
                any(end is not None and type(end) is not int for end in self.crossing_ends)):
            raise ContractError('crossing_ends requires four null or integer candidate indices')
        Schedule.from_seed(self.arm, self.timing, self.seed_rows, self.crossing)

    @property
    def crossing(self):
        return {lane: end for lane, end in enumerate(self.crossing_ends) if end is not None}

    def payload(self):
        return dict(format=CONDITION_FORMAT, arm=self.arm.value, timing=asdict(self.timing),
                    seed_rows=[asdict(row) for row in self.seed_rows], crossing_ends=self.crossing_ends)

    @classmethod
    def from_payload(cls, value):
        exact_fields(value, ('format', 'arm', 'timing', 'seed_rows', 'crossing_ends'), 'Condition')
        if value['format'] != CONDITION_FORMAT:
            raise ContractError('Unsupported bounded generation condition format')
        exact_fields(value['timing'], ('times_ms', 'onsets'), 'Timing')
        if (not isinstance(value['timing']['times_ms'], list) or
                value['timing']['onsets'] is not None and not isinstance(value['timing']['onsets'], list) or
                not isinstance(value['seed_rows'], list) or not isinstance(value['crossing_ends'], list)):
            raise ContractError('Condition times, roles, rows and crossing ends must use JSON arrays')
        rows = []
        for row in value['seed_rows']:
            exact_fields(row, ('time_ms', 'actions'), 'Seed row')
            if not isinstance(row['actions'], list):
                raise ContractError('Seed actions must use a JSON array')
            rows.append(CompleteRow(row['time_ms'], tuple(row['actions'])))
        try:
            arm = Arm(value['arm'])
        except (TypeError, ValueError) as error:
            raise ContractError('Condition arm must be r0, r1 or o1') from error
        roles = value['timing']['onsets']
        timing = Timing(tuple(value['timing']['times_ms']), None if roles is None else tuple(roles))
        return cls(arm, timing, tuple(rows), tuple(value['crossing_ends']))


def condition_from_source(data: bytes, source_sha256: str, arm: Arm, *, seed_notes=30):
    """Use source R/H and whole seed objects; omit every other suffix assignment.

    This is an optional preparation operation. The generation runner reads only
    the resulting condition and never needs this source or its corpus allocation.
    """
    if type(seed_notes) is not int or seed_notes <= 0:
        raise ContractError('seed_notes must be a positive integer')
    source = parse_source(data, source_sha256)
    rows = source_rows(source.objects)
    times = tuple(row.time_ms for row in rows)
    roles = None if arm == Arm.R0 else tuple(any(a in (1, 2) for a in row.actions) for row in rows)
    count = stop = 0
    for stop, row in enumerate(rows, 1):
        count += sum(a in (1, 2) for a in row.actions)
        if count >= seed_notes:
            break
    if count < seed_notes or stop == len(rows):
        raise ContractError('Source needs the requested whole-row seed and a remaining suffix')
    crossing = [None] * 4
    if arm != Arm.R0:
        indices = {time: i for i, time in enumerate(times)}
        boundary = times[stop - 1]
        for note in source.objects:
            if note.kind == 'long' and note.start_ms <= boundary < note.end_ms:
                crossing[note.column] = indices[note.end_ms]
    return GenerationCondition(arm, Timing(times, roles), rows[:stop], tuple(crossing))


def write_source_condition(source_file, source_sha256, output_file, *, arm=Arm.R1, seed_notes=30):
    condition = condition_from_source(pinned_bytes(source_file, source_sha256), source_sha256,
                                      arm, seed_notes=seed_notes)
    data = (json.dumps(condition.payload(), indent=2, allow_nan=False) + '\n').encode()
    if len(data) > MAX_INPUT_BYTES:
        raise ContractError('Prepared condition exceeds its input-size envelope')
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
    return dict(condition_file=str(path), condition_sha256=hashlib.sha256(data).hexdigest(),
                source_sha256=source_sha256, arm=arm.value, candidates=len(condition.timing.times_ms),
                seed_rows=len(condition.seed_rows), seed_notes=sum(a in (1, 2) for row in condition.seed_rows for a in row.actions))
