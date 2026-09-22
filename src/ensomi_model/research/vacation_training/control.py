"""Atomic receipts, exclusive ownership and safe-boundary stop requests."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import time

from ..oracle_time_continuation.publication import staging_directory
from ..oracle_time_continuation.runtime import ResourceGuard, ResourceLimit, sync_directory
from ..bounded_typed_continuation.memory import footprint_bytes
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError


def read_json(path, sha, *, limit=128 * 1024**2):
    path = Path(path)
    if not sha or file_digest(path, limit) != sha:
        raise ContractError(f'Pinned input differs: {path}')
    return json.loads(path.read_text())


def publish_json(path, value):
    path = Path(path)
    with staging_directory(path) as stage:
        temporary = stage / 'value.json'
        with temporary.open('w') as stream:
            json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    return file_digest(path)


@contextmanager
def run_lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError('Another process owns this vacation queue') from error
        yield


def tree_bytes(root):
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                raise ContractError('Queue output must not contain symlinked products')
            total += path.stat().st_size
    return total


def host_pause_reason(require_ac, thermal):
    if platform.system() != 'Darwin':
        if require_ac or thermal:
            raise ContractError('Mac power/thermal checks require Darwin; disable explicitly on another host')
        return None
    if require_ac:
        result = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True, timeout=5, check=True)
        if "AC Power" not in result.stdout:
            return 'ac_power_required'
    if thermal:
        result = subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True, timeout=5, check=True)
        for name in ('CPU_Speed_Limit', 'CPU_Scheduler_Limit'):
            match = re.search(name + r'\s*=\s*(\d+)', result.stdout)
            if match and int(match.group(1)) < 100:
                return 'thermal_pressure'
        if re.search(r'(Thermal|Performance) (?:warning|level)\s*[:=]\s*[1-9]', result.stdout, re.I):
            return 'thermal_pressure'
    return None


class Control:
    def __init__(self, config, root, started_at, stage_deadline, initial_swap_bytes=None):
        self.config, self.root = config, root
        self.deadline = min(started_at + config.max_seconds, stage_deadline)
        self.queue_deadline = started_at + config.max_seconds
        self.signal_reason = None
        self.last_host_check = -float('inf')
        self.host_reason = None
        self.guard = None
        self.initial_swap_bytes = initial_swap_bytes

    def boundary(self):
        if self.signal_reason:
            return self.signal_reason
        if (self.root / 'PAUSE').exists():
            return 'pause_file'
        now = time.time()
        if now >= self.queue_deadline:
            return 'queue_time_limit'
        if now >= self.deadline:
            return 'stage_time_limit'
        if time.monotonic() - self.last_host_check >= 5:
            self.host_reason = host_pause_reason(self.config.require_ac_power, self.config.check_thermal)
            self.storage()
            self.last_host_check = time.monotonic()
        return self.host_reason

    def storage(self, additional=0):
        if self.guard is not None:
            footprint = footprint_bytes()
            self.guard.check('queue-boundary', footprint_bytes=footprint)
            if footprint is not None and footprint > self.config.footprint_limit_bytes:
                raise ResourceLimit('Vacation worker exceeds its physical-footprint budget')
        if tree_bytes(self.root) + additional > self.config.max_bytes:
            raise ResourceLimit('Vacation products exceed the aggregate byte budget')
        if shutil.disk_usage(self.root).free < self.config.disk_reserve_bytes + additional:
            raise ResourceLimit('Vacation products would consume the disk reserve')

    @contextmanager
    def signals(self):
        saved = {}
        def request(signum, _frame):
            self.signal_reason = 'signal_' + signal.Signals(signum).name
        for sig in (signal.SIGINT, signal.SIGTERM):
            saved[sig] = signal.signal(sig, request)
        try:
            with (self.root / 'resources.jsonl').open('a') as journal:
                self.guard = ResourceGuard('cpu', self.config.resources, journal)
                if self.initial_swap_bytes is not None:
                    self.guard.initial_swap = self.initial_swap_bytes
                yield
        finally:
            self.guard = None
            for sig, handler in saved.items():
                signal.signal(sig, handler)
