"""Resource envelope and atomic, size-limited durable publications."""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import psutil
import torch

from ..scoped_style_modeling.dataset import ContractError
from .config import BackboneConfig
from .publication import staging_directory

MiB = 1024**2
GiB = 1024**3


@dataclass(frozen=True)
class ResourceConfig:
    driver_limit_bytes: int = 4 * GiB
    rss_limit_bytes: int = 6 * GiB
    allocator_ceiling_bytes: int = 8 * GiB
    min_available_bytes: int = 512 * MiB
    max_swap_growth_bytes: int = GiB
    checkpoint_max_bytes: int = 128 * MiB
    disk_reserve_bytes: int = 512 * MiB
    output_max_bytes: int = 2 * GiB
    check_every_rows: int = 128
    release_mps_cache_at_boundaries: bool = True

    def __post_init__(self):
        if type(self.release_mps_cache_at_boundaries) is not bool:
            raise ContractError("release_mps_cache_at_boundaries must be boolean")
        for name, value in vars(self).items():
            if name == "release_mps_cache_at_boundaries":
                continue
            if type(value) is not int or value <= 0:
                raise ContractError(f'{name} must be a positive integer')
        if self.driver_limit_bytes > 8 * GiB or self.rss_limit_bytes > 12 * GiB:
            raise ContractError('Resource guard exceeds the 8 GiB driver / 12 GiB RSS outer limits')
        if not self.driver_limit_bytes <= self.allocator_ceiling_bytes <= 8 * GiB:
            raise ContractError('MPS allocator ceiling must cover the driver guard and be at most 8 GiB')
        if self.check_every_rows > 128:
            raise ContractError('Resource checks must occur at least every 128 rows')


class ResourceLimit(ContractError):
    """Abort the operation and recover from the last already durable checkpoint."""


def training_checkpoint_tensor_bytes(model) -> int:
    """Conservative model plus two FP32 AdamW moments, before the first update.

    Serialized metadata and step counters require additional space; the bounded
    writer remains the authority for the actual publication size.
    """
    return (3 * sum(p.numel() * p.element_size() for p in model.parameters()) +
            sum(b.numel() * b.element_size() for b in model.buffers()))


def validate_envelope(model: BackboneConfig, microbatch: int = 1):
    maximum = BackboneConfig()
    limits = {f.name: getattr(maximum, f.name) for f in fields(model)}
    limits.update(heads=8, temporal_layers=6, temporal_expansion=8, temporal_bias_hidden=128,
                  time_lookahead_rows=16, clock_readout_hidden=128)
    if any(value is not None and value > limits[name]
           for name, value in vars(model).items()) or microbatch > 2:
        raise ContractError('Requested model/microbatch exceeds the supported FP32 envelope')
    if model.temporal_hidden > 128 or model.temporal_layers > 2:
        bias_width = model.temporal_bias_hidden or model.temporal_hidden
        if microbatch > 1 or model.max_chunk > 64 or bias_width > 64:
            raise ContractError('Expanded temporal envelope requires microbatch 1, max_chunk <=64 and bias width <=64')


def snapshot(device: str) -> dict:
    device = torch.device(device).type
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    pressure = None
    if platform.system() == 'Darwin':
        result = subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            pressure = int(result.stdout.strip())
    return dict(time=time.time(), rss_bytes=psutil.Process().memory_info().rss,
                available_bytes=psutil.virtual_memory().available, swap_bytes=psutil.swap_memory().used,
                pressure_level=pressure,
                active_bytes=torch.mps.current_allocated_memory() if device == 'mps' else
                             torch.cuda.memory_allocated() if device == 'cuda' else 0,
                driver_bytes=torch.mps.driver_allocated_memory() if device == 'mps' else
                             torch.cuda.memory_reserved() if device == 'cuda' else 0)


class ResourceGuard:
    def __init__(self, device: str, config: ResourceConfig, log=None):
        device = torch.device(device).type
        self.device, self.config, self.log = device, config, log
        if device == 'mps':
            if not torch.backends.mps.is_available():
                raise ContractError('Requested MPS device is unavailable')
            torch.mps.set_per_process_memory_fraction(config.allocator_ceiling_bytes /
                                                      torch.mps.recommended_max_memory())
        if device == 'cuda' and not torch.cuda.is_available():
            raise ContractError('Requested CUDA device is unavailable')
        self.initial_swap = psutil.swap_memory().used
        self.check('startup')

    def check(self, phase: str, **metadata):
        value = snapshot(self.device)
        value.update(phase=phase, swap_growth_bytes=value['swap_bytes'] - self.initial_swap, **metadata)
        if self.log is not None:
            record = json.dumps(value, allow_nan=False) + '\n'
            size = len(record.encode('utf-8'))
            if self.log.tell() + size > self.config.output_max_bytes:
                raise ResourceLimit('Resource journal exceeds output_max_bytes')
            if shutil.disk_usage(Path(self.log.name).parent).free < self.config.disk_reserve_bytes + size:
                raise ResourceLimit('Resource journal would consume disk reserve')
            self.log.write(record)
            self.log.flush()
        limits = [('driver_bytes', self.config.driver_limit_bytes), ('rss_bytes', self.config.rss_limit_bytes),
                  ('swap_growth_bytes', self.config.max_swap_growth_bytes)]
        for name, limit in limits:
            if value[name] > limit:
                raise ResourceLimit(f'{phase}: {name}={value[name]} exceeds {limit}; resume last durable boundary')
        if value['available_bytes'] < self.config.min_available_bytes or value['pressure_level'] == 4:
            raise ResourceLimit(f'{phase}: insufficient available memory or critical system pressure')
        return value


    def release_idle_cache(self, phase, **metadata):
        """Release unused allocations at a caller-owned graph-free phase boundary."""
        if self.device == 'mps' and self.config.release_mps_cache_at_boundaries:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        return self.check(phase, **metadata)

    def update_boundary(self, **metadata):
        return self.release_idle_cache('update-boundary', **metadata)

    def prepare_target(self, **metadata):
        return self.release_idle_cache('target-boundary', **metadata)


def owned_cpu(value):
    """Compact every tensor view before serialization; never save a backing graph/bank."""
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu').clone(memory_format=torch.contiguous_format)
    if isinstance(value, dict):
        return {k: owned_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(owned_cpu(v) for v in value)
    return value


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class LimitedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, data):
        if self.stream.tell() + len(data) > self.limit:
            raise ResourceLimit('Checkpoint exceeds checkpoint_max_bytes; previous publication is intact')
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def sync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_checkpoint(path: Path, payload: dict, config: ResourceConfig) -> dict:
    """Fsync one compact temporary file, then replace one retained checkpoint.

    The existing file stays usable on serialization/space failure. Staging must
    fit beside it while preserving disk_reserve_bytes. There is no history list.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor_bytes(payload) > config.checkpoint_max_bytes:
        raise ResourceLimit('Checkpoint tensors exceed checkpoint_max_bytes')
    with staging_directory(path) as staging:
        if shutil.disk_usage(path.parent).free < config.disk_reserve_bytes + config.checkpoint_max_bytes:
            raise ResourceLimit('Insufficient disk space for checkpoint staging and reserve')
        temporary = staging / 'checkpoint.pt'
        with temporary.open('wb') as stream:
            torch.save(owned_cpu(payload), LimitedWriter(stream, config.checkpoint_max_bytes))
            stream.flush()
            os.fsync(stream.fileno())
        size = temporary.stat().st_size
        os.replace(temporary, path)
        sync_directory(path.parent)
        return dict(checkpoint_bytes=size, tensor_bytes=tensor_bytes(payload))


def log_boundary(stream) -> dict:
    stream.flush()
    os.fsync(stream.fileno())
    size = stream.tell()
    return dict(bytes=size, sha256=prefix_digest(Path(stream.name), size))


def prefix_digest(path: Path, size: int):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        remaining = size
        while remaining:
            block = stream.read(min(remaining, MiB))
            if not block:
                raise ContractError(f'{path}: durable output is shorter than checkpoint boundary')
            h.update(block)
            remaining -= len(block)
    return h.hexdigest()


def verify_boundary(path: Path, boundary: dict):
    if prefix_digest(path, boundary['bytes']) != boundary['sha256']:
        raise ContractError(f'{path}: durable output digest differs from checkpoint')


def align_boundary(path: Path, boundary: dict):
    verify_boundary(path, boundary)
    with path.open('r+b') as stream:
        stream.truncate(boundary['bytes'])
        stream.flush()
        os.fsync(stream.fileno())


def write_record(stream, record: dict, config: ResourceConfig):
    data = (json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if stream.tell() + len(data) > config.output_max_bytes:
        raise ResourceLimit('Output exceeds output_max_bytes; last durable boundary is intact')
    if shutil.disk_usage(Path(stream.name).parent).free < config.disk_reserve_bytes + len(data):
        raise ResourceLimit('Output would consume disk reserve')
    stream.write(data)
