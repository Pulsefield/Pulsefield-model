from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .vocab import KEY_COUNT


_OLD_MAPPER_KEYS = {
    "target_tokens",
    "target_token_mask",
    "teacher_current_ms",
    "teacher_open_mask",
    "teacher_open_age_ms",
}


@dataclass(frozen=True)
class MapperTokenContract:
    name: str
    vocab: Any
    requires_sparse_lane_state: bool = False
    uses_chart_end_for_terminal_windows: bool = False
    sparse_padded_last_lane_index: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MapperTokenContract.name must be non-empty")
        if self.vocab is None:
            raise ValueError("MapperTokenContract.vocab is required")
        vocab_size = self.vocab_size
        for attr in ("pad_id", "bos_id", "eos_id"):
            value = _vocab_int_attr(self.vocab, attr)
            if not 0 <= value < vocab_size:
                raise ValueError(f"MapperTokenContract.vocab.{attr} must be inside vocab size {vocab_size}")

    @property
    def pad_id(self) -> int:
        return _vocab_int_attr(self.vocab, "pad_id")

    @property
    def bos_id(self) -> int:
        return _vocab_int_attr(self.vocab, "bos_id")

    @property
    def eos_id(self) -> int:
        return _vocab_int_attr(self.vocab, "eos_id")

    @property
    def vocab_size(self) -> int:
        if hasattr(self.vocab, "size"):
            size = _vocab_int_attr(self.vocab, "size")
        elif hasattr(self.vocab, "vocab_size"):
            size = _vocab_int_attr(self.vocab, "vocab_size")
        elif hasattr(self.vocab, "id_to_token"):
            size = len(getattr(self.vocab, "id_to_token"))
        else:
            raise ValueError("MapperTokenContract.vocab must define size, vocab_size, or id_to_token")
        if size <= 0:
            raise ValueError("MapperTokenContract.vocab_size must be positive")
        return size


@dataclass(frozen=True)
class MapperFragmentState:
    current_ms: torch.Tensor
    open_mask: torch.Tensor
    open_start_ms: torch.Tensor
    open_age_ms: torch.Tensor
    emitted_lane_mask: torch.Tensor | None = None
    last_lane_index: torch.Tensor | None = None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        batch_shape: Sequence[int],
        device: torch.device | str,
        requires_sparse_lane_state: bool,
    ) -> "MapperFragmentState":
        if not isinstance(mapping, Mapping):
            raise ValueError("target_fragment_states must be a mapping")
        expected_batch_shape = _normalize_batch_shape(batch_shape)
        resolved_device = torch.device(device)
        current_ms = _require_state_tensor(mapping, "current_ms", ndim=2).to(device=resolved_device, dtype=torch.long)
        open_mask = _require_state_tensor(mapping, "open_mask", ndim=3).to(device=resolved_device, dtype=torch.bool)
        open_start_ms = _require_state_tensor(mapping, "open_start_ms", ndim=3).to(
            device=resolved_device,
            dtype=torch.long,
        )
        open_age_ms = _require_state_tensor(mapping, "open_age_ms", ndim=3).to(
            device=resolved_device,
            dtype=torch.long,
        )

        if tuple(current_ms.shape) != expected_batch_shape:
            raise ValueError("target_fragment_states.current_ms must align with decoder_input_tokens")
        expected_lane_shape = (*expected_batch_shape, KEY_COUNT)
        if tuple(open_mask.shape) != expected_lane_shape:
            raise ValueError(f"target_fragment_states.open_mask must have shape [B,S,{KEY_COUNT}]")
        if tuple(open_start_ms.shape) != tuple(open_mask.shape):
            raise ValueError("target_fragment_states.open_start_ms must align with target_fragment_states.open_mask")
        if tuple(open_age_ms.shape) != tuple(open_mask.shape):
            raise ValueError("target_fragment_states.open_age_ms must align with target_fragment_states.open_mask")

        emitted_lane_mask = _optional_state_tensor(mapping, "emitted_lane_mask", ndim=3)
        last_lane_index = _optional_state_tensor(mapping, "last_lane_index", ndim=2)
        if requires_sparse_lane_state:
            if emitted_lane_mask is None:
                raise ValueError("target_fragment_states.emitted_lane_mask must be a torch.Tensor")
            if last_lane_index is None:
                raise ValueError("target_fragment_states.last_lane_index must be a torch.Tensor")

        if emitted_lane_mask is not None:
            emitted_lane_mask = emitted_lane_mask.to(device=resolved_device, dtype=torch.bool)
            if tuple(emitted_lane_mask.shape) != tuple(open_mask.shape):
                raise ValueError(
                    "target_fragment_states.emitted_lane_mask must align with target_fragment_states.open_mask",
                )
        if last_lane_index is not None:
            last_lane_index = last_lane_index.to(device=resolved_device, dtype=torch.long)
            if tuple(last_lane_index.shape) != expected_batch_shape:
                raise ValueError("target_fragment_states.last_lane_index must align with decoder_input_tokens")

        return cls(
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            emitted_lane_mask=emitted_lane_mask,
            last_lane_index=last_lane_index,
        )

    def sanitized(
        self,
        target_end_ms: torch.Tensor,
        valid_input_mask: torch.Tensor,
        sparse_padded_last_lane_index: int = -1,
    ) -> "MapperFragmentState":
        target_end = _as_batch_vector(
            target_end_ms,
            name="target_end_ms",
            batch_size=int(self.current_ms.shape[0]),
            device=self.current_ms.device,
            dtype=torch.long,
        )
        valid = valid_input_mask.to(device=self.current_ms.device, dtype=torch.bool)
        if tuple(valid.shape) != tuple(self.current_ms.shape):
            raise ValueError(f"valid_input_mask must have shape {tuple(self.current_ms.shape)}")
        if bool(valid.all()):
            return self

        padded = ~valid
        current_ms = torch.where(padded, target_end.reshape(-1, 1), self.current_ms)
        open_mask = torch.where(padded.unsqueeze(-1), torch.zeros_like(self.open_mask), self.open_mask)
        open_start_ms = torch.where(padded.unsqueeze(-1), torch.full_like(self.open_start_ms, -1), self.open_start_ms)
        open_age_ms = torch.where(padded.unsqueeze(-1), torch.zeros_like(self.open_age_ms), self.open_age_ms)

        emitted_lane_mask = self.emitted_lane_mask
        if emitted_lane_mask is not None:
            emitted_lane_mask = torch.where(
                padded.unsqueeze(-1),
                torch.zeros_like(emitted_lane_mask),
                emitted_lane_mask,
            )
        last_lane_index = self.last_lane_index
        if last_lane_index is not None:
            last_lane_index = torch.where(
                padded,
                torch.full_like(last_lane_index, int(sparse_padded_last_lane_index)),
                last_lane_index,
            )
        return MapperFragmentState(
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            emitted_lane_mask=emitted_lane_mask,
            last_lane_index=last_lane_index,
        )

    def as_mapping(self) -> dict[str, torch.Tensor]:
        mapping = {
            "current_ms": self.current_ms,
            "open_mask": self.open_mask,
            "open_start_ms": self.open_start_ms,
            "open_age_ms": self.open_age_ms,
        }
        if self.emitted_lane_mask is not None:
            mapping["emitted_lane_mask"] = self.emitted_lane_mask
        if self.last_lane_index is not None:
            mapping["last_lane_index"] = self.last_lane_index
        return mapping


@dataclass(frozen=True)
class MapperCarryStateBatch:
    current_ms: torch.Tensor
    open_mask: torch.Tensor
    open_start_ms: torch.Tensor
    open_age_ms: torch.Tensor

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        key: str,
        device: torch.device | str,
    ) -> "MapperCarryStateBatch":
        raw = _require_state_mapping(mapping, key)
        resolved_device = torch.device(device)
        current_ms = _require_nested_tensor(raw, key, "current_ms", ndim=1).to(
            device=resolved_device,
            dtype=torch.long,
        )
        open_mask = _require_nested_tensor(raw, key, "open_mask", ndim=2).to(device=resolved_device, dtype=torch.bool)
        open_start_ms = _require_nested_tensor(raw, key, "open_start_ms", ndim=2).to(
            device=resolved_device,
            dtype=torch.long,
        )
        open_age_ms = _require_nested_tensor(raw, key, "open_age_ms", ndim=2).to(
            device=resolved_device,
            dtype=torch.long,
        )
        if tuple(open_mask.shape) != tuple(open_start_ms.shape) or tuple(open_mask.shape) != tuple(open_age_ms.shape):
            raise ValueError(f"{key}.open_mask, open_start_ms, and open_age_ms must have matching shapes")
        if open_mask.ndim != 2 or int(open_mask.shape[-1]) != KEY_COUNT:
            raise ValueError(f"{key}.open_mask must have shape [B,{KEY_COUNT}]")
        if int(open_mask.shape[0]) != int(current_ms.shape[0]):
            raise ValueError(f"{key} tensors must share batch size")
        return cls(
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
        )

    @property
    def batch_size(self) -> int:
        return int(self.current_ms.shape[0])

    def as_mapping(self) -> dict[str, torch.Tensor]:
        return {
            "current_ms": self.current_ms,
            "open_mask": self.open_mask,
            "open_start_ms": self.open_start_ms,
            "open_age_ms": self.open_age_ms,
        }


@dataclass(frozen=True)
class MapperBatch:
    contract: MapperTokenContract
    decoder_input_tokens: torch.Tensor
    target_fragment_tokens: torch.Tensor
    target_fragment_mask: torch.Tensor
    target_fragment_states: MapperFragmentState
    ln_carry_in: MapperCarryStateBatch
    ln_carry_out: MapperCarryStateBatch
    write_start_ms: torch.Tensor
    write_end_ms: torch.Tensor
    is_full_chart_start: torch.Tensor
    is_full_chart_end: torch.Tensor
    chart_end_ms: torch.Tensor
    target_end_ms: torch.Tensor

    @classmethod
    def from_mapping(
        cls,
        batch: Mapping[str, Any],
        *,
        contract: MapperTokenContract,
        device: torch.device | str | None = None,
    ) -> "MapperBatch":
        if not isinstance(batch, Mapping):
            raise ValueError("batch must be a mapping")
        if not isinstance(contract, MapperTokenContract):
            raise ValueError("contract must be a MapperTokenContract")
        _reject_old_mapper_contract(batch)

        decoder_raw = _require_batch_tensor(batch, "decoder_input_tokens", ndim=2)
        resolved_device = decoder_raw.device if device is None else torch.device(device)
        decoder_input_tokens = decoder_raw.to(device=resolved_device, dtype=torch.long)
        if int(decoder_input_tokens.shape[1]) < 1:
            raise ValueError("decoder_input_tokens must contain at least one fragment position")

        target_fragment_tokens = _require_batch_tensor(batch, "target_fragment_tokens", ndim=2).to(
            device=resolved_device,
            dtype=torch.long,
        )
        target_fragment_mask = _require_batch_tensor(batch, "target_fragment_mask", ndim=2).to(
            device=resolved_device,
            dtype=torch.bool,
        )
        if tuple(target_fragment_tokens.shape) != tuple(decoder_input_tokens.shape):
            raise ValueError("target_fragment_tokens must match decoder_input_tokens shape")
        if tuple(target_fragment_mask.shape) != tuple(decoder_input_tokens.shape):
            raise ValueError("target_fragment_mask must match decoder_input_tokens shape")

        batch_size = int(decoder_input_tokens.shape[0])
        states = MapperFragmentState.from_mapping(
            _require_state_mapping(batch, "target_fragment_states"),
            batch_shape=decoder_input_tokens.shape,
            device=resolved_device,
            requires_sparse_lane_state=contract.requires_sparse_lane_state,
        )
        ln_carry_in = MapperCarryStateBatch.from_mapping(batch, key="ln_carry_in", device=resolved_device)
        ln_carry_out = MapperCarryStateBatch.from_mapping(batch, key="ln_carry_out", device=resolved_device)

        write_start_ms = _require_batch_vector(
            batch,
            "write_start_ms",
            batch_size=batch_size,
            device=resolved_device,
            dtype=torch.long,
        )
        write_end_ms = _require_batch_vector(
            batch,
            "write_end_ms",
            batch_size=batch_size,
            device=resolved_device,
            dtype=torch.long,
        )
        is_full_chart_start = _require_batch_vector(
            batch,
            "is_full_chart_start",
            batch_size=batch_size,
            device=resolved_device,
            dtype=torch.bool,
        )
        is_full_chart_end = _require_batch_vector(
            batch,
            "is_full_chart_end",
            batch_size=batch_size,
            device=resolved_device,
            dtype=torch.bool,
        )
        chart_end_ms = _optional_batch_vector(
            batch,
            "chart_end_ms",
            batch_size=batch_size,
            device=resolved_device,
            dtype=torch.long,
        )
        if chart_end_ms is None:
            chart_end_ms = write_end_ms

        if ln_carry_in.batch_size != batch_size:
            raise ValueError("ln_carry_in tensors must share decoder_input_tokens batch size")
        if ln_carry_out.batch_size != batch_size:
            raise ValueError("ln_carry_out tensors must share decoder_input_tokens batch size")

        if contract.uses_chart_end_for_terminal_windows:
            target_end_ms = torch.where(is_full_chart_end, chart_end_ms, write_end_ms)
        else:
            target_end_ms = write_end_ms

        return cls(
            contract=contract,
            decoder_input_tokens=decoder_input_tokens,
            target_fragment_tokens=target_fragment_tokens,
            target_fragment_mask=target_fragment_mask,
            target_fragment_states=states,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            chart_end_ms=chart_end_ms,
            target_end_ms=target_end_ms,
        )

    @property
    def fragment_states(self) -> MapperFragmentState:
        return self.target_fragment_states

    @property
    def batch_size(self) -> int:
        return int(self.decoder_input_tokens.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.decoder_input_tokens.shape[1])

    @property
    def input_padding_mask(self) -> torch.Tensor:
        return ~self.target_fragment_mask

    def positions(self) -> torch.Tensor:
        positions = torch.arange(self.seq_len, dtype=torch.long, device=self.decoder_input_tokens.device).reshape(1, -1)
        return positions.expand(self.batch_size, -1)


def _reject_old_mapper_contract(batch: Mapping[str, Any]) -> None:
    present = sorted(key for key in _OLD_MAPPER_KEYS if key in batch)
    if present:
        raise ValueError(
            "old target_tokens/teacher_* mapper contract is not supported; "
            "supply decoder_input_tokens, target_fragment_tokens, target_fragment_mask, and target_fragment_states "
            f"instead of {present}"
        )


def _require_batch_tensor(batch: Mapping[str, Any], key: str, *, ndim: int) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch[{key!r}] must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"batch[{key!r}] must be rank {ndim}, got shape {tuple(value.shape)}")
    return value


def _require_batch_vector(
    batch: Mapping[str, Any],
    key: str,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = _require_batch_tensor(batch, key, ndim=1).to(device=device, dtype=dtype)
    if tuple(tensor.shape) != (int(batch_size),):
        raise ValueError(f"{key} must have shape [B] matching decoder_input_tokens")
    return tensor


def _optional_batch_vector(
    batch: Mapping[str, Any],
    key: str,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a torch.Tensor")
    tensor = value.to(device=device, dtype=dtype).reshape(-1)
    if tuple(tensor.shape) != (int(batch_size),):
        raise ValueError(f"{key} must have shape [B] matching decoder_input_tokens")
    return tensor


def _require_state_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"batch[{key!r}] must be a mapping of state tensors")
    return value


def _require_state_tensor(mapping: Mapping[str, Any], key: str, *, ndim: int) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"target_fragment_states.{key} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"target_fragment_states.{key} must be rank {ndim}, got shape {tuple(value.shape)}")
    return value


def _optional_state_tensor(mapping: Mapping[str, Any], key: str, *, ndim: int) -> torch.Tensor | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"target_fragment_states.{key} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"target_fragment_states.{key} must be rank {ndim}, got shape {tuple(value.shape)}")
    return value


def _require_nested_tensor(mapping: Mapping[str, Any], prefix: str, key: str, *, ndim: int) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{prefix}.{key} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{prefix}.{key} must be rank {ndim}, got shape {tuple(value.shape)}")
    return value


def _as_batch_vector(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    tensor = value.to(device=device, dtype=dtype).reshape(-1)
    if tuple(tensor.shape) != (int(batch_size),):
        raise ValueError(f"{name} must have shape [B]")
    return tensor


def _normalize_batch_shape(batch_shape: Sequence[int]) -> tuple[int, int]:
    shape = tuple(int(dim) for dim in batch_shape)
    if len(shape) != 2:
        raise ValueError(f"batch_shape must be [B,S], got {shape}")
    return shape


def _vocab_int_attr(vocab: Any, attr: str) -> int:
    if not hasattr(vocab, attr):
        raise ValueError(f"MapperTokenContract.vocab must define {attr}")
    value = getattr(vocab, attr)
    if callable(value):
        value = value()
    if isinstance(value, bool):
        raise ValueError(f"MapperTokenContract.vocab.{attr} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MapperTokenContract.vocab.{attr} must be an integer") from exc
