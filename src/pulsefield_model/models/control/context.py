from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from pulsefield_model.data.control_windows import (
    CONTEXT_LENGTH_FRAMES,
    DENSE_TIMING_V2_CHANNELS,
    PACKED_MEL_CHANNELS,
    TARGET_OFFSET_IN_CONTEXT,
    TARGET_WINDOW_LENGTH_FRAMES,
)


def prepare_control_context_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    full_mel = _require_tensor(batch, "full_mel", ndim=3, channels=PACKED_MEL_CHANNELS)
    full_dense_timing_v2 = _require_tensor(
        batch,
        "full_dense_timing_v2",
        ndim=3,
        channels=DENSE_TIMING_V2_CHANNELS,
    )
    if full_mel.shape[:2] != full_dense_timing_v2.shape[:2]:
        raise ValueError("full_mel and full_dense_timing_v2 must share batch/frame dimensions")
    if full_mel.device != full_dense_timing_v2.device:
        raise ValueError("full_mel and full_dense_timing_v2 must be on the same device")
    if full_mel.dtype != torch.float32:
        raise ValueError(f"full_mel must be float32, got {full_mel.dtype}")
    if full_dense_timing_v2.dtype != torch.float32:
        raise ValueError(f"full_dense_timing_v2 must be float32, got {full_dense_timing_v2.dtype}")
    batch_size = int(full_mel.shape[0])
    if "padding_mask" in batch:
        padding_mask = batch["padding_mask"]
        if not isinstance(padding_mask, torch.Tensor):
            raise ValueError("padding_mask must be a torch.Tensor")
        if padding_mask.dtype != torch.bool:
            raise ValueError(f"padding_mask must be bool, got {padding_mask.dtype}")
        if tuple(padding_mask.shape) != tuple(full_mel.shape[:2]):
            raise ValueError(f"padding_mask must have shape {tuple(full_mel.shape[:2])}, got {tuple(padding_mask.shape)}")
        if padding_mask.device != full_mel.device:
            raise ValueError("padding_mask must be on the same device as full_mel")
    else:
        padding_mask = torch.zeros(full_mel.shape[:2], dtype=torch.bool, device=full_mel.device)

    target_start_frame = _require_integer_tensor(batch, "target_start_frame", device=full_mel.device)
    frame_count = _require_integer_tensor(batch, "frame_count", device=full_mel.device)
    if target_start_frame.shape[0] != full_mel.shape[0] or frame_count.shape[0] != full_mel.shape[0]:
        raise ValueError("target_start_frame and frame_count must have one value per batch item")
    if torch.any(frame_count <= 0):
        raise ValueError("frame_count must be positive")
    if torch.any(frame_count > full_mel.shape[1]):
        raise ValueError("frame_count cannot exceed padded full-song length")
    if torch.any(target_start_frame < 0):
        raise ValueError("target_start_frame must be non-negative")
    if torch.any(target_start_frame >= frame_count):
        raise ValueError("target_start_frame must be less than frame_count")

    _validate_unpadded_finite(full_mel, padding_mask=padding_mask, frame_count=frame_count, name="full_mel")
    _validate_unpadded_finite(
        full_dense_timing_v2,
        padding_mask=padding_mask,
        frame_count=frame_count,
        name="full_dense_timing_v2",
    )

    context_mel = full_mel.new_zeros((batch_size, CONTEXT_LENGTH_FRAMES, PACKED_MEL_CHANNELS))
    context_dense_timing_v2 = full_dense_timing_v2.new_zeros(
        (batch_size, CONTEXT_LENGTH_FRAMES, DENSE_TIMING_V2_CHANNELS)
    )
    context_padding_mask = torch.ones((batch_size, CONTEXT_LENGTH_FRAMES), dtype=torch.bool, device=full_mel.device)

    context_start_frame = target_start_frame - TARGET_OFFSET_IN_CONTEXT
    context_offsets = torch.arange(CONTEXT_LENGTH_FRAMES, dtype=torch.long, device=full_mel.device)

    for batch_index in range(batch_size):
        source_frames = context_start_frame[batch_index] + context_offsets
        in_song = (source_frames >= 0) & (source_frames < frame_count[batch_index])
        if bool(in_song.any()):
            in_song_destination = in_song.nonzero(as_tuple=False).flatten()
            in_song_source = source_frames[in_song_destination]
            unpadded = ~padding_mask[batch_index, in_song_source]
            destination = in_song_destination[unpadded]
            source = in_song_source[unpadded]
        else:
            destination = torch.empty((0,), dtype=torch.long, device=full_mel.device)
            source = torch.empty((0,), dtype=torch.long, device=full_mel.device)
        if int(destination.numel()) > 0:
            context_mel[batch_index, destination] = full_mel[batch_index, source]
            context_dense_timing_v2[batch_index, destination] = full_dense_timing_v2[batch_index, source]
            context_padding_mask[batch_index, destination] = False

    target_offsets = torch.arange(TARGET_WINDOW_LENGTH_FRAMES, dtype=torch.long, device=full_mel.device)
    target_frames = target_start_frame.unsqueeze(1) + target_offsets.unsqueeze(0)
    target_in_frame_count = target_frames < frame_count.unsqueeze(1)
    safe_target_frames = target_frames.clamp(min=0, max=max(int(full_mel.shape[1]) - 1, 0))
    target_unpadded = ~padding_mask.gather(1, safe_target_frames)
    target_valid_mask = target_in_frame_count & target_unpadded

    prepared = dict(batch)
    prepared["context_mel"] = context_mel
    prepared["context_dense_timing_v2"] = context_dense_timing_v2
    prepared["context_padding_mask"] = context_padding_mask
    prepared["target_valid_mask"] = target_valid_mask
    prepared["context_start_frame"] = context_start_frame
    prepared["target_offset_in_context"] = torch.full(
        (batch_size,),
        TARGET_OFFSET_IN_CONTEXT,
        dtype=torch.long,
        device=full_mel.device,
    )
    return prepared


def _require_integer_tensor(batch: Mapping[str, Any], key: str, *, device: torch.device) -> torch.Tensor:
    value = _require_tensor(batch, key, ndim=1)
    if value.dtype == torch.bool:
        raise ValueError(f"{key} must be an integer tensor, got bool")
    if not value.dtype.is_floating_point and not value.dtype.is_complex:
        return value.to(dtype=torch.long, device=device)
    if value.dtype.is_complex:
        raise ValueError(f"{key} must be an integer tensor, got {value.dtype}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{key} must contain only finite values")
    if not torch.equal(value, value.round()):
        raise ValueError(f"{key} must contain integer frame indexes")
    return value.to(dtype=torch.long, device=device)


def _validate_unpadded_finite(
    tensor: torch.Tensor,
    *,
    padding_mask: torch.Tensor,
    frame_count: torch.Tensor,
    name: str,
) -> None:
    frame_offsets = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(0)
    in_frame_count = frame_offsets < frame_count.unsqueeze(1)
    valid = in_frame_count & ~padding_mask
    if bool(valid.any()) and not torch.isfinite(tensor[valid]).all():
        raise ValueError(f"{name} must contain only finite values in unpadded frames")


def _require_tensor(
    batch: Mapping[str, Any],
    key: str,
    *,
    ndim: int,
    channels: int | None = None,
) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{key} must be rank {ndim}, got shape {tuple(value.shape)}")
    if channels is not None and int(value.shape[-1]) != channels:
        raise ValueError(f"{key} must have {channels} channels, got shape {tuple(value.shape)}")
    return value
