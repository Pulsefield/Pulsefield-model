from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .replay import LNCarryState, empty_ln_carry_state, replay_tokens
from .tokenizer import MAPPER_DENSITY_FRAME_MS, MAPPER_DENSITY_FRAMES, TokenizedMapperWindow
from .vocab import MapperTupleVocab


@dataclass(frozen=True)
class DensityCalibration:
    scale: float
    bias: float
    radius: int = 5
    kernel: str = "triangular"

    def predict(self, raw_mass: torch.Tensor) -> torch.Tensor:
        smoothed = smooth_density_mass(raw_mass, radius=self.radius, kernel=self.kernel)
        return float(self.scale) * smoothed + float(self.bias)

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def scatter_gold_onset_mass(
    token_ids: Sequence[int],
    *,
    vocab: MapperTupleVocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState | None = None,
    ln_carry_out: LNCarryState | None = None,
    is_full_chart_end: bool = False,
    frame_count: int = MAPPER_DENSITY_FRAMES,
    frame_ms: int = MAPPER_DENSITY_FRAME_MS,
) -> torch.Tensor:
    carry_in = empty_ln_carry_state(write_start_ms) if ln_carry_in is None else ln_carry_in
    carry_out = empty_ln_carry_state(write_end_ms) if ln_carry_out is None else ln_carry_out
    states = replay_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        ln_carry_in=carry_in,
        ln_carry_out=carry_out,
        is_full_chart_end=is_full_chart_end,
        validate_terminal=True,
    )
    mass = torch.zeros(frame_count, dtype=torch.float32)
    for token_index, token_id in enumerate(token_ids):
        if not vocab.is_event_token(int(token_id)):
            continue
        event_ms = states[token_index].current_ms
        frame_index = (event_ms - int(write_start_ms)) // int(frame_ms)
        if 0 <= frame_index < frame_count:
            mass[frame_index] += float(vocab.event_onset_weight(int(token_id)))
    return mass


def scatter_tokenized_gold_onset_mass(
    tokenized: TokenizedMapperWindow,
    *,
    vocab: MapperTupleVocab,
    frame_count: int = MAPPER_DENSITY_FRAMES,
) -> torch.Tensor:
    return scatter_gold_onset_mass(
        tokenized.target_fragment_ids,
        vocab=vocab,
        write_start_ms=tokenized.write_start_ms,
        write_end_ms=tokenized.write_end_ms,
        ln_carry_in=tokenized.ln_carry_in,
        ln_carry_out=tokenized.ln_carry_out,
        is_full_chart_end=tokenized.is_full_chart_end,
        frame_count=frame_count,
    )


def smooth_density_mass(raw_mass: torch.Tensor, *, radius: int = 5, kernel: str = "triangular") -> torch.Tensor:
    if radius < 0:
        raise ValueError(f"radius must be non-negative: {radius}")
    if radius == 0:
        return raw_mass.to(dtype=torch.float32)
    values = raw_mass.to(dtype=torch.float32)
    squeeze = False
    if values.ndim == 1:
        values = values.unsqueeze(0)
        squeeze = True
    if values.ndim != 2:
        raise ValueError(f"raw_mass must have shape [F] or [B,F], got {tuple(raw_mass.shape)}")

    weights = _kernel_weights(radius=radius, kernel=kernel, device=values.device, dtype=values.dtype)
    smoothed = F.conv1d(values.unsqueeze(1), weights.view(1, 1, -1), padding=radius).squeeze(1)
    return smoothed.squeeze(0) if squeeze else smoothed


def fit_monotonic_affine_calibration(
    smoothed_mass: torch.Tensor,
    density_target: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    radius: int = 5,
    kernel: str = "triangular",
) -> DensityCalibration:
    x = smoothed_mass.detach().to(dtype=torch.float64).reshape(-1)
    target_raw = density_target.detach().to(dtype=torch.float64).reshape(-1)
    if not torch.isfinite(x).all():
        raise ValueError("smoothed_mass must contain only finite values")
    if not torch.isfinite(target_raw).all():
        raise ValueError("density_target must contain only finite values")
    if bool((target_raw < 0.0).any()):
        raise ValueError("density_target must contain raw log1p density values, which must be non-negative")
    if confidence is None:
        weight = torch.ones_like(target_raw)
    else:
        weight = confidence.detach().to(dtype=torch.float64).reshape(-1)
        if not torch.isfinite(weight).all():
            raise ValueError("density confidence must contain only finite values")
    finite = torch.isfinite(x) & torch.isfinite(target_raw) & torch.isfinite(weight) & (weight > 0)
    if not bool(finite.any()):
        raise ValueError("cannot fit density calibration without finite weighted samples")
    x = x[finite]
    y = target_raw[finite]
    weight = weight[finite]

    sw = weight.sum()
    sx = (weight * x).sum()
    sy = (weight * y).sum()
    sxx = (weight * x * x).sum()
    sxy = (weight * x * y).sum()
    denom = sw * sxx - sx * sx
    if abs(float(denom.item())) < 1e-12:
        scale = 0.0
        bias = float((sy / sw).item())
    else:
        scale = float(((sw * sxy - sx * sy) / denom).item())
        bias = float(((sy - scale * sx) / sw).item())
        if scale < 0.0:
            scale = 0.0
            bias = float((sy / sw).item())
    return DensityCalibration(scale=scale, bias=bias, radius=radius, kernel=kernel)


def density_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> dict[str, float]:
    pred = prediction.detach().to(dtype=torch.float64).reshape(-1)
    tgt = target.detach().to(dtype=torch.float64).reshape(-1)
    if confidence is None:
        weight = torch.ones_like(tgt)
    else:
        weight = confidence.detach().to(dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(pred) & torch.isfinite(tgt) & torch.isfinite(weight) & (weight > 0)
    if not bool(finite.any()):
        return {
            "density_frame_mae": math.nan,
            "density_frame_smooth_l1": math.nan,
            "density_window_mean_error": math.nan,
            "density_pearson_corr": math.nan,
            "density_spearman_corr": math.nan,
        }
    pred = pred[finite]
    tgt = tgt[finite]
    weight = weight[finite]
    weight = weight / weight.sum()
    diff = pred - tgt
    return {
        "density_frame_mae": float((weight * diff.abs()).sum().item()),
        "density_frame_smooth_l1": float((weight * F.smooth_l1_loss(pred, tgt, reduction="none")).sum().item()),
        "density_window_mean_error": float((pred.mean() - tgt.mean()).item()),
        "density_pearson_corr": _pearson_corr(pred, tgt),
        "density_spearman_corr": _pearson_corr(_rank_ordinal(pred), _rank_ordinal(tgt)),
    }


def _kernel_weights(*, radius: int, kernel: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel == "triangular":
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype).abs()
        weights = (radius + 1) - offsets
    elif kernel == "gaussian":
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        sigma = max(float(radius) / 2.0, 1.0)
        weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    else:
        raise ValueError(f"unsupported density smoothing kernel: {kernel}")
    return weights / weights.sum()


def _pearson_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return math.nan
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denom.item()) == 0.0:
        return math.nan
    return float((left_centered * right_centered).sum().div(denom).item())


def _rank_ordinal(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), dtype=values.dtype, device=values.device)
    return ranks
