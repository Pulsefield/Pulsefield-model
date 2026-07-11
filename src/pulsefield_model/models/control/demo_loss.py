from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from pulsefield_model.data.control_demo_global_windows import (
    CONTROL_DEMO_TARGET_CHANNELS,
    DENSITY_CONFIDENCE_TARGET_INDEX,
    DENSITY_LEVEL_TARGET_INDEX,
)


class _ControlDemoLossOutputLike(Protocol):
    value_pred: torch.Tensor


@dataclass(frozen=True)
class ControlDemoLossConfig:
    density_loss_weight: float = 1.0
    smooth_l1_delta: float = 0.20


@dataclass(frozen=True)
class ControlDemoLossOutput:
    total_loss: torch.Tensor
    value_loss: torch.Tensor
    confidence_loss: torch.Tensor
    metrics: dict[str, float]
    metric_numerators: dict[str, float] = field(default_factory=dict)
    metric_denominators: dict[str, float] = field(default_factory=dict)


class ControlDemoModelLoss(nn.Module):
    def __init__(self, config: ControlDemoLossConfig | None = None) -> None:
        super().__init__()
        config = ControlDemoLossConfig() if config is None else config
        _validate_config(config)
        self.config = config

    def forward(
        self,
        output: _ControlDemoLossOutputLike,
        *,
        control_demo_target: torch.Tensor,
        target_valid_mask: torch.Tensor,
    ) -> ControlDemoLossOutput:
        _validate_shapes(output, control_demo_target=control_demo_target, target_valid_mask=target_valid_mask)
        valid_bool = target_valid_mask.to(device=output.value_pred.device, dtype=torch.bool)
        target = control_demo_target.to(dtype=output.value_pred.dtype, device=output.value_pred.device)
        _validate_finite_valid_frames(target, valid_bool, "control_demo_target")
        _validate_finite_valid_frames(output.value_pred, valid_bool, "value_pred")
        target = _zero_invalid_frames(target, valid_bool)
        value_pred = _zero_invalid_frames(output.value_pred.squeeze(-1), valid_bool)
        valid = valid_bool.to(dtype=output.value_pred.dtype)

        density_target = target[..., DENSITY_LEVEL_TARGET_INDEX]
        density_confidence = target[..., DENSITY_CONFIDENCE_TARGET_INDEX].clamp(0.0, 1.0)
        weights = self.value_weights(control_demo_target=target, target_valid_mask=valid_bool)
        per_frame_loss = F.smooth_l1_loss(
            value_pred,
            density_target,
            beta=self.config.smooth_l1_delta,
            reduction="none",
        )

        metrics: dict[str, float] = {}
        metric_numerators: dict[str, float] = {}
        metric_denominators: dict[str, float] = {}
        value_loss = _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "loss/value",
            (per_frame_loss * weights).sum(),
            weights.sum(),
        )
        mae_numerator, mae_denominator = _masked_abs_error_parts(value_pred, density_target, valid_bool)
        _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "value/density_level/mae",
            mae_numerator,
            mae_denominator,
        )
        _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "value/density_level/weighted_smooth_l1",
            (per_frame_loss * weights).sum(),
            weights.sum(),
        )
        confidence_numerator, confidence_denominator = _masked_mean_parts(density_confidence, valid_bool)
        _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "target/density_confidence_mean",
            confidence_numerator,
            confidence_denominator,
        )
        confidence_loss = value_loss.new_zeros(())
        total_loss = value_loss
        metrics["loss/confidence"] = 0.0
        metric_numerators["loss/confidence"] = 0.0
        metric_denominators["loss/confidence"] = float(valid.sum().detach().cpu())
        metrics["loss/total"] = float(total_loss.detach().cpu())
        _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "target/valid_frame_rate",
            valid.sum(),
            valid.new_tensor(float(valid.numel())),
        )
        metrics["target/valid_frame_count"] = int(valid_bool.sum().detach().cpu())
        metrics["target/masked_frame_count"] = int((valid_bool.numel() - valid_bool.sum()).detach().cpu())

        return ControlDemoLossOutput(
            total_loss=total_loss,
            value_loss=value_loss,
            confidence_loss=confidence_loss,
            metrics=metrics,
            metric_numerators=metric_numerators,
            metric_denominators=metric_denominators,
        )

    def value_weights(
        self,
        *,
        control_demo_target: torch.Tensor,
        target_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if control_demo_target.shape[-1] != CONTROL_DEMO_TARGET_CHANNELS:
            raise ValueError(f"control_demo_target must have {CONTROL_DEMO_TARGET_CHANNELS} channels")
        valid_bool = target_valid_mask.to(dtype=torch.bool, device=control_demo_target.device)
        _validate_finite_valid_frames(control_demo_target, valid_bool, "control_demo_target")
        target = _zero_invalid_frames(control_demo_target, valid_bool)
        valid = valid_bool.to(dtype=target.dtype)
        density_confidence = target[..., DENSITY_CONFIDENCE_TARGET_INDEX].clamp(0.0, 1.0)
        return valid * density_confidence * self.config.density_loss_weight


def _validate_shapes(
    output: _ControlDemoLossOutputLike,
    *,
    control_demo_target: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> None:
    expected_value_shape = (*control_demo_target.shape[:2], 1)
    if control_demo_target.shape[-1] != CONTROL_DEMO_TARGET_CHANNELS:
        raise ValueError(f"control_demo_target must have {CONTROL_DEMO_TARGET_CHANNELS} channels")
    if output.value_pred.shape != expected_value_shape:
        raise ValueError(f"value_pred must have shape {expected_value_shape}, got {tuple(output.value_pred.shape)}")
    if target_valid_mask.shape != control_demo_target.shape[:2] or target_valid_mask.dtype != torch.bool:
        raise ValueError("target_valid_mask must be bool with shape [B,100]")


def _validate_config(config: ControlDemoLossConfig) -> None:
    _require_finite_number(config.density_loss_weight, "density_loss_weight")
    _require_finite_number(config.smooth_l1_delta, "smooth_l1_delta")
    if config.density_loss_weight < 0.0:
        raise ValueError("density_loss_weight must be non-negative")
    if config.smooth_l1_delta <= 0.0:
        raise ValueError("smooth_l1_delta must be positive")


def _validate_finite_valid_frames(value: torch.Tensor, valid_mask: torch.Tensor, name: str) -> None:
    mask = valid_mask.to(dtype=torch.bool, device=value.device)
    if bool(mask.any()) and not bool(torch.isfinite(value[mask]).all()):
        raise ValueError(f"{name} must contain only finite values on valid target frames")


def _zero_invalid_frames(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(dtype=torch.bool, device=value.device)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, value, torch.zeros_like(value))


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)


def _record_mean_metric(
    metrics: dict[str, float],
    metric_numerators: dict[str, float],
    metric_denominators: dict[str, float],
    key: str,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    value = _safe_div(numerator, denominator)
    metrics[key] = float(value.detach().cpu())
    metric_numerators[key] = float(numerator.detach().cpu())
    metric_denominators[key] = float(denominator.detach().cpu())
    return value


def _masked_abs_error_parts(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_bool = mask.to(dtype=torch.bool, device=pred.device)
    safe_pred = _zero_invalid_frames(pred, mask_bool)
    safe_target = _zero_invalid_frames(target.to(dtype=pred.dtype, device=pred.device), mask_bool)
    return (safe_pred - safe_target).abs().sum(), mask_bool.to(dtype=pred.dtype).sum()


def _masked_mean_parts(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_bool = mask.to(dtype=torch.bool, device=value.device)
    return _zero_invalid_frames(value, mask_bool).sum(), mask_bool.to(dtype=value.dtype).sum()


def _require_finite_number(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
