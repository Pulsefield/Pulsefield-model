from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from pulsefield_model.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from pulsefield_model.models.control.encoder import ControlEncoderOutput


CONFIDENCE_BY_VALUE_FEATURE = {
    "density_level": "density_confidence",
    "density_burst": "density_confidence",
    "ln_change_rate_gated": "ln_change_confidence",
    "chord_ratio": "chord_confidence",
    "jack_excess": "jack_confidence",
    "jack_streak_exposure": "jack_streak_confidence",
    "hand_balance_signed": "hand_confidence",
    "hand_imbalance_abs": "hand_confidence",
    "repeat_exact": "repeat_confidence",
    "repeat_shift": "repeat_confidence",
    "repeat_motion": "repeat_confidence",
}
SPARSE_VALUE_FEATURES = (
    "jack_excess",
    "hand_imbalance_abs",
    "repeat_exact",
    "repeat_shift",
    "repeat_motion",
)


@dataclass(frozen=True)
class ControlLossConfig:
    confidence_loss_weight: float = 0.25
    feature_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_FEATURE_WEIGHTS))
    smooth_l1_deltas: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_SMOOTH_L1_DELTAS))
    sparse_low: float = 0.10
    sparse_high: float = 0.50
    sparse_boost: float = 4.0
    hand_balance_imbalance_full_weight: float = 0.25
    ln_change_support_min: float = 2.0
    ln_change_support_full: float = 3.0


@dataclass(frozen=True)
class ControlLossOutput:
    total_loss: torch.Tensor
    value_loss: torch.Tensor
    confidence_loss: torch.Tensor
    metrics: dict[str, float]
    metric_numerators: dict[str, float] = field(default_factory=dict)
    metric_denominators: dict[str, float] = field(default_factory=dict)


class ControlModelLoss(nn.Module):
    def __init__(self, config: ControlLossConfig | None = None) -> None:
        super().__init__()
        config = ControlLossConfig() if config is None else config
        self.config = config
        _validate_config(config)
        self._value_indexes = {name: index for index, name in enumerate(VALUE_FEATURE_NAMES)}
        self._confidence_indexes = {name: index for index, name in enumerate(CONFIDENCE_FEATURE_NAMES)}
        self._target_confidence_indexes = {
            name: MODEL_FEATURE_NAMES.index(name)
            for name in CONFIDENCE_FEATURE_NAMES
        }

    def forward(
        self,
        output: ControlEncoderOutput,
        *,
        control_v3_target: torch.Tensor,
        target_valid_mask: torch.Tensor,
        ln_change_n_eff_target: torch.Tensor | None = None,
    ) -> ControlLossOutput:
        _validate_shapes(output, control_v3_target=control_v3_target, target_valid_mask=target_valid_mask)
        valid_bool = target_valid_mask.to(device=output.value_pred.device, dtype=torch.bool)
        target = control_v3_target.to(dtype=output.value_pred.dtype, device=output.value_pred.device)
        _validate_finite_valid_frames(target, valid_bool, "control_v3_target")
        _validate_finite_valid_frames(output.value_pred, valid_bool, "value_pred")
        _validate_finite_valid_frames(output.confidence_pred, valid_bool, "confidence_pred")
        _validate_finite_valid_frames(output.compound_confidence_pred, valid_bool, "compound_confidence_pred")
        target = _zero_invalid_frames(target, valid_bool)
        value_pred = _zero_invalid_frames(output.value_pred, valid_bool)
        confidence_pred = _zero_invalid_frames(output.confidence_pred, valid_bool)
        compound_confidence_pred = _zero_invalid_frames(output.compound_confidence_pred.squeeze(-1), valid_bool)
        valid = valid_bool.to(dtype=output.value_pred.dtype)
        value_target = target[..., : len(VALUE_FEATURE_NAMES)]
        confidence_target = target[..., len(VALUE_FEATURE_NAMES) :]
        weights = self.value_weights(
            control_v3_target=target,
            target_valid_mask=valid_bool,
            ln_change_n_eff_target=ln_change_n_eff_target,
            dtype=output.value_pred.dtype,
            device=output.value_pred.device,
        )

        value_numerators: list[torch.Tensor] = []
        value_denominators: list[torch.Tensor] = []
        metrics: dict[str, float] = {}
        metric_numerators: dict[str, float] = {}
        metric_denominators: dict[str, float] = {}
        for feature_index, feature_name in enumerate(VALUE_FEATURE_NAMES):
            delta = self.config.smooth_l1_deltas[feature_name]
            loss = _masked_smooth_l1_loss(
                value_pred[..., feature_index],
                value_target[..., feature_index],
                valid_bool,
                beta=delta,
            )
            feature_weight = weights[..., feature_index]
            numerator = (loss * feature_weight).sum()
            denominator = feature_weight.sum()
            value_numerators.append(numerator)
            value_denominators.append(denominator)
            _record_mean_metric(
                metrics,
                metric_numerators,
                metric_denominators,
                f"value/{feature_name}/weighted_smooth_l1",
                numerator,
                denominator,
            )
            mae_numerator, mae_denominator = _masked_abs_error_parts(
                value_pred[..., feature_index],
                value_target[..., feature_index],
                valid_bool,
            )
            _record_mean_metric(
                metrics,
                metric_numerators,
                metric_denominators,
                f"value/{feature_name}/mae",
                mae_numerator,
                mae_denominator,
            )

        value_loss = _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "loss/value",
            torch.stack(value_numerators).sum(),
            torch.stack(value_denominators).sum(),
        )
        confidence_numerator, confidence_denominator = _masked_squared_error_parts(
            confidence_pred,
            confidence_target,
            valid_bool,
        )
        confidence_loss = _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "loss/confidence",
            confidence_numerator,
            confidence_denominator,
        )
        total_loss = value_loss + self.config.confidence_loss_weight * confidence_loss

        for feature_index, feature_name in enumerate(CONFIDENCE_FEATURE_NAMES):
            mae_numerator, mae_denominator = _masked_abs_error_parts(
                confidence_pred[..., feature_index],
                confidence_target[..., feature_index],
                valid_bool,
            )
            _record_mean_metric(
                metrics,
                metric_numerators,
                metric_denominators,
                f"confidence/{feature_name}/mae",
                mae_numerator,
                mae_denominator,
            )
        control_confidence_index = self._confidence_indexes["control_confidence"]
        compound_numerator, compound_denominator = _masked_abs_error_parts(
            compound_confidence_pred,
            confidence_target[..., control_confidence_index],
            valid_bool,
        )
        _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "confidence/compound_control_confidence/mae",
            compound_numerator,
            compound_denominator,
        )
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

        ln_support = self.ln_change_support_weight(
            ln_change_n_eff_target,
            target_valid_mask=target_valid_mask,
            dtype=output.value_pred.dtype,
            device=output.value_pred.device,
        )
        support_numerator, support_denominator = _masked_mean_parts(ln_support, valid_bool)
        _record_mean_metric(
            metrics,
            metric_numerators,
            metric_denominators,
            "value/ln_change_rate_gated/support_weight_mean",
            support_numerator,
            support_denominator,
        )
        metrics["value/ln_change_rate_gated/support_masked_frame_count"] = int(
            ((ln_support <= 0.0) & valid_bool).sum().detach().cpu()
        )

        for feature_name in SPARSE_VALUE_FEATURES:
            feature_index = self._value_indexes[feature_name]
            positive_mask = (value_target[..., feature_index] >= 0.50) & valid_bool
            positive_numerator, positive_denominator = _masked_abs_error_parts(
                value_pred[..., feature_index],
                value_target[..., feature_index],
                positive_mask,
            )
            _record_mean_metric(
                metrics,
                metric_numerators,
                metric_denominators,
                f"value/{feature_name}/positive_frame_mae",
                positive_numerator,
                positive_denominator,
            )
            max_numerator, max_denominator = _positive_window_max_mae_parts(
                value_pred[..., feature_index],
                value_target[..., feature_index],
                valid_bool,
            )
            _record_mean_metric(
                metrics,
                metric_numerators,
                metric_denominators,
                f"value/{feature_name}/positive_window_max_mae",
                max_numerator,
                max_denominator,
            )

        return ControlLossOutput(
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
        control_v3_target: torch.Tensor,
        target_valid_mask: torch.Tensor,
        ln_change_n_eff_target: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        dtype = control_v3_target.dtype if dtype is None else dtype
        device = control_v3_target.device if device is None else device
        valid_bool = target_valid_mask.to(dtype=torch.bool, device=device)
        target = control_v3_target.to(dtype=dtype, device=device)
        _validate_finite_valid_frames(target, valid_bool, "control_v3_target")
        target = _zero_invalid_frames(target, valid_bool)
        valid = valid_bool.to(dtype=dtype)
        value_target = target[..., : len(VALUE_FEATURE_NAMES)]
        weights = target.new_zeros((*target.shape[:2], len(VALUE_FEATURE_NAMES)))

        for feature_index, feature_name in enumerate(VALUE_FEATURE_NAMES):
            weight = valid * self.config.feature_weights[feature_name]
            confidence_name = CONFIDENCE_BY_VALUE_FEATURE.get(feature_name)
            if confidence_name is not None:
                weight = weight * target[..., self._target_confidence_indexes[confidence_name]]
            if feature_name in SPARSE_VALUE_FEATURES:
                weight = weight * self._sparse_multiplier(value_target[..., feature_index])
            if feature_name == "ln_change_rate_gated":
                weight = weight * self.ln_change_support_weight(
                    ln_change_n_eff_target,
                    target_valid_mask=target_valid_mask,
                    dtype=dtype,
                    device=device,
                )
            if feature_name == "hand_balance_signed":
                hand_imbalance = value_target[..., self._value_indexes["hand_imbalance_abs"]]
                weight = (
                    valid
                    * target[..., self._target_confidence_indexes["hand_confidence"]]
                    * self.config.feature_weights[feature_name]
                    * torch.clamp(hand_imbalance / self.config.hand_balance_imbalance_full_weight, 0.0, 1.0)
                )
            weights[..., feature_index] = weight
        return weights

    def ln_change_support_weight(
        self,
        ln_change_n_eff_target: torch.Tensor | None,
        *,
        target_valid_mask: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if ln_change_n_eff_target is None:
            raise ValueError("ln_change_n_eff_target is required for support-aware ln_change_rate_gated loss")
        n_eff = ln_change_n_eff_target.to(dtype=dtype, device=device)
        if n_eff.shape != target_valid_mask.shape:
            raise ValueError("ln_change_n_eff_target must have shape [B,100]")
        valid_bool = target_valid_mask.to(dtype=torch.bool, device=device)
        _validate_finite_valid_frames(n_eff, valid_bool, "ln_change_n_eff_target")
        n_eff = _zero_invalid_frames(n_eff, valid_bool)
        span = self.config.ln_change_support_full - self.config.ln_change_support_min
        return torch.clamp((n_eff - self.config.ln_change_support_min) / span, 0.0, 1.0)

    def _sparse_multiplier(self, target: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.config.sparse_boost * _smoothstep(target, self.config.sparse_low, self.config.sparse_high)


_DEFAULT_FEATURE_WEIGHTS = {
    "density_level": 1.00,
    "density_burst": 0.35,
    "hold_occupancy": 1.00,
    "ln_change_rate_gated": 0.60,
    "chord_ratio": 1.00,
    "jack_excess": 0.90,
    "jack_streak_exposure": 1.00,
    "hand_balance_signed": 0.60,
    "hand_imbalance_abs": 0.90,
    "repeat_exact": 1.00,
    "repeat_shift": 1.00,
    "repeat_motion": 1.00,
}
_DEFAULT_SMOOTH_L1_DELTAS = {
    "density_level": 0.20,
    "density_burst": 0.10,
    "hold_occupancy": 0.10,
    "ln_change_rate_gated": 0.20,
    "chord_ratio": 0.10,
    "jack_excess": 0.10,
    "jack_streak_exposure": 0.10,
    "hand_balance_signed": 0.10,
    "hand_imbalance_abs": 0.10,
    "repeat_exact": 0.10,
    "repeat_shift": 0.10,
    "repeat_motion": 0.10,
}


def _smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    scaled = torch.clamp((value - low) / (high - low), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


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


def _masked_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    mask_bool = mask.to(dtype=torch.bool, device=pred.device)
    safe_pred = _zero_invalid_frames(pred, mask_bool)
    safe_target = _zero_invalid_frames(target.to(dtype=pred.dtype, device=pred.device), mask_bool)
    return F.smooth_l1_loss(safe_pred, safe_target, beta=beta, reduction="none")


def _masked_squared_error_parts(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_bool = mask.to(dtype=torch.bool, device=pred.device)
    safe_pred = _zero_invalid_frames(pred, mask_bool)
    safe_target = _zero_invalid_frames(target.to(dtype=pred.dtype, device=pred.device), mask_bool)
    squared_error = (safe_pred - safe_target).square()
    elements_per_frame = int(math.prod(pred.shape[mask_bool.ndim:]))
    denominator = mask_bool.to(dtype=pred.dtype).sum() * elements_per_frame
    return squared_error.sum(), denominator


def _masked_abs_error_parts(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_bool = mask.to(dtype=torch.bool, device=pred.device)
    safe_pred = _zero_invalid_frames(pred, mask_bool)
    safe_target = _zero_invalid_frames(target.to(dtype=pred.dtype, device=pred.device), mask_bool)
    return (safe_pred - safe_target).abs().sum(), mask_bool.to(dtype=pred.dtype).sum()


def _masked_mean_parts(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_bool = mask.to(dtype=torch.bool, device=value.device)
    return _zero_invalid_frames(value, mask_bool).sum(), mask_bool.to(dtype=value.dtype).sum()


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    numerator, denominator = _masked_abs_error_parts(pred, target, mask)
    value = _safe_div(numerator, denominator)
    return float(value.detach().cpu())


def _masked_mean_float(value: torch.Tensor, mask: torch.Tensor) -> float:
    numerator, denominator = _masked_mean_parts(value, mask)
    mean = _safe_div(numerator, denominator)
    return float(mean.detach().cpu())


def _positive_window_max_mae(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> float:
    numerator, denominator = _positive_window_max_mae_parts(pred, target, valid_mask)
    return float(_safe_div(numerator, denominator).detach().cpu())


def _positive_window_max_mae_parts(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values: list[torch.Tensor] = []
    for batch_index in range(int(pred.shape[0])):
        valid = valid_mask[batch_index]
        if not bool(valid.any()):
            continue
        target_values = target[batch_index][valid]
        if not bool((target_values >= 0.50).any()):
            continue
        pred_values = pred[batch_index][valid]
        values.append((pred_values.max() - target_values.max()).abs())
    if not values:
        return pred.new_zeros(()), pred.new_zeros(())
    return torch.stack(values).sum(), pred.new_tensor(float(len(values)))


def _validate_shapes(
    output: ControlEncoderOutput,
    *,
    control_v3_target: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> None:
    expected_value_shape = (*control_v3_target.shape[:2], len(VALUE_FEATURE_NAMES))
    expected_confidence_shape = (*control_v3_target.shape[:2], len(CONFIDENCE_FEATURE_NAMES))
    if control_v3_target.shape[-1] != len(MODEL_FEATURE_NAMES):
        raise ValueError(f"control_v3_target must have {len(MODEL_FEATURE_NAMES)} channels")
    if output.value_pred.shape != expected_value_shape:
        raise ValueError(f"value_pred must have shape {expected_value_shape}, got {tuple(output.value_pred.shape)}")
    if output.confidence_pred.shape != expected_confidence_shape:
        raise ValueError(
            f"confidence_pred must have shape {expected_confidence_shape}, got {tuple(output.confidence_pred.shape)}"
        )
    if output.compound_confidence_pred.shape != (*control_v3_target.shape[:2], 1):
        raise ValueError("compound_confidence_pred must have shape [B,100,1]")
    if target_valid_mask.shape != control_v3_target.shape[:2] or target_valid_mask.dtype != torch.bool:
        raise ValueError("target_valid_mask must be bool with shape [B,100]")


def _validate_config(config: ControlLossConfig) -> None:
    _require_finite_number(config.confidence_loss_weight, "confidence_loss_weight")
    _require_finite_number(config.sparse_low, "sparse_low")
    _require_finite_number(config.sparse_high, "sparse_high")
    _require_finite_number(config.sparse_boost, "sparse_boost")
    _require_finite_number(config.hand_balance_imbalance_full_weight, "hand_balance_imbalance_full_weight")
    _require_finite_number(config.ln_change_support_min, "ln_change_support_min")
    _require_finite_number(config.ln_change_support_full, "ln_change_support_full")
    if config.confidence_loss_weight < 0.0:
        raise ValueError("confidence_loss_weight must be non-negative")
    if config.sparse_high <= config.sparse_low:
        raise ValueError("sparse_high must be greater than sparse_low")
    if config.sparse_boost < 0.0:
        raise ValueError("sparse_boost must be non-negative")
    if config.hand_balance_imbalance_full_weight <= 0.0:
        raise ValueError("hand_balance_imbalance_full_weight must be positive")
    if config.ln_change_support_full <= config.ln_change_support_min:
        raise ValueError("ln_change_support_full must be greater than ln_change_support_min")
    _require_feature_keys(config.feature_weights, "feature_weights")
    _require_feature_keys(config.smooth_l1_deltas, "smooth_l1_deltas")
    for feature_name, value in config.feature_weights.items():
        _require_finite_number(value, f"feature_weights[{feature_name}]")
        if value < 0.0:
            raise ValueError("feature_weights must be non-negative")
    for feature_name, value in config.smooth_l1_deltas.items():
        _require_finite_number(value, f"smooth_l1_deltas[{feature_name}]")
        if value <= 0.0:
            raise ValueError("smooth_l1_deltas must be positive")


def _require_feature_keys(mapping: dict[str, float], name: str) -> None:
    missing = sorted(set(VALUE_FEATURE_NAMES) - set(mapping))
    unknown = sorted(set(mapping) - set(VALUE_FEATURE_NAMES))
    if missing or unknown:
        raise ValueError(f"{name} keys must match VALUE_FEATURE_NAMES, missing={missing}, unknown={unknown}")


def _require_finite_number(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
