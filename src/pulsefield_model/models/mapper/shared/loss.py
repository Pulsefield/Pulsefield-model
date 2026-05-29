from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .tokenizer import MAPPER_DENSITY_FRAME_MS, MAPPER_DENSITY_FRAMES
from .vocab import MapperTupleVocab


@dataclass(frozen=True)
class MapperTupleLossConfig:
    lambda_density: float = 0.0
    lambda_ln_close: float = 0.05
    lambda_adapter_reg: float = 1e-5
    ln_close_pos_weight: float = 1.0
    ln_close_focal_gamma: float = 1.5
    density_calibration_scale: float = 1.0
    density_calibration_bias: float = 0.0


@dataclass(frozen=True)
class MapperTupleLossOutput:
    total_loss: torch.Tensor
    token_loss: torch.Tensor
    ln_close_loss: torch.Tensor
    density_loss: torch.Tensor
    adapter_reg_loss: torch.Tensor
    metrics: dict[str, float]
    metric_numerators: dict[str, float] = field(default_factory=dict)
    metric_denominators: dict[str, float] = field(default_factory=dict)


class MapperTupleModelLoss(nn.Module):
    def __init__(self, config: MapperTupleLossConfig | None = None, *, vocab: MapperTupleVocab | None = None) -> None:
        super().__init__()
        config = MapperTupleLossConfig() if config is None else config
        _validate_config(config)
        self.config = config
        self.vocab = MapperTupleVocab() if vocab is None else vocab

    def forward(self, output: Any, batch: Mapping[str, torch.Tensor]) -> MapperTupleLossOutput:
        logits_final = _require_tensor_attr(output, "logits_final")
        _reject_old_mapper_contract(batch)
        target = _require_batch_tensor(batch, "target_fragment_tokens")
        if target.ndim != 2:
            raise ValueError(f"target_fragment_tokens must have shape [B,T], got {tuple(target.shape)}")
        if logits_final.ndim != 3 or tuple(logits_final.shape[:2]) != tuple(target.shape):
            raise ValueError(
                "logits_final must have shape [B,T,V] matching target_fragment_tokens, "
                f"got logits={tuple(logits_final.shape)} target={tuple(target.shape)}"
            )

        target = target.to(device=logits_final.device, dtype=torch.long)
        target_mask = _target_loss_mask(batch, target=target, pad_id=self.vocab.pad_id)
        token_loss = token_cross_entropy(
            logits_final,
            target,
            pad_id=self.vocab.pad_id,
            target_mask=target_mask,
        )

        disabled_zero = logits_final.new_zeros(())
        if self.config.lambda_ln_close > 0.0:
            close_logits = _require_tensor_attr(output, "ln_close_logits")
            close_labels = _require_batch_tensor(batch, "close_labels")
            close_mask = _require_batch_tensor(batch, "close_label_mask").to(
                device=close_logits.device,
                dtype=torch.bool,
            )
            close_mask = close_mask & target_mask.to(device=close_logits.device, dtype=torch.bool).unsqueeze(-1)
            ln_close_loss = ln_close_focal_bce_loss(
                close_logits=close_logits,
                labels=close_labels,
                mask=close_mask,
                pos_weight=self.config.ln_close_pos_weight,
                gamma=self.config.ln_close_focal_gamma,
            )
            close_open_count = int(close_mask.to(dtype=torch.bool).sum().detach().cpu())
            close_positive_count = int(
                (
                    close_labels.to(device=close_mask.device, dtype=torch.bool)
                    & close_mask.to(dtype=torch.bool)
                )
                .sum()
                .detach()
                .cpu()
            )
        else:
            ln_close_loss = disabled_zero
            close_open_count = 0
            close_positive_count = 0

        if self.config.lambda_adapter_reg > 0.0:
            input_mask = _input_loss_mask(batch, steps=logits_final.shape[1], device=logits_final.device)
            adapter_reg_loss = adapter_bias_regularization(
                getattr(output, "state_prior_bias", None),
                getattr(output, "ln_close_event_bias", None),
                getattr(output, "ln_close_time_shift_bias", None),
                mask=input_mask,
            )
        else:
            adapter_reg_loss = disabled_zero

        if self.config.lambda_density > 0.0:
            density_target = batch.get("density_target_8s")
            density_confidence = batch.get("density_confidence_8s")
            if density_target is None or density_confidence is None:
                raise ValueError("density_target_8s and density_confidence_8s are required when lambda_density > 0")
            if not isinstance(density_target, torch.Tensor) or not isinstance(density_confidence, torch.Tensor):
                raise ValueError("density_target_8s and density_confidence_8s must be tensors")
            density_loss = density_auxiliary_loss(
                logits_final=logits_final,
                current_ms=_require_fragment_state_tensor(batch, "current_ms").to(device=logits_final.device),
                write_start_ms=_require_batch_tensor(batch, "write_start_ms").to(device=logits_final.device),
                target=density_target.to(device=logits_final.device),
                confidence=density_confidence.to(device=logits_final.device),
                vocab=self.vocab,
                target_mask=target_mask,
                calibration_scale=self.config.density_calibration_scale,
                calibration_bias=self.config.density_calibration_bias,
            )
            density_weight = _density_loss_weight(
                confidence=density_confidence.to(device=density_loss.device),
            )
        else:
            density_loss = disabled_zero
            density_weight = logits_final.new_zeros(())
        total_loss = (
            token_loss
            + float(self.config.lambda_ln_close) * ln_close_loss
            + float(self.config.lambda_density) * density_loss
            + float(self.config.lambda_adapter_reg) * adapter_reg_loss
        )

        metrics: dict[str, float] = {}
        numerators: dict[str, float] = {}
        denominators: dict[str, float] = {}
        _record_scalar(metrics, "loss/total", total_loss)
        _record_scalar(metrics, "loss/token", token_loss)
        _record_scalar(metrics, "loss/ln_close", ln_close_loss)
        _record_scalar(metrics, "loss/density", density_loss)
        _record_scalar(metrics, "loss/adapter_reg", adapter_reg_loss)
        metrics["phase/lambda_density"] = float(self.config.lambda_density)
        metrics["phase/lambda_ln_close"] = float(self.config.lambda_ln_close)
        metrics["token/valid_count"] = int(target_mask.sum().detach().cpu())
        metrics["ln_close/open_lane_count"] = close_open_count
        metrics["ln_close/positive_count"] = close_positive_count
        token_weight = target_mask.to(dtype=token_loss.dtype).sum().clamp_min(1)
        numerators["loss/token"] = float((token_loss.detach() * token_weight).cpu())
        denominators["loss/token"] = float(target_mask.sum().detach().cpu())
        numerators["loss/ln_close"] = float(
            (ln_close_loss.detach() * max(close_open_count, 1)).cpu()
        )
        denominators["loss/ln_close"] = float(close_open_count)
        numerators["loss/density"] = float((density_loss.detach() * density_weight.clamp_min(1)).cpu())
        denominators["loss/density"] = float(density_weight.detach().cpu())

        return MapperTupleLossOutput(
            total_loss=total_loss,
            token_loss=token_loss,
            ln_close_loss=ln_close_loss,
            density_loss=density_loss,
            adapter_reg_loss=adapter_reg_loss,
            metrics=metrics,
            metric_numerators=numerators,
            metric_denominators=denominators,
        )


def token_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pad_id: int | None = None,
    target_mask: torch.Tensor | None = None,
    grammar_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,T,V], got {tuple(logits.shape)}")
    if tuple(target.shape) != tuple(logits.shape[:2]):
        raise ValueError(f"target must have shape {tuple(logits.shape[:2])}, got {tuple(target.shape)}")
    if grammar_mask is not None:
        if tuple(grammar_mask.shape) != tuple(logits.shape):
            raise ValueError(f"grammar_mask must have shape {tuple(logits.shape)}, got {tuple(grammar_mask.shape)}")
        logits = logits + grammar_mask.to(device=logits.device, dtype=logits.dtype)
    target = target.to(device=logits.device, dtype=torch.long)
    if target_mask is None:
        valid = torch.ones_like(target, dtype=torch.bool) if pad_id is None else target != int(pad_id)
    else:
        if tuple(target_mask.shape) != tuple(target.shape):
            raise ValueError(f"target_mask must have shape {tuple(target.shape)}, got {tuple(target_mask.shape)}")
        valid = target_mask.to(device=logits.device, dtype=torch.bool)
        if pad_id is not None:
            valid = valid & (target != int(pad_id))
    if not bool(valid.any()):
        return logits.reshape(-1)[:0].sum() * 0.0

    flat_valid = valid.reshape(-1)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1)
    return F.cross_entropy(
        flat_logits[flat_valid],
        flat_target[flat_valid],
        reduction="mean",
    )


def ln_close_focal_bce_loss(
    *,
    close_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
    gamma: float = 1.5,
) -> torch.Tensor:
    if close_logits.ndim != 3:
        raise ValueError(f"close_logits must have shape [B,T,4], got {tuple(close_logits.shape)}")
    if tuple(labels.shape) != tuple(close_logits.shape) or tuple(mask.shape) != tuple(close_logits.shape):
        raise ValueError("labels and mask must match close_logits shape")
    _require_finite_number(pos_weight, "pos_weight")
    _require_finite_number(gamma, "gamma")
    if pos_weight <= 0.0:
        raise ValueError("pos_weight must be positive")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")

    labels_f = labels.to(device=close_logits.device, dtype=close_logits.dtype)
    mask_f = mask.to(device=close_logits.device, dtype=close_logits.dtype)
    if not bool(mask_f.to(dtype=torch.bool).any()):
        return close_logits.sum() * 0.0
    pos_weight_tensor = close_logits.new_tensor(float(pos_weight))
    bce = F.binary_cross_entropy_with_logits(
        close_logits,
        labels_f,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    if gamma > 0.0:
        prob = torch.sigmoid(close_logits)
        p_t = torch.where(labels_f > 0.5, prob, 1.0 - prob)
        bce = bce * (1.0 - p_t).clamp_min(0.0).pow(float(gamma))
    return (bce * mask_f).sum() / mask_f.sum().clamp_min(torch.finfo(mask_f.dtype).eps)


def ln_close_aux_loss(
    *,
    close_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float | torch.Tensor | None = None,
    gamma: float = 1.5,
    focal: bool = True,
) -> torch.Tensor:
    if close_logits.ndim != 3:
        raise ValueError(f"close_logits must have shape [B,T,4], got {tuple(close_logits.shape)}")
    if tuple(labels.shape) != tuple(close_logits.shape) or tuple(mask.shape) != tuple(close_logits.shape):
        raise ValueError("labels and mask must match close_logits shape")
    mask_bool = mask.to(device=close_logits.device, dtype=torch.bool)
    if not bool(mask_bool.any()):
        return close_logits.sum() * 0.0
    labels_f = labels.to(device=close_logits.device, dtype=close_logits.dtype)
    resolved_pos_weight = (
        close_pos_weight(labels=labels, mask=mask).to(device=close_logits.device, dtype=close_logits.dtype)
        if pos_weight is None
        else torch.as_tensor(pos_weight, device=close_logits.device, dtype=close_logits.dtype)
    )
    bce = F.binary_cross_entropy_with_logits(
        close_logits,
        labels_f,
        pos_weight=resolved_pos_weight,
        reduction="none",
    )
    if focal and float(gamma) > 0.0:
        prob = torch.sigmoid(close_logits)
        p_t = torch.where(labels_f > 0.5, prob, 1.0 - prob)
        bce = bce * (1.0 - p_t).clamp_min(0.0).pow(float(gamma))
    return bce[mask_bool].mean()


def close_pos_weight(
    *,
    labels: torch.Tensor,
    mask: torch.Tensor,
    min_weight: float = 1.0,
    max_weight: float = 20.0,
) -> torch.Tensor:
    if tuple(labels.shape) != tuple(mask.shape):
        raise ValueError(f"labels and mask must have the same shape, got {tuple(labels.shape)} and {tuple(mask.shape)}")
    mask_bool = mask.to(dtype=torch.bool)
    labels_bool = labels.to(dtype=torch.bool)
    positives = (labels_bool & mask_bool).sum().to(dtype=torch.float32)
    negatives = ((~labels_bool) & mask_bool).sum().to(dtype=torch.float32)
    if float(positives.item()) <= 0.0:
        return torch.tensor(float(min_weight), dtype=torch.float32, device=labels.device)
    return (negatives / positives).clamp(min=float(min_weight), max=float(max_weight))


def adapter_bias_regularization(*biases: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
    present = [bias for bias in biases if isinstance(bias, torch.Tensor)]
    if not present:
        raise ValueError("at least one adapter bias tensor is required")
    total = present[0].sum() * 0.0
    for bias in present:
        assert isinstance(bias, torch.Tensor)
        if mask is None:
            total = total + bias.square().mean()
        else:
            if tuple(mask.shape) != tuple(bias.shape[:2]):
                raise ValueError(f"mask must have shape {tuple(bias.shape[:2])}, got {tuple(mask.shape)}")
            mask_f = mask.to(device=bias.device, dtype=bias.dtype)
            while mask_f.ndim < bias.ndim:
                mask_f = mask_f.unsqueeze(-1)
            denom = mask_f.sum() * float(bias.shape[-1])
            total = total + (bias.square() * mask_f).sum() / denom.clamp_min(torch.finfo(bias.dtype).eps)
    return total


def adapter_regularization(*biases: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
    return adapter_bias_regularization(*biases, mask=mask)


def adapter_reg(*biases: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
    return adapter_bias_regularization(*biases, mask=mask)


def token_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pad_id: int | None = None,
    target_mask: torch.Tensor | None = None,
    grammar_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return token_cross_entropy(
        logits,
        target,
        pad_id=pad_id,
        target_mask=target_mask,
        grammar_mask=grammar_mask,
    )


def density_auxiliary_loss(
    *,
    logits_final: torch.Tensor,
    current_ms: torch.Tensor,
    write_start_ms: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    vocab: MapperTupleVocab,
    target_mask: torch.Tensor | None = None,
    calibration_scale: float = 1.0,
    calibration_bias: float = 0.0,
) -> torch.Tensor:
    prediction = expected_density_from_logits(
        logits_final=logits_final,
        current_ms=current_ms,
        write_start_ms=write_start_ms,
        vocab=vocab,
        target_mask=target_mask,
        calibration_scale=calibration_scale,
        calibration_bias=calibration_bias,
    )
    if tuple(target.shape) != tuple(prediction.shape):
        raise ValueError(f"density_target_8s must have shape {tuple(prediction.shape)}, got {tuple(target.shape)}")
    if tuple(confidence.shape) != tuple(prediction.shape):
        raise ValueError(f"density_confidence_8s must have shape {tuple(prediction.shape)}, got {tuple(confidence.shape)}")
    weight = confidence.to(device=prediction.device, dtype=prediction.dtype).clamp_min(0.0)
    if not bool((weight > 0).any()):
        return logits_final.reshape(-1)[:0].sum() * 0.0
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * weight
    return loss.sum() / weight.sum().clamp_min(torch.finfo(weight.dtype).eps)


def expected_density_from_logits(
    *,
    logits_final: torch.Tensor,
    current_ms: torch.Tensor,
    write_start_ms: torch.Tensor,
    vocab: MapperTupleVocab,
    target_mask: torch.Tensor | None = None,
    calibration_scale: float = 1.0,
    calibration_bias: float = 0.0,
    frame_count: int = MAPPER_DENSITY_FRAMES,
    frame_ms: int = MAPPER_DENSITY_FRAME_MS,
) -> torch.Tensor:
    if logits_final.ndim != 3:
        raise ValueError(f"logits_final must have shape [B,T,V], got {tuple(logits_final.shape)}")
    if tuple(current_ms.shape) != tuple(logits_final.shape[:2]):
        raise ValueError(f"current_ms must have shape {tuple(logits_final.shape[:2])}, got {tuple(current_ms.shape)}")
    if write_start_ms.ndim != 1 or int(write_start_ms.shape[0]) != int(logits_final.shape[0]):
        raise ValueError(f"write_start_ms must have shape [{logits_final.shape[0]}]")
    if int(logits_final.shape[-1]) != vocab.size:
        raise ValueError(f"logits_final vocab dim must be {vocab.size}, got {logits_final.shape[-1]}")
    if target_mask is None:
        valid = torch.ones_like(current_ms, dtype=torch.bool)
    else:
        if tuple(target_mask.shape) != tuple(current_ms.shape):
            raise ValueError(f"target_mask must have shape {tuple(current_ms.shape)}, got {tuple(target_mask.shape)}")
        valid = target_mask.to(device=logits_final.device, dtype=torch.bool)

    density_logits = logits_final.masked_fill(~valid.unsqueeze(-1), 0.0)
    probs = torch.softmax(density_logits, dim=-1)
    onset_weights = logits_final.new_zeros((vocab.size,))
    if vocab.event_token_ids:
        event_ids = torch.tensor(vocab.event_token_ids, dtype=torch.long, device=logits_final.device)
        weights = [float(vocab.event_onset_weight(token_id)) for token_id in vocab.event_token_ids]
        onset_weights[event_ids] = torch.tensor(weights, dtype=logits_final.dtype, device=logits_final.device)
    expected_onset_mass = (probs * onset_weights.reshape(1, 1, -1)).sum(dim=-1)
    expected_onset_mass = expected_onset_mass * valid.to(dtype=expected_onset_mass.dtype)

    batch_size, steps = current_ms.shape
    write_start = write_start_ms.to(device=logits_final.device, dtype=torch.long).reshape(batch_size, 1)
    relative_ms = current_ms.to(device=logits_final.device, dtype=torch.long) - write_start
    frame_index = torch.div(relative_ms, int(frame_ms), rounding_mode="floor")
    in_window = (relative_ms >= 0) & (frame_index >= 0) & (frame_index < int(frame_count)) & valid
    safe_index = frame_index.clamp(0, int(frame_count) - 1)

    density = logits_final.new_zeros((batch_size, int(frame_count), 1))
    density.scatter_add_(
        dim=1,
        index=safe_index.reshape(batch_size, steps, 1),
        src=(expected_onset_mass * in_window.to(dtype=expected_onset_mass.dtype)).reshape(batch_size, steps, 1),
    )
    return float(calibration_scale) * density + float(calibration_bias)


def _target_loss_mask(batch: Mapping[str, torch.Tensor], *, target: torch.Tensor, pad_id: int) -> torch.Tensor:
    raw_mask = batch.get("target_fragment_mask")
    if raw_mask is None:
        return target != int(pad_id)
    if not isinstance(raw_mask, torch.Tensor) or raw_mask.ndim != 2:
        raise ValueError("target_fragment_mask must have shape [B,T]")
    mask = raw_mask.to(device=target.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(target.shape):
        raise ValueError(f"target_fragment_mask must have shape {tuple(target.shape)}")
    return mask & (target != int(pad_id))


def _input_loss_mask(batch: Mapping[str, torch.Tensor], *, steps: int, device: torch.device) -> torch.Tensor:
    raw_mask = batch.get("target_fragment_mask")
    if raw_mask is None:
        return torch.ones((int(_require_batch_tensor(batch, "target_fragment_tokens").shape[0]), steps), dtype=torch.bool, device=device)
    if not isinstance(raw_mask, torch.Tensor) or raw_mask.ndim != 2:
        raise ValueError("target_fragment_mask must have shape [B,T]")
    mask = raw_mask.to(device=device, dtype=torch.bool)
    if int(mask.shape[1]) != int(steps):
        raise ValueError(f"target_fragment_mask must have {steps} steps")
    return mask


def _require_tensor_attr(output: Any, name: str) -> torch.Tensor:
    value = getattr(output, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"output.{name} must be a torch.Tensor")
    return value


def _reject_old_mapper_contract(batch: Mapping[str, Any]) -> None:
    old_keys = {
        "target_tokens",
        "target_token_mask",
        "teacher_current_ms",
        "teacher_open_mask",
        "teacher_open_age_ms",
    }
    present = sorted(key for key in old_keys if key in batch)
    if present:
        raise ValueError(
            "old target_tokens/teacher_* mapper contract is not supported; "
            f"received {present}"
        )


def _require_batch_tensor(batch: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch[{name!r}] must be a torch.Tensor")
    return value


def _require_fragment_state_tensor(batch: Mapping[str, Any], name: str) -> torch.Tensor:
    states = batch.get("target_fragment_states")
    if not isinstance(states, Mapping):
        raise ValueError("batch['target_fragment_states'] must be a mapping")
    value = states.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"target_fragment_states[{name!r}] must be a torch.Tensor")
    return value


def _density_loss_weight(*, confidence: torch.Tensor) -> torch.Tensor:
    return confidence.detach().to(dtype=torch.float32).clamp_min(0.0).sum()


def _record_scalar(metrics: dict[str, float], key: str, value: torch.Tensor) -> None:
    metrics[key] = float(value.detach().cpu())


def _validate_config(config: MapperTupleLossConfig) -> None:
    for name in (
        "lambda_density",
        "lambda_ln_close",
        "lambda_adapter_reg",
        "ln_close_pos_weight",
        "ln_close_focal_gamma",
        "density_calibration_scale",
        "density_calibration_bias",
    ):
        _require_finite_number(getattr(config, name), name)
    if config.lambda_density < 0.0:
        raise ValueError("lambda_density must be non-negative")
    if config.lambda_ln_close < 0.0:
        raise ValueError("lambda_ln_close must be non-negative")
    if config.lambda_adapter_reg < 0.0:
        raise ValueError("lambda_adapter_reg must be non-negative")
    if config.ln_close_pos_weight <= 0.0:
        raise ValueError("ln_close_pos_weight must be positive")
    if config.ln_close_focal_gamma < 0.0:
        raise ValueError("ln_close_focal_gamma must be non-negative")
    if config.density_calibration_scale < 0.0:
        raise ValueError("density_calibration_scale must be non-negative")


def _require_finite_number(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
