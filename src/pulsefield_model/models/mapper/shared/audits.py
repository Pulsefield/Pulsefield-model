from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from .density_calibration import (
    density_metrics,
    fit_monotonic_affine_calibration,
    smooth_density_mass,
)
from .generation import (
    CarryStateError,
    LNCarryState,
    RecoveryCEReport,
    carry_aware_valid_token_mask,
    carry_states_equal,
    coerce_ln_carry_state,
    reconstruct_ln_carry_states,
    transition_carry_state,
    window_is_complete,
)
from .replay import CLOSED_OPEN_START_MS
from .tokenizer import MAPPER_DENSITY_FRAMES, TokenizedMapperWindow
from .vocab import MapperTupleVocab


@dataclass(frozen=True)
class TokenizerAuditReport:
    num_windows: int
    num_eligible_windows: int
    num_dropped_cross_window_ln_windows: int
    max_seq_len: int
    mean_seq_len: float
    p95_seq_len: int
    p99_seq_len: int
    event_vocab_coverage: int
    open_mask_nonzero_before_eos_count: int
    invalid_time_delta_count: int
    noncanonical_time_shift_count: int
    invalid_event_count: int
    time_shift_vocab_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GrammarAuditReport:
    num_windows: int
    checked_token_count: int
    violation_count: int
    violations: list[dict[str, object]]
    dead_end_count: int = 0
    bos_inside_ordinary_window_count: int = 0
    eos_inside_ordinary_window_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CarryAuditReport:
    num_windows: int
    num_windows_with_carry_in: int
    num_windows_with_carry_out: int
    num_windows_with_same_lane_carry_through: int
    carry_in_open_lane_rate: float
    carry_out_open_lane_rate: float
    carry_reconstruction_checked_count: int
    carry_reconstruction_failure_count: int
    carry_reconstruction_failure_examples: list[dict[str, object]]
    terminal_state_mismatch_count: int
    terminal_state_mismatch_examples: list[dict[str, object]]
    boundary_exact_start_count: int
    boundary_exact_end_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryTokenAuditReport:
    bos_token_count: int
    eos_token_count: int
    non_initial_window_bos_count: int
    non_final_window_eos_count: int
    window_start_bos_count: int
    window_end_eos_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedRolloutAuditReport:
    num_generated_windows: int
    generated_invalid_token_count: int
    generated_dead_end_count: int
    generated_completed_window_count: int
    generated_carry_out_mismatch_count: int
    generated_bos_count: int
    generated_eos_count: int
    generated_validity_rate: float
    generated_dead_end_rate: float
    generated_carry_out_match_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryAuditReport:
    generated_prefix_match_rate_500ms: float
    generated_prefix_match_rate_1000ms: float
    generated_current_ms_drift_mae: float
    generated_open_mask_mismatch_rate: float
    generated_open_start_mismatch_rate: float
    generated_open_age_mae_when_open_mask_matches: float
    recovery_ce: float
    recovery_batch_valid_fraction: float
    rollout_token_edit_distance: float
    mismatch_reasons: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LNCloseImbalanceReport:
    num_open_lane_steps: int
    num_close_positive_steps: int
    close_positive_rate: float
    pos_weight: float
    ln_duration_distribution: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseAAuditGateDecision:
    status: str
    tokenizer_status: str
    grammar_status: str
    failure_reasons: list[str]
    open_mask_nonzero_before_eos_count: int
    invalid_time_delta_count: int
    invalid_event_count: int
    noncanonical_time_shift_count: int
    grammar_violation_count: int
    carry_status: str = "PASS"
    boundary_status: str = "PASS"
    generation_status: str = "PASS"
    carry_reconstruction_failure_count: int = 0
    terminal_state_mismatch_count: int = 0
    non_initial_window_bos_count: int = 0
    non_final_window_eos_count: int = 0
    generated_invalid_token_count: int = 0
    generated_dead_end_count: int = 0
    generated_carry_out_mismatch_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_tokenized_windows(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperTupleVocab,
    filter_report: object,
) -> TokenizerAuditReport:
    window_list = _window_records(windows)
    filter_payload = _filter_report_payload(filter_report)
    num_total_windows = _filter_report_int(filter_payload, "num_total_windows")
    num_eligible_windows = _filter_report_int(filter_payload, "num_mapper_eligible_windows")
    num_dropped_cross_window_ln_windows = _filter_report_int(filter_payload, "num_dropped_cross_window_ln_windows")
    if num_eligible_windows != len(window_list):
        raise ValueError(
            "filter_report num_mapper_eligible_windows must match audited windows: "
            f"{num_eligible_windows} != {len(window_list)}",
        )

    lengths = [_window_seq_len(window) for window in window_list]
    event_ids = {
        token_id
        for window in window_list
        for token_id in _window_raw_tokens(window)
        if vocab.is_event_token(int(token_id))
    }
    ts_counter: Counter[str] = Counter()
    tokenizer_counts = _TokenizerTokenCounts()
    for window in window_list:
        for token_id in _window_raw_tokens(window):
            if vocab.is_time_shift_token(int(token_id)):
                ts_counter[vocab.token_name(int(token_id))] += 1
        tokenizer_counts += _audit_token_sequence(window, vocab=vocab)

    return TokenizerAuditReport(
        num_windows=num_total_windows,
        num_eligible_windows=num_eligible_windows,
        num_dropped_cross_window_ln_windows=num_dropped_cross_window_ln_windows,
        max_seq_len=max(lengths, default=0),
        mean_seq_len=float(sum(lengths) / len(lengths)) if lengths else 0.0,
        p95_seq_len=_percentile_int(lengths, 0.95),
        p99_seq_len=_percentile_int(lengths, 0.99),
        event_vocab_coverage=len(event_ids),
        open_mask_nonzero_before_eos_count=tokenizer_counts.open_mask_nonzero_before_eos_count,
        invalid_time_delta_count=tokenizer_counts.invalid_time_delta_count,
        noncanonical_time_shift_count=tokenizer_counts.noncanonical_time_shift_count,
        invalid_event_count=tokenizer_counts.invalid_event_count,
        time_shift_vocab_distribution=dict(sorted(ts_counter.items())),
    )


def audit_grammar_replay(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperTupleVocab,
    max_violations: int = 20,
) -> GrammarAuditReport:
    window_list = _window_records(windows)
    violations: list[dict[str, object]] = []
    checked = 0
    violation_count = 0
    dead_end_count = 0
    boundary = audit_boundary_tokens(window_list, vocab=vocab)
    for window_index, window in enumerate(window_list):
        fragment_tokens = _window_fragment_tokens(window, vocab=vocab)
        ln_carry_in = _window_ln_carry_in(window)
        ln_carry_out = _window_ln_carry_out(window)
        write_start_ms = _window_int(window, "write_start_ms")
        write_end_ms = _window_int(window, "write_end_ms")
        state = ln_carry_in
        for token_index, token_id in enumerate(fragment_tokens):
            checked += 1
            mask = carry_aware_valid_token_mask(
                state=state,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                ln_carry_out=ln_carry_out,
                vocab=vocab,
            )
            if not bool(mask[int(token_id)].item()):
                violation_count += 1
                if len(violations) < max_violations:
                    violations.append(
                        {
                            "window_index": window_index,
                            "token_index": token_index,
                            "token_id": int(token_id),
                            "token_name": vocab.token_name(int(token_id)),
                            "current_ms": state.current_ms,
                            "open_mask": state.open_mask,
                            "open_start_ms": state.open_start_ms,
                        }
                    )
            try:
                state = transition_carry_state(
                    state,
                    int(token_id),
                    vocab=vocab,
                    write_start_ms=write_start_ms,
                    write_end_ms=write_end_ms,
                )
            except CarryStateError:
                continue
        if not window_is_complete(state, write_end_ms=write_end_ms, ln_carry_out=ln_carry_out):
            next_mask = carry_aware_valid_token_mask(
                state=state,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                ln_carry_out=ln_carry_out,
                vocab=vocab,
            )
            if not bool(next_mask.any().item()):
                dead_end_count += 1
    return GrammarAuditReport(
        num_windows=len(window_list),
        checked_token_count=checked,
        violation_count=violation_count,
        violations=violations,
        dead_end_count=dead_end_count,
        bos_inside_ordinary_window_count=boundary.non_initial_window_bos_count,
        eos_inside_ordinary_window_count=boundary.non_final_window_eos_count,
    )


def audit_carry_windows(
    windows: Sequence[Any],
    *,
    vocab: MapperTupleVocab,
    max_examples: int = 20,
) -> CarryAuditReport:
    window_list = _window_records(windows)
    windows_with_carry_in = 0
    windows_with_carry_out = 0
    same_lane_carry_through = 0
    carry_in_open_lanes = 0
    carry_out_open_lanes = 0
    reconstruction_checked = 0
    reconstruction_failures = 0
    reconstruction_examples: list[dict[str, object]] = []
    terminal_mismatches = 0
    terminal_examples: list[dict[str, object]] = []
    boundary_start_count = 0
    boundary_end_count = 0

    for window_index, window in enumerate(window_list):
        ln_carry_in = _window_ln_carry_in(window)
        ln_carry_out = _window_ln_carry_out(window)
        write_start_ms = _window_int(window, "write_start_ms")
        write_end_ms = _window_int(window, "write_end_ms")
        carry_in_lanes = sum(ln_carry_in.open_mask)
        carry_out_lanes = sum(ln_carry_out.open_mask)
        carry_in_open_lanes += carry_in_lanes
        carry_out_open_lanes += carry_out_lanes
        windows_with_carry_in += int(carry_in_lanes > 0)
        windows_with_carry_out += int(carry_out_lanes > 0)
        if any(
            ln_carry_in.open_mask[lane]
            and ln_carry_out.open_mask[lane]
            and ln_carry_in.open_start_ms[lane] == ln_carry_out.open_start_ms[lane]
            for lane in range(4)
        ):
            same_lane_carry_through += 1

        timepoints = _window_timepoints(window)
        if timepoints is not None:
            boundary_start_count += sum(1 for timepoint in timepoints if int(timepoint.time_ms) == write_start_ms)
            boundary_end_count += sum(1 for timepoint in timepoints if int(timepoint.time_ms) == write_end_ms)
            reconstruction_checked += 1
            try:
                reconstructed_in, reconstructed_out = reconstruct_ln_carry_states(
                    timepoints,
                    write_start_ms=write_start_ms,
                    write_end_ms=write_end_ms,
                )
            except Exception as exc:  # noqa: BLE001 - audit records source-data failures.
                reconstruction_failures += 1
                if len(reconstruction_examples) < max_examples:
                    reconstruction_examples.append(
                        {"window_index": window_index, "reason": str(exc)},
                    )
            else:
                if not carry_states_equal(reconstructed_in, ln_carry_in) or not carry_states_equal(reconstructed_out, ln_carry_out):
                    reconstruction_failures += 1
                    if len(reconstruction_examples) < max_examples:
                        reconstruction_examples.append(
                            {
                                "window_index": window_index,
                                "expected_in": ln_carry_in.to_dict(),
                                "reconstructed_in": reconstructed_in.to_dict(),
                                "expected_out": ln_carry_out.to_dict(),
                                "reconstructed_out": reconstructed_out.to_dict(),
                            }
                        )

        terminal_state = _window_terminal_state(window)
        if terminal_state is not None:
            terminal_match = carry_states_equal(terminal_state, ln_carry_out)
        else:
            terminal_match = _window_replays_to_carry_out(
                window,
                vocab=vocab,
                ln_carry_in=ln_carry_in,
                ln_carry_out=ln_carry_out,
            )
        if not terminal_match:
            terminal_mismatches += 1
            if len(terminal_examples) < max_examples:
                terminal_examples.append(
                    {
                        "window_index": window_index,
                        "terminal_state": None if terminal_state is None else terminal_state.to_dict(),
                        "ln_carry_out": ln_carry_out.to_dict(),
                    }
                )

    denominator = max(len(window_list) * 4, 1)
    return CarryAuditReport(
        num_windows=len(window_list),
        num_windows_with_carry_in=windows_with_carry_in,
        num_windows_with_carry_out=windows_with_carry_out,
        num_windows_with_same_lane_carry_through=same_lane_carry_through,
        carry_in_open_lane_rate=float(carry_in_open_lanes / denominator),
        carry_out_open_lane_rate=float(carry_out_open_lanes / denominator),
        carry_reconstruction_checked_count=reconstruction_checked,
        carry_reconstruction_failure_count=reconstruction_failures,
        carry_reconstruction_failure_examples=reconstruction_examples,
        terminal_state_mismatch_count=terminal_mismatches,
        terminal_state_mismatch_examples=terminal_examples,
        boundary_exact_start_count=boundary_start_count,
        boundary_exact_end_count=boundary_end_count,
    )


def audit_boundary_tokens(
    windows: Sequence[Any],
    *,
    vocab: MapperTupleVocab,
) -> BoundaryTokenAuditReport:
    window_list = _window_records(windows)
    bos_count = 0
    eos_count = 0
    non_initial_bos = 0
    non_final_eos = 0
    window_start_bos = 0
    window_end_eos = 0
    for window in window_list:
        tokens = _window_raw_tokens(window)
        if not tokens:
            continue
        is_initial = _window_is_full_chart_start(window)
        is_final = _window_is_full_chart_end(window)
        for token_index, token_id in enumerate(tokens):
            if int(token_id) == vocab.bos_id:
                bos_count += 1
                if token_index == 0:
                    window_start_bos += 1
                if not is_initial or token_index != 0:
                    non_initial_bos += 1
            elif int(token_id) == vocab.eos_id:
                eos_count += 1
                if token_index == len(tokens) - 1:
                    window_end_eos += 1
                if not is_final or token_index != len(tokens) - 1:
                    non_final_eos += 1
    return BoundaryTokenAuditReport(
        bos_token_count=bos_count,
        eos_token_count=eos_count,
        non_initial_window_bos_count=non_initial_bos,
        non_final_window_eos_count=non_final_eos,
        window_start_bos_count=window_start_bos,
        window_end_eos_count=window_end_eos,
    )


def audit_generated_rollouts(
    generated_windows: Sequence[Any] | None,
    *,
    vocab: MapperTupleVocab,
) -> GeneratedRolloutAuditReport:
    if not generated_windows:
        return GeneratedRolloutAuditReport(
            num_generated_windows=0,
            generated_invalid_token_count=0,
            generated_dead_end_count=0,
            generated_completed_window_count=0,
            generated_carry_out_mismatch_count=0,
            generated_bos_count=0,
            generated_eos_count=0,
            generated_validity_rate=math.nan,
            generated_dead_end_rate=math.nan,
            generated_carry_out_match_rate=math.nan,
        )

    invalid_tokens = 0
    total_tokens = 0
    dead_ends = 0
    completed = 0
    carry_mismatches = 0
    bos_count = 0
    eos_count = 0
    for window in generated_windows:
        tokens = [int(token_id) for token_id in getattr(window, "tokens", [])]
        total_tokens += len(tokens)
        bos_count += sum(token_id == vocab.bos_id for token_id in tokens)
        eos_count += sum(token_id == vocab.eos_id for token_id in tokens)
        invalid_tokens += sum(token_id in {vocab.pad_id, vocab.bos_id, vocab.eos_id} for token_id in tokens)
        dead_ends += int(bool(getattr(window, "dead_end", False)))
        completed_flag = bool(getattr(window, "completed", False))
        completed += int(completed_flag)
        terminal_state = _coerce_optional_carry_state(getattr(window, "terminal_state", None))
        ln_carry_out = _coerce_optional_carry_state(getattr(window, "ln_carry_out", None))
        if terminal_state is not None and ln_carry_out is not None:
            carry_match = carry_states_equal(terminal_state, ln_carry_out)
        else:
            carry_match = completed_flag
        carry_mismatches += int(not carry_match)
    total = len(generated_windows)
    return GeneratedRolloutAuditReport(
        num_generated_windows=total,
        generated_invalid_token_count=invalid_tokens,
        generated_dead_end_count=dead_ends,
        generated_completed_window_count=completed,
        generated_carry_out_mismatch_count=carry_mismatches,
        generated_bos_count=bos_count,
        generated_eos_count=eos_count,
        generated_validity_rate=float((total_tokens - invalid_tokens) / total_tokens) if total_tokens else 1.0,
        generated_dead_end_rate=float(dead_ends / total),
        generated_carry_out_match_rate=float((total - carry_mismatches) / total),
    )


def audit_recovery_metrics(
    recovery_report: RecoveryCEReport | Mapping[str, object] | None = None,
    *,
    generated_states_500ms: Sequence[LNCarryState] | None = None,
    generated_states_1000ms: Sequence[LNCarryState] | None = None,
    gold_states: Sequence[LNCarryState] | None = None,
    rollout_token_edit_distance: float = math.nan,
) -> RecoveryAuditReport:
    payload = recovery_report.to_dict() if isinstance(recovery_report, RecoveryCEReport) else dict(recovery_report or {})
    recovery_ce = float(payload.get("recovery_ce", math.nan))
    valid_fraction = float(payload.get("recovery_batch_valid_fraction", math.nan))
    mismatch_reasons = payload.get("mismatch_reasons", {})
    if not isinstance(mismatch_reasons, Mapping):
        mismatch_reasons = {}

    return RecoveryAuditReport(
        generated_prefix_match_rate_500ms=_prefix_match_rate(generated_states_500ms, gold_states),
        generated_prefix_match_rate_1000ms=_prefix_match_rate(generated_states_1000ms, gold_states),
        generated_current_ms_drift_mae=_current_ms_drift_mae(generated_states_1000ms, gold_states),
        generated_open_mask_mismatch_rate=_open_mask_mismatch_rate(generated_states_1000ms, gold_states),
        generated_open_start_mismatch_rate=_open_start_mismatch_rate(generated_states_1000ms, gold_states),
        generated_open_age_mae_when_open_mask_matches=_open_age_mae(generated_states_1000ms, gold_states),
        recovery_ce=recovery_ce,
        recovery_batch_valid_fraction=valid_fraction,
        rollout_token_edit_distance=float(rollout_token_edit_distance),
        mismatch_reasons={str(key): int(value) for key, value in mismatch_reasons.items()},
    )


def audit_ln_close_imbalance(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperTupleVocab | None = None,
) -> LNCloseImbalanceReport:
    resolved_vocab = MapperTupleVocab() if vocab is None else vocab
    window_list = _window_records(windows)
    open_steps = 0
    positive_steps = 0
    durations_ms: list[int] = []
    for window in window_list:
        mask = torch.as_tensor(_window_value(window, "close_label_mask"), dtype=torch.bool)
        labels = torch.as_tensor(_window_value(window, "close_labels"), dtype=torch.bool)
        open_steps += int(mask.sum().item())
        positive_steps += int((labels & mask).sum().item())
        durations_ms.extend(_ln_durations_ms(window, vocab=resolved_vocab))
    negative_steps = max(open_steps - positive_steps, 0)
    if positive_steps <= 0:
        pos_weight = 20.0 if open_steps > 0 else 1.0
    else:
        pos_weight = min(max(negative_steps / positive_steps, 1.0), 20.0)
    return LNCloseImbalanceReport(
        num_open_lane_steps=open_steps,
        num_close_positive_steps=positive_steps,
        close_positive_rate=float(positive_steps / open_steps) if open_steps else 0.0,
        pos_weight=float(pos_weight),
        ln_duration_distribution=_duration_distribution(durations_ms),
    )


def audit_density_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float]:
    report = density_metrics(prediction, target, confidence)
    confidence_flat = confidence.detach().reshape(-1)
    report["density_target_missing_rate"] = float((confidence_flat <= 0).sum().item() / max(confidence_flat.numel(), 1))
    report["density_confidence_mean"] = float(confidence_flat.to(dtype=torch.float32).mean().item())
    report["density_confidence_min"] = float(confidence_flat.to(dtype=torch.float32).min().item())
    report["density_confidence_max"] = float(confidence_flat.to(dtype=torch.float32).max().item())
    report["density_confidence_distribution"] = {
        "min": report["density_confidence_min"],
        "mean": report["density_confidence_mean"],
        "max": report["density_confidence_max"],
    }
    return report


def build_phase_a_report(
    *,
    windows: Sequence[TokenizedMapperWindow],
    vocab: MapperTupleVocab,
    filter_report: object,
    density_prediction: torch.Tensor | None = None,
    density_target: torch.Tensor | None = None,
    density_confidence: torch.Tensor | None = None,
    calibration: object | None = None,
    generated_windows: Sequence[Any] | None = None,
    recovery_report: RecoveryCEReport | Mapping[str, object] | None = None,
) -> dict[str, object]:
    filter_payload = _filter_report_payload(filter_report)
    tokenizer = audit_tokenized_windows(windows, vocab=vocab, filter_report=filter_payload).to_dict()
    carry = audit_carry_windows(windows, vocab=vocab).to_dict()
    boundary = audit_boundary_tokens(windows, vocab=vocab).to_dict()
    grammar = audit_grammar_replay(windows, vocab=vocab).to_dict()
    generation = audit_generated_rollouts(generated_windows, vocab=vocab).to_dict()
    recovery = audit_recovery_metrics(recovery_report).to_dict()
    ln_close = audit_ln_close_imbalance(windows, vocab=vocab).to_dict()
    density: dict[str, object] = {
        "gold_mass_to_density_mae": math.nan,
        "gold_mass_to_density_corr": math.nan,
        "density_target_missing_rate": math.nan,
        "density_confidence_distribution": {},
    }
    confidence_for_density = _density_confidence_or_ones(density_target, density_confidence)
    gold_mass: torch.Tensor | None = None
    resolved_calibration = calibration
    if density_target is not None:
        gold_mass = _gold_onset_mass_batch(windows, vocab=vocab)
        _validate_density_frame_count(
            gold_mass,
            density_target=density_target,
            density_confidence=confidence_for_density,
        )
        if resolved_calibration is None:
            resolved_calibration = fit_monotonic_affine_calibration(
                smooth_density_mass(gold_mass),
                density_target,
                confidence_for_density,
            )

    if density_prediction is not None and density_target is not None:
        density_metrics_report = audit_density_prediction(density_prediction, density_target, confidence_for_density)
        density.update(density_metrics_report)
    if gold_mass is not None and density_target is not None and hasattr(resolved_calibration, "predict"):
        gold_prediction = resolved_calibration.predict(gold_mass)
        gold_metrics = density_metrics(gold_prediction, density_target, confidence_for_density)
        if density_prediction is None:
            density.update(audit_density_prediction(gold_prediction, density_target, confidence_for_density))
        density["gold_mass_to_density_mae"] = gold_metrics["density_frame_mae"]
        density["gold_mass_to_density_corr"] = gold_metrics["density_pearson_corr"]

    calibration_payload = resolved_calibration.to_dict() if hasattr(resolved_calibration, "to_dict") else {}
    gate_decision = build_phase_a_gate_decision(
        tokenizer=tokenizer,
        grammar=grammar,
        carry=carry,
        boundary=boundary,
        generation=generation,
    ).to_dict()
    report = {
        "window_filter": filter_payload,
        "carry": carry,
        "boundary_tokens": boundary,
        "tokenizer": tokenizer,
        "grammar": grammar,
        "generation": generation,
        "recovery": recovery,
        "density": density,
        "ln_close": ln_close,
        "density_calibration": calibration_payload,
        "gate_decision": gate_decision,
    }
    return report


def _gold_onset_mass_batch(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperTupleVocab,
) -> torch.Tensor:
    window_list = _window_records(windows)
    if not window_list:
        return torch.zeros((0, MAPPER_DENSITY_FRAMES), dtype=torch.float32)
    return torch.stack([_scatter_window_gold_onset_mass(window, vocab=vocab) for window in window_list])


def _scatter_window_gold_onset_mass(
    window: Any,
    *,
    vocab: MapperTupleVocab,
) -> torch.Tensor:
    mass = torch.zeros(MAPPER_DENSITY_FRAMES, dtype=torch.float32)
    tokens = _window_raw_tokens(window)
    current_ms = _window_current_ms_tensor(window)
    state = _window_ln_carry_in(window)
    write_start_ms = _window_int(window, "write_start_ms")
    write_end_ms = _window_int(window, "write_end_ms")
    for index, token_id in enumerate(tokens):
        if vocab.is_event_token(int(token_id)):
            event_ms = int(current_ms[index].item()) if index < int(current_ms.numel()) else state.current_ms
            frame_index = (event_ms - write_start_ms) // 20
            if 0 <= frame_index < MAPPER_DENSITY_FRAMES:
                mass[frame_index] += float(vocab.event_onset_weight(int(token_id)))
        try:
            state = transition_carry_state(
                state,
                int(token_id),
                vocab=vocab,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                allow_bos=_window_is_full_chart_start(window),
                allow_eos=_window_is_full_chart_end(window),
            )
        except CarryStateError:
            continue
    return mass


def _density_confidence_or_ones(
    density_target: torch.Tensor | None,
    density_confidence: torch.Tensor | None,
) -> torch.Tensor | None:
    if density_target is None:
        return None
    if density_confidence is not None:
        return density_confidence
    return torch.ones_like(density_target, dtype=torch.float32)


def _validate_density_frame_count(
    gold_mass: torch.Tensor,
    *,
    density_target: torch.Tensor,
    density_confidence: torch.Tensor | None,
) -> None:
    expected = int(gold_mass.numel())
    target_count = int(density_target.detach().numel())
    if target_count != expected:
        raise ValueError(f"density_target must contain {expected} gold density frames, got {target_count}")
    if density_confidence is None:
        return
    confidence_count = int(density_confidence.detach().numel())
    if confidence_count != expected:
        raise ValueError(f"density_confidence must contain {expected} gold density frames, got {confidence_count}")


def _percentile_int(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, math.ceil(percentile * len(sorted_values)) - 1))
    return int(sorted_values[index])


def build_phase_a_gate_decision(
    *,
    tokenizer: TokenizerAuditReport | Mapping[str, object],
    grammar: GrammarAuditReport | Mapping[str, object],
    carry: CarryAuditReport | Mapping[str, object] | None = None,
    boundary: BoundaryTokenAuditReport | Mapping[str, object] | None = None,
    generation: GeneratedRolloutAuditReport | Mapping[str, object] | None = None,
) -> PhaseAAuditGateDecision:
    tokenizer_payload = tokenizer.to_dict() if isinstance(tokenizer, TokenizerAuditReport) else tokenizer
    grammar_payload = grammar.to_dict() if isinstance(grammar, GrammarAuditReport) else grammar
    carry_payload = {} if carry is None else carry.to_dict() if isinstance(carry, CarryAuditReport) else carry
    boundary_payload = (
        {} if boundary is None else boundary.to_dict() if isinstance(boundary, BoundaryTokenAuditReport) else boundary
    )
    generation_payload = (
        {}
        if generation is None
        else generation.to_dict()
        if isinstance(generation, GeneratedRolloutAuditReport)
        else generation
    )
    open_mask_count = _report_int(tokenizer_payload, "open_mask_nonzero_before_eos_count")
    invalid_time_delta_count = _report_int(tokenizer_payload, "invalid_time_delta_count")
    invalid_event_count = _report_int(tokenizer_payload, "invalid_event_count")
    noncanonical_time_shift_count = _report_int(tokenizer_payload, "noncanonical_time_shift_count")
    grammar_violation_count = _report_int(grammar_payload, "violation_count")
    carry_reconstruction_failure_count = _optional_report_int(carry_payload, "carry_reconstruction_failure_count")
    terminal_state_mismatch_count = _optional_report_int(carry_payload, "terminal_state_mismatch_count")
    non_initial_window_bos_count = _optional_report_int(boundary_payload, "non_initial_window_bos_count")
    non_final_window_eos_count = _optional_report_int(boundary_payload, "non_final_window_eos_count")
    generated_invalid_token_count = _optional_report_int(generation_payload, "generated_invalid_token_count")
    generated_dead_end_count = _optional_report_int(generation_payload, "generated_dead_end_count")
    generated_carry_out_mismatch_count = _optional_report_int(
        generation_payload,
        "generated_carry_out_mismatch_count",
    )

    tokenizer_failures: list[str] = []
    if open_mask_count > 0:
        tokenizer_failures.append("open_mask_nonzero_before_eos_count > 0")
    if invalid_time_delta_count > 0:
        tokenizer_failures.append("invalid_time_delta_count > 0")
    if invalid_event_count > 0:
        tokenizer_failures.append("invalid_event_count > 0")
    if noncanonical_time_shift_count > 0:
        tokenizer_failures.append("noncanonical_time_shift_count > 0")

    grammar_failures = ["grammar violation_count > 0"] if grammar_violation_count > 0 else []
    carry_failures: list[str] = []
    if carry_reconstruction_failure_count > 0:
        carry_failures.append("carry_reconstruction_failure_count > 0")
    if terminal_state_mismatch_count > 0:
        carry_failures.append("terminal_state_mismatch_count > 0")
    boundary_failures: list[str] = []
    if non_initial_window_bos_count > 0:
        boundary_failures.append("non_initial_window_bos_count > 0")
    if non_final_window_eos_count > 0:
        boundary_failures.append("non_final_window_eos_count > 0")
    generation_failures: list[str] = []
    if generated_invalid_token_count > 0:
        generation_failures.append("generated_invalid_token_count > 0")
    if generated_dead_end_count > 0:
        generation_failures.append("generated_dead_end_count > 0")
    if generated_carry_out_mismatch_count > 0:
        generation_failures.append("generated_carry_out_mismatch_count > 0")
    failure_reasons = tokenizer_failures + grammar_failures + carry_failures + boundary_failures + generation_failures
    return PhaseAAuditGateDecision(
        status="PASS" if not failure_reasons else "FAIL",
        tokenizer_status="PASS" if not tokenizer_failures else "FAIL",
        grammar_status="PASS" if not grammar_failures else "FAIL",
        failure_reasons=failure_reasons,
        open_mask_nonzero_before_eos_count=open_mask_count,
        invalid_time_delta_count=invalid_time_delta_count,
        invalid_event_count=invalid_event_count,
        noncanonical_time_shift_count=noncanonical_time_shift_count,
        grammar_violation_count=grammar_violation_count,
        carry_status="PASS" if not carry_failures else "FAIL",
        boundary_status="PASS" if not boundary_failures else "FAIL",
        generation_status="PASS" if not generation_failures else "FAIL",
        carry_reconstruction_failure_count=carry_reconstruction_failure_count,
        terminal_state_mismatch_count=terminal_state_mismatch_count,
        non_initial_window_bos_count=non_initial_window_bos_count,
        non_final_window_eos_count=non_final_window_eos_count,
        generated_invalid_token_count=generated_invalid_token_count,
        generated_dead_end_count=generated_dead_end_count,
        generated_carry_out_mismatch_count=generated_carry_out_mismatch_count,
    )


@dataclass
class _TokenizerTokenCounts:
    open_mask_nonzero_before_eos_count: int = 0
    invalid_time_delta_count: int = 0
    noncanonical_time_shift_count: int = 0
    invalid_event_count: int = 0

    def __iadd__(self, other: "_TokenizerTokenCounts") -> "_TokenizerTokenCounts":
        self.open_mask_nonzero_before_eos_count += other.open_mask_nonzero_before_eos_count
        self.invalid_time_delta_count += other.invalid_time_delta_count
        self.noncanonical_time_shift_count += other.noncanonical_time_shift_count
        self.invalid_event_count += other.invalid_event_count
        return self


def _audit_token_sequence(window: TokenizedMapperWindow, *, vocab: MapperTupleVocab) -> _TokenizerTokenCounts:
    counts = _TokenizerTokenCounts()
    state = _window_ln_carry_in(window)
    ln_carry_out = _window_ln_carry_out(window)
    write_start_ms = _window_int(window, "write_start_ms")
    write_end_ms = _window_int(window, "write_end_ms")
    time_shift_run: list[int] = []

    def flush_time_shift_run() -> None:
        nonlocal time_shift_run
        if not time_shift_run:
            return
        total_delta_ms = sum(time_shift_run)
        try:
            canonical = vocab.decompose_time_shift_delta(total_delta_ms)
        except ValueError:
            counts.noncanonical_time_shift_count += 1
        else:
            if time_shift_run != canonical:
                counts.noncanonical_time_shift_count += 1
        time_shift_run = []

    for position, raw_token_id in enumerate(_window_raw_tokens(window)):
        token_id = int(raw_token_id)
        is_time_shift = vocab.is_time_shift_token(token_id)
        if is_time_shift:
            time_shift_run.append(vocab.time_shift_value(token_id))
        else:
            flush_time_shift_run()

        mask = carry_aware_valid_token_mask(
            state=state,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_out=ln_carry_out,
            vocab=vocab,
            is_full_chart_start=_window_is_full_chart_start(window),
            is_full_chart_end=_window_is_full_chart_end(window),
            token_position=position,
        )
        is_known_token = 0 <= token_id < vocab.size
        is_valid = is_known_token and bool(mask[token_id].item())
        if not is_valid:
            if is_time_shift:
                counts.invalid_time_delta_count += 1
            elif vocab.is_event_token(token_id) or not _is_special_token(token_id, vocab=vocab):
                counts.invalid_event_count += 1

        if token_id == vocab.eos_id and any(state.open_mask):
            counts.open_mask_nonzero_before_eos_count += 1

        try:
            state = transition_carry_state(
                state,
                token_id,
                vocab=vocab,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                allow_bos=_window_is_full_chart_start(window),
                allow_eos=_window_is_full_chart_end(window),
            )
        except CarryStateError:
            continue

    flush_time_shift_run()
    return counts


def _is_special_token(token_id: int, *, vocab: MapperTupleVocab) -> bool:
    return token_id in {vocab.pad_id, vocab.bos_id, vocab.eos_id}


def _ln_durations_ms(window: TokenizedMapperWindow, *, vocab: MapperTupleVocab) -> list[int]:
    open_start_by_lane: dict[int, int] = {}
    durations: list[int] = []
    current_ms = _window_current_ms_tensor(window)
    write_start_ms = _window_int(window, "write_start_ms")
    for index, token_id in enumerate(_window_raw_tokens(window)):
        if not vocab.is_event_token(int(token_id)):
            continue
        event_ms = int(current_ms[min(index, max(len(current_ms) - 1, 0))].item()) if len(current_ms) else write_start_ms
        for lane, action in enumerate(vocab.decode_event(int(token_id))):
            if action.value == "HOLD_START":
                open_start_by_lane[lane] = event_ms
            elif action.value == "HOLD_END" and lane in open_start_by_lane:
                durations.append(event_ms - open_start_by_lane.pop(lane))
    return durations


def _duration_distribution(durations_ms: Sequence[int]) -> dict[str, float]:
    if not durations_ms:
        return {"count": 0.0, "mean_ms": math.nan, "p50_ms": math.nan, "p95_ms": math.nan}
    values = sorted(int(value) for value in durations_ms)
    return {
        "count": float(len(values)),
        "mean_ms": float(sum(values) / len(values)),
        "p50_ms": float(values[len(values) // 2]),
        "p95_ms": float(values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]),
    }


def _filter_report_payload(filter_report: object | None) -> dict[str, object]:
    if filter_report is None:
        raise ValueError(
            "filter_report is required for Phase A mapper audits; post-filtered windows cannot reconstruct "
            "cross-window LN exclusion counts or drop rates.",
        )
    if hasattr(filter_report, "to_dict"):
        return filter_report.to_dict()  # type: ignore[no-any-return]
    if hasattr(filter_report, "__dataclass_fields__"):
        return asdict(filter_report)
    if isinstance(filter_report, Mapping):
        return dict(filter_report)
    return dict(filter_report)  # type: ignore[arg-type]


def _filter_report_int(filter_payload: Mapping[str, object], key: str) -> int:
    return _report_int(filter_payload, key, report_name="filter_report")


def _report_int(report_payload: Mapping[str, object], key: str, *, report_name: str = "audit report") -> int:
    try:
        value = report_payload[key]
    except KeyError as exc:
        raise ValueError(f"{report_name} is missing required field {key!r}") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{report_name} field {key!r} must be an integer-compatible value, got {value!r}") from exc


def _optional_report_int(report_payload: Mapping[str, object], key: str) -> int:
    if key not in report_payload:
        return 0
    try:
        return int(report_payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"audit report field {key!r} must be an integer-compatible value") from exc


@dataclass(frozen=True)
class _IndexedWindowMapping:
    source: Mapping[str, Any]
    index: int


_WINDOW_TOKEN_KEYS = ("target_ids", "target_fragment_ids", "target_fragment_tokens")
_TARGET_FRAGMENT_STATE_KEYS = {
    "target_fragment_current_ms": "current_ms",
    "target_fragment_open_mask": "open_mask",
    "target_fragment_open_start_ms": "open_start_ms",
    "target_fragment_open_age_ms": "open_age_ms",
}


def _window_records(windows: Any) -> list[Any]:
    if _is_window_mapping(windows):
        return _mapping_window_records(windows)
    records: list[Any] = []
    for window in windows:
        if _is_window_mapping(window):
            records.extend(_mapping_window_records(window))
        else:
            records.append(window)
    return records


def _is_window_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and any(key in value for key in (*_WINDOW_TOKEN_KEYS, "ln_carry_in", "ln_carry_out"))


def _mapping_window_records(mapping: Mapping[str, Any]) -> list[Any]:
    batch_size = _mapping_batch_size(mapping)
    if batch_size is None:
        return [mapping]
    return [_IndexedWindowMapping(mapping, index) for index in range(batch_size)]


def _mapping_batch_size(mapping: Mapping[str, Any]) -> int | None:
    for key in ("target_fragment_tokens", "target_ids", "target_fragment_ids"):
        value = _mapping_value(mapping, key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.ndim >= 2 else None
        if _is_nested_sequence(value):
            return len(value)

    mask = _mapping_value(mapping, "target_fragment_mask")
    if isinstance(mask, torch.Tensor) and mask.ndim >= 2:
        return int(mask.shape[0])

    carry = mapping.get("ln_carry_in")
    if isinstance(carry, Mapping):
        open_mask = carry.get("open_mask")
        if isinstance(open_mask, torch.Tensor) and open_mask.ndim >= 2:
            return int(open_mask.shape[0])
        current_ms = carry.get("current_ms")
        if isinstance(current_ms, torch.Tensor) and current_ms.ndim >= 1 and int(current_ms.numel()) != 1:
            return int(current_ms.shape[0])
    return None


def _is_nested_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and isinstance(value[0], (list, tuple, Mapping, torch.Tensor))


def _window_raw_tokens(window: Any) -> list[int]:
    value = _window_value(window, "target_ids")
    apply_fragment_mask = False
    if value is None:
        value = _window_value(window, "target_fragment_ids")
        apply_fragment_mask = True
    if value is None:
        value = _window_value(window, "target_fragment_tokens")
        apply_fragment_mask = True
    if value is None:
        return []
    return _window_token_list(window, value, apply_fragment_mask=apply_fragment_mask)


def _window_fragment_tokens(window: Any, *, vocab: MapperTupleVocab) -> list[int]:
    explicit = _window_value(window, "target_fragment_tokens")
    if explicit is None:
        explicit = _window_value(window, "target_fragment_ids")
    if explicit is not None:
        return _window_token_list(window, explicit, apply_fragment_mask=True)
    tokens = _window_raw_tokens(window)
    if tokens and tokens[0] == vocab.bos_id:
        tokens = tokens[1:]
    if tokens and tokens[-1] == vocab.eos_id:
        tokens = tokens[:-1]
    return tokens


def _window_ln_carry_in(window: Any) -> LNCarryState:
    value = _window_value(window, "ln_carry_in")
    write_start_ms = _window_int(window, "write_start_ms")
    if value is not None:
        return _coerce_window_carry_state(value, current_ms=write_start_ms)
    return LNCarryState.closed(write_start_ms)


def _window_ln_carry_out(window: Any) -> LNCarryState:
    value = _window_value(window, "ln_carry_out")
    write_end_ms = _window_int(window, "write_end_ms")
    if value is not None:
        return _coerce_window_carry_state(value, current_ms=write_end_ms)
    return LNCarryState.closed(write_end_ms)


def _window_terminal_state(window: Any) -> LNCarryState | None:
    return _coerce_optional_carry_state(_window_value(window, "terminal_state"))


def _coerce_optional_carry_state(value: Any) -> LNCarryState | None:
    if value is None:
        return None
    return _coerce_window_carry_state(value, current_ms=None)


def _window_replays_to_carry_out(
    window: Any,
    *,
    vocab: MapperTupleVocab,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
) -> bool:
    state = ln_carry_in
    write_start_ms = _window_int(window, "write_start_ms")
    write_end_ms = _window_int(window, "write_end_ms")
    for token_id in _window_fragment_tokens(window, vocab=vocab):
        try:
            state = transition_carry_state(
                state,
                int(token_id),
                vocab=vocab,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
            )
        except CarryStateError:
            return False
    return window_is_complete(state, write_end_ms=write_end_ms, ln_carry_out=ln_carry_out)


def _window_timepoints(window: Any) -> Sequence[Any] | None:
    value = _window_value(window, "timepoints")
    if value is None:
        value = _window_value(window, "source_timepoints")
    if value is None:
        return None
    return list(value)


def _window_is_full_chart_start(window: Any) -> bool:
    value = _window_value(window, "is_full_chart_start")
    if value is None:
        value = _window_value(window, "full_chart_start")
    if value is not None:
        return _scalar_bool(value, field_name="is_full_chart_start")
    return _window_int(window, "write_start_ms") == 0


def _window_is_full_chart_end(window: Any) -> bool:
    value = _window_value(window, "is_full_chart_end")
    if value is None:
        value = _window_value(window, "full_chart_end")
    return _scalar_bool(value, field_name="is_full_chart_end") if value is not None else False


def _window_value(window: Any, key: str) -> Any:
    if isinstance(window, _IndexedWindowMapping):
        return _mapping_value(window.source, key, batch_index=window.index)
    if isinstance(window, Mapping):
        return _mapping_value(window, key)
    return getattr(window, key, None)


def _mapping_value(mapping: Mapping[str, Any], key: str, *, batch_index: int | None = None) -> Any:
    value = mapping.get(key)
    if value is None and key in _TARGET_FRAGMENT_STATE_KEYS:
        states = mapping.get("target_fragment_states")
        if isinstance(states, Mapping):
            value = states.get(_TARGET_FRAGMENT_STATE_KEYS[key])
    if batch_index is None or value is None:
        return value
    return _slice_batch_value(value, batch_index)


def _slice_batch_value(value: Any, batch_index: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _slice_batch_value(item, batch_index) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value if value.ndim == 0 else value[batch_index]
    if _is_nested_sequence(value):
        return value[batch_index]
    return value


def _window_seq_len(window: Any) -> int:
    value = _window_value(window, "seq_len")
    if value is not None:
        return _scalar_int(value, field_name="seq_len")
    mask = _window_value(window, "target_fragment_mask")
    if mask is not None:
        return sum(_as_bool_list(mask))
    return len(_window_raw_tokens(window))


def _window_token_list(window: Any, value: Any, *, apply_fragment_mask: bool) -> list[int]:
    tokens = _as_int_list(value)
    if not apply_fragment_mask:
        return tokens
    mask_value = _window_value(window, "target_fragment_mask")
    if mask_value is None:
        return tokens
    mask = _as_bool_list(mask_value)
    if len(mask) != len(tokens):
        raise ValueError(f"target_fragment_mask length must match target tokens: {len(mask)} != {len(tokens)}")
    return [token for token, keep in zip(tokens, mask, strict=True) if keep]


def _coerce_window_carry_state(value: Any, *, current_ms: int | None) -> LNCarryState:
    if isinstance(value, LNCarryState):
        return value
    if isinstance(value, Mapping):
        resolved_current_ms = _scalar_int(value.get("current_ms"), default=current_ms, field_name="ln_carry.current_ms")
        return LNCarryState(
            current_ms=resolved_current_ms,
            open_mask=_as_bool_tuple4(value["open_mask"], field_name="ln_carry.open_mask"),
            open_start_ms=_as_open_start_tuple4(value["open_start_ms"], field_name="ln_carry.open_start_ms"),
            open_age_ms=_as_int_tuple4(value["open_age_ms"], field_name="ln_carry.open_age_ms"),
        )
    return coerce_ln_carry_state(value, current_ms=current_ms)


def _window_int(window: Any, key: str) -> int:
    return _scalar_int(_window_value(window, key), field_name=key)


def _scalar_int(value: Any, *, default: int | None = None, field_name: str = "value") -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} is required")
        return int(default)
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        if int(flat.numel()) != 1:
            raise ValueError(f"{field_name} must contain exactly one value, got {int(flat.numel())}")
        return int(flat.item())
    return int(value)


def _scalar_bool(value: Any, *, field_name: str = "value") -> bool:
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        if int(flat.numel()) != 1:
            raise ValueError(f"{field_name} must contain exactly one value, got {int(flat.numel())}")
        return bool(flat.item())
    return bool(value)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [int(item) for item in value]


def _as_bool_list(value: Any) -> list[bool]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [bool(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [bool(item) for item in value]


def _as_bool_tuple4(value: Any, *, field_name: str) -> tuple[bool, bool, bool, bool]:
    items = _as_bool_list(value)
    if len(items) != 4:
        raise ValueError(f"{field_name} must contain 4 lanes, got {len(items)}")
    return tuple(items)  # type: ignore[return-value]


def _as_int_tuple4(value: Any, *, field_name: str) -> tuple[int, int, int, int]:
    items = _as_int_list(value)
    if len(items) != 4:
        raise ValueError(f"{field_name} must contain 4 lanes, got {len(items)}")
    return tuple(items)  # type: ignore[return-value]


def _as_open_start_tuple4(value: Any, *, field_name: str) -> tuple[int | None, int | None, int | None, int | None]:
    if isinstance(value, torch.Tensor):
        raw_items: list[Any] = value.detach().cpu().reshape(-1).tolist()
    else:
        raw_items = list(value)
    if len(raw_items) != 4:
        raise ValueError(f"{field_name} must contain 4 lanes, got {len(raw_items)}")
    items: list[int | None] = []
    for item in raw_items:
        if item is None:
            items.append(None)
            continue
        item_int = int(item)
        items.append(None if item_int == CLOSED_OPEN_START_MS else item_int)
    return tuple(items)  # type: ignore[return-value]


def _window_current_ms_tensor(window: Any) -> torch.Tensor:
    value = _window_value(window, "target_fragment_current_ms")
    if value is None:
        states = _window_value(window, "target_fragment_states")
        if isinstance(states, Mapping):
            value = states.get("current_ms")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)
    if value is None:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor([int(item) for item in value], dtype=torch.long)


def _prefix_match_rate(
    generated_states: Sequence[LNCarryState] | None,
    gold_states: Sequence[LNCarryState] | None,
) -> float:
    if not generated_states or not gold_states:
        return math.nan
    matched = 0
    for generated_state in generated_states:
        matched += int(any(carry_states_equal(generated_state, gold_state, age_tolerance_ms=10) for gold_state in gold_states))
    return float(matched / len(generated_states))


def _current_ms_drift_mae(
    generated_states: Sequence[LNCarryState] | None,
    gold_states: Sequence[LNCarryState] | None,
) -> float:
    pairs = _paired_states(generated_states, gold_states)
    if not pairs:
        return math.nan
    return float(sum(abs(g.current_ms - y.current_ms) for g, y in pairs) / len(pairs))


def _open_mask_mismatch_rate(
    generated_states: Sequence[LNCarryState] | None,
    gold_states: Sequence[LNCarryState] | None,
) -> float:
    pairs = _paired_states(generated_states, gold_states)
    if not pairs:
        return math.nan
    return float(sum(g.open_mask != y.open_mask for g, y in pairs) / len(pairs))


def _open_start_mismatch_rate(
    generated_states: Sequence[LNCarryState] | None,
    gold_states: Sequence[LNCarryState] | None,
) -> float:
    pairs = _paired_states(generated_states, gold_states)
    if not pairs:
        return math.nan
    return float(sum(g.open_start_ms != y.open_start_ms for g, y in pairs) / len(pairs))


def _open_age_mae(
    generated_states: Sequence[LNCarryState] | None,
    gold_states: Sequence[LNCarryState] | None,
) -> float:
    pairs = _paired_states(generated_states, gold_states)
    errors: list[int] = []
    for generated_state, gold_state in pairs:
        if generated_state.open_mask != gold_state.open_mask:
            continue
        for lane, is_open in enumerate(generated_state.open_mask):
            if is_open:
                errors.append(abs(generated_state.open_age_ms[lane] - gold_state.open_age_ms[lane]))
    if not errors:
        return math.nan
    return float(sum(errors) / len(errors))


def _paired_states(
    generated_states: Sequence[LNCarryState] | None,
    gold_states: Sequence[LNCarryState] | None,
) -> list[tuple[LNCarryState, LNCarryState]]:
    if not generated_states or not gold_states:
        return []
    return list(zip(generated_states, gold_states, strict=False))
