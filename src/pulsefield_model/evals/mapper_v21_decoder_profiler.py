from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, record_function

from pulsefield_model.inference.mapper_v2_1_rollout import (
    generate_full_song_rollout_v2_1,
    rollout_to_timepoints_v2_1,
    zero_control_batch_provider_v2_1,
)
from pulsefield_model.models.mapper.v2_1 import (
    LaneAction,
    MapperTimepoint,
    MapperV21Config,
    MapperV21Model,
    MapperV21Vocab,
    TokenizedMapperWindow,
    encode_mapper_window,
    ln_carry_state_tensors,
)
from pulsefield_model.models.mapper.v2_1.grammar import build_grammar_mask
from pulsefield_model.models.mapper.v2_1.replay import MapperReplayState
from pulsefield_model.models.mapper.v2_1.tokenizer import MAPPER_DENSITY_FRAMES


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_PROFILER_TOP_EVENTS = 24


@dataclass(frozen=True)
class ProfileRunConfig:
    repeat: int = 1
    warmup: int = 0
    use_profiler: bool = True
    profiler_top_events: int = DEFAULT_PROFILER_TOP_EVENTS

    def __post_init__(self) -> None:
        if int(self.repeat) <= 0:
            raise ValueError("repeat must be positive")
        if int(self.warmup) < 0:
            raise ValueError("warmup must be non-negative")
        if int(self.profiler_top_events) <= 0:
            raise ValueError("profiler_top_events must be positive")


@dataclass(frozen=True)
class ProfiledCallResult:
    output: Any
    wall_ms: tuple[float, ...]
    profiler_events: tuple[dict[str, Any], ...]

    def wall_summary(self) -> dict[str, float | int]:
        return _wall_time_summary(self.wall_ms)


def tiny_mapper_v21_config(**overrides: Any) -> MapperV21Config:
    values: dict[str, Any] = {
        "control_dim": 16,
        "d_model": 16,
        "heads": 4,
        "layers": 1,
        "ffn_dim": 32,
        "dropout": 0.0,
        "max_seq_len": 80,
        "state_prior_hidden_dim": 16,
        "ln_close_hidden_dim": 16,
        "lane_embedding_dim": 4,
        "age_embedding_dim": 4,
        "use_global_context": False,
        "global_conv_blocks": 0,
    }
    values.update(overrides)
    return MapperV21Config(**values)


def synthetic_probe_window(
    *,
    vocab: MapperV21Vocab,
    event_count: int = 20,
    step_ms: int = 100,
    chart_end_padding_ms: int = 400,
) -> TokenizedMapperWindow:
    if int(event_count) <= 0:
        raise ValueError("event_count must be positive")
    if int(step_ms) <= 0 or int(step_ms) % 10 != 0:
        raise ValueError("step_ms must be positive and 10ms-aligned")
    timepoints = []
    for index in range(int(event_count)):
        first_lane = index % 4
        second_lane = (index + 2) % 4
        actions = [LaneAction.NONE, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE]
        actions[first_lane] = LaneAction.TAP
        actions[second_lane] = LaneAction.TAP
        timepoints.append(MapperTimepoint((index + 1) * int(step_ms), tuple(actions)))  # type: ignore[arg-type]
    chart_end_ms = int(event_count) * int(step_ms) + int(chart_end_padding_ms)
    return encode_mapper_window(
        timepoints,
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=8_000,
        chart_end_ms=chart_end_ms,
    )


def batch_for_tokenized_window(
    tokenized: TokenizedMapperWindow,
    *,
    config: MapperV21Config,
    prefix_len: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    seq_len = int(tokenized.seq_len if prefix_len is None else prefix_len)
    if seq_len <= 0:
        raise ValueError("prefix_len must be positive")
    if seq_len > int(tokenized.seq_len):
        raise ValueError(f"prefix_len {seq_len} exceeds tokenized sequence length {tokenized.seq_len}")

    target_tokens = tokenized.target_fragment_tensor()[:seq_len].unsqueeze(0).to(resolved_device)
    decoder_tokens = tokenized.decoder_input_tensor()[:seq_len].unsqueeze(0).to(resolved_device)
    target_mask = torch.ones((1, seq_len), dtype=torch.bool, device=resolved_device)
    states = {
        "current_ms": tokenized.target_fragment_current_ms[:seq_len].unsqueeze(0).to(resolved_device),
        "open_mask": tokenized.target_fragment_open_mask[:seq_len].unsqueeze(0).to(resolved_device),
        "open_start_ms": tokenized.target_fragment_open_start_ms[:seq_len].unsqueeze(0).to(resolved_device),
        "open_age_ms": tokenized.target_fragment_open_age_ms[:seq_len].unsqueeze(0).to(resolved_device),
        "emitted_lane_mask": tokenized.target_fragment_emitted_lane_mask[:seq_len].unsqueeze(0).to(resolved_device),
        "last_lane_index": tokenized.target_fragment_last_lane_index[:seq_len].unsqueeze(0).to(resolved_device),
    }
    batch: dict[str, Any] = {
        "decoder_input_tokens": decoder_tokens,
        "target_fragment_tokens": target_tokens,
        "target_fragment_mask": target_mask,
        "target_fragment_states": states,
        "ln_carry_in": _batched_carry(tokenized.ln_carry_in, device=resolved_device),
        "ln_carry_out": _batched_carry(tokenized.ln_carry_out, device=resolved_device),
        "close_labels": tokenized.close_labels[:seq_len].unsqueeze(0).to(resolved_device),
        "close_label_mask": tokenized.close_label_mask[:seq_len].unsqueeze(0).to(resolved_device),
        "write_start_ms": torch.tensor([tokenized.write_start_ms], dtype=torch.long, device=resolved_device),
        "write_end_ms": torch.tensor([tokenized.write_end_ms], dtype=torch.long, device=resolved_device),
        "chart_end_ms": torch.tensor([tokenized.chart_end_ms], dtype=torch.long, device=resolved_device),
        "is_full_chart_start": torch.tensor([tokenized.is_full_chart_start], dtype=torch.bool, device=resolved_device),
        "is_full_chart_end": torch.tensor([tokenized.is_full_chart_end], dtype=torch.bool, device=resolved_device),
        "difficulty": torch.tensor([[3.2]], dtype=torch.float32, device=resolved_device),
        "normalized_difficulty": torch.tensor([[0.1]], dtype=torch.float32, device=resolved_device),
        "projected_control_memory_8s": torch.zeros(
            (1, MAPPER_DENSITY_FRAMES, int(config.d_model)),
            dtype=torch.float32,
            device=resolved_device,
        ),
        "density_teacher_8s": torch.zeros((1, MAPPER_DENSITY_FRAMES, 1), dtype=torch.float32, device=resolved_device),
    }
    if bool(config.use_global_context):
        batch.update(
            {
                "global_memory": torch.zeros((1, 4, int(config.d_model)), dtype=torch.float32, device=resolved_device),
                "global_memory_padding_mask": torch.zeros((1, 4), dtype=torch.bool, device=resolved_device),
                "global_position_features": torch.zeros((1, 4), dtype=torch.float32, device=resolved_device),
            }
        )
    return batch


def clone_batch(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: clone_batch(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_batch(item) for item in value)
    if isinstance(value, list):
        return [clone_batch(item) for item in value]
    return value


def run_prefix_length_sweep(
    *,
    model: MapperV21Model,
    vocab: MapperV21Vocab,
    prefix_lengths: Sequence[int],
    run_config: ProfileRunConfig,
    device: torch.device | str = "cpu",
    apply_grammar_mask: bool = True,
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    _prepare_model(model, resolved_device)
    max_runnable_prefix_len = max(
        (int(prefix_len) for prefix_len in prefix_lengths if int(prefix_len) <= int(model.config.max_seq_len)),
        default=1,
    )
    tokenized = synthetic_probe_window(
        vocab=vocab,
        event_count=max(20, math.ceil(max_runnable_prefix_len / 3.0) + 8),
        step_ms=10,
    )
    rows: list[dict[str, Any]] = []
    for raw_prefix_len in prefix_lengths:
        prefix_len = int(raw_prefix_len)
        if prefix_len > int(model.config.max_seq_len):
            rows.append(
                {
                    "prefix_len": prefix_len,
                    "status": "skipped",
                    "skip_reason": f"prefix_len exceeds model max_seq_len {model.config.max_seq_len}",
                }
            )
            continue
        if prefix_len > int(tokenized.seq_len):
            rows.append(
                {
                    "prefix_len": prefix_len,
                    "status": "skipped",
                    "skip_reason": f"prefix_len exceeds probe sequence length {tokenized.seq_len}",
                }
            )
            continue
        batch = batch_for_tokenized_window(tokenized, config=model.config, prefix_len=prefix_len, device=resolved_device)
        batch["apply_grammar_mask"] = bool(apply_grammar_mask)

        def forward_once() -> Any:
            with torch.no_grad():
                return model(batch)

        profiled = profiled_call(
            forward_once,
            scope_name=f"mapper_v21.prefix_length_sweep.forward.prefix_{prefix_len}",
            config=run_config,
            device=resolved_device,
        )
        output = profiled.output
        rows.append(
            {
                "prefix_len": prefix_len,
                "seq_len": int(batch["decoder_input_tokens"].shape[1]),
                "status": "ok",
                "wall_ms": profiled.wall_summary(),
                "logits_shape": list(output.logits_final.shape),
                "finite_target_logits": _finite_target_logits(output.logits_final, batch["target_fragment_tokens"]),
                "profiler_events": list(profiled.profiler_events),
            }
        )
    return _summary(
        experiment="prefix_length_sweep",
        model=model,
        device=device,
        status="ok",
        repeat=run_config.repeat,
        profiler_enabled=run_config.use_profiler,
        apply_grammar_mask=bool(apply_grammar_mask),
        rows=rows,
    )


def run_kernel_overhead_probe(
    *,
    run_config: ProfileRunConfig,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    tensor = torch.ones((16, 16), dtype=torch.float32, device=resolved_device)

    def empty_scope() -> None:
        return None

    def tensor_kernel() -> torch.Tensor:
        return (tensor + 1.0).relu().sum()

    rows = []
    for name, fn in (
        ("empty_record_function_scope", empty_scope),
        ("tiny_tensor_kernel", tensor_kernel),
    ):
        profiled = profiled_call(
            fn,
            scope_name=f"mapper_v21.kernel_overhead_probe.{name}",
            config=run_config,
            device=resolved_device,
        )
        rows.append(
            {
                "probe": name,
                "status": "ok",
                "wall_ms": profiled.wall_summary(),
                "profiler_events": list(profiled.profiler_events),
            }
        )
    return _summary(
        experiment="kernel_overhead_probe",
        model=None,
        device=device,
        status="ok",
        repeat=run_config.repeat,
        profiler_enabled=run_config.use_profiler,
        rows=rows,
    )


def run_constraint_sampling_split(
    *,
    model: MapperV21Model,
    vocab: MapperV21Vocab,
    run_config: ProfileRunConfig,
    prefix_len: int = 16,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    _prepare_model(model, resolved_device)
    tokenized = synthetic_probe_window(vocab=vocab)
    batch = batch_for_tokenized_window(tokenized, config=model.config, prefix_len=prefix_len, device=resolved_device)

    def forward_logits() -> Any:
        with torch.no_grad():
            return model(batch)

    forward_profile = profiled_call(
        forward_logits,
        scope_name="mapper_v21.constraint_sampling_split.forward_logits",
        config=run_config,
        device=resolved_device,
    )
    output = forward_profile.output

    def constraints() -> torch.Tensor:
        states = batch["target_fragment_states"]
        return build_grammar_mask(
            current_ms=states["current_ms"],
            open_mask=states["open_mask"],
            open_start_ms=states["open_start_ms"],
            open_age_ms=states["open_age_ms"],
            emitted_lane_mask=states["emitted_lane_mask"],
            last_lane_index=states["last_lane_index"],
            write_start_ms=batch["write_start_ms"],
            write_end_ms=batch["write_end_ms"],
            chart_end_ms=batch["chart_end_ms"],
            ln_carry_in=batch["ln_carry_in"],
            ln_carry_out=batch["ln_carry_out"],
            is_full_chart_start=batch["is_full_chart_start"],
            is_full_chart_end=batch["is_full_chart_end"],
            vocab=vocab,
            positions=torch.arange(states["current_ms"].shape[1], dtype=torch.long, device=states["current_ms"].device).reshape(1, -1),
        )

    constraint_profile = profiled_call(
        constraints,
        scope_name="mapper_v21.constraint_sampling_split.constraints",
        config=run_config,
        device=resolved_device,
    )
    grammar_mask = constraint_profile.output
    valid_mask = torch.isfinite(grammar_mask[:, -1, :])

    def sampling() -> torch.Tensor:
        return sample_next_tokens(output.logits_final[:, -1, :], valid_mask=valid_mask)

    sampling_profile = profiled_call(
        sampling,
        scope_name="mapper_v21.constraint_sampling_split.sampling",
        config=run_config,
        device=resolved_device,
    )
    selected = sampling_profile.output.detach().cpu().reshape(-1).tolist()
    return _summary(
        experiment="constraint_sampling_split",
        model=model,
        device=device,
        status="ok",
        repeat=run_config.repeat,
        profiler_enabled=run_config.use_profiler,
        rows=[
            {
                "section": "forward_logits",
                "status": "ok",
                "wall_ms": forward_profile.wall_summary(),
                "profiler_events": list(forward_profile.profiler_events),
            },
            {
                "section": "constraints",
                "status": "ok",
                "valid_token_count_last_step": int(valid_mask.sum().item()),
                "wall_ms": constraint_profile.wall_summary(),
                "profiler_events": list(constraint_profile.profiler_events),
            },
            {
                "section": "sampling",
                "status": "ok",
                "selected_token_ids": [int(token_id) for token_id in selected],
                "wall_ms": sampling_profile.wall_summary(),
                "profiler_events": list(sampling_profile.profiler_events),
            },
        ],
    )


def run_no_ts_full_rollout_metrics(
    *,
    model: MapperV21Model,
    vocab: MapperV21Vocab,
    run_config: ProfileRunConfig,
    device: torch.device | str = "cpu",
    rollout_fn: Callable[[], Any] | None = None,
    chart_end_ms: int = 16_000,
    max_tokens_per_window: int | None = None,
    output_osu_path: str | Path | None = None,
    render_reamber_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    _prepare_model(model, resolved_device)
    if rollout_fn is None:
        rollout_fn = lambda: generate_full_song_rollout_v2_1(
            model=model,
            vocab=vocab,
            chart_end_ms=int(chart_end_ms),
            window_batch_provider=zero_control_batch_provider_v2_1(model=model, device=resolved_device),
            device=resolved_device,
            normalized_difficulty=0.0,
            max_tokens_per_window=max_tokens_per_window,
            temperature=0.0,
            top_p=None,
            time_shift_length_penalty_alpha=0.0,
        )

    profiled = profiled_call(
        rollout_fn,
        scope_name="mapper_v21.no_ts_full_rollout_metrics.rollout",
        config=run_config,
        device=resolved_device,
    )
    rollout = profiled.output
    tokens = _rollout_tokens(rollout)
    no_ts_tokens = [token_id for token_id in tokens if not vocab.is_time_shift_token(token_id)]
    timepoints = rollout_to_timepoints_v2_1(rollout, vocab) if hasattr(rollout, "windows") else []
    export_paths = _export_rollout_outputs_v2_1(
        rollout=rollout,
        vocab=vocab,
        timepoints=timepoints,
        output_osu_path=output_osu_path,
        render_reamber_dir=render_reamber_dir,
    )
    return _summary(
        experiment="no_ts_full_rollout_metrics",
        model=model,
        device=device,
        status="ok",
        repeat=run_config.repeat,
        profiler_enabled=run_config.use_profiler,
        rows=[
            {
                "section": "rollout",
                "status": "ok",
                "token_count": len(tokens),
                "no_ts_token_count": len(no_ts_tokens),
                "eos_count": sum(1 for token_id in tokens if token_id == vocab.eos_id),
                "lane_action_count": sum(1 for token_id in no_ts_tokens if vocab.is_lane_action_token(token_id)),
                "timepoint_count": len(timepoints),
                "window_count": _rollout_window_count(rollout),
                "completed_window_count": _rollout_completed_window_count(rollout),
                "empty_window_count": _rollout_empty_window_count(rollout, vocab),
                "completion_rate": _rollout_completion_rate(rollout),
                "longest_event_gap_ms": _longest_event_gap_ms(timepoints, chart_end_ms=int(chart_end_ms)),
                "tokens_per_window": _rollout_tokens_per_window(rollout),
                "events_per_window": _rollout_events_per_window(rollout, vocab),
                "dead_end_window_count": _rollout_dead_end_window_count(rollout),
                "max_token_hit_window_count": _rollout_max_token_hit_window_count(rollout),
                "completed": bool(getattr(rollout, "completed", False)),
                "dead_end": bool(getattr(rollout, "dead_end", False)),
                "max_tokens_exceeded": bool(getattr(rollout, "max_tokens_exceeded", False)),
                **export_paths,
                "wall_ms": profiled.wall_summary(),
                "profiler_events": list(profiled.profiler_events),
            }
        ],
    )


def run_eos_probe(
    *,
    model: MapperV21Model,
    vocab: MapperV21Vocab,
    run_config: ProfileRunConfig,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    _prepare_model(model, resolved_device)
    tokenized = synthetic_probe_window(vocab=vocab)
    batch = batch_for_tokenized_window(tokenized, config=model.config, device=resolved_device)

    def forward_once() -> Any:
        with torch.no_grad():
            return model(batch)

    profiled = profiled_call(
        forward_once,
        scope_name="mapper_v21.eos_probe.forward",
        config=run_config,
        device=resolved_device,
    )
    output = profiled.output
    eos_positions = (batch["target_fragment_tokens"][0] == int(vocab.eos_id)).nonzero(as_tuple=False).reshape(-1)
    rows: list[dict[str, Any]] = []
    for position_tensor in eos_positions:
        position = int(position_tensor.item())
        logits = output.logits_final[0, position]
        eos_logit = float(logits[vocab.eos_id].detach().cpu().item())
        rank = _descending_rank(logits, vocab.eos_id)
        rows.append(
            {
                "position": position,
                "current_ms": int(output.state_current_ms[0, position].detach().cpu().item()),
                "eos_logit": eos_logit,
                "eos_rank": rank,
                "eos_allowed_by_grammar": bool(torch.isfinite(output.grammar_mask[0, position, vocab.eos_id]).item()),
            }
        )
    return _summary(
        experiment="eos_probe",
        model=model,
        device=device,
        status="ok" if rows else "skipped",
        repeat=run_config.repeat,
        profiler_enabled=run_config.use_profiler,
        rows=rows,
        skip_reason=None if rows else "synthetic probe did not contain an EOS target",
        wall_ms=profiled.wall_summary(),
        profiler_events=list(profiled.profiler_events),
    )


def _export_rollout_outputs_v2_1(
    *,
    rollout: Any,
    vocab: MapperV21Vocab,
    timepoints: Sequence[Any],
    output_osu_path: str | Path | None,
    render_reamber_dir: str | Path | None,
) -> dict[str, Any]:
    del vocab
    result: dict[str, Any] = {}
    if output_osu_path is None:
        return result

    from pulsefield_model.inference.osu_export import OsuExportMetadata, format_osu_export

    osu_path = Path(output_osu_path)
    osu_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        osu_text = format_osu_export(
            timepoints=timepoints,
            metadata=OsuExportMetadata(
                audio_filename="audio.mp3",
                title="Mapper v2.1 No TS Penalty Rollout",
                creator="Pulsefield",
                version="Mapper v2.1 greedy no-ts-penalty eval",
            ),
        )
    except Exception as exc:
        result["osu_export_status"] = "error"
        result["osu_export_error"] = str(exc)
        return result

    osu_path.write_text(osu_text, encoding="utf-8")
    result["osu_export_status"] = "ok"
    result["osu_path"] = osu_path.as_posix()

    if render_reamber_dir is None:
        return result
    try:
        from pulsefield_model.evals.mapper_render_reamber import render_named_spans

        rendered = render_named_spans(osu_path, Path(render_reamber_dir))
    except Exception as exc:
        result["reamber_render_status"] = "error"
        result["reamber_render_error"] = str(exc)
    else:
        result["reamber_render_status"] = "ok"
        result["reamber_paths"] = {name: path.as_posix() for name, path in rendered.items()}
    return result


def _rollout_windows(rollout: Any) -> list[Any]:
    if isinstance(rollout, Mapping):
        raw = rollout.get("windows", ())
    else:
        raw = getattr(rollout, "windows", ())
    return list(raw)


def _rollout_window_count(rollout: Any) -> int:
    return len(_rollout_windows(rollout))


def _rollout_completed_window_count(rollout: Any) -> int:
    return sum(1 for window in _rollout_windows(rollout) if bool(getattr(window, "completed", False)))


def _rollout_empty_window_count(rollout: Any, vocab: MapperV21Vocab) -> int:
    count = 0
    for window in _rollout_windows(rollout):
        tokens = [int(token_id) for token_id in getattr(window, "tokens", ())]
        if not any(vocab.is_lane_action_token(token_id) for token_id in tokens):
            count += 1
    return count


def _rollout_completion_rate(rollout: Any) -> float:
    windows = _rollout_windows(rollout)
    if not windows:
        return 0.0
    return float(sum(1 for window in windows if bool(getattr(window, "completed", False))) / len(windows))


def _rollout_tokens_per_window(rollout: Any) -> list[int]:
    return [len(getattr(window, "tokens", ())) for window in _rollout_windows(rollout)]


def _rollout_events_per_window(rollout: Any, vocab: MapperV21Vocab) -> list[int]:
    return [
        sum(1 for token_id in getattr(window, "tokens", ()) if vocab.is_lane_action_token(int(token_id)))
        for window in _rollout_windows(rollout)
    ]


def _rollout_dead_end_window_count(rollout: Any) -> int:
    return sum(1 for window in _rollout_windows(rollout) if bool(getattr(window, "dead_end", False)))


def _rollout_max_token_hit_window_count(rollout: Any) -> int:
    return sum(1 for window in _rollout_windows(rollout) if bool(getattr(window, "max_tokens_exceeded", False)))


def _longest_event_gap_ms(timepoints: Sequence[Any], *, chart_end_ms: int) -> int:
    event_times = sorted({int(item.time_ms) for item in timepoints})
    anchors = [0, *event_times, int(chart_end_ms)]
    if len(anchors) < 2:
        return 0
    return max(max(0, right - left) for left, right in zip(anchors, anchors[1:]))


def profiled_call(
    fn: Callable[[], Any],
    *,
    scope_name: str,
    config: ProfileRunConfig,
    device: torch.device,
) -> ProfiledCallResult:
    if not scope_name:
        raise ValueError("scope_name must be non-empty")
    with torch.no_grad():
        for _ in range(int(config.warmup)):
            with record_function(f"{scope_name}.warmup"):
                fn()
    _synchronize_device(device)
    profiler = (
        torch.profiler.profile(
            activities=_profiler_activities(device),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            acc_events=True,
        )
        if bool(config.use_profiler)
        else nullcontext(None)
    )
    wall_ms: list[float] = []
    output: Any = None
    with profiler as active_profiler:
        for _ in range(int(config.repeat)):
            _synchronize_device(device)
            start = time.perf_counter()
            with record_function(scope_name):
                output = fn()
            _synchronize_device(device)
            wall_ms.append((time.perf_counter() - start) * 1_000.0)
    events = _profiler_events(active_profiler, limit=int(config.profiler_top_events), scope_prefix=scope_name)
    return ProfiledCallResult(output=output, wall_ms=tuple(wall_ms), profiler_events=tuple(events))


def sample_next_tokens(
    logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B,V], got {tuple(logits.shape)}")
    if tuple(valid_mask.shape) != tuple(logits.shape):
        raise ValueError(f"valid_mask must match logits shape, got {tuple(valid_mask.shape)}")
    masked = logits.masked_fill(~valid_mask.to(device=logits.device, dtype=torch.bool), -torch.inf)
    if float(temperature) <= 0.0:
        return torch.argmax(masked, dim=-1)
    probs = torch.softmax(masked / float(temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).reshape(-1)


def write_json_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.suffix == ".jsonl":
        raise ValueError("decoder eval summaries must be aggregate .json files, not JSONL traces")
    if output_path.suffix != ".json":
        raise ValueError("decoder eval summary path must use a .json suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _batched_carry(carry: Any, *, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = ln_carry_state_tensors(carry)
    return {
        "current_ms": tensors["current_ms"].reshape(1).to(device),
        "open_mask": tensors["open_mask"].reshape(1, 4).to(device),
        "open_start_ms": tensors["open_start_ms"].reshape(1, 4).to(device),
        "open_age_ms": tensors["open_age_ms"].reshape(1, 4).to(device),
    }


def _summary(
    *,
    experiment: str,
    model: MapperV21Model | None,
    device: torch.device | str,
    status: str,
    repeat: int,
    profiler_enabled: bool,
    rows: Sequence[Mapping[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": experiment,
        "status": status,
        "device": str(torch.device(device)),
        "repeat": int(repeat),
        "profiler_enabled": bool(profiler_enabled),
        "rows": list(rows),
    }
    if config is not None:
        payload["model"] = {
            "class": type(model).__name__,
            "d_model": int(config.d_model),
            "layers": int(config.layers),
            "heads": int(config.heads),
            "max_seq_len": int(config.max_seq_len),
            "use_global_context": bool(config.use_global_context),
        }
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def _profiler_activities(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


def _prepare_model(model: MapperV21Model, device: torch.device) -> None:
    model.to(device)
    model.eval()


def _profiler_events(active_profiler: Any, *, limit: int, scope_prefix: str) -> list[dict[str, Any]]:
    if active_profiler is None:
        return []
    try:
        averages = active_profiler.key_averages()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for event in averages:
        key = str(getattr(event, "key", ""))
        if key != scope_prefix and not key.startswith("mapper_v21."):
            continue
        rows.append(
            {
                "key": key,
                "count": int(getattr(event, "count", 0)),
                "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
                "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
            }
        )
    rows.sort(key=lambda item: item["cpu_time_total_us"], reverse=True)
    return rows[: int(limit)]


def _wall_time_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": math.nan, "min": math.nan, "max": math.nan}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _finite_target_logits(logits: torch.Tensor, targets: torch.Tensor) -> bool:
    positions = torch.arange(targets.shape[1], dtype=torch.long, device=targets.device)
    gathered = logits[0, positions, targets[0]]
    return bool(torch.isfinite(gathered).all().item())


def _descending_rank(logits: torch.Tensor, token_id: int) -> int:
    order = torch.argsort(logits.detach(), descending=True)
    matches = (order == int(token_id)).nonzero(as_tuple=False)
    if int(matches.numel()) == 0:
        return -1
    return int(matches[0].item()) + 1


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            return


def _v21_full_rollout_available(model: MapperV21Model) -> bool:
    for name in ("full_rollout", "generate_full_rollout", "rollout_full_chart"):
        value = getattr(model, name, None)
        if callable(value):
            return True
    try:
        value = getattr(model, "incremental_decode_next_token")
    except AttributeError:
        return False
    return callable(value)


def _rollout_tokens(rollout: Any) -> list[int]:
    if isinstance(rollout, Mapping):
        raw = rollout.get("tokens", ())
    else:
        raw = getattr(rollout, "tokens", ())
    return [int(token_id) for token_id in raw]


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, MapperReplayState):
        return {
            "position": int(value.position),
            "current_ms": int(value.current_ms),
            "open_mask": list(value.open_mask),
            "open_start_ms": list(value.open_start_ms),
            "open_age_ms": list(value.open_age_ms),
            "emitted_lane_mask": list(value.emitted_lane_mask),
            "last_lane_index": int(value.last_lane_index),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
