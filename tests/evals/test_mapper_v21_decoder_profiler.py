from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("mapper v2.1 decoder profiler tests require torch", allow_module_level=True)

from pulsefield_model.evals.mapper_v21_decoder_profiler import (
    ProfileRunConfig,
    run_constraint_sampling_split,
    run_eos_probe,
    run_kernel_overhead_probe,
    run_no_ts_full_rollout_metrics,
    run_prefix_length_sweep,
    write_json_summary,
)


pytestmark = pytest.mark.mapper_v21_decoder_eval


def test_prefix_length_sweep_writes_aggregate_json(
    mapper_v21_decoder_model: Any,
    mapper_v21_decoder_model_state: dict[str, Any],
    mapper_v21_decoder_eval_options: dict[str, Any],
    mapper_v21_decoder_eval_output_dir: Path,
) -> None:
    run_config = _run_config(mapper_v21_decoder_eval_options)
    summary = run_prefix_length_sweep(
        model=mapper_v21_decoder_model,
        vocab=mapper_v21_decoder_model_state["vocab"],
        prefix_lengths=mapper_v21_decoder_eval_options["prefix_lengths"],
        run_config=run_config,
        device=mapper_v21_decoder_eval_options["device"],
    )
    path = write_json_summary(summary, mapper_v21_decoder_eval_output_dir / "mapper_v21_prefix_length_sweep.json")

    payload = _read_json(path)
    assert payload["experiment"] == "prefix_length_sweep"
    assert payload["status"] == "ok"
    assert any(row["status"] == "ok" for row in payload["rows"])
    assert _has_record_function_event(payload)
    assert not list(mapper_v21_decoder_eval_output_dir.glob("*.jsonl"))


def test_kernel_overhead_probe_writes_aggregate_json(
    mapper_v21_decoder_eval_options: dict[str, Any],
    mapper_v21_decoder_eval_output_dir: Path,
) -> None:
    summary = run_kernel_overhead_probe(
        run_config=_run_config(mapper_v21_decoder_eval_options),
        device=mapper_v21_decoder_eval_options["device"],
    )
    path = write_json_summary(summary, mapper_v21_decoder_eval_output_dir / "mapper_v21_kernel_overhead_probe.json")

    payload = _read_json(path)
    assert payload["experiment"] == "kernel_overhead_probe"
    assert {row["probe"] for row in payload["rows"]} == {"empty_record_function_scope", "tiny_tensor_kernel"}
    assert _has_record_function_event(payload)


def test_constraint_sampling_split_writes_aggregate_json(
    mapper_v21_decoder_model: Any,
    mapper_v21_decoder_model_state: dict[str, Any],
    mapper_v21_decoder_eval_options: dict[str, Any],
    mapper_v21_decoder_eval_output_dir: Path,
) -> None:
    summary = run_constraint_sampling_split(
        model=mapper_v21_decoder_model,
        vocab=mapper_v21_decoder_model_state["vocab"],
        run_config=_run_config(mapper_v21_decoder_eval_options),
        device=mapper_v21_decoder_eval_options["device"],
    )
    path = write_json_summary(summary, mapper_v21_decoder_eval_output_dir / "mapper_v21_constraint_sampling_split.json")

    payload = _read_json(path)
    assert payload["experiment"] == "constraint_sampling_split"
    assert {row["section"] for row in payload["rows"]} == {"forward_logits", "constraints", "sampling"}
    assert next(row for row in payload["rows"] if row["section"] == "constraints")["valid_token_count_last_step"] > 0
    assert _has_record_function_event(payload)


def test_no_ts_full_rollout_metrics_writes_sparse_rollout_outputs(
    mapper_v21_decoder_model: Any,
    mapper_v21_decoder_model_state: dict[str, Any],
    mapper_v21_decoder_eval_options: dict[str, Any],
    mapper_v21_decoder_eval_output_dir: Path,
) -> None:
    osu_path = mapper_v21_decoder_eval_output_dir / "mapper_v21_no_ts_full_rollout.osu"
    render_dir = (
        mapper_v21_decoder_eval_output_dir / "reamber"
        if mapper_v21_decoder_eval_options["render_reamber"]
        else None
    )
    summary = run_no_ts_full_rollout_metrics(
        model=mapper_v21_decoder_model,
        vocab=mapper_v21_decoder_model_state["vocab"],
        run_config=_run_config(mapper_v21_decoder_eval_options),
        device=mapper_v21_decoder_eval_options["device"],
        chart_end_ms=mapper_v21_decoder_eval_options["rollout_ms"],
        max_tokens_per_window=mapper_v21_decoder_eval_options["rollout_max_tokens_per_window"],
        output_osu_path=osu_path,
        render_reamber_dir=render_dir,
    )
    path = write_json_summary(summary, mapper_v21_decoder_eval_output_dir / "mapper_v21_no_ts_full_rollout_metrics.json")
    payload = _read_json(path)
    assert payload["experiment"] == "no_ts_full_rollout_metrics"
    row = payload["rows"][0]
    assert row["no_ts_token_count"] <= row["token_count"]
    assert row["window_count"] >= 1
    assert "tokens_per_window" in row
    assert row["osu_export_status"] in {"ok", "error"}
    if row["osu_export_status"] == "ok":
        assert osu_path.exists()


def test_eos_probe_writes_aggregate_json(
    mapper_v21_decoder_model: Any,
    mapper_v21_decoder_model_state: dict[str, Any],
    mapper_v21_decoder_eval_options: dict[str, Any],
    mapper_v21_decoder_eval_output_dir: Path,
) -> None:
    summary = run_eos_probe(
        model=mapper_v21_decoder_model,
        vocab=mapper_v21_decoder_model_state["vocab"],
        run_config=_run_config(mapper_v21_decoder_eval_options),
        device=mapper_v21_decoder_eval_options["device"],
    )
    path = write_json_summary(summary, mapper_v21_decoder_eval_output_dir / "mapper_v21_eos_probe.json")

    payload = _read_json(path)
    assert payload["experiment"] == "eos_probe"
    assert payload["status"] == "ok"
    assert payload["rows"]
    assert payload["rows"][0]["eos_allowed_by_grammar"] is True
    assert _has_record_function_event(payload)


def _run_config(options: dict[str, Any]) -> ProfileRunConfig:
    return ProfileRunConfig(
        repeat=options["repeat"],
        warmup=options["warmup"],
        use_profiler=options["use_profiler"],
    )


def _read_json(path: Path) -> dict[str, Any]:
    assert path.suffix == ".json"
    return json.loads(path.read_text(encoding="utf-8"))


def _has_record_function_event(payload: dict[str, Any]) -> bool:
    if not payload["profiler_enabled"]:
        return True
    rows = payload.get("rows", [])
    for row in rows:
        for event in row.get("profiler_events", []):
            if str(event.get("key", "")).startswith("mapper_v21."):
                return True
    for event in payload.get("profiler_events", []):
        if str(event.get("key", "")).startswith("mapper_v21."):
            return True
    return False
