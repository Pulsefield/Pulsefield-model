from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from pulsefield_model.timing.evaluation.exp013_pilot import (
    _PilotRunMetadata,
    _stable_json,
    run_exp013_pilot,
)


EXP014_PILOT_RESULT_SCHEMA = "pulsefield_model.timing_v3_exp014_pilot_result_v1"
EXP014_PILOT_SUMMARY_SCHEMA = "pulsefield_model.timing_v3_exp014_pilot_summary_v1"
EXP014_FROZEN_INFERENCE_SCHEMA = (
    "pulsefield_model.timing_v3_exp014_frozen_inference_v1"
)
EXP014_INFERENCE_FAMILY = "timing_v3_exp014_boundary_complete_jump_candidates_v1"
EXP014_WEAK_ORACLE_POLICY = "post_frozen_inference_representative_redline_v1"

EXP014_RUN_METADATA = _PilotRunMetadata(
    result_schema=EXP014_PILOT_RESULT_SCHEMA,
    summary_schema=EXP014_PILOT_SUMMARY_SCHEMA,
    frozen_inference_schema=EXP014_FROZEN_INFERENCE_SCHEMA,
    inference_family=EXP014_INFERENCE_FAMILY,
    weak_oracle_policy=EXP014_WEAK_ORACLE_POLICY,
)


def run_exp014_pilot(**kwargs: Any) -> dict[str, Any]:
    """Run the Exp014 exposed pilot adapter with Exp014 provenance schemas."""

    if "run_metadata" in kwargs:
        raise TypeError("run_exp014_pilot owns run_metadata")
    return run_exp013_pilot(**kwargs, run_metadata=EXP014_RUN_METADATA)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the narrow Timing v3 Exp014 exposed pilot adapter. "
            "No full-corpus default is provided."
        )
    )
    parser.add_argument("--pilot-jsonl", required=True, help="Exposed pilot JSONL input")
    parser.add_argument(
        "--baseline-v2-jsonl",
        default=None,
        help="Optional v2 baseline JSONL, parsed only after frozen inference per row.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Atomic per-row JSONL output")
    parser.add_argument("--summary-json", required=True, help="Atomic summary JSON output")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cache-audio-key",
        action="append",
        dest="cache_audio_keys",
        default=None,
        help="Optional explicit cache_audio_key filter; may be repeated.",
    )
    args = parser.parse_args(argv)
    summary = run_exp014_pilot(
        pilot_jsonl_path=args.pilot_jsonl,
        baseline_v2_jsonl_path=args.baseline_v2_jsonl,
        output_jsonl_path=args.output_jsonl,
        summary_json_path=args.summary_json,
        repo_root=args.repo_root,
        limit=args.limit,
        explicit_cache_audio_keys=args.cache_audio_keys,
    )
    print(_stable_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
