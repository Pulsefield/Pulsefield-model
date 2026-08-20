from __future__ import annotations

import ast
import datetime as _datetime
import hashlib
import importlib
import json
import math
import platform as _platform
import re
import subprocess
import sys
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


EXP007_EXPERIMENT_ID = "timing_v3_experiment_007"
EXP007_PROTOCOL_STAGE = "protocol_freeze"
EXP007_SCHEDULE_STAGE = "schedule16"
EXP007_REPAIR_STAGE = "repair80"
EXP007_SELECTOR_SEED = "timing-v3-exp005-schedule16-v1"
EXP007_SCHEDULE_ARMS = ("S30", "S60", "S90", "S64")
EXP007_EXECUTION_ORDER = ("S30", "S60", "S90", "S64")
EXP007_TIE_RANK = {"S64": 0, "S90": 1, "S60": 2, "S30": 3}
EXP007_WORKER_COUNT = 4
EXP007_WORKER_START_METHOD = "spawn"
EXP007_IMAP_CHUNKSIZE = 1
EXP007_MAXTASKSPERCHILD = None
EXP007_PER_AUDIO_ARM_TIMEOUT_S = 180.0
EXP007_SCHEDULE_FOUR_ARM_STOP_S = 1200.0
EXP007_REPAIR_STOP_S = 1800.0
EXP007_WORKER_RSS_CAP_BYTES = 4_294_967_296
EXP007_ROW_JSON_BYTE_CAP = 1_048_576
EXP007_CANDIDATE_PAYLOAD_BYTE_CAP = 67_108_864
EXP007_CANDIDATE_BUNDLE_BYTE_CAP = 69_206_016
EXP007_CANDIDATE_REFERENCE_MANIFEST_BYTE_CAP = 1_048_576
EXP007_CANDIDATE_GLOBAL_MANIFEST_BYTE_CAP = 1_048_576
EXP007_PARENT_POLL_MAX_SECONDS = 0.25
EXP007_FINISH_RESULT_DELIVERY_S = 5.0
EXP007_WORKER_TERMINATE_GRACE_S = 5.0
EXP007_WORKER_KILL_GRACE_S = 5.0
EXP007_CANDIDATE_METHOD_ID = "exp006_pair_conditioned_change_floor_1_4"
EXP007_BASELINE_METHOD_ID = "current_v2_grid_fitter"
EXP007_SELECTED_METHOD_ID = "exp006_or_current_v2_fallback"
EXP007_WEAK_METHOD_ID = "weak_osu_redline_object_grid_v1"
EXP007_ELIMINATION_REASONS_ORDER = (
    "candidate_fallback_guard",
    "no_origin_or_path_guard",
    "runtime_nonfinite",
    "runtime_p90_guard",
    "row_timeout_guard",
    "rss_nonfinite",
    "rss_cap_guard",
    "seam_guard",
    "section_cap_guard",
    "row_consistency_guard",
    "overlap_common_minimum",
    "section_common_minimum",
    "overlap_e1_guard",
)

SOURCE_REF_SCHEMA = "pulsefield_model.timing_v3_exp007_source_ref_v1"
REPAIR80_INPUT_BINDING_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_repair80_input_binding_v1"
)
IDENTITY_SCHEMA = "pulsefield_model.timing_v3_exp007_identity_v1"
RUN_CONFIG_SCHEMA = "pulsefield_model.timing_v3_exp007_run_config_v1"
ROW_RESULT_SCHEMA = "pulsefield_model.timing_v3_exp007_row_result_v1"
ARM_FAILURE_SCHEMA = "pulsefield_model.timing_v3_exp007_arm_failure_v1"
ARM_STAGE_SUCCESS_SCHEMA = "pulsefield_model.timing_v3_exp007_arm_stage_success_v1"
ARM_STAGE_HARD_FAILURE_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_arm_stage_hard_failure_v1"
)
NOT_RUN_ARM_SCHEMA = "pulsefield_model.timing_v3_exp007_not_run_arm_v1"
CONFIG_SELECTION_SCHEMA = "pulsefield_model.timing_v3_exp007_config_selection_v1"
FOUR_ARM_STAGE_SUMMARY_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_four_arm_stage_summary_v1"
)
SOURCE_ARM_STAGE_SUMMARY_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_source_arm_summary_v1"
)
CACHE_IDENTITY_SCHEMA = "pulsefield_model.timing_v3_exp007_cache_identity_v1"
RESTRICTED_PREDICTION_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_restricted_prediction_v1"
)
CANDIDATE_PAYLOAD_SCHEMA = "pulsefield_model.timing_v3_exp007_candidate_payload_v1"
TIMING_V3_GRID_SCHEMA = "pulsefield_model.timing_v3_grid_v1"
V2_GRID_SCHEMA = "pulsefield_model.timing_fitted_grid_v1"
SOURCE_CLOSURE_SCHEMA = "pulsefield_model.timing_v3_exp007_source_closure_v1"
CANDIDATE_GLOBAL_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_candidate_global_manifest_v1"
)
CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_candidate_reference_row_bundle_v1"
)
CANDIDATE_REFERENCE_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_candidate_reference_manifest_v1"
)
SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_schedule_weak_veto_summary_v1"
)
SCHEDULE_WEAK_SUCCESS_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_schedule_weak_success_v1"
)
SCHEDULE_WEAK_HARD_FAILURE_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_schedule_weak_hard_failure_v1"
)
SCHEDULE_WEAK_FAILURE_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_schedule_weak_failure_v1"
)
WEAK_ROW_SCHEMA = "pulsefield_model.timing_v3_exp007_weak_evidence_row_v1"
REPAIR80_SUMMARY_SCHEMA = "pulsefield_model.timing_v3_exp007_repair80_summary_v1"
SELECTOR_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_schedule16_selector_manifest_v1"
)
SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_repair80_identity_rows_v1"
)
SOURCE_LABELS_ARTIFACT_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_source_label_rows_v1"
)

LABEL_STRATA = frozenset(
    {"stable", "jump_candidate", "dense", "ramp_candidate", "ambiguous"}
)
STAGES = frozenset({EXP007_SCHEDULE_STAGE, EXP007_REPAIR_STAGE})
METHOD_IDS = {
    "candidate": EXP007_CANDIDATE_METHOD_ID,
    "baseline": EXP007_BASELINE_METHOD_ID,
    "selected": EXP007_SELECTED_METHOD_ID,
    "weak": EXP007_WEAK_METHOD_ID,
}
SCHEDULE_ARM_SET = frozenset(EXP007_SCHEDULE_ARMS)
FAILURE_KINDS = frozenset(
    {
        "row_timeout",
        "row_hard_failure",
        "worker_death",
        "pool_replacement",
        "broken_stream",
        "missing_envelope",
        "duplicate_envelope",
        "arm_deadline",
        "schedule_deadline",
        "identity_mismatch",
        "source_mismatch",
        "cache_mismatch",
        "config_mismatch",
        "restricted_input_mismatch",
        "candidate_mismatch",
        "current_v2_mismatch",
        "weak_input_failure",
        "weak_comparator_failure",
        "weak_metrics_failure",
        "weak_schema_failure",
        "weak_publication_failure",
        "schema_failure",
        "rss_failure",
        "diagnostics_integrity_failure",
        "artifact_resource_cap",
        "atomic_publication_failure",
        "summary_publication_failure",
    }
)
FAILURE_STAGES = frozenset(
    {
        "preflight",
        "pool_start",
        "row_source_check",
        "cache_load",
        "restricted_prediction",
        "candidate",
        "current_v2",
        "local_frontier",
        "diagnostics",
        "row_serialization",
        "row_publication",
        "pool_stream",
        "pool_join",
        "weak_input",
        "weak_comparator",
        "weak_metrics",
        "weak_schema",
        "weak_publication",
        "arm_summary",
        "repair_summary",
        "schedule_deadline",
    }
)
SCHEDULE_WEAK_FAILURE_KINDS = frozenset(
    {
        "weak_input_failure",
        "comparator_failure",
        "metrics_failure",
        "schema_failure",
        "publication_failure",
    }
)
SCHEDULE_WEAK_FAILURE_STAGES = frozenset(
    {"weak_input", "comparator", "metrics", "schema", "publication"}
)
CANDIDATE_FALLBACK_REASONS = frozenset(
    {
        "no_origin_candidate",
        "no_local_frontier_path",
        "local_frontier_resource_cap_exceeded",
    }
)
NO_ORIGIN_OR_PATH_REASONS = frozenset(
    {"no_origin_candidate", "no_local_frontier_path"}
)
BASELINE_UNAVAILABLE_REASONS = frozenset(
    {"prediction_too_short", "beat_signal_flat"}
)
PLATFORM_RULES = frozenset({"macos_bytes", "linux_kib_times_1024"})
GRID_KINDS = frozenset({"timing_v3", "current_v2"})
WEAK_DECISIONS = frozenset({"pass", "ambiguous", "negative"})
WEAK_ACTIONS = frozenset(
    {"authorize_repair80", "stop_ambiguous", "stop_negative"}
)
SOURCE_CLOSURE_ENTRY_MODULES = (
    "pulsefield_model.timing.evaluation.exp007_protocol",
    "pulsefield_model.timing.evaluation.exp007_selector",
    "pulsefield_model.timing.evaluation.exp007_runner",
    "pulsefield_model.timing.evaluation.exp007_metrics",
    "pulsefield_model.timing.evaluation.exp007_weak_evidence",
    "pulsefield_model.timing.evaluation.exp007_artifacts",
    "pulsefield_model.timing.v3.local_frontier",
    "pulsefield_model.timing.v3.global_constant_jump",
    "pulsefield_model.timing.v3.schema",
    "pulsefield_model.timing.providers.beatthis_cache",
    "pulsefield_model.timing.grid_fitting",
)
SOURCE_CLOSURE_REQUIRED_NON_IMPORT_FILES = (
    "docs/research/timing_v3_experiment_007_real_cache_schedule_repair.md",
)
SELECTOR_CLASSES = ("long", "dense", "jump", "stable")
SELECTOR_BUCKETS = SELECTOR_CLASSES + ("deficit_fill",)
SELECTION_SUBSTAGES = frozenset(
    {
        "long_quota",
        "dense_quota",
        "jump_quota",
        "stable_quota",
        "long_deficit_from_dense",
        "long_deficit_from_jump",
        "long_deficit_from_stable",
        "dense_deficit_from_jump",
        "dense_deficit_from_stable",
        "jump_deficit_from_stable",
        "deficit_remaining",
    }
)

SOURCE_REF_FIELDS = frozenset(
    {"schema", "artifact_schema", "sha256", "row_count", "ordered_rows_sha256"}
)
IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "row_index",
        "source_row_index",
        "cache_audio_key",
        "audio_group_key",
        "label_stratum",
        "source_long_track",
        "duration_ms",
        "label_source_sha256",
        "identity_payload_sha256",
    }
)
RUN_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "method_ids",
        "candidate_policy",
        "pool_policy",
        "limits",
        "selector_manifest_sha256",
        "input_manifest_sha256",
        "schedule_weak_veto_outcome_sha256",
        "source_closure_fingerprint_sha256",
        "cache_config_sha256",
        "grid_fitter_config_sha256",
        "local_frontier_config",
        "weak_config_sha256",
        "run_config_fingerprint_sha256",
    }
)
LOCAL_FRONTIER_CONFIG_FIELDS = frozenset(
    {
        "schedule_arm",
        "exported_frontier_width",
        "local_beam_width",
        "max_boundary_candidates_per_block",
        "max_tempo_candidates_per_block",
        "max_blocks",
        "max_sections",
        "max_section_score_misses_per_block",
        "max_section_score_misses_per_audio",
    }
)
COMPLETED_ROW_REF_FIELDS = frozenset(
    {
        "row_index",
        "cache_audio_key",
        "identity_payload_sha256",
        "row_payload_sha256",
        "candidate_reference_entry_payload_sha256",
    }
)
PENDING_IDENTITY_REF_FIELDS = frozenset(
    {"row_index", "cache_audio_key", "identity_payload_sha256"}
)
ARM_FAILURE_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "run_config_fingerprint_sha256",
        "source_closure_fingerprint_sha256",
        "input_manifest_sha256",
        "expected_row_count",
        "failure_kind",
        "failure_stage",
        "causing_row_index",
        "causing_cache_audio_key",
        "causing_worker_slot",
        "causing_worker_generation_nonce",
        "causing_worker_pid",
        "causing_dispatch_token",
        "causing_worker_rss_bytes",
        "completed_prefix_count",
        "completed_prefix_rows",
        "completed_prefix_rows_sha256",
        "completed_reference_entry_count",
        "completed_reference_entry_payload_sha256s",
        "completed_reference_entry_payloads_sha256",
        "pending_identity_count",
        "pending_identities",
        "pending_identities_sha256",
        "prefix_candidate_fallback_count",
        "prefix_no_origin_or_path_count",
        "prefix_resource_cap_fallback_count",
        "worker_rss_snapshot",
        "failure_deterministic_fingerprint_sha256",
        "full_payload_sha256",
    }
)
WORKER_RSS_FAILURE_SNAPSHOT_FIELDS = frozenset(
    {"worker_slot_lifetime_bytes", "observed_worker_max_bytes"}
)
ARM_STAGE_SUCCESS_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "status",
        "expected_row_count",
        "row_count",
        "row_payloads_sha256",
        "candidate_reference_manifest_sha256",
        "stage_summary_sha256",
        "outcome_fingerprint_sha256",
    }
)
ARM_STAGE_HARD_FAILURE_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "status",
        "arm_failure_record",
        "arm_failure_record_sha256",
        "outcome_fingerprint_sha256",
    }
)
NOT_RUN_ARM_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "status",
        "reason",
        "causing_arm",
        "causing_outcome_sha256",
        "expected_row_count",
        "pending_identity_count",
        "pending_identities",
        "pending_identities_sha256",
        "record_fingerprint_sha256",
    }
)
ARM_OUTCOME_SHA_MAP_FIELDS = frozenset(EXP007_EXECUTION_ORDER)
CONFIG_SELECTION_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "arm_outcome_sha256_by_execution_order",
        "candidate_global_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "selector_manifest_sha256",
        "overlap_common",
        "section_common",
        "source_decision",
        "arm_order_values",
        "selected_schedule_arm",
        "selected_run_config_fingerprint_sha256",
        "source_winner_selected_before_weak",
        "selection_fingerprint_sha256",
    }
)
ARM_ORDER_VALUES_FIELDS = frozenset(
    {
        "schedule_arm",
        "e0_eligible",
        "e1_eligible",
        "elimination_reasons",
        "candidate_fallback_count",
        "no_origin_or_path_count",
        "p90_overlap_ms",
        "section_inflation_violation_count",
        "p90_section_excess",
        "p90_runtime",
        "max_worker_rss",
        "tie_rank",
        "order_tuple_sha256",
    }
)
FOUR_ARM_FAILURE_DETAILS_FIELDS = frozenset(
    {
        "failure_kind",
        "first_failure_arm",
        "causing_outcome_sha256",
        "mismatch_cache_audio_key",
        "mismatch_field",
        "completed_success_arm_count",
        "deterministic_failure_sha256",
        "full_failure_sha256",
    }
)
FOUR_ARM_STAGE_SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "status",
        "arm_outcome_sha256_by_execution_order",
        "candidate_global_manifest_sha256",
        "failure_details",
        "source_selection_status",
        "config_selection_sha256",
        "summary_fingerprint_sha256",
    }
)
REPAIR80_INPUT_BINDING_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "identity_source",
        "label_source",
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_selection_sha256",
        "schedule_weak_veto_outcome_sha256",
        "row_count",
        "binding_fingerprint_sha256",
    }
)
CACHE_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "relative_cache_path",
        "exists",
        "size_bytes",
        "mtime_ns",
        "inode",
        "device",
        "sha256",
        "cache_config_sha256",
        "audio_cache_key_sha256",
    }
)
RESTRICTED_PREDICTION_FIELDS = frozenset(
    {
        "schema",
        "frame_count",
        "frame_rate_hz",
        "beat_dtype",
        "downbeat_dtype",
        "input_signal_sha256",
        "beat_bytes_sha256",
        "downbeat_bytes_sha256",
        "source_path_is_none",
        "arrays_read_only",
        "shares_loaded_memory",
    }
)
TIMING_V3_SECTION_FIELDS = frozenset({"start_beat", "end_beat", "bpm"})
TIMING_V3_GRID_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "origin_beat",
        "origin_time_ms",
        "coverage_start_ms",
        "coverage_end_ms",
        "sections",
    }
)
V2_SEGMENT_FIELDS = frozenset({"offset_ms", "beat_length_ms", "meter"})
V2_GRID_PAYLOAD_FIELDS = frozenset({"schema", "segments"})
GRID_ENVELOPE_FIELDS = frozenset(
    {"kind", "payload", "grid_sha256", "deterministic_projection_sha256"}
)
GRID_SUMMARY_FIELDS = frozenset(
    {
        "grid_kind",
        "section_count",
        "jump_count",
        "coverage_start_ms",
        "coverage_end_ms",
        "maximum_seam_discontinuity_ms",
    }
)
METHOD_RESULT_FIELDS = frozenset(
    {
        "method_id",
        "method_kind",
        "status",
        "reason",
        "fallback_kind",
        "grid",
        "grid_summary",
        "deterministic_projection_sha256",
    }
)
METHODS_FIELDS = frozenset({"candidate", "baseline", "selected"})
RESUME_BINDING_FIELDS = frozenset(
    {
        "row_input_fingerprint_sha256",
        "reused",
        "prior_row_payload_sha256",
        "validated_source_closure_fingerprint_sha256",
        "validated_config_sha256",
        "validated_cache_sha256",
        "validated_selector_sha256",
    }
)
DENOMINATOR_FLAGS_FIELDS = frozenset(
    {
        "cache_valid",
        "projection_evaluable",
        "candidate_accepted",
        "candidate_tagged_fallback",
        "baseline_accepted",
        "product_grid_available",
        "overlap_available",
        "current_v2_phase_matched",
        "pure_exp006_phase_matched",
        "selected_safety_phase_matched",
    }
)
OVERLAP_SUMMARY_FIELDS = frozenset(
    {
        "metric_version",
        "record_count",
        "available_record_count",
        "unavailable_record_count",
        "comparable_beat_count",
        "p90_ms",
        "p90_beats",
        "residual_vector_sha256",
        "records_sha256",
    }
)
BOUNDED_DIAGNOSTICS_SUMMARY_FIELDS = frozenset(
    {
        "local_frontier_contract_version",
        "bounded_contract_version",
        "objective_variant",
        "schedule_arm",
        "result_reason",
        "selected_section_count",
        "block_count",
        "candidate_fingerprint",
        "grid_fingerprint",
        "replay_fingerprint",
        "transition_cache_size",
        "actual_scored_edge_count",
        "selected_terminal_objective",
        "runner_up_terminal_objective",
        "selected_runner_up_margin",
        "block_resource_records_sha256",
        "class_coverage_records_sha256",
        "overlap",
        "deterministic_fingerprint",
    }
)
RUNTIME_TELEMETRY_FIELDS = frozenset(
    {
        "platform_rule",
        "worker_pid",
        "audio_arm_seconds",
        "cache_load_seconds",
        "candidate_seconds",
        "current_v2_seconds",
        "exp006_seconds",
        "serialization_seconds",
    }
)
RSS_TELEMETRY_FIELDS = frozenset(
    {
        "platform_rule",
        "worker_pid",
        "initial_ru_maxrss_bytes",
        "final_ru_maxrss_bytes",
    }
)
HARD_GUARDS_FIELDS = frozenset(
    {
        "timed_out",
        "worker_alive",
        "cache_unchanged",
        "source_unchanged",
        "resume_valid",
        "schema_valid",
        "row_within_byte_cap",
        "rss_within_cap",
        "grid_seam_zero",
        "section_cap_valid",
        "diagnostics_caps_valid",
    }
)
ROW_RESULT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "row_index",
        "cache_audio_key",
        "audio_group_key",
        "identity_payload_sha256",
        "cache_identity",
        "source_closure_fingerprint_sha256",
        "run_config_fingerprint_sha256",
        "selector_manifest_sha256",
        "input_manifest_sha256",
        "resume",
        "restricted_prediction",
        "candidate_payload_schema",
        "candidate_payload_byte_count",
        "candidate_payload_field_set_sha256",
        "candidate_payload_sha256",
        "candidate_fingerprint",
        "methods",
        "denominator_flags",
        "diagnostics_summary",
        "diagnostics_summary_sha256",
        "deterministic_projection_sha256",
        "runtime",
        "rss",
        "hard_guards",
        "row_payload_sha256",
    }
)
RELATIVE_SOURCE_FILE_FIELDS = frozenset({"relative_path", "sha256"})
IMPORT_EDGE_FIELDS = frozenset(
    {"importer_relative_path", "imported_module", "resolved_relative_path"}
)
MODULE_IDENTITY_FIELDS = frozenset({"module_name", "relative_path", "sha256"})
SOURCE_BEHAVIOR_FIELDS = frozenset(
    {
        "entry_modules",
        "required_non_import_files",
        "relative_source_files",
        "relative_source_files_sha256",
        "import_edges",
        "import_graph_sha256",
        "module_identities",
        "module_identities_sha256",
        "python_behavior_version",
        "numpy_behavior_version",
        "canonical_json_contract_sha256",
    }
)
SOURCE_AUDIT_FIELDS = frozenset(
    {
        "generated_at_utc",
        "absolute_root_path",
        "git_commit",
        "dirty_files",
        "platform",
        "python_full_version",
        "numpy_full_version",
    }
)
SOURCE_CLOSURE_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "schema_descriptor_sha256",
        "behavior",
        "audit",
        "source_closure_fingerprint_sha256",
        "full_payload_sha256",
    }
)
RUNTIME_SUMMARY_FIELDS = frozenset({"row_seconds", "aggregate_wall_seconds"})
RSS_SUMMARY_FIELDS = frozenset(
    {"worker_count", "worker_lifetime_bytes", "arm_max_worker_bytes"}
)
SOURCE_ARM_DENOMINATORS_FIELDS = frozenset(
    {
        "stage_audio_count",
        "stage_audio",
        "cache_valid_audio",
        "projection_evaluable_audio",
        "candidate_accepted_audio",
        "candidate_fallback_audio",
        "selected_product_fallback_audio",
        "baseline_accepted_audio",
        "product_grid_available_audio",
        "no_origin_or_path_audio",
        "resource_cap_fallback_audio",
        "overlap_available_audio",
    }
)
SOURCE_ARM_GATES_FIELDS = frozenset(
    {
        "candidate_fallback_rate",
        "selected_product_fallback_rate",
        "no_origin_or_path_rate",
        "runtime_seconds",
        "worker_rss_bytes",
        "candidate_seam_ms",
        "candidate_section_count",
        "row_json_bytes",
        "every_row_under_180_seconds",
        "seam_zero",
        "section_cap_valid",
        "row_byte_cap_valid",
        "replay_schema_source_cache_candidate_v2_consistent",
    }
)
SOURCE_ARM_STAGE_SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "run_config_fingerprint_sha256",
        "source_closure_fingerprint_sha256",
        "selector_manifest_sha256",
        "candidate_reference_manifest_sha256",
        "row_count",
        "row_refs",
        "row_payloads_sha256",
        "denominators",
        "gates",
        "runtime_summary",
        "rss_summary",
        "summary_fingerprint_sha256",
    }
)
ARM_ROW_SHA_MAP_FIELDS = frozenset(EXP007_EXECUTION_ORDER)
CANDIDATE_GLOBAL_ENTRY_FIELDS = frozenset(
    {
        "row_index",
        "cache_audio_key",
        "audio_group_key",
        "input_signal_sha256",
        "candidate_payload_schema",
        "candidate_payload_field_set_sha256",
        "candidate_payload_byte_count",
        "candidate_payload_sha256",
        "candidate_fingerprint",
        "candidate_reference_entry_payload_sha256",
        "arm_row_payload_sha256",
    }
)
CANDIDATE_GLOBAL_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "selector_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "candidate_reference_manifest_sha256",
        "row_count",
        "entries",
        "ordered_entries_sha256",
        "manifest_fingerprint_sha256",
    }
)
WEAK_ROW_REF_FIELDS = frozenset(
    {
        "row_index",
        "cache_audio_key",
        "prediction_row_sha256",
        "weak_row_payload_sha256",
    }
)
WEAK_PENDING_ROW_REF_FIELDS = frozenset(
    {"row_index", "cache_audio_key", "prediction_row_sha256"}
)
SCHEDULE_WEAK_DENOMINATORS_FIELDS = frozenset(
    {
        "stage_audio_count",
        "stage_audio",
        "comparator_available_audio",
        "comparator_unavailable_audio",
        "comparator_conflicting_audio",
        "current_v2_phase_matched",
        "pure_exp006_phase_matched",
        "selected_safety_phase_matched",
        "phase_common",
        "alias_drift_common",
        "weak_change_boundary_audio",
    }
)
SCHEDULE_WEAK_GATES_FIELDS = frozenset(
    {
        "pure_mean_phase_ratio",
        "pure_p90_phase_ratio",
        "pure_phase_coverage",
        "current_v2_phase_mean_ms",
        "pure_exp006_phase_mean_ms",
        "current_v2_phase_p90_ms",
        "pure_exp006_phase_p90_ms",
        "current_v2_alias_drift_mean_ms",
        "pure_exp006_alias_drift_mean_ms",
        "current_v2_alias_drift_p90_ms",
        "pure_exp006_alias_drift_p90_ms",
        "alias_max_prefix_drift_mean_ratio",
        "alias_max_prefix_drift_p90_ratio",
        "current_v2_boundary_f1_mean",
        "pure_exp006_boundary_f1_mean",
        "selected_boundary_f1_mean",
        "pure_minus_v2_boundary_f1_delta",
    }
)
SCHEDULE_WEAK_VETO_SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "source_selection_sha256",
        "selected_row_refs_sha256",
        "row_weak_pairs_sha256",
        "weak_row_count",
        "weak_row_refs",
        "weak_payloads_sha256",
        "denominators",
        "gates",
        "decision",
        "action",
        "summary_fingerprint_sha256",
    }
)
SCHEDULE_WEAK_SUCCESS_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "status",
        "summary",
        "summary_payload_sha256",
        "outcome_fingerprint_sha256",
    }
)
SCHEDULE_WEAK_HARD_FAILURE_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "status",
        "failure",
        "failure_payload_sha256",
        "outcome_fingerprint_sha256",
    }
)
SCHEDULE_WEAK_FAILURE_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_selection_sha256",
        "source_closure_fingerprint_sha256",
        "expected_row_count",
        "failure_kind",
        "failure_stage",
        "causing_row_index",
        "causing_cache_audio_key",
        "completed_prefix_count",
        "completed_prefix",
        "completed_prefix_sha256",
        "pending_count",
        "pending",
        "pending_sha256",
        "failure_deterministic_fingerprint_sha256",
        "full_payload_sha256",
    }
)
COMPARATOR_AVAILABILITY_FIELDS = frozenset(
    {
        "state",
        "valid_difficulty_count",
        "invalid_difficulty_count",
        "reason",
        "comparator_payloads_sha256",
    }
)
PHASE_SUMMARY_FIELDS = frozenset(
    {"current_v2_ms", "product_ms", "pure_exp006_ms"}
)
DRIFT_SUMMARY_FIELDS = frozenset(
    {
        "current_v2_alias_max_prefix_ms",
        "product_alias_max_prefix_ms",
        "pure_exp006_alias_max_prefix_ms",
    }
)
BOUNDARY_SUMMARY_FIELDS = frozenset(
    {
        "eligible",
        "valid_difficulty_count",
        "tp",
        "fp",
        "fn",
        "f1",
        "matched_error_ms",
        "weak_consensus_supported_count",
    }
)
OBJECT_GRID_SUMMARY_FIELDS = frozenset(
    {
        "eligible",
        "object_count",
        "start_residual_ms",
        "end_residual_ms",
        "inlier_count",
        "inlier_rate",
    }
)
WEAK_ROW_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "row_index",
        "cache_audio_key",
        "audio_group_key",
        "prediction_row_sha256",
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_selection_sha256",
        "comparator_availability",
        "current_v2_phase_matched",
        "pure_exp006_phase_matched",
        "selected_safety_phase_matched",
        "phase_metrics_summary",
        "drift_metrics_summary",
        "current_v2_boundary_summary",
        "pure_exp006_boundary_summary",
        "selected_boundary_summary",
        "object_grid_summary",
        "deterministic_projection_sha256",
        "weak_row_payload_sha256",
    }
)
REPAIR80_DENOMINATORS_FIELDS = frozenset(
    {
        "stage_audio_count",
        "stage_audio",
        "cache_valid_audio",
        "projection_evaluable_audio",
        "candidate_accepted_audio",
        "candidate_fallback_audio",
        "selected_product_fallback_audio",
        "baseline_accepted_audio",
        "product_grid_available_audio",
        "no_origin_or_path_audio",
        "resource_cap_fallback_audio",
        "overlap_available_audio",
        "current_v2_phase_matched",
        "pure_exp006_phase_matched",
        "selected_safety_phase_matched",
        "phase_common",
        "stable_pure_paired",
        "jump_pure_paired",
        "long_pure_paired",
        "jump_alias_drift_common",
        "long_alias_drift_common",
        "repair_boundary_common",
    }
)
REPAIR80_GATES_FIELDS = frozenset(
    {
        "candidate_fallback_rate",
        "selected_product_fallback_rate",
        "no_origin_or_path_rate",
        "runtime_seconds",
        "worker_rss_bytes",
        "overlap_ms",
        "stable_section_excess",
        "pure_mean_phase_ratio",
        "pure_p90_phase_ratio",
        "pure_phase_coverage",
        "current_v2_phase_mean_ms",
        "pure_exp006_phase_mean_ms",
        "current_v2_phase_p90_ms",
        "pure_exp006_phase_p90_ms",
        "stable_phase_mean_ratio",
        "stable_phase_p90_ratio",
        "jump_phase_mean_ratio",
        "current_v2_jump_alias_drift_mean_ms",
        "pure_exp006_jump_alias_drift_mean_ms",
        "jump_alias_drift_mean_ratio",
        "current_v2_long_alias_drift_mean_ms",
        "pure_exp006_long_alias_drift_mean_ms",
        "current_v2_long_alias_drift_p90_ms",
        "pure_exp006_long_alias_drift_p90_ms",
        "long_alias_drift_mean_ratio",
        "long_alias_drift_p90_ratio",
        "current_v2_boundary_f1_mean",
        "pure_exp006_boundary_f1_mean",
        "selected_boundary_f1_mean",
        "pure_minus_v2_boundary_f1_delta",
        "every_row_under_180_seconds",
        "seam_zero",
        "section_cap_valid",
        "replay_schema_source_cache_integrity",
    }
)
REPAIR80_SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_selection_sha256",
        "schedule_weak_veto_outcome_sha256",
        "run_config_fingerprint_sha256",
        "source_closure_fingerprint_sha256",
        "repair_input_binding_sha256",
        "repair_identity_source",
        "repair_label_source",
        "candidate_reference_manifest_sha256",
        "row_count",
        "row_refs",
        "row_payloads_sha256",
        "weak_row_count",
        "weak_row_refs",
        "weak_payloads_sha256",
        "row_weak_pairs_sha256",
        "denominators",
        "gates",
        "decision",
        "action",
        "runtime_summary",
        "rss_summary",
        "summary_fingerprint_sha256",
    }
)
SELECTOR_ENTRY_FIELDS = frozenset(
    {
        "row_index",
        "source_row_index",
        "cache_audio_key",
        "audio_group_key",
        "bucket",
        "selection_substage",
        "selection_rank",
        "selection_hash_sha256",
        "label_stratum",
        "source_long_track",
        "duration_ms",
        "label_source_sha256",
        "identity_payload_sha256",
    }
)
BUCKET_COUNT_FIELDS = frozenset(
    {"bucket", "requested", "available", "selected", "deficit"}
)
SELECTOR_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "seed",
        "source_repair80_identity",
        "source_labels",
        "source_repair80_row_count",
        "selected_count",
        "bucket_counts",
        "deficit_count",
        "selected_cache_audio_keys_sha256",
        "selected_ordered_cache_audio_keys_sha256",
        "selected_ordered_entries_sha256",
        "selected",
        "manifest_fingerprint_sha256",
    }
)

FORBIDDEN_SELECTOR_FIELD_EXACT = {
    "boundary_score",
    "cache_array",
    "candidate_outcome",
    "comparator",
    "drift",
    "failure",
    "grid",
    "hitobject",
    "metric",
    "metrics",
    "object_grid",
    "osu",
    "phase",
    "prediction",
    "redline",
    "result",
    "rss",
    "runtime",
    "score",
    "weak",
}
FORBIDDEN_SELECTOR_FIELD_SUBSTRINGS = (
    "boundary",
    "beatmap",
    "cache_array",
    "comparator",
    "drift",
    "hitobject",
    "object",
    "object_grid",
    "osu",
    "phase",
    "prediction",
    "redline",
    "runtime",
    "weak",
)
_SELECTOR_KEY_CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_SELECTOR_KEY_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")


def _selector_key_forms(key: str) -> tuple[str, tuple[str, ...], str]:
    camel_split = _SELECTOR_KEY_CAMEL_BOUNDARY_RE.sub("_", key)
    lowered = camel_split.lower()
    tokens = tuple(
        token
        for token in _SELECTOR_KEY_NON_ALNUM_RE.split(lowered)
        if token
    )
    return key.lower(), tokens, "".join(tokens)


_FORBIDDEN_SELECTOR_FIELD_EXACT_COMPACT = frozenset(
    _selector_key_forms(key)[2] for key in FORBIDDEN_SELECTOR_FIELD_EXACT
)
_FORBIDDEN_SELECTOR_FIELD_SUBSTRING_COMPACT = frozenset(
    _selector_key_forms(key)[2] for key in FORBIDDEN_SELECTOR_FIELD_SUBSTRINGS
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def object_complete_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(payload)


def payload_hash(payload: Mapping[str, Any], hash_field: str) -> str:
    body = dict(payload)
    body.pop(hash_field, None)
    return canonical_json_sha256(body)


def with_payload_hash(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[hash_field] = payload_hash(result, hash_field)
    return result


def validate_payload_hash(
    payload: Mapping[str, Any],
    hash_field: str,
    *,
    context: str,
) -> None:
    expected = require_sha256(payload.get(hash_field), f"{context}.{hash_field}")
    actual = payload_hash(payload, hash_field)
    if expected != actual:
        raise ValueError(f"{context}.{hash_field} mismatch")


def schema_descriptor_payload(schema_id: str) -> dict[str, Any]:
    descriptors = _schema_descriptors()
    if schema_id not in descriptors:
        raise ValueError(f"unknown Exp007 schema descriptor: {schema_id}")
    return _jsonable(descriptors[schema_id])


def schema_descriptor_sha256(schema_id: str) -> str:
    return canonical_json_sha256(schema_descriptor_payload(schema_id))


def candidate_payload_field_set_sha256() -> str:
    return canonical_json_sha256(_candidate_payload_field_set_descriptor())


def _candidate_payload_field_set_descriptor() -> dict[str, Any]:
    return {
        "schema": CANDIDATE_PAYLOAD_SCHEMA,
        "fields": [
            "beat_peaks",
            "boundary_candidates",
            "diagnostics",
            "downbeat_peaks",
            "origin_candidates",
            "schema",
            "tempo_candidates",
        ],
        "materialized_peak_fields": [
            "confidence",
            "frame_index",
            "refined_frame",
            "time_ms",
        ],
        "tempo_candidate_fields": ["bpm", "score", "source"],
        "origin_candidate_fields": ["anchor_id", "bpm", "score", "time_ms"],
        "boundary_candidate_fields": [
            "anchor_id",
            "downbeat_bonus",
            "evidence_mode",
            "left_period_ms",
            "nearest_downbeat_distance_ms",
            "ordinary_score",
            "rank_score",
            "right_period_ms",
            "source_peak_confidence",
            "source_peak_index",
            "source_peak_time_ms",
            "super_score",
            "time_ms",
        ],
        "candidate_diagnostics_fields": [
            "beat_peak_count",
            "boundary_candidate_count",
            "boundary_candidate_score_version",
            "candidate_contract_version",
            "candidate_fingerprint",
            "constants_json_sha256",
            "coverage_end_ms",
            "coverage_start_ms",
            "downbeat_peak_count",
            "frame_count",
            "frame_rate_hz",
            "input_signal_sha256",
            "max_period_frames",
            "min_period_frames",
            "origin_candidate_count",
            "pulse_correlation_version",
            "tempo_candidate_count",
        ],
    }


def _schema_descriptors() -> dict[str, Any]:
    common = {
        "descriptor_contract_version": "timing-v3-exp007-schema-descriptor-v2",
        "experiment_id": EXP007_EXPERIMENT_ID,
        "canonical_json": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": True,
            "allow_nan": False,
            "duplicate_keys": "reject",
            "nonfinite": "reject",
        },
        "scalar_rules": {
            "sha256": "lowercase [0-9a-f]{64}",
            "int": "reject bool; range checked per field",
            "number": "reject bool; finite binary64",
            "string": "nonempty unless enum/null branch says otherwise",
        },
    }

    def exact(
        schema: str,
        fields: frozenset[str] | Sequence[str],
        *,
        nested: Mapping[str, Any] | None = None,
        enums: Mapping[str, Any] | None = None,
        constants: Mapping[str, Any] | None = None,
        branches: Mapping[str, Any] | None = None,
        hash_preimages: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        descriptor: dict[str, Any] = {
            **common,
            "schema": schema,
            "type": "exact_object",
            "fields": sorted(fields),
        }
        if nested:
            descriptor["nested_descriptors"] = nested
        if enums:
            descriptor["enums"] = enums
        if constants:
            descriptor["constants"] = constants
        if branches:
            descriptor["branches"] = branches
        if hash_preimages:
            descriptor["hash_preimages"] = hash_preimages
        return descriptor

    rate_value = {
        "type": "exact_object",
        "fields": ["denominator", "numerator", "value"],
        "rule": "0 <= numerator <= denominator; denominator > 0; value=n/d",
    }
    ratio_value = {
        "type": "exact_object",
        "fields": ["denominator", "numerator", "state", "value"],
        "branches": {
            "finite": "denominator>0; value=n/d",
            "both_zero": "numerator=denominator=0; value=1.0",
            "positive_infinity": "numerator>0; denominator=0; value=null",
            "undefined": "numerator=denominator=value=null",
        },
    }
    stats_value = {
        "type": "exact_object",
        "fields": ["count", "maximum", "mean", "p50", "p90"],
        "branches": {
            "count=0": "mean,p50,p90,maximum are null",
            "count>0": "all numeric fields finite nonnegative",
        },
    }
    audio_set = {
        "type": "exact_object",
        "fields": ["count", "sorted_cache_audio_keys_sha256"],
        "hash_preimage": "canonical JSON of sorted unique cache_audio_key list",
    }
    grid_envelope = {
        "type": "exact_object",
        "fields": sorted(GRID_ENVELOPE_FIELDS),
        "branches": {
            "kind=timing_v3": "payload is TimingV3GridPayload",
            "kind=current_v2": "payload is V2GridPayload",
        },
        "nested_descriptors": {
            "TimingV3GridPayload": {
                "fields": sorted(TIMING_V3_GRID_PAYLOAD_FIELDS),
                "section_fields": sorted(TIMING_V3_SECTION_FIELDS),
            },
            "V2GridPayload": {
                "fields": sorted(V2_GRID_PAYLOAD_FIELDS),
                "segment_fields": sorted(V2_SEGMENT_FIELDS),
            },
        },
        "hash_preimages": {
            "grid_sha256": "canonical validated grid payload",
            "deterministic_projection_sha256": "canonical {kind,payload}",
        },
    }
    method_result = {
        "type": "exact_object",
        "fields": sorted(METHOD_RESULT_FIELDS),
        "nested_descriptors": {
            "GridEnvelope": grid_envelope,
            "GridSummary": {"fields": sorted(GRID_SUMMARY_FIELDS)},
        },
        "branches": {
            "candidate.accepted": "reason=fallback_kind=null; timing_v3 grid present",
            "candidate.tagged_fallback": (
                "reason=fallback_kind in candidate fallback reasons; grids null"
            ),
            "baseline.accepted": "reason=fallback_kind=null; current_v2 grid present",
            "baseline.unavailable": (
                "reason in baseline unavailable reasons; fallback_kind/grid null"
            ),
            "selected.accepted.candidate": "candidate accepted; timing_v3 grid present",
            "selected.accepted.current_v2": (
                "candidate tagged_fallback and baseline accepted; current_v2 grid present"
            ),
            "selected.unavailable": (
                "candidate tagged_fallback and baseline unavailable; fixed composite reason"
            ),
        },
        "hash_preimages": {
            "deterministic_projection_sha256": (
                "canonical method branch without this hash field"
            )
        },
    }
    descriptors: dict[str, Any] = {
        SOURCE_REF_SCHEMA: exact(SOURCE_REF_SCHEMA, SOURCE_REF_FIELDS),
        IDENTITY_SCHEMA: exact(
            IDENTITY_SCHEMA,
            IDENTITY_FIELDS,
            enums={
                "stage": sorted(STAGES),
                "label_stratum": sorted(LABEL_STRATA),
            },
            hash_preimages={
                "identity_payload_sha256": (
                    "canonical Identity with identity_payload_sha256 omitted"
                )
            },
        ),
        RUN_CONFIG_SCHEMA: exact(
            RUN_CONFIG_SCHEMA,
            RUN_CONFIG_FIELDS,
            nested={
                "MethodIds": {"fields": sorted(METHOD_IDS), "values": METHOD_IDS},
                "CandidatePolicy": _default_candidate_policy(),
                "PoolPolicy": _default_pool_policy(),
                "LimitPolicy": _default_limits(),
                "LocalFrontierConfigPayload": {
                    "fields": sorted(LOCAL_FRONTIER_CONFIG_FIELDS),
                    "per_arm_values": {
                        arm: make_local_frontier_config(arm)
                        for arm in EXP007_EXECUTION_ORDER
                    },
                },
            },
            enums={"stage": sorted(STAGES), "schedule_arm": list(EXP007_SCHEDULE_ARMS)},
            branches={
                "schedule16": (
                    "input_manifest_sha256 == selector_manifest_sha256 and "
                    "schedule_weak_veto_outcome_sha256 is null"
                ),
                "repair80_base_shape": (
                    "input_manifest_sha256 is Repair80InputBinding hash and weak "
                    "outcome SHA is non-null; execution validator checks objects"
                ),
            },
            hash_preimages={
                "run_config_fingerprint_sha256": (
                    "canonical RunConfig with run_config_fingerprint_sha256 omitted"
                )
            },
        ),
        CACHE_IDENTITY_SCHEMA: exact(
            CACHE_IDENTITY_SCHEMA,
            CACHE_IDENTITY_FIELDS,
            branches={
                "exists=true": "size,mtime,inode,device nonnegative ints; sha256 non-null",
                "exists=false": "size,mtime,inode,device,sha256 all null",
            },
        ),
        RESTRICTED_PREDICTION_SCHEMA: exact(
            RESTRICTED_PREDICTION_SCHEMA,
            RESTRICTED_PREDICTION_FIELDS,
            constants={
                "beat_dtype": "<f4",
                "downbeat_dtype": "<f4",
                "source_path_is_none": True,
                "arrays_read_only": True,
                "shares_loaded_memory": True,
            },
        ),
        CANDIDATE_PAYLOAD_SCHEMA: exact(
            CANDIDATE_PAYLOAD_SCHEMA,
            _candidate_payload_field_set_descriptor()["fields"],
            nested={
                "field_set_descriptor": _candidate_payload_field_set_descriptor(),
                "MaterializedPeak": (
                    _candidate_payload_field_set_descriptor()[
                        "materialized_peak_fields"
                    ]
                ),
                "TempoCandidate": (
                    _candidate_payload_field_set_descriptor()[
                        "tempo_candidate_fields"
                    ]
                ),
                "OriginCandidate": (
                    _candidate_payload_field_set_descriptor()[
                        "origin_candidate_fields"
                    ]
                ),
                "BoundaryCandidate": (
                    _candidate_payload_field_set_descriptor()[
                        "boundary_candidate_fields"
                    ]
                ),
                "CandidateDiagnostics": (
                    _candidate_payload_field_set_descriptor()[
                        "candidate_diagnostics_fields"
                    ]
                ),
            },
            enums={
                "TempoCandidate.source": ["autocorrelation", "peak_interval"],
                "BoundaryCandidate.evidence_mode": ["ordinary", "super"],
            },
            hash_preimages={
                "candidate_payload_sha256": "canonical CandidatePayload bytes",
                "candidate_payload_field_set_sha256": (
                    "canonical JSON of nested field_set_descriptor"
                ),
                "diagnostics.candidate_fingerprint": (
                    "existing source-owned Exp004 candidate fingerprint"
                ),
            },
        ),
        TIMING_V3_GRID_SCHEMA: exact(
            TIMING_V3_GRID_SCHEMA,
            TIMING_V3_GRID_PAYLOAD_FIELDS,
            nested={"TimingV3Section": {"fields": sorted(TIMING_V3_SECTION_FIELDS)}},
            constants={"schema": TIMING_V3_GRID_SCHEMA, "version": 1},
            branches={
                "sections": (
                    "nonempty; <=20; contiguous integer beats; 20<=bpm<=1000"
                )
            },
        ),
        V2_GRID_SCHEMA: exact(
            V2_GRID_SCHEMA,
            V2_GRID_PAYLOAD_FIELDS,
            nested={"V2Segment": {"fields": sorted(V2_SEGMENT_FIELDS)}},
            constants={"schema": V2_GRID_SCHEMA},
            branches={"segments": "nonempty; strictly increasing offsets"},
        ),
        ROW_RESULT_SCHEMA: exact(
            ROW_RESULT_SCHEMA,
            ROW_RESULT_FIELDS,
            nested={
                "CacheIdentity": {"schema": CACHE_IDENTITY_SCHEMA},
                "RestrictedPrediction": {"schema": RESTRICTED_PREDICTION_SCHEMA},
                "ResumeBinding": {"fields": sorted(RESUME_BINDING_FIELDS)},
                "MethodResult": method_result,
                "DenominatorFlags": {"fields": sorted(DENOMINATOR_FLAGS_FIELDS)},
                "BoundedDiagnosticsSummary": {
                    "fields": sorted(BOUNDED_DIAGNOSTICS_SUMMARY_FIELDS),
                    "overlap": {"fields": sorted(OVERLAP_SUMMARY_FIELDS)},
                },
                "RuntimeTelemetry": {"fields": sorted(RUNTIME_TELEMETRY_FIELDS)},
                "RssTelemetry": {"fields": sorted(RSS_TELEMETRY_FIELDS)},
                "HardGuards": {"fields": sorted(HARD_GUARDS_FIELDS)},
            },
            hash_preimages={
                "diagnostics_summary_sha256": "canonical validated diagnostics_summary",
                "deterministic_projection_sha256": "canonical source-owned row projection",
                "row_payload_sha256": "canonical RowResult with row_payload_sha256 omitted",
            },
        ),
        ARM_FAILURE_SCHEMA: exact(
            ARM_FAILURE_SCHEMA,
            ARM_FAILURE_FIELDS,
            nested={
                "CompletedRowRef": {"fields": sorted(COMPLETED_ROW_REF_FIELDS)},
                "PendingIdentityRef": {"fields": sorted(PENDING_IDENTITY_REF_FIELDS)},
                "WorkerRssFailureSnapshot": {
                    "fields": sorted(WORKER_RSS_FAILURE_SNAPSHOT_FIELDS)
                },
            },
            enums={
                "failure_kind": sorted(FAILURE_KINDS),
                "failure_stage": sorted(FAILURE_STAGES),
            },
            branches={
                "reference_arm": (
                    "schedule S30 or repair80 requires one non-null reference SHA "
                    "per completed prefix row"
                ),
                "later_schedule_arm": (
                    "completed reference list empty and row reference fields null"
                ),
            },
            hash_preimages={
                "failure_deterministic_fingerprint_sha256": (
                    "canonical payload excluding PID,RSS snapshot,and both hashes"
                ),
                "full_payload_sha256": "canonical payload excluding only itself",
            },
        ),
        ARM_STAGE_SUCCESS_SCHEMA: exact(
            ARM_STAGE_SUCCESS_SCHEMA,
            ARM_STAGE_SUCCESS_FIELDS,
            hash_preimages={
                "outcome_fingerprint_sha256": (
                    "canonical ArmStageSuccess with outcome_fingerprint_sha256 omitted"
                )
            },
        ),
        ARM_STAGE_HARD_FAILURE_SCHEMA: exact(
            ARM_STAGE_HARD_FAILURE_SCHEMA,
            ARM_STAGE_HARD_FAILURE_FIELDS,
            nested={"ArmFailureRecord": {"schema": ARM_FAILURE_SCHEMA}},
            hash_preimages={
                "arm_failure_record_sha256": "canonical complete ArmFailureRecord",
                "outcome_fingerprint_sha256": (
                    "canonical ArmStageHardFailure with outcome_fingerprint_sha256 omitted"
                ),
            },
        ),
        NOT_RUN_ARM_SCHEMA: exact(
            NOT_RUN_ARM_SCHEMA,
            NOT_RUN_ARM_FIELDS,
            nested={"PendingIdentityRef": {"fields": sorted(PENDING_IDENTITY_REF_FIELDS)}},
            hash_preimages={
                "pending_identities_sha256": "canonical ordered pending identity refs",
                "record_fingerprint_sha256": (
                    "canonical NotRunArmRecord with record_fingerprint_sha256 omitted"
                ),
            },
        ),
        CONFIG_SELECTION_SCHEMA: exact(
            CONFIG_SELECTION_SCHEMA,
            CONFIG_SELECTION_FIELDS,
            nested={
                "ArmOutcomeShaMap": {"fields": list(EXP007_EXECUTION_ORDER)},
                "AudioSetBinding": audio_set,
                "ArmOrderValues": {
                    "fields": sorted(ARM_ORDER_VALUES_FIELDS),
                    "elimination_reasons_order": list(EXP007_ELIMINATION_REASONS_ORDER),
                    "order_tuple_preimage": [
                        "candidate_fallback_count",
                        "no_origin_or_path_count",
                        "p90_overlap_ms",
                        "section_inflation_violation_count",
                        "p90_section_excess",
                        "tie_rank",
                    ],
                },
            },
            branches={
                "positive": "selected fields non-null and first E1 arm is selected",
                "ambiguous": "selected fields null; weak forbidden",
                "negative": "selected fields null; weak forbidden",
            },
            hash_preimages={
                "selection_fingerprint_sha256": (
                    "canonical ConfigSelection with selection_fingerprint_sha256 omitted"
                )
            },
        ),
        FOUR_ARM_STAGE_SUMMARY_SCHEMA: exact(
            FOUR_ARM_STAGE_SUMMARY_SCHEMA,
            FOUR_ARM_STAGE_SUMMARY_FIELDS,
            nested={
                "ArmOutcomeShaMap": {"fields": list(EXP007_EXECUTION_ORDER)},
                "FourArmFailureDetails": {
                    "fields": sorted(FOUR_ARM_FAILURE_DETAILS_FIELDS)
                },
            },
            hash_preimages={
                "summary_fingerprint_sha256": (
                    "canonical FourArmStageSummary with summary_fingerprint_sha256 omitted"
                )
            },
        ),
        REPAIR80_INPUT_BINDING_SCHEMA: exact(
            REPAIR80_INPUT_BINDING_SCHEMA,
            REPAIR80_INPUT_BINDING_FIELDS,
            nested={"SourceRef": {"schema": SOURCE_REF_SCHEMA}},
            hash_preimages={
                "binding_fingerprint_sha256": (
                    "canonical Repair80InputBinding with binding_fingerprint_sha256 omitted"
                )
            },
        ),
        SELECTOR_MANIFEST_SCHEMA: exact(
            SELECTOR_MANIFEST_SCHEMA,
            SELECTOR_MANIFEST_FIELDS,
            nested={
                "SourceRef": {"schema": SOURCE_REF_SCHEMA},
                "SelectorEntry": {"fields": sorted(SELECTOR_ENTRY_FIELDS)},
                "BucketCount": {"fields": sorted(BUCKET_COUNT_FIELDS)},
            },
            enums={
                "bucket": list(SELECTOR_BUCKETS),
                "bucket_count_bucket": list(SELECTOR_CLASSES),
                "selection_substage": sorted(SELECTION_SUBSTAGES),
                "label_stratum": sorted(LABEL_STRATA),
            },
            constants={
                "seed": EXP007_SELECTOR_SEED,
                "stage": EXP007_SCHEDULE_STAGE,
                "source_repair80_row_count": 80,
                "selected_count": 16,
                "bucket_count_order": list(SELECTOR_CLASSES),
            },
            hash_preimages={
                "selection_hash_sha256": (
                    "sha256(EXP007_SELECTOR_SEED + NUL + cache_audio_key)"
                ),
                "selected_cache_audio_keys_sha256": (
                    "canonical JSON of sorted selected cache_audio_key values"
                ),
                "selected_ordered_cache_audio_keys_sha256": (
                    "canonical JSON of selected cache_audio_key values in append order"
                ),
                "selected_ordered_entries_sha256": (
                    "canonical JSON of complete selected SelectorEntry list"
                ),
                "manifest_fingerprint_sha256": (
                    "canonical SelectorManifest with manifest_fingerprint_sha256 omitted"
                ),
            },
        ),
        SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA: exact(
            SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
            ["rows", "schema"],
            nested={"Identity": {"schema": IDENTITY_SCHEMA}},
            constants={"row_count": 80, "stage": EXP007_REPAIR_STAGE},
        ),
        SOURCE_LABELS_ARTIFACT_SCHEMA: exact(
            SOURCE_LABELS_ARTIFACT_SCHEMA,
            ["rows", "schema"],
            enums={"label_stratum": sorted(LABEL_STRATA)},
            constants={"row_count": 80},
        ),
        SOURCE_CLOSURE_SCHEMA: exact(
            SOURCE_CLOSURE_SCHEMA,
            SOURCE_CLOSURE_FIELDS,
            nested={
                "SourceBehavior": {
                    "fields": sorted(SOURCE_BEHAVIOR_FIELDS),
                    "RelativeSourceFile": sorted(RELATIVE_SOURCE_FILE_FIELDS),
                    "ImportEdge": sorted(IMPORT_EDGE_FIELDS),
                    "ModuleIdentity": sorted(MODULE_IDENTITY_FIELDS),
                },
                "SourceAudit": {"fields": sorted(SOURCE_AUDIT_FIELDS)},
            },
            constants={"entry_modules": list(SOURCE_CLOSURE_ENTRY_MODULES)},
            hash_preimages={
                "source_closure_fingerprint_sha256": "canonical SourceBehavior bytes",
                "full_payload_sha256": "canonical SourceClosure with full_payload_sha256 omitted",
            },
        ),
        SOURCE_ARM_STAGE_SUMMARY_SCHEMA: exact(
            SOURCE_ARM_STAGE_SUMMARY_SCHEMA,
            SOURCE_ARM_STAGE_SUMMARY_FIELDS,
            nested={
                "CompletedRowRef": {"fields": sorted(COMPLETED_ROW_REF_FIELDS)},
                "SourceArmDenominators": {
                    "fields": sorted(SOURCE_ARM_DENOMINATORS_FIELDS),
                    "AudioSetBinding": audio_set,
                },
                "SourceArmGates": {
                    "fields": sorted(SOURCE_ARM_GATES_FIELDS),
                    "RateValue": rate_value,
                    "StatsValue": stats_value,
                },
                "RuntimeSummary": {
                    "fields": sorted(RUNTIME_SUMMARY_FIELDS),
                    "StatsValue": stats_value,
                },
                "RssSummary": {"fields": sorted(RSS_SUMMARY_FIELDS)},
            },
            hash_preimages={
                "row_payloads_sha256": "canonical ordered CompletedRowRef list",
                "summary_fingerprint_sha256": (
                    "canonical SourceArmStageSummary with summary_fingerprint_sha256 omitted"
                ),
            },
        ),
        CANDIDATE_GLOBAL_MANIFEST_SCHEMA: exact(
            CANDIDATE_GLOBAL_MANIFEST_SCHEMA,
            CANDIDATE_GLOBAL_MANIFEST_FIELDS,
            nested={
                "CandidateGlobalEntry": {
                    "fields": sorted(CANDIDATE_GLOBAL_ENTRY_FIELDS),
                    "ArmRowShaMap": list(EXP007_EXECUTION_ORDER),
                }
            },
            hash_preimages={
                "ordered_entries_sha256": "canonical ordered CandidateGlobalEntry list",
                "manifest_fingerprint_sha256": (
                    "canonical CandidateGlobalManifest with manifest_fingerprint_sha256 omitted"
                ),
            },
        ),
        CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA: exact(
            CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA,
            [
                "bundle_fingerprint_sha256",
                "entry",
                "experiment_id",
                "row",
                "row_index",
                "schedule_arm",
                "schema",
                "schema_descriptor_sha256",
                "stage",
            ],
            hash_preimages={
                "bundle_fingerprint_sha256": (
                    "canonical CandidateReferenceRowBundle with bundle_fingerprint_sha256 omitted"
                )
            },
        ),
        CANDIDATE_REFERENCE_MANIFEST_SCHEMA: exact(
            CANDIDATE_REFERENCE_MANIFEST_SCHEMA,
            [
                "entries",
                "experiment_id",
                "input_manifest_sha256",
                "manifest_fingerprint_sha256",
                "ordered_entries_sha256",
                "reference_arm",
                "row_count",
                "schema",
                "schema_descriptor_sha256",
                "source_closure_fingerprint_sha256",
                "stage",
            ],
            hash_preimages={
                "ordered_entries_sha256": "canonical ordered CandidateReferenceRef list",
                "manifest_fingerprint_sha256": (
                    "canonical CandidateReferenceManifest with manifest_fingerprint_sha256 omitted"
                ),
            },
        ),
        SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA: exact(
            SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA,
            SCHEDULE_WEAK_VETO_SUMMARY_FIELDS,
            nested={
                "WeakRowRef": {"fields": sorted(WEAK_ROW_REF_FIELDS)},
                "ScheduleWeakDenominators": {
                    "fields": sorted(SCHEDULE_WEAK_DENOMINATORS_FIELDS),
                    "AudioSetBinding": audio_set,
                },
                "ScheduleWeakGates": {
                    "fields": sorted(SCHEDULE_WEAK_GATES_FIELDS),
                    "RatioValue": ratio_value,
                    "RateValue": rate_value,
                },
            },
            branches={
                "pass": "action=authorize_repair80",
                "ambiguous": "action=stop_ambiguous",
                "negative": "action=stop_negative",
            },
            hash_preimages={
                "selected_row_refs_sha256": (
                    "canonical ordered row_index/cache_audio_key/prediction_row_sha256 tuples"
                ),
                "weak_payloads_sha256": "canonical ordered WeakRowRef list",
                "summary_fingerprint_sha256": (
                    "canonical ScheduleWeakVetoSummary with summary_fingerprint_sha256 omitted"
                ),
            },
        ),
        SCHEDULE_WEAK_SUCCESS_SCHEMA: exact(
            SCHEDULE_WEAK_SUCCESS_SCHEMA,
            SCHEDULE_WEAK_SUCCESS_FIELDS,
            nested={"ScheduleWeakVetoSummary": {"schema": SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA}},
            hash_preimages={
                "summary_payload_sha256": "canonical complete ScheduleWeakVetoSummary",
                "outcome_fingerprint_sha256": (
                    "canonical ScheduleWeakVetoSuccess with outcome_fingerprint_sha256 omitted"
                ),
            },
        ),
        SCHEDULE_WEAK_HARD_FAILURE_SCHEMA: exact(
            SCHEDULE_WEAK_HARD_FAILURE_SCHEMA,
            SCHEDULE_WEAK_HARD_FAILURE_FIELDS,
            nested={"ScheduleWeakFailureRecord": {"schema": SCHEDULE_WEAK_FAILURE_SCHEMA}},
            hash_preimages={
                "failure_payload_sha256": "canonical complete ScheduleWeakFailureRecord",
                "outcome_fingerprint_sha256": (
                    "canonical ScheduleWeakVetoHardFailure with outcome_fingerprint_sha256 omitted"
                ),
            },
        ),
        SCHEDULE_WEAK_FAILURE_SCHEMA: exact(
            SCHEDULE_WEAK_FAILURE_SCHEMA,
            SCHEDULE_WEAK_FAILURE_FIELDS,
            nested={
                "WeakRowRef": {"fields": sorted(WEAK_ROW_REF_FIELDS)},
                "WeakPendingRowRef": {"fields": sorted(WEAK_PENDING_ROW_REF_FIELDS)},
            },
            enums={
                "failure_kind": sorted(SCHEDULE_WEAK_FAILURE_KINDS),
                "failure_stage": sorted(SCHEDULE_WEAK_FAILURE_STAGES),
            },
            hash_preimages={
                "failure_deterministic_fingerprint_sha256": (
                    "canonical failure payload excluding both hash fields"
                ),
                "full_payload_sha256": "canonical payload excluding only itself",
            },
        ),
        WEAK_ROW_SCHEMA: exact(
            WEAK_ROW_SCHEMA,
            WEAK_ROW_FIELDS,
            nested={
                "ComparatorAvailability": {
                    "fields": sorted(COMPARATOR_AVAILABILITY_FIELDS)
                },
                "PhaseSummary": {
                    "fields": sorted(PHASE_SUMMARY_FIELDS),
                    "StatsValue": stats_value,
                },
                "DriftSummary": {
                    "fields": sorted(DRIFT_SUMMARY_FIELDS),
                    "StatsValue": stats_value,
                },
                "BoundarySummary": {
                    "fields": sorted(BOUNDARY_SUMMARY_FIELDS),
                    "RatioValue": ratio_value,
                    "StatsValue": stats_value,
                },
                "ObjectGridSummary": {
                    "fields": sorted(OBJECT_GRID_SUMMARY_FIELDS),
                    "RateValue": rate_value,
                    "StatsValue": stats_value,
                },
            },
            branches={
                "schedule16": "binds committed FourArmStageSummary and ConfigSelection",
                "repair80": "repeats schedule bindings through Repair80InputBinding",
            },
            hash_preimages={
                "deterministic_projection_sha256": (
                    "canonical WeakRow projection excluding payload hash"
                ),
                "weak_row_payload_sha256": (
                    "canonical WeakRow with weak_row_payload_sha256 omitted"
                ),
            },
        ),
        REPAIR80_SUMMARY_SCHEMA: exact(
            REPAIR80_SUMMARY_SCHEMA,
            REPAIR80_SUMMARY_FIELDS,
            nested={
                "SourceRef": {"schema": SOURCE_REF_SCHEMA},
                "CompletedRowRef": {"fields": sorted(COMPLETED_ROW_REF_FIELDS)},
                "WeakRowRef": {"fields": sorted(WEAK_ROW_REF_FIELDS)},
                "Repair80Denominators": {
                    "fields": sorted(REPAIR80_DENOMINATORS_FIELDS),
                    "AudioSetBinding": audio_set,
                },
                "Repair80Gates": {
                    "fields": sorted(REPAIR80_GATES_FIELDS),
                    "RatioValue": ratio_value,
                    "RateValue": rate_value,
                    "StatsValue": stats_value,
                },
                "RuntimeSummary": {
                    "fields": sorted(RUNTIME_SUMMARY_FIELDS),
                    "StatsValue": stats_value,
                },
                "RssSummary": {"fields": sorted(RSS_SUMMARY_FIELDS)},
            },
            branches={
                "pass": "action=write_result_and_next_no_data_card",
                "ambiguous": "action=stop_ambiguous",
                "negative": "action=stop_negative",
            },
            hash_preimages={
                "row_payloads_sha256": "canonical ordered CompletedRowRef list",
                "weak_payloads_sha256": "canonical ordered WeakRowRef list",
                "row_weak_pairs_sha256": (
                    "canonical ordered row/ref pairing tuples"
                ),
                "summary_fingerprint_sha256": (
                    "canonical Repair80Summary with summary_fingerprint_sha256 omitted"
                ),
            },
        ),
    }
    return descriptors


def load_json_strict(data: bytes | str) -> Any:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return _jsonable(
        json.loads(
            data,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_json_constant,
        )
    )


def load_json_file_strict(path: str | Path) -> Any:
    return load_json_strict(Path(path).read_bytes())


def validate_exact_fields(
    payload: Mapping[str, Any],
    expected_fields: frozenset[str],
    context: str,
) -> None:
    actual = set(payload)
    if actual != expected_fields:
        missing = sorted(expected_fields.difference(actual))
        extra = sorted(actual.difference(expected_fields))
        raise ValueError(
            f"{context} fields are incomplete or unsupported: "
            f"missing={missing!r} extra={extra!r}"
        )


def require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def require_positive_int(value: Any, field_name: str) -> int:
    result = require_nonnegative_int(value, field_name)
    if result <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return result


def require_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def make_source_ref(
    *,
    artifact_schema: str,
    sha256: str,
    row_count: int,
    ordered_rows_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": SOURCE_REF_SCHEMA,
        "artifact_schema": require_nonempty_string(
            artifact_schema,
            "artifact_schema",
        ),
        "sha256": require_sha256(sha256, "sha256"),
        "row_count": require_nonnegative_int(row_count, "row_count"),
        "ordered_rows_sha256": require_sha256(
            ordered_rows_sha256,
            "ordered_rows_sha256",
        ),
    }
    return validate_source_ref(payload)


def validate_source_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "SourceRef")
    validate_exact_fields(payload, SOURCE_REF_FIELDS, "SourceRef")
    if payload.get("schema") != SOURCE_REF_SCHEMA:
        raise ValueError("SourceRef schema is invalid")
    return {
        "schema": SOURCE_REF_SCHEMA,
        "artifact_schema": require_nonempty_string(
            payload.get("artifact_schema"),
            "SourceRef.artifact_schema",
        ),
        "sha256": require_sha256(payload.get("sha256"), "SourceRef.sha256"),
        "row_count": require_nonnegative_int(
            payload.get("row_count"),
            "SourceRef.row_count",
        ),
        "ordered_rows_sha256": require_sha256(
            payload.get("ordered_rows_sha256"),
            "SourceRef.ordered_rows_sha256",
        ),
    }


def validate_repair80_identity_sources_for_execution(
    *,
    repair80_identity_source_artifact: bytes,
    repair80_label_source_artifact: bytes,
    repair80_identity_rows: Sequence[Mapping[str, Any]],
    repair80_label_rows: Sequence[Mapping[str, Any]],
    identity_source: Mapping[str, Any],
    label_source: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    selector = importlib.import_module(
        "pulsefield_model.timing.evaluation.exp007_selector"
    )
    identities = selector.validate_repair80_identity_label_sources(
        repair80_identity_source_artifact=repair80_identity_source_artifact,
        label_source_artifact=repair80_label_source_artifact,
        repair80_identity_rows=repair80_identity_rows,
        label_rows=repair80_label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )
    return tuple(validate_identity(row) for row in identities)


def make_identity(
    *,
    stage: str,
    row_index: int,
    source_row_index: int,
    cache_audio_key: str,
    audio_group_key: str,
    label_stratum: str,
    source_long_track: bool,
    duration_ms: int | float,
    label_source_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": IDENTITY_SCHEMA,
        "stage": stage,
        "row_index": row_index,
        "source_row_index": source_row_index,
        "cache_audio_key": cache_audio_key,
        "audio_group_key": audio_group_key,
        "label_stratum": label_stratum,
        "source_long_track": source_long_track,
        "duration_ms": duration_ms,
        "label_source_sha256": label_source_sha256,
    }
    return validate_identity(with_payload_hash(payload, "identity_payload_sha256"))


def validate_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "Identity")
    validate_exact_fields(payload, IDENTITY_FIELDS, "Identity")
    if payload.get("schema") != IDENTITY_SCHEMA:
        raise ValueError("Identity schema is invalid")
    stage = _require_stage(payload.get("stage"), "Identity.stage")
    label = payload.get("label_stratum")
    if label not in LABEL_STRATA:
        raise ValueError("Identity.label_stratum is invalid")
    source_long_track = payload.get("source_long_track")
    if not isinstance(source_long_track, bool):
        raise ValueError("Identity.source_long_track must be a bool")
    duration_ms = payload.get("duration_ms")
    duration_ms_value = require_finite_number(duration_ms, "Identity.duration_ms")
    if duration_ms_value <= 0:
        raise ValueError("Identity.duration_ms must be positive")
    result = {
        "schema": IDENTITY_SCHEMA,
        "stage": stage,
        "row_index": require_nonnegative_int(
            payload.get("row_index"),
            "Identity.row_index",
        ),
        "source_row_index": require_nonnegative_int(
            payload.get("source_row_index"),
            "Identity.source_row_index",
        ),
        "cache_audio_key": require_nonempty_string(
            payload.get("cache_audio_key"),
            "Identity.cache_audio_key",
        ),
        "audio_group_key": require_nonempty_string(
            payload.get("audio_group_key"),
            "Identity.audio_group_key",
        ),
        "label_stratum": label,
        "source_long_track": source_long_track,
        "duration_ms": duration_ms,
        "label_source_sha256": require_sha256(
            payload.get("label_source_sha256"),
            "Identity.label_source_sha256",
        ),
        "identity_payload_sha256": require_sha256(
            payload.get("identity_payload_sha256"),
            "Identity.identity_payload_sha256",
        ),
    }
    validate_payload_hash(result, "identity_payload_sha256", context="Identity")
    return result


def make_rate_value(numerator: int | float, denominator: int | float) -> dict[str, Any]:
    numerator_value = require_finite_number(numerator, "RateValue.numerator")
    denominator_value = require_finite_number(denominator, "RateValue.denominator")
    if denominator_value <= 0 or numerator_value < 0 or numerator_value > denominator_value:
        raise ValueError("RateValue numerator/denominator are invalid")
    return {
        "numerator": numerator_value,
        "denominator": denominator_value,
        "value": float(numerator_value / denominator_value),
    }


def validate_rate_value(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RateValue")
    validate_exact_fields(payload, frozenset({"numerator", "denominator", "value"}), "RateValue")
    expected = make_rate_value(payload.get("numerator"), payload.get("denominator"))
    if require_finite_number(payload.get("value"), "RateValue.value") != expected["value"]:
        raise ValueError("RateValue.value mismatch")
    return expected


def make_ratio_value(
    numerator: int | float | None,
    denominator: int | float | None,
) -> dict[str, Any]:
    if numerator is None and denominator is None:
        return {"state": "undefined", "numerator": None, "denominator": None, "value": None}
    numerator_value = require_finite_number(numerator, "RatioValue.numerator")
    denominator_value = require_finite_number(denominator, "RatioValue.denominator")
    if numerator_value < 0 or denominator_value < 0:
        raise ValueError("RatioValue numerator/denominator must be nonnegative")
    if denominator_value > 0:
        return {
            "state": "finite",
            "numerator": numerator_value,
            "denominator": denominator_value,
            "value": float(numerator_value / denominator_value),
        }
    if numerator_value == 0:
        return {
            "state": "both_zero",
            "numerator": 0.0,
            "denominator": 0.0,
            "value": 1.0,
        }
    return {
        "state": "positive_infinity",
        "numerator": numerator_value,
        "denominator": 0.0,
        "value": None,
    }


def validate_ratio_value(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RatioValue")
    validate_exact_fields(
        payload,
        frozenset({"state", "numerator", "denominator", "value"}),
        "RatioValue",
    )
    state = payload.get("state")
    if state == "undefined":
        if payload.get("numerator") is not None or payload.get("denominator") is not None:
            raise ValueError("RatioValue undefined operands must be null")
        if payload.get("value") is not None:
            raise ValueError("RatioValue undefined value must be null")
        return {"state": "undefined", "numerator": None, "denominator": None, "value": None}
    expected = make_ratio_value(payload.get("numerator"), payload.get("denominator"))
    if payload.get("state") != expected["state"] or payload.get("value") != expected["value"]:
        raise ValueError("RatioValue branch mismatch")
    return expected


def make_coverage_value(
    numerator: int | float | None,
    denominator: int | float | None,
) -> dict[str, Any]:
    if denominator is None or denominator == 0:
        return make_ratio_value(None, None)
    return make_rate_value(numerator, denominator)


def validate_coverage_value(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "state" in payload:
        ratio = validate_ratio_value(payload)
        if ratio["state"] != "undefined":
            raise ValueError("CoverageValue RatioValue branch must be undefined")
        return ratio
    return validate_rate_value(payload)


def make_stats_value(values: Sequence[int | float]) -> dict[str, Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, SequenceABC):
        raise ValueError("StatsValue input must be a sequence")
    numbers = [require_finite_number(value, "StatsValue input") for value in values]
    if not numbers:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "maximum": None}
    return {
        "count": len(numbers),
        "mean": float(sum(numbers) / len(numbers)),
        "p50": _linear_quantile(numbers, 0.5),
        "p90": _linear_quantile(numbers, 0.9),
        "maximum": max(numbers),
    }


def validate_stats_value(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "StatsValue")
    validate_exact_fields(
        payload,
        frozenset({"count", "mean", "p50", "p90", "maximum"}),
        "StatsValue",
    )
    count = require_nonnegative_int(payload.get("count"), "StatsValue.count")
    if count == 0:
        if any(payload.get(key) is not None for key in ("mean", "p50", "p90", "maximum")):
            raise ValueError("StatsValue count=0 values must be null")
    else:
        for key in ("mean", "p50", "p90", "maximum"):
            value = require_finite_number(payload.get(key), f"StatsValue.{key}")
            if value < 0:
                raise ValueError(f"StatsValue.{key} must be nonnegative")
    return dict(payload)


def make_audio_set_binding(keys: Sequence[str]) -> dict[str, Any]:
    unique_keys = sorted({require_nonempty_string(key, "AudioSetBinding key") for key in keys})
    return {
        "count": len(unique_keys),
        "sorted_cache_audio_keys_sha256": canonical_json_sha256(unique_keys),
    }


def validate_audio_set_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "AudioSetBinding")
    validate_exact_fields(
        payload,
        frozenset({"count", "sorted_cache_audio_keys_sha256"}),
        "AudioSetBinding",
    )
    return {
        "count": require_nonnegative_int(payload.get("count"), "AudioSetBinding.count"),
        "sorted_cache_audio_keys_sha256": require_sha256(
            payload.get("sorted_cache_audio_keys_sha256"),
            "AudioSetBinding.sorted_cache_audio_keys_sha256",
        ),
    }


def make_local_frontier_config(schedule_arm: str) -> dict[str, Any]:
    arm = _require_schedule_arm(schedule_arm, "local_frontier_config.schedule_arm")
    return {
        "schedule_arm": arm,
        "exported_frontier_width": 16,
        "local_beam_width": 64,
        "max_boundary_candidates_per_block": 32,
        "max_tempo_candidates_per_block": 64,
        "max_blocks": 192,
        "max_sections": 20,
        "max_section_score_misses_per_block": 30_000,
        "max_section_score_misses_per_audio": 500_000,
    }


def make_run_config(
    *,
    stage: str,
    schedule_arm: str,
    selector_manifest_sha256: str,
    input_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    cache_config_sha256: str,
    grid_fitter_config_sha256: str,
    weak_config_sha256: str,
    schedule_weak_veto_outcome_sha256: str | None = None,
    local_frontier_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    arm = _require_schedule_arm(schedule_arm, "RunConfig.schedule_arm")
    stage_value = _require_stage(stage, "RunConfig.stage")
    payload = {
        "schema": RUN_CONFIG_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": stage_value,
        "schema_descriptor_sha256": schema_descriptor_sha256(RUN_CONFIG_SCHEMA),
        "schedule_arm": arm,
        "method_ids": dict(METHOD_IDS),
        "candidate_policy": _default_candidate_policy(),
        "pool_policy": _default_pool_policy(),
        "limits": _default_limits(),
        "selector_manifest_sha256": selector_manifest_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "schedule_weak_veto_outcome_sha256": schedule_weak_veto_outcome_sha256,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "cache_config_sha256": cache_config_sha256,
        "grid_fitter_config_sha256": grid_fitter_config_sha256,
        "local_frontier_config": (
            make_local_frontier_config(arm)
            if local_frontier_config is None
            else dict(local_frontier_config)
        ),
        "weak_config_sha256": weak_config_sha256,
    }
    return validate_run_config(with_payload_hash(payload, "run_config_fingerprint_sha256"))


def validate_run_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RunConfig")
    validate_exact_fields(payload, RUN_CONFIG_FIELDS, "RunConfig")
    if payload.get("schema") != RUN_CONFIG_SCHEMA:
        raise ValueError("RunConfig schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("RunConfig experiment_id is invalid")
    stage = _require_stage(payload.get("stage"), "RunConfig.stage")
    arm = _require_schedule_arm(payload.get("schedule_arm"), "RunConfig.schedule_arm")
    _require_descriptor(payload, RUN_CONFIG_SCHEMA, "RunConfig")
    _validate_method_ids(payload.get("method_ids"))
    _validate_candidate_policy(payload.get("candidate_policy"))
    _validate_pool_policy(payload.get("pool_policy"))
    _validate_limits(payload.get("limits"))
    _validate_local_frontier_config(payload.get("local_frontier_config"), arm=arm)
    selector_sha = require_sha256(
        payload.get("selector_manifest_sha256"),
        "RunConfig.selector_manifest_sha256",
    )
    input_sha = require_sha256(
        payload.get("input_manifest_sha256"),
        "RunConfig.input_manifest_sha256",
    )
    weak_outcome_sha = payload.get("schedule_weak_veto_outcome_sha256")
    if stage == EXP007_SCHEDULE_STAGE:
        if input_sha != selector_sha:
            raise ValueError("schedule16 RunConfig input manifest must equal selector SHA")
        if weak_outcome_sha is not None:
            raise ValueError("schedule16 RunConfig weak outcome SHA must be null")
    else:
        require_sha256(weak_outcome_sha, "RunConfig.schedule_weak_veto_outcome_sha256")
        if input_sha == selector_sha:
            raise ValueError("repair80 RunConfig input manifest must be repair binding SHA")
    for field_name in (
        "source_closure_fingerprint_sha256",
        "cache_config_sha256",
        "grid_fitter_config_sha256",
        "weak_config_sha256",
    ):
        require_sha256(payload.get(field_name), f"RunConfig.{field_name}")
    validate_payload_hash(payload, "run_config_fingerprint_sha256", context="RunConfig")
    return dict(payload)


def make_completed_row_ref(
    *,
    row_index: int,
    cache_audio_key: str,
    identity_payload_sha256: str,
    row_payload_sha256: str,
    candidate_reference_entry_payload_sha256: str | None,
) -> dict[str, Any]:
    payload = {
        "row_index": row_index,
        "cache_audio_key": cache_audio_key,
        "identity_payload_sha256": identity_payload_sha256,
        "row_payload_sha256": row_payload_sha256,
        "candidate_reference_entry_payload_sha256": candidate_reference_entry_payload_sha256,
    }
    return validate_completed_row_ref(payload)


def validate_completed_row_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "CompletedRowRef")
    validate_exact_fields(payload, COMPLETED_ROW_REF_FIELDS, "CompletedRowRef")
    candidate_sha = payload.get("candidate_reference_entry_payload_sha256")
    if candidate_sha is not None:
        candidate_sha = require_sha256(
            candidate_sha,
            "CompletedRowRef.candidate_reference_entry_payload_sha256",
        )
    return {
        "row_index": require_nonnegative_int(
            payload.get("row_index"),
            "CompletedRowRef.row_index",
        ),
        "cache_audio_key": require_nonempty_string(
            payload.get("cache_audio_key"),
            "CompletedRowRef.cache_audio_key",
        ),
        "identity_payload_sha256": require_sha256(
            payload.get("identity_payload_sha256"),
            "CompletedRowRef.identity_payload_sha256",
        ),
        "row_payload_sha256": require_sha256(
            payload.get("row_payload_sha256"),
            "CompletedRowRef.row_payload_sha256",
        ),
        "candidate_reference_entry_payload_sha256": candidate_sha,
    }


def make_pending_identity_ref(
    *,
    row_index: int,
    cache_audio_key: str,
    identity_payload_sha256: str,
) -> dict[str, Any]:
    return validate_pending_identity_ref(
        {
            "row_index": row_index,
            "cache_audio_key": cache_audio_key,
            "identity_payload_sha256": identity_payload_sha256,
        }
    )


def validate_pending_identity_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "PendingIdentityRef")
    validate_exact_fields(payload, PENDING_IDENTITY_REF_FIELDS, "PendingIdentityRef")
    return {
        "row_index": require_nonnegative_int(
            payload.get("row_index"),
            "PendingIdentityRef.row_index",
        ),
        "cache_audio_key": require_nonempty_string(
            payload.get("cache_audio_key"),
            "PendingIdentityRef.cache_audio_key",
        ),
        "identity_payload_sha256": require_sha256(
            payload.get("identity_payload_sha256"),
            "PendingIdentityRef.identity_payload_sha256",
        ),
    }


def make_cache_identity(
    *,
    relative_cache_path: str,
    exists: bool,
    cache_config_sha256: str,
    audio_cache_key_sha256: str,
    size_bytes: int | None = None,
    mtime_ns: int | None = None,
    inode: int | None = None,
    device: int | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": CACHE_IDENTITY_SCHEMA,
        "relative_cache_path": relative_cache_path,
        "exists": exists,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
        "inode": inode,
        "device": device,
        "sha256": sha256,
        "cache_config_sha256": cache_config_sha256,
        "audio_cache_key_sha256": audio_cache_key_sha256,
    }
    return validate_cache_identity(payload)


def validate_cache_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "CacheIdentity")
    validate_exact_fields(payload, CACHE_IDENTITY_FIELDS, "CacheIdentity")
    if payload.get("schema") != CACHE_IDENTITY_SCHEMA:
        raise ValueError("CacheIdentity schema is invalid")
    exists = payload.get("exists")
    if not isinstance(exists, bool):
        raise ValueError("CacheIdentity.exists must be a bool")
    result: dict[str, Any] = {
        "schema": CACHE_IDENTITY_SCHEMA,
        "relative_cache_path": require_relative_posix_path(
            payload.get("relative_cache_path"),
            "CacheIdentity.relative_cache_path",
        ),
        "exists": exists,
        "cache_config_sha256": require_sha256(
            payload.get("cache_config_sha256"),
            "CacheIdentity.cache_config_sha256",
        ),
        "audio_cache_key_sha256": require_sha256(
            payload.get("audio_cache_key_sha256"),
            "CacheIdentity.audio_cache_key_sha256",
        ),
    }
    nullable_names = ("size_bytes", "mtime_ns", "inode", "device", "sha256")
    if exists:
        for name in ("size_bytes", "mtime_ns", "inode", "device"):
            result[name] = require_nonnegative_int(
                payload.get(name),
                f"CacheIdentity.{name}",
            )
        result["sha256"] = require_sha256(
            payload.get("sha256"),
            "CacheIdentity.sha256",
        )
    else:
        for name in nullable_names:
            if payload.get(name) is not None:
                raise ValueError("CacheIdentity missing-cache fields must be null")
            result[name] = None
    return {
        "schema": result["schema"],
        "relative_cache_path": result["relative_cache_path"],
        "exists": result["exists"],
        "size_bytes": result["size_bytes"],
        "mtime_ns": result["mtime_ns"],
        "inode": result["inode"],
        "device": result["device"],
        "sha256": result["sha256"],
        "cache_config_sha256": result["cache_config_sha256"],
        "audio_cache_key_sha256": result["audio_cache_key_sha256"],
    }


def make_restricted_prediction(
    *,
    frame_count: int,
    frame_rate_hz: int | float,
    input_signal_sha256: str,
    beat_bytes_sha256: str,
    downbeat_bytes_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": RESTRICTED_PREDICTION_SCHEMA,
        "frame_count": frame_count,
        "frame_rate_hz": frame_rate_hz,
        "beat_dtype": "<f4",
        "downbeat_dtype": "<f4",
        "input_signal_sha256": input_signal_sha256,
        "beat_bytes_sha256": beat_bytes_sha256,
        "downbeat_bytes_sha256": downbeat_bytes_sha256,
        "source_path_is_none": True,
        "arrays_read_only": True,
        "shares_loaded_memory": True,
    }
    return validate_restricted_prediction(payload)


def validate_restricted_prediction(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RestrictedPrediction")
    validate_exact_fields(payload, RESTRICTED_PREDICTION_FIELDS, "RestrictedPrediction")
    if payload.get("schema") != RESTRICTED_PREDICTION_SCHEMA:
        raise ValueError("RestrictedPrediction schema is invalid")
    if payload.get("beat_dtype") != "<f4" or payload.get("downbeat_dtype") != "<f4":
        raise ValueError("RestrictedPrediction dtypes must be <f4")
    for name in ("source_path_is_none", "arrays_read_only", "shares_loaded_memory"):
        if payload.get(name) is not True:
            raise ValueError(f"RestrictedPrediction.{name} must be true")
    frame_rate = require_finite_number(
        payload.get("frame_rate_hz"),
        "RestrictedPrediction.frame_rate_hz",
    )
    if frame_rate <= 0:
        raise ValueError("RestrictedPrediction.frame_rate_hz must be positive")
    return {
        "schema": RESTRICTED_PREDICTION_SCHEMA,
        "frame_count": require_nonnegative_int(
            payload.get("frame_count"),
            "RestrictedPrediction.frame_count",
        ),
        "frame_rate_hz": payload.get("frame_rate_hz"),
        "beat_dtype": "<f4",
        "downbeat_dtype": "<f4",
        "input_signal_sha256": require_sha256(
            payload.get("input_signal_sha256"),
            "RestrictedPrediction.input_signal_sha256",
        ),
        "beat_bytes_sha256": require_sha256(
            payload.get("beat_bytes_sha256"),
            "RestrictedPrediction.beat_bytes_sha256",
        ),
        "downbeat_bytes_sha256": require_sha256(
            payload.get("downbeat_bytes_sha256"),
            "RestrictedPrediction.downbeat_bytes_sha256",
        ),
        "source_path_is_none": True,
        "arrays_read_only": True,
        "shares_loaded_memory": True,
    }


def make_timing_v3_grid_payload(
    *,
    origin_beat: int | float,
    origin_time_ms: int | float,
    coverage_start_ms: int | float,
    coverage_end_ms: int | float,
    sections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": TIMING_V3_GRID_SCHEMA,
        "version": 1,
        "origin_beat": origin_beat,
        "origin_time_ms": origin_time_ms,
        "coverage_start_ms": coverage_start_ms,
        "coverage_end_ms": coverage_end_ms,
        "sections": [dict(section) for section in sections],
    }
    return validate_timing_v3_grid_payload(payload)


def validate_timing_v3_grid_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "TimingV3GridPayload")
    validate_exact_fields(payload, TIMING_V3_GRID_PAYLOAD_FIELDS, "TimingV3GridPayload")
    if payload.get("schema") != TIMING_V3_GRID_SCHEMA:
        raise ValueError("TimingV3GridPayload schema is invalid")
    if payload.get("version") != 1:
        raise ValueError("TimingV3GridPayload version is invalid")
    origin_beat = require_finite_number(
        payload.get("origin_beat"),
        "TimingV3GridPayload.origin_beat",
    )
    origin_time = require_finite_number(
        payload.get("origin_time_ms"),
        "TimingV3GridPayload.origin_time_ms",
    )
    coverage_start = require_finite_number(
        payload.get("coverage_start_ms"),
        "TimingV3GridPayload.coverage_start_ms",
    )
    coverage_end = require_finite_number(
        payload.get("coverage_end_ms"),
        "TimingV3GridPayload.coverage_end_ms",
    )
    if coverage_end <= coverage_start:
        raise ValueError("TimingV3GridPayload coverage must be increasing")
    raw_sections = _require_sequence(payload.get("sections"), "TimingV3GridPayload.sections")
    if not raw_sections or len(raw_sections) > 20:
        raise ValueError("TimingV3GridPayload sections must be nonempty and <=20")
    sections = [_validate_timing_v3_section(section) for section in raw_sections]
    previous_end: int | None = None
    for section in sections:
        if previous_end is not None and section["start_beat"] != previous_end:
            raise ValueError("TimingV3GridPayload sections must be contiguous")
        previous_end = section["end_beat"]
    result = {
        "schema": TIMING_V3_GRID_SCHEMA,
        "version": 1,
        "origin_beat": payload.get("origin_beat"),
        "origin_time_ms": payload.get("origin_time_ms"),
        "coverage_start_ms": payload.get("coverage_start_ms"),
        "coverage_end_ms": payload.get("coverage_end_ms"),
        "sections": sections,
    }
    _require_canonical_roundtrip(result, "TimingV3GridPayload")
    return result


def _validate_timing_v3_section(payload: Any) -> dict[str, Any]:
    _require_mapping(payload, "TimingV3Section")
    validate_exact_fields(payload, TIMING_V3_SECTION_FIELDS, "TimingV3Section")
    start = require_nonnegative_int(payload.get("start_beat"), "TimingV3Section.start_beat")
    end = require_nonnegative_int(payload.get("end_beat"), "TimingV3Section.end_beat")
    if end <= start:
        raise ValueError("TimingV3Section end_beat must exceed start_beat")
    bpm = require_finite_number(payload.get("bpm"), "TimingV3Section.bpm")
    if bpm < 20.0 or bpm > 1000.0:
        raise ValueError("TimingV3Section bpm is outside 20..1000")
    return {"start_beat": start, "end_beat": end, "bpm": payload.get("bpm")}


def make_v2_grid_payload(*, segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return validate_v2_grid_payload(
        {"schema": V2_GRID_SCHEMA, "segments": [dict(segment) for segment in segments]}
    )


def validate_v2_grid_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "V2GridPayload")
    validate_exact_fields(payload, V2_GRID_PAYLOAD_FIELDS, "V2GridPayload")
    if payload.get("schema") != V2_GRID_SCHEMA:
        raise ValueError("V2GridPayload schema is invalid")
    raw_segments = _require_sequence(payload.get("segments"), "V2GridPayload.segments")
    if not raw_segments:
        raise ValueError("V2GridPayload segments must be nonempty")
    segments = [_validate_v2_segment(segment) for segment in raw_segments]
    previous_offset: float | None = None
    for segment in segments:
        offset = float(segment["offset_ms"])
        if previous_offset is not None and offset <= previous_offset:
            raise ValueError("V2GridPayload segment offsets must increase")
        previous_offset = offset
    result = {"schema": V2_GRID_SCHEMA, "segments": segments}
    _require_canonical_roundtrip(result, "V2GridPayload")
    return result


def _validate_v2_segment(payload: Any) -> dict[str, Any]:
    _require_mapping(payload, "V2Segment")
    validate_exact_fields(payload, V2_SEGMENT_FIELDS, "V2Segment")
    offset = require_finite_number(payload.get("offset_ms"), "V2Segment.offset_ms")
    beat_length = require_finite_number(
        payload.get("beat_length_ms"),
        "V2Segment.beat_length_ms",
    )
    if beat_length <= 0:
        raise ValueError("V2Segment beat_length_ms must be positive")
    return {
        "offset_ms": payload.get("offset_ms"),
        "beat_length_ms": payload.get("beat_length_ms"),
        "meter": require_positive_int(payload.get("meter"), "V2Segment.meter"),
    }


def make_grid_envelope(*, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    grid_payload = (
        validate_timing_v3_grid_payload(payload)
        if kind == "timing_v3"
        else validate_v2_grid_payload(payload)
        if kind == "current_v2"
        else None
    )
    if grid_payload is None:
        raise ValueError("GridEnvelope kind is invalid")
    result = {
        "kind": kind,
        "payload": grid_payload,
        "grid_sha256": canonical_json_sha256(grid_payload),
        "deterministic_projection_sha256": canonical_json_sha256(
            {"kind": kind, "payload": grid_payload}
        ),
    }
    return validate_grid_envelope(result)


def validate_grid_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "GridEnvelope")
    validate_exact_fields(payload, GRID_ENVELOPE_FIELDS, "GridEnvelope")
    kind = payload.get("kind")
    if kind == "timing_v3":
        grid_payload = validate_timing_v3_grid_payload(payload.get("payload"))
    elif kind == "current_v2":
        grid_payload = validate_v2_grid_payload(payload.get("payload"))
    else:
        raise ValueError("GridEnvelope kind is invalid")
    if payload.get("grid_sha256") != canonical_json_sha256(grid_payload):
        raise ValueError("GridEnvelope grid_sha256 mismatch")
    expected_projection = canonical_json_sha256({"kind": kind, "payload": grid_payload})
    if payload.get("deterministic_projection_sha256") != expected_projection:
        raise ValueError("GridEnvelope deterministic_projection_sha256 mismatch")
    return {
        "kind": kind,
        "payload": grid_payload,
        "grid_sha256": payload["grid_sha256"],
        "deterministic_projection_sha256": payload[
            "deterministic_projection_sha256"
        ],
    }


def make_grid_summary(
    *,
    grid: Mapping[str, Any],
    coverage_start_ms: int | float | None = None,
    coverage_end_ms: int | float | None = None,
    maximum_seam_discontinuity_ms: int | float = 0.0,
) -> dict[str, Any]:
    envelope = validate_grid_envelope(grid)
    if envelope["kind"] == "timing_v3":
        grid_payload = envelope["payload"]
        start = grid_payload["coverage_start_ms"]
        end = grid_payload["coverage_end_ms"]
        count = len(grid_payload["sections"])
    else:
        if coverage_start_ms is None or coverage_end_ms is None:
            raise ValueError("current_v2 GridSummary requires coverage bounds")
        start = coverage_start_ms
        end = coverage_end_ms
        count = len(envelope["payload"]["segments"])
    return validate_grid_summary(
        {
            "grid_kind": envelope["kind"],
            "section_count": count,
            "jump_count": count - 1,
            "coverage_start_ms": start,
            "coverage_end_ms": end,
            "maximum_seam_discontinuity_ms": maximum_seam_discontinuity_ms,
        },
        grid=envelope,
    )


def validate_grid_summary(
    payload: Mapping[str, Any],
    *,
    grid: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_mapping(payload, "GridSummary")
    validate_exact_fields(payload, GRID_SUMMARY_FIELDS, "GridSummary")
    kind = payload.get("grid_kind")
    if kind not in GRID_KINDS:
        raise ValueError("GridSummary.grid_kind is invalid")
    section_count = require_positive_int(
        payload.get("section_count"),
        "GridSummary.section_count",
    )
    if payload.get("jump_count") != section_count - 1:
        raise ValueError("GridSummary.jump_count mismatch")
    coverage_start = require_finite_number(
        payload.get("coverage_start_ms"),
        "GridSummary.coverage_start_ms",
    )
    coverage_end = require_finite_number(
        payload.get("coverage_end_ms"),
        "GridSummary.coverage_end_ms",
    )
    if coverage_end <= coverage_start:
        raise ValueError("GridSummary coverage must be increasing")
    seam = require_finite_number(
        payload.get("maximum_seam_discontinuity_ms"),
        "GridSummary.maximum_seam_discontinuity_ms",
    )
    if seam < 0:
        raise ValueError("GridSummary seam must be nonnegative")
    if grid is not None:
        envelope = validate_grid_envelope(grid)
        if envelope["kind"] != kind:
            raise ValueError("GridSummary kind mismatch")
        if kind == "timing_v3":
            grid_payload = envelope["payload"]
            if section_count != len(grid_payload["sections"]):
                raise ValueError("GridSummary section_count mismatch")
            if (
                payload.get("coverage_start_ms") != grid_payload["coverage_start_ms"]
                or payload.get("coverage_end_ms") != grid_payload["coverage_end_ms"]
            ):
                raise ValueError("GridSummary coverage mismatch")
            if seam != 0.0:
                raise ValueError("Timing-v3 GridSummary seam must be 0.0")
        elif section_count != len(envelope["payload"]["segments"]):
            raise ValueError("GridSummary v2 segment count mismatch")
    return dict(payload)


def make_method_result(
    *,
    method_kind: str,
    status: str,
    reason: str | None = None,
    fallback_kind: str | None = None,
    grid: Mapping[str, Any] | None = None,
    grid_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    method_id = METHOD_IDS.get(method_kind)
    if method_id is None:
        raise ValueError("MethodResult method_kind is invalid")
    payload = {
        "method_id": method_id,
        "method_kind": method_kind,
        "status": status,
        "reason": reason,
        "fallback_kind": fallback_kind,
        "grid": None if grid is None else dict(grid),
        "grid_summary": None if grid_summary is None else dict(grid_summary),
    }
    payload["deterministic_projection_sha256"] = canonical_json_sha256(payload)
    return validate_method_result(payload, expected_kind=method_kind)


def validate_method_result(
    payload: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    _require_mapping(payload, "MethodResult")
    validate_exact_fields(payload, METHOD_RESULT_FIELDS, "MethodResult")
    kind = payload.get("method_kind")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError("MethodResult method_kind mismatch")
    if kind not in METHODS_FIELDS:
        raise ValueError("MethodResult method_kind is invalid")
    if payload.get("method_id") != METHOD_IDS[kind]:
        raise ValueError("MethodResult method_id mismatch")
    status = require_nonempty_string(payload.get("status"), "MethodResult.status")
    reason = payload.get("reason")
    fallback = payload.get("fallback_kind")
    if reason is not None:
        reason = require_nonempty_string(reason, "MethodResult.reason")
    if fallback is not None:
        fallback = require_nonempty_string(fallback, "MethodResult.fallback_kind")
    grid = None if payload.get("grid") is None else validate_grid_envelope(payload.get("grid"))
    summary = (
        None
        if payload.get("grid_summary") is None
        else validate_grid_summary(payload.get("grid_summary"), grid=grid)
    )
    if (grid is None) != (summary is None):
        raise ValueError("MethodResult grid and grid_summary must both be present or null")
    _validate_method_null_matrix(
        kind=kind,
        status=status,
        reason=reason,
        fallback_kind=fallback,
        grid=grid,
        grid_summary=summary,
    )
    result = {
        "method_id": payload["method_id"],
        "method_kind": kind,
        "status": status,
        "reason": reason,
        "fallback_kind": fallback,
        "grid": grid,
        "grid_summary": summary,
        "deterministic_projection_sha256": require_sha256(
            payload.get("deterministic_projection_sha256"),
            "MethodResult.deterministic_projection_sha256",
        ),
    }
    expected_projection = payload_hash(result, "deterministic_projection_sha256")
    if result["deterministic_projection_sha256"] != expected_projection:
        raise ValueError("MethodResult deterministic_projection_sha256 mismatch")
    return result


def _validate_method_null_matrix(
    *,
    kind: str,
    status: str,
    reason: str | None,
    fallback_kind: str | None,
    grid: Mapping[str, Any] | None,
    grid_summary: Mapping[str, Any] | None,
) -> None:
    if kind == "candidate":
        if status == "accepted":
            if reason is not None or fallback_kind is not None:
                raise ValueError("candidate accepted branch requires null reason")
            if grid is None or grid["kind"] != "timing_v3":
                raise ValueError("candidate accepted branch requires Timing-v3 grid")
            return
        if status == "tagged_fallback":
            if reason not in CANDIDATE_FALLBACK_REASONS or fallback_kind != reason:
                raise ValueError("candidate fallback branch reason mismatch")
            if grid is not None or grid_summary is not None:
                raise ValueError("candidate fallback branch grid must be null")
            return
    elif kind == "baseline":
        if status == "accepted":
            if reason is not None or fallback_kind is not None:
                raise ValueError("baseline accepted branch requires null reason")
            if grid is None or grid["kind"] != "current_v2":
                raise ValueError("baseline accepted branch requires current-v2 grid")
            return
        if status == "unavailable":
            if reason not in BASELINE_UNAVAILABLE_REASONS or fallback_kind is not None:
                raise ValueError("baseline unavailable branch reason mismatch")
            if grid is not None or grid_summary is not None:
                raise ValueError("baseline unavailable branch grid must be null")
            return
    elif kind == "selected":
        if status == "accepted":
            if grid is None or grid["kind"] not in GRID_KINDS:
                raise ValueError("selected accepted branch requires a grid")
            return
        if status == "unavailable":
            if reason != "candidate_fallback_and_baseline_unavailable":
                raise ValueError("selected unavailable branch reason mismatch")
            if fallback_kind not in CANDIDATE_FALLBACK_REASONS:
                raise ValueError("selected unavailable fallback_kind mismatch")
            if grid is not None or grid_summary is not None:
                raise ValueError("selected unavailable branch grid must be null")
            return
    raise ValueError("MethodResult status branch is invalid")


def validate_methods(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "methods")
    validate_exact_fields(payload, METHODS_FIELDS, "methods")
    candidate = validate_method_result(payload.get("candidate"), expected_kind="candidate")
    baseline = validate_method_result(payload.get("baseline"), expected_kind="baseline")
    selected = validate_method_result(payload.get("selected"), expected_kind="selected")
    if candidate["status"] == "accepted":
        if selected["status"] != "accepted" or selected["grid"]["kind"] != "timing_v3":
            raise ValueError("selected product must use candidate grid when candidate accepted")
        if (
            selected["reason"] is not None
            or selected["fallback_kind"] is not None
            or selected["grid"] != candidate["grid"]
            or selected["grid_summary"] != candidate["grid_summary"]
        ):
            raise ValueError("selected candidate product branch mismatch")
    elif baseline["status"] == "accepted":
        if (
            selected["status"] != "accepted"
            or selected["reason"] != candidate["reason"]
            or selected["fallback_kind"] != candidate["reason"]
            or selected["grid"] != baseline["grid"]
            or selected["grid_summary"] != baseline["grid_summary"]
        ):
            raise ValueError("selected v2 fallback product branch mismatch")
    else:
        if (
            selected["status"] != "unavailable"
            or selected["reason"] != "candidate_fallback_and_baseline_unavailable"
            or selected["fallback_kind"] != candidate["reason"]
            or selected["grid"] is not None
            or selected["grid_summary"] is not None
        ):
            raise ValueError("selected unavailable branch mismatch")
    return {"candidate": candidate, "baseline": baseline, "selected": selected}


def make_resume_binding(
    *,
    row_input_fingerprint_sha256: str,
    reused: bool,
    prior_row_payload_sha256: str | None,
    validated_source_closure_fingerprint_sha256: str,
    validated_config_sha256: str,
    validated_cache_sha256: str,
    validated_selector_sha256: str,
) -> dict[str, Any]:
    return validate_resume_binding(
        {
            "row_input_fingerprint_sha256": row_input_fingerprint_sha256,
            "reused": reused,
            "prior_row_payload_sha256": prior_row_payload_sha256,
            "validated_source_closure_fingerprint_sha256": (
                validated_source_closure_fingerprint_sha256
            ),
            "validated_config_sha256": validated_config_sha256,
            "validated_cache_sha256": validated_cache_sha256,
            "validated_selector_sha256": validated_selector_sha256,
        }
    )


def validate_resume_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ResumeBinding")
    validate_exact_fields(payload, RESUME_BINDING_FIELDS, "ResumeBinding")
    reused = payload.get("reused")
    if not isinstance(reused, bool):
        raise ValueError("ResumeBinding.reused must be a bool")
    prior = payload.get("prior_row_payload_sha256")
    if reused:
        prior = require_sha256(prior, "ResumeBinding.prior_row_payload_sha256")
    elif prior is not None:
        raise ValueError("ResumeBinding prior row SHA must be null when not reused")
    return {
        "row_input_fingerprint_sha256": require_sha256(
            payload.get("row_input_fingerprint_sha256"),
            "ResumeBinding.row_input_fingerprint_sha256",
        ),
        "reused": reused,
        "prior_row_payload_sha256": prior,
        "validated_source_closure_fingerprint_sha256": require_sha256(
            payload.get("validated_source_closure_fingerprint_sha256"),
            "ResumeBinding.validated_source_closure_fingerprint_sha256",
        ),
        "validated_config_sha256": require_sha256(
            payload.get("validated_config_sha256"),
            "ResumeBinding.validated_config_sha256",
        ),
        "validated_cache_sha256": require_sha256(
            payload.get("validated_cache_sha256"),
            "ResumeBinding.validated_cache_sha256",
        ),
        "validated_selector_sha256": require_sha256(
            payload.get("validated_selector_sha256"),
            "ResumeBinding.validated_selector_sha256",
        ),
    }


def make_denominator_flags(**flags: bool) -> dict[str, Any]:
    return validate_denominator_flags(flags)


def validate_denominator_flags(
    payload: Mapping[str, Any],
    *,
    methods: Mapping[str, Any] | None = None,
    diagnostics_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_mapping(payload, "DenominatorFlags")
    validate_exact_fields(payload, DENOMINATOR_FLAGS_FIELDS, "DenominatorFlags")
    result: dict[str, bool] = {}
    for name in sorted(DENOMINATOR_FLAGS_FIELDS):
        value = payload.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"DenominatorFlags.{name} must be a bool")
        result[name] = value
    if result["pure_exp006_phase_matched"] and (
        not result["candidate_accepted"] or not result["current_v2_phase_matched"]
    ):
        raise ValueError("DenominatorFlags pure phase implication failed")
    if result["selected_safety_phase_matched"] and (
        not result["product_grid_available"] or not result["current_v2_phase_matched"]
    ):
        raise ValueError("DenominatorFlags selected safety implication failed")
    if result["product_grid_available"] and not (
        result["candidate_accepted"]
        or (result["candidate_tagged_fallback"] and result["baseline_accepted"])
    ):
        raise ValueError("DenominatorFlags product availability implication failed")
    if result["overlap_available"] and not result["candidate_accepted"]:
        raise ValueError("DenominatorFlags overlap implication failed")
    if result["candidate_accepted"] == result["candidate_tagged_fallback"]:
        raise ValueError("DenominatorFlags candidate branch must be exclusive")
    if methods is not None:
        method_result = validate_methods(methods)
        expected = {
            "candidate_accepted": method_result["candidate"]["status"] == "accepted",
            "candidate_tagged_fallback": (
                method_result["candidate"]["status"] == "tagged_fallback"
            ),
            "baseline_accepted": method_result["baseline"]["status"] == "accepted",
            "product_grid_available": method_result["selected"]["status"] == "accepted",
        }
        for name, value in expected.items():
            if result[name] != value:
                raise ValueError(f"DenominatorFlags {name} mismatch")
    if diagnostics_summary is not None:
        diagnostics = validate_bounded_diagnostics_summary(diagnostics_summary)
        overlap_available = diagnostics["overlap"]["available_record_count"] > 0
        if result["overlap_available"] != overlap_available:
            raise ValueError("DenominatorFlags overlap_available mismatch")
    return {name: result[name] for name in DENOMINATOR_FLAGS_FIELDS}


def make_overlap_summary(
    *,
    metric_version: str = "timing-v3-exp007-local-frontier-overlap-v1",
    record_count: int,
    available_record_count: int,
    unavailable_record_count: int,
    comparable_beat_count: int,
    p90_ms: int | float | None,
    p90_beats: int | float | None,
    residual_vector_sha256: str | None,
    records_sha256: str,
) -> dict[str, Any]:
    return validate_overlap_summary(
        {
            "metric_version": metric_version,
            "record_count": record_count,
            "available_record_count": available_record_count,
            "unavailable_record_count": unavailable_record_count,
            "comparable_beat_count": comparable_beat_count,
            "p90_ms": p90_ms,
            "p90_beats": p90_beats,
            "residual_vector_sha256": residual_vector_sha256,
            "records_sha256": records_sha256,
        }
    )


def validate_overlap_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "OverlapSummary")
    validate_exact_fields(payload, OVERLAP_SUMMARY_FIELDS, "OverlapSummary")
    result = {
        "metric_version": require_nonempty_string(
            payload.get("metric_version"),
            "OverlapSummary.metric_version",
        ),
        "record_count": require_nonnegative_int(
            payload.get("record_count"),
            "OverlapSummary.record_count",
        ),
        "available_record_count": require_nonnegative_int(
            payload.get("available_record_count"),
            "OverlapSummary.available_record_count",
        ),
        "unavailable_record_count": require_nonnegative_int(
            payload.get("unavailable_record_count"),
            "OverlapSummary.unavailable_record_count",
        ),
        "comparable_beat_count": require_nonnegative_int(
            payload.get("comparable_beat_count"),
            "OverlapSummary.comparable_beat_count",
        ),
        "records_sha256": require_sha256(
            payload.get("records_sha256"),
            "OverlapSummary.records_sha256",
        ),
    }
    if result["record_count"] != (
        result["available_record_count"] + result["unavailable_record_count"]
    ):
        raise ValueError("OverlapSummary record counts mismatch")
    if result["available_record_count"] > 0:
        p90_ms = require_finite_number(payload.get("p90_ms"), "OverlapSummary.p90_ms")
        p90_beats = require_finite_number(
            payload.get("p90_beats"),
            "OverlapSummary.p90_beats",
        )
        if p90_ms < 0 or p90_beats < 0:
            raise ValueError("OverlapSummary p90 values must be nonnegative")
        residual = require_sha256(
            payload.get("residual_vector_sha256"),
            "OverlapSummary.residual_vector_sha256",
        )
    else:
        if (
            payload.get("p90_ms") is not None
            or payload.get("p90_beats") is not None
            or payload.get("residual_vector_sha256") is not None
        ):
            raise ValueError("OverlapSummary unavailable aggregate fields must be null")
        if result["comparable_beat_count"] != 0:
            raise ValueError("OverlapSummary unavailable comparable count must be zero")
        p90_ms = None
        p90_beats = None
        residual = None
    return {
        **result,
        "p90_ms": p90_ms,
        "p90_beats": p90_beats,
        "residual_vector_sha256": residual,
    }


def make_bounded_diagnostics_summary(
    *,
    schedule_arm: str,
    result_reason: str | None,
    selected_section_count: int | None,
    block_count: int,
    candidate_fingerprint: str,
    grid_fingerprint: str,
    replay_fingerprint: str,
    transition_cache_size: int,
    actual_scored_edge_count: int,
    selected_terminal_objective: int | float | None,
    runner_up_terminal_objective: int | float | None,
    selected_runner_up_margin: int | float | None,
    block_resource_records_sha256: str,
    class_coverage_records_sha256: str,
    overlap: Mapping[str, Any],
    local_frontier_contract_version: str = "timing-v3-local-frontier-v1",
    bounded_contract_version: str = "timing-v3-exp007-boundary-pair-bounded-v1",
    objective_variant: str = "exp006_pair_conditioned_change_floor_1_4",
) -> dict[str, Any]:
    payload = {
        "local_frontier_contract_version": local_frontier_contract_version,
        "bounded_contract_version": bounded_contract_version,
        "objective_variant": objective_variant,
        "schedule_arm": schedule_arm,
        "result_reason": result_reason,
        "selected_section_count": selected_section_count,
        "block_count": block_count,
        "candidate_fingerprint": candidate_fingerprint,
        "grid_fingerprint": grid_fingerprint,
        "replay_fingerprint": replay_fingerprint,
        "transition_cache_size": transition_cache_size,
        "actual_scored_edge_count": actual_scored_edge_count,
        "selected_terminal_objective": selected_terminal_objective,
        "runner_up_terminal_objective": runner_up_terminal_objective,
        "selected_runner_up_margin": selected_runner_up_margin,
        "block_resource_records_sha256": block_resource_records_sha256,
        "class_coverage_records_sha256": class_coverage_records_sha256,
        "overlap": validate_overlap_summary(overlap),
    }
    payload["deterministic_fingerprint"] = payload_hash(
        payload,
        "deterministic_fingerprint",
    )
    return validate_bounded_diagnostics_summary(payload)


def validate_bounded_diagnostics_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "BoundedDiagnosticsSummary")
    validate_exact_fields(
        payload,
        BOUNDED_DIAGNOSTICS_SUMMARY_FIELDS,
        "BoundedDiagnosticsSummary",
    )
    arm = _require_schedule_arm(
        payload.get("schedule_arm"),
        "BoundedDiagnosticsSummary.schedule_arm",
    )
    result_reason = payload.get("result_reason")
    if result_reason is not None:
        result_reason = require_nonempty_string(
            result_reason,
            "BoundedDiagnosticsSummary.result_reason",
        )
    selected_section_count = payload.get("selected_section_count")
    if selected_section_count is not None:
        selected_section_count = require_positive_int(
            selected_section_count,
            "BoundedDiagnosticsSummary.selected_section_count",
        )
    selected_objective = payload.get("selected_terminal_objective")
    runner_up = payload.get("runner_up_terminal_objective")
    margin = payload.get("selected_runner_up_margin")
    if selected_objective is None:
        if runner_up is not None or margin is not None:
            raise ValueError("BoundedDiagnosticsSummary failed objective branch mismatch")
    else:
        selected_objective = require_finite_number(
            selected_objective,
            "BoundedDiagnosticsSummary.selected_terminal_objective",
        )
        if (runner_up is None) != (margin is None):
            raise ValueError("BoundedDiagnosticsSummary runner-up branch mismatch")
        if runner_up is not None:
            runner_up = require_finite_number(
                runner_up,
                "BoundedDiagnosticsSummary.runner_up_terminal_objective",
            )
            margin = require_finite_number(
                margin,
                "BoundedDiagnosticsSummary.selected_runner_up_margin",
            )
    result = {
        "local_frontier_contract_version": require_nonempty_string(
            payload.get("local_frontier_contract_version"),
            "BoundedDiagnosticsSummary.local_frontier_contract_version",
        ),
        "bounded_contract_version": require_nonempty_string(
            payload.get("bounded_contract_version"),
            "BoundedDiagnosticsSummary.bounded_contract_version",
        ),
        "objective_variant": require_nonempty_string(
            payload.get("objective_variant"),
            "BoundedDiagnosticsSummary.objective_variant",
        ),
        "schedule_arm": arm,
        "result_reason": result_reason,
        "selected_section_count": selected_section_count,
        "block_count": require_nonnegative_int(
            payload.get("block_count"),
            "BoundedDiagnosticsSummary.block_count",
        ),
        "candidate_fingerprint": require_sha256(
            payload.get("candidate_fingerprint"),
            "BoundedDiagnosticsSummary.candidate_fingerprint",
        ),
        "grid_fingerprint": require_sha256(
            payload.get("grid_fingerprint"),
            "BoundedDiagnosticsSummary.grid_fingerprint",
        ),
        "replay_fingerprint": require_sha256(
            payload.get("replay_fingerprint"),
            "BoundedDiagnosticsSummary.replay_fingerprint",
        ),
        "transition_cache_size": require_nonnegative_int(
            payload.get("transition_cache_size"),
            "BoundedDiagnosticsSummary.transition_cache_size",
        ),
        "actual_scored_edge_count": require_nonnegative_int(
            payload.get("actual_scored_edge_count"),
            "BoundedDiagnosticsSummary.actual_scored_edge_count",
        ),
        "selected_terminal_objective": selected_objective,
        "runner_up_terminal_objective": runner_up,
        "selected_runner_up_margin": margin,
        "block_resource_records_sha256": require_sha256(
            payload.get("block_resource_records_sha256"),
            "BoundedDiagnosticsSummary.block_resource_records_sha256",
        ),
        "class_coverage_records_sha256": require_sha256(
            payload.get("class_coverage_records_sha256"),
            "BoundedDiagnosticsSummary.class_coverage_records_sha256",
        ),
        "overlap": validate_overlap_summary(payload.get("overlap")),
        "deterministic_fingerprint": require_sha256(
            payload.get("deterministic_fingerprint"),
            "BoundedDiagnosticsSummary.deterministic_fingerprint",
        ),
    }
    if result["deterministic_fingerprint"] != payload_hash(
        result,
        "deterministic_fingerprint",
    ):
        raise ValueError("BoundedDiagnosticsSummary deterministic_fingerprint mismatch")
    return result


def make_runtime_telemetry(
    *,
    platform_rule: str,
    worker_pid: int,
    audio_arm_seconds: int | float,
    cache_load_seconds: int | float = 0.0,
    candidate_seconds: int | float = 0.0,
    current_v2_seconds: int | float = 0.0,
    exp006_seconds: int | float = 0.0,
    serialization_seconds: int | float = 0.0,
) -> dict[str, Any]:
    return validate_runtime_telemetry(
        {
            "platform_rule": platform_rule,
            "worker_pid": worker_pid,
            "audio_arm_seconds": audio_arm_seconds,
            "cache_load_seconds": cache_load_seconds,
            "candidate_seconds": candidate_seconds,
            "current_v2_seconds": current_v2_seconds,
            "exp006_seconds": exp006_seconds,
            "serialization_seconds": serialization_seconds,
        }
    )


def validate_runtime_telemetry(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RuntimeTelemetry")
    validate_exact_fields(payload, RUNTIME_TELEMETRY_FIELDS, "RuntimeTelemetry")
    rule = payload.get("platform_rule")
    if rule not in PLATFORM_RULES:
        raise ValueError("RuntimeTelemetry platform_rule is invalid")
    result = {
        "platform_rule": rule,
        "worker_pid": require_positive_int(
            payload.get("worker_pid"),
            "RuntimeTelemetry.worker_pid",
        ),
    }
    for name in (
        "audio_arm_seconds",
        "cache_load_seconds",
        "candidate_seconds",
        "current_v2_seconds",
        "exp006_seconds",
        "serialization_seconds",
    ):
        value = require_finite_number(payload.get(name), f"RuntimeTelemetry.{name}")
        if value < 0:
            raise ValueError(f"RuntimeTelemetry.{name} must be nonnegative")
        result[name] = payload.get(name)
    return result


def make_rss_telemetry(
    *,
    platform_rule: str,
    worker_pid: int,
    initial_ru_maxrss_bytes: int,
    final_ru_maxrss_bytes: int,
) -> dict[str, Any]:
    return validate_rss_telemetry(
        {
            "platform_rule": platform_rule,
            "worker_pid": worker_pid,
            "initial_ru_maxrss_bytes": initial_ru_maxrss_bytes,
            "final_ru_maxrss_bytes": final_ru_maxrss_bytes,
        }
    )


def validate_rss_telemetry(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RssTelemetry")
    validate_exact_fields(payload, RSS_TELEMETRY_FIELDS, "RssTelemetry")
    rule = payload.get("platform_rule")
    if rule not in PLATFORM_RULES:
        raise ValueError("RssTelemetry platform_rule is invalid")
    initial = require_nonnegative_int(
        payload.get("initial_ru_maxrss_bytes"),
        "RssTelemetry.initial_ru_maxrss_bytes",
    )
    final = require_nonnegative_int(
        payload.get("final_ru_maxrss_bytes"),
        "RssTelemetry.final_ru_maxrss_bytes",
    )
    if final < initial:
        raise ValueError("RssTelemetry final RSS must be >= initial")
    return {
        "platform_rule": rule,
        "worker_pid": require_positive_int(
            payload.get("worker_pid"),
            "RssTelemetry.worker_pid",
        ),
        "initial_ru_maxrss_bytes": initial,
        "final_ru_maxrss_bytes": final,
    }


def make_hard_guards(**overrides: bool) -> dict[str, Any]:
    payload = {
        "timed_out": False,
        "worker_alive": True,
        "cache_unchanged": True,
        "source_unchanged": True,
        "resume_valid": True,
        "schema_valid": True,
        "row_within_byte_cap": True,
        "rss_within_cap": True,
        "grid_seam_zero": True,
        "section_cap_valid": True,
        "diagnostics_caps_valid": True,
    }
    payload.update(overrides)
    return validate_hard_guards(payload, require_complete_row=True)


def validate_hard_guards(
    payload: Mapping[str, Any],
    *,
    require_complete_row: bool = False,
) -> dict[str, Any]:
    _require_mapping(payload, "HardGuards")
    validate_exact_fields(payload, HARD_GUARDS_FIELDS, "HardGuards")
    result: dict[str, bool] = {}
    for name in HARD_GUARDS_FIELDS:
        value = payload.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"HardGuards.{name} must be a bool")
        result[name] = value
    if require_complete_row:
        if result["timed_out"] is not False:
            raise ValueError("RowResult cannot persist a timed_out guard")
        for name, value in result.items():
            if name != "timed_out" and value is not True:
                raise ValueError("RowResult hard guards must all pass")
    return {name: result[name] for name in HARD_GUARDS_FIELDS}


def make_row_result(
    *,
    stage: str,
    schedule_arm: str,
    row_index: int,
    cache_audio_key: str,
    audio_group_key: str,
    identity_payload_sha256: str,
    cache_identity: Mapping[str, Any],
    source_closure_fingerprint_sha256: str,
    run_config_fingerprint_sha256: str,
    selector_manifest_sha256: str,
    input_manifest_sha256: str,
    resume: Mapping[str, Any],
    restricted_prediction: Mapping[str, Any],
    candidate_payload_schema: str,
    candidate_payload_byte_count: int,
    candidate_payload_field_set_sha256: str,
    candidate_payload_sha256: str,
    candidate_fingerprint: str,
    methods: Mapping[str, Any],
    denominator_flags: Mapping[str, Any],
    diagnostics_summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
    rss: Mapping[str, Any],
    hard_guards: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics = validate_bounded_diagnostics_summary(diagnostics_summary)
    payload = {
        "schema": ROW_RESULT_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": _require_stage(stage, "RowResult.stage"),
        "schema_descriptor_sha256": schema_descriptor_sha256(ROW_RESULT_SCHEMA),
        "schedule_arm": _require_schedule_arm(schedule_arm, "RowResult.schedule_arm"),
        "row_index": require_nonnegative_int(row_index, "RowResult.row_index"),
        "cache_audio_key": require_nonempty_string(
            cache_audio_key,
            "RowResult.cache_audio_key",
        ),
        "audio_group_key": require_nonempty_string(
            audio_group_key,
            "RowResult.audio_group_key",
        ),
        "identity_payload_sha256": require_sha256(
            identity_payload_sha256,
            "RowResult.identity_payload_sha256",
        ),
        "cache_identity": validate_cache_identity(cache_identity),
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "run_config_fingerprint_sha256": run_config_fingerprint_sha256,
        "selector_manifest_sha256": selector_manifest_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "resume": validate_resume_binding(resume),
        "restricted_prediction": validate_restricted_prediction(restricted_prediction),
        "candidate_payload_schema": candidate_payload_schema,
        "candidate_payload_byte_count": candidate_payload_byte_count,
        "candidate_payload_field_set_sha256": candidate_payload_field_set_sha256,
        "candidate_payload_sha256": candidate_payload_sha256,
        "candidate_fingerprint": candidate_fingerprint,
        "methods": validate_methods(methods),
        "denominator_flags": validate_denominator_flags(
            denominator_flags,
            methods=methods,
            diagnostics_summary=diagnostics,
        ),
        "diagnostics_summary": diagnostics,
        "diagnostics_summary_sha256": object_complete_sha256(diagnostics),
        "runtime": validate_runtime_telemetry(runtime),
        "rss": validate_rss_telemetry(rss),
        "hard_guards": validate_hard_guards(hard_guards, require_complete_row=True),
    }
    payload["deterministic_projection_sha256"] = canonical_json_sha256(
        _row_deterministic_projection(payload)
    )
    return validate_row_result(with_payload_hash(payload, "row_payload_sha256"))


def minimal_row_result(
    *,
    stage: str,
    schedule_arm: str,
    row_index: int,
    cache_audio_key: str,
    audio_group_key: str,
    identity_payload_sha256: str,
    source_closure_fingerprint_sha256: str,
    run_config_fingerprint_sha256: str,
    selector_manifest_sha256: str,
    input_manifest_sha256: str,
    candidate_payload_schema: str,
    candidate_payload_byte_count: int,
    candidate_payload_field_set_sha256: str,
    candidate_payload_sha256: str,
    candidate_fingerprint: str,
) -> dict[str, Any]:
    stage_value = _require_stage(stage, "RowResult.stage")
    arm = _require_schedule_arm(schedule_arm, "RowResult.schedule_arm")
    cache_sha = canonical_json_sha256(
        {
            "stage": stage_value,
            "row_index": row_index,
            "cache_audio_key": cache_audio_key,
        }
    )
    timing_grid = make_grid_envelope(
        kind="timing_v3",
        payload=make_timing_v3_grid_payload(
            origin_beat=0,
            origin_time_ms=0.0,
            coverage_start_ms=0.0,
            coverage_end_ms=120_000.0,
            sections=[{"start_beat": 0, "end_beat": 128, "bpm": 128.0}],
        ),
    )
    v2_grid = make_grid_envelope(
        kind="current_v2",
        payload=make_v2_grid_payload(
            segments=[{"offset_ms": 0.0, "beat_length_ms": 468.75, "meter": 4}]
        ),
    )
    timing_summary = make_grid_summary(grid=timing_grid)
    v2_summary = make_grid_summary(
        grid=v2_grid,
        coverage_start_ms=0.0,
        coverage_end_ms=120_000.0,
    )
    methods = {
        "candidate": make_method_result(
            method_kind="candidate",
            status="accepted",
            grid=timing_grid,
            grid_summary=timing_summary,
        ),
        "baseline": make_method_result(
            method_kind="baseline",
            status="accepted",
            grid=v2_grid,
            grid_summary=v2_summary,
        ),
        "selected": make_method_result(
            method_kind="selected",
            status="accepted",
            grid=timing_grid,
            grid_summary=timing_summary,
        ),
    }
    overlap = make_overlap_summary(
        record_count=1,
        available_record_count=1,
        unavailable_record_count=0,
        comparable_beat_count=8,
        p90_ms=0.0,
        p90_beats=0.0,
        residual_vector_sha256=canonical_json_sha256(
            {"row_index": row_index, "residual": "zero"}
        ),
        records_sha256=canonical_json_sha256(
            {"row_index": row_index, "records": "synthetic"}
        ),
    )
    diagnostics = make_bounded_diagnostics_summary(
        schedule_arm=arm,
        result_reason=None,
        selected_section_count=1,
        block_count=1,
        candidate_fingerprint=candidate_fingerprint,
        grid_fingerprint=timing_grid["grid_sha256"],
        replay_fingerprint=canonical_json_sha256(
            {"row_index": row_index, "replay": "synthetic"}
        ),
        transition_cache_size=1,
        actual_scored_edge_count=1,
        selected_terminal_objective=0.0,
        runner_up_terminal_objective=None,
        selected_runner_up_margin=None,
        block_resource_records_sha256=canonical_json_sha256([]),
        class_coverage_records_sha256=canonical_json_sha256([]),
        overlap=overlap,
    )
    cache_identity = make_cache_identity(
        relative_cache_path=f"cache/{cache_audio_key}.npz",
        exists=True,
        size_bytes=1,
        mtime_ns=0,
        inode=0,
        device=0,
        sha256=cache_sha,
        cache_config_sha256=cache_sha,
        audio_cache_key_sha256=canonical_json_sha256(cache_audio_key),
    )
    return make_row_result(
        stage=stage_value,
        schedule_arm=arm,
        row_index=row_index,
        cache_audio_key=cache_audio_key,
        audio_group_key=audio_group_key,
        identity_payload_sha256=identity_payload_sha256,
        cache_identity=cache_identity,
        source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        selector_manifest_sha256=selector_manifest_sha256,
        input_manifest_sha256=input_manifest_sha256,
        resume=make_resume_binding(
            row_input_fingerprint_sha256=canonical_json_sha256(
                {
                    "row_index": row_index,
                    "cache_audio_key": cache_audio_key,
                    "input_manifest_sha256": input_manifest_sha256,
                    "run_config_fingerprint_sha256": run_config_fingerprint_sha256,
                    "source_closure_fingerprint_sha256": (
                        source_closure_fingerprint_sha256
                    ),
                }
            ),
            reused=False,
            prior_row_payload_sha256=None,
            validated_source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
            validated_config_sha256=run_config_fingerprint_sha256,
            validated_cache_sha256=cache_sha,
            validated_selector_sha256=selector_manifest_sha256,
        ),
        restricted_prediction=make_restricted_prediction(
            frame_count=1024,
            frame_rate_hz=100.0,
            input_signal_sha256=canonical_json_sha256(
                {"row_index": row_index, "signal": "synthetic"}
            ),
            beat_bytes_sha256=canonical_json_sha256(
                {"row_index": row_index, "beat": "synthetic"}
            ),
            downbeat_bytes_sha256=canonical_json_sha256(
                {"row_index": row_index, "downbeat": "synthetic"}
            ),
        ),
        candidate_payload_schema=candidate_payload_schema,
        candidate_payload_byte_count=candidate_payload_byte_count,
        candidate_payload_field_set_sha256=candidate_payload_field_set_sha256,
        candidate_payload_sha256=candidate_payload_sha256,
        candidate_fingerprint=candidate_fingerprint,
        methods=methods,
        denominator_flags=make_denominator_flags(
            cache_valid=True,
            projection_evaluable=True,
            candidate_accepted=True,
            candidate_tagged_fallback=False,
            baseline_accepted=True,
            product_grid_available=True,
            overlap_available=True,
            current_v2_phase_matched=False,
            pure_exp006_phase_matched=False,
            selected_safety_phase_matched=False,
        ),
        diagnostics_summary=diagnostics,
        runtime=make_runtime_telemetry(
            platform_rule="macos_bytes",
            worker_pid=1,
            audio_arm_seconds=0.0,
        ),
        rss=make_rss_telemetry(
            platform_rule="macos_bytes",
            worker_pid=1,
            initial_ru_maxrss_bytes=0,
            final_ru_maxrss_bytes=0,
        ),
        hard_guards=make_hard_guards(),
    )


def validate_row_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RowResult")
    validate_exact_fields(payload, ROW_RESULT_FIELDS, "RowResult")
    if payload.get("schema") != ROW_RESULT_SCHEMA:
        raise ValueError("RowResult schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("RowResult experiment_id is invalid")
    stage = _require_stage(payload.get("stage"), "RowResult.stage")
    arm = _require_schedule_arm(payload.get("schedule_arm"), "RowResult.schedule_arm")
    _require_descriptor(payload, ROW_RESULT_SCHEMA, "RowResult")
    row_index = require_nonnegative_int(payload.get("row_index"), "RowResult.row_index")
    cache_audio_key = require_nonempty_string(
        payload.get("cache_audio_key"),
        "RowResult.cache_audio_key",
    )
    audio_group_key = require_nonempty_string(
        payload.get("audio_group_key"),
        "RowResult.audio_group_key",
    )
    cache_identity = validate_cache_identity(payload.get("cache_identity"))
    if cache_identity["exists"] is not True:
        raise ValueError("RowResult requires an existing cache identity")
    resume = validate_resume_binding(payload.get("resume"))
    restricted_prediction = validate_restricted_prediction(
        payload.get("restricted_prediction")
    )
    methods = validate_methods(payload.get("methods"))
    diagnostics = validate_bounded_diagnostics_summary(payload.get("diagnostics_summary"))
    denominator_flags = validate_denominator_flags(
        payload.get("denominator_flags"),
        methods=methods,
        diagnostics_summary=diagnostics,
    )
    runtime = validate_runtime_telemetry(payload.get("runtime"))
    rss = validate_rss_telemetry(payload.get("rss"))
    if runtime["platform_rule"] != rss["platform_rule"] or runtime["worker_pid"] != rss["worker_pid"]:
        raise ValueError("RowResult runtime/RSS telemetry mismatch")
    hard_guards = validate_hard_guards(
        payload.get("hard_guards"),
        require_complete_row=True,
    )
    for field_name in (
        "identity_payload_sha256",
        "source_closure_fingerprint_sha256",
        "run_config_fingerprint_sha256",
        "selector_manifest_sha256",
        "input_manifest_sha256",
        "candidate_payload_field_set_sha256",
        "candidate_payload_sha256",
        "candidate_fingerprint",
        "diagnostics_summary_sha256",
        "deterministic_projection_sha256",
    ):
        require_sha256(payload.get(field_name), f"RowResult.{field_name}")
    require_nonnegative_int(
        payload.get("candidate_payload_byte_count"),
        "RowResult.candidate_payload_byte_count",
    )
    if payload.get("candidate_payload_byte_count") >= EXP007_CANDIDATE_PAYLOAD_BYTE_CAP:
        raise ValueError("RowResult candidate payload at or above byte cap")
    if payload.get("candidate_payload_schema") != CANDIDATE_PAYLOAD_SCHEMA:
        raise ValueError("RowResult candidate_payload_schema is invalid")
    if payload.get("candidate_payload_field_set_sha256") != candidate_payload_field_set_sha256():
        raise ValueError("RowResult candidate_payload_field_set_sha256 mismatch")
    if payload.get("diagnostics_summary_sha256") != object_complete_sha256(diagnostics):
        raise ValueError("RowResult diagnostics_summary_sha256 mismatch")
    if resume["validated_source_closure_fingerprint_sha256"] != payload.get(
        "source_closure_fingerprint_sha256"
    ):
        raise ValueError("RowResult resume source closure mismatch")
    if resume["validated_config_sha256"] != payload.get("run_config_fingerprint_sha256"):
        raise ValueError("RowResult resume config mismatch")
    if resume["validated_selector_sha256"] != payload.get("selector_manifest_sha256"):
        raise ValueError("RowResult resume selector mismatch")
    if resume["validated_cache_sha256"] != cache_identity["sha256"]:
        raise ValueError("RowResult resume cache mismatch")
    result = {
        "schema": ROW_RESULT_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": stage,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "schedule_arm": arm,
        "row_index": row_index,
        "cache_audio_key": cache_audio_key,
        "audio_group_key": audio_group_key,
        "identity_payload_sha256": payload["identity_payload_sha256"],
        "cache_identity": cache_identity,
        "source_closure_fingerprint_sha256": payload[
            "source_closure_fingerprint_sha256"
        ],
        "run_config_fingerprint_sha256": payload["run_config_fingerprint_sha256"],
        "selector_manifest_sha256": payload["selector_manifest_sha256"],
        "input_manifest_sha256": payload["input_manifest_sha256"],
        "resume": resume,
        "restricted_prediction": restricted_prediction,
        "candidate_payload_schema": payload["candidate_payload_schema"],
        "candidate_payload_byte_count": payload["candidate_payload_byte_count"],
        "candidate_payload_field_set_sha256": payload[
            "candidate_payload_field_set_sha256"
        ],
        "candidate_payload_sha256": payload["candidate_payload_sha256"],
        "candidate_fingerprint": payload["candidate_fingerprint"],
        "methods": methods,
        "denominator_flags": denominator_flags,
        "diagnostics_summary": diagnostics,
        "diagnostics_summary_sha256": payload["diagnostics_summary_sha256"],
        "deterministic_projection_sha256": require_sha256(
            payload.get("deterministic_projection_sha256"),
            "RowResult.deterministic_projection_sha256",
        ),
        "runtime": runtime,
        "rss": rss,
        "hard_guards": hard_guards,
        "row_payload_sha256": require_sha256(
            payload.get("row_payload_sha256"),
            "RowResult.row_payload_sha256",
        ),
    }
    if result["deterministic_projection_sha256"] != canonical_json_sha256(
        _row_deterministic_projection(result)
    ):
        raise ValueError("RowResult deterministic_projection_sha256 mismatch")
    validate_payload_hash(result, "row_payload_sha256", context="RowResult")
    if len(canonical_json_bytes(result)) >= EXP007_ROW_JSON_BYTE_CAP:
        raise ValueError("RowResult at or above byte cap")
    return result


def make_arm_failure_record(
    *,
    stage: str,
    schedule_arm: str,
    run_config_fingerprint_sha256: str,
    source_closure_fingerprint_sha256: str,
    input_manifest_sha256: str,
    failure_kind: str,
    failure_stage: str,
    completed_prefix_rows: Sequence[Mapping[str, Any]],
    pending_identities: Sequence[Mapping[str, Any]],
    completed_reference_entry_payload_sha256s: Sequence[str] = (),
    causing_row_index: int | None = None,
    causing_cache_audio_key: str | None = None,
    causing_worker_slot: int | None = None,
    causing_worker_generation_nonce: str | None = None,
    causing_worker_pid: int | None = None,
    causing_dispatch_token: str | None = None,
    causing_worker_rss_bytes: int | None = None,
    prefix_candidate_fallback_count: int = 0,
    prefix_no_origin_or_path_count: int = 0,
    prefix_resource_cap_fallback_count: int = 0,
    worker_rss_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prefix = [validate_completed_row_ref(row) for row in completed_prefix_rows]
    pending = [validate_pending_identity_ref(row) for row in pending_identities]
    refs = [require_sha256(value, "completed_reference_entry_payload_sha256") for value in completed_reference_entry_payload_sha256s]
    payload = {
        "schema": ARM_FAILURE_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": _require_stage(stage, "ArmFailureRecord.stage"),
        "schema_descriptor_sha256": schema_descriptor_sha256(ARM_FAILURE_SCHEMA),
        "schedule_arm": _require_schedule_arm(schedule_arm, "ArmFailureRecord.schedule_arm"),
        "run_config_fingerprint_sha256": run_config_fingerprint_sha256,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "expected_row_count": expected_row_count_for_stage(stage),
        "failure_kind": failure_kind,
        "failure_stage": failure_stage,
        "causing_row_index": causing_row_index,
        "causing_cache_audio_key": causing_cache_audio_key,
        "causing_worker_slot": causing_worker_slot,
        "causing_worker_generation_nonce": causing_worker_generation_nonce,
        "causing_worker_pid": causing_worker_pid,
        "causing_dispatch_token": causing_dispatch_token,
        "causing_worker_rss_bytes": causing_worker_rss_bytes,
        "completed_prefix_count": len(prefix),
        "completed_prefix_rows": prefix,
        "completed_prefix_rows_sha256": canonical_json_sha256(prefix),
        "completed_reference_entry_count": len(refs),
        "completed_reference_entry_payload_sha256s": refs,
        "completed_reference_entry_payloads_sha256": canonical_json_sha256(refs),
        "pending_identity_count": len(pending),
        "pending_identities": pending,
        "pending_identities_sha256": canonical_json_sha256(pending),
        "prefix_candidate_fallback_count": prefix_candidate_fallback_count,
        "prefix_no_origin_or_path_count": prefix_no_origin_or_path_count,
        "prefix_resource_cap_fallback_count": prefix_resource_cap_fallback_count,
        "worker_rss_snapshot": worker_rss_snapshot
        if worker_rss_snapshot is not None
        else {
            "worker_slot_lifetime_bytes": [None, None, None, None],
            "observed_worker_max_bytes": None,
        },
    }
    payload["failure_deterministic_fingerprint_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "causing_worker_pid",
                "worker_rss_snapshot",
                "failure_deterministic_fingerprint_sha256",
                "full_payload_sha256",
            }
        }
    )
    payload["full_payload_sha256"] = payload_hash(payload, "full_payload_sha256")
    return validate_arm_failure_record(payload)


def validate_arm_failure_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ArmFailureRecord")
    validate_exact_fields(payload, ARM_FAILURE_FIELDS, "ArmFailureRecord")
    if payload.get("schema") != ARM_FAILURE_SCHEMA:
        raise ValueError("ArmFailureRecord schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("ArmFailureRecord experiment_id is invalid")
    stage = _require_stage(payload.get("stage"), "ArmFailureRecord.stage")
    arm = _require_schedule_arm(payload.get("schedule_arm"), "ArmFailureRecord.schedule_arm")
    _require_descriptor(payload, ARM_FAILURE_SCHEMA, "ArmFailureRecord")
    if payload.get("expected_row_count") != expected_row_count_for_stage(stage):
        raise ValueError("ArmFailureRecord expected_row_count is invalid")
    failure_kind = payload.get("failure_kind")
    failure_stage = payload.get("failure_stage")
    if failure_kind not in FAILURE_KINDS:
        raise ValueError("ArmFailureRecord failure_kind is invalid")
    if failure_stage not in FAILURE_STAGES:
        raise ValueError("ArmFailureRecord failure_stage is invalid")
    if failure_kind == "diagnostics_integrity_failure" and failure_stage != "diagnostics":
        raise ValueError("diagnostics_integrity_failure requires diagnostics stage")
    if failure_kind == "artifact_resource_cap" and failure_stage == "diagnostics":
        raise ValueError("artifact_resource_cap cannot represent diagnostics caps")
    for field_name in (
        "run_config_fingerprint_sha256",
        "source_closure_fingerprint_sha256",
        "input_manifest_sha256",
    ):
        require_sha256(payload.get(field_name), f"ArmFailureRecord.{field_name}")
    completed = [validate_completed_row_ref(row) for row in _require_sequence(payload.get("completed_prefix_rows"), "completed_prefix_rows")]
    pending = [validate_pending_identity_ref(row) for row in _require_sequence(payload.get("pending_identities"), "pending_identities")]
    if payload.get("completed_prefix_count") != len(completed):
        raise ValueError("ArmFailureRecord completed_prefix_count mismatch")
    if payload.get("pending_identity_count") != len(pending):
        raise ValueError("ArmFailureRecord pending_identity_count mismatch")
    if len(completed) + len(pending) != payload.get("expected_row_count"):
        raise ValueError("ArmFailureRecord completed/pending counts do not sum")
    if payload.get("completed_prefix_rows_sha256") != canonical_json_sha256(completed):
        raise ValueError("ArmFailureRecord completed_prefix_rows_sha256 mismatch")
    if payload.get("pending_identities_sha256") != canonical_json_sha256(pending):
        raise ValueError("ArmFailureRecord pending_identities_sha256 mismatch")
    _validate_failure_prefix_order(
        completed=completed,
        pending=pending,
        expected_count=payload["expected_row_count"],
        failure_kind=failure_kind,
        causing_row_index=payload.get("causing_row_index"),
        causing_cache_audio_key=payload.get("causing_cache_audio_key"),
    )
    refs = _require_sequence(
        payload.get("completed_reference_entry_payload_sha256s"),
        "completed_reference_entry_payload_sha256s",
    )
    for value in refs:
        require_sha256(value, "completed_reference_entry_payload_sha256s[]")
    if payload.get("completed_reference_entry_count") != len(refs):
        raise ValueError("ArmFailureRecord completed_reference_entry_count mismatch")
    if payload.get("completed_reference_entry_payloads_sha256") != canonical_json_sha256(list(refs)):
        raise ValueError("ArmFailureRecord completed_reference_entry_payloads_sha256 mismatch")
    is_reference_arm = stage == EXP007_REPAIR_STAGE or arm == "S30"
    completed_ref_shas = [
        row["candidate_reference_entry_payload_sha256"] for row in completed
    ]
    if is_reference_arm:
        if any(value is None for value in completed_ref_shas):
            raise ValueError("ArmFailureRecord reference-arm row reference cannot be null")
        if list(refs) != completed_ref_shas:
            raise ValueError(
                "ArmFailureRecord reference-arm completed reference count/list mismatch"
            )
    else:
        if refs or any(value is not None for value in completed_ref_shas):
            raise ValueError("ArmFailureRecord later schedule arms cannot carry references")
    _validate_worker_rss_snapshot(payload.get("worker_rss_snapshot"))
    for name in (
        "prefix_candidate_fallback_count",
        "prefix_no_origin_or_path_count",
        "prefix_resource_cap_fallback_count",
    ):
        require_nonnegative_int(payload.get(name), f"ArmFailureRecord.{name}")
    _validate_causing_fields(payload)
    require_sha256(
        payload.get("failure_deterministic_fingerprint_sha256"),
        "ArmFailureRecord.failure_deterministic_fingerprint_sha256",
    )
    deterministic = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "causing_worker_pid",
            "worker_rss_snapshot",
            "failure_deterministic_fingerprint_sha256",
            "full_payload_sha256",
        }
    }
    if payload.get("failure_deterministic_fingerprint_sha256") != canonical_json_sha256(
        deterministic
    ):
        raise ValueError("ArmFailureRecord deterministic fingerprint mismatch")
    validate_payload_hash(payload, "full_payload_sha256", context="ArmFailureRecord")
    return dict(payload)


def make_arm_stage_success(
    *,
    stage: str,
    schedule_arm: str,
    row_payloads_sha256: str,
    candidate_reference_manifest_sha256: str,
    stage_summary_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": ARM_STAGE_SUCCESS_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": _require_stage(stage, "ArmStageSuccess.stage"),
        "schema_descriptor_sha256": schema_descriptor_sha256(ARM_STAGE_SUCCESS_SCHEMA),
        "schedule_arm": _require_schedule_arm(schedule_arm, "ArmStageSuccess.schedule_arm"),
        "status": "success",
        "expected_row_count": expected_row_count_for_stage(stage),
        "row_count": expected_row_count_for_stage(stage),
        "row_payloads_sha256": row_payloads_sha256,
        "candidate_reference_manifest_sha256": candidate_reference_manifest_sha256,
        "stage_summary_sha256": stage_summary_sha256,
    }
    return validate_arm_stage_outcome(with_payload_hash(payload, "outcome_fingerprint_sha256"))


def make_arm_stage_hard_failure(
    arm_failure_record: Mapping[str, Any],
) -> dict[str, Any]:
    record = validate_arm_failure_record(arm_failure_record)
    payload = {
        "schema": ARM_STAGE_HARD_FAILURE_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": record["stage"],
        "schema_descriptor_sha256": schema_descriptor_sha256(ARM_STAGE_HARD_FAILURE_SCHEMA),
        "schedule_arm": record["schedule_arm"],
        "status": "hard_failure",
        "arm_failure_record": record,
        "arm_failure_record_sha256": object_complete_sha256(record),
    }
    return validate_arm_stage_outcome(with_payload_hash(payload, "outcome_fingerprint_sha256"))


def make_not_run_arm_record(
    *,
    schedule_arm: str,
    reason: str,
    causing_arm: str,
    causing_outcome_sha256: str,
    pending_identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pending = [validate_pending_identity_ref(row) for row in pending_identities]
    payload = {
        "schema": NOT_RUN_ARM_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": schema_descriptor_sha256(NOT_RUN_ARM_SCHEMA),
        "schedule_arm": _require_schedule_arm(schedule_arm, "NotRunArm.schedule_arm"),
        "status": "not_run_due_prior_hard_failure",
        "reason": reason,
        "causing_arm": _require_schedule_arm(causing_arm, "NotRunArm.causing_arm"),
        "causing_outcome_sha256": causing_outcome_sha256,
        "expected_row_count": 16,
        "pending_identity_count": len(pending),
        "pending_identities": pending,
        "pending_identities_sha256": canonical_json_sha256(pending),
    }
    return validate_arm_stage_outcome(with_payload_hash(payload, "record_fingerprint_sha256"))


def validate_arm_stage_outcome(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ArmStageOutcome")
    schema = payload.get("schema")
    if schema == ARM_STAGE_SUCCESS_SCHEMA:
        validate_exact_fields(payload, ARM_STAGE_SUCCESS_FIELDS, "ArmStageSuccess")
        _validate_common_outcome(payload, expected_status="success")
        stage = _require_stage(payload.get("stage"), "ArmStageSuccess.stage")
        expected_count = expected_row_count_for_stage(stage)
        if payload.get("expected_row_count") != expected_count:
            raise ValueError("ArmStageSuccess expected_row_count is invalid")
        if payload.get("row_count") != expected_count:
            raise ValueError("ArmStageSuccess row_count is invalid")
        for name in (
            "row_payloads_sha256",
            "candidate_reference_manifest_sha256",
            "stage_summary_sha256",
        ):
            require_sha256(payload.get(name), f"ArmStageSuccess.{name}")
        validate_payload_hash(payload, "outcome_fingerprint_sha256", context="ArmStageSuccess")
        return dict(payload)
    if schema == ARM_STAGE_HARD_FAILURE_SCHEMA:
        validate_exact_fields(payload, ARM_STAGE_HARD_FAILURE_FIELDS, "ArmStageHardFailure")
        _validate_common_outcome(payload, expected_status="hard_failure")
        record = validate_arm_failure_record(payload.get("arm_failure_record"))
        if payload.get("stage") != record["stage"] or payload.get("schedule_arm") != record["schedule_arm"]:
            raise ValueError("ArmStageHardFailure record common fields mismatch")
        if payload.get("arm_failure_record_sha256") != object_complete_sha256(record):
            raise ValueError("ArmStageHardFailure arm_failure_record_sha256 mismatch")
        validate_payload_hash(payload, "outcome_fingerprint_sha256", context="ArmStageHardFailure")
        return dict(payload)
    if schema == NOT_RUN_ARM_SCHEMA:
        validate_exact_fields(payload, NOT_RUN_ARM_FIELDS, "NotRunArmRecord")
        if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
            raise ValueError("NotRunArmRecord experiment_id is invalid")
        if payload.get("stage") != EXP007_SCHEDULE_STAGE:
            raise ValueError("NotRunArmRecord stage is invalid")
        if payload.get("status") != "not_run_due_prior_hard_failure":
            raise ValueError("NotRunArmRecord status is invalid")
        _require_descriptor(payload, NOT_RUN_ARM_SCHEMA, "NotRunArmRecord")
        _require_schedule_arm(payload.get("schedule_arm"), "NotRunArm.schedule_arm")
        causing_arm = _require_schedule_arm(payload.get("causing_arm"), "NotRunArm.causing_arm")
        if EXP007_EXECUTION_ORDER.index(causing_arm) >= EXP007_EXECUTION_ORDER.index(payload["schedule_arm"]):
            raise ValueError("NotRunArmRecord causing_arm must be earlier")
        if payload.get("reason") not in {"prior_arm_hard_failure", "schedule_deadline_already_crossed"}:
            raise ValueError("NotRunArmRecord reason is invalid")
        require_sha256(payload.get("causing_outcome_sha256"), "NotRunArm.causing_outcome_sha256")
        if payload.get("expected_row_count") != 16:
            raise ValueError("NotRunArmRecord expected_row_count is invalid")
        pending = [validate_pending_identity_ref(row) for row in _require_sequence(payload.get("pending_identities"), "pending_identities")]
        if payload.get("pending_identity_count") != len(pending) or len(pending) != 16:
            raise ValueError("NotRunArmRecord pending count is invalid")
        if payload.get("pending_identities_sha256") != canonical_json_sha256(pending):
            raise ValueError("NotRunArmRecord pending_identities_sha256 mismatch")
        validate_payload_hash(payload, "record_fingerprint_sha256", context="NotRunArmRecord")
        return dict(payload)
    raise ValueError("ArmStageOutcome variant is invalid")


def make_config_selection(
    *,
    arm_outcome_sha256_by_execution_order: Mapping[str, str],
    candidate_global_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    selector_manifest_sha256: str,
    overlap_common: Mapping[str, Any],
    section_common: Mapping[str, Any],
    source_decision: str,
    arm_order_values: Sequence[Mapping[str, Any]],
    selected_schedule_arm: str | None,
    selected_run_config_fingerprint_sha256: str | None,
) -> dict[str, Any]:
    winner_selected = source_decision == "positive"
    payload = {
        "schema": CONFIG_SELECTION_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": schema_descriptor_sha256(CONFIG_SELECTION_SCHEMA),
        "arm_outcome_sha256_by_execution_order": dict(arm_outcome_sha256_by_execution_order),
        "candidate_global_manifest_sha256": candidate_global_manifest_sha256,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "selector_manifest_sha256": selector_manifest_sha256,
        "overlap_common": dict(overlap_common),
        "section_common": dict(section_common),
        "source_decision": source_decision,
        "arm_order_values": [dict(value) for value in arm_order_values],
        "selected_schedule_arm": selected_schedule_arm,
        "selected_run_config_fingerprint_sha256": selected_run_config_fingerprint_sha256,
        "source_winner_selected_before_weak": winner_selected,
    }
    return validate_config_selection(with_payload_hash(payload, "selection_fingerprint_sha256"))


def validate_config_selection(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ConfigSelection")
    validate_exact_fields(payload, CONFIG_SELECTION_FIELDS, "ConfigSelection")
    if payload.get("schema") != CONFIG_SELECTION_SCHEMA:
        raise ValueError("ConfigSelection schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("ConfigSelection experiment_id is invalid")
    if payload.get("stage") != EXP007_SCHEDULE_STAGE:
        raise ValueError("ConfigSelection stage is invalid")
    _require_descriptor(payload, CONFIG_SELECTION_SCHEMA, "ConfigSelection")
    _validate_arm_outcome_sha_map(payload.get("arm_outcome_sha256_by_execution_order"))
    for name in (
        "candidate_global_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "selector_manifest_sha256",
    ):
        require_sha256(payload.get(name), f"ConfigSelection.{name}")
    overlap_common = validate_audio_set_binding(payload.get("overlap_common"))
    section_common = validate_audio_set_binding(payload.get("section_common"))
    decision = payload.get("source_decision")
    if decision not in {"positive", "ambiguous", "negative"}:
        raise ValueError("ConfigSelection source_decision is invalid")
    values = _require_sequence(payload.get("arm_order_values"), "arm_order_values")
    if len(values) != 4:
        raise ValueError("ConfigSelection must have four arm_order_values")
    common_ready = overlap_common["count"] >= 5 and section_common["count"] >= 8
    validated_values: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        validated_values.append(
            _validate_arm_order_value(
                value,
                expected_arm=EXP007_EXECUTION_ORDER[index],
                common_ready=common_ready,
                overlap_common_count=overlap_common["count"],
                section_common_count=section_common["count"],
            )
        )
    e0_values = [value for value in validated_values if value["e0_eligible"]]
    e1_values = [value for value in validated_values if value["e1_eligible"]]
    if len(e0_values) < 2 or not common_ready:
        derived_decision = "ambiguous"
    elif not e1_values:
        derived_decision = "negative"
    else:
        derived_decision = "positive"
    if decision != derived_decision:
        raise ValueError("ConfigSelection source_decision mismatch")
    selected_arm = payload.get("selected_schedule_arm")
    selected_run_config = payload.get("selected_run_config_fingerprint_sha256")
    selected_before_weak = payload.get("source_winner_selected_before_weak")
    if decision == "positive":
        selected_arm = _require_schedule_arm(
            selected_arm,
            "ConfigSelection.selected_schedule_arm",
        )
        require_sha256(
            selected_run_config,
            "ConfigSelection.selected_run_config_fingerprint_sha256",
        )
        if selected_before_weak is not True:
            raise ValueError("positive ConfigSelection must select before weak")
        winner = min(
            e1_values,
            key=lambda value: value["_order_tuple"],
        )
        if selected_arm != winner["schedule_arm"]:
            raise ValueError("ConfigSelection selected arm is not source winner")
    else:
        if selected_arm is not None or selected_run_config is not None:
            raise ValueError("non-positive ConfigSelection selected fields must be null")
        if selected_before_weak is not False:
            raise ValueError("non-positive ConfigSelection weak flag must be false")
    validate_payload_hash(payload, "selection_fingerprint_sha256", context="ConfigSelection")
    return dict(payload)


def make_four_arm_stage_summary(
    *,
    status: str,
    arm_outcome_sha256_by_execution_order: Mapping[str, str],
    candidate_global_manifest_sha256: str | None,
    source_selection_status: str,
    config_selection_sha256: str | None,
    failure_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": FOUR_ARM_STAGE_SUMMARY_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": schema_descriptor_sha256(
            FOUR_ARM_STAGE_SUMMARY_SCHEMA
        ),
        "status": status,
        "arm_outcome_sha256_by_execution_order": dict(arm_outcome_sha256_by_execution_order),
        "candidate_global_manifest_sha256": candidate_global_manifest_sha256,
        "failure_details": None if failure_details is None else dict(failure_details),
        "source_selection_status": source_selection_status,
        "config_selection_sha256": config_selection_sha256,
    }
    return validate_four_arm_stage_summary(
        with_payload_hash(payload, "summary_fingerprint_sha256")
    )


def validate_four_arm_stage_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "FourArmStageSummary")
    validate_exact_fields(payload, FOUR_ARM_STAGE_SUMMARY_FIELDS, "FourArmStageSummary")
    if payload.get("schema") != FOUR_ARM_STAGE_SUMMARY_SCHEMA:
        raise ValueError("FourArmStageSummary schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("FourArmStageSummary experiment_id is invalid")
    if payload.get("stage") != EXP007_SCHEDULE_STAGE:
        raise ValueError("FourArmStageSummary stage is invalid")
    _require_descriptor(payload, FOUR_ARM_STAGE_SUMMARY_SCHEMA, "FourArmStageSummary")
    _validate_arm_outcome_sha_map(payload.get("arm_outcome_sha256_by_execution_order"))
    status = payload.get("status")
    if status == "success":
        require_sha256(
            payload.get("candidate_global_manifest_sha256"),
            "FourArmStageSummary.candidate_global_manifest_sha256",
        )
        if payload.get("failure_details") is not None:
            raise ValueError("successful FourArmStageSummary failure_details must be null")
        if payload.get("source_selection_status") not in {"positive", "ambiguous", "negative"}:
            raise ValueError("FourArmStageSummary source_selection_status is invalid")
        require_sha256(
            payload.get("config_selection_sha256"),
            "FourArmStageSummary.config_selection_sha256",
        )
    elif status == "hard_failure":
        if payload.get("candidate_global_manifest_sha256") is not None:
            raise ValueError("hard-failure FourArmStageSummary candidate manifest must be null")
        _validate_four_arm_failure_details(payload.get("failure_details"))
        if payload.get("source_selection_status") != "not_run":
            raise ValueError("hard-failure FourArmStageSummary source selection must not run")
        if payload.get("config_selection_sha256") is not None:
            raise ValueError("hard-failure FourArmStageSummary config selection must be null")
    else:
        raise ValueError("FourArmStageSummary status is invalid")
    validate_payload_hash(payload, "summary_fingerprint_sha256", context="FourArmStageSummary")
    return dict(payload)


def make_repair80_input_binding(
    *,
    identity_source: Mapping[str, Any],
    label_source: Mapping[str, Any],
    four_arm_stage_summary_sha256: str,
    candidate_global_manifest_sha256: str,
    source_selection_sha256: str,
    schedule_weak_veto_outcome_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": REPAIR80_INPUT_BINDING_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_REPAIR_STAGE,
        "identity_source": validate_source_ref(identity_source),
        "label_source": validate_source_ref(label_source),
        "four_arm_stage_summary_sha256": four_arm_stage_summary_sha256,
        "candidate_global_manifest_sha256": candidate_global_manifest_sha256,
        "source_selection_sha256": source_selection_sha256,
        "schedule_weak_veto_outcome_sha256": schedule_weak_veto_outcome_sha256,
        "row_count": 80,
    }
    return validate_repair80_input_binding(
        with_payload_hash(payload, "binding_fingerprint_sha256")
    )


def validate_repair80_input_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "Repair80InputBinding")
    validate_exact_fields(payload, REPAIR80_INPUT_BINDING_FIELDS, "Repair80InputBinding")
    if payload.get("schema") != REPAIR80_INPUT_BINDING_SCHEMA:
        raise ValueError("Repair80InputBinding schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("Repair80InputBinding experiment_id is invalid")
    if payload.get("stage") != EXP007_REPAIR_STAGE:
        raise ValueError("Repair80InputBinding stage is invalid")
    identity_source = validate_source_ref(payload.get("identity_source"))
    label_source = validate_source_ref(payload.get("label_source"))
    if identity_source["artifact_schema"] != SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA:
        raise ValueError("Repair80InputBinding identity_source artifact_schema is invalid")
    if label_source["artifact_schema"] != SOURCE_LABELS_ARTIFACT_SCHEMA:
        raise ValueError("Repair80InputBinding label_source artifact_schema is invalid")
    if identity_source["row_count"] != 80 or label_source["row_count"] != 80:
        raise ValueError("Repair80InputBinding sources must have row_count 80")
    if payload.get("row_count") != 80:
        raise ValueError("Repair80InputBinding row_count is invalid")
    for name in (
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_selection_sha256",
        "schedule_weak_veto_outcome_sha256",
    ):
        require_sha256(payload.get(name), f"Repair80InputBinding.{name}")
    validate_payload_hash(
        payload,
        "binding_fingerprint_sha256",
        context="Repair80InputBinding",
    )
    return dict(payload)


def validate_run_config_for_execution(
    payload: Mapping[str, Any],
    *,
    source_closure: Mapping[str, Any] | None = None,
    source_repo_root: str | Path | None = None,
    repair80_input_binding: Mapping[str, Any] | None = None,
    repair80_identity_source_artifact: bytes | None = None,
    repair80_label_source_artifact: bytes | None = None,
    repair80_identity_rows: Sequence[Mapping[str, Any]] | None = None,
    repair80_label_rows: Sequence[Mapping[str, Any]] | None = None,
    schedule_weak_veto_outcome: Mapping[str, Any] | None = None,
    four_arm_stage_summary: Mapping[str, Any] | None = None,
    config_selection: Mapping[str, Any] | None = None,
    candidate_global_manifest: Mapping[str, Any] | None = None,
    candidate_reference_manifest: Mapping[str, Any] | None = None,
    artifact_root: str | Path | None = None,
    run_configs_by_arm: Mapping[str, Mapping[str, Any]] | None = None,
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    arm_stage_outcomes_by_execution_order: Mapping[str, Mapping[str, Any]] | None = None,
    source_arm_stage_summaries_by_execution_order: Mapping[str, Mapping[str, Any]] | None = None,
    authoritative_candidate_global_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    config = validate_run_config(payload)
    _validate_run_config_source_closure(
        config,
        source_closure=source_closure,
        source_repo_root=source_repo_root,
    )
    if config["stage"] == EXP007_SCHEDULE_STAGE:
        return config
    missing = [
        name
        for name, value in (
            ("repair80_input_binding", repair80_input_binding),
            ("repair80_identity_source_artifact", repair80_identity_source_artifact),
            ("repair80_label_source_artifact", repair80_label_source_artifact),
            ("repair80_identity_rows", repair80_identity_rows),
            ("repair80_label_rows", repair80_label_rows),
            ("schedule_weak_veto_outcome", schedule_weak_veto_outcome),
            ("four_arm_stage_summary", four_arm_stage_summary),
            ("config_selection", config_selection),
            ("candidate_global_manifest", candidate_global_manifest),
            ("candidate_reference_manifest", candidate_reference_manifest),
            ("artifact_root", artifact_root),
            ("run_configs_by_arm", run_configs_by_arm),
            ("arm_rows_by_arm", arm_rows_by_arm),
            ("candidate_payloads_by_arm", candidate_payloads_by_arm),
            (
                "arm_stage_outcomes_by_execution_order",
                arm_stage_outcomes_by_execution_order,
            ),
            (
                "source_arm_stage_summaries_by_execution_order",
                source_arm_stage_summaries_by_execution_order,
            ),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "repair80 RunConfig execution validation requires dependency objects: "
            + ", ".join(missing)
        )
    binding = validate_repair80_input_binding_for_execution(
        repair80_input_binding,
        repair80_identity_source_artifact=repair80_identity_source_artifact,
        repair80_label_source_artifact=repair80_label_source_artifact,
        repair80_identity_rows=repair80_identity_rows,
        repair80_label_rows=repair80_label_rows,
        schedule_weak_veto_outcome=schedule_weak_veto_outcome,
        four_arm_stage_summary=four_arm_stage_summary,
        config_selection=config_selection,
        candidate_global_manifest=candidate_global_manifest,
        candidate_reference_manifest=candidate_reference_manifest,
        artifact_root=artifact_root,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
        arm_stage_outcomes_by_execution_order=arm_stage_outcomes_by_execution_order,
        source_arm_stage_summaries_by_execution_order=(
            source_arm_stage_summaries_by_execution_order
        ),
        authoritative_candidate_global_validator=(
            authoritative_candidate_global_validator
        ),
    )
    if config["input_manifest_sha256"] != binding["binding_fingerprint_sha256"]:
        raise ValueError("repair80 RunConfig input manifest does not match binding")
    if (
        config["schedule_weak_veto_outcome_sha256"]
        != binding["schedule_weak_veto_outcome_sha256"]
    ):
        raise ValueError("repair80 RunConfig weak outcome SHA mismatch")
    if config["selector_manifest_sha256"] != config_selection["selector_manifest_sha256"]:
        raise ValueError("repair80 RunConfig selector SHA mismatch")
    if (
        config["source_closure_fingerprint_sha256"]
        != config_selection["source_closure_fingerprint_sha256"]
    ):
        raise ValueError("repair80 RunConfig source closure mismatch")
    if config["schedule_arm"] != config_selection["selected_schedule_arm"]:
        raise ValueError("repair80 RunConfig schedule arm must be selected winner")
    return config


def _validate_run_config_source_closure(
    config: Mapping[str, Any],
    *,
    source_closure: Mapping[str, Any] | None,
    source_repo_root: str | Path | None,
) -> dict[str, Any]:
    if source_closure is None or source_repo_root is None:
        raise ValueError(
            "RunConfig execution validation requires source_closure and "
            "source_repo_root"
        )
    closure = validate_source_closure_for_repo(source_closure, source_repo_root)
    if (
        config["source_closure_fingerprint_sha256"]
        != closure["source_closure_fingerprint_sha256"]
    ):
        raise ValueError("RunConfig source closure fingerprint mismatch")
    return closure


def validate_repair80_input_binding_for_execution(
    payload: Mapping[str, Any],
    *,
    repair80_identity_source_artifact: bytes | None = None,
    repair80_label_source_artifact: bytes | None = None,
    repair80_identity_rows: Sequence[Mapping[str, Any]] | None = None,
    repair80_label_rows: Sequence[Mapping[str, Any]] | None = None,
    schedule_weak_veto_outcome: Mapping[str, Any],
    four_arm_stage_summary: Mapping[str, Any],
    config_selection: Mapping[str, Any],
    candidate_global_manifest: Mapping[str, Any],
    candidate_reference_manifest: Mapping[str, Any] | None = None,
    artifact_root: str | Path | None = None,
    run_configs_by_arm: Mapping[str, Mapping[str, Any]] | None = None,
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    arm_stage_outcomes_by_execution_order: Mapping[str, Mapping[str, Any]],
    source_arm_stage_summaries_by_execution_order: Mapping[str, Mapping[str, Any]],
    authoritative_candidate_global_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    binding = validate_repair80_input_binding(payload)
    missing = [
        name
        for name, value in (
            ("repair80_identity_source_artifact", repair80_identity_source_artifact),
            ("repair80_label_source_artifact", repair80_label_source_artifact),
            ("repair80_identity_rows", repair80_identity_rows),
            ("repair80_label_rows", repair80_label_rows),
            ("candidate_reference_manifest", candidate_reference_manifest),
            ("artifact_root", artifact_root),
            ("run_configs_by_arm", run_configs_by_arm),
            ("arm_rows_by_arm", arm_rows_by_arm),
            ("candidate_payloads_by_arm", candidate_payloads_by_arm),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Repair80InputBinding candidate global authoritative validation "
            "requires dependency objects: "
            + ", ".join(missing)
        )
    validate_repair80_identity_sources_for_execution(
        repair80_identity_source_artifact=repair80_identity_source_artifact,
        repair80_label_source_artifact=repair80_label_source_artifact,
        repair80_identity_rows=repair80_identity_rows,
        repair80_label_rows=repair80_label_rows,
        identity_source=binding["identity_source"],
        label_source=binding["label_source"],
    )
    summary = validate_four_arm_stage_summary(four_arm_stage_summary)
    if summary["status"] != "success":
        raise ValueError("Repair80InputBinding requires a successful FourArm summary")
    if summary["source_selection_status"] != "positive":
        raise ValueError("Repair80InputBinding requires a positive source selection")
    summary_sha = object_complete_sha256(summary)
    if binding["four_arm_stage_summary_sha256"] != summary_sha:
        raise ValueError("Repair80InputBinding four-arm summary SHA mismatch")
    selection = validate_config_selection(config_selection)
    selection_sha = object_complete_sha256(selection)
    if binding["source_selection_sha256"] != selection_sha:
        raise ValueError("Repair80InputBinding source selection SHA mismatch")
    if summary["config_selection_sha256"] != selection_sha:
        raise ValueError("Repair80InputBinding summary/config selection mismatch")
    if selection["source_decision"] != "positive":
        raise ValueError("Repair80InputBinding requires positive ConfigSelection")
    global_manifest = _validate_candidate_global_manifest_authoritatively(
        candidate_global_manifest,
        candidate_reference_manifest=candidate_reference_manifest,
        artifact_root=artifact_root,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
        authoritative_candidate_global_validator=(
            authoritative_candidate_global_validator
        ),
    )
    global_sha = object_complete_sha256(global_manifest)
    if binding["candidate_global_manifest_sha256"] != global_sha:
        raise ValueError("Repair80InputBinding candidate global manifest SHA mismatch")
    if summary["candidate_global_manifest_sha256"] != global_sha:
        raise ValueError("Repair80InputBinding summary/global manifest mismatch")
    if selection["candidate_global_manifest_sha256"] != global_sha:
        raise ValueError("Repair80InputBinding selection/global manifest mismatch")
    if global_manifest["selector_manifest_sha256"] != selection["selector_manifest_sha256"]:
        raise ValueError("Repair80InputBinding global selector SHA mismatch")
    if (
        global_manifest["source_closure_fingerprint_sha256"]
        != selection["source_closure_fingerprint_sha256"]
    ):
        raise ValueError("Repair80InputBinding global source closure mismatch")
    _validate_four_success_outcome_dependencies(
        arm_stage_outcomes_by_execution_order,
        summary=summary,
        selection=selection,
        source_arm_stage_summaries_by_execution_order=(
            source_arm_stage_summaries_by_execution_order
        ),
    )
    _validate_config_selection_against_source_summaries(
        selection,
        source_arm_stage_summaries_by_execution_order,
    )
    _validate_config_selection_against_full_rows(
        selection,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_global_manifest=global_manifest,
        arm_stage_outcomes_by_execution_order=arm_stage_outcomes_by_execution_order,
        source_arm_stage_summaries_by_execution_order=(
            source_arm_stage_summaries_by_execution_order
        ),
    )
    weak_outcome = validate_schedule_weak_veto_outcome(schedule_weak_veto_outcome)
    weak_sha = object_complete_sha256(weak_outcome)
    if binding["schedule_weak_veto_outcome_sha256"] != weak_sha:
        raise ValueError("Repair80InputBinding weak outcome SHA mismatch")
    if weak_outcome["status"] != "success":
        raise ValueError("Repair80InputBinding requires weak success outcome")
    weak_summary = weak_outcome["summary"]
    if weak_summary["decision"] != "pass" or weak_summary["action"] != "authorize_repair80":
        raise ValueError("Repair80InputBinding weak outcome must pass")
    if weak_summary["schedule_arm"] != selection["selected_schedule_arm"]:
        raise ValueError("Repair80InputBinding weak summary arm mismatch")
    _validate_schedule_weak_summary_against_selected_rows(
        weak_summary,
        selection=selection,
        arm_rows_by_arm=arm_rows_by_arm,
    )
    if weak_summary["four_arm_stage_summary_sha256"] != summary_sha:
        raise ValueError("Repair80InputBinding weak summary four-arm mismatch")
    if weak_summary["candidate_global_manifest_sha256"] != global_sha:
        raise ValueError("Repair80InputBinding weak summary global mismatch")
    if weak_summary["source_selection_sha256"] != selection_sha:
        raise ValueError("Repair80InputBinding weak summary selection mismatch")
    return binding


def _validate_candidate_global_manifest_authoritatively(
    payload: Mapping[str, Any],
    *,
    candidate_reference_manifest: Mapping[str, Any],
    artifact_root: str | Path,
    run_configs_by_arm: Mapping[str, Mapping[str, Any]],
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    authoritative_candidate_global_validator: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    shape = validate_candidate_global_manifest_non_authoritative_shape(payload)
    default_validated = _default_authoritative_candidate_global_validator(
        payload,
        reference_manifest=candidate_reference_manifest,
        root=artifact_root,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
    )
    try:
        authoritative = validate_candidate_global_manifest_non_authoritative_shape(
            default_validated
        )
    except ValueError as exc:
        raise ValueError(
            "authoritative candidate global validator must return a "
            "CandidateGlobalManifest object"
        ) from exc
    if authoritative != shape:
        raise ValueError("default candidate global validator returned divergent CandidateGlobalManifest")
    if authoritative_candidate_global_validator is not None:
        if not callable(authoritative_candidate_global_validator):
            raise ValueError("authoritative candidate global validator must be callable")
        extra = authoritative_candidate_global_validator(
            payload,
            reference_manifest=candidate_reference_manifest,
            root=artifact_root,
            run_configs_by_arm=run_configs_by_arm,
            arm_rows_by_arm=arm_rows_by_arm,
            candidate_payloads_by_arm=candidate_payloads_by_arm,
        )
        try:
            extra_authoritative = validate_candidate_global_manifest_non_authoritative_shape(
                extra
            )
        except ValueError as exc:
            raise ValueError(
                "authoritative candidate global validator must return a "
                "CandidateGlobalManifest object"
            ) from exc
        if extra_authoritative != authoritative:
            raise ValueError(
                "authoritative candidate global validator returned divergent "
                "CandidateGlobalManifest"
            )
    return authoritative


def _default_authoritative_candidate_global_validator(
    payload: Mapping[str, Any],
    *,
    reference_manifest: Mapping[str, Any],
    root: str | Path,
    run_configs_by_arm: Mapping[str, Mapping[str, Any]],
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    artifacts = importlib.import_module(
        "pulsefield_model.timing.evaluation.exp007_artifacts"
    )
    return artifacts.validate_candidate_global_manifest(
        payload,
        reference_manifest=reference_manifest,
        root=root,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
    )


def validate_candidate_global_manifest_non_authoritative_shape(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require_mapping(payload, "CandidateGlobalManifest")
    validate_exact_fields(
        payload,
        CANDIDATE_GLOBAL_MANIFEST_FIELDS,
        "CandidateGlobalManifest",
    )
    if payload.get("schema") != CANDIDATE_GLOBAL_MANIFEST_SCHEMA:
        raise ValueError("CandidateGlobalManifest schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("CandidateGlobalManifest experiment_id is invalid")
    if payload.get("stage") != EXP007_SCHEDULE_STAGE:
        raise ValueError("CandidateGlobalManifest stage is invalid")
    _require_descriptor(payload, CANDIDATE_GLOBAL_MANIFEST_SCHEMA, "CandidateGlobalManifest")
    selector_sha = require_sha256(
        payload.get("selector_manifest_sha256"),
        "CandidateGlobalManifest.selector_manifest_sha256",
    )
    source_sha = require_sha256(
        payload.get("source_closure_fingerprint_sha256"),
        "CandidateGlobalManifest.source_closure_fingerprint_sha256",
    )
    reference_sha = require_sha256(
        payload.get("candidate_reference_manifest_sha256"),
        "CandidateGlobalManifest.candidate_reference_manifest_sha256",
    )
    if payload.get("row_count") != 16:
        raise ValueError("CandidateGlobalManifest row_count is invalid")
    raw_entries = _require_sequence(payload.get("entries"), "CandidateGlobalManifest.entries")
    if len(raw_entries) != 16:
        raise ValueError("CandidateGlobalManifest entries length mismatch")
    entries: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for index, entry in enumerate(raw_entries):
        validated = validate_candidate_global_entry(entry, expected_row_index=index)
        if validated["cache_audio_key"] in seen_keys:
            raise ValueError("CandidateGlobalManifest duplicate cache key")
        seen_keys.add(validated["cache_audio_key"])
        entries.append(validated)
    if payload.get("ordered_entries_sha256") != canonical_json_sha256(entries):
        raise ValueError("CandidateGlobalManifest ordered_entries_sha256 mismatch")
    result = {
        "schema": CANDIDATE_GLOBAL_MANIFEST_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "selector_manifest_sha256": selector_sha,
        "source_closure_fingerprint_sha256": source_sha,
        "candidate_reference_manifest_sha256": reference_sha,
        "row_count": 16,
        "entries": entries,
        "ordered_entries_sha256": payload["ordered_entries_sha256"],
        "manifest_fingerprint_sha256": require_sha256(
            payload.get("manifest_fingerprint_sha256"),
            "CandidateGlobalManifest.manifest_fingerprint_sha256",
        ),
    }
    validate_payload_hash(
        result,
        "manifest_fingerprint_sha256",
        context="CandidateGlobalManifest",
    )
    if len(canonical_json_bytes(result)) >= EXP007_CANDIDATE_GLOBAL_MANIFEST_BYTE_CAP:
        raise ValueError("CandidateGlobalManifest at or above byte cap")
    return result


def validate_candidate_global_entry(
    payload: Mapping[str, Any],
    *,
    expected_row_index: int | None = None,
) -> dict[str, Any]:
    _require_mapping(payload, "CandidateGlobalEntry")
    validate_exact_fields(payload, CANDIDATE_GLOBAL_ENTRY_FIELDS, "CandidateGlobalEntry")
    row_index = require_nonnegative_int(
        payload.get("row_index"),
        "CandidateGlobalEntry.row_index",
    )
    if expected_row_index is not None and row_index != expected_row_index:
        raise ValueError("CandidateGlobalEntry row_index order mismatch")
    arm_map = validate_arm_row_sha_map(payload.get("arm_row_payload_sha256"))
    return {
        "row_index": row_index,
        "cache_audio_key": require_nonempty_string(
            payload.get("cache_audio_key"),
            "CandidateGlobalEntry.cache_audio_key",
        ),
        "audio_group_key": require_nonempty_string(
            payload.get("audio_group_key"),
            "CandidateGlobalEntry.audio_group_key",
        ),
        "input_signal_sha256": require_sha256(
            payload.get("input_signal_sha256"),
            "CandidateGlobalEntry.input_signal_sha256",
        ),
        "candidate_payload_schema": _require_candidate_payload_schema(
            payload.get("candidate_payload_schema"),
            "CandidateGlobalEntry.candidate_payload_schema",
        ),
        "candidate_payload_field_set_sha256": require_sha256(
            payload.get("candidate_payload_field_set_sha256"),
            "CandidateGlobalEntry.candidate_payload_field_set_sha256",
        ),
        "candidate_payload_byte_count": require_nonnegative_int(
            payload.get("candidate_payload_byte_count"),
            "CandidateGlobalEntry.candidate_payload_byte_count",
        ),
        "candidate_payload_sha256": require_sha256(
            payload.get("candidate_payload_sha256"),
            "CandidateGlobalEntry.candidate_payload_sha256",
        ),
        "candidate_fingerprint": require_sha256(
            payload.get("candidate_fingerprint"),
            "CandidateGlobalEntry.candidate_fingerprint",
        ),
        "candidate_reference_entry_payload_sha256": require_sha256(
            payload.get("candidate_reference_entry_payload_sha256"),
            "CandidateGlobalEntry.candidate_reference_entry_payload_sha256",
        ),
        "arm_row_payload_sha256": arm_map,
    }


def validate_arm_row_sha_map(payload: Mapping[str, Any]) -> dict[str, str]:
    _require_mapping(payload, "ArmRowShaMap")
    validate_exact_fields(payload, ARM_ROW_SHA_MAP_FIELDS, "ArmRowShaMap")
    return {
        arm: require_sha256(payload.get(arm), f"ArmRowShaMap.{arm}")
        for arm in EXP007_EXECUTION_ORDER
    }


def _validate_four_success_outcome_dependencies(
    payload: Mapping[str, Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_arm_stage_summaries_by_execution_order: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_mapping(payload, "arm_stage_outcomes_by_execution_order")
    validate_exact_fields(
        payload,
        ARM_OUTCOME_SHA_MAP_FIELDS,
        "arm_stage_outcomes_by_execution_order",
    )
    _require_mapping(
        source_arm_stage_summaries_by_execution_order,
        "source_arm_stage_summaries_by_execution_order",
    )
    validate_exact_fields(
        source_arm_stage_summaries_by_execution_order,
        ARM_OUTCOME_SHA_MAP_FIELDS,
        "source_arm_stage_summaries_by_execution_order",
    )
    hashes: list[str] = []
    for arm in EXP007_EXECUTION_ORDER:
        outcome = validate_arm_stage_outcome(payload[arm])
        if outcome["status"] != "success":
            raise ValueError("FourArm dependency outcome must be success")
        if outcome["stage"] != EXP007_SCHEDULE_STAGE or outcome["schedule_arm"] != arm:
            raise ValueError("FourArm dependency arm/stage mismatch")
        source_summary = validate_source_arm_stage_summary(
            source_arm_stage_summaries_by_execution_order[arm]
        )
        if source_summary["schedule_arm"] != arm:
            raise ValueError("SourceArmStageSummary arm mismatch")
        if outcome["stage_summary_sha256"] != object_complete_sha256(source_summary):
            raise ValueError("ArmStageSuccess source summary SHA mismatch")
        if outcome["row_payloads_sha256"] != source_summary["row_payloads_sha256"]:
            raise ValueError("ArmStageSuccess row payload hash mismatch")
        outcome_sha = object_complete_sha256(outcome)
        if summary["arm_outcome_sha256_by_execution_order"][arm] != outcome_sha:
            raise ValueError("FourArm summary outcome SHA mismatch")
        if selection["arm_outcome_sha256_by_execution_order"][arm] != outcome_sha:
            raise ValueError("ConfigSelection outcome SHA mismatch")
        hashes.append(outcome_sha)
    if len(set(hashes)) != len(hashes):
        raise ValueError("FourArm dependency outcome hashes must be unique")


def _validate_config_selection_against_source_summaries(
    selection: Mapping[str, Any],
    source_arm_stage_summaries_by_execution_order: Mapping[str, Mapping[str, Any]],
) -> None:
    common_ready = (
        selection["overlap_common"]["count"] >= 5
        and selection["section_common"]["count"] >= 8
    )
    summaries = {
        arm: validate_source_arm_stage_summary(source_arm_stage_summaries_by_execution_order[arm])
        for arm in EXP007_EXECUTION_ORDER
    }
    for index, arm in enumerate(EXP007_EXECUTION_ORDER):
        value = selection["arm_order_values"][index]
        summary = summaries[arm]
        denominators = summary["denominators"]
        gates = summary["gates"]
        if (
            value["candidate_fallback_count"]
            != denominators["candidate_fallback_audio"]["count"]
        ):
            raise ValueError("ConfigSelection candidate_fallback_count summary mismatch")
        if (
            value["no_origin_or_path_count"]
            != denominators["no_origin_or_path_audio"]["count"]
        ):
            raise ValueError("ConfigSelection no_origin_or_path_count summary mismatch")
        if value["p90_runtime"] != gates["runtime_seconds"]["p90"]:
            raise ValueError("ConfigSelection p90_runtime summary mismatch")
        if value["max_worker_rss"] != summary["rss_summary"]["arm_max_worker_bytes"]:
            raise ValueError("ConfigSelection max_worker_rss summary mismatch")
        derived_reasons = _source_summary_elimination_reasons(
            summary,
            selection_value=value,
            common_ready=common_ready,
            overlap_common_count=selection["overlap_common"]["count"],
            section_common_count=selection["section_common"]["count"],
        )
        if value["elimination_reasons"] != derived_reasons:
            raise ValueError("ConfigSelection elimination reasons summary mismatch")


def _validate_config_selection_against_full_rows(
    selection: Mapping[str, Any],
    *,
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_global_manifest: Mapping[str, Any],
    arm_stage_outcomes_by_execution_order: Mapping[str, Mapping[str, Any]],
    source_arm_stage_summaries_by_execution_order: Mapping[str, Mapping[str, Any]],
) -> None:
    metrics = importlib.import_module(
        "pulsefield_model.timing.evaluation.exp007_metrics"
    )
    _require_mapping(arm_rows_by_arm, "arm_rows_by_arm")
    validate_exact_fields(arm_rows_by_arm, ARM_OUTCOME_SHA_MAP_FIELDS, "arm_rows_by_arm")
    _require_mapping(
        arm_stage_outcomes_by_execution_order,
        "arm_stage_outcomes_by_execution_order",
    )
    validate_exact_fields(
        arm_stage_outcomes_by_execution_order,
        ARM_OUTCOME_SHA_MAP_FIELDS,
        "arm_stage_outcomes_by_execution_order",
    )
    global_manifest = validate_candidate_global_manifest_non_authoritative_shape(
        candidate_global_manifest
    )
    if (
        selection["candidate_global_manifest_sha256"]
        != object_complete_sha256(global_manifest)
    ):
        raise ValueError("ConfigSelection row rederive global manifest mismatch")
    reference_manifest_sha = global_manifest["candidate_reference_manifest_sha256"]
    global_entries = global_manifest["entries"]
    evaluations: dict[str, Any] = {}
    source_summaries = {
        arm: validate_source_arm_stage_summary(
            source_arm_stage_summaries_by_execution_order[arm]
        )
        for arm in EXP007_EXECUTION_ORDER
    }
    arm_outcomes = {
        arm: validate_arm_stage_outcome(arm_stage_outcomes_by_execution_order[arm])
        for arm in EXP007_EXECUTION_ORDER
    }
    for arm in EXP007_EXECUTION_ORDER:
        rows = [
            validate_row_result(row)
            for row in _require_sequence(arm_rows_by_arm[arm], f"arm_rows_by_arm.{arm}")
        ]
        if len(rows) != 16:
            raise ValueError("ConfigSelection row rederive requires 16 rows per arm")
        summary = source_summaries[arm]
        outcome = arm_outcomes[arm]
        if outcome["status"] != "success":
            raise ValueError("ConfigSelection row rederive requires successful outcomes")
        if outcome["stage"] != EXP007_SCHEDULE_STAGE or outcome["schedule_arm"] != arm:
            raise ValueError("ConfigSelection row rederive outcome arm/stage mismatch")
        if (
            summary["candidate_reference_manifest_sha256"]
            != reference_manifest_sha
            or outcome["candidate_reference_manifest_sha256"]
            != reference_manifest_sha
        ):
            raise ValueError(
                "ConfigSelection candidate reference manifest linkage mismatch"
            )
        if outcome["stage_summary_sha256"] != object_complete_sha256(summary):
            raise ValueError("ConfigSelection ArmStageSuccess source summary mismatch")
        expected_refs: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            entry = global_entries[index]
            if row["stage"] != EXP007_SCHEDULE_STAGE:
                raise ValueError("ConfigSelection row rederive stage mismatch")
            if row["schedule_arm"] != arm:
                raise ValueError("ConfigSelection row rederive arm mismatch")
            if row["row_index"] != index:
                raise ValueError("ConfigSelection row rederive row order mismatch")
            if (
                entry["row_index"] != index
                or row["cache_audio_key"] != entry["cache_audio_key"]
                or row["audio_group_key"] != entry["audio_group_key"]
            ):
                raise ValueError("ConfigSelection row/global identity mismatch")
            if row["row_payload_sha256"] != entry["arm_row_payload_sha256"][arm]:
                raise ValueError("ConfigSelection row/global payload SHA mismatch")
            if (
                row["candidate_payload_sha256"]
                != entry["candidate_payload_sha256"]
                or row["candidate_fingerprint"] != entry["candidate_fingerprint"]
            ):
                raise ValueError("ConfigSelection row/global candidate mismatch")
            if (
                row["run_config_fingerprint_sha256"]
                != summary["run_config_fingerprint_sha256"]
            ):
                raise ValueError("ConfigSelection row run config mismatch")
            if row["selector_manifest_sha256"] != selection["selector_manifest_sha256"]:
                raise ValueError("ConfigSelection row rederive selector mismatch")
            if row["selector_manifest_sha256"] != summary["selector_manifest_sha256"]:
                raise ValueError("ConfigSelection row summary selector mismatch")
            if row["input_manifest_sha256"] != summary["selector_manifest_sha256"]:
                raise ValueError("ConfigSelection row input manifest mismatch")
            if (
                row["source_closure_fingerprint_sha256"]
                != selection["source_closure_fingerprint_sha256"]
            ):
                raise ValueError("ConfigSelection row rederive source mismatch")
            if (
                row["source_closure_fingerprint_sha256"]
                != summary["source_closure_fingerprint_sha256"]
            ):
                raise ValueError("ConfigSelection row summary source mismatch")
            expected_refs.append(
                make_completed_row_ref(
                    row_index=row["row_index"],
                    cache_audio_key=row["cache_audio_key"],
                    identity_payload_sha256=row["identity_payload_sha256"],
                    row_payload_sha256=row["row_payload_sha256"],
                    candidate_reference_entry_payload_sha256=entry[
                        "candidate_reference_entry_payload_sha256"
                    ],
                )
            )
        expected_row_payloads_sha = canonical_json_sha256(expected_refs)
        if summary["row_refs"] != expected_refs:
            raise ValueError("ConfigSelection source row refs mismatch")
        if summary["row_payloads_sha256"] != expected_row_payloads_sha:
            raise ValueError("ConfigSelection source row payload hash mismatch")
        if outcome["row_payloads_sha256"] != expected_row_payloads_sha:
            raise ValueError("ConfigSelection ArmStageSuccess row payload hash mismatch")
        evaluations[arm] = metrics.evaluate_source_arm(
            tuple(_source_metric_row_from_row_result(row, metrics) for row in rows),
            schedule_arm=arm,
            worker_lifetime_rss_bytes=tuple(
                summary["rss_summary"]["worker_lifetime_bytes"]
            ),
            aggregate_wall_seconds=summary["runtime_summary"][
                "aggregate_wall_seconds"
            ],
        )
    rederived = metrics.select_source_schedule(evaluations)
    selected_arm = rederived.selected_schedule_arm
    selected_run_config = (
        None
        if selected_arm is None
        else source_summaries[selected_arm]["run_config_fingerprint_sha256"]
    )
    expected = make_config_selection(
        arm_outcome_sha256_by_execution_order=selection[
            "arm_outcome_sha256_by_execution_order"
        ],
        candidate_global_manifest_sha256=selection[
            "candidate_global_manifest_sha256"
        ],
        source_closure_fingerprint_sha256=selection[
            "source_closure_fingerprint_sha256"
        ],
        selector_manifest_sha256=selection["selector_manifest_sha256"],
        overlap_common=_metrics_audio_binding_to_dict(rederived.overlap_common),
        section_common=_metrics_audio_binding_to_dict(rederived.section_common),
        source_decision=rederived.source_decision,
        arm_order_values=[
            _metrics_order_value_to_dict(value)
            for value in rederived.arm_order_values
        ],
        selected_schedule_arm=selected_arm,
        selected_run_config_fingerprint_sha256=selected_run_config,
    )
    if selection != expected:
        raise ValueError("ConfigSelection row rederive mismatch")


def _validate_schedule_weak_summary_against_selected_rows(
    weak_summary: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    selected_arm = _require_schedule_arm(
        selection.get("selected_schedule_arm"),
        "ConfigSelection.selected_schedule_arm",
    )
    rows = [
        validate_row_result(row)
        for row in _require_sequence(
            arm_rows_by_arm[selected_arm],
            f"arm_rows_by_arm.{selected_arm}",
        )
    ]
    weak_refs = [
        validate_weak_row_ref(ref)
        for ref in _require_sequence(
            weak_summary.get("weak_row_refs"),
            "ScheduleWeakVetoSummary.weak_row_refs",
        )
    ]
    if len(rows) != 16 or len(weak_refs) != 16:
        raise ValueError("Repair80InputBinding weak selected row count mismatch")
    selected_ref_tuples = []
    pair_tuples = []
    for row, weak_ref in zip(rows, weak_refs, strict=True):
        if (
            row["row_index"] != weak_ref["row_index"]
            or row["cache_audio_key"] != weak_ref["cache_audio_key"]
        ):
            raise ValueError("Repair80InputBinding weak selected row ref mismatch")
        if weak_ref["prediction_row_sha256"] != row["row_payload_sha256"]:
            raise ValueError("Repair80InputBinding weak prediction row SHA mismatch")
        selected_ref_tuples.append(
            {
                "row_index": row["row_index"],
                "cache_audio_key": row["cache_audio_key"],
                "prediction_row_sha256": row["row_payload_sha256"],
            }
        )
        pair_tuples.append(
            {
                "row_index": row["row_index"],
                "cache_audio_key": row["cache_audio_key"],
                "row_payload_sha256": row["row_payload_sha256"],
                "prediction_row_sha256": row["row_payload_sha256"],
                "weak_row_payload_sha256": weak_ref["weak_row_payload_sha256"],
            }
        )
    if weak_summary["selected_row_refs_sha256"] != canonical_json_sha256(
        selected_ref_tuples
    ):
        raise ValueError("Repair80InputBinding weak selected refs hash mismatch")
    if weak_summary["row_weak_pairs_sha256"] != canonical_json_sha256(pair_tuples):
        raise ValueError("Repair80InputBinding weak row-pair hash mismatch")


def _source_metric_row_from_row_result(row: Mapping[str, Any], metrics: Any) -> Any:
    validated = validate_row_result(row)
    methods = validated["methods"]
    candidate = methods["candidate"]
    baseline = methods["baseline"]
    selected = methods["selected"]
    flags = validated["denominator_flags"]
    diagnostics = validated["diagnostics_summary"]
    candidate_accepted = candidate["status"] == "accepted"
    baseline_accepted = baseline["status"] == "accepted"
    return metrics.SourceMetricRow(
        schedule_arm=validated["schedule_arm"],
        row_index=validated["row_index"],
        cache_audio_key=validated["cache_audio_key"],
        audio_group_key=validated["audio_group_key"],
        cache_valid=flags["cache_valid"],
        projection_evaluable=flags["projection_evaluable"],
        candidate_status=candidate["status"],
        candidate_fallback_reason=(
            None if candidate_accepted else candidate["reason"]
        ),
        baseline_status=baseline["status"],
        selected_status=selected["status"],
        candidate_section_count=(
            candidate["grid_summary"]["section_count"] if candidate_accepted else None
        ),
        current_v2_segment_count=(
            len(baseline["grid"]["payload"]["segments"])
            if baseline_accepted
            else None
        ),
        current_v2_projection_sha256=(
            baseline["deterministic_projection_sha256"]
            if baseline_accepted
            else None
        ),
        candidate_seam_ms=(
            candidate["grid_summary"]["maximum_seam_discontinuity_ms"]
            if candidate_accepted
            else None
        ),
        overlap_p90_ms=(
            diagnostics["overlap"]["p90_ms"] if flags["overlap_available"] else None
        ),
        audio_arm_seconds=validated["runtime"]["audio_arm_seconds"],
        row_json_bytes=len(canonical_json_bytes(validated)),
        replay_schema_source_cache_candidate_v2_consistent=all(
            value
            for name, value in validated["hard_guards"].items()
            if name != "timed_out"
        ),
    )


def _metrics_audio_binding_to_dict(value: Any) -> dict[str, Any]:
    return {
        "count": value.count,
        "sorted_cache_audio_keys_sha256": value.sorted_cache_audio_keys_sha256,
    }


def _metrics_order_value_to_dict(value: Any) -> dict[str, Any]:
    return {
        "schedule_arm": value.schedule_arm,
        "e0_eligible": value.e0_eligible,
        "e1_eligible": value.e1_eligible,
        "elimination_reasons": list(value.elimination_reasons),
        "candidate_fallback_count": value.candidate_fallback_count,
        "no_origin_or_path_count": value.no_origin_or_path_count,
        "p90_overlap_ms": value.p90_overlap_ms,
        "section_inflation_violation_count": (
            value.section_inflation_violation_count
        ),
        "p90_section_excess": value.p90_section_excess,
        "p90_runtime": value.p90_runtime,
        "max_worker_rss": value.max_worker_rss,
        "tie_rank": value.tie_rank,
        "order_tuple_sha256": value.order_tuple_sha256,
    }


def _source_summary_elimination_reasons(
    summary: Mapping[str, Any],
    *,
    selection_value: Mapping[str, Any],
    common_ready: bool,
    overlap_common_count: int,
    section_common_count: int,
) -> list[str]:
    denominators = summary["denominators"]
    gates = summary["gates"]
    reasons: list[str] = []
    if denominators["candidate_fallback_audio"]["count"] > 1:
        reasons.append("candidate_fallback_guard")
    if denominators["no_origin_or_path_audio"]["count"] > 0:
        reasons.append("no_origin_or_path_guard")
    runtime_p90 = gates["runtime_seconds"]["p90"]
    if runtime_p90 is None:
        reasons.append("runtime_nonfinite")
    elif runtime_p90 > 60.0:
        reasons.append("runtime_p90_guard")
    if not gates["every_row_under_180_seconds"]:
        reasons.append("row_timeout_guard")
    rss_max = summary["rss_summary"]["arm_max_worker_bytes"]
    if rss_max > EXP007_WORKER_RSS_CAP_BYTES:
        reasons.append("rss_cap_guard")
    if not gates["seam_zero"]:
        reasons.append("seam_guard")
    if not gates["section_cap_valid"]:
        reasons.append("section_cap_guard")
    if not gates["replay_schema_source_cache_candidate_v2_consistent"]:
        reasons.append("row_consistency_guard")
    if not any(
        reason
        in {
            "candidate_fallback_guard",
            "no_origin_or_path_guard",
            "runtime_nonfinite",
            "runtime_p90_guard",
            "row_timeout_guard",
            "rss_nonfinite",
            "rss_cap_guard",
            "seam_guard",
            "section_cap_guard",
            "row_consistency_guard",
        }
        for reason in reasons
    ) and not common_ready:
        if overlap_common_count < 5:
            reasons.append("overlap_common_minimum")
        if section_common_count < 8:
            reasons.append("section_common_minimum")
    if (
        selection_value["p90_overlap_ms"] is not None
        and selection_value["p90_overlap_ms"] > 90.0
    ):
        reasons.append("overlap_e1_guard")
    return _sort_elimination_reasons(reasons)


def validate_schedule_weak_veto_outcome(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ScheduleWeakVetoOutcome")
    schema = payload.get("schema")
    if schema == SCHEDULE_WEAK_SUCCESS_SCHEMA:
        validate_exact_fields(
            payload,
            SCHEDULE_WEAK_SUCCESS_FIELDS,
            "ScheduleWeakVetoSuccess",
        )
        if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
            raise ValueError("ScheduleWeakVetoSuccess experiment_id is invalid")
        if payload.get("stage") != EXP007_SCHEDULE_STAGE:
            raise ValueError("ScheduleWeakVetoSuccess stage is invalid")
        if payload.get("status") != "success":
            raise ValueError("ScheduleWeakVetoSuccess status is invalid")
        _require_descriptor(payload, SCHEDULE_WEAK_SUCCESS_SCHEMA, "ScheduleWeakVetoSuccess")
        summary = validate_schedule_weak_veto_summary(payload.get("summary"))
        if payload.get("summary_payload_sha256") != object_complete_sha256(summary):
            raise ValueError("ScheduleWeakVetoSuccess summary hash mismatch")
        result = {
            "schema": SCHEDULE_WEAK_SUCCESS_SCHEMA,
            "experiment_id": EXP007_EXPERIMENT_ID,
            "stage": EXP007_SCHEDULE_STAGE,
            "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
            "status": "success",
            "summary": summary,
            "summary_payload_sha256": payload["summary_payload_sha256"],
            "outcome_fingerprint_sha256": require_sha256(
                payload.get("outcome_fingerprint_sha256"),
                "ScheduleWeakVetoSuccess.outcome_fingerprint_sha256",
            ),
        }
        validate_payload_hash(
            result,
            "outcome_fingerprint_sha256",
            context="ScheduleWeakVetoSuccess",
        )
        return result
    if schema == SCHEDULE_WEAK_HARD_FAILURE_SCHEMA:
        validate_exact_fields(
            payload,
            SCHEDULE_WEAK_HARD_FAILURE_FIELDS,
            "ScheduleWeakVetoHardFailure",
        )
        if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
            raise ValueError("ScheduleWeakVetoHardFailure experiment_id is invalid")
        if payload.get("stage") != EXP007_SCHEDULE_STAGE:
            raise ValueError("ScheduleWeakVetoHardFailure stage is invalid")
        if payload.get("status") != "hard_failure":
            raise ValueError("ScheduleWeakVetoHardFailure status is invalid")
        _require_descriptor(
            payload,
            SCHEDULE_WEAK_HARD_FAILURE_SCHEMA,
            "ScheduleWeakVetoHardFailure",
        )
        failure = validate_schedule_weak_failure_record(payload.get("failure"))
        if payload.get("failure_payload_sha256") != object_complete_sha256(failure):
            raise ValueError("ScheduleWeakVetoHardFailure failure hash mismatch")
        result = {
            "schema": SCHEDULE_WEAK_HARD_FAILURE_SCHEMA,
            "experiment_id": EXP007_EXPERIMENT_ID,
            "stage": EXP007_SCHEDULE_STAGE,
            "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
            "status": "hard_failure",
            "failure": failure,
            "failure_payload_sha256": payload["failure_payload_sha256"],
            "outcome_fingerprint_sha256": require_sha256(
                payload.get("outcome_fingerprint_sha256"),
                "ScheduleWeakVetoHardFailure.outcome_fingerprint_sha256",
            ),
        }
        validate_payload_hash(
            result,
            "outcome_fingerprint_sha256",
            context="ScheduleWeakVetoHardFailure",
        )
        return result
    raise ValueError("ScheduleWeakVetoOutcome variant is invalid")


def validate_schedule_weak_veto_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ScheduleWeakVetoSummary")
    validate_exact_fields(
        payload,
        SCHEDULE_WEAK_VETO_SUMMARY_FIELDS,
        "ScheduleWeakVetoSummary",
    )
    if payload.get("schema") != SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA:
        raise ValueError("ScheduleWeakVetoSummary schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("ScheduleWeakVetoSummary experiment_id is invalid")
    if payload.get("stage") != EXP007_SCHEDULE_STAGE:
        raise ValueError("ScheduleWeakVetoSummary stage is invalid")
    _require_descriptor(payload, SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA, "ScheduleWeakVetoSummary")
    arm = _require_schedule_arm(payload.get("schedule_arm"), "ScheduleWeakVetoSummary.schedule_arm")
    for name in (
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "source_selection_sha256",
    ):
        require_sha256(payload.get(name), f"ScheduleWeakVetoSummary.{name}")
    weak_refs = [
        validate_weak_row_ref(ref)
        for ref in _require_sequence(
            payload.get("weak_row_refs"),
            "ScheduleWeakVetoSummary.weak_row_refs",
        )
    ]
    if payload.get("weak_row_count") != 16 or len(weak_refs) != 16:
        raise ValueError("ScheduleWeakVetoSummary weak row count is invalid")
    _validate_contiguous_ref_order(weak_refs, context="ScheduleWeakVetoSummary")
    if payload.get("weak_payloads_sha256") != canonical_json_sha256(weak_refs):
        raise ValueError("ScheduleWeakVetoSummary weak_payloads_sha256 mismatch")
    selected_ref_tuples = [
        {
            "row_index": ref["row_index"],
            "cache_audio_key": ref["cache_audio_key"],
            "prediction_row_sha256": ref["prediction_row_sha256"],
        }
        for ref in weak_refs
    ]
    if payload.get("selected_row_refs_sha256") != canonical_json_sha256(
        selected_ref_tuples
    ):
        raise ValueError("ScheduleWeakVetoSummary selected row refs hash mismatch")
    pair_tuples = [
        {
            "row_index": ref["row_index"],
            "cache_audio_key": ref["cache_audio_key"],
            "row_payload_sha256": ref["prediction_row_sha256"],
            "prediction_row_sha256": ref["prediction_row_sha256"],
            "weak_row_payload_sha256": ref["weak_row_payload_sha256"],
        }
        for ref in weak_refs
    ]
    if payload.get("row_weak_pairs_sha256") != canonical_json_sha256(pair_tuples):
        raise ValueError("ScheduleWeakVetoSummary row weak pair hash mismatch")
    denominators = validate_schedule_weak_denominators(payload.get("denominators"))
    gates = validate_schedule_weak_gates(payload.get("gates"))
    decision = payload.get("decision")
    action = payload.get("action")
    expected_action = {
        "pass": "authorize_repair80",
        "ambiguous": "stop_ambiguous",
        "negative": "stop_negative",
    }.get(decision)
    if expected_action is None or action != expected_action:
        raise ValueError("ScheduleWeakVetoSummary decision/action mismatch")
    result = {
        "schema": SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "schedule_arm": arm,
        "four_arm_stage_summary_sha256": payload["four_arm_stage_summary_sha256"],
        "candidate_global_manifest_sha256": payload["candidate_global_manifest_sha256"],
        "source_closure_fingerprint_sha256": payload["source_closure_fingerprint_sha256"],
        "source_selection_sha256": payload["source_selection_sha256"],
        "selected_row_refs_sha256": payload["selected_row_refs_sha256"],
        "row_weak_pairs_sha256": payload["row_weak_pairs_sha256"],
        "weak_row_count": 16,
        "weak_row_refs": weak_refs,
        "weak_payloads_sha256": payload["weak_payloads_sha256"],
        "denominators": denominators,
        "gates": gates,
        "decision": decision,
        "action": action,
        "summary_fingerprint_sha256": require_sha256(
            payload.get("summary_fingerprint_sha256"),
            "ScheduleWeakVetoSummary.summary_fingerprint_sha256",
        ),
    }
    validate_payload_hash(
        result,
        "summary_fingerprint_sha256",
        context="ScheduleWeakVetoSummary",
    )
    return result


def validate_schedule_weak_failure_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ScheduleWeakFailureRecord")
    validate_exact_fields(payload, SCHEDULE_WEAK_FAILURE_FIELDS, "ScheduleWeakFailureRecord")
    if payload.get("schema") != SCHEDULE_WEAK_FAILURE_SCHEMA:
        raise ValueError("ScheduleWeakFailureRecord schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("ScheduleWeakFailureRecord experiment_id is invalid")
    if payload.get("stage") != EXP007_SCHEDULE_STAGE:
        raise ValueError("ScheduleWeakFailureRecord stage is invalid")
    _require_descriptor(payload, SCHEDULE_WEAK_FAILURE_SCHEMA, "ScheduleWeakFailureRecord")
    _require_schedule_arm(payload.get("schedule_arm"), "ScheduleWeakFailureRecord.schedule_arm")
    for name in (
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_selection_sha256",
        "source_closure_fingerprint_sha256",
    ):
        require_sha256(payload.get(name), f"ScheduleWeakFailureRecord.{name}")
    if payload.get("expected_row_count") != 16:
        raise ValueError("ScheduleWeakFailureRecord expected row count is invalid")
    if payload.get("failure_kind") not in SCHEDULE_WEAK_FAILURE_KINDS:
        raise ValueError("ScheduleWeakFailureRecord failure_kind is invalid")
    if payload.get("failure_stage") not in SCHEDULE_WEAK_FAILURE_STAGES:
        raise ValueError("ScheduleWeakFailureRecord failure_stage is invalid")
    causing_row = payload.get("causing_row_index")
    causing_key = payload.get("causing_cache_audio_key")
    if causing_row is None:
        if causing_key is not None:
            raise ValueError("ScheduleWeakFailureRecord causing key must be null")
    else:
        require_nonnegative_int(causing_row, "ScheduleWeakFailureRecord.causing_row_index")
        require_nonempty_string(causing_key, "ScheduleWeakFailureRecord.causing_cache_audio_key")
    completed = [
        validate_weak_row_ref(ref)
        for ref in _require_sequence(
            payload.get("completed_prefix"),
            "ScheduleWeakFailureRecord.completed_prefix",
        )
    ]
    pending = [
        validate_weak_pending_row_ref(ref)
        for ref in _require_sequence(
            payload.get("pending"),
            "ScheduleWeakFailureRecord.pending",
        )
    ]
    if payload.get("completed_prefix_count") != len(completed):
        raise ValueError("ScheduleWeakFailureRecord completed count mismatch")
    if payload.get("pending_count") != len(pending):
        raise ValueError("ScheduleWeakFailureRecord pending count mismatch")
    if len(completed) + len(pending) != 16:
        raise ValueError("ScheduleWeakFailureRecord counts do not sum")
    if payload.get("completed_prefix_sha256") != canonical_json_sha256(completed):
        raise ValueError("ScheduleWeakFailureRecord completed prefix hash mismatch")
    if payload.get("pending_sha256") != canonical_json_sha256(pending):
        raise ValueError("ScheduleWeakFailureRecord pending hash mismatch")
    deterministic = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "failure_deterministic_fingerprint_sha256",
            "full_payload_sha256",
        }
    }
    if payload.get("failure_deterministic_fingerprint_sha256") != canonical_json_sha256(deterministic):
        raise ValueError("ScheduleWeakFailureRecord deterministic hash mismatch")
    validate_payload_hash(payload, "full_payload_sha256", context="ScheduleWeakFailureRecord")
    return dict(payload)


def validate_weak_row_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "WeakRowRef")
    validate_exact_fields(payload, WEAK_ROW_REF_FIELDS, "WeakRowRef")
    return {
        "row_index": require_nonnegative_int(
            payload.get("row_index"),
            "WeakRowRef.row_index",
        ),
        "cache_audio_key": require_nonempty_string(
            payload.get("cache_audio_key"),
            "WeakRowRef.cache_audio_key",
        ),
        "prediction_row_sha256": require_sha256(
            payload.get("prediction_row_sha256"),
            "WeakRowRef.prediction_row_sha256",
        ),
        "weak_row_payload_sha256": require_sha256(
            payload.get("weak_row_payload_sha256"),
            "WeakRowRef.weak_row_payload_sha256",
        ),
    }


def validate_weak_pending_row_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "WeakPendingRowRef")
    validate_exact_fields(payload, WEAK_PENDING_ROW_REF_FIELDS, "WeakPendingRowRef")
    return {
        "row_index": require_nonnegative_int(
            payload.get("row_index"),
            "WeakPendingRowRef.row_index",
        ),
        "cache_audio_key": require_nonempty_string(
            payload.get("cache_audio_key"),
            "WeakPendingRowRef.cache_audio_key",
        ),
        "prediction_row_sha256": require_sha256(
            payload.get("prediction_row_sha256"),
            "WeakPendingRowRef.prediction_row_sha256",
        ),
    }


def validate_schedule_weak_denominators(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ScheduleWeakDenominators")
    validate_exact_fields(
        payload,
        SCHEDULE_WEAK_DENOMINATORS_FIELDS,
        "ScheduleWeakDenominators",
    )
    if payload.get("stage_audio_count") != 16:
        raise ValueError("ScheduleWeakDenominators stage_audio_count is invalid")
    result = {"stage_audio_count": 16}
    for name in sorted(SCHEDULE_WEAK_DENOMINATORS_FIELDS - {"stage_audio_count"}):
        result[name] = validate_audio_set_binding(payload.get(name))
    if result["stage_audio"]["count"] != 16:
        raise ValueError("ScheduleWeakDenominators stage_audio count is invalid")
    comparator_total = (
        result["comparator_available_audio"]["count"]
        + result["comparator_unavailable_audio"]["count"]
        + result["comparator_conflicting_audio"]["count"]
    )
    if comparator_total != 16:
        raise ValueError("ScheduleWeakDenominators comparator partition mismatch")
    return {name: result[name] for name in SCHEDULE_WEAK_DENOMINATORS_FIELDS}


def validate_schedule_weak_gates(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ScheduleWeakGates")
    validate_exact_fields(payload, SCHEDULE_WEAK_GATES_FIELDS, "ScheduleWeakGates")
    result = {
        "pure_mean_phase_ratio": validate_ratio_value(
            payload.get("pure_mean_phase_ratio")
        ),
        "pure_p90_phase_ratio": validate_ratio_value(payload.get("pure_p90_phase_ratio")),
        "pure_phase_coverage": validate_coverage_value(
            payload.get("pure_phase_coverage")
        ),
        "alias_max_prefix_drift_mean_ratio": validate_ratio_value(
            payload.get("alias_max_prefix_drift_mean_ratio")
        ),
        "alias_max_prefix_drift_p90_ratio": validate_ratio_value(
            payload.get("alias_max_prefix_drift_p90_ratio")
        ),
    }
    numeric_or_null = (
        "current_v2_phase_mean_ms",
        "pure_exp006_phase_mean_ms",
        "current_v2_phase_p90_ms",
        "pure_exp006_phase_p90_ms",
        "current_v2_alias_drift_mean_ms",
        "pure_exp006_alias_drift_mean_ms",
        "current_v2_alias_drift_p90_ms",
        "pure_exp006_alias_drift_p90_ms",
        "current_v2_boundary_f1_mean",
        "pure_exp006_boundary_f1_mean",
        "selected_boundary_f1_mean",
        "pure_minus_v2_boundary_f1_delta",
    )
    for name in numeric_or_null:
        value = payload.get(name)
        result[name] = None if value is None else require_finite_number(value, f"ScheduleWeakGates.{name}")
    if (
        result["current_v2_boundary_f1_mean"] is not None
        and result["pure_exp006_boundary_f1_mean"] is not None
    ):
        expected_delta = (
            result["pure_exp006_boundary_f1_mean"]
            - result["current_v2_boundary_f1_mean"]
        )
        if result["pure_minus_v2_boundary_f1_delta"] != expected_delta:
            raise ValueError("ScheduleWeakGates boundary delta mismatch")
    return {name: result[name] for name in SCHEDULE_WEAK_GATES_FIELDS}


def make_schedule_weak_veto_summary(
    *,
    schedule_arm: str,
    four_arm_stage_summary_sha256: str,
    candidate_global_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    source_selection_sha256: str,
    weak_row_refs: Sequence[Mapping[str, Any]],
    denominators: Mapping[str, Any],
    gates: Mapping[str, Any],
    decision: str,
) -> dict[str, Any]:
    refs = [validate_weak_row_ref(ref) for ref in weak_row_refs]
    action = {
        "pass": "authorize_repair80",
        "ambiguous": "stop_ambiguous",
        "negative": "stop_negative",
    }[decision]
    selected_ref_tuples = [
        {
            "row_index": ref["row_index"],
            "cache_audio_key": ref["cache_audio_key"],
            "prediction_row_sha256": ref["prediction_row_sha256"],
        }
        for ref in refs
    ]
    pair_tuples = [
        {
            "row_index": ref["row_index"],
            "cache_audio_key": ref["cache_audio_key"],
            "row_payload_sha256": ref["prediction_row_sha256"],
            "prediction_row_sha256": ref["prediction_row_sha256"],
            "weak_row_payload_sha256": ref["weak_row_payload_sha256"],
        }
        for ref in refs
    ]
    payload = {
        "schema": SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": schema_descriptor_sha256(
            SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA
        ),
        "schedule_arm": schedule_arm,
        "four_arm_stage_summary_sha256": four_arm_stage_summary_sha256,
        "candidate_global_manifest_sha256": candidate_global_manifest_sha256,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "source_selection_sha256": source_selection_sha256,
        "selected_row_refs_sha256": canonical_json_sha256(selected_ref_tuples),
        "row_weak_pairs_sha256": canonical_json_sha256(pair_tuples),
        "weak_row_count": len(refs),
        "weak_row_refs": refs,
        "weak_payloads_sha256": canonical_json_sha256(refs),
        "denominators": denominators,
        "gates": gates,
        "decision": decision,
        "action": action,
    }
    return validate_schedule_weak_veto_summary(
        with_payload_hash(payload, "summary_fingerprint_sha256")
    )


def make_schedule_weak_success_outcome(summary: Mapping[str, Any]) -> dict[str, Any]:
    summary_payload = validate_schedule_weak_veto_summary(summary)
    payload = {
        "schema": SCHEDULE_WEAK_SUCCESS_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": schema_descriptor_sha256(
            SCHEDULE_WEAK_SUCCESS_SCHEMA
        ),
        "status": "success",
        "summary": summary_payload,
        "summary_payload_sha256": object_complete_sha256(summary_payload),
    }
    return validate_schedule_weak_veto_outcome(
        with_payload_hash(payload, "outcome_fingerprint_sha256")
    )


def make_source_closure(
    repo_root: str | Path,
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    root = _resolve_source_repo_root(repo_root)
    behavior = _build_source_behavior(root)
    audit = _build_source_audit(root, generated_at_utc=generated_at_utc)
    payload = {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "schema_descriptor_sha256": schema_descriptor_sha256(SOURCE_CLOSURE_SCHEMA),
        "behavior": behavior,
        "audit": audit,
        "source_closure_fingerprint_sha256": canonical_json_sha256(behavior),
    }
    return validate_source_closure(
        with_payload_hash(payload, "full_payload_sha256")
    )


def validate_source_closure_for_repo(
    payload: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    root = _resolve_source_repo_root(repo_root)
    closure = validate_source_closure(payload)
    expected_behavior = _build_source_behavior(root)
    if closure["behavior"] != expected_behavior:
        raise ValueError("SourceClosure behavior does not match repo source bytes")
    if closure["audit"]["absolute_root_path"] != str(root):
        raise ValueError("SourceClosure audit root mismatch")
    return closure


def validate_source_closure(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "SourceClosure")
    validate_exact_fields(payload, SOURCE_CLOSURE_FIELDS, "SourceClosure")
    if payload.get("schema") != SOURCE_CLOSURE_SCHEMA:
        raise ValueError("SourceClosure schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("SourceClosure experiment_id is invalid")
    _require_descriptor(payload, SOURCE_CLOSURE_SCHEMA, "SourceClosure")
    behavior = validate_source_behavior(payload.get("behavior"))
    audit = validate_source_audit(payload.get("audit"))
    if payload.get("source_closure_fingerprint_sha256") != canonical_json_sha256(behavior):
        raise ValueError("SourceClosure source_closure_fingerprint_sha256 mismatch")
    result = {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "behavior": behavior,
        "audit": audit,
        "source_closure_fingerprint_sha256": payload[
            "source_closure_fingerprint_sha256"
        ],
        "full_payload_sha256": require_sha256(
            payload.get("full_payload_sha256"),
            "SourceClosure.full_payload_sha256",
        ),
    }
    validate_payload_hash(result, "full_payload_sha256", context="SourceClosure")
    return result


def validate_source_behavior(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "SourceBehavior")
    validate_exact_fields(payload, SOURCE_BEHAVIOR_FIELDS, "SourceBehavior")
    entry_modules = _validate_string_sequence(
        payload.get("entry_modules"),
        "SourceBehavior.entry_modules",
    )
    if tuple(entry_modules) != SOURCE_CLOSURE_ENTRY_MODULES:
        raise ValueError("SourceBehavior entry_modules mismatch")
    required_files = _validate_relative_source_file_list(
        payload.get("required_non_import_files"),
        context="SourceBehavior.required_non_import_files",
    )
    relative_files = _validate_relative_source_file_list(
        payload.get("relative_source_files"),
        context="SourceBehavior.relative_source_files",
        require_nonempty=True,
    )
    import_edges = _validate_import_edges(payload.get("import_edges"))
    modules = _validate_module_identities(payload.get("module_identities"))
    if payload.get("relative_source_files_sha256") != canonical_json_sha256(relative_files):
        raise ValueError("SourceBehavior relative source files hash mismatch")
    if payload.get("import_graph_sha256") != canonical_json_sha256(import_edges):
        raise ValueError("SourceBehavior import graph hash mismatch")
    if payload.get("module_identities_sha256") != canonical_json_sha256(modules):
        raise ValueError("SourceBehavior module identities hash mismatch")
    return {
        "entry_modules": entry_modules,
        "required_non_import_files": required_files,
        "relative_source_files": relative_files,
        "relative_source_files_sha256": require_sha256(
            payload.get("relative_source_files_sha256"),
            "SourceBehavior.relative_source_files_sha256",
        ),
        "import_edges": import_edges,
        "import_graph_sha256": require_sha256(
            payload.get("import_graph_sha256"),
            "SourceBehavior.import_graph_sha256",
        ),
        "module_identities": modules,
        "module_identities_sha256": require_sha256(
            payload.get("module_identities_sha256"),
            "SourceBehavior.module_identities_sha256",
        ),
        "python_behavior_version": require_nonempty_string(
            payload.get("python_behavior_version"),
            "SourceBehavior.python_behavior_version",
        ),
        "numpy_behavior_version": require_nonempty_string(
            payload.get("numpy_behavior_version"),
            "SourceBehavior.numpy_behavior_version",
        ),
        "canonical_json_contract_sha256": require_sha256(
            payload.get("canonical_json_contract_sha256"),
            "SourceBehavior.canonical_json_contract_sha256",
        ),
    }


def validate_source_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "SourceAudit")
    validate_exact_fields(payload, SOURCE_AUDIT_FIELDS, "SourceAudit")
    git_commit = payload.get("git_commit")
    if git_commit is not None:
        git_commit = require_lower_hex(git_commit, "SourceAudit.git_commit")
    dirty_files = _validate_sorted_unique_strings(
        payload.get("dirty_files"),
        context="SourceAudit.dirty_files",
    )
    return {
        "generated_at_utc": require_nonempty_string(
            payload.get("generated_at_utc"),
            "SourceAudit.generated_at_utc",
        ),
        "absolute_root_path": require_nonempty_string(
            payload.get("absolute_root_path"),
            "SourceAudit.absolute_root_path",
        ),
        "git_commit": git_commit,
        "dirty_files": dirty_files,
        "platform": require_nonempty_string(payload.get("platform"), "SourceAudit.platform"),
        "python_full_version": require_nonempty_string(
            payload.get("python_full_version"),
            "SourceAudit.python_full_version",
        ),
        "numpy_full_version": require_nonempty_string(
            payload.get("numpy_full_version"),
            "SourceAudit.numpy_full_version",
        ),
    }


def _resolve_source_repo_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError("SourceClosure repo_root must be an existing directory")
    if not (root / "src" / "pulsefield_model").is_dir():
        raise ValueError("SourceClosure repo_root does not contain src/pulsefield_model")
    return root


def _build_source_behavior(repo_root: Path) -> dict[str, Any]:
    module_paths = _collect_source_closure_module_paths(repo_root)
    relative_source_files = _source_file_records(repo_root, module_paths.values())
    required_files = _source_file_records(
        repo_root,
        [repo_root / path for path in SOURCE_CLOSURE_REQUIRED_NON_IMPORT_FILES],
    )
    module_identities = [
        {
            "module_name": module_name,
            "relative_path": _relative_repo_path(path, repo_root),
            "sha256": _file_sha256(path),
        }
        for module_name, path in sorted(module_paths.items())
    ]
    import_edges = _collect_source_closure_import_edges(repo_root, module_paths)
    behavior = {
        "entry_modules": list(SOURCE_CLOSURE_ENTRY_MODULES),
        "required_non_import_files": required_files,
        "relative_source_files": relative_source_files,
        "relative_source_files_sha256": canonical_json_sha256(relative_source_files),
        "import_edges": import_edges,
        "import_graph_sha256": canonical_json_sha256(import_edges),
        "module_identities": module_identities,
        "module_identities_sha256": canonical_json_sha256(module_identities),
        "python_behavior_version": (
            f"{_platform.python_implementation()} "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "numpy_behavior_version": _numpy_version(),
        "canonical_json_contract_sha256": _canonical_json_contract_sha256(),
    }
    return validate_source_behavior(behavior)


def _build_source_audit(
    repo_root: Path,
    *,
    generated_at_utc: str | None,
) -> dict[str, Any]:
    timestamp = (
        generated_at_utc
        if generated_at_utc is not None
        else _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    audit = {
        "generated_at_utc": timestamp,
        "absolute_root_path": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "dirty_files": _git_dirty_files(repo_root),
        "platform": _platform.platform(),
        "python_full_version": sys.version,
        "numpy_full_version": _numpy_version(),
    }
    return validate_source_audit(audit)


def _collect_source_closure_module_paths(repo_root: Path) -> dict[str, Path]:
    pending = list(SOURCE_CLOSURE_ENTRY_MODULES)
    module_paths: dict[str, Path] = {}
    while pending:
        module_name = pending.pop(0)
        if module_name in module_paths:
            continue
        path = _resolve_owned_module_path(repo_root, module_name)
        if path is None:
            raise ValueError(f"SourceClosure entry module is not repo-owned: {module_name}")
        module_paths[module_name] = path
        dependencies = (
            _owned_parent_package_modules(repo_root, module_name)
            + _owned_imports_for_module(
                repo_root,
                module_name=module_name,
                module_path=path,
            )
        )
        for dependency in dependencies:
            if dependency not in module_paths and dependency not in pending:
                pending.append(dependency)
    return module_paths


def _owned_parent_package_modules(repo_root: Path, module_name: str) -> list[str]:
    parts = module_name.split(".")
    parents = []
    for index in range(1, len(parts)):
        parent = ".".join(parts[:index])
        path = _resolve_owned_module_path(repo_root, parent)
        if path is not None and path.name == "__init__.py":
            parents.append(parent)
    return parents


def _collect_source_closure_import_edges(
    repo_root: Path,
    module_paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    edges: set[tuple[str, str, str | None]] = set()
    for module_name, module_path in module_paths.items():
        importer = _relative_repo_path(module_path, repo_root)
        for imported_module in _source_imports_for_module(
            module_name=module_name,
            module_path=module_path,
        ):
            resolved_path = _resolve_owned_module_path(repo_root, imported_module)
            if resolved_path is not None:
                edges.add(
                    (
                        importer,
                        imported_module,
                        _relative_repo_path(resolved_path, repo_root),
                    )
                )
            elif not imported_module.startswith("pulsefield_model."):
                _require_source_closure_external_import_bound(
                    imported_module,
                    importer_relative_path=importer,
                )
                edges.add((importer, imported_module, None))
    return [
        {
            "importer_relative_path": importer,
            "imported_module": imported,
            "resolved_relative_path": resolved,
        }
        for importer, imported, resolved in sorted(
            edges,
            key=lambda item: (item[0], item[1], "" if item[2] is None else item[2]),
        )
    ]


def _owned_imports_for_module(
    repo_root: Path,
    *,
    module_name: str,
    module_path: Path,
) -> list[str]:
    result = []
    for imported_module in _source_imports_for_module(
        module_name=module_name,
        module_path=module_path,
    ):
        if _resolve_owned_module_path(repo_root, imported_module) is not None:
            result.append(imported_module)
    return sorted(set(result))


def _source_imports_for_module(
    *,
    module_name: str,
    module_path: Path,
) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    package_name = _module_package_name(module_name, module_path)
    imports: set[str] = set()
    for node in _import_time_import_nodes(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.update(
                _import_from_candidates(
                    module_name=module_name,
                    package_name=package_name,
                    node=node,
                )
            )
    return sorted(imports)


def _import_time_import_nodes(statements: Sequence[ast.stmt]) -> list[ast.Import | ast.ImportFrom]:
    result: list[ast.Import | ast.ImportFrom] = []
    pending = list(statements)
    while pending:
        node = pending.pop(0)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            result.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        elif isinstance(node, ast.ClassDef):
            pending[0:0] = list(node.body)
        elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            pending[0:0] = list(node.body) + list(node.orelse)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            pending[0:0] = list(node.body)
        elif isinstance(node, ast.Try):
            handler_bodies = [
                statement
                for handler in node.handlers
                for statement in handler.body
            ]
            pending[0:0] = (
                list(node.body)
                + handler_bodies
                + list(node.orelse)
                + list(node.finalbody)
            )
        elif isinstance(node, ast.Match):
            pending[0:0] = [
                statement
                for case in node.cases
                for statement in case.body
            ]
    return result


def _require_source_closure_external_import_bound(
    imported_module: str,
    *,
    importer_relative_path: str,
) -> None:
    top_level = imported_module.split(".", 1)[0]
    stdlib_names = getattr(sys, "stdlib_module_names", frozenset())
    if top_level == "numpy" or top_level in stdlib_names:
        return
    raise ValueError(
        "SourceClosure import-time external dependency is not behavior-bound: "
        f"{imported_module} imported by {importer_relative_path}"
    )


def _import_from_candidates(
    *,
    module_name: str,
    package_name: str,
    node: ast.ImportFrom,
) -> list[str]:
    base = node.module or ""
    if node.level:
        package_parts = package_name.split(".")
        keep = len(package_parts) - node.level + 1
        if keep < 0:
            return []
        prefix = ".".join(package_parts[:keep])
        base = f"{prefix}.{base}" if base else prefix
    if not base:
        return []
    result = []
    for alias in node.names:
        if alias.name == "*":
            result.append(base)
            continue
        result.append(f"{base}.{alias.name}")
        result.append(base)
    return result


def _module_package_name(module_name: str, module_path: Path) -> str:
    if module_path.name == "__init__.py":
        return module_name
    if "." not in module_name:
        return module_name
    return module_name.rsplit(".", 1)[0]


def _resolve_owned_module_path(repo_root: Path, module_name: str) -> Path | None:
    if not module_name.startswith("pulsefield_model"):
        return None
    parts = module_name.split(".")
    base = repo_root / "src"
    module_file = base.joinpath(*parts).with_suffix(".py")
    package_file = base.joinpath(*parts, "__init__.py")
    for candidate in (module_file, package_file):
        try:
            resolved = candidate.resolve()
            resolved.relative_to(repo_root)
        except ValueError:
            continue
        if candidate.is_file():
            return resolved
    return None


def _source_file_records(repo_root: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = [
        {
            "relative_path": _relative_repo_path(path, repo_root),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    records.sort(key=lambda row: row["relative_path"])
    return _validate_relative_source_file_list(records, context="SourceClosure source files")


def _relative_repo_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("SourceClosure path is outside repo root") from exc
    return PurePosixPath(relative).as_posix()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"SourceClosure required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numpy_version() -> str:
    numpy = importlib.import_module("numpy")
    return require_nonempty_string(
        getattr(numpy, "__version__", None),
        "numpy.__version__",
    )


def _canonical_json_contract_sha256() -> str:
    return canonical_json_sha256(
        {
            "allow_nan": False,
            "ensure_ascii": True,
            "separators": [",", ":"],
            "sort_keys": True,
            "json_contract": "exp007_canonical_json_v1",
        }
    )


def _git_commit(repo_root: Path) -> str | None:
    result = _run_git(repo_root, "rev-parse", "HEAD")
    if result is None:
        return None
    value = result.strip().lower()
    if not value:
        return None
    return value


def _git_dirty_files(repo_root: Path) -> list[str]:
    result = _run_git(repo_root, "status", "--short", "--untracked-files=all")
    if result is None:
        return []
    files = []
    for line in result.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            files.append(path)
    return sorted(set(files))


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def validate_source_arm_stage_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "SourceArmStageSummary")
    validate_exact_fields(payload, SOURCE_ARM_STAGE_SUMMARY_FIELDS, "SourceArmStageSummary")
    if payload.get("schema") != SOURCE_ARM_STAGE_SUMMARY_SCHEMA:
        raise ValueError("SourceArmStageSummary schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("SourceArmStageSummary experiment_id is invalid")
    if payload.get("stage") != EXP007_SCHEDULE_STAGE:
        raise ValueError("SourceArmStageSummary stage is invalid")
    _require_descriptor(payload, SOURCE_ARM_STAGE_SUMMARY_SCHEMA, "SourceArmStageSummary")
    arm = _require_schedule_arm(payload.get("schedule_arm"), "SourceArmStageSummary.schedule_arm")
    for name in (
        "run_config_fingerprint_sha256",
        "source_closure_fingerprint_sha256",
        "selector_manifest_sha256",
        "candidate_reference_manifest_sha256",
    ):
        require_sha256(payload.get(name), f"SourceArmStageSummary.{name}")
    refs = [
        validate_completed_row_ref(ref)
        for ref in _require_sequence(payload.get("row_refs"), "SourceArmStageSummary.row_refs")
    ]
    if payload.get("row_count") != 16 or len(refs) != 16:
        raise ValueError("SourceArmStageSummary row_count is invalid")
    _validate_completed_ref_order(refs, context="SourceArmStageSummary")
    if payload.get("row_payloads_sha256") != canonical_json_sha256(refs):
        raise ValueError("SourceArmStageSummary row payload hash mismatch")
    denominators = validate_source_arm_denominators(payload.get("denominators"))
    gates = validate_source_arm_gates(payload.get("gates"), denominators=denominators)
    runtime_summary = validate_runtime_summary(payload.get("runtime_summary"))
    rss_summary = validate_rss_summary(payload.get("rss_summary"))
    if runtime_summary["row_seconds"] != gates["runtime_seconds"]:
        raise ValueError("SourceArmStageSummary runtime summary mismatch")
    expected_worker_rss = make_stats_value(rss_summary["worker_lifetime_bytes"])
    if gates["worker_rss_bytes"] != expected_worker_rss:
        raise ValueError("SourceArmStageSummary worker RSS summary mismatch")
    if gates["worker_rss_bytes"]["maximum"] != rss_summary["arm_max_worker_bytes"]:
        raise ValueError("SourceArmStageSummary worker RSS max mismatch")
    result = {
        "schema": SOURCE_ARM_STAGE_SUMMARY_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "schedule_arm": arm,
        "run_config_fingerprint_sha256": payload["run_config_fingerprint_sha256"],
        "source_closure_fingerprint_sha256": payload["source_closure_fingerprint_sha256"],
        "selector_manifest_sha256": payload["selector_manifest_sha256"],
        "candidate_reference_manifest_sha256": payload[
            "candidate_reference_manifest_sha256"
        ],
        "row_count": 16,
        "row_refs": refs,
        "row_payloads_sha256": payload["row_payloads_sha256"],
        "denominators": denominators,
        "gates": gates,
        "runtime_summary": runtime_summary,
        "rss_summary": rss_summary,
        "summary_fingerprint_sha256": require_sha256(
            payload.get("summary_fingerprint_sha256"),
            "SourceArmStageSummary.summary_fingerprint_sha256",
        ),
    }
    validate_payload_hash(
        result,
        "summary_fingerprint_sha256",
        context="SourceArmStageSummary",
    )
    return result


def validate_source_arm_stage_summary_authoritatively(
    payload: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a schedule16 source-arm summary against exact RowResult payloads."""

    summary = validate_source_arm_stage_summary(payload)
    validated_rows = [
        validate_row_result(row)
        for row in _require_sequence(rows, "SourceArmStageSummary.rows")
    ]
    if len(validated_rows) != 16:
        raise ValueError("SourceArmStageSummary authority requires 16 exact rows")
    expected_refs: list[dict[str, Any]] = []
    for row, ref in zip(validated_rows, summary["row_refs"], strict=True):
        expected_ref = make_completed_row_ref(
            row_index=row["row_index"],
            cache_audio_key=row["cache_audio_key"],
            identity_payload_sha256=row["identity_payload_sha256"],
            row_payload_sha256=row["row_payload_sha256"],
            candidate_reference_entry_payload_sha256=(
                ref["candidate_reference_entry_payload_sha256"]
            ),
        )
        if ref != expected_ref:
            raise ValueError("SourceArmStageSummary row refs do not match exact rows")
        if row["stage"] != EXP007_SCHEDULE_STAGE:
            raise ValueError("SourceArmStageSummary authority row stage mismatch")
        if row["schedule_arm"] != summary["schedule_arm"]:
            raise ValueError("SourceArmStageSummary authority row arm mismatch")
        if row["run_config_fingerprint_sha256"] != summary["run_config_fingerprint_sha256"]:
            raise ValueError("SourceArmStageSummary authority row run config mismatch")
        if (
            row["source_closure_fingerprint_sha256"]
            != summary["source_closure_fingerprint_sha256"]
        ):
            raise ValueError("SourceArmStageSummary authority row source mismatch")
        if row["selector_manifest_sha256"] != summary["selector_manifest_sha256"]:
            raise ValueError("SourceArmStageSummary authority row selector mismatch")
        if row["input_manifest_sha256"] != summary["selector_manifest_sha256"]:
            raise ValueError("SourceArmStageSummary authority row input mismatch")
        expected_refs.append(expected_ref)
    expected_row_payloads_sha = canonical_json_sha256(expected_refs)
    if summary["row_payloads_sha256"] != expected_row_payloads_sha:
        raise ValueError("SourceArmStageSummary authority row payload hash mismatch")

    metrics = importlib.import_module(
        "pulsefield_model.timing.evaluation.exp007_metrics"
    )
    evaluation = metrics.evaluate_source_arm(
        tuple(_source_metric_row_from_row_result(row, metrics) for row in validated_rows),
        schedule_arm=summary["schedule_arm"],
        worker_lifetime_rss_bytes=tuple(
            summary["rss_summary"]["worker_lifetime_bytes"]
        ),
        aggregate_wall_seconds=summary["runtime_summary"]["aggregate_wall_seconds"],
    )
    expected_summary = validate_source_arm_stage_summary(
        with_payload_hash(
            _source_arm_summary_payload_from_evaluation(
                summary=summary,
                evaluation=evaluation,
                row_refs=expected_refs,
            ),
            "summary_fingerprint_sha256",
        )
    )
    if summary != expected_summary:
        raise ValueError("SourceArmStageSummary row authority mismatch")
    return summary


def _source_arm_summary_payload_from_evaluation(
    *,
    summary: Mapping[str, Any],
    evaluation: Any,
    row_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SOURCE_ARM_STAGE_SUMMARY_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": summary["schema_descriptor_sha256"],
        "schedule_arm": evaluation.schedule_arm,
        "run_config_fingerprint_sha256": summary["run_config_fingerprint_sha256"],
        "source_closure_fingerprint_sha256": summary[
            "source_closure_fingerprint_sha256"
        ],
        "selector_manifest_sha256": summary["selector_manifest_sha256"],
        "candidate_reference_manifest_sha256": summary[
            "candidate_reference_manifest_sha256"
        ],
        "row_count": 16,
        "row_refs": list(row_refs),
        "row_payloads_sha256": canonical_json_sha256(row_refs),
        "denominators": _source_arm_denominators_from_metrics(
            evaluation.denominators
        ),
        "gates": _source_arm_gates_from_metrics(evaluation.gates),
        "runtime_summary": {
            "row_seconds": evaluation.runtime_summary.row_seconds.to_dict(),
            "aggregate_wall_seconds": (
                evaluation.runtime_summary.aggregate_wall_seconds
            ),
        },
        "rss_summary": {
            "worker_count": evaluation.rss_summary.worker_count,
            "worker_lifetime_bytes": list(evaluation.rss_summary.worker_lifetime_bytes),
            "arm_max_worker_bytes": evaluation.rss_summary.arm_max_worker_bytes,
        },
    }


def _source_arm_denominators_from_metrics(value: Any) -> dict[str, Any]:
    return {
        "stage_audio_count": value.stage_audio_count,
        "stage_audio": value.stage_audio.to_dict(),
        "cache_valid_audio": value.cache_valid_audio.to_dict(),
        "projection_evaluable_audio": value.projection_evaluable_audio.to_dict(),
        "candidate_accepted_audio": value.candidate_accepted_audio.to_dict(),
        "candidate_fallback_audio": value.candidate_fallback_audio.to_dict(),
        "selected_product_fallback_audio": (
            value.selected_product_fallback_audio.to_dict()
        ),
        "baseline_accepted_audio": value.baseline_accepted_audio.to_dict(),
        "product_grid_available_audio": value.product_grid_available_audio.to_dict(),
        "no_origin_or_path_audio": value.no_origin_or_path_audio.to_dict(),
        "resource_cap_fallback_audio": value.resource_cap_fallback_audio.to_dict(),
        "overlap_available_audio": value.overlap_available_audio.to_dict(),
    }


def _source_arm_gates_from_metrics(value: Any) -> dict[str, Any]:
    return {
        "candidate_fallback_rate": make_rate_value(
            value.candidate_fallback_rate.numerator,
            value.candidate_fallback_rate.denominator,
        ),
        "selected_product_fallback_rate": make_rate_value(
            value.selected_product_fallback_rate.numerator,
            value.selected_product_fallback_rate.denominator,
        ),
        "no_origin_or_path_rate": make_rate_value(
            value.no_origin_or_path_rate.numerator,
            value.no_origin_or_path_rate.denominator,
        ),
        "runtime_seconds": value.runtime_seconds.to_dict(),
        "worker_rss_bytes": value.worker_rss_bytes.to_dict(),
        "candidate_seam_ms": value.candidate_seam_ms.to_dict(),
        "candidate_section_count": value.candidate_section_count.to_dict(),
        "row_json_bytes": value.row_json_bytes.to_dict(),
        "every_row_under_180_seconds": value.every_row_under_180_seconds,
        "seam_zero": value.seam_zero,
        "section_cap_valid": value.section_cap_valid,
        "row_byte_cap_valid": value.row_byte_cap_valid,
        "replay_schema_source_cache_candidate_v2_consistent": (
            value.replay_schema_source_cache_candidate_v2_consistent
        ),
    }


def validate_runtime_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RuntimeSummary")
    validate_exact_fields(payload, RUNTIME_SUMMARY_FIELDS, "RuntimeSummary")
    aggregate = require_finite_number(
        payload.get("aggregate_wall_seconds"),
        "RuntimeSummary.aggregate_wall_seconds",
    )
    if aggregate < 0:
        raise ValueError("RuntimeSummary aggregate wall seconds must be nonnegative")
    return {
        "row_seconds": validate_stats_value(payload.get("row_seconds")),
        "aggregate_wall_seconds": payload.get("aggregate_wall_seconds"),
    }


def validate_rss_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "RssSummary")
    validate_exact_fields(payload, RSS_SUMMARY_FIELDS, "RssSummary")
    if payload.get("worker_count") != EXP007_WORKER_COUNT:
        raise ValueError("RssSummary worker_count is invalid")
    values = [
        require_nonnegative_int(value, "RssSummary.worker_lifetime_bytes[]")
        for value in _require_sequence(
            payload.get("worker_lifetime_bytes"),
            "RssSummary.worker_lifetime_bytes",
        )
    ]
    if len(values) != EXP007_WORKER_COUNT:
        raise ValueError("RssSummary worker_lifetime_bytes length mismatch")
    if payload.get("arm_max_worker_bytes") != max(values):
        raise ValueError("RssSummary arm max mismatch")
    return {
        "worker_count": EXP007_WORKER_COUNT,
        "worker_lifetime_bytes": values,
        "arm_max_worker_bytes": payload["arm_max_worker_bytes"],
    }


def validate_source_arm_denominators(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "SourceArmDenominators")
    validate_exact_fields(payload, SOURCE_ARM_DENOMINATORS_FIELDS, "SourceArmDenominators")
    if payload.get("stage_audio_count") != 16:
        raise ValueError("SourceArmDenominators stage_audio_count is invalid")
    result = {"stage_audio_count": 16}
    for name in sorted(SOURCE_ARM_DENOMINATORS_FIELDS - {"stage_audio_count"}):
        result[name] = validate_audio_set_binding(payload.get(name))
    if result["stage_audio"]["count"] != 16:
        raise ValueError("SourceArmDenominators stage audio count is invalid")
    if result["cache_valid_audio"]["count"] != 16 or result["projection_evaluable_audio"]["count"] != 16:
        raise ValueError("SourceArmDenominators valid/evaluable counts are invalid")
    if (
        result["candidate_accepted_audio"]["count"]
        + result["candidate_fallback_audio"]["count"]
        != 16
    ):
        raise ValueError("SourceArmDenominators candidate partition mismatch")
    return {name: result[name] for name in SOURCE_ARM_DENOMINATORS_FIELDS}


def validate_source_arm_gates(
    payload: Mapping[str, Any],
    *,
    denominators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_mapping(payload, "SourceArmGates")
    validate_exact_fields(payload, SOURCE_ARM_GATES_FIELDS, "SourceArmGates")
    candidate_rate = validate_rate_value(payload.get("candidate_fallback_rate"))
    selected_rate = validate_rate_value(payload.get("selected_product_fallback_rate"))
    no_origin_rate = validate_rate_value(payload.get("no_origin_or_path_rate"))
    if candidate_rate["denominator"] != 16 or selected_rate["denominator"] != 16 or no_origin_rate["denominator"] != 16:
        raise ValueError("SourceArmGates rates must use denominator 16")
    runtime = validate_stats_value(payload.get("runtime_seconds"))
    worker_rss = validate_stats_value(payload.get("worker_rss_bytes"))
    candidate_seam = validate_stats_value(payload.get("candidate_seam_ms"))
    section_count = validate_stats_value(payload.get("candidate_section_count"))
    row_bytes = validate_stats_value(payload.get("row_json_bytes"))
    if runtime["count"] != 16 or row_bytes["count"] != 16 or worker_rss["count"] != 4:
        raise ValueError("SourceArmGates stats counts are invalid")
    result = {
        "candidate_fallback_rate": candidate_rate,
        "selected_product_fallback_rate": selected_rate,
        "no_origin_or_path_rate": no_origin_rate,
        "runtime_seconds": runtime,
        "worker_rss_bytes": worker_rss,
        "candidate_seam_ms": candidate_seam,
        "candidate_section_count": section_count,
        "row_json_bytes": row_bytes,
    }
    for name in (
        "every_row_under_180_seconds",
        "seam_zero",
        "section_cap_valid",
        "row_byte_cap_valid",
        "replay_schema_source_cache_candidate_v2_consistent",
    ):
        value = payload.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"SourceArmGates.{name} must be a bool")
        result[name] = value
    if denominators is not None:
        expected_candidate = denominators["candidate_fallback_audio"]["count"]
        if candidate_rate["numerator"] != float(expected_candidate):
            raise ValueError("SourceArmGates candidate fallback numerator mismatch")
        expected_selected = denominators["selected_product_fallback_audio"]["count"]
        if selected_rate["numerator"] != float(expected_selected):
            raise ValueError("SourceArmGates selected fallback numerator mismatch")
        expected_no_origin = denominators["no_origin_or_path_audio"]["count"]
        if no_origin_rate["numerator"] != float(expected_no_origin):
            raise ValueError("SourceArmGates no-origin numerator mismatch")
    return {name: result[name] for name in SOURCE_ARM_GATES_FIELDS}


def validate_repair80_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "Repair80Summary")
    validate_exact_fields(payload, REPAIR80_SUMMARY_FIELDS, "Repair80Summary")
    if payload.get("schema") != REPAIR80_SUMMARY_SCHEMA:
        raise ValueError("Repair80Summary schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("Repair80Summary experiment_id is invalid")
    if payload.get("stage") != EXP007_REPAIR_STAGE:
        raise ValueError("Repair80Summary stage is invalid")
    _require_descriptor(payload, REPAIR80_SUMMARY_SCHEMA, "Repair80Summary")
    arm = _require_schedule_arm(payload.get("schedule_arm"), "Repair80Summary.schedule_arm")
    for name in (
        "four_arm_stage_summary_sha256",
        "candidate_global_manifest_sha256",
        "source_selection_sha256",
        "schedule_weak_veto_outcome_sha256",
        "run_config_fingerprint_sha256",
        "source_closure_fingerprint_sha256",
        "repair_input_binding_sha256",
        "candidate_reference_manifest_sha256",
    ):
        require_sha256(payload.get(name), f"Repair80Summary.{name}")
    identity_source = validate_source_ref(payload.get("repair_identity_source"))
    label_source = validate_source_ref(payload.get("repair_label_source"))
    if identity_source["row_count"] != 80 or label_source["row_count"] != 80:
        raise ValueError("Repair80Summary sources must have row_count 80")
    row_refs = [
        validate_completed_row_ref(ref)
        for ref in _require_sequence(payload.get("row_refs"), "Repair80Summary.row_refs")
    ]
    weak_refs = [
        validate_weak_row_ref(ref)
        for ref in _require_sequence(
            payload.get("weak_row_refs"),
            "Repair80Summary.weak_row_refs",
        )
    ]
    if payload.get("row_count") != 80 or len(row_refs) != 80:
        raise ValueError("Repair80Summary row_count is invalid")
    if payload.get("weak_row_count") != 80 or len(weak_refs) != 80:
        raise ValueError("Repair80Summary weak_row_count is invalid")
    _validate_completed_ref_order(row_refs, context="Repair80Summary.row_refs")
    _validate_contiguous_ref_order(weak_refs, context="Repair80Summary.weak_row_refs")
    if payload.get("row_payloads_sha256") != canonical_json_sha256(row_refs):
        raise ValueError("Repair80Summary row_payloads_sha256 mismatch")
    if payload.get("weak_payloads_sha256") != canonical_json_sha256(weak_refs):
        raise ValueError("Repair80Summary weak_payloads_sha256 mismatch")
    pair_tuples = []
    for row_ref, weak_ref in zip(row_refs, weak_refs, strict=True):
        if row_ref["row_index"] != weak_ref["row_index"] or row_ref["cache_audio_key"] != weak_ref["cache_audio_key"]:
            raise ValueError("Repair80Summary row/weak ref order mismatch")
        if row_ref["row_payload_sha256"] != weak_ref["prediction_row_sha256"]:
            raise ValueError("Repair80Summary weak prediction row SHA mismatch")
        pair_tuples.append(
            {
                "row_index": row_ref["row_index"],
                "cache_audio_key": row_ref["cache_audio_key"],
                "row_payload_sha256": row_ref["row_payload_sha256"],
                "prediction_row_sha256": weak_ref["prediction_row_sha256"],
                "weak_row_payload_sha256": weak_ref["weak_row_payload_sha256"],
            }
        )
    if payload.get("row_weak_pairs_sha256") != canonical_json_sha256(pair_tuples):
        raise ValueError("Repair80Summary row_weak_pairs_sha256 mismatch")
    denominators = validate_repair80_denominators(payload.get("denominators"))
    gates = validate_repair80_gates(payload.get("gates"), denominators=denominators)
    decision = payload.get("decision")
    action = payload.get("action")
    expected_action = {
        "pass": "write_result_and_next_no_data_card",
        "ambiguous": "stop_ambiguous",
        "negative": "stop_negative",
    }.get(decision)
    if expected_action is None or action != expected_action:
        raise ValueError("Repair80Summary decision/action mismatch")
    runtime_summary = validate_runtime_summary(payload.get("runtime_summary"))
    rss_summary = validate_rss_summary(payload.get("rss_summary"))
    result = {
        "schema": REPAIR80_SUMMARY_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": EXP007_REPAIR_STAGE,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "schedule_arm": arm,
        "four_arm_stage_summary_sha256": payload["four_arm_stage_summary_sha256"],
        "candidate_global_manifest_sha256": payload["candidate_global_manifest_sha256"],
        "source_selection_sha256": payload["source_selection_sha256"],
        "schedule_weak_veto_outcome_sha256": payload[
            "schedule_weak_veto_outcome_sha256"
        ],
        "run_config_fingerprint_sha256": payload["run_config_fingerprint_sha256"],
        "source_closure_fingerprint_sha256": payload["source_closure_fingerprint_sha256"],
        "repair_input_binding_sha256": payload["repair_input_binding_sha256"],
        "repair_identity_source": identity_source,
        "repair_label_source": label_source,
        "candidate_reference_manifest_sha256": payload[
            "candidate_reference_manifest_sha256"
        ],
        "row_count": 80,
        "row_refs": row_refs,
        "row_payloads_sha256": payload["row_payloads_sha256"],
        "weak_row_count": 80,
        "weak_row_refs": weak_refs,
        "weak_payloads_sha256": payload["weak_payloads_sha256"],
        "row_weak_pairs_sha256": payload["row_weak_pairs_sha256"],
        "denominators": denominators,
        "gates": gates,
        "decision": decision,
        "action": action,
        "runtime_summary": runtime_summary,
        "rss_summary": rss_summary,
        "summary_fingerprint_sha256": require_sha256(
            payload.get("summary_fingerprint_sha256"),
            "Repair80Summary.summary_fingerprint_sha256",
        ),
    }
    validate_payload_hash(result, "summary_fingerprint_sha256", context="Repair80Summary")
    return result


def validate_repair80_denominators(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "Repair80Denominators")
    validate_exact_fields(payload, REPAIR80_DENOMINATORS_FIELDS, "Repair80Denominators")
    if payload.get("stage_audio_count") != 80:
        raise ValueError("Repair80Denominators stage_audio_count is invalid")
    result = {"stage_audio_count": 80}
    for name in sorted(REPAIR80_DENOMINATORS_FIELDS - {"stage_audio_count"}):
        result[name] = validate_audio_set_binding(payload.get(name))
    if result["stage_audio"]["count"] != 80:
        raise ValueError("Repair80Denominators stage_audio count is invalid")
    if result["cache_valid_audio"]["count"] != 80 or result["projection_evaluable_audio"]["count"] != 80:
        raise ValueError("Repair80Denominators valid/evaluable counts are invalid")
    if (
        result["candidate_accepted_audio"]["count"]
        + result["candidate_fallback_audio"]["count"]
        != 80
    ):
        raise ValueError("Repair80Denominators candidate partition mismatch")
    return {name: result[name] for name in REPAIR80_DENOMINATORS_FIELDS}


def validate_repair80_gates(
    payload: Mapping[str, Any],
    *,
    denominators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_mapping(payload, "Repair80Gates")
    validate_exact_fields(payload, REPAIR80_GATES_FIELDS, "Repair80Gates")
    result = {
        "candidate_fallback_rate": validate_rate_value(
            payload.get("candidate_fallback_rate")
        ),
        "selected_product_fallback_rate": validate_rate_value(
            payload.get("selected_product_fallback_rate")
        ),
        "no_origin_or_path_rate": validate_rate_value(
            payload.get("no_origin_or_path_rate")
        ),
        "runtime_seconds": validate_stats_value(payload.get("runtime_seconds")),
        "worker_rss_bytes": validate_stats_value(payload.get("worker_rss_bytes")),
        "overlap_ms": validate_stats_value(payload.get("overlap_ms")),
        "stable_section_excess": validate_stats_value(
            payload.get("stable_section_excess")
        ),
        "pure_mean_phase_ratio": validate_ratio_value(
            payload.get("pure_mean_phase_ratio")
        ),
        "pure_p90_phase_ratio": validate_ratio_value(payload.get("pure_p90_phase_ratio")),
        "pure_phase_coverage": validate_coverage_value(
            payload.get("pure_phase_coverage")
        ),
        "current_v2_phase_mean_ms": _validate_nullable_finite_number(
            payload.get("current_v2_phase_mean_ms"),
            "Repair80Gates.current_v2_phase_mean_ms",
        ),
        "pure_exp006_phase_mean_ms": _validate_nullable_finite_number(
            payload.get("pure_exp006_phase_mean_ms"),
            "Repair80Gates.pure_exp006_phase_mean_ms",
        ),
        "current_v2_phase_p90_ms": _validate_nullable_finite_number(
            payload.get("current_v2_phase_p90_ms"),
            "Repair80Gates.current_v2_phase_p90_ms",
        ),
        "pure_exp006_phase_p90_ms": _validate_nullable_finite_number(
            payload.get("pure_exp006_phase_p90_ms"),
            "Repair80Gates.pure_exp006_phase_p90_ms",
        ),
        "stable_phase_mean_ratio": validate_ratio_value(
            payload.get("stable_phase_mean_ratio")
        ),
        "stable_phase_p90_ratio": validate_ratio_value(
            payload.get("stable_phase_p90_ratio")
        ),
        "jump_phase_mean_ratio": validate_ratio_value(
            payload.get("jump_phase_mean_ratio")
        ),
        "current_v2_jump_alias_drift_mean_ms": _validate_nullable_finite_number(
            payload.get("current_v2_jump_alias_drift_mean_ms"),
            "Repair80Gates.current_v2_jump_alias_drift_mean_ms",
        ),
        "pure_exp006_jump_alias_drift_mean_ms": _validate_nullable_finite_number(
            payload.get("pure_exp006_jump_alias_drift_mean_ms"),
            "Repair80Gates.pure_exp006_jump_alias_drift_mean_ms",
        ),
        "jump_alias_drift_mean_ratio": validate_ratio_value(
            payload.get("jump_alias_drift_mean_ratio")
        ),
        "current_v2_long_alias_drift_mean_ms": _validate_nullable_finite_number(
            payload.get("current_v2_long_alias_drift_mean_ms"),
            "Repair80Gates.current_v2_long_alias_drift_mean_ms",
        ),
        "pure_exp006_long_alias_drift_mean_ms": _validate_nullable_finite_number(
            payload.get("pure_exp006_long_alias_drift_mean_ms"),
            "Repair80Gates.pure_exp006_long_alias_drift_mean_ms",
        ),
        "current_v2_long_alias_drift_p90_ms": _validate_nullable_finite_number(
            payload.get("current_v2_long_alias_drift_p90_ms"),
            "Repair80Gates.current_v2_long_alias_drift_p90_ms",
        ),
        "pure_exp006_long_alias_drift_p90_ms": _validate_nullable_finite_number(
            payload.get("pure_exp006_long_alias_drift_p90_ms"),
            "Repair80Gates.pure_exp006_long_alias_drift_p90_ms",
        ),
        "long_alias_drift_mean_ratio": validate_ratio_value(
            payload.get("long_alias_drift_mean_ratio")
        ),
        "long_alias_drift_p90_ratio": validate_ratio_value(
            payload.get("long_alias_drift_p90_ratio")
        ),
        "current_v2_boundary_f1_mean": _validate_nullable_finite_number(
            payload.get("current_v2_boundary_f1_mean"),
            "Repair80Gates.current_v2_boundary_f1_mean",
        ),
        "pure_exp006_boundary_f1_mean": _validate_nullable_finite_number(
            payload.get("pure_exp006_boundary_f1_mean"),
            "Repair80Gates.pure_exp006_boundary_f1_mean",
        ),
        "selected_boundary_f1_mean": _validate_nullable_finite_number(
            payload.get("selected_boundary_f1_mean"),
            "Repair80Gates.selected_boundary_f1_mean",
        ),
    }
    result["pure_minus_v2_boundary_f1_delta"] = _validate_nullable_finite_number(
        payload.get("pure_minus_v2_boundary_f1_delta"),
        "Repair80Gates.pure_minus_v2_boundary_f1_delta",
    )
    for name in (
        "every_row_under_180_seconds",
        "seam_zero",
        "section_cap_valid",
        "replay_schema_source_cache_integrity",
    ):
        value = payload.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"Repair80Gates {name} must be a bool")
        result[name] = value
    if result["candidate_fallback_rate"]["denominator"] != 80:
        raise ValueError("Repair80Gates candidate fallback denominator must be 80")
    if result["selected_product_fallback_rate"]["denominator"] != 80:
        raise ValueError("Repair80Gates selected fallback denominator must be 80")
    if result["no_origin_or_path_rate"]["denominator"] != 80:
        raise ValueError("Repair80Gates no-origin denominator must be 80")
    if result["runtime_seconds"]["count"] != 80:
        raise ValueError("Repair80Gates runtime count must be 80")
    if result["worker_rss_bytes"]["count"] != EXP007_WORKER_COUNT:
        raise ValueError("Repair80Gates worker RSS count mismatch")
    if denominators is not None:
        denominator_payload = validate_repair80_denominators(denominators)
        _validate_repair80_gate_denominator_bindings(result, denominator_payload)
    return {name: result[name] for name in REPAIR80_GATES_FIELDS}


def _validate_repair80_gate_denominator_bindings(
    gates: Mapping[str, Any],
    denominators: Mapping[str, Any],
) -> None:
    if gates["candidate_fallback_rate"]["numerator"] != float(
        denominators["candidate_fallback_audio"]["count"]
    ):
        raise ValueError("Repair80Gates candidate fallback numerator mismatch")
    if gates["selected_product_fallback_rate"]["numerator"] != float(
        denominators["selected_product_fallback_audio"]["count"]
    ):
        raise ValueError("Repair80Gates selected fallback numerator mismatch")
    if gates["no_origin_or_path_rate"]["numerator"] != float(
        denominators["no_origin_or_path_audio"]["count"]
    ):
        raise ValueError("Repair80Gates no-origin numerator mismatch")
    if gates["overlap_ms"]["count"] != denominators["overlap_available_audio"]["count"]:
        raise ValueError("Repair80Gates overlap count mismatch")
    if (
        gates["stable_section_excess"]["count"]
        != denominators["stable_pure_paired"]["count"]
    ):
        raise ValueError("Repair80Gates stable section count mismatch")
    current_count = denominators["current_v2_phase_matched"]["count"]
    pure_count = denominators["pure_exp006_phase_matched"]["count"]
    coverage = gates["pure_phase_coverage"]
    if current_count == 0:
        if coverage.get("state") != "undefined":
            raise ValueError("Repair80Gates pure coverage must be undefined")
    elif coverage.get("denominator") != float(current_count) or coverage.get(
        "numerator"
    ) != float(pure_count):
        raise ValueError("Repair80Gates pure coverage operands mismatch")
    _require_repair80_ratio_and_raw_fields(
        gates,
        count=denominators["phase_common"]["count"],
        minimum=40,
        ratio_fields=("pure_mean_phase_ratio", "pure_p90_phase_ratio"),
        raw_fields=(
            "current_v2_phase_mean_ms",
            "pure_exp006_phase_mean_ms",
            "current_v2_phase_p90_ms",
            "pure_exp006_phase_p90_ms",
        ),
        context="phase",
    )
    _require_repair80_ratio_and_raw_fields(
        gates,
        count=denominators["stable_pure_paired"]["count"],
        minimum=5,
        ratio_fields=("stable_phase_mean_ratio", "stable_phase_p90_ratio"),
        raw_fields=(),
        context="stable phase",
    )
    _require_repair80_ratio_and_raw_fields(
        gates,
        count=denominators["jump_pure_paired"]["count"],
        minimum=15,
        ratio_fields=("jump_phase_mean_ratio",),
        raw_fields=(),
        context="jump phase",
    )
    _require_repair80_ratio_and_raw_fields(
        gates,
        count=denominators["jump_alias_drift_common"]["count"],
        minimum=15,
        ratio_fields=("jump_alias_drift_mean_ratio",),
        raw_fields=(
            "current_v2_jump_alias_drift_mean_ms",
            "pure_exp006_jump_alias_drift_mean_ms",
        ),
        context="jump drift",
    )
    _require_repair80_ratio_and_raw_fields(
        gates,
        count=denominators["long_alias_drift_common"]["count"],
        minimum=5,
        ratio_fields=(
            "long_alias_drift_mean_ratio",
            "long_alias_drift_p90_ratio",
        ),
        raw_fields=(
            "current_v2_long_alias_drift_mean_ms",
            "pure_exp006_long_alias_drift_mean_ms",
            "current_v2_long_alias_drift_p90_ms",
            "pure_exp006_long_alias_drift_p90_ms",
        ),
        context="long drift",
    )
    boundary_fields = (
        "current_v2_boundary_f1_mean",
        "pure_exp006_boundary_f1_mean",
        "selected_boundary_f1_mean",
        "pure_minus_v2_boundary_f1_delta",
    )
    if denominators["repair_boundary_common"]["count"] >= 15:
        if any(gates[name] is None for name in boundary_fields):
            raise ValueError("Repair80Gates boundary fields must be finite")
        expected_delta = (
            gates["pure_exp006_boundary_f1_mean"]
            - gates["current_v2_boundary_f1_mean"]
        )
        if gates["pure_minus_v2_boundary_f1_delta"] != expected_delta:
            raise ValueError("Repair80Gates boundary delta mismatch")
    elif any(gates[name] is not None for name in boundary_fields):
        raise ValueError("Repair80Gates boundary fields must be null")


def _require_repair80_ratio_and_raw_fields(
    gates: Mapping[str, Any],
    *,
    count: int,
    minimum: int,
    ratio_fields: Sequence[str],
    raw_fields: Sequence[str],
    context: str,
) -> None:
    if count < minimum:
        for name in ratio_fields:
            if gates[name]["state"] != "undefined":
                raise ValueError(f"Repair80Gates {context} ratio must be undefined")
        for name in raw_fields:
            if gates[name] is not None:
                raise ValueError(f"Repair80Gates {context} raw fields must be null")
        return
    for name in ratio_fields:
        if gates[name]["state"] == "undefined":
            raise ValueError(f"Repair80Gates {context} ratio must be defined")
    for name in raw_fields:
        if gates[name] is None:
            raise ValueError(f"Repair80Gates {context} raw fields must be finite")


def _validate_nullable_finite_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return require_finite_number(value, field_name)


def expected_row_count_for_stage(stage: str) -> int:
    stage = _require_stage(stage, "stage")
    return 16 if stage == EXP007_SCHEDULE_STAGE else 80


def reject_forbidden_selector_fields(value: Any, *, context: str) -> None:
    if isinstance(value, MappingABC):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered, tokens, compact = _selector_key_forms(key)
            if (
                lowered in FORBIDDEN_SELECTOR_FIELD_EXACT
                or any(token in FORBIDDEN_SELECTOR_FIELD_EXACT for token in tokens)
                or compact in _FORBIDDEN_SELECTOR_FIELD_EXACT_COMPACT
                or any(
                    marker in compact
                    for marker in _FORBIDDEN_SELECTOR_FIELD_SUBSTRING_COMPACT
                )
            ):
                raise ValueError(f"forbidden selector field {context}.{key}")
            reject_forbidden_selector_fields(child, context=f"{context}.{key}")
    elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            reject_forbidden_selector_fields(child, context=f"{context}[{index}]")


def _default_candidate_policy() -> dict[str, Any]:
    return {
        "restricted_fields": ["beat_prob", "downbeat_prob", "frame_rate_hz"],
        "extract_exactly_once": True,
        "explicit_candidate_argument": True,
        "canonical_candidate_schema": (
            "pulsefield_model.timing_v3_exp007_candidate_payload_v1"
        ),
    }


def _default_pool_policy() -> dict[str, Any]:
    return {
        "worker_count": EXP007_WORKER_COUNT,
        "start_method": EXP007_WORKER_START_METHOD,
        "imap_chunksize": EXP007_IMAP_CHUNKSIZE,
        "maxtasksperchild": EXP007_MAXTASKSPERCHILD,
        "fixed_input_order": True,
        "arm_execution_order": list(EXP007_EXECUTION_ORDER),
    }


def _default_limits() -> dict[str, Any]:
    return {
        "per_audio_arm_timeout_seconds": EXP007_PER_AUDIO_ARM_TIMEOUT_S,
        "schedule_four_arm_stop_seconds": EXP007_SCHEDULE_FOUR_ARM_STOP_S,
        "repair_stop_seconds": EXP007_REPAIR_STOP_S,
        "worker_rss_cap_bytes": EXP007_WORKER_RSS_CAP_BYTES,
        "row_json_byte_cap": EXP007_ROW_JSON_BYTE_CAP,
        "candidate_payload_byte_cap": EXP007_CANDIDATE_PAYLOAD_BYTE_CAP,
        "candidate_bundle_byte_cap": EXP007_CANDIDATE_BUNDLE_BYTE_CAP,
        "candidate_reference_manifest_byte_cap": (
            EXP007_CANDIDATE_REFERENCE_MANIFEST_BYTE_CAP
        ),
        "candidate_global_manifest_byte_cap": EXP007_CANDIDATE_GLOBAL_MANIFEST_BYTE_CAP,
        "parent_poll_max_seconds": EXP007_PARENT_POLL_MAX_SECONDS,
        "finish_result_delivery_seconds": EXP007_FINISH_RESULT_DELIVERY_S,
        "worker_terminate_grace_seconds": EXP007_WORKER_TERMINATE_GRACE_S,
        "worker_kill_grace_seconds": EXP007_WORKER_KILL_GRACE_S,
    }


def _validate_method_ids(payload: Any) -> None:
    _require_mapping(payload, "MethodIds")
    validate_exact_fields(payload, frozenset(METHOD_IDS), "MethodIds")
    if dict(payload) != METHOD_IDS:
        raise ValueError("MethodIds values are invalid")


def _validate_candidate_policy(payload: Any) -> None:
    _require_mapping(payload, "CandidatePolicy")
    expected = _default_candidate_policy()
    validate_exact_fields(payload, frozenset(expected), "CandidatePolicy")
    if dict(payload) != expected:
        raise ValueError("CandidatePolicy values are invalid")


def _validate_pool_policy(payload: Any) -> None:
    _require_mapping(payload, "PoolPolicy")
    expected = _default_pool_policy()
    validate_exact_fields(payload, frozenset(expected), "PoolPolicy")
    if dict(payload) != expected:
        raise ValueError("PoolPolicy values are invalid")


def _validate_limits(payload: Any) -> None:
    _require_mapping(payload, "LimitPolicy")
    expected = _default_limits()
    validate_exact_fields(payload, frozenset(expected), "LimitPolicy")
    if dict(payload) != expected:
        raise ValueError("LimitPolicy values are invalid")


def _validate_local_frontier_config(payload: Any, *, arm: str) -> None:
    _require_mapping(payload, "LocalFrontierConfigPayload")
    expected = make_local_frontier_config(arm)
    validate_exact_fields(payload, LOCAL_FRONTIER_CONFIG_FIELDS, "LocalFrontierConfigPayload")
    if dict(payload) != expected:
        raise ValueError("LocalFrontierConfigPayload values are invalid")


def _validate_worker_rss_snapshot(payload: Any) -> None:
    _require_mapping(payload, "WorkerRssFailureSnapshot")
    validate_exact_fields(
        payload,
        WORKER_RSS_FAILURE_SNAPSHOT_FIELDS,
        "WorkerRssFailureSnapshot",
    )
    values = _require_sequence(
        payload.get("worker_slot_lifetime_bytes"),
        "worker_slot_lifetime_bytes",
    )
    if len(values) != 4:
        raise ValueError("WorkerRssFailureSnapshot must have four slots")
    non_null: list[int] = []
    for value in values:
        if value is None:
            continue
        non_null.append(require_nonnegative_int(value, "worker_slot_lifetime_bytes[]"))
    observed = payload.get("observed_worker_max_bytes")
    if non_null:
        if observed != max(non_null):
            raise ValueError("WorkerRssFailureSnapshot observed max mismatch")
    elif observed is not None:
        raise ValueError("WorkerRssFailureSnapshot observed max must be null")


def _validate_failure_prefix_order(
    *,
    completed: Sequence[Mapping[str, Any]],
    pending: Sequence[Mapping[str, Any]],
    expected_count: int,
    failure_kind: str,
    causing_row_index: Any,
    causing_cache_audio_key: Any,
) -> None:
    expected_completed_indexes = list(range(len(completed)))
    actual_completed_indexes = [row["row_index"] for row in completed]
    if actual_completed_indexes != expected_completed_indexes:
        raise ValueError("ArmFailureRecord completed prefix must be contiguous")
    expected_pending_indexes = list(range(len(completed), expected_count))
    actual_pending_indexes = [row["row_index"] for row in pending]
    if actual_pending_indexes != expected_pending_indexes:
        raise ValueError("ArmFailureRecord pending suffix must be contiguous")
    completed_keys = [row["cache_audio_key"] for row in completed]
    pending_keys = [row["cache_audio_key"] for row in pending]
    all_keys = completed_keys + pending_keys
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("ArmFailureRecord cache_audio_key values must be unique")
    identity_hashes = [
        row["identity_payload_sha256"] for row in list(completed) + list(pending)
    ]
    if len(set(identity_hashes)) != len(identity_hashes):
        raise ValueError("ArmFailureRecord identity payload hashes must be unique")
    if causing_row_index is None:
        return
    row_index = require_nonnegative_int(causing_row_index, "causing_row_index")
    cache_key = require_nonempty_string(causing_cache_audio_key, "causing_cache_audio_key")
    matching_pending = [
        row
        for row in pending
        if row["row_index"] == row_index and row["cache_audio_key"] == cache_key
    ]
    if matching_pending:
        return
    matching_completed = [
        row
        for row in completed
        if row["row_index"] == row_index and row["cache_audio_key"] == cache_key
    ]
    if matching_completed:
        if failure_kind == "row_timeout":
            raise ValueError("ArmFailureRecord row_timeout causing row must be pending")
        return
    if row_index >= expected_count:
        raise ValueError("ArmFailureRecord causing row is outside expected rows")
    raise ValueError(
        "ArmFailureRecord causing row must be in completed or pending identities"
    )


def _validate_causing_fields(payload: Mapping[str, Any]) -> None:
    row = payload.get("causing_row_index")
    if row is None:
        for name in (
            "causing_cache_audio_key",
            "causing_worker_slot",
            "causing_worker_generation_nonce",
            "causing_worker_pid",
            "causing_dispatch_token",
            "causing_worker_rss_bytes",
        ):
            if payload.get(name) is not None:
                raise ValueError("ArmFailureRecord causing fields must all be null")
        return
    require_nonnegative_int(row, "causing_row_index")
    require_nonempty_string(payload.get("causing_cache_audio_key"), "causing_cache_audio_key")
    require_nonnegative_int(payload.get("causing_worker_slot"), "causing_worker_slot")
    require_sha256(
        payload.get("causing_worker_generation_nonce"),
        "causing_worker_generation_nonce",
    )
    require_positive_int(payload.get("causing_worker_pid"), "causing_worker_pid")
    require_sha256(payload.get("causing_dispatch_token"), "causing_dispatch_token")
    rss = payload.get("causing_worker_rss_bytes")
    if rss is not None:
        require_nonnegative_int(rss, "causing_worker_rss_bytes")


def _validate_common_outcome(payload: Mapping[str, Any], *, expected_status: str) -> None:
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("ArmStageOutcome experiment_id is invalid")
    if payload.get("status") != expected_status:
        raise ValueError("ArmStageOutcome status is invalid")
    _require_stage(payload.get("stage"), "ArmStageOutcome.stage")
    _require_schedule_arm(payload.get("schedule_arm"), "ArmStageOutcome.schedule_arm")
    _require_descriptor(payload, payload["schema"], "ArmStageOutcome")


def _validate_arm_outcome_sha_map(payload: Any) -> None:
    _require_mapping(payload, "ArmOutcomeShaMap")
    validate_exact_fields(payload, ARM_OUTCOME_SHA_MAP_FIELDS, "ArmOutcomeShaMap")
    for arm in EXP007_EXECUTION_ORDER:
        require_sha256(payload.get(arm), f"ArmOutcomeShaMap.{arm}")


def _validate_arm_order_value(
    payload: Any,
    *,
    expected_arm: str,
    common_ready: bool,
    overlap_common_count: int,
    section_common_count: int,
) -> dict[str, Any]:
    _require_mapping(payload, "ArmOrderValues")
    validate_exact_fields(payload, ARM_ORDER_VALUES_FIELDS, "ArmOrderValues")
    arm = _require_schedule_arm(payload.get("schedule_arm"), "ArmOrderValues.schedule_arm")
    if arm != expected_arm:
        raise ValueError("ArmOrderValues arm order is invalid")
    if not isinstance(payload.get("e0_eligible"), bool) or not isinstance(payload.get("e1_eligible"), bool):
        raise ValueError("ArmOrderValues eligibility values must be bool")
    candidate_fallback_count = require_nonnegative_int(
        payload.get("candidate_fallback_count"),
        "ArmOrderValues.candidate_fallback_count",
    )
    no_origin_or_path_count = require_nonnegative_int(
        payload.get("no_origin_or_path_count"),
        "ArmOrderValues.no_origin_or_path_count",
    )
    tie_rank = require_nonnegative_int(payload.get("tie_rank"), "ArmOrderValues.tie_rank")
    if payload.get("tie_rank") != EXP007_TIE_RANK[arm]:
        raise ValueError("ArmOrderValues tie_rank is invalid")
    p90_runtime = require_finite_number(
        payload.get("p90_runtime"),
        "ArmOrderValues.p90_runtime",
    )
    max_worker_rss = require_nonnegative_int(
        payload.get("max_worker_rss"),
        "ArmOrderValues.max_worker_rss",
    )
    reasons = _validate_string_sequence(
        payload.get("elimination_reasons"),
        "elimination_reasons",
    )
    allowed_order = {reason: index for index, reason in enumerate(EXP007_ELIMINATION_REASONS_ORDER)}
    previous = -1
    for reason in reasons:
        if reason not in allowed_order:
            raise ValueError("ArmOrderValues elimination_reasons contains unknown reason")
        current = allowed_order[reason]
        if current <= previous:
            raise ValueError("ArmOrderValues elimination_reasons order is invalid")
        previous = current
    hard_reason_set = {
        "runtime_nonfinite",
        "row_timeout_guard",
        "rss_nonfinite",
        "seam_guard",
        "section_cap_guard",
        "row_consistency_guard",
    }
    retained_hard_reasons = [reason for reason in reasons if reason in hard_reason_set]
    derived_reasons: list[str] = []
    if candidate_fallback_count > 1:
        derived_reasons.append("candidate_fallback_guard")
    if no_origin_or_path_count > 0:
        derived_reasons.append("no_origin_or_path_guard")
    derived_reasons.extend(retained_hard_reasons)
    if p90_runtime > 60.0:
        derived_reasons.append("runtime_p90_guard")
    if max_worker_rss > EXP007_WORKER_RSS_CAP_BYTES:
        derived_reasons.append("rss_cap_guard")
    derived_reasons = _sort_elimination_reasons(derived_reasons)
    e0_expected = not any(
        reason
        in {
            "candidate_fallback_guard",
            "no_origin_or_path_guard",
            "runtime_nonfinite",
            "runtime_p90_guard",
            "row_timeout_guard",
            "rss_nonfinite",
            "rss_cap_guard",
            "seam_guard",
            "section_cap_guard",
            "row_consistency_guard",
        }
        for reason in derived_reasons
    )
    if payload.get("e0_eligible") != e0_expected:
        raise ValueError("ArmOrderValues e0_eligible mismatch")
    if e0_expected and not common_ready:
        if overlap_common_count < 5:
            derived_reasons.append("overlap_common_minimum")
        if section_common_count < 8:
            derived_reasons.append("section_common_minimum")
        derived_reasons = _sort_elimination_reasons(derived_reasons)
    p90_overlap = payload.get("p90_overlap_ms")
    section_violation_count = payload.get("section_inflation_violation_count")
    p90_section_excess = payload.get("p90_section_excess")
    if e0_expected and common_ready:
        p90_overlap = require_finite_number(
            p90_overlap,
            "ArmOrderValues.p90_overlap_ms",
        )
        section_violation_count = require_nonnegative_int(
            section_violation_count,
            "ArmOrderValues.section_inflation_violation_count",
        )
        p90_section_excess = require_finite_number(
            p90_section_excess,
            "ArmOrderValues.p90_section_excess",
        )
        if p90_overlap > 90.0:
            derived_reasons.append("overlap_e1_guard")
            derived_reasons = _sort_elimination_reasons(derived_reasons)
    else:
        if (
            p90_overlap is not None
            or section_violation_count is not None
            or p90_section_excess is not None
        ):
            raise ValueError("ArmOrderValues E0/common null matrix mismatch")
        p90_overlap = None
        section_violation_count = None
        p90_section_excess = None
    e1_expected = e0_expected and common_ready and p90_overlap is not None and p90_overlap <= 90.0
    if payload.get("e1_eligible") != e1_expected:
        raise ValueError("ArmOrderValues e1_eligible mismatch")
    if reasons != derived_reasons:
        raise ValueError("ArmOrderValues elimination_reasons mismatch")
    order_tuple = None
    order_tuple_sha = payload.get("order_tuple_sha256")
    if e1_expected:
        order_tuple = [
            candidate_fallback_count,
            no_origin_or_path_count,
            p90_overlap,
            section_violation_count,
            p90_section_excess,
            tie_rank,
        ]
        if order_tuple_sha != canonical_json_sha256(order_tuple):
            raise ValueError("ArmOrderValues order_tuple_sha256 mismatch")
    elif order_tuple_sha is not None:
        raise ValueError("ArmOrderValues order_tuple_sha256 must be null")
    result = dict(payload)
    result["_order_tuple"] = tuple(order_tuple) if order_tuple is not None else None
    return result


def _validate_four_arm_failure_details(payload: Any) -> None:
    _require_mapping(payload, "FourArmFailureDetails")
    validate_exact_fields(
        payload,
        FOUR_ARM_FAILURE_DETAILS_FIELDS,
        "FourArmFailureDetails",
    )
    if payload.get("failure_kind") not in {
        "arm_hard_failure",
        "schedule_deadline",
        "cross_arm_identity_mismatch",
        "cross_arm_cache_mismatch",
        "cross_arm_source_config_mismatch",
        "cross_arm_restricted_input_mismatch",
        "cross_arm_candidate_mismatch",
        "cross_arm_current_v2_mismatch",
        "cross_arm_schema_mismatch",
        "candidate_global_publication_failure",
    }:
        raise ValueError("FourArmFailureDetails failure_kind is invalid")
    require_nonnegative_int(
        payload.get("completed_success_arm_count"),
        "completed_success_arm_count",
    )
    if payload["completed_success_arm_count"] > 4:
        raise ValueError("completed_success_arm_count cannot exceed 4")
    for name in ("deterministic_failure_sha256", "full_failure_sha256"):
        require_sha256(payload.get(name), f"FourArmFailureDetails.{name}")


def _require_descriptor(payload: Mapping[str, Any], schema_id: str, context: str) -> None:
    expected = schema_descriptor_sha256(schema_id)
    actual = require_sha256(
        payload.get("schema_descriptor_sha256"),
        f"{context}.schema_descriptor_sha256",
    )
    if actual != expected:
        raise ValueError(f"{context}.schema_descriptor_sha256 mismatch")


def _require_stage(value: Any, field_name: str) -> str:
    if value not in STAGES:
        raise ValueError(f"{field_name} must be schedule16 or repair80")
    return str(value)


def _require_schedule_arm(value: Any, field_name: str) -> str:
    if value not in SCHEDULE_ARM_SET:
        raise ValueError(f"{field_name} must be one of {EXP007_SCHEDULE_ARMS!r}")
    return str(value)


def require_relative_posix_path(value: Any, field_name: str) -> str:
    text = require_nonempty_string(value, field_name)
    if "\\" in text:
        raise ValueError(f"{field_name} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() != text:
        raise ValueError(f"{field_name} must be normalized POSIX-relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must not contain empty, '.', or '..'")
    return text


def require_lower_hex(value: Any, field_name: str) -> str:
    text = require_nonempty_string(value, field_name)
    if any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be lower-case hex")
    return text


def _require_candidate_payload_schema(value: Any, field_name: str) -> str:
    text = require_nonempty_string(value, field_name)
    if text != CANDIDATE_PAYLOAD_SCHEMA:
        raise ValueError(f"{field_name} is invalid")
    return text


def _row_deterministic_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    cache = validate_cache_identity(payload["cache_identity"])
    return {
        "schema": payload["schema"],
        "experiment_id": payload["experiment_id"],
        "stage": payload["stage"],
        "schedule_arm": payload["schedule_arm"],
        "row_index": payload["row_index"],
        "cache_audio_key": payload["cache_audio_key"],
        "audio_group_key": payload["audio_group_key"],
        "identity_payload_sha256": payload["identity_payload_sha256"],
        "cache_identity": {
            "relative_cache_path": cache["relative_cache_path"],
            "exists": cache["exists"],
            "size_bytes": cache["size_bytes"],
            "sha256": cache["sha256"],
            "cache_config_sha256": cache["cache_config_sha256"],
            "audio_cache_key_sha256": cache["audio_cache_key_sha256"],
        },
        "source_closure_fingerprint_sha256": payload[
            "source_closure_fingerprint_sha256"
        ],
        "run_config_fingerprint_sha256": payload["run_config_fingerprint_sha256"],
        "selector_manifest_sha256": payload["selector_manifest_sha256"],
        "input_manifest_sha256": payload["input_manifest_sha256"],
        "resume": payload["resume"],
        "restricted_prediction": payload["restricted_prediction"],
        "candidate_payload_schema": payload["candidate_payload_schema"],
        "candidate_payload_byte_count": payload["candidate_payload_byte_count"],
        "candidate_payload_field_set_sha256": payload[
            "candidate_payload_field_set_sha256"
        ],
        "candidate_payload_sha256": payload["candidate_payload_sha256"],
        "candidate_fingerprint": payload["candidate_fingerprint"],
        "methods": payload["methods"],
        "denominator_flags": payload["denominator_flags"],
        "diagnostics_summary": payload["diagnostics_summary"],
        "diagnostics_summary_sha256": payload["diagnostics_summary_sha256"],
    }


def _require_canonical_roundtrip(payload: Mapping[str, Any], context: str) -> None:
    if load_json_strict(canonical_json_bytes(payload)) != dict(payload):
        raise ValueError(f"{context} canonical round-trip mismatch")


def _validate_contiguous_ref_order(
    refs: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    indexes = [ref["row_index"] for ref in refs]
    if indexes != list(range(len(refs))):
        raise ValueError(f"{context} refs must be contiguous from zero")
    keys = [ref["cache_audio_key"] for ref in refs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{context} refs must have unique cache_audio_key values")


def _validate_completed_ref_order(
    refs: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    _validate_contiguous_ref_order(refs, context=context)
    hashes = [ref["row_payload_sha256"] for ref in refs]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{context} refs must have unique row payload hashes")


def _validate_sorted_unique_strings(value: Any, *, context: str) -> list[str]:
    result = [
        require_nonempty_string(item, f"{context}[]")
        for item in _require_sequence(value, context)
    ]
    if result != sorted(result) or len(set(result)) != len(result):
        raise ValueError(f"{context} must be sorted and unique")
    return result


def _validate_relative_source_file_list(
    value: Any,
    *,
    context: str,
    require_nonempty: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    seen_paths: set[str] = set()
    for item in _require_sequence(value, context):
        _require_mapping(item, context)
        validate_exact_fields(item, RELATIVE_SOURCE_FILE_FIELDS, context)
        relative_path = require_relative_posix_path(
            item.get("relative_path"),
            f"{context}.relative_path",
        )
        if relative_path in seen_paths:
            raise ValueError(f"{context} contains duplicate relative_path")
        seen_paths.add(relative_path)
        rows.append(
            {
                "relative_path": relative_path,
                "sha256": require_sha256(item.get("sha256"), f"{context}.sha256"),
            }
        )
    if require_nonempty and not rows:
        raise ValueError(f"{context} must be nonempty")
    if [row["relative_path"] for row in rows] != sorted(row["relative_path"] for row in rows):
        raise ValueError(f"{context} must be sorted by relative_path")
    return rows


def _validate_import_edges(value: Any) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in _require_sequence(value, "SourceBehavior.import_edges"):
        _require_mapping(item, "ImportEdge")
        validate_exact_fields(item, IMPORT_EDGE_FIELDS, "ImportEdge")
        importer = require_relative_posix_path(
            item.get("importer_relative_path"),
            "ImportEdge.importer_relative_path",
        )
        imported = require_nonempty_string(
            item.get("imported_module"),
            "ImportEdge.imported_module",
        )
        resolved = item.get("resolved_relative_path")
        if resolved is not None:
            resolved = require_relative_posix_path(
                resolved,
                "ImportEdge.resolved_relative_path",
            )
        key = (importer, imported, resolved)
        if key in seen:
            raise ValueError("SourceBehavior import_edges contains duplicate")
        seen.add(key)
        rows.append(
            {
                "importer_relative_path": importer,
                "imported_module": imported,
                "resolved_relative_path": resolved,
            }
        )
    if rows != sorted(
        rows,
        key=lambda row: (
            row["importer_relative_path"],
            row["imported_module"],
            "" if row["resolved_relative_path"] is None else row["resolved_relative_path"],
        ),
    ):
        raise ValueError("SourceBehavior import_edges must be sorted")
    return rows


def _validate_module_identities(value: Any) -> list[dict[str, Any]]:
    rows = []
    seen_modules: set[str] = set()
    for item in _require_sequence(value, "SourceBehavior.module_identities"):
        _require_mapping(item, "ModuleIdentity")
        validate_exact_fields(item, MODULE_IDENTITY_FIELDS, "ModuleIdentity")
        module = require_nonempty_string(
            item.get("module_name"),
            "ModuleIdentity.module_name",
        )
        if module in seen_modules:
            raise ValueError("SourceBehavior module_identities contains duplicate module")
        seen_modules.add(module)
        rows.append(
            {
                "module_name": module,
                "relative_path": require_relative_posix_path(
                    item.get("relative_path"),
                    "ModuleIdentity.relative_path",
                ),
                "sha256": require_sha256(item.get("sha256"), "ModuleIdentity.sha256"),
            }
        )
    if [row["module_name"] for row in rows] != sorted(row["module_name"] for row in rows):
        raise ValueError("SourceBehavior module_identities must be sorted")
    return rows


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, SequenceABC):
        raise ValueError(f"{context} must be a sequence")
    return value


def _validate_string_sequence(value: Any, context: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(_require_sequence(value, context)):
        text = require_nonempty_string(item, f"{context}[{index}]")
        if text in seen:
            raise ValueError(f"{context} contains duplicate value {text!r}")
        seen.add(text)
        result.append(text)
    return result


def _sort_elimination_reasons(reasons: Sequence[str]) -> list[str]:
    order = {reason: index for index, reason in enumerate(EXP007_ELIMINATION_REASONS_ORDER)}
    unique: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in order:
            raise ValueError("unknown elimination reason")
        if reason not in seen:
            seen.add(reason)
            unique.append(reason)
    return sorted(unique, key=lambda reason: order[reason])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, MappingABC):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are forbidden in Exp007 JSON")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _reject_duplicate_object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _linear_quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))
