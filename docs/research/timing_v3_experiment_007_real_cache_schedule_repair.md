# Timing v3 Experiment 007: Real-Cache Protocol Freeze

Date: 2026-08-12

## Mode

- Mode: planner
- Route: `TEST`
- Source idea: turn the positive Exp006 synthetic result into a decision-complete,
  source-owned execution protocol before opening any real BeatThis cache row.
- Acceptance source: Exp006 passed `44/44` source-owned fixture/schedule arms and
  its diagnostics repairs closed the scored-edge observability gap. It authorizes
  only a new real-cache protocol card, not a real-cache run.
- Source snapshot / evidence grade: high for the Exp006 search/grid behavior and
  five frozen parity oracles; medium for the proposed schedule/repair evaluation,
  because Exp007 tests it only with source-owned synthetic fixtures.
- Parent-card SHA-256:
  `789c1ffe08d2ed05c2268b6dc0a2fe4d0f4cd9bdf0b0d4a84dd57a62f9f895f7`
- Measurement Repair 001 SHA-256:
  `c1eda039a57c95931ca6d0cca6289e507a5a474a86d5ef339ee11004256b2280`
- Measurement Repair 002 SHA-256:
  `3711ef9dd8dac4df932e8b50990b573589cbf642be24858f9ed69c63d780d852`
- Exp006 result source SHA-256 for `local_frontier.py`:
  `bd6b1610cf929c8d80a5d0ded616c8aa68288225944246c998d7641fa6966dc0`
- Pre-execution real-data boundary: `NO`. Exp007 forbids opening `artifacts/`,
  BeatThis cache files, audio, `.osu`, identity/label/metric manifests,
  schedule16, repair80, holdout, broad500, full5050, API snapshots, or network
  sources. Exp008 will be the first card that may execute schedule16 and
  repair80.

## Hypothesis

A diagnostics-only bounded Exp006 entry point plus strict source-owned protocol,
selector, runner, evaluator, artifact, resume, and exposure contracts can make
the future schedule16 -> winner-only weak veto -> repair80 run mechanically
decidable without altering Exp006 search behavior or using mapper-authored
`.osu` evidence to optimize block geometry.

## Root Objective

Implement and synthetically verify the smallest protocol that lets a later card
choose one Exp006 block schedule and regression-test it on the already exposed
repair80 set. Exp007 delivers code, tests, reviewed hashes, and this frozen
contract only. It does not create an Exp008 card and does not execute real data.

## Goal Decomposition

1. Freeze recursive exact schemas, constants, source closure, canonical JSON,
   atomic artifacts, resume keys, telemetry, and exposure-delta semantics.
2. Preserve the Exp005 schedule16 identities by replaying its exact selector
   seed and exclusive bucket/deficit-fill rules from the existing repair80
   identity and source-label inputs, without metric fields.
3. Implement four independent schedule arms with one restricted prediction,
   one candidate extraction, and one current-v2 fit per `(audio, arm)`, then
   reject the entire schedule set on any cross-arm source/candidate/v2 mismatch.
4. Add a bounded diagnostics-only Exp006 API that measures overlap from exact
   exported-state lineage without allocating the full Exp006 objective ledgers,
   while proving the old API, grid, base diagnostics, search, and oracles remain
   unchanged.
5. Freeze schedule selection as source-only. Commit the source winner only via
   the successful FourArm final marker before any weak comparator access, then
   permit `.osu` only to veto that winner.
6. Implement all calculations against synthetic rows, including denominators,
   product truth, weak veto, hard failures, exact p90, row byte caps, and resume
   invalidation. Leave schedule16/repair80 execution to a separately reviewed
   Exp008 card.

## Candidate Variants

### E7-A: execute schedule16 and repair80 while repairing protocol

Reject. Exposing real metrics while fields, reducers, timeout semantics, and
selection order remain mutable would allow post-outcome protocol changes.

### E7-B: strict protocol and bounded-diagnostics implementation only

Selected. Implement only the future execution surface and synthetic contract
tests. No real selector input or prediction row is materialized in Exp007.

### E7-C: select schedule with weak `.osu` phase/drift/boundary metrics

Mutate and reject. This was inherited from Exp005, but `.osu` files are
correlated mapper annotations. They can detect an unacceptable source winner,
but optimizing block geometry against them would tune to mapper conventions and
would entangle schedule choice with comparator coverage. Exp007 therefore
selects by cache-only/source-owned behavior and uses winner-only weak evidence
as a non-promoting veto.

### E7-D: reuse the Exp006 full objective ledger on real rows

Reject. Full scored-edge occurrences, component entries, terminal ledgers, and
provisional ledgers are measurement-repair evidence, not a scalable per-row
artifact. A new bounded mode preserves the required scalar/resource/class and
overlap measurements without allocating those ledgers.

## Local Verification Matrix

| Variant | Smallest local verification | Decision |
| --- | --- | --- |
| E7-A | Requires opening real identities or caches. | Reject before execution. |
| E7-B | Strict-schema mutation tests; selector replay fixtures; spawn/timeout/death tests; bounded/full differential matrix; metric/gate truth tables; artifact/resume tests. | Selected. |
| E7-C | Static dependency test proves the source selector never imports or consumes weak rows; winner hash exists before weak evaluation. | Mutate to winner-only veto. |
| E7-D | Monkeypatch every full-ledger builder to raise, then prove bounded mode still returns the same grid/base diagnostics. | Reject full persistence; select bounded mode. |

## Selected Variant

- Selected: `E7-B protocol_freeze_only`, including the mutated source-only
  schedule selector and the diagnostics-only bounded Exp006 API.
- Rejected: immediate real execution, weak-metric schedule optimization,
  Exp004 schema reuse, and full objective-ledger persistence.
- Why this is the smallest useful test: it changes no candidate math, score,
  search order, frontier/cap, section schema, production fitter/provider/config,
  or ramp behavior. It closes only the measurement and execution protocol that
  Exp008 needs.

## Selection Pressure

- Primary pressure: every future schedule/repair decision must be replayable
  from immutable row bytes and exact source-owned reducers.
- Guard pressure: no real-data access; no inference leakage; strict recursive
  schemas; old Exp006 behavior/oracles unchanged; hard failures never become a
  fallback or product grid.
- Runtime pressure: bounded diagnostics must avoid full objective-ledger
  allocation; each future audio/arm remains under 180 seconds and 4 GiB, and
  the four-arm schedule sweep under 20 minutes.
- Kill pressure: if the bounded API perturbs search, if selector replay needs a
  metric, or if a required decision cannot be represented without unbounded
  state, stop and return to planner mode.

## Research Question

Can Exp006 enter a real-cache schedule/repair loop under a pre-registered,
source-only selection protocol whose overlap, fallback, section-inflation,
phase-safety, runtime, and provenance claims are independently auditable?

## Closest Analogies / Novelty Layer

- Closest analogies: preregistered evaluation harnesses, fixed-lag Viterbi
  lineage checks, strict manifest schemas, and exposed repair-set regression.
- Relevant taxonomy bucket: evaluation/protocol engineering.
- Novelty layer: none claimed.
- Representation novelty versus engineering variation: the half-open absolute
  beat grid is project-specific; this card is engineering around an existing
  structured decoder.

## Authorization Boundary

Exp007 may change source, tests, and this card only. All fixtures must be
created in tests or a temporary directory. It may not read or write any real
identity, cache, audio, `.osu`, label, metric, exposure, or run artifact. It may
not create the Exp008 card. After Exp007 passes and receives independent review,
the next planner turn may write Exp008; only that accepted card may authorize
schedule16 -> source-winner freeze -> winner-only weak veto -> repair80.

## Frozen IDs and Constants

```text
EXP007_EXPERIMENT_ID                 = "timing_v3_experiment_007"
EXP007_PROTOCOL_STAGE               = "protocol_freeze"
EXP007_SCHEDULE_STAGE               = "schedule16"
EXP007_REPAIR_STAGE                 = "repair80"
EXP007_SELECTOR_SEED                = "timing-v3-exp005-schedule16-v1"
EXP007_SCHEDULE_ARMS                = ("S30","S60","S90","S64")
EXP007_EXECUTION_ORDER              = ("S30","S60","S90","S64")
EXP007_TIE_RANK                     = {"S64":0,"S90":1,"S60":2,"S30":3}
EXP007_WORKER_COUNT                 = 4
EXP007_WORKER_START_METHOD          = "spawn"
EXP007_IMAP_CHUNKSIZE               = 1
EXP007_MAXTASKSPERCHILD             = None
EXP007_PER_AUDIO_ARM_TIMEOUT_S      = 180.0
EXP007_SCHEDULE_FOUR_ARM_STOP_S     = 1200.0
EXP007_REPAIR_STOP_S                = 1800.0
EXP007_WORKER_RSS_CAP_BYTES         = 4294967296
EXP007_ROW_JSON_BYTE_CAP            = 1048576
EXP007_CANDIDATE_PAYLOAD_BYTE_CAP   = 67108864
EXP007_CANDIDATE_BUNDLE_BYTE_CAP    = 69206016
EXP007_CANDIDATE_REFERENCE_MANIFEST_BYTE_CAP = 1048576
EXP007_CANDIDATE_GLOBAL_MANIFEST_BYTE_CAP = 1048576
EXP007_PARENT_POLL_MAX_SECONDS      = 0.25
EXP007_FINISH_RESULT_DELIVERY_S     = 5.0
EXP007_WORKER_TERMINATE_GRACE_S     = 5.0
EXP007_WORKER_KILL_GRACE_S          = 5.0
EXP007_CANDIDATE_METHOD_ID          = "exp006_pair_conditioned_change_floor_1_4"
EXP007_BASELINE_METHOD_ID           = "current_v2_grid_fitter"
EXP007_SELECTED_METHOD_ID           = "exp006_or_current_v2_fallback"
EXP007_WEAK_METHOD_ID               = "weak_osu_redline_object_grid_v1"
```

The selector seed deliberately remains the Exp005 literal. Exp006 changed the
transition objective, not the exposed tuning identities. Re-seeding would
silently resample after a synthetic outcome and defeat the registered
schedule16 intent.

## Canonical Encoding and Strict Validation

All schemas in this card are recursive exact schemas. `exact{...}` means the
object's key set equals the displayed set; no missing or extra key is allowed.
`list[T]` and `map[K=>T]` validate every element/value recursively. A union is
accepted only when exactly one named variant validates. Enums are literal
strings. `int` rejects `bool`; `number` is a binary64-convertible finite real
and rejects `bool`; SHA fields match lower-case `[0-9a-f]{64}`. Strings are
nonempty UTF-8 unless an enum says otherwise. Counts/ranks/indexes are
nonnegative integers unless a stricter range is stated.

Canonical JSON bytes are:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("utf-8")
```

The strict loader rejects duplicate object keys with `object_pairs_hook` and
rejects `NaN`, `Infinity`, and `-Infinity` with `parse_constant`. Validation
rejects path aliases, symlink escapes, wrong schema/version/stage/method IDs,
bool-as-int, nonfinite floats at construction and after parsing, unordered or
duplicate identities where order is contractual, and extra fields at every
depth. Every artifact binds a `schema_descriptor_sha256` computed from the
source-owned canonical descriptor for its complete recursive schema. The
descriptor is never inferred from an observed payload.

Unless a schema states a different preimage, a field named
`*_fingerprint_sha256`, `*_payload_sha256`, or `full_payload_sha256` hashes the
canonical containing object with only that hash field omitted. Ordered-row,
grid, projection, source, and field-set hashes use their explicitly named
preimages. Validators recompute every hash bottom-up and reject dependency
cycles or a payload that includes its own digest in the preimage.

### Common leaf schemas

```text
SourceRef := exact{
  schema, artifact_schema, sha256, row_count, ordered_rows_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_source_ref_v1"
  artifact_schema is a source-owned schema ID; row_count is nonnegative;
  ordered_rows_sha256 hashes exact source row order

Repair80InputBinding := exact{
  schema,experiment_id,stage,identity_source,label_source,
  four_arm_stage_summary_sha256,candidate_global_manifest_sha256,
  source_selection_sha256,schedule_weak_veto_outcome_sha256,row_count,
  binding_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_repair80_input_binding_v1";
  experiment_id=EXP007_EXPERIMENT_ID; stage="repair80";
  identity_source/label_source are SourceRef;
  four_arm_stage_summary_sha256 validates the unique committed schedule marker:
  FourArmStageSummary.status=success, source_selection_status=positive,
  config_selection_sha256=source_selection_sha256, and
  candidate_global_manifest_sha256 equals this binding's manifest SHA;
  source_selection_sha256 is that marker's positive ConfigSelection;
  schedule_weak_veto_outcome_sha256 validates a
  ScheduleWeakVetoSuccess whose summary decision is `pass` and whose source
  selection SHA matches;
  row_count=identity_source.row_count=label_source.row_count=80;
  the identity_source validator opens exactly 80 stage=repair80 Identity rows
  with 80 distinct cache_audio_key values; audio_group_key may repeat and is
  audit/association metadata only, never a grouping or denominator key;
  binding_fingerprint_sha256 hashes this object with itself omitted

RatioValue := exact{state,numerator,denominator,value}
  state in {finite,both_zero,positive_infinity,undefined}
  numerator/denominator are nonnegative finite numbers or null
  finite: numerator>=0, denominator>0, value=numerator/denominator
  both_zero: numerator=0, denominator=0, value=1.0
  positive_infinity: numerator>0, denominator=0, value=null
  undefined: numerator=null, denominator=null, value=null

RateValue := exact{numerator,denominator,value}
  0<=numerator<=denominator, denominator>0,
  value=binary64(numerator/denominator)

CoverageValue := one_of{RateValue,RatioValue}
  RatioValue variant is allowed only with state=undefined and is used exactly
  when the reference coverage denominator is zero

StatsValue := exact{count,mean,p50,p90,maximum}
  count=0 => mean=p50=p90=maximum=null
  count>0 => all values finite, nonnegative, and mechanically recomputed

AudioSetBinding := exact{count,sorted_cache_audio_keys_sha256}
  count is nonnegative; the SHA hashes canonical JSON of the sorted unique
  cache_audio_key list, including `[]` when count=0

Identity := exact{
  schema,stage,row_index,source_row_index,cache_audio_key,audio_group_key,
  label_stratum,source_long_track,duration_ms,label_source_sha256,
  identity_payload_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_identity_v1"
  stage in {schedule16,repair80}; label_stratum in
  {stable,jump_candidate,dense,ramp_candidate,ambiguous}
  source_long_track is bool; duration_ms>0
  identity_payload_sha256 hashes the same object with that field omitted;
  validating a Repair80InputBinding requires exactly 80 such repair80 rows,
  unique row_index and unique cache_audio_key, while duplicate audio_group_key
  is valid and has no counting semantics
```

`SourceRef` hashes bytes already supplied to the future Exp008 run; it is not
permission for Exp007 to open those sources.

### Strict grid schemas

```text
TimingV3Section := exact{start_beat,end_beat,bpm}

TimingV3GridPayload := exact{
  schema,version,origin_beat,origin_time_ms,coverage_start_ms,
  coverage_end_ms,sections
}
  schema = "pulsefield_model.timing_v3_grid_v1"; version=1
  sections=list[TimingV3Section], nonempty, <=20, contiguous integer beats,
  end_beat>start_beat, 20<=bpm<=1000, exact derived half-open seams

V2Segment := exact{offset_ms,beat_length_ms,meter}

V2GridPayload := exact{schema,segments}
  schema = "pulsefield_model.timing_fitted_grid_v1"
  segments=list[V2Segment], nonempty, strictly increasing offsets,
  beat_length_ms>0, meter is positive int
```

Grid validators perform canonical serialize/parse/reconstruct/serialize
round-trip equality. Accepted candidate and selected Exp006 grids use
`TimingV3GridPayload`; accepted baseline and a fallback-selected product use
`V2GridPayload` as appropriate. The product records its grid kind explicitly.

### Cache, restricted input, candidate, and source schemas

```text
CacheIdentity := exact{
  schema,relative_cache_path,exists,size_bytes,mtime_ns,inode,device,sha256,
  cache_config_sha256,audio_cache_key_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_cache_identity_v1"
  relative_cache_path is normalized POSIX-relative, with no empty, ".", "..",
  absolute, alias, or symlink-escape component
  exists=true => size_bytes,mtime_ns,inode,device are nonnegative ints and
                 sha256 is SHA-256
  exists=false => size_bytes=mtime_ns=inode=device=sha256=null and execution
                  creates a cache ArmFailureRecord, not a RowResult

RestrictedPrediction := exact{
  schema,frame_count,frame_rate_hz,beat_dtype,downbeat_dtype,
  input_signal_sha256,beat_bytes_sha256,downbeat_bytes_sha256,
  source_path_is_none,arrays_read_only,shares_loaded_memory
}
  schema = "pulsefield_model.timing_v3_exp007_restricted_prediction_v1"
  beat/downbeat dtype is "<f4"; source_path_is_none=true;
  arrays_read_only=true; shares_loaded_memory=true

CandidatePayload := exact{
  schema,beat_peaks,downbeat_peaks,tempo_candidates,origin_candidates,
  boundary_candidates,diagnostics
}
  schema = "pulsefield_model.timing_v3_exp007_candidate_payload_v1"
  beat_peaks/downbeat_peaks=list[MaterializedPeak]
  tempo_candidates=list[TempoCandidate]
  origin_candidates=list[OriginCandidate]
  boundary_candidates=list[BoundaryCandidate]
  diagnostics=CandidateDiagnostics
MaterializedPeak := exact{frame_index,refined_frame,time_ms,confidence}
  frame_index is nonnegative int; other fields finite; confidence in [0,1]
TempoCandidate := exact{bpm,source,score}
  source in {autocorrelation,peak_interval}
  20<=bpm<=1000; score finite
OriginCandidate := exact{anchor_id,time_ms,bpm,score}
  anchor_id nonnegative int; time/score finite; 20<=bpm<=1000
BoundaryCandidate := exact{
  anchor_id,time_ms,source_peak_index,source_peak_time_ms,
  source_peak_confidence,rank_score,evidence_mode,left_period_ms,
  right_period_ms,ordinary_score,super_score,downbeat_bonus,
  nearest_downbeat_distance_ms
}
  evidence_mode in {ordinary,super}; indexes/anchor IDs are nonnegative ints;
  BPM/period/time/confidence/score fields obey existing strict candidate
  finite/range validation; optional fields are finite number or null
CandidateDiagnostics := exact{
  candidate_contract_version,constants_json_sha256,
  pulse_correlation_version,boundary_candidate_score_version,frame_count,
  frame_rate_hz,coverage_start_ms,coverage_end_ms,min_period_frames,
  max_period_frames,beat_peak_count,downbeat_peak_count,
  tempo_candidate_count,origin_candidate_count,boundary_candidate_count,
  input_signal_sha256,candidate_fingerprint
}
  all versions are exact current source constants; all counts/frame bounds are
  nonnegative ints and equal payload lengths; all times/rates finite with
  positive frame rate and increasing coverage; both digests are SHA-256

RelativeSourceFile := exact{relative_path,sha256}
  relative_path is normalized unique POSIX-relative path
ImportEdge := exact{
  importer_relative_path,imported_module,resolved_relative_path
}
  resolved_relative_path is normalized relative path or null for a recorded
  external Python/NumPy module already bound by behavior version
ModuleIdentity := exact{module_name,relative_path,sha256}
  module_name is nonempty; relative_path obeys RelativeSourceFile rules

SourceClosure := exact{
  schema,experiment_id,schema_descriptor_sha256,behavior,
  audit,source_closure_fingerprint_sha256,full_payload_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_source_closure_v1"
  behavior=SourceBehavior; audit=SourceAudit
SourceBehavior := exact{
  entry_modules,required_non_import_files,
  relative_source_files,relative_source_files_sha256,import_edges,
  import_graph_sha256,module_identities,module_identities_sha256,
  python_behavior_version,numpy_behavior_version,canonical_json_contract_sha256
}
  entry_modules=list[string] exactly, in this order:
  [pulsefield_model.timing.evaluation.exp007_protocol,
   pulsefield_model.timing.evaluation.exp007_selector,
   pulsefield_model.timing.evaluation.exp007_runner,
   pulsefield_model.timing.evaluation.exp007_metrics,
   pulsefield_model.timing.evaluation.exp007_weak_evidence,
   pulsefield_model.timing.evaluation.exp007_artifacts,
   pulsefield_model.timing.v3.local_frontier,
   pulsefield_model.timing.v3.global_constant_jump,
   pulsefield_model.timing.v3.schema,
   pulsefield_model.timing.providers.beatthis_cache,
   pulsefield_model.timing.grid_fitting]
  required_non_import_files=list[RelativeSourceFile] exactly containing this
  frozen card and every source-owned schema-descriptor data file, sorted by
  relative path; no generated result or artifact is allowed
  relative_source_files=list[RelativeSourceFile], sorted by relative_path,
  nonempty and unique
  import_edges=list[ImportEdge], sorted by all three fields and unique
  module_identities=list[ModuleIdentity], sorted by module_name and unique
  python_behavior_version is implementation plus full major.minor.micro;
  numpy_behavior_version is exact `numpy.__version__`
SourceAudit := exact{
  generated_at_utc,absolute_root_path,git_commit,dirty_files,platform,
  python_full_version,numpy_full_version
}
  generated_at_utc,absolute_root_path,platform,python_full_version,
  numpy_full_version are nonempty strings
  git_commit is lower-case Git object hex or null when unavailable;
  dirty_files=list[string], sorted and unique
```

`CandidatePayload` is the Exp007 canonical projection of `asdict` over the
complete `GlobalConstantJumpCandidateSet`, plus the displayed top-level schema
ID. Every candidate list has the exact dataclass field set shown, retains tuple
order, and has no schedule field. `ordinary_score`, `super_score`, and nearest
downbeat distance are finite or null under the existing candidate contract.
`candidate_payload_schema` lives beside the payload in the transient worker
envelope and persisted row/global-manifest binding and must equal the displayed
schema ID. `candidate_payload_field_set_sha256` is owned by those envelope/
artifact bindings and hashes this complete recursive descriptor; it is not a
self-referential field inside `CandidatePayload` and is never discovered from
runtime keys. `candidate_payload_sha256` hashes canonical `CandidatePayload`.
The existing `diagnostics.candidate_fingerprint` is recomputed independently
by the Exp004 candidate validator and cross-checked; it is not replaced by the
Exp007 payload SHA. The current
tagged input signal digest remains the existing little-endian float32 hash over
beat then downbeat bytes; separate byte hashes expose accidental swapping.

`source_closure_fingerprint_sha256` is exactly SHA-256 over canonical
`SourceBehavior` bytes; every downstream field with that name must byte-equal
`SourceClosure.source_closure_fingerprint_sha256`. `SourceBehavior` binds
relative source paths and byte SHAs, the transitive import graph, module
identities, and Python/NumPy behavior versions required by tested semantics. It
excludes generated time, absolute root, PID, process start, wall/RSS telemetry,
Git commit, and unrelated dirty filenames. Those volatile/audit values live in
`SourceAudit`; `SourceClosure.full_payload_sha256` protects them and is used
only for audit-object integrity. It is excluded from deterministic resume,
row, candidate, source-selection, and config fingerprints. Tests prove one
transitive byte change changes the
behavior closure and an audit-only change leaves that closure stable.

## Selector Contract

### Selector schemas

```text
SelectorEntry := exact{
  row_index,source_row_index,cache_audio_key,audio_group_key,bucket,
  selection_substage,selection_rank,selection_hash_sha256,label_stratum,
  source_long_track,duration_ms,label_source_sha256,identity_payload_sha256
}
  bucket in {long,dense,jump,stable,deficit_fill}
  selection_substage in
  {long_quota,dense_quota,jump_quota,stable_quota,
   long_deficit_from_dense,long_deficit_from_jump,
   long_deficit_from_stable,dense_deficit_from_jump,
   dense_deficit_from_stable,jump_deficit_from_stable,
   deficit_remaining}

BucketCount := exact{bucket,requested,available,selected,deficit}
  bucket in {long,dense,jump,stable}; requested=4;
  available is total source rows with that exclusive class; selected is the
  number taken in that class's own quota after prior deficit donors; deficit
  is `4-selected`; all are replayed nonnegative ints

SelectorManifest := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,seed,
  source_repair80_identity,source_labels,source_repair80_row_count,
  selected_count,bucket_counts,deficit_count,selected_cache_audio_keys_sha256,
  selected_ordered_cache_audio_keys_sha256,selected_ordered_entries_sha256,
  selected,manifest_fingerprint_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_schedule16_selector_manifest_v1"
  stage="schedule16"; selected_count=16; selected=list[SelectorEntry]
  source_repair80_identity/source_labels are SourceRef
  source_repair80_row_count =
    source_repair80_identity.row_count = source_labels.row_count = 80
  bucket_counts=list[BucketCount] in exact long,dense,jump,stable order
```

The selector validates and replays the complete repair80 identity/label join,
requires unique `cache_audio_key`, unique `(stage,row_index)`, one label row per
identity, exact source SHA/count/order, and recursively forbids metric,
prediction, cache-array, `.osu`, redline, hitobject, comparator, phase, drift,
boundary-score, runtime, RSS, failure, result, or grid fields. The manifest is
identity-only; its inclusion never depends on a candidate outcome.

The frozen rank is:

```text
h(key) = sha256("timing-v3-exp005-schedule16-v1\0" + key)
rank rows by (h(cache_audio_key), cache_audio_key)
```

The mechanically replayed Exp005 semantics are:

```python
classes = ("long", "dense", "jump", "stable")

def class_of(r):
    # Exclusive priority is classification, not only selection order.
    if r.source_long_track is True:
        return "long"
    elif r.label_stratum == "dense":
        return "dense"
    elif r.label_stratum == "jump_candidate":
        return "jump"
    elif r.label_stratum == "stable":
        return "stable"
    return "remaining"

selected = []
used = set()

for i, quota in enumerate(classes):
    own = ranked(r for r in rows if key(r) not in used and class_of(r) == quota)
    take(own[:4], bucket=quota, substage=f"{quota}_quota")
    deficit = 4 - min(4, len(own))

    # "fill from the next class in fixed order": later predicates are tried
    # in long->dense->jump->stable order, without stealing an already used row.
    for donor in classes[i + 1:]:
        if deficit == 0:
            break
        donor_rows = ranked(
            r for r in rows if key(r) not in used and class_of(r) == donor
        )
        chosen = donor_rows[:deficit]
        take(chosen, bucket="deficit_fill",
             substage=f"{quota}_deficit_from_{donor}")
        deficit -= len(chosen)

if len(selected) < 16:
    take(ranked(r for r in rows if key(r) not in used)[:16-len(selected)],
         bucket="deficit_fill", substage="deficit_remaining")

require len(selected) == 16
assign row_index by append order
```

`selection_rank` is zero-based append order within each uniquely named
`selection_substage`; each substage's rows are in the displayed `(hash,key)`
order. `source_row_index` points to the exact repair80
identity row. `selection_hash_sha256` must equal `h(key)`.
The three ordered selector hash preimages are frozen as follows:

```text
selected_cache_audio_keys_sha256 = SHA256(canonical_json(
  sorted([entry.cache_audio_key for entry in selected])))
selected_ordered_cache_audio_keys_sha256 = SHA256(canonical_json(
  [entry.cache_audio_key for entry in selected]))
selected_ordered_entries_sha256 = SHA256(canonical_json(
  [the complete validated SelectorEntry object for entry in selected]))
```

The first list is sorted lexicographically and unique; the latter two are in
selector append order. No hash uses a set, map iteration order, or concatenated
unframed strings.
`bucket_counts` contains exactly long/dense/jump/stable in that order and is
recomputed by replay; `deficit_count` is the number selected outside an own
quota. All selected hashes, ranks, counts, provenance, and the final fingerprint
are validator-derived. Any disagreement is schema failure, never repaired from
a majority or observed result.

Selector tests include the representable overlap rows long+dense, long+jump,
and long+stable. Each must enter the long `class_of` branch, and an unused
overlap row may not reappear as a later donor or quota member. Because
`label_stratum` is a scalar exact enum, synthetic multi-valued dense+jump or
dense+stable payloads are schema failures rather than selector inputs.

## Run Config and Four-Arm Runner Contract

### Run config

```text
MethodIds := exact{candidate,baseline,selected,weak}
  each value equals its frozen method ID
CandidatePolicy := exact{
  restricted_fields,extract_exactly_once,explicit_candidate_argument,
  canonical_candidate_schema
}
  restricted_fields=list[string] exactly
  [beat_prob,downbeat_prob,frame_rate_hz]; both booleans=true;
  canonical_candidate_schema is the frozen CandidatePayload schema ID
PoolPolicy := exact{
  worker_count,start_method,imap_chunksize,maxtasksperchild,
  fixed_input_order,arm_execution_order
}
  worker_count=4; start_method="spawn"; imap_chunksize=1;
  maxtasksperchild=null; fixed_input_order=true;
  arm_execution_order=list[string] exactly [S30,S60,S90,S64]
LimitPolicy := exact{
  per_audio_arm_timeout_seconds,schedule_four_arm_stop_seconds,
  repair_stop_seconds,worker_rss_cap_bytes,row_json_byte_cap,
  candidate_payload_byte_cap,candidate_bundle_byte_cap,
  candidate_reference_manifest_byte_cap,candidate_global_manifest_byte_cap,
  parent_poll_max_seconds,
  finish_result_delivery_seconds,worker_terminate_grace_seconds,
  worker_kill_grace_seconds
}
  every field equals the frozen numeric constant

RunConfig := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,method_ids,
  candidate_policy,pool_policy,limits,selector_manifest_sha256,
  input_manifest_sha256,schedule_weak_veto_outcome_sha256,
  source_closure_fingerprint_sha256,cache_config_sha256,
  grid_fitter_config_sha256,
  local_frontier_config,weak_config_sha256,run_config_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_run_config_v1";
  experiment_id=EXP007_EXPERIMENT_ID; stage in {schedule16,repair80};
  method_ids=MethodIds; candidate_policy=CandidatePolicy;
  pool_policy=PoolPolicy; limits=LimitPolicy;
  local_frontier_config=LocalFrontierConfigPayload; every SHA field except
  schedule_weak_veto_outcome_sha256 is a non-null SHA-256;
  schedule_weak_veto_outcome_sha256 follows the exact stage branch below
LocalFrontierConfigPayload := exact{
  schedule_arm,exported_frontier_width,local_beam_width,
  max_boundary_candidates_per_block,max_tempo_candidates_per_block,
  max_blocks,max_sections,max_section_score_misses_per_block,
  max_section_score_misses_per_audio
}
  every field equals the frozen Exp006/Exp005 config; schedule_arm equals the
  enclosing RunConfig arm
```

`stage` is `schedule16` or `repair80`; `schedule_arm` is one frozen arm. All
method/config constants equal this card and the frozen Exp006 values. A
schedule16 execution has four immutable `RunConfig` artifacts, one per arm.
Repair80 has one config for the already committed selected arm; it cannot
reselect. For schedule16, `input_manifest_sha256` equals
`selector_manifest_sha256` and `schedule_weak_veto_outcome_sha256=null`; for
repair80, `input_manifest_sha256` equals the complete
`Repair80InputBinding.binding_fingerprint_sha256`, while
`selector_manifest_sha256` continues to bind the schedule selector that
selected the arm, and `schedule_weak_veto_outcome_sha256` is SHA-256 of the
canonical JSON bytes of a complete validated `ScheduleWeakVetoSuccess` whose
embedded summary decision is `pass`, including its internal
`outcome_fingerprint_sha256`, and equals the corresponding field opened from
the validated `Repair80InputBinding`. Schedule16/non-null and repair80/null,
hard-failure, ambiguous, negative, stale, or mismatched branches are invalid.

### Process model C

Future Exp008 executes arms in fixed order `S30 -> S60 -> S90 -> S64` as four
separate arm/stage executions. They are neither nested nor concurrent. Each arm
creates exactly four fresh `spawn` worker processes, passes input identities in
manifest order through ordered `imap(chunksize=1)`, and keeps those workers
persistent for that arm with `maxtasksperchild=None`. Pool cleanup, summary
fsync, and process exit finish before the next arm starts. Repair80 uses the
same four-worker model for the single selected arm.

For every `(audio,arm)`, one worker performs, in this exact order:

1. arm/row source-closure and cache identity check;
2. load exactly one shift-zero cache;
3. construct exactly one array-only restricted prediction;
4. hash the little-endian float32 arrays;
5. extract and canonicalize exactly one complete candidate set;
6. run current-v2 exactly once on that same restricted prediction;
7. call `fit_local_frontier_boundary_pair_transition_bounded` exactly once,
   passing `candidate_set=` explicitly;
8. construct selected-product, strict grids, deterministic projection, bounded
   diagnostics summary, telemetry, and the complete row;
9. recheck cache identity and source closure; canonicalize and return one
   worker envelope.

Candidate extraction is shared between current processing steps, not rerun by
the core. It is intentionally repeated once per arm because the arms are four
separate Exp005-compatible executions. The stage-global join below proves
those independently built inputs and baseline products are identical.

Before the standard `multiprocessing.Pool`, the parent creates one
`multiprocessing.connection.Listener`. Every initially spawned worker
initializer creates its exclusive duplex `Connection`, sends `WorkerHello`,
and blocks. The parent reads `pool._pool` only under the exact source-closed
CPython version frozen here, accepts exactly four HELLOs whose PIDs equal the
initial four PID set, assigns slot by initial `pool._pool` order, and replies
with `WorkerHelloAck`. An unknown PID, extra/duplicate HELLO, changed initial
PID set, automatic replacement, or generation-nonce mismatch is hard failure.
If this exact CPython Pool ownership/ordering behavior cannot be preflighted,
the arm fails before input access. `Queue` and `SimpleQueue` are forbidden as
authoritative control channels. Parent and workers use the same OS monotonic
clock domain. The tested CPython micro version and the exact runner bytes that
read `pool._pool` are behavior-source-closure inputs, not portable assumptions.

For each item the worker synchronously sends `RowStartedEvent`, then blocks for
the matching parent ACK. Parent receipt time defines `parent_start_ns` and
`deadline_ns=parent_start_ns+180_000_000_000`; ACK supplies the unique token
and deadline. Only after a valid ACK does the worker install
`SIGALRM`/`setitimer(ITIMER_REAL)` for the positive remaining duration and
begin cache/source validation. Thus the 180-second budget includes everything
after parent ACK: cache/source validation, cache load, restricted view,
candidate extraction, current-v2, all Exp006 blocks, diagnostics,
serialization, cache/source recheck, row construction, canonical envelope,
and the finish event. After canonical envelope bytes are final, the worker
records `worker_elapsed_ns`, synchronously sends `RowFinishedEvent`, and only
then returns that same envelope through ordered `imap`.

```text
WorkerHello := exact{pid,generation_nonce}
  pid is positive; generation_nonce is a worker-generated SHA-256
WorkerHelloAck := exact{pid,generation_nonce,slot}
  fields match HELLO; slot is the worker's index in initial pool._pool order
RowStartedEvent := exact{row,slot,generation_nonce,pid,worker_ready_ns}
  row=PendingIdentityRef; slot is int in 0..3; generation_nonce matches HELLO;
  pid is positive; worker_ready_ns is a nonnegative monotonic-ns int
RowStartAck := exact{
  row,slot,generation_nonce,pid,token,parent_start_ns,deadline_ns
}
  row/slot/generation_nonce/pid equal RowStartedEvent; token is the parent's unique
  SHA-256 dispatch token; deadline_ns=parent_start_ns+180_000_000_000
RowFinishedEvent := exact{
  row,slot,generation_nonce,pid,token,worker_elapsed_ns,envelope_sha256
}
  all identity fields equal the active ACK; worker_elapsed_ns is a
  nonnegative int measured from ACK receipt to immediately before synchronous
  finish send; envelope_sha256 hashes the exact returned envelope
```

The parent maintains an active ledger per slot and calls
`multiprocessing.connection.wait(active_connections,
timeout=timeout_to_nearest_deadline)`. A readable connection is drained and
validated immediately. At wait timeout, the parent performs one final
nonblocking drain
of every active connection and one final nonblocking ordered-imap poll before
deciding. Authoritative success requires both (a) the parent received one
valid matching finish event before its final
deadline decision and (b) `worker_elapsed_ns < 180_000_000_000`. Equality is
timeout. A valid under-limit finish received in the final drain succeeds;
absent or over-limit finish times out. Every active slot has its own deadline,
and one
timeout terminates and joins the whole pool. The resulting
`ArmFailureRecord(row_timeout)` binds the known
row/slot/generation-nonce/PID/token.

Ordered `imap` remains the data stream, but its iterator timeout is never used
for attribution and polling is nonblocking or at most 0.25 seconds. A valid
finish disarms only that row's 180-second deadline; the slot may ACK its next
row. Finish events and result envelopes for rows beyond
`next_expected_row_index` are buffered as audit/control state, do not enter the
prefix, and do not start a 5-second guard. When a row becomes next expected,
the first already-present side (finish or envelope) starts an exact 5.0-second
join/result-delivery guard; if the other side does not arrive and match before
that guard expires, the arm fails `broken_stream`. If the next expected
envelope arrives before finish, its original 180-second row deadline also
remains active. Only matching finish+envelope SHA commits that row and advances
the contiguous prefix. Thus a row-3 finish cannot time out delivery while row
0 is still running.

Late control events or envelopes after termination are ignored and quarantined;
they cannot resurrect a row. Missing/duplicate/out-of-order/corrupt events,
connection EOF, unknown token, slot/generation-nonce/PID mismatch, bad ACK, or
envelope SHA mismatch is hard `broken_stream`/`schema_failure`.
Worker SIGALRM remains a second fail-closed guard, is cancelled/restored in
`finally`, and never substitutes for the parent watchdog. The parent therefore
times out a live worker hung in a native-like call even when Python cannot run
the signal handler.

Multiple slots may have independent active rows. Control finish order and
result arrival order have no scientific semantics; completed-prefix order is
exactly ordered-imap identity order.

The parent supervises all four worker PIDs/sentinels. Any unexpected worker
exit, broken pipe, missing envelope, duplicate envelope, or pool replacement
invalidates the entire arm/stage immediately, because ordered `imap` cannot
prove which queued row crossed the integrity boundary. Rows after that point
are not represented as rows. The parent stops consumption, closes the
Listener and every control `Connection`, calls `Pool.terminate()`, joins known
worker `Process` objects under one 5-second terminate deadline, calls
`Process.kill()` on survivors, and joins again under one 5-second kill
deadline. If every known PID exits, it publishes the `ArmFailureRecord`; if any
PID remains alive, kill is unsupported, or bounded teardown itself errors, the
run reports fatal teardown/publication quarantine and claims no durable
`ArmStageOutcome`. It never manufactures rows, candidate payloads, or telemetry
for pending identities. Only a successful schedule arm has exactly 16 complete
`RowResult` objects; only a successful repair arm has exactly 80. A failed
arm's validated contiguous prefix is audit evidence only and may not enter
E0/E1, weak, repair, or any success denominator. The
schedule-level monotonic deadline is 20 minutes from before the S30 pool through
cross-arm join, candidate-global/config-selection publication, and
`FourArmStageSummary` directory fsync. Crossing it invalidates the full
schedule set. The repair80 arm has the frozen 30-minute stage stop through its
final summary directory fsync.

At worker initialization and after every row, read
`resource.getrusage(RUSAGE_SELF).ru_maxrss`. macOS already reports bytes;
Linux reports KiB and is multiplied by 1024. Unsupported platforms fail
preflight. A successful arm has four non-null worker-slot lifetime values and
its arm RSS is their maximum. A failed arm preserves the last observed
four-slot vector with nullable slots and the maximum of its non-null values, or
null if none was observed. Missing telemetry on a nominal success or any
observed worker above 4 GiB is hard failure. RSS and runtime remain safety
gates and reports; they are not source-order keys.

The runner outcome union is exact:

```text
CompletedRowRef := exact{
  row_index,cache_audio_key,identity_payload_sha256,row_payload_sha256,
  candidate_reference_entry_payload_sha256
}
  indexes are a contiguous prefix from zero;
  candidate_reference_entry_payload_sha256 equals the matching
  CandidateReferenceEntry.entry_payload_sha256 on schedule S30 and repair80
  reference arms, and is null on later schedule arms
PendingIdentityRef := exact{
  row_index,cache_audio_key,identity_payload_sha256
}
WorkerRssFailureSnapshot := exact{
  worker_slot_lifetime_bytes,observed_worker_max_bytes
}
  worker_slot_lifetime_bytes=list[int|null] of length 4 in worker-slot order;
  observed_worker_max_bytes=max(non-null values), or null iff all are null
ArmFailureRecord := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,
  run_config_fingerprint_sha256,source_closure_fingerprint_sha256,
  input_manifest_sha256,expected_row_count,failure_kind,failure_stage,
  causing_row_index,causing_cache_audio_key,causing_worker_slot,
  causing_worker_generation_nonce,causing_worker_pid,causing_dispatch_token,
  causing_worker_rss_bytes,
  completed_prefix_count,
  completed_prefix_rows,completed_prefix_rows_sha256,
  completed_reference_entry_count,
  completed_reference_entry_payload_sha256s,
  completed_reference_entry_payloads_sha256,pending_identity_count,
  pending_identities,pending_identities_sha256,
  prefix_candidate_fallback_count,prefix_no_origin_or_path_count,
  prefix_resource_cap_fallback_count,worker_rss_snapshot,
  failure_deterministic_fingerprint_sha256,full_payload_sha256
}
  schema="pulsefield_model.timing_v3_exp007_arm_failure_v1";
  stage in {schedule16,repair80}; expected_row_count is 16 or 80 respectively;
  failure_kind in
  {row_timeout,row_hard_failure,worker_death,pool_replacement,broken_stream,
   missing_envelope,duplicate_envelope,arm_deadline,schedule_deadline,
   identity_mismatch,source_mismatch,cache_mismatch,config_mismatch,
   restricted_input_mismatch,candidate_mismatch,current_v2_mismatch,
   weak_input_failure,weak_comparator_failure,weak_metrics_failure,
   weak_schema_failure,weak_publication_failure,
   schema_failure,rss_failure,diagnostics_integrity_failure,
   artifact_resource_cap,
   atomic_publication_failure,
   summary_publication_failure};
  failure_stage in
  {preflight,pool_start,row_source_check,cache_load,restricted_prediction,
   candidate,current_v2,local_frontier,diagnostics,row_serialization,
   row_publication,pool_stream,pool_join,weak_input,weak_comparator,
   weak_metrics,weak_schema,weak_publication,arm_summary,repair_summary,
   schedule_deadline};
  failure_kind=diagnostics_integrity_failure requires
  failure_stage=diagnostics; every malformed/nonfinite/duplicate/inconsistent/
  unknown or bounded-diagnostics count/trace/record/residual cap violation uses
  exactly that pair. A row_timeout may still name diagnostics as its earliest
  active stage and remains row_timeout. artifact_resource_cap is reserved
  exclusively for canonical JSON/candidate payload/bundle/manifest byte caps
  and never represents a bounded-diagnostics structural or memory cap;
  each causing_* field is null exactly when the earliest failure has no known
  row/worker/value; otherwise index/slot/generation-nonce/PID/bytes has its strict
  scalar type;
  completed_prefix_rows=list[CompletedRowRef] in identity order;
  causing_dispatch_token is SHA-256 exactly when a causing row is known, else
  null; completed_reference_entry_payload_sha256s=list[SHA-256] in the corresponding
  committed-prefix order; pending_identities=list[PendingIdentityRef] in the
  remaining identity order and includes the causing identity when it did not
  complete, followed by every unexecuted identity; all three list hashes hash
  their complete ordered lists;
  completed_prefix_count+pending_identity_count=expected_row_count;
  completed_reference_entry_count=
    len(completed_reference_entry_payload_sha256s)
  and equals completed_prefix_count for schedule S30/repair80 reference arms,
  but is zero for later schedule arms; prefix fallback counters are recomputed
  only from completed rows and are never success denominators
ArmStageSuccess := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,status,
  expected_row_count,row_count,row_payloads_sha256,
  candidate_reference_manifest_sha256,stage_summary_sha256,
  outcome_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_arm_stage_success_v1";
  status="success"; row_count=expected_row_count; expected/row count is 16
  for schedule16 and 80 for repair80; all ordered row/reference/summary bytes
  validate before publication; candidate_reference_manifest_sha256 is the
  completed arm's own reference manifest for S30/repair80 and the same S30
  reference-manifest SHA for each later schedule arm after direct comparison;
  for stage=schedule16, stage_summary_sha256 is SHA-256 of canonical JSON bytes
  of the complete validated SourceArmStageSummary object; for stage=repair80,
  it is SHA-256 of canonical JSON bytes of the complete validated
  Repair80Summary object. "Complete" includes the summary's own internally
  validated fingerprint field and omits nothing.
ArmStageHardFailure := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,status,
  arm_failure_record,arm_failure_record_sha256,
  outcome_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_arm_stage_hard_failure_v1";
  status="hard_failure"; arm_failure_record=ArmFailureRecord and common
  experiment/stage/arm fields agree; arm_failure_record_sha256 is SHA-256 of
  canonical JSON bytes of the complete validated ArmFailureRecord, including
  its deterministic and full-payload hash fields, with no omitted field
NotRunArmRecord := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,status,
  reason,causing_arm,causing_outcome_sha256,expected_row_count,
  pending_identity_count,pending_identities,pending_identities_sha256,
  record_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_not_run_arm_v1";
  stage="schedule16";
  status="not_run_due_prior_hard_failure";
  reason in {prior_arm_hard_failure,schedule_deadline_already_crossed};
  causing_arm is an earlier execution-order arm; no RowResult exists;
  pending_identity_count=expected_row_count=16 and pending_identities_sha256
  hashes pending_identities=list[PendingIdentityRef] containing the complete
  ordered selector identity refs
ArmStageOutcome := one_of{
  ArmStageSuccess,ArmStageHardFailure,NotRunArmRecord
}
```

`NotRunArmRecord` exists only for a later schedule arm suppressed by an earlier
schedule hard failure/deadline; it is never a repair80 outcome. A weak-veto
ambiguous/negative/hard outcome prevents repair authorization without creating
an arm record. Once repair80 starts, any failure is
`ArmStageHardFailure(expected_row_count=80)`.

`failure_deterministic_fingerprint_sha256` excludes worker PID, the nullable
RSS snapshot, and publication timing, but includes the enum/stage, all input
bindings, exact completed/pending ordered lists, and prefix counters.
`full_payload_sha256` includes every field. If the filesystem failure that
caused or followed an arm failure also prevents the failure artifact from
being atomically fsynced, the process reports a fatal publication error to its
caller and claims no durable `ArmStageOutcome`; it must not pretend that a
failure record was persisted.

### Stage-global cross-arm join

After all four arm artifacts are immutable and before any arm eligibility is
computed, join each schedule identity by `cache_audio_key`. Require exact
equality across the S30 candidate reference, each later worker's candidate
envelope, and all four row bindings for:

- identity and row order;
- cache SHA/size/config/audio-key identity (mtime/inode/device are audit-only);
- behavior source closure and all non-arm config projection fields;
- restricted input signal, beat bytes, downbeat bytes, frame count/rate;
- complete canonical `CandidatePayload` bytes, schema-descriptor SHA,
  field-set SHA, payload SHA, and candidate fingerprint;
- current-v2 status, reason, exact deterministic grid/projection bytes and SHA.

Any mismatch invalidates the entire four-arm schedule set. There is no trusted
majority, arm-local elimination, or normalization after the fact. Runtime/RSS,
PID, timestamps, absolute paths, arm-specific local-frontier config/grid, and
bounded overlap observations are excluded from this equality. The joined
candidate manifest is one schedule-global artifact with exactly 16 entries,
each binding the four equal per-arm candidate hashes; it is the authority for
candidate consistency and must validate before summaries.

```text
CandidateGlobalEntry := exact{
  row_index,cache_audio_key,audio_group_key,input_signal_sha256,
  candidate_payload_schema,candidate_payload_field_set_sha256,
  candidate_payload_byte_count,candidate_payload_sha256,
  candidate_fingerprint,candidate_reference_entry_payload_sha256,
  arm_row_payload_sha256
}
  entry identity/candidate fields equal the referenced S30 bundle entry;
  candidate_reference_entry_payload_sha256 equals that
  CandidateReferenceEntry.entry_payload_sha256; its ordered canonical bytes
  are the candidate-global hash unit
ArmRowShaMap := exact{S30,S60,S90,S64}
  every value is SHA-256
  CandidateGlobalEntry.arm_row_payload_sha256 is ArmRowShaMap

CandidateReferenceEntry := exact{
  row_index,cache_audio_key,audio_group_key,input_signal_sha256,
  candidate_payload_schema,candidate_payload_field_set_sha256,
  candidate_payload_byte_count,candidate_payload_sha256,
  candidate_fingerprint,candidate_payload,bound_row_payload_sha256,
  entry_payload_sha256
}
  candidate_payload=CandidatePayload; every digest is recomputed;
  candidate_payload_byte_count=len(canonical CandidatePayload bytes);
  candidate_payload_sha256 hashes those bytes;
  entry_payload_sha256 hashes this complete entry with only that field omitted

CandidateReferenceRef := exact{
  row_index,cache_audio_key,audio_group_key,input_signal_sha256,
  entry_payload_sha256,candidate_payload_byte_count,candidate_payload_sha256,
  candidate_fingerprint,bundle_relative_path,bundle_fingerprint_sha256
}
  bundle_relative_path is normalized unique POSIX-relative with no empty,
  ".", "..", absolute, alias, or symlink-escape component; every identity,
  candidate, entry, and bundle field is recomputed by opening and strictly
  validating the referenced CandidateReferenceRowBundle

CandidateReferenceRowBundle := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,row_index,
  entry,row,bundle_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_candidate_reference_row_bundle_v1";
  entry=CandidateReferenceEntry; row=RowResult;
  entry.bound_row_payload_sha256=row.row_payload_sha256 and all identity,
  candidate,source/config,stage,arm bindings agree exactly

CandidateReferenceManifest := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,
  input_manifest_sha256,source_closure_fingerprint_sha256,reference_arm,
  row_count,entries,
  ordered_entries_sha256,manifest_fingerprint_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_candidate_reference_manifest_v1"
  stage in {schedule16,repair80}; reference_arm is S30 for schedule16 and the
  committed selected source arm for repair80; row_count is 16 or 80
  respectively; entries=list[CandidateReferenceRef] in input identity order;
  ordered_entries_sha256 hashes canonical JSON of that complete list;
  manifest_fingerprint_sha256 hashes the complete manifest with only itself
  omitted

CandidateGlobalManifest := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,
  selector_manifest_sha256,source_closure_fingerprint_sha256,
  candidate_reference_manifest_sha256,row_count,entries,
  ordered_entries_sha256,manifest_fingerprint_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_candidate_global_manifest_v1"
  stage="schedule16"; row_count=16;
  entries=list[CandidateGlobalEntry] in selector order
  ordered_entries_sha256 hashes canonical JSON of the complete entries list;
  manifest_fingerprint_sha256 hashes the complete manifest with only itself
  omitted
```

Artifact caps are exclusive and checked on the single canonical byte buffer
before any temporary file is opened:

```text
len(canonical CandidatePayload bytes) < 67_108_864
len(canonical CandidateReferenceRowBundle bytes) < 69_206_016
len(canonical CandidateReferenceManifest bytes) < 1_048_576
len(canonical CandidateGlobalManifest bytes) < 1_048_576
```

Equality at a cap fails `artifact_resource_cap`; the payload is not truncated,
split, compressed, or externalized. These are artifact/serialization caps only
and do not change candidate extraction, search, scoring, or grids. The
reference manifest remains small because it contains refs only. Every normal
row retains its existing `<1,048,576` canonical JSON cap. Each capped artifact
is written from exactly the already-measured
canonical buffer; a second semantically reconstructed buffer is forbidden.

The complete `CandidateReferenceEntry` and candidate payload exist only inside
their `CandidateReferenceRowBundle`; the manifest embeds bounded refs, never
full entries or payloads. Each schedule S30 worker envelope and each repair80 selected-arm envelope
returns complete canonical candidate bytes to the parent. The parent constructs
the row and its complete reference entry and publishes them as one canonical
`CandidateReferenceRowBundle` file by same-directory atomic replacement and
directory fsync. This stronger transaction has no state in which a committed
entry can authorize skipping an absent row, or vice versa. After all 16
schedule bundles, or all 80 repair bundles, validate, the parent writes the
corresponding immutable `CandidateReferenceManifest`. Repair80 recomputes and
binds its own complete 80-entry reference; it never reuses a 16-entry schedule
manifest. Before repair metrics, every repair row's candidate field-set,
payload SHA/byte count, fingerprint, input SHA, and bound row SHA are
recomputed from its matching entry in that 80-entry manifest. S60/S90/S64
schedule workers again return complete bytes; before
their rows may publish, the parent requires direct byte equality with the S30
reference as well as equality of schema, field-set SHA, payload SHA, existing
candidate fingerprint, and input SHA. This remains replayable across a parent
restart without embedding the potentially large peak list in every row.

After S64, the schedule-only `CandidateGlobalManifest` binds each reference ref
and all four immutable arm rows. The reference/global manifests are primary
input-integrity artifacts, not diagnostic sidecars; their independent
exclusive 1 MiB caps do not consume the RowResult cap.
Candidate payloads are not consumed by the weak evaluator; all grids it needs
are persisted in the rows. Any reference/global disagreement invalidates the
entire schedule set.

The two ordered candidate-manifest hashes have these exact preimages:

```text
ordered_entries_sha256 = SHA256(canonical_json(
  [the complete validated CandidateReferenceRef in input order]))
candidate global ordered_entries_sha256 = SHA256(canonical_json(
  [the complete validated CandidateGlobalEntry in selector order]))
```

Every candidate-reference manifest validation opens each ref's bundle from its
relative path, checks containment and exact canonical bytes, reconstructs its
full entry/payload/row, and recomputes the ref. A missing, swapped, stale,
aliased, or mismatched bundle is hard failure. Within each
`CandidateGlobalEntry`, `arm_row_payload_sha256` has exactly the keys
S30,S60,S90,S64; no other map order is significant because canonical JSON sorts
object keys. A restart first reads the arm outcome. `success` is fully
revalidated and returned byte-for-byte; `hard_failure` forbids append or
resume; `not_run_due_prior_hard_failure` remains not run. With no outcome,
resume is arm-type specific: schedule S30 and repair80 reference arms validate
a contiguous prefix of complete `CandidateReferenceRowBundle` files and
continue at the first missing identity; later schedule arms validate a
contiguous prefix of `RowResult` files and recompare each row's candidate
schema, field-set SHA, payload SHA, candidate fingerprint, input SHA, and
complete candidate bytes against the immutable S30 reference before reuse. A
later-arm prefix is invalid unless the S30 reference manifest is already
fsynced and byte-valid. A stale, gapped, row-only artifact in a reference arm,
entry-only bundle, or mismatched row/bundle fails closed. Temp files never
count. Synthetic crash-window tests interrupt before file fsync, after file
fsync but before replace, and after replace but before directory fsync, proving
that only one complete entry+row bundle or validated later-arm row can be
reused.

## Bounded Diagnostics-Only Exp006 API

### Public and internal mode contract

Add this public API without changing the old entry point or result types:

```python
fit_local_frontier_boundary_pair_transition_bounded(
    prediction,
    *,
    config=LocalFrontierConfig(),
    candidate_set=None,
) -> LocalFrontierBoundedResult
```

```text
LocalFrontierBoundedResult := exact-object dataclass{
  fit_result: LocalFrontierResult,
  diagnostics: BoundaryPairBoundedDiagnostics
}
```

Internally, `FULL` and `BOUNDED` are diagnostics modes only. They execute
identical candidate validation, candidate/order bytes, score/cache lookup,
successor generation, local beam, dominance, class reservation, export order,
traceback, grid construction, and base `LocalFrontierDiagnostics`. Diagnostics
never enter an objective, cache key, future-equivalence key, ordering key,
tie-break, prune decision, grid, or fallback reason.

`FULL` remains the old
`fit_local_frontier_boundary_pair_transition` behavior and allocates the
Exp006 v3 component cache entries, actual scored edges, path transition
ledgers, occurrence-reference ledgers, and full objective ledgers. `BOUNDED`
does not allocate any of those collections. It keeps only scalar actual scored
edge count; unique transition-cache size; selected and runner-up objective and
margin scalars; at most 192 existing block-resource records; at most 192
class-coverage records; and the bounded overlap state below. The old public
entry point, `LocalFrontierResult`, `LocalFrontierDiagnostics`, default Exp005
entry point/payloads, and all frozen oracle hashes remain byte-identical.

### Exact bounded diagnostics dataclasses

```text
LookaheadOverlapRecord := exact{
  previous_block_index,next_block_index,previous_export_ordinal,
  lineage_sha256,comparison_start_ms,comparison_end_ms,
  provisional_trace_sha256,recomputed_trace_sha256,
  comparison_domain_sha256,comparable_beat_count,residual_vector_sha256,
  p90_ms,p90_beats,unavailable_reason
}

LocalFrontierOverlapDiagnostics := exact{
  metric_version,record_contract_version,record_count,
  available_record_count,unavailable_record_count,comparable_beat_count,
  p90_ms,p90_beats,residual_vector_sha256,records_sha256,records
}

BoundaryPairBoundedDiagnostics := exact{
  contract_version,objective_variant,candidate_fingerprint,
  transition_cache_size,actual_scored_edge_count,
  selected_terminal_objective,runner_up_terminal_objective,
  selected_runner_up_margin,block_resource_records,
  class_coverage_records,overlap,deterministic_fingerprint
}
  block_resource_records=list[BoundaryPairBlockResourceRecord] in block order
  class_coverage_records=list[BoundaryPairClassCoverageRecord] in block order
  overlap=LocalFrontierOverlapDiagnostics
  transition_cache_size/actual_scored_edge_count are nonnegative ints
  candidate_fingerprint/deterministic_fingerprint are SHA-256
  accepted fit => selected_terminal_objective is finite;
  runner-up objective and margin are both finite or both null
  failed fit => all three objective scalars are null

BoundaryPairBlockResourceRecord := exact{
  block_index,core_start_ms,core_end_ms,lookahead_end_ms,
  incoming_path_count,raw_committed_path_count,
  dominant_committed_path_count,lookahead_call_count,
  lookahead_successor_count,pre_export_state_count,exported_state_count,
  block_score_miss_count_before,block_score_miss_count_after,
  row_score_miss_count_before,row_score_miss_count_after,
  transition_component_cache_count_before,
  transition_component_cache_count_after,scored_edge_count_before,
  scored_edge_count_after,dominance_pruned_state_count,
  width_pruned_state_count,exported_frontier_width_cap,
  local_beam_width_cap,max_boundary_candidates_per_block_cap,
  max_tempo_candidates_per_block_cap,max_blocks_cap,max_sections_cap,
  max_section_score_misses_per_block_cap,
  max_section_score_misses_per_audio_cap
}
  all count/index/cap fields are nonnegative ints; time fields are finite;
  before<=after for each cumulative counter and frozen caps equal Exp006

FrontierClassKey := tuple[number, int-in-0..3-or-null]
BoundaryPairClassCoverageRecord := exact{
  block_index,cut_time_ms,input_state_count,input_unique_class_keys,
  post_future_equivalence_state_count,
  post_future_equivalence_unique_class_keys,reserved_state_count,
  reserved_unique_class_keys,final_state_count,final_unique_class_keys
}
  every *_unique_class_keys is list[FrontierClassKey] in exact
  numeric/phase order
  indexes/counts are nonnegative ints, cut_time_ms finite, and each unique-key
  count is <= its matching state count
```

These are strict projections of the existing source records; adding them to
the separate bounded result does not change either existing FULL dataclass or
its oracle bytes.

Indexes/counts are nonnegative ints; lineage/provisional hashes are always
SHA-256; comparison times are finite. The exact record union is:

- available: nonempty domain, recomputed/domain/residual SHAs present,
  `comparable_beat_count>=8`, both p90s finite/nonnegative, reason null;
- `empty_common_time_domain`: `comparison_end_ms<=comparison_start_ms`, domain
  SHA hashes the exact empty interval, recomputed/residual SHAs and p90s null,
  comparable count zero;
- `lineage_not_retained_at_next_cut`: nonempty domain and its SHA present,
  recomputed/residual SHAs and p90s null, comparable count zero;
- `fewer_than_8_comparable_beats`: nonempty domain, recomputed/domain/residual
  SHAs present, comparable count in `0..7`, p90s null.

The only unavailable enum is:

```text
empty_common_time_domain
lineage_not_retained_at_next_cut
fewer_than_8_comparable_beats
```

Malformed, nonfinite, duplicate, inconsistent, over-cap, or unknown-reason
diagnostics are hard `diagnostics_integrity_failure`, not unavailable.

`LocalFrontierOverlapDiagnostics.records` is
`list[LookaheadOverlapRecord]` in `(previous_block_index,
previous_export_ordinal)` order. Its three counts are nonnegative ints with
`record_count=available+unavailable=len(records)` and its comparable count is
the sum of available records. With at least one available record, both audio
p90s and residual SHA are non-null/finite; with none they are null. All version
strings are exact source constants and every displayed digest is SHA-256.

### Lineage and overlap algorithm

Every state in a prior cut's final exported tuple receives a private frozen
lineage token after the normal export order is complete:

```text
(record_contract_version,
 prior_block_index,
 prior_export_ordinal,
 future_equivalence_sha256,
 committed_replay_sha256)
```

The token and exact ranked provisional lookahead trace travel privately with
the state and descendants. The lineage field is `compare=False`, absent from
future-equivalence/order/replay/objective/serialization, and cannot change
search semantics. The provisional trace is the exact lookahead continuation
already chosen to rank that exported state, not the union of all selected
lookahead paths.

At the next normal cut export, group final exported descendants by the incoming
lineage token. For each prior exported state, select the first descendant in
the existing next-export order. Its newly committed core trace is the
recomputed trace. Do not rescore, backfill, retain an otherwise pruned state,
or change export order for diagnostics. If no descendant survives, record
`lineage_not_retained_at_next_cut`. After recording, discard the old lineage
and attach the new cut's token; no older lineage may leak forward.

The comparison time domain is exactly:

```text
[previous_core_end_ms,
 min(previous_lookahead_end_ms, next_core_end_ms))
```

If empty, use `empty_common_time_domain`. A trace contains ordered beat tuples
`(absolute_integer_beat, float.hex(time_ms), float.hex(local_bpm))` and ordered
real-boundary tuples. At a jump beat, the right section owns that beat and BPM.
Trace canonical JSON rejects duplicate beat keys, nonfinite values, invalid
boundaries, more than 501 beats, or more than 19 boundaries.

Intersect provisional and recomputed traces by exact absolute integer beat.
A pair is comparable only when both beat times lie in the half-open domain.
There is no nearest-beat, tempo-alias, or phase rematching. For beat `k`:

```text
r_ms    = abs(t_provisional(k) - t_recomputed(k))
period1 = 60000 / q_provisional(k)
period2 = 60000 / q_recomputed(k)
r_beats = r_ms / min(period1, period2)
```

Fewer than 8 pairs is unavailable. Otherwise record both p90s. Per-record
domain and trace hashes use their exact canonical payloads. The audio residual
SHA hashes packed canonical tuples
`(previous_block_index,previous_export_ordinal,absolute_beat,
float.hex(r_ms),float.hex(r_beats))` in record/beat order. `records_sha256`
hashes all records including unavailable records. Audio p90 is computed once
over the concatenation of every available record's residuals; it is not a p90
of per-record p90s.

All p90 values use exact linear interpolation. Sort finite binary64 values,
set `h=(n-1)*0.9`, `lo=floor(h)`, `hi=ceil(h)`, and return
`x[lo] + (h-lo)*(x[hi]-x[lo])` in the displayed evaluation order.

### Bounds and persisted projection

- maximum overlap records per audio: `16 * (192 - 1) = 3056`;
- maximum beats per trace: `501`;
- maximum real boundaries per trace: `19`;
- maximum stored residual pairs: `3056 * 501 = 1,531,056`;
- residual storage: two packed float64 arrays, approximately `24.5 MiB`;
- block-resource records: at most 192;
- class-coverage records: at most 192.

Bounds are checked before append/allocation. A `502`nd trace beat, `20`th
boundary, `3057`th record, or `1,531,057`th residual pair is hard failure.
Packed vectors are released after aggregate hashes/stats are constructed. The
row persists only overlap aggregate counts/p90/residual SHA/records SHA, never
individual records, traces, vectors, full ledgers, or sidecars.

### Required differential tests

1. All 44 Exp006 matrix arms compare `FULL` and `BOUNDED` canonical bytes for
   `{reason, grid, base LocalFrontierDiagnostics}` exactly.
2. The five aggregate oracles remain exactly:
   `a7401b...`, `f72574...`, `3911d9...`, `abf249...`, and `e1f813...`; the four
   original-kill Exp005 full-payload hashes remain unchanged.
3. Selected/runner-up objective scalars, transition-cache size, actual edge
   count, block-resource records, and class-coverage records match FULL.
4. Monkeypatch all full-ledger/occurrence collection builders to raise;
   BOUNDED still succeeds.
5. Test constant zero residual; jump at/either side of cut; right-BPM ownership;
   exact-k intersection; smaller-period divisor; 7 versus 8 pairs; every
   unavailable reason; retained versus pruned lineage; no phantom final-cut
   record; linear p90; finite/hash replay; and all `limit/limit+1` caps.

## Row and Method Schemas

### Method results

Candidate, baseline, and selected statuses are intentionally different, and
exist only inside a successful complete row:

```text
CandidateStatus in {accepted,tagged_fallback}
BaselineStatus in {accepted,unavailable}
SelectedStatus in {accepted,unavailable}

GridEnvelope := exact{kind,payload,grid_sha256,deterministic_projection_sha256}
  kind="timing_v3" <=> payload is TimingV3GridPayload
  kind="current_v2" <=> payload is V2GridPayload
  both SHAs are recomputed from their source-owned projections

MethodResult := exact{
  method_id,method_kind,status,reason,fallback_kind,
  grid,grid_summary,deterministic_projection_sha256
}
GridSummary := exact{
  grid_kind,section_count,jump_count,coverage_start_ms,coverage_end_ms,
  maximum_seam_discontinuity_ms
}
  grid_kind in {timing_v3,current_v2}; section_count>=1;
  jump_count=section_count-1; coverage bounds finite and increasing;
  maximum_seam_discontinuity_ms finite and >=0
```

The exact null/status matrix is:

| kind/status | reason | fallback_kind | grid / summary |
| --- | --- | --- | --- |
| candidate accepted | null | null | Timing-v3 present |
| candidate tagged_fallback | same frozen fallback reason | same reason | null |
| baseline accepted | null | null | current-v2 present |
| baseline unavailable | typed known reason | null | null |
| selected accepted, candidate product | null | null | Timing-v3 present |
| selected accepted, v2 fallback product | candidate fallback reason | same reason | current-v2 present |
| selected unavailable | exact `candidate_fallback_and_baseline_unavailable` | candidate fallback reason | null |

Candidate fallback reasons are exactly `no_origin_candidate`,
`no_local_frontier_path`, and `local_frontier_resource_cap_exceeded`; baseline
unavailable reasons are exactly `prediction_too_short` and `beat_signal_flat`.
The selected-unavailable deterministic branch additionally binds both source
reasons, even though its displayed selected reason is the fixed composite.
For every accepted status, `grid=GridEnvelope`,
`grid_summary=GridSummary`, and kinds agree; for every non-accepted status both
are null. `method_kind` equals the containing key and `method_id` equals its
frozen ID. Candidate/selected Timing-v3 seam is `0.0`. Current-v2 coverage is
the restricted prediction half-open coverage, and its `0.0` representation
seam does not claim integer-beat phase. A hard exception, timeout, worker/pool
failure, or publication failure creates no method or row; it creates an arm
failure outcome. `deterministic_projection_sha256` hashes the exact valid
status branch.

Baseline unavailability is determined by source-owned typed prevalidation
predicates before calling current v2: `prediction_too_short` reproduces
`frame_count < max_period_frames`; `beat_signal_flat` reproduces exact centered
signal norm zero. Tests prove predicate/legacy-exception equivalence. String
matching an exception is forbidden. Any other current-v2 exception is hard
failure.

### Diagnostics summary, runtime, and guards

```text
OverlapSummary := exact{
  metric_version,record_count,available_record_count,
  unavailable_record_count,comparable_beat_count,p90_ms,p90_beats,
  residual_vector_sha256,records_sha256
}
BoundedDiagnosticsSummary := exact{
  local_frontier_contract_version,bounded_contract_version,
  objective_variant,schedule_arm,result_reason,selected_section_count,
  block_count,candidate_fingerprint,grid_fingerprint,replay_fingerprint,
  transition_cache_size,actual_scored_edge_count,
  selected_terminal_objective,runner_up_terminal_objective,
  selected_runner_up_margin,block_resource_records_sha256,
  class_coverage_records_sha256,overlap,deterministic_fingerprint
}
RuntimeTelemetry := exact{
  platform_rule,worker_pid,audio_arm_seconds,cache_load_seconds,
  candidate_seconds,current_v2_seconds,exp006_seconds,serialization_seconds
}
  platform_rule in {macos_bytes,linux_kib_times_1024}; worker_pid is positive;
  every seconds field is finite and >=0
RssTelemetry := exact{
  platform_rule,worker_pid,initial_ru_maxrss_bytes,final_ru_maxrss_bytes
}
  platform_rule matches RuntimeTelemetry; worker_pid matches;
  both byte fields are nonnegative ints and final>=initial
HardGuards := exact{
  timed_out,worker_alive,cache_unchanged,source_unchanged,resume_valid,
  schema_valid,row_within_byte_cap,rss_within_cap,grid_seam_zero,
  section_cap_valid,diagnostics_caps_valid
}
  every field is bool; a persisted complete RowResult requires timed_out=false
  and every other field=true. Any opposite value is represented only by an
  ArmFailureRecord, never by a complete row.
```

`worker_pid`, all duration fields, RSS, timestamps, absolute paths, inode,
device, mtime, audit source fields, and pool scheduling are telemetry. They are
protected by the full row payload SHA but excluded from deterministic replay.
All mathematical grids, the canonical candidate byte count/SHA/field-set SHA
and fingerprint, fallback/status/reasons, trace and overlap hashes, schema IDs,
source/config identities, and bounded diagnostics are included in the
deterministic projection.

### Row result

```text
ResumeBinding := exact{
  row_input_fingerprint_sha256,reused,prior_row_payload_sha256,
  validated_source_closure_fingerprint_sha256,validated_config_sha256,
  validated_cache_sha256,
  validated_selector_sha256
}
  reused is bool
  reused=false => prior_row_payload_sha256=null
  reused=true => prior_row_payload_sha256 is SHA-256 and the exact immutable
                 prior row is returned after full validation
  every validated_* field is the current expected SHA-256
DenominatorFlags := exact{
  cache_valid,projection_evaluable,candidate_accepted,
  candidate_tagged_fallback,baseline_accepted,product_grid_available,
  overlap_available,current_v2_phase_matched,pure_exp006_phase_matched,
  selected_safety_phase_matched
}
  every field is bool and all implications below validate

RowResult := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,
  row_index,cache_audio_key,audio_group_key,identity_payload_sha256,
  cache_identity,source_closure_fingerprint_sha256,
  run_config_fingerprint_sha256,
  selector_manifest_sha256,input_manifest_sha256,resume,restricted_prediction,
  candidate_payload_schema,candidate_payload_byte_count,
  candidate_payload_field_set_sha256,candidate_payload_sha256,
  candidate_fingerprint,methods,denominator_flags,diagnostics_summary,
  diagnostics_summary_sha256,deterministic_projection_sha256,
  runtime,rss,hard_guards,row_payload_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_row_result_v1"
  methods = exact{candidate:MethodResult,baseline:MethodResult,
                  selected:MethodResult}
```

The row persists the complete candidate-method grid, selected-product grid,
and current-v2 grid inside
their method results, not only their SHAs. This is required so a later
immutable weak evaluator never reruns inference. It persists the canonical
candidate-set digest/field contract but not the potentially large peak lists,
which are transiently validated by the cross-arm join above. Duplicate grid
payloads may be canonical content-addressed references only if the referenced
payload is inside the same row object and validates recursively; no external
sidecar is allowed. Canonical row bytes, including every full grid, must be
`< 1,048,576` bytes. Equality at exactly the cap fails. Truncating a grid or
candidate identity is forbidden.

`row_payload_sha256` hashes the exact row without that field.
`deterministic_projection_sha256` hashes its exact source-owned projection.
`diagnostics_summary_sha256` hashes the strict summary. Every SHA is
recomputed, never trusted. The candidate reference entry binds this row through
`bound_row_payload_sha256`; the row deliberately does not point back to the
entry, avoiding a hash cycle. Later schedule arms have no per-row reference
entry SHA; their direct byte comparison and final global binding supply that
relationship.

## Product Truth and Denominators

For schedule16 source summaries (`SourceArmDenominators`, `SourceArmGates`, and
the source-selection common sets/order) only, counts are grouped by unique
`audio_group_key`; the schedule selector has already enforced a one-to-one
`cache_audio_key`/`audio_group_key` mapping. Repair80 instead uses exactly 80
unique `cache_audio_key` Identity rows and identity-level denominators: it
never groups or deduplicates by `audio_group_key`. Duplicate repair
`audio_group_key` values are allowed as audit/association metadata and still
contribute one denominator unit per distinct cache identity. Summaries rebuild
flags from exact method status and weak rows; row-supplied flags are checked
but never trusted.

| Candidate | Baseline | Selected product | Projection evaluable | Candidate fallback numerator | Selected-product fallback numerator | Pure paired possible | Selected paired possible |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| accepted | accepted | accepted candidate grid | yes | 0 | 0 | yes | yes |
| accepted | unavailable | accepted candidate grid | yes | 0 | 0 | no | no |
| tagged fallback | accepted | accepted baseline grid | yes | 1 | 1 | no | yes |
| tagged fallback | unavailable | unavailable | yes | 1 | 0 | no | no |

These are the only four complete-row branches. Failure-prefix counters are
retained in `ArmFailureRecord`, but a failed arm produces no denominators and
cannot enter this truth table. Mandatory implications are:

```text
pure_exp006_phase_matched
  => candidate_accepted and current_v2_phase_matched
selected_safety_phase_matched
  => product_grid_available and current_v2_phase_matched
product_grid_available
  => candidate_accepted or
     (candidate_tagged_fallback and baseline_accepted)
overlap_available => candidate_accepted
```

Exact source denominators/reducers for one 16-row arm are:

```text
stage_audio_count = 16
cache_valid_count = count(cache_valid)
projection_evaluable_count = count(projection_evaluable)
candidate_accepted_count = count(candidate_accepted)
candidate_fallback_count = count(candidate_tagged_fallback)
selected_product_fallback_count = count(
  candidate_tagged_fallback and baseline_accepted)
no_origin_or_path_count = count(candidate fallback reason in
  {no_origin_candidate,no_local_frontier_path})
resource_cap_fallback_count = count(reason ==
  local_frontier_resource_cap_exceeded)
candidate_fallback_rate = RateValue(candidate_fallback_count, 16)
selected_product_fallback_rate =
  RateValue(selected_product_fallback_count, 16)
no_origin_or_path_rate = RateValue(no_origin_or_path_count, 16)
```

Resource-cap fallback counts in `candidate_fallback_count` but not in
`no_origin_or_path_count`. E0 and repair gates always consume candidate
fallback count/rate, never selected-product fallback. A failed arm has no
success rates, so it cannot shrink or fabricate a denominator.

## Source-Only Schedule Selection

### Source arm eligibility E0

Schedule ordering consumes only immutable prediction rows and source labels
already present in the identity manifest. It must not open or import weak
evidence. An arm is `E0` eligible iff all are true:

```text
ArmStageOutcome.status == success      # therefore hard failure count is zero
all 16 audio_arm_seconds and arm max RSS are finite
candidate_fallback_count <= 1          # <=10% with N=16
no_origin_or_path_count == 0           # <=5% with N=16
p90(audio_arm_seconds over all 16) <= 60.0
arm_max_worker_ru_maxrss_bytes <= 4294967296
every audio_arm_seconds < 180.0
every accepted Timing-v3 maximum seam discontinuity == 0.0 ms
every accepted Timing-v3 section_count <= 20
all row schema/replay/source/cache/candidate/v2 consistency guards pass
```

The 10% and 5% guards are integer-count consequences, not rounded percentages:
`1/16=6.25%` is allowed for fallback, while `1/16>5%` makes
no-origin/no-path count necessarily zero. If fewer than two arms are E0
eligible, source selection is `ambiguous` and no source winner exists.

### Common sets and section inflation

After E0 is fixed, define sets without labels or weak data:

```text
overlap_common = audio keys where every E0 arm has candidate accepted and a
                 finite available audio overlap p90_ms
section_common = audio keys where current-v2 is accepted identically and
                 every E0 arm has candidate accepted
```

Require `len(overlap_common) >= 5` and `len(section_common) >= 8`; otherwise
selection is ambiguous. Common membership is the same for every E0 arm. For
each arm/key in `section_common`:

```text
section_excess = max(0, exp006_section_count - current_v2_segment_count)
section_inflation_violation = (section_excess > 1)
```

Reducers are:

```text
p90_overlap_ms = linear_p90(per-audio overlap p90_ms on overlap_common)
section_inflation_violation_count = sum(violation on section_common)
p90_section_excess = linear_p90(section_excess on section_common)
p90_runtime = linear_p90(audio_arm_seconds over all 16)
max_worker_rss = maximum four-worker lifetime ru_maxrss bytes
```

Every value must be finite. `label_stratum`, `source_long_track`, `.osu`,
weak-boundary F1, phase, drift, object grids, API BPM, and stable/jump/long
membership do not enter source arm eligibility, common sets, reducers, or
ordering.

### E1 and total order

Eliminate any E0 arm with `p90_overlap_ms > 90.0`; this forms E1. If E1 is
empty, source decision is `negative`. Otherwise choose the unique first arm by:

```text
(
  candidate_fallback_count,
  no_origin_or_path_count,
  p90_overlap_ms,
  section_inflation_violation_count,
  p90_section_excess,
  tie_rank,  # S64=0,S90=1,S60=2,S30=3
)
```

Runtime and RSS remain mandatory E0 gates and are reported, but are excluded
from this tuple because fixed arm order can create warm-cache bias. The second
key is explicitly retained even though E0 forces it to zero; this makes replay
stable if a future card mutates only the E0 threshold. Source
decision status is exactly:

- `ambiguous`: fewer than two E0 arms, a common-set minimum fails, or a
  required source reducer is unavailable/nonfinite;
- `negative`: the schedule set is valid but every E0 arm is removed by E1;
- `positive`: E1 has a mechanically selected first arm.

Before any `.osu` path, comparator manifest, or weak row is opened, write and
directory-fsync one immutable `ConfigSelection` as a pending dependency, then
write and directory-fsync the enclosing successful `FourArmStageSummary`.
Only that final summary is the schedule/source-selection commit marker. There
is no weak-aware retry and no runner-up promotion.

## Config Selection Schema

```text
ArmOrderValues := exact{
  schedule_arm,e0_eligible,e1_eligible,elimination_reasons,
  candidate_fallback_count,no_origin_or_path_count,p90_overlap_ms,
  section_inflation_violation_count,p90_section_excess,p90_runtime,
  max_worker_rss,tie_rank,order_tuple_sha256
}
  schedule_arm is one frozen arm; e0_eligible/e1_eligible are bool;
  candidate_fallback_count/no_origin_or_path_count/tie_rank are nonnegative
  ints; section_inflation_violation_count is a nonnegative int or null;
  elimination_reasons=list[enum] in this exact order when applicable:
  [candidate_fallback_guard,no_origin_or_path_guard,
   runtime_nonfinite,runtime_p90_guard,row_timeout_guard,rss_nonfinite,
   rss_cap_guard,seam_guard,section_cap_guard,row_consistency_guard,
   overlap_common_minimum,section_common_minimum,overlap_e1_guard];
  no enum repeats

ConfigSelection := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,
  arm_outcome_sha256_by_execution_order,candidate_global_manifest_sha256,
  source_closure_fingerprint_sha256,selector_manifest_sha256,
  overlap_common,section_common,
  source_decision,arm_order_values,selected_schedule_arm,
  selected_run_config_fingerprint_sha256,
  source_winner_selected_before_weak,selection_fingerprint_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_config_selection_v1"
  source_decision in {positive,ambiguous,negative}
  arm_outcome_sha256_by_execution_order=ArmOutcomeShaMap and every referenced
  outcome is success; this breaks any ConfigSelection/FourArm hash cycle
  arm_order_values=list[ArmOrderValues] exactly in execution order
  overlap_common/section_common are AudioSetBinding
  selected_schedule_arm is one frozen arm or null;
  selected_run_config_fingerprint_sha256 is SHA-256 or null
```

For positive, selected fields are non-null and
`source_winner_selected_before_weak=true`. For ambiguous/negative they are
null/false, weak evaluation is forbidden, and Exp008 stops. `ConfigSelection`
exists only after all four arm outcomes, the candidate-global join, and schema
validation succeed. Thus every listed arm has finite
`candidate_fallback_count`, `no_origin_or_path_count`, `p90_runtime`, and
`max_worker_rss`. `p90_overlap_ms`, `section_inflation_violation_count`, and
`p90_section_excess` are non-null iff the arm is E0 and both global common-set
minima hold; otherwise all three are null. `order_tuple_sha256` is non-null iff
E1 and hashes canonical JSON of the exact displayed total-order tuple;
otherwise null. `tie_rank` is always the frozen arm integer. The validator
derives elimination reasons and this null matrix mechanically.
`arm_order_values` has exactly four entries in S30,S60,S90,S64 order, one per
arm, with no duplicates.

A standalone `ConfigSelection` never commits or authorizes anything. Weak
evaluation and every resume path first open and validate a canonical
`FourArmStageSummary(status=success)`, then verify its
`config_selection_sha256` and `candidate_global_manifest_sha256`, then open and
validate those exact dependencies and all four success outcomes. If the final
summary is absent, hard-failure, stale, mismatched, timed out before fsync, or
unreadable, any already-fsynced ConfigSelection is orphaned, quarantined, and
ignored; no weak row or repair input may be read.

## Weak Evidence: Winner-Only Veto

Weak evidence is evaluated only from the 16 immutable selected-winner rows
after the successful FourArm commit marker and its exact ConfigSelection/
candidate-global dependencies validate. Losing-arm grids are never read by the
weak evaluator. `.osu` redlines and hitobjects remain weak correlated
mapper annotations; they cannot change candidates, source metrics, ordering,
winner, thresholds, or the repair80 schedule.

### Weak schemas

```text
ComparatorAvailability := exact{
  state,valid_difficulty_count,invalid_difficulty_count,reason,
  comparator_payloads_sha256
}
  state in {available,unavailable,conflicting}
  available => valid_difficulty_count>0, reason=null,
               comparator_payloads_sha256 is SHA-256
  unavailable/conflicting => reason is typed nonempty string;
  comparator_payloads_sha256 is SHA-256 when any payload was parsed, else null

PhaseSummary := exact{
  current_v2_ms,product_ms,pure_exp006_ms
}
  every field is StatsValue or null under the exact matrix below

DriftSummary := exact{
  current_v2_alias_max_prefix_ms,product_alias_max_prefix_ms,
  pure_exp006_alias_max_prefix_ms
}
  every field is StatsValue or null under the exact matrix below

BoundarySummary := exact{
  eligible,valid_difficulty_count,tp,fp,fn,f1,matched_error_ms,
  weak_consensus_supported_count
}
  matched_error_ms is StatsValue; f1 is RatioValue over
  numerator=2*tp, denominator=2*tp+fp+fn
  eligible=false => valid_difficulty_count=tp=fp=fn=
                    weak_consensus_supported_count=0,
                    f1=RatioValue(undefined), matched_error_ms count=0
  eligible=true => valid_difficulty_count>0 and all counts are nonnegative

ObjectGridSummary := exact{
  eligible,object_count,start_residual_ms,end_residual_ms,inlier_count,
  inlier_rate
}
  residuals are StatsValue; inlier_rate is RateValue or null
  eligible=false => object_count=inlier_count=0, both residual counts=0,
                    inlier_rate=null
  eligible=true => object_count>0, 0<=inlier_count<=object_count,
                   inlier_rate=RateValue(inlier_count,object_count)

WeakRowRef := exact{
  row_index,cache_audio_key,prediction_row_sha256,weak_row_payload_sha256
}
  every SHA is recomputed; prediction_row_sha256 equals the matching
  RowResult.row_payload_sha256 and the index/key match that row

WeakRow := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,
  row_index,cache_audio_key,audio_group_key,prediction_row_sha256,
  four_arm_stage_summary_sha256,candidate_global_manifest_sha256,
  source_selection_sha256,comparator_availability,
  current_v2_phase_matched,pure_exp006_phase_matched,
  selected_safety_phase_matched,phase_metrics_summary,
  drift_metrics_summary,current_v2_boundary_summary,
  pure_exp006_boundary_summary,selected_boundary_summary,object_grid_summary,
  deterministic_projection_sha256,weak_row_payload_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_weak_evidence_row_v1"
  stage in {schedule16,repair80};
  every *_phase_matched field is bool; schedule_arm/source_selection matches
  the committed winner; prediction_row_sha256 equals the exact paired
  RowResult.row_payload_sha256 from the same stage/index/key
  schedule16 rows bind the validated successful FourArm commit marker and its
  exact candidate-global/config-selection SHAs; repair80 rows repeat those
  same schedule bindings through Repair80InputBinding
```

The exact phase/drift null matrix is:

| Comparator/current-v2/pure/selected flags | current-v2 fields | pure fields | product fields |
| --- | --- | --- | --- |
| comparator unavailable or conflicting; all flags false | null | null | null |
| comparator available, current-v2 false; all flags false | null | null | null |
| comparator available, current-v2 true, pure false, selected false | nonempty `StatsValue` | null | null |
| comparator available, current-v2 true, pure true, selected false | nonempty `StatsValue` | nonempty `StatsValue` | null |
| comparator available, current-v2 true, pure false, selected true | nonempty `StatsValue` | null | nonempty `StatsValue` |
| comparator available, all three true | nonempty `StatsValue` | nonempty `StatsValue` | nonempty `StatsValue` |

The matrix applies independently to the three phase fields and three drift
fields. Pure/selected true implies current-v2 true. A false flag requires the
corresponding phase and drift fields null; it is never encoded as an empty
`StatsValue`. Boundary summaries follow grid availability independently:
comparator unavailable/conflicting makes all three ineligible; with an
available comparator, each is eligible iff its named grid exists and the
frozen weak-change precondition holds. Thus current-v2 unavailable forces its
boundary summary ineligible, candidate fallback forces pure ineligible, and
selected unavailable forces selected ineligible. Every ineligible branch uses
the exact zero/undefined form defined above.

Weak metrics inherit Exp004/005 contracts: valid uninherited redlines; 0.5%
log-BPM change threshold; one-to-one greedy boundary matching; tolerance
`min(750 ms, 0.5 * minimum adjacent predicted/reference period)`; 20 ms
half-open phase sampling; audio-first aggregation across difficulties; alias
normalization bound to the recorded canonicalization source. Object placement
is phase/subdivision corroboration only and does not establish BPM alias.

All `RatioValue` and `RateValue` instances use their exact schemas, so positive
infinity is structured JSON with `value=null`; no IEEE infinity is serialized.

### Schedule weak-veto reducers and truth

Required winner-only schedule weak denominators are:

```text
current_v2_phase_matched_count >= 8
pure_exp006_phase_matched_count >= 8
selected_safety_phase_matched_count >= 8
alias_drift_common_count >= 8
weak_change_boundary_audio_count >= 5
```

Let `phase_common` be the exact audio-key intersection where the source-winner
candidate and current v2 are both accepted against the same usable comparator
and both phase summaries are finite. Let `alias_drift_common` be the same
intersection with both alias-normalized max-prefix values finite. Let
`weak_change_boundary_audio` contain audio where current-v2 and pure Exp006
both have at least one valid weak redline change and finite audio-first boundary
F1. No difficulty is an independent row: per-difficulty values are first
reduced to their audio key under the frozen weak-comparator rules. Each common
set is recorded as an `AudioSetBinding`; its hash preimage is canonical JSON of
the sorted unique keys, never weak-row iteration order.

Compute:

```text
pure_mean_phase_ratio = RatioValue(
  mean per-audio pure Exp006 mean phase ms on phase_common,
  mean per-audio current-v2 mean phase ms on phase_common)
pure_p90_phase_ratio = RatioValue(
  linear_p90 of per-audio pure Exp006 p90 phase ms on phase_common,
  linear_p90 of per-audio current-v2 p90 phase ms on phase_common)
pure_phase_coverage = CoverageValue(
  pure_exp006_phase_matched_count,current_v2_phase_matched_count)
alias_max_prefix_drift_mean_ratio = RatioValue(
  mean per-audio pure alias-normalized max-prefix ms on alias_drift_common,
  mean per-audio current-v2 value on that exact set)
alias_max_prefix_drift_p90_ratio = RatioValue(
  linear_p90 of per-audio pure values on alias_drift_common,
  linear_p90 of per-audio current-v2 values on that exact set)
current_v2_boundary_f1_mean = arithmetic mean of per-audio current-v2 F1 on
  weak_change_boundary_audio
pure_exp006_boundary_f1_mean = arithmetic mean of per-audio pure Exp006 F1 on
  weak_change_boundary_audio
pure_minus_v2_boundary_f1_delta =
  pure_exp006_boundary_f1_mean - current_v2_boundary_f1_mean
```

The winner-only veto preserves the existing phase/cumulative-drift bands:

| Weak gate | Pass | Ambiguous | Negative |
| --- | --- | --- | --- |
| pure mean phase ratio, `phase_common>=8` | `<=1.05` | `(1.05,1.10]` | `>1.10` or positive infinity |
| pure p90 phase ratio, same set | `<=1.10` | `(1.10,1.15]` | `>1.15` or positive infinity |
| pure phase coverage, current-v2 matched `>=8` | `>=95%` | `[90%,95%)` | `<90%` |
| max of alias max-prefix drift mean/p90 ratios, `alias_drift_common>=8` | `<=1.15` | `(1.15,1.30]` | `>1.30` or positive infinity |
| pure-minus-v2 mean boundary F1 delta, common `n>=5` | `>=-0.05` | `[-0.10,-0.05)` | `<-0.10` |

At exactly `-0.10` the boundary gate is ambiguous; at exactly `-0.05` it
passes. Absolute current-v2, pure, and selected mean F1 values remain reported
diagnostics and cannot by themselves support an adequacy claim: a near-zero
pure F1 can still pass the delta gate when current v2 is also near zero.
Object-grid summaries are diagnostic-only. A `both_zero` ratio has value
`1.0`; `undefined` is ambiguous; structured
`positive_infinity` enters the displayed negative band.

The selected source winner passes the weak veto only when every row/schema
guard holds and every table row is in its pass band. Any ambiguous band stops
ambiguous; any negative band stops negative.

Typed comparator absence/conflict is ordinary ambiguous evidence. An
unexpected exception, source/row/schema mutation, nonfinite malformed metric,
or weak-summary publication failure is instead a fatal protocol error: no
`ScheduleWeakVetoSummary` is claimed, the already committed source winner is
retained, repair80 is forbidden, and no runner-up is evaluated. It publishes a
`ScheduleWeakVetoHardFailure` when possible. If that failure outcome itself
cannot be atomically fsynced, the run reports fatal publication/quarantine and
claims no durable weak outcome.

Weak resume first validates the unique successful FourArm commit marker and
its exact ConfigSelection/candidate-global dependencies, before opening any
`ScheduleWeakVetoOutcome`, comparator, WeakRow, or other weak input. It then
reads the outcome: a valid success or hard failure is reused byte-for-byte;
hard failure forbids append. With no outcome, resume validates the exact
contiguous `WeakRowRef` prefix and its RowResult pairings, then continues at
the first pending selected row. A missing/failing marker or stale, swapped,
cross-stage, gapped, or mismatched weak prefix fails closed. Only a success
outcome whose embedded summary decision is `pass` authorizes repair80;
ambiguous/negative are successful completed measurements but block repair.

Weak-veto truth is exact:

| Source decision | Weak denominator/ratio state | Action |
| --- | --- | --- |
| ambiguous | not read | stop `ambiguous` |
| negative | not read | stop `negative` |
| positive | insufficient, undefined, or comparator-conflicting | stop `ambiguous` |
| positive | any phase/coverage/drift/boundary negative threshold fails | stop `negative` |
| positive | no negative but any gate is in an ambiguous band | stop `ambiguous` |
| positive | all weak-veto gates pass | authorize selected-arm repair80 in Exp008 |

There is no row, reducer, or truth branch that promotes an S30/S60/S90/S64
runner-up. The source winner remains recorded even when weak evidence vetoes
it.

## Repair80 Contract Frozen for Exp008

Repair80 executes exactly the selected source arm, only after its winner-only
weak veto passes. It uses all existing exposed repair80 identities, the same
source/config/candidate/row schemas, one independent four-worker arm/stage, and
no reselection. Any behavior-affecting code/config/schema/reducer/candidate
change invalidates the source selection and requires a new complete synthetic
verification plus all four schedule16 arms. A proven byte-equivalent audit or
serialization repair may retain the winner only when deterministic rows,
grids, candidates, trace hashes, and behavior closure are identical under a
separately reviewed repair card.

For repair80, the `ArmStageOutcome` spans row execution through weak evaluation
and final `Repair80Summary` publication. An unexpected weak/schema/publication
failure after all rows therefore yields an `ArmFailureRecord` with
`completed_prefix_count=80`, `pending_identity_count=0`, and the appropriate
weak/repair failure stage; the completed rows remain audit-only and no repair
denominator or decision is published.

Repair80 weak denominator minima remain:

| Denominator | Minimum |
| --- | ---: |
| `pure_exp006_phase_matched_count` | 40 |
| `selected_safety_phase_matched_count` | 40 |
| stable pure paired | 5 |
| jump pure paired | 15 |
| long pure paired | 5 |
| overlap-available accepted audio | 20 |
| jump weak-change boundary common | 15 |

The exact Repair80 table is:

| Metric and exact denominator | Pass | Ambiguous | Negative |
| --- | --- | --- | --- |
| pure mean phase ratio on pure/current-v2 intersection, `n>=40` | `<=1.05` | `(1.05,1.10]` | `>1.10` |
| pure p90 phase ratio on same set | `<=1.10` | `(1.10,1.15]` | `>1.15` |
| pure coverage = pure matched/current-v2 matched, current-v2 `n>=40` | `>=95%` | `[90%,95%)` | `<90%` |
| max stable mean/p90 phase ratio, stable pure `n>=5` | `<=1.10` | `(1.10,1.20]` | `>1.20` |
| jump mean phase ratio, jump pure `n>=15` | `<=1.05` | `(1.05,1.15]` | `>1.15` |
| jump alias-normalized max-prefix drift mean ratio, same jump set | `<=0.90` | `(0.90,1.15]` | `>1.15` |
| pure-minus-v2 mean boundary F1 delta, jump common `n>=15` | `>=-0.05` | `[-0.10,-0.05)` | `<-0.10` |
| max long alias-normalized max-prefix drift mean/p90 ratio, long pure `n>=5` | `<=1.15` | `(1.15,1.30]` | `>1.30` |
| candidate tagged-fallback rate / all 80 identities | `<=5%` | `(5%,10%]` | `>10%` |
| no-origin/no-path rate / all 80 identities | `<=3%` | `(3%,5%]` | `>5%` |
| p90 audio-arm runtime / all 80 identities | `<=30 s` | `(30,60] s` | `>60 s` |
| audio overlap p90, accepted+available `n>=20` | `<=45 ms` | `(45,90] ms` | `>90 ms` |

Structured positive infinity is in the negative band for the applicable ratio;
undefined is ambiguous. Every metric names and hashes its exact audio set.
Selected-safety phase metrics with `n>=40` are reported as product safety but
do not replace the pure acceptance ratios. The repair boundary common set is
the sorted-key intersection of jump-paired audio with finite current-v2 and
pure Exp006 audio-first boundary F1; absolute F1 is reported but never an
adequacy threshold. At exactly `-0.10` delta is ambiguous and exactly `-0.05`
passes. Any denominator below its minimum is ambiguous, never pass.

The decomposition hard guards are zero hard failures, exact accepted
Timing-v3 seam `0.0 ms`, section count `<=20`, all rows `<180 s`, worker RSS
`<=4 GiB`, and strict replay/schema/source/cache integrity. On stable pure
paired rows, any candidate section excess over current v2 greater than one is
negative. Any hard failure or negative band blocks the later holdout; any
ambiguous band also stops for a new no-data acceptance card. A passing Exp008
authorizes only recording the repair result and drafting the next no-data card;
it cannot claim boundary localization adequacy or directly authorize a fresh
holdout. Repair80 cannot accept Exp006 for production.

## Summary Schemas and Exact Reducers

```text
RuntimeSummary := exact{row_seconds,aggregate_wall_seconds}
  row_seconds=StatsValue; aggregate_wall_seconds is finite and >=0
RssSummary := exact{worker_count,worker_lifetime_bytes,arm_max_worker_bytes}
  worker_count=4; worker_lifetime_bytes=list[int] length 4 in worker-slot
  order, each nonnegative; arm_max_worker_bytes=max(the list)

SourceArmDenominators := exact{
  stage_audio_count,stage_audio,cache_valid_audio,
  projection_evaluable_audio,candidate_accepted_audio,
  candidate_fallback_audio,selected_product_fallback_audio,
  baseline_accepted_audio,product_grid_available_audio,
  no_origin_or_path_audio,resource_cap_fallback_audio,
  overlap_available_audio
}
  stage_audio_count=16; every remaining field is AudioSetBinding;
  stage_audio.count=cache_valid_audio.count=
  projection_evaluable_audio.count=16;
  candidate_accepted_audio.count+candidate_fallback_audio.count=16;
  source formulas use `<name>_count` as the corresponding binding `.count`;
  all bindings are recomputed from complete rows
SourceArmGates := exact{
  candidate_fallback_rate,selected_product_fallback_rate,
  no_origin_or_path_rate,runtime_seconds,worker_rss_bytes,
  candidate_seam_ms,candidate_section_count,row_json_bytes,
  every_row_under_180_seconds,seam_zero,section_cap_valid,row_byte_cap_valid,
  replay_schema_source_cache_candidate_v2_consistent
}
  rates are RateValue over denominator 16; runtime_seconds is StatsValue with
  count=16; worker_rss_bytes is StatsValue with count=4; seam/section stats
  count candidate_accepted_count; row_json_bytes count=16; every final field
  is bool and must be true for E0
SourceArmStageSummary := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,
  run_config_fingerprint_sha256,source_closure_fingerprint_sha256,
  selector_manifest_sha256,candidate_reference_manifest_sha256,
  row_count,row_refs,row_payloads_sha256,denominators,gates,
  runtime_summary,rss_summary,summary_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_source_arm_summary_v1";
  stage="schedule16"; row_count=16;
  row_refs=list[CompletedRowRef] exactly selector order;
  row_payloads_sha256 hashes canonical JSON of those complete ordered refs;
  denominators=SourceArmDenominators; gates=SourceArmGates;
  runtime_summary=RuntimeSummary; rss_summary=RssSummary

ScheduleWeakDenominators := exact{
  stage_audio_count,stage_audio,comparator_available_audio,
  comparator_unavailable_audio,comparator_conflicting_audio,
  current_v2_phase_matched,pure_exp006_phase_matched,
  selected_safety_phase_matched,phase_common,alias_drift_common,
  weak_change_boundary_audio
}
  stage_audio_count=stage_audio.count=16; every field after stage_audio_count
  is AudioSetBinding; the three
  comparator sets partition all 16 keys; truth-table names
  current_v2_phase_matched_count, pure_exp006_phase_matched_count,
  selected_safety_phase_matched_count, alias_drift_common_count, and
  weak_change_boundary_audio_count mean the corresponding `.count` fields
ScheduleWeakGates := exact{
  pure_mean_phase_ratio,pure_p90_phase_ratio,pure_phase_coverage,
  current_v2_phase_mean_ms,pure_exp006_phase_mean_ms,
  current_v2_phase_p90_ms,pure_exp006_phase_p90_ms,
  current_v2_alias_drift_mean_ms,pure_exp006_alias_drift_mean_ms,
  current_v2_alias_drift_p90_ms,pure_exp006_alias_drift_p90_ms,
  alias_max_prefix_drift_mean_ratio,alias_max_prefix_drift_p90_ratio,
  current_v2_boundary_f1_mean,pure_exp006_boundary_f1_mean,
  selected_boundary_f1_mean,pure_minus_v2_boundary_f1_delta
}
  ratio fields are RatioValue; pure_phase_coverage is CoverageValue;
  raw phase/drift/F1/delta fields are finite numbers iff their named common set
  meets its minimum and all operands are finite, otherwise null; delta is exactly
  pure_exp006_boundary_f1_mean-current_v2_boundary_f1_mean in that order
  if phase_common.count<8 or either phase operand is unavailable,
  pure_mean_phase_ratio and pure_p90_phase_ratio are
  RatioValue(undefined); if current_v2_phase_matched.count<8,
  pure_phase_coverage is still the exact RateValue when its denominator is
  positive but its gate is ambiguous; if that denominator is zero the schema
  uses RatioValue(undefined) in place of RateValue under the named
  `pure_phase_coverage` union; if alias_drift_common.count<8 or an operand is
  unavailable, both drift ratios are RatioValue(undefined)
ScheduleWeakVetoSummary := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,
  four_arm_stage_summary_sha256,candidate_global_manifest_sha256,
  source_closure_fingerprint_sha256,source_selection_sha256,
  selected_row_refs_sha256,row_weak_pairs_sha256,weak_row_count,
  weak_row_refs,weak_payloads_sha256,denominators,gates,decision,action,
  summary_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_schedule_weak_veto_summary_v1";
  stage="schedule16"; weak_row_count=16;
  four_arm_stage_summary_sha256 validates the unique success commit marker;
  its config-selection/global-manifest SHAs equal this summary's
  source_selection_sha256/candidate_global_manifest_sha256;
  weak_row_refs=list[WeakRowRef] in selected-winner row order;
  each ref opens a strict WeakRow with stage="schedule16" and matching
  index/key/prediction/weak payload SHAs;
  selected_row_refs_sha256 hashes canonical JSON of ordered
  `(row_index,cache_audio_key,prediction_row_sha256)` tuples;
  row_weak_pairs_sha256 hashes canonical JSON of ordered
  `(row_index,cache_audio_key,row_payload_sha256,prediction_row_sha256,
  weak_row_payload_sha256)` tuples, requiring the two row SHAs equal;
  weak_payloads_sha256 hashes canonical JSON of complete ordered weak_row_refs;
  denominators=ScheduleWeakDenominators; gates=ScheduleWeakGates;
  decision in {pass,ambiguous,negative};
  action in {authorize_repair80,stop_ambiguous,stop_negative}
  decision/action pairs are exactly
  {pass/authorize_repair80,ambiguous/stop_ambiguous,negative/stop_negative}

WeakPendingRowRef := exact{
  row_index,cache_audio_key,prediction_row_sha256
}
ScheduleWeakFailureRecord := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,
  four_arm_stage_summary_sha256,candidate_global_manifest_sha256,
  source_selection_sha256,source_closure_fingerprint_sha256,
  expected_row_count,failure_kind,failure_stage,causing_row_index,
  causing_cache_audio_key,completed_prefix_count,completed_prefix,
  completed_prefix_sha256,pending_count,pending,pending_sha256,
  failure_deterministic_fingerprint_sha256,full_payload_sha256
}
  schema="pulsefield_model.timing_v3_exp007_schedule_weak_failure_v1";
  stage="schedule16"; expected_row_count=16;
  commit marker/config-selection/global-manifest bindings validate exactly as
  in ScheduleWeakVetoSummary;
  failure_kind in
  {weak_input_failure,comparator_failure,metrics_failure,schema_failure,
   publication_failure};
  failure_stage in
  {weak_input,comparator,metrics,schema,publication};
  causing row/key are both typed or both null;
  completed_prefix=list[WeakRowRef] as the contiguous selected-row prefix;
  pending=list[WeakPendingRowRef] as causing-if-incomplete then remaining rows;
  counts sum to 16 and list SHAs hash complete ordered lists;
  deterministic hash hashes the exact mathematical/protocol branch with both
  hash fields omitted; full hash includes the deterministic hash and omits
  only itself; this schema contains no path/PID/time field
ScheduleWeakVetoSuccess := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,status,summary,
  summary_payload_sha256,outcome_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_schedule_weak_success_v1";
  stage="schedule16"; status="success";
  summary=ScheduleWeakVetoSummary and summary_payload_sha256 hashes it
ScheduleWeakVetoHardFailure := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,status,failure,
  failure_payload_sha256,outcome_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_schedule_weak_hard_failure_v1";
  stage="schedule16"; status="hard_failure";
  failure=ScheduleWeakFailureRecord and failure_payload_sha256 hashes it
ScheduleWeakVetoOutcome := one_of{
  ScheduleWeakVetoSuccess,ScheduleWeakVetoHardFailure
}

Repair80Denominators := exact{
  stage_audio_count,stage_audio,cache_valid_audio,projection_evaluable_audio,
  candidate_accepted_audio,candidate_fallback_audio,
  selected_product_fallback_audio,baseline_accepted_audio,
  product_grid_available_audio,no_origin_or_path_audio,
  resource_cap_fallback_audio,overlap_available_audio,
  current_v2_phase_matched,pure_exp006_phase_matched,
  selected_safety_phase_matched,phase_common,stable_pure_paired,
  jump_pure_paired,long_pure_paired,jump_alias_drift_common,
  long_alias_drift_common,repair_boundary_common
}
  stage_audio_count=stage_audio.count=80; every field after stage_audio_count
  is AudioSetBinding;
  cache_valid_audio.count=projection_evaluable_audio.count=80;
  candidate_accepted_audio.count+candidate_fallback_audio.count=80;
  each named Repair80 truth-table count is its binding's `.count`

Repair80 set membership is recomputed by exact `cache_audio_key` from immutable
repair Identity rows, RowResult method statuses/diagnostics, and paired WeakRow
objects. Define independent identity predicates, not one exclusive class:

stable_label(k) := identity(k).label_stratum == "stable"
jump_label(k)   := identity(k).label_stratum == "jump_candidate"
long_label(k)   := identity(k).source_long_track is true
pure_pair(k)    := candidate status accepted
                   and baseline status accepted
                   and WeakRow.comparator_availability.state == available
                   and WeakRow.current_v2_phase_matched
                   and WeakRow.pure_exp006_phase_matched
stable_pure_paired := sorted keys k where stable_label(k) and pure_pair(k)
jump_pure_paired   := sorted keys k where jump_label(k) and pure_pair(k)
long_pure_paired   := sorted keys k where long_label(k) and pure_pair(k)
phase_common       := sorted keys k where pure_pair(k) and both current-v2 and
                      pure phase StatsValue operands are present and finite
jump_alias_drift_common := sorted keys k in jump_pure_paired where current-v2
                           and pure alias-max-prefix drift operands are present
                           and finite
long_alias_drift_common := sorted keys k in long_pure_paired where current-v2
                           and pure alias-max-prefix drift operands are present
                           and finite
repair_boundary_common := sorted keys k in jump_pure_paired where the same
                          comparator has >=1 valid weak redline change and both
                          current-v2 and pure boundary summaries are eligible
                          with finite audio-first F1
overlap_available_audio := sorted keys k where candidate status accepted and
                           diagnostics overlap is available with finite p90_ms

`long_label` is deliberately orthogonal: a long track may also be stable or
jump and then appears in both exact sets; no priority/exclusive `class_of` from
the schedule selector is used here. Dense/ramp/ambiguous identities enter none
of stable/jump unless independently long. Classification happens first from
identity bytes, then each displayed method/comparator/finite intersection is
applied. Candidate tagged fallback is excluded from every pure, drift,
boundary, overlap, and stable-section set; selected-product fallback never
substitutes for a pure candidate. `selected_safety_phase_matched` remains its
separate WeakRow-flag set and does not enter these pure predicates.

Every displayed set is an `AudioSetBinding`: `count=len(sorted unique keys)`
and `sorted_cache_audio_keys_sha256=SHA256(canonical_json(sorted unique keys))`.
The remaining bindings are equally exact:

stage_audio := all 80 validated repair identity keys
cache_valid_audio := keys with RowResult.denominator_flags.cache_valid
projection_evaluable_audio := keys with projection_evaluable
candidate_accepted_audio := keys with candidate status accepted
candidate_fallback_audio := keys with candidate status tagged_fallback
selected_product_fallback_audio := keys with candidate tagged_fallback,
                                   baseline accepted, selected accepted/current-v2
baseline_accepted_audio := keys with baseline status accepted
product_grid_available_audio := keys with selected status accepted and
                                product_grid_available
no_origin_or_path_audio := candidate fallback keys whose reason is exactly
                           no_origin_candidate or no_local_frontier_path
resource_cap_fallback_audio := candidate fallback keys whose reason is exactly
                               local_frontier_resource_cap_exceeded
current_v2_phase_matched := keys with comparator available, baseline accepted,
                            and WeakRow.current_v2_phase_matched
pure_exp006_phase_matched := keys with comparator available, candidate accepted,
                             baseline accepted, and both current-v2 and pure flags
selected_safety_phase_matched := keys with comparator available,
                                 product_grid_available, baseline accepted,
                                 and both current-v2 and selected-safety flags

Each phrase `keys with` means sort/unique by cache key only after all displayed
conditions intersect. The row/weak schema implications are revalidated rather
than trusting a supplied denominator flag.
Repair80Gates := exact{
  candidate_fallback_rate,selected_product_fallback_rate,
  no_origin_or_path_rate,runtime_seconds,worker_rss_bytes,overlap_ms,
  stable_section_excess,pure_mean_phase_ratio,pure_p90_phase_ratio,
  pure_phase_coverage,current_v2_phase_mean_ms,pure_exp006_phase_mean_ms,
  current_v2_phase_p90_ms,pure_exp006_phase_p90_ms,
  stable_phase_mean_ratio,stable_phase_p90_ratio,
  jump_phase_mean_ratio,current_v2_jump_alias_drift_mean_ms,
  pure_exp006_jump_alias_drift_mean_ms,jump_alias_drift_mean_ratio,
  current_v2_long_alias_drift_mean_ms,pure_exp006_long_alias_drift_mean_ms,
  current_v2_long_alias_drift_p90_ms,pure_exp006_long_alias_drift_p90_ms,
  long_alias_drift_mean_ratio,long_alias_drift_p90_ratio,
  current_v2_boundary_f1_mean,pure_exp006_boundary_f1_mean,
  selected_boundary_f1_mean,pure_minus_v2_boundary_f1_delta,
  every_row_under_180_seconds,seam_zero,section_cap_valid,
  replay_schema_source_cache_integrity
}
  rates are RateValue; pure_phase_coverage is CoverageValue: RateValue when current-v2 matched
  denominator is positive and RatioValue(undefined) when it is zero;
  phase/drift ratios are RatioValue;
  runtime/RSS/overlap/section are StatsValue over their exact named bindings;
  boundary fields are finite iff repair_boundary_common.count>=15, else null;
  booleans are true on a successful evaluable repair summary; every
  phase/drift RatioValue whose named common-set minimum fails or whose operand
  is unavailable has exactly the RatioValue(undefined) branch, never null;
  candidate/selected fallback rates and no-origin/path rate are always finite
  RateValue over 80

`stable_section_excess` has exactly the `stable_pure_paired` key set and one
value per key:
`max(0, candidate Timing-v3 section_count - current-v2 segment_count)`.
Its StatsValue count equals `stable_pure_paired.count`; no fallback/product,
non-stable, or long-only key is added. `overlap_ms` contains exactly each
`overlap_available_audio` key's finite audio overlap p90_ms. All class-specific
phase/drift reducers use exactly the correspondingly named bindings above.
Repair80Summary := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,schedule_arm,
  four_arm_stage_summary_sha256,candidate_global_manifest_sha256,
  source_selection_sha256,schedule_weak_veto_outcome_sha256,
  run_config_fingerprint_sha256,source_closure_fingerprint_sha256,
  repair_input_binding_sha256,repair_identity_source,repair_label_source,
  candidate_reference_manifest_sha256,row_count,row_refs,row_payloads_sha256,
  weak_row_count,weak_row_refs,weak_payloads_sha256,row_weak_pairs_sha256,
  denominators,gates,
  decision,action,runtime_summary,rss_summary,summary_fingerprint_sha256
}
  schema="pulsefield_model.timing_v3_exp007_repair80_summary_v1";
  stage="repair80"; repair identity/label fields are SourceRef with row_count
  80 and repair_input_binding_sha256 equals their validated
  Repair80InputBinding fingerprint; row_count=weak_row_count=80;
  four-arm/global/selection/weak-outcome fields equal that binding and the
  successful committed schedule/weak dependency chain;
  schedule_weak_veto_outcome_sha256 validates a ScheduleWeakVetoSuccess whose
  embedded summary decision=`pass` and source-selection SHA matches;
  row_refs=list[CompletedRowRef] and weak_row_refs=list[WeakRowRef] in repair
  identity order; each weak ref has stage repair80 and its prediction SHA
  equals the matching RowResult row SHA;
  row_weak_pairs_sha256 hashes canonical JSON of ordered
  `(row_index,cache_audio_key,row_payload_sha256,prediction_row_sha256,
  weak_row_payload_sha256)` tuples and requires row/prediction SHA equality;
  row/weak payload hashes use their exact ordered refs;
  denominators=Repair80Denominators; gates=Repair80Gates;
  decision in {pass,ambiguous,negative};
  action in {write_result_and_next_no_data_card,stop_ambiguous,stop_negative};
  decision/action pairs are exactly
  {pass/write_result_and_next_no_data_card,
   ambiguous/stop_ambiguous,
   negative/stop_negative}; every other cross-pair is schema-invalid;
  runtime_summary=RuntimeSummary; rss_summary=RssSummary

ArmOutcomeShaMap := exact{S30,S60,S90,S64}
  each value is the SHA-256 of one ArmStageOutcome and map keys are exact
FourArmFailureDetails := exact{
  failure_kind,first_failure_arm,causing_outcome_sha256,
  mismatch_cache_audio_key,mismatch_field,completed_success_arm_count,
  deterministic_failure_sha256,full_failure_sha256
}
  failure_kind in
  {arm_hard_failure,schedule_deadline,cross_arm_identity_mismatch,
   cross_arm_cache_mismatch,cross_arm_source_config_mismatch,
   cross_arm_restricted_input_mismatch,cross_arm_candidate_mismatch,
   cross_arm_current_v2_mismatch,cross_arm_schema_mismatch,
   candidate_global_publication_failure};
  first_failure_arm/causing_outcome_sha256 are non-null for arm_hard_failure
  and for schedule_deadline represented by an ArmStageHardFailure; they are
  null for post-arm stage-level schedule_deadline, cross-arm mismatch, and
  candidate-global publication failure. A post-arm schedule_deadline requires
  completed_success_arm_count=4, the enclosing FourArmStageSummary
  candidate_global_manifest_sha256=null, and no ConfigSelection. mismatch
  key/field are non-null exactly for a cross-arm mismatch;
  completed_success_arm_count is 0..4; full hash includes audit details while
  deterministic hash excludes telemetry/path/time

FourArmStageSummary := exact{
  schema,experiment_id,stage,schema_descriptor_sha256,
  status,arm_outcome_sha256_by_execution_order,
  candidate_global_manifest_sha256,failure_details,
  source_selection_status,config_selection_sha256,
  summary_fingerprint_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_four_arm_stage_summary_v1"
  stage="schedule16";
  status in {success,hard_failure};
  arm_outcome_sha256_by_execution_order=ArmOutcomeShaMap;
  success => all four outcomes are success,
             candidate_global_manifest_sha256 is SHA-256,
             failure_details=null,
             source_selection_status in {positive,ambiguous,negative},
             config_selection_sha256 is SHA-256;
  hard_failure => candidate_global_manifest_sha256=null,
                  failure_details=FourArmFailureDetails,
                  source_selection_status="not_run",
                  config_selection_sha256=null
```

Every external `four_arm_stage_summary_sha256` in this card is SHA-256 of the
canonical JSON bytes of the complete validated FourArmStageSummary, including
its `summary_fingerprint_sha256`; no field is omitted for this external object
hash. In the success branch, `config_selection_sha256` and
`candidate_global_manifest_sha256` likewise hash the complete canonical
dependency artifacts. The source-owned final relative path plus exclusive run
lock permits exactly one such final marker.

The hard-failure branch permits four successful arm outcomes followed by a
cross-arm mismatch, post-arm schedule deadline, or candidate-global
publication failure; it still has no candidate-global manifest and no
committed `ConfigSelection` (an orphan file may exist but is ignored). If an
earlier arm fails, every later map value must be a
`NotRunArmRecord` chained to that failure. No majority arm is trusted. The
success branch is the only schedule/source-selection commit marker and the
only path to source selection; global join/schema mismatches are never
reclassified as source ambiguity. Exactly one canonical final
FourArmStageSummary path is permitted per schedule run root; a divergent or
second final marker is hard publication failure.

All means are arithmetic means in manifest/audio-key order using Python
binary64. All p50/p90 use the frozen linear interpolation. Maximum is exact.
Empty sets yield `StatsValue(count=0,...null)` and cannot satisfy a required
denominator. Ratios and rates never coerce missing, undefined, or infinity to a
finite sentinel. Each reducer validates membership hashes and recomputes its
denominator from complete success rows. Every ordered row/weak reference hash
uses canonical JSON of the complete displayed ref objects in manifest order;
every common-set hash uses canonical JSON of sorted unique audio keys. Failed
prefixes are excluded from all three summary families.

## Hard Failures

The following are hard failures, never tagged fallbacks or baseline
unavailability:

- timeout at or above 180 seconds; four-arm/stage deadline crossing;
- worker death, replacement, broken pipe, missing/duplicate envelope;
- cache/source/config/selector/identity mutation or cross-arm mismatch;
- unrestricted prediction, implicit candidate extraction, candidate schema or
  fingerprint mismatch;
- path alias, duplicate resolved target, containment failure, symlink escape;
- stale resume or mismatched behavior/source/config/cache/input fingerprint;
- unknown current-v2 exception;
- invalid/extra/missing/nonfinite/duplicate-key JSON; bool-as-int; wrong schema;
- grid/schema/serialization failure;
- any malformed or over-cap bounded diagnostics
  (`diagnostics_integrity_failure`, stage `diagnostics`);
- RowResult at/above 1 MiB (`artifact_resource_cap`);
- candidate payload at/above 64 MiB, reference bundle at/above 66 MiB, or
  reference/global manifest at/above 1 MiB (`artifact_resource_cap`);
- Listener/HELLO/ACK/control-Connection corruption or EOF, initial Pool PID
  mutation/replacement, watchdog timeout, or 5-second join/delivery failure;
- missing/nonfinite runtime/RSS; unsupported RSS rule; RSS over 4 GiB;
- nonzero accepted Timing-v3 seam or more than 20 sections;
- fsync, atomic replace, lock, or summary publication failure.

No partial, truncated, best-so-far, pre-timeout, or previous-arm product may be
marked accepted. A hard failure produces no `RowResult`, method statuses,
denominator flags, product, projection, or weak pairs. An arm-level integrity
failure invalidates the full arm, and a cross-arm integrity failure invalidates
the full schedule set.

## Atomic Output, Resume, and Source/Cache Closure

Every future run uses an exclusive run lock bound to experiment/stage/arm,
selector, behavior source closure, config, and output root fingerprint. An
artifact is written from one canonical byte buffer to a unique same-directory
`O_CREAT|O_EXCL` temporary path, file-fsynced, atomically replaced, then its
parent directory fsynced. Existing immutable destinations must match exact
bytes or fail; they are never overwritten with different bytes. No diagnostic
sidecars are authorized.

Top-level schedule/source-selection/weak resume first reads the unique
FourArmStageSummary commit-marker path. A valid success marker causes exact
global-manifest, ConfigSelection, and four-arm outcome validation before any
selection or weak input; a hard marker stops. If no valid final marker exists,
resume quarantines any standalone ConfigSelection and may resume only
pre-commit arm/join/publication work—never source-selection consumption, weak
evaluation, or repair authorization.

Arm-level resume then reads the exact `ArmStageOutcome` as specified in the
candidate bundle contract: success returns the validated immutable arm, hard
failure forbids append, and a not-run arm remains not run. Only when no outcome
exists does it validate the arm-type-specific contiguous prefix and continue:
complete entry+row bundles for schedule S30 and repair80 reference arms, and
row files plus S30 candidate-reference byte comparison for later schedule
arms. Each reusable prefix artifact is read from one byte snapshot and
rechecks schema, payload SHA, deterministic projection, identity,
selector/input manifest, behavior source, run config, cache identity/content
SHA, restricted input, candidate payload binding, method/grid, and all guard
bindings. A stale, gapped, or partially valid prefix is hard failure; it is not
silently recomputed in the same immutable run path. New behavior receives a
new fingerprinted run directory.

Cache identity is checked before load, after candidate extraction, after
current-v2, after Exp006, and before row publication; source closure is checked
before the pool, at row start/end, and before summary publication. Cache bytes
are parsed and hashed from a stable snapshot; stat-before/stat-after must match.
Stage publication rescans every declared cache/source identity. Absolute paths,
mtime/inode/device are audit telemetry and do not enter deterministic replay;
content/config/audio-key SHAs do.

Atomic publication order is fixed. A reference-arm row uses the single
entry+row bundle transaction, then its reference manifest, then its success
arm summary/outcome. A later schedule arm publishes rows, then its success
summary/outcome. After all four success outcomes, the parent publishes the
candidate-global manifest, then `ConfigSelection`, then the FourArm success
summary. The ConfigSelection is pending/uncommitted until the final success
summary and its parent directory are fsynced. Schedule weak/resume first
validates that success marker and its exact selection/global-manifest
dependencies; only then may weak inputs be opened. Schedule weak rows publish
only after that commit; their completed
summary/failure record publishes before its `ScheduleWeakVetoOutcome`. Repair
RunConfig/input binding/rows may publish only after a fsynced weak success
outcome with decision `pass`; a weak or repair summary can only reference
already-fsynced inputs.
If the process crashes, times out, or fails publication after ConfigSelection
fsync but before final FourArm success-summary directory fsync, the standalone
selection is quarantined on resume and never authorizes weak or repair. The
same is true when a hard-failure final marker exists.
On arm failure, no later success artifact may publish; the failure record and
outcome publish before any later-arm not-run records. On cross-arm failure the
FourArm hard-failure summary is the only final schedule summary. Every artifact
uses a source-owned relative path derived solely from experiment/stage/arm and
content fingerprints, takes an exclusive lock, refuses a divergent existing
destination, and directory-fsyncs before a dependent SHA may be published.

## Exposure Delta Contract

```text
ExposureEntry := exact{
  cache_audio_key,audio_group_key,exposure_stage,exposure_reason,
  first_exposed_at_or_run_id,observed_payload_kind,source_manifest_sha256
}
  exposure_stage in {schedule16,repair80,accidental_batch}
  observed_payload_kind in
  {identity,cache,prediction,grid,metric,diagnostic,runtime,failure,trace,osu,
   rendering,batch_aggregate}
  keys/reason/run-id are nonempty strings; source_manifest_sha256 is SHA-256

ExposureDelta := exact{
  schema,experiment_id,schema_descriptor_sha256,generated_at_utc,
  prior_exposure_manifest_sha256,source_closure_fingerprint_sha256,
  delta_reason,
  entry_count,cache_audio_keys_sha256,entries_sha256,entries,
  manifest_fingerprint_sha256
}
  schema = "pulsefield_model.timing_v3_exp007_exposure_delta_v1"
  entries=list[ExposureEntry] sorted uniquely by cache_audio_key;
  entry_count=len(entries); all source/prior/entry/key hashes are recomputed
```

Exp007 tests this only with synthetic keys and writes no real delta. Exp008
must add every key whose identity, cache, prediction, grid, metric, diagnostic,
runtime, failure, trace, `.osu`, rendering, or batch aggregate is observed. If
an accidental batch exposes more keys, all keys in that batch enter the delta.
Uncertain means exposed. Future selection uses the exact union of prior
exposure and Exp008 delta before materializing a holdout.

`cache_audio_keys_sha256` hashes canonical JSON of the sorted unique entry
keys; `entries_sha256` hashes canonical JSON of the complete sorted entries
list. An exposure update is lock-protected and uses the same
file-fsync/atomic-replace/directory-fsync contract; a divergent existing delta
or failure to publish it is fatal and cannot be bypassed by a later stage.

## Minimal Change

Add one bounded diagnostics mode and Exp007 evaluation modules/tests. Do not
change Exp006 objective/search/candidate behavior, v3 grid schema, current-v2,
production Timing-v3 fitter/provider, BeatThis cache format, Hydra config,
mapper, ramp primitive, or any real split.

## Files Likely to Change

- `src/pulsefield_model/timing/v3/local_frontier.py`
- `src/pulsefield_model/timing/v3/__init__.py` only if the new public API is
  exported there
- `src/pulsefield_model/timing/evaluation/exp007_protocol.py`
- `src/pulsefield_model/timing/evaluation/exp007_selector.py`
- `src/pulsefield_model/timing/evaluation/exp007_runner.py`
- `src/pulsefield_model/timing/evaluation/exp007_metrics.py`
- `src/pulsefield_model/timing/evaluation/exp007_weak_evidence.py`
- `src/pulsefield_model/timing/evaluation/exp007_artifacts.py`
- `tests/timing/test_timing_v3_exp007_protocol.py`
- `tests/timing/test_timing_v3_exp007_selector.py`
- `tests/timing/test_timing_v3_exp007_runner.py`
- `tests/timing/test_timing_v3_exp007_metrics.py`
- `tests/timing/test_timing_v3_exp007_weak_evidence.py`
- `tests/timing/test_timing_v3_exp007_artifacts.py`
- existing local-frontier/Exp006 matrix tests for bounded/full parity
- this card; a result/problem-log update only after execution and review

Exp007 must not create or edit an Exp008 document. Existing unrelated dirty
files are preserved.

## Read-Only Context Files

- `docs/research/timing_v3_task_definition.md`
- Exp004 card, protocol clarification, result, runner, metrics, weak evaluator,
  and split modules as reference only
- Exp005 card and result
- Exp006 parent card, both measurement-repair cards, and result
- `src/pulsefield_model/timing/v3/schema.py`
- `src/pulsefield_model/timing/v3/global_constant_jump.py`
- `src/pulsefield_model/timing/providers/beatthis_cache.py`
- current-v2 grid-fitting modules

## Dataset Slice

Source-owned synthetic arrays, strict JSON fixtures, fake cache identities,
temporary directories, and short-lived synthetic spawn workers only. Tests
must inject fake loaders and fail if any configured path resolves into
`artifacts/` or a real asset root. No schedule16/repair80 identity manifest is
materialized in Exp007.

## Baseline / Comparator

- Behavior baseline: old full-ledger
  `fit_local_frontier_boundary_pair_transition` on the frozen Exp006 synthetic
  matrix and five parity oracles.
- Future real primary comparator: current-v2 `GridFitter` on the identical
  restricted cache activation.
- Candidate: bounded Exp006 E6-D with explicit identical candidate set.
- Product: accepted Exp006 or tagged-fallback plus accepted current-v2.
- Weak `.osu`: selected-winner veto and later evaluation only; never inference
  or schedule optimization.
- Network/catalog BPM: outside Exp007 and Exp008 schedule/repair inference and
  selection. A later separately frozen audit may corroborate recording/alias
  family, never define change structure.

## Primary Metric

Exp007 passes only if all strict protocol tests pass and every one of 44
FULL/BOUNDED arms has exact canonical equality for reason/grid/base diagnostics,
with the five Exp006 parity oracles unchanged and no real-data access.

## Secondary Metric

- recursive schema rejection coverage;
- exact selector replay including deficits and provenance;
- one restricted/candidate/v2 call per `(audio,arm)` and explicit candidate
  injection;
- worker alarm/death/ordered-imap/RSS/four-arm stop behavior;
- exact overlap lineage, hash, p90, cap, and unavailable semantics;
- source E0/E1/common-set/order and winner-only weak-veto truth tables;
- product/hard-failure denominator truth;
- deterministic/audit projection separation;
- atomic/resume/source/cache/exposure synthetic tests;
- canonical rows below the frozen byte cap.

### Required synthetic protocol tests

In addition to the differential tests above, implementation is incomplete
until all of these exact families pass:

1. strict recursive valid/extra/missing/duplicate-key/nonfinite/bool-as-int
   tests for every schema and every union/null branch, including the exact
   RunConfig stage branch: schedule16 requires null weak-outcome SHA, repair80
   requires the complete validated weak-success/pass outcome SHA, and every
   reversed, null, hard/ambiguous/negative, stale, or mismatched case rejects;
2. selector replay, exact seed, selected sorted/ordered key hashes, entry hash,
   exclusive overlap-label classification, and each deficit-donor fixture;
3. four-arm fixed execution order, four fresh persistent workers per stage,
   Listener plus exactly four initial-PID HELLO/slot handshakes, unsupported
   CPython preflight, unknown/extra/corrupt HELLO, Connection EOF, bad ACK/token/
   generation nonce, automatic replacement, ordered-imap attribution, one
   cache/restricted/candidate/current-v2 call, exact alarm restore, and
   four-slot RSS normalization;
4. watchdog live native-like hang, normal finish/deadline race, equality at
   180 seconds, final Connection drain, late quarantined result, result before
   finish, finish before result, 5-second next-row join guard, out-of-order
   row-3 finish while row 0 hangs, multiple active slot deadlines, worker death,
   broken stream, bounded terminate-then-kill teardown, schedule deadline, and
   proof imap poll never attributes time;
5. all four successful RowResult product branches, candidate versus selected-
   product fallback counting, resource fallback exclusion from no-origin/path,
   and proof that a failure-prefix candidate fallback remains in
   `ArmFailureRecord` without becoming a success denominator; repair80 rejects
   duplicate cache_audio_key but accepts duplicate audio_group_key and still
   produces exactly 80 identity-level denominator units in the synthetic
   fixture, while schedule16 retains its one-to-one source grouping;
6. every `ArmFailureRecord` causing-field nullable branch, exact contiguous
   completed/pending preimages and prefix counts, each `ArmStageOutcome` union
   branch, all later-arm `NotRunArmRecord` chains, and fatal failure-publication
   behavior; bounded diagnostics malformed/cap failures map exactly to
   diagnostics_integrity_failure/diagnostics while JSON/candidate artifact byte
   caps alone map to artifact_resource_cap; complete-object round trips prove
   schedule/repair `stage_summary_sha256` and
   `arm_failure_record_sha256` preimages;
7. candidate reference 16-entry schedule and 80-entry repair ref-only
   manifests, bundle reopening/recomputed refs, exact entry-payload SHA naming,
   bound-row equality, later-arm direct S30 byte compare, global entry arm map,
   payload/bundle/manifest caps at limit-1 and limit using a large dense
   synthetic candidate, and crash windows before fsync, before replace, and
   before directory fsync; stale/swapped/gapped/orphan refs fail closed;
   reference-arm row-only prefixes fail closed while S60/S90/S64 row prefixes
   resume only after S30 reference-manifest byte comparison;
8. every cross-arm identity/cache/source/config/restricted/candidate/v2/schema
   mismatch, including four successful arms followed by join failure; each must
   yield FourArm hard failure with null global manifest, not-run source
   selection, null ConfigSelection, and no trusted majority; post-arm schedule
   deadline fixtures cover cross-arm join, candidate-global publication,
   pending ConfigSelection, and FourArm summary fsync; crash after selection
   fsync/before final-marker fsync, missing/failing/mismatched marker, and
   orphan-selection resume all quarantine the selection and forbid weak/repair;
   only one validated FourArm success marker commits source selection;
9. SourceClosure behavior/audit mutation matrix and proof all downstream
   fingerprints bind only `source_closure_fingerprint_sha256`, never full audit
   payload, absolute root, PID, time, Git commit, or unrelated dirty state;
10. ArmOrderValues E0/E1/null matrix, exact elimination enum order, runtime/RSS
   gates but exclusion from tuple, common-set sorted-key hashes, fewer-than-two
   ambiguity, E1 negative, tie ranks, pending selection before final-marker
   commit, and weak/resume validation order marker -> global/selection -> weak;
11. weak comparator unavailable/conflicting/current-v2/pure/selected flag null
    matrix for both PhaseSummary and DriftSummary; three distinct boundary
    summaries; RatioValue finite/both-zero/positive-infinity/undefined; and
    sorted-key bindings for every weak common set;
12. all five schedule weak fatal stages/kinds, contiguous completed/pending
    refs, weak success/hard outcome hashes, outcome-first resume and fatal
    publication quarantine; ambiguous/negative success blocks repair and only
    success+pass authorizes repair RunConfig/input/summary;
13. WeakRow schedule/repair stage union, stale/swapped/cross-stage RowResult
    pairing, exact WeakRowRef and row_weak_pairs SHA preimages for schedule and
    repair, and prediction-row SHA equality;
14. schedule and repair boundary delta tests at `-0.100...`, `-0.05`, either
    side, insufficient 4/14 versus sufficient 5/15, and a near-zero absolute
    current-v2/pure F1 fixture proving delta can pass without an absolute
    boundary-adequacy claim;
15. all `SourceArm*`, `ScheduleWeak*`, `Repair80*`, FourArm success/failure,
    action/decision, count/hash, resume, atomic, source/cache closure, exposure,
    publication null matrices, and repair80 weak-input/comparator/metrics/
    schema/publication ArmFailureRecord kind/stage encodings; Repair80 fixtures
    cover stable/jump/long identity predicates, long+stable and long+jump
    overlap, dense/ramp/ambiguous exclusion, class-before-intersection,
    fallback exclusion, every exact common-set sorted-key hash,
    stable-section operand membership, and rejection of all six invalid
    decision/action cross-pairs.

## Verify Command / Evaluation Procedure

1. Freeze this card bytes and receive independent blocker/scientific review.
2. Add tests first for bounded/full equality and strict protocol edge cases.
3. Implement only the scoped source modules.
4. Run the smallest focused test per module.
5. Run the complete local-frontier/Exp006 guard and five oracle checks.
6. Run every Exp007 synthetic test, then the full Timing-v3 suite.
7. Run compilation and document/diff checks.
8. Audit commands and tests for absence of real asset/network access.
9. Write a result/problem-log entry only after reviewed verification. Do not
   write Exp008 or execute real data in this card.

Expected commands, source/synthetic only:

```sh
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py \
  tests/timing/test_timing_v3_boundary_pair_transition_matrix.py --tb=short
.venv/bin/python -m pytest -q tests/timing/test_timing_v3_exp007_*.py --tb=short
.venv/bin/python -m pytest -q tests/timing/test_timing_v3_*.py --tb=short
.venv/bin/python -m py_compile \
  src/pulsefield_model/timing/v3/local_frontier.py \
  src/pulsefield_model/timing/evaluation/exp007_*.py
git diff --check
```

No allowed command names an `artifacts/`, cache, audio, `.osu`, label/identity
manifest, schedule16, repair80, holdout, broad, full5050, API, or network input.

## Guard Check

- source/doc/synthetic boundary is enforced by tests;
- strict recursive schema and descriptor SHAs reject every mutation class;
- old API/base result/default Exp005 and all Exp006 oracles are unchanged;
- bounded diagnostics are excluded from inference semantics and memory-bounded;
- runner uses four separate fixed-order arms, four fresh workers each, exact
  timeout/death semantics, and stage-global cross-arm equality;
- source selector has no weak-evidence dependency; only the final successful
  FourArm marker commits its pending winner before weak access;
- hard failures produce no projection/product/pairs;
- rows retain immutable candidate-method/product/current-v2 grids and
  candidate digests while remaining below 1 MiB; full candidate payloads live
  only in the capped bound reference bundles;
- atomic, resume, source/cache closure, and exposure tests pass;
- no Exp008 card or real artifact is produced.

## Qualitative Check

Review source, test fixtures, schema descriptors, synthetic worker envelopes,
overlap lineage traces, and immutable synthetic artifacts only. Do not inspect
real audio/cache/`.osu` rows or earlier generated metric artifacts.

## Positive Signal

- all bounded/full behavior and oracle guards pass;
- overlap is measurable from exact state lineage without full ledgers;
- selector, source ordering, weak veto, product truth, and artifact replay are
  mechanical and complete;
- no real input was opened and no production path changed;
- a later Exp008 can be written without changing this protocol.

## Negative Signal

- bounded mode changes any grid, reason, base diagnostic, search count, or
  frozen oracle;
- overlap needs rescore/backfill or unbounded ledgers;
- cross-arm candidate/v2 equality cannot be proven;
- selection needs weak labels, network, or outcome-dependent normalization;
- worker timeout/death cannot be attributed fail-closed;
- required immutable row content exceeds the cap;
- resume/source/cache/atomic integrity cannot be enforced.

## Kill Criteria

Kill or mutate Exp007 before real execution if any negative signal occurs; if
passing requires changing Exp006 objective, candidate, schedule, cap, search,
tie-break, grid schema, production integration, ramp behavior, real-data gate,
or any frozen threshold; or if any test opens a forbidden real input.

## Expected Failure Modes

- a private lineage field may accidentally enter dataclass equality or future
  equivalence;
- the prior lookahead trace may be a union instead of the exact ranked path;
- a pruned lineage may be incorrectly backfilled;
- candidate canonicalization may include schedule/volatile state;
- current-v2 unavailable exceptions may be classified by text;
- `imap` may hide a dead worker behind ordered output;
- a recursive validator may accept an extra nested key or bool as int;
- deterministic closure may bind absolute root or unrelated dirty files;
- full grid payloads may approach 1 MiB on synthetic cap fixtures;
- a weak evaluator import may leak into source ordering.

## Confounders

- Exp007 establishes protocol correctness, not real-cache quality.
- Synthetic overlap traces do not establish real long-track disagreement.
- macOS/Linux `ru_maxrss` is a lifetime high-water mark and depends on runtime
  versions and cache state.
- schedule16 and repair80 are exposed diagnostic sets, not fresh acceptance.
- `.osu` maps remain correlated annotations; winner-only veto reduces but does
  not eliminate comparator bias.
- the bounded overlap metric measures recomputation of retained lineage, not a
  counterfactual state pruned at the next cut.

## Error Attribution

Exp007 failures are assigned to the earliest confirmed layer:

1. schema/canonical encoding;
2. selector/source-label identity;
3. source/cache/config closure;
4. restricted prediction/candidate sharing;
5. current-v2 availability;
6. Exp006 bounded/full parity;
7. lineage/overlap diagnostics;
8. worker timeout/death/RSS;
9. cross-arm join;
10. source E0/E1/order;
11. weak winner veto;
12. product/denominator reducer;
13. atomic/resume/exposure publication.

Later weak evidence cannot overwrite an earlier source/protocol failure.

## Expected Runtime / Runtime Budget

Exp007 source tests should remain under the existing Timing-v3 guard budget;
the 44-arm differential matrix should remain under two minutes on the recorded
CPU environment. Synthetic worker alarm/death tests use deliberately short
injected deadlines, never the real 180-second wait.

Future Exp008 budgets are frozen here for protocol tests only:

- each audio/arm: `<180 s`;
- four schedule16 arms total: `<20 min` hard stop;
- selected repair80 arm: `<30 min` hard stop;
- exactly four spawn workers per independent arm/stage;
- worker lifetime peak RSS `<=4 GiB`.

## Result Interpretation Plan

- Positive result would suggest: the Exp006 candidate is ready for a separately
  preregistered schedule16/repair80 execution card, not that it is accepted.
- Negative result would suggest: repair only the attributed protocol or bounded
  measurement layer before opening real cache rows.
- Ambiguous result would require: one narrower no-data protocol card; do not
  weaken a threshold or create Exp008.
- Human owner decides: whether reviewed source/test/card hashes authorize
  writing Exp008.
- Next-loop action if positive: write Exp007 result/problem-log entries, then
  separately draft and review Exp008.
- Next-loop action if negative: `MUTATE` the smallest failed layer.
- Next-loop action if ambiguous: preserve the no-data boundary and stop.

## Result Log Template

- Experiment: Timing v3 Experiment 007
- Date:
- Frozen card SHA-256:
- Source closure / schema descriptor hashes:
- Changed source/test files and SHAs:
- FULL/BOUNDED matrix equality:
- Five oracle results:
- Overlap lineage/cap tests:
- Strict schema/selector replay tests:
- Runner timeout/death/RSS/candidate-sharing tests:
- Source ordering/common-set/winner tests:
- Winner-only weak-veto tests:
- Product truth/denominator/reducer tests:
- Atomic/resume/source/cache/exposure tests:
- Row byte-cap tests:
- Full Timing-v3 guard:
- Real-data access audit:
- Production-path change audit:
- Decision: `TEST | MUTATE | KILL`
- Exp008 created: `no`
- Human owner decision:

## Pre-Execution Gate

- Card status: pending renewed independent blocker and scientific review of
  this revision; completeness is not claimed before that review.
- Code execution allowed after review: yes, only if that review accepts this
  card, and only for the source/doc/synthetic implementation and tests named
  here.
- Real cache/audio/`.osu`/identity/label/artifact/network access: no.
- Schedule16 execution: no.
- Repair80 execution: no.
- Holdout/broad/full5050 execution: no.
- Production fitter/provider/config change: no.
- Ramp change: no.
- Exp008 creation in this card: no.
- Closed loop complete: no; review remains before implementation may begin.
- Remaining ambiguity: reviewer findings plus later empirical real-cache
  results; real evidence belongs only to an accepted Exp008 card.

## Next-Loop Action

- If positive: freeze Exp007 result and problem log, then draft a separate
  Exp008 card for exactly schedule16 -> source-winner commit -> winner-only weak
  veto -> selected-arm repair80.
- If negative: mutate only the failed protocol/diagnostics layer without real
  data.
- If ambiguous: stop at the no-data boundary for human decision.

## Novelty Notes

- Closest analogies: preregistered evaluation, fixed-lag lineage diagnostics,
  content-addressed artifacts, and repair-set execution.
- Novelty layer: none.
- Representation novelty versus engineering variation: engineering protocol
  for Pulsefield's phase-continuous absolute-beat representation.
