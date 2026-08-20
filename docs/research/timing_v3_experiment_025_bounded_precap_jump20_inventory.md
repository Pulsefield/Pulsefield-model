# Timing v3 Experiment 025: Bounded pre-cap jump20 inventory repair

Status: completed / valid diagnostic / stopped before mutation or broader rows

## Mode

- Mode: executor result freeze
- Route: TEST diagnostic repair
- Source idea: Exp023 failed as an observability run because its JSON
  occurrence inventory exceeded its per-row and total storage guards. Preserve
  that invalid run and retry the same already-exposed jump20 question with a
  deterministic bounded binary encoding.
- Acceptance source: the human owner authorized drafting and exploratory work.
  Independent card review, synthetic self-check, harness review, and run-plan
  review approved exactly one real exact20 run. Post-run integrity audit found
  no invalidation.
- Source snapshot / evidence grade: frozen local Exp022 evidence and the frozen
  invalid Exp023 result log. Exp022 remains invalid for advancement because of
  runtime; Exp023 produced no lifecycle classification.

## Hypothesis

Replacing only Exp023's verbose occurrence serialization with streamed,
fixed-width, source-only row shards will preserve the unchanged candidate
lifecycle while fitting the already-observed 28k--73k unique
`(fingerprint_sha256, source)` occurrences per row under a strict artifact
bound. The resulting frozen boundary signatures are sufficient to classify
direct candidates after, and only after, all source and inference outputs are
durable.

## Root Objective

Recover the unanswered Exp023 lifecycle diagnosis for exactly the same 20
already-exposed jump rows without changing proposal generation, candidates,
scores, retention, eligibility, selection, evaluator logic, thresholds,
fallback, or product output.

## Goal Decomposition

- Freeze every distinct pre-cap `(fingerprint_sha256, source)` occurrence's
  direct-relevant boundary signature and source/family/rank metadata, plus
  duplicate counts and a tuple-order digest covering every raw proposal, in
  bounded deterministic storage.
- Freeze the unchanged retained batch losslessly and prove hooked inference is
  behavior-identical to an uninstrumented source shadow on all 20 rows.
- Only after 20/20 source, runner, shadow, and inference freezes are durable,
  apply the existing weak comparator to produce one lifecycle bucket per row.

## Candidate Variants

- Variant A: repeat Exp023's canonical JSON occurrence table. Rejected: the
  frozen invalid run already hit 11 per-row guards and 6 total-size guards.
- Variant B: one compressed NPZ archive. Rejected: compression adds library-
  and timestamp-sensitive bytes, peak-memory risk, and poor row-level failure
  durability.
- Variant C: one uncompressed, fixed-width NPY shard per row plus canonical
  JSON schema/SHA manifests and lossless retained JSONL. Selected.
- Variant D: sample, hash-set, Bloom-filter, or boundary-only-positive records.
  Rejected: lossy storage cannot distinguish not-generated from pruned or
  reconstruct source-family/rank diagnostics. Variant C's equivalence-class
  aggregation is not sampling: every unique `(fingerprint, source)` key is
  retained, duplicate multiplicity and first/last/best ranks are retained, and
  a rolling digest commits to the full raw tuple order.

## Local Verification Matrix

- A is already negative under the frozen Exp023 result.
- B fails unless two independent writes are byte-identical and each completed
  row is independently durable; the proposed format does not make that easy.
- C passes only if an `80,001`-unique-occurrence synthetic row round-trips exactly,
  independent writes have the same SHA-256, the static byte proof passes, and
  hooked versus unhooked behavior is exactly equal.
- D fails any fixture in which the only direct candidate is a discarded
  occurrence.

## Selected Variant

Select Variant C only. This is a storage-encoding and streaming change to a
temporary diagnostic harness. It is not a Timing-v3 algorithm change and does
not authorize edits to package source, tests, configs, or evaluators.

## Selection Pressure

- Primary: complete and deterministic lifecycle observability for 20/20 rows.
- Guard: zero product-behavior difference from unchanged source.
- Storage: source bundle at most `125 MiB`; all Exp025 artifacts at most
  `150 MiB`, both stricter than the requested `200 MiB` ceiling.
- Kill: any need to sample proposals, alter inference/evaluation, or read an
  oracle before the complete source/inference freeze.

## Research Question

Can fixed-width source streaming repair Exp023's failed observability without
changing what Timing v3 generates or selects?

## Closest Analogies / Novelty Layer

- Closest analogies: append-only trace shards, columnar event logs, compiler
  pass provenance, and beam-survivor audits.
- Relevant taxonomy bucket: diagnostic observability repair.
- Novelty layer: none claimed.
- Representation novelty vs engineering variation: engineering variation only.

## Minimal Change

After independent acceptance of this card, create one new temporary harness.
It may intercept the same pre-retention call as Exp023 inside `try/finally`.
For each row it must call the original retention delegate exactly once with the
identical proposal tuple/config, stream the source shard, serialize the
returned retained batch, durably close/hash both, and return the identical
retained batch object. The package source remains unchanged.

The hook must be restored before a second, uninstrumented shadow inference over
the same row inputs. The shadow must run in a fresh Python interpreter that
never installs the hook; an in-process post-hook call is insufficient. The
canonical `behavior_projection_v1` defined below must be exactly equal between
hooked and clean-shadow inference; only excluded harness/runtime fields may
differ.

## Files Likely to Change

Only after independent card acceptance:

- new temporary harness
  `/private/tmp/timing_v3_exp025_bounded_precap_jump20_inventory.py`;
- this card's result-log section after an independently approved run;
- new Exp025 artifact paths listed below.

No package source, tests, configs, evaluator, Exp022 artifact, or Exp023 path
may change.

## Read-Only Context Files

- `docs/research/timing_v3_experiment_022_pareto_retention_pilot42_replay.md`
- `docs/research/timing_v3_experiment_023_precap_jump20_inventory.md`
- `docs/research/timing_v3_problem_log.md`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- `src/pulsefield_model/timing/evaluation/exp013_pilot.py`
- `src/pulsefield_model/timing/evaluation/curve_metrics.py`
- `src/pulsefield_model/timing/evaluation/exp004_metrics.py`

## Dataset Slice And Identity Semantics

Use exactly the same 20 high/medium, non-ambiguous `jump_candidate` identities
already exposed by Exp022/Exp023. Do not infer on stable rows or open any new
identity.

`local_ordinal` is newly assigned `0..19` after sorting the authorized jump
records by their Exp022 `execution_index`. It is never interchangeable with
`exp022_execution_index`; the field name `ordinal` is forbidden in new fixed
schemas. The frozen sanity mapping is:

- Exp022 `execution_index=14` is Exp025 `local_ordinal=4`, cache-key SHA-256
  `6d03cb10fbcfb1372cb1e632828f271da436ada4776b610b3c9db05ca7e7a788`,
  and is the known selected-direct row.
- Exp025 local ordinals `0` and `1` are the known retained-but-ineligible rows.

These are post-freeze consistency checks only. The harness must not branch on
them or on weak truth.

Routing must reuse Exp023's corrected authority: verify the five frozen Exp022
artifact SHAs, select `.identities[*].execution_index` from the 42-record
identity root, join output/freeze by that execution index, and project only the
sanctioned source/input fields into a new 20-row execution snapshot. The
documented sorted-cache manifest is not ordering authority.

The five self-contained Exp022 authorities are fixed as:

- identity
  `artifacts/reports/timing/timing_v3_exp022_pilot42_identity_v1.json`,
  SHA-256
  `6b315460900d7a569c0e3523b0de0b4f1c902b398c93f1f7bf10763a06d1c4f6`;
- output
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_v1.jsonl`,
  SHA-256
  `df3cb61bc8aec41284f27cf2c75d0281046d043216db83196617fbd06989e173`;
- summary
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_summary_v1.json`,
  SHA-256
  `00e956ab4fdf66e5cca75cf359a910c40132675957b3906e6174fbf213afb057`;
- durable inference freeze
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_freeze_v1.jsonl`,
  SHA-256
  `bbefbc76aad2469ab4af02c7ab4fe6fda805d370dc9d9cc48aea05eef9188988`;
- fixed-1-second audit
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_audit_v1.json`,
  SHA-256
  `2b2e8cada2768827e1d78254ebfabb3cde9689b0ca95662e40d4fcfce8424d77`.

No holdout100-v2, structure-manifest6, repair80, broad500, full5050,
ramp/dense identity, network source, audio listening, or render is authorized.
Pilot80 may only be streamed post-freeze for the exact 20 cache hashes.

## New Authoritative Paths

Every path is new and preflight must refuse if any exists:

- `artifacts/reports/timing/timing_v3_exp025_jump20_identity_v1.json`
- `/private/tmp/timing_v3_exp025_jump20_execution_snapshot_v1.jsonl`
- `artifacts/reports/timing/timing_v3_exp025_jump20_resource_pre_v1.json`
- `artifacts/reports/timing/timing_v3_exp025_jump20_source_inventory_v1/`
- `artifacts/reports/timing/timing_v3_exp025_jump20_source_summary_v1.json`
- `artifacts/reports/timing/timing_v3_exp025_jump20_hooked_runner_v1.jsonl`
- `artifacts/reports/timing/timing_v3_exp025_jump20_hooked_runner_summary_v1.json`
- `artifacts/reports/timing/timing_v3_exp025_jump20_hooked_freeze_v1.jsonl`
- `artifacts/reports/timing/timing_v3_exp025_jump20_shadow_runner_v1.jsonl`
- `artifacts/reports/timing/timing_v3_exp025_jump20_shadow_runner_summary_v1.json`
- `artifacts/reports/timing/timing_v3_exp025_jump20_shadow_freeze_v1.jsonl`
- `artifacts/reports/timing/timing_v3_exp025_jump20_behavior_shadow_v1.json`
- `artifacts/reports/timing/timing_v3_exp025_jump20_resource_terminal_v1.json`
- `artifacts/reports/timing/timing_v3_exp025_jump20_postfreeze_audit_v1.json`
- `artifacts/reports/timing/timing_v3_exp025_jump20_run_manifest_v1.json`

The run manifest is written last and records `complete`, `invalid`, or
`hard_failure`. Completed row shards and terminal resource snapshots must be
preserved even on invalid/exception paths; rerunning into the same paths is
forbidden.

## Deterministic Fixed-Width Source Schema

The source directory has an exact closed layout: root `schema.json` and
`manifest.json`, plus `row_00` through `row_19`; each row directory contains
only `occurrences.npy`, `retained.jsonl`, and `manifest.json`. Temporary sibling
names are never referenced by the root manifest and must be absent at terminal
closure. Extra members invalidate the run.

Each `row_XX/occurrences.npy` uses NPY v2.0, C order,
`allow_pickle=False`, little-endian numeric fields, `align=False`, and exact
`itemsize=120`. It contains one record for every unique
`(fingerprint_sha256, source)` key, ordered lexicographically by raw
fingerprint bytes and then UTF-8 source bytes. Field offsets are normative:

| offset | field | dtype / bytes |
| ---: | --- | --- |
| 0 | `fingerprint_sha256` | `u1[32]` / 32 |
| 32 | `boundary_time_ms` | `<f8[3]` / 24 |
| 56 | `left_bpm` | `<f4[3]` / 12 |
| 68 | `right_bpm` | `<f4[3]` / 12 |
| 80 | `best_generation_score` | `<f4` / 4 |
| 84 | `collapse_bpm` | `<f4` / 4 |
| 88 | `aba_support_delta` | `<f4` / 4 |
| 92 | `minimum_generation_rank` | `<u4` / 4 |
| 96 | `maximum_generation_rank` | `<u4` / 4 |
| 100 | `duplicate_count` | `<u4` / 4 |
| 104 | `source_code` | `<u2` / 2 |
| 106 | `family_code` | `<u2` / 2 |
| 108 | `curve_class_code` | `<u2` / 2 |
| 110 | `section_count` | `u1` / 1 |
| 111 | `boundary_count` | `u1` / 1 |
| 112 | `finite_mask` | `u1` / 1 |
| 113 | `flags` | `u1` / 1 |
| 114 | `best_score_generation_rank` | `<u4` / 4 |
| 118 | `reserved` | `u1[2]` / 2 |

Generation ranks are zero-based positions in the original proposal tuple;
`minimum_generation_rank` is also the first occurrence rank. Duplicate records
must agree exactly on fingerprint, source, family, curve class, section count,
and boundary signature. Their `duplicate_count`, minimum/maximum rank, maximum
finite generation score, and earliest rank attaining that score are
aggregated. `collapse_bpm` is taken from the minimum-rank occurrence and
`aba_support_delta` is the maximum finite value; flag bits and row-manifest
counts record any duplicate variation or finite/absent mixture in those
diagnostic-only values. The tuple digest still commits to every exact value.
Any disagreement in an equivalence-required structural field invalidates the
diagnostic before oracle access.

Source-rank diagnostics are derived after mmap by sorting records within each
source code by `(minimum_generation_rank, fingerprint_sha256)`; family ranks
are derived within each family code by
`(minimum_generation_rank, fingerprint_sha256, source_code)`. They are not
lossily narrowed into the fixed record. Source and family codebooks are ordered
by first raw-tuple occurrence and recorded losslessly in the row manifest.
`source_code`, `family_code`, `curve_class_code`, every `u4` count/rank, and
the codebook lengths have explicit overflow guards. `reserved` and unused flag
bits are zero.

The row manifest stores the raw `proposal_count` and a
`proposal_tuple_sha256`. That digest is updated in original tuple order with a
length-prefixed canonical source-only record containing the zero-based rank,
fingerprint, UTF-8 source, exact `float.hex()` encodings of score and collapse
BPM, nullable exact `float.hex()` ABA support, and complete analytic-curve
canonical bytes. It commits to every raw proposal and its order without making
duplicate storage control lifecycle classification.

All current proposals have at most four sections and therefore at most three
boundaries. Populated boundary slots contain source `time_ms` as float64 and
left/right BPM as float32; unused slots are exact positive zero and ignored by
`boundary_count`. The harness must record the maximum float32 BPM round-trip
error. If any post-freeze BPM decision lies within twice that row's maximum
round-trip error of the tolerance boundary, the result is encoding-ambiguous
and the diagnostic is invalid rather than silently rounded.

`finite_mask` marks presence of `best_generation_score`, `collapse_bpm`, and
`aba_support_delta`; absent values encode positive zero, and present values
must be finite. These values are source-only diagnostics, not classifier
inputs. Fingerprint and boundary/BPM signatures are classifier inputs only
after oracle access is unlocked.

Mask/flag bits are closed and normative. `finite_mask` bits `0,1,2` mean
best-score, minimum-rank collapse-BPM, and any finite ABA-support value are
present; bits `3..7` are zero. `flags` bit `0` means `duplicate_count > 1`, bit
`1` means collapse values varied, bit `2` means ABA values mixed finite and
absent, bit `3` means finite ABA values varied, and bit `4` means finite score
values varied; bits `5..7` are zero. `duplicate_count` is at least one.
Curve-class codes use the root codebook in fixed order `constant`, `jump`,
`ramp`; unknown classes are rejected rather than dynamically assigned.

Each row also contains `retained.jsonl`, a lossless canonical source-only
serialization of every returned retained record in exact delegate order,
including its complete analytic sections and available proposal metadata.
The row manifest must prove count, order, and fingerprint-by-fingerprint
equality with the delegate batch. No eligible, selected, lane, fallback,
oracle, title, map, or weak-comparator field is permitted.

Each row shard is streamed to sibling temporary files, flushed and fsynced,
hashed, then atomically replaced into its final row path before the delegate
returns. Canonical row/root JSON manifests use sorted keys, compact separators,
UTF-8, one trailing newline, finite JSON numbers, explicit dtype descriptor,
shape, byte count, codebooks, and SHA-256 for every member. No ZIP container,
compression, object array, implicit native endian, or pickle is permitted.

## Static Byte Bound

The frozen Exp023 evidence bounds unique `(fingerprint, source)` occurrence
records, not unaggregated proposal-tuple length. The 11 rows stopped by the
per-row guard reported counts summing to `633,834`; the three durable rows
reported `39,585 + 39,036 + 27,946 = 106,567`; and each of the six rows that
reached the later total-byte guard necessarily passed the inclusive
`<=40,000` per-row guard. Therefore the conservative exposed envelope is:

```text
633,834 + 106,567 + 6 * 40,000 = 980,401 occurrence records
980,401 records * 120 bytes = 117,648,120 bytes = 112.20 MiB
```

Raw proposal count is separately committed by each row manifest and tuple
digest but does not determine shard bytes. Each unique occurrence has at most
three boundaries.

Reserve at most `8 MiB` for all 880-or-fewer retained full records and at most
`2 MiB` for NPY headers, schemas, codebooks, and manifests. The source bundle
hard cap is `125 MiB`. All remaining new paths, including the `/private/tmp`
snapshot and both runner/freeze lanes, have a combined `25 MiB` cap. Total
Exp025 bytes are therefore capped at `150 MiB < 200 MiB`.

The post-freeze audit must not drop zero-direct candidates when computing
secondary diagnostics. For each row it embeds, in the existing audit JSON and
without adding a new artifact path, one packed base64 blob aligned one-to-one
with the source `occurrences.npy` record order. The packed dtype is exactly
six bytes per source occurrence:

| offset | field | dtype / bytes |
| ---: | --- | --- |
| 0 | `accepted_direct` | `u1` / 1 |
| 1 | `accepted_non_direct` | `u1` / 1 |
| 2 | `unmatched_predicted` | `u1` / 1 |
| 3 | `unmatched_weak` | `<u2` / 2 |
| 5 | `flags` | `u1` / 1 |

Flag bit `0` records all accepted matches direct, bit `1` records at least
one direct accepted match, bit `2` records at least one non-direct accepted
match, bit `3` records any unmatched predicted or weak boundary, and bit `4`
records zero predicted significant in-coverage boundaries. Bits `5..7` are
zero. Each row audit records the packed byte count, SHA-256, dtype descriptor,
source occurrence NPY SHA/count, and a row candidate-metric tuple SHA.
Duplicate source occurrences with the same fingerprint must have identical
packed metric bytes; any disagreement invalidates the diagnostic before
bucket interpretation. The worst-case packed payload is
`980,401 * 6 = 5,882,406` raw bytes; base64 expansion is
`ceil(5,882,406 / 3) * 4 = 7,843,208` bytes, about `7.48 MiB`, which remains
inside the unchanged `25 MiB` non-source reservation and therefore preserves
the `150 MiB` total cap.

Before real execution, a checked integer byte calculator must reject any
schema itemsize other than 120, unique-occurrence total above 980,401, section
count above four, retained total above 880, source forecast above 125 MiB,
other-artifact reservation above 25 MiB, or total forecast above 150 MiB.
Runtime enforcement repeats the same guards while streaming. A breach freezes
the partial invalid evidence, writes terminal snapshots/manifests, and forbids
oracle access. Raw proposal counts and duplicate counts must fit unsigned
64-bit manifest integers; fixed-record ranks/counts must fit their declared
`u4` fields.

## Baseline / Comparator

Baseline is unchanged Exp021/Exp022 source plus the frozen invalid Exp023
observability run. Exp023 persisted only 3 source/freeze rows, produced 17
instrumentation hard failures, wrote no post-freeze audit, and therefore has
no lifecycle result. Its artifacts must not be completed or reinterpreted.

Behavior comparator is a second uninstrumented invocation of the unchanged
source on each same sanitized input in a fresh interpreter that never installs
the hook. Hooked and shadow `behavior_projection_v1` canonical bytes must be
exactly equal for 20/20 rows.

For each row, `behavior_projection_v1` contains exact identity plus every
behavior-bearing product field: result/frozen schema and product status;
failure stage/error; fallback; candidate count and complete serialized
candidate identities/order/classes/sources/generation scores/raw score fields;
eligible indices; production-selection lane, ranks, paired gains, and raw-run
diagnostic; selected index/fingerprint/class and complete selected analytic
curve; all tempo-track diagnostics/version fields; maximum seam and complete
seam/serialization report; and the complete `frozen_inference.tempo_track`
payload. The matching freeze-row candidate inventory digest/class counts,
selected curve digest, tempo schema/version, and frozen inference digest are
also included.

Only wall-clock/runtime fields, write sizes/times, RSS, resource snapshots,
weak-evaluation/oracle fields, environment timestamps, and harness-only
metadata are excluded. A fixture must prove that changing any included field
breaks equality and changing only an explicitly excluded field does not. The
clean shadow is launched through a fixed `--shadow-worker` entrypoint after
hook restoration; the worker must reject hook installation and write only the
fixed shadow runner/freeze paths.

## Primary Metric

Exactly one lifecycle bucket per local ordinal:

- `not_generated`
- `generated_pruned`
- `retained_ineligible`
- `eligible_not_selected`
- `selected`

Greedy fixed-window matching and the two-side BPM tolerance remain the existing
evaluator semantics: `+/-1000 ms` and `max(1 BPM, 1%)` independently on each
side. Before matching, candidate seams use the frozen
`curve_metrics._predicted_boundaries` coverage rule and its significant-change
test `abs(log2(right_bpm / left_bpm)) >= log2(1.005)`; weak boundaries use the
frozen oracle extraction over the same coverage. Fixed matching then sorts all
within-window pairs by
`(absolute_error, predicted_time, weak_time, predicted_index, weak_index)` and
greedily accepts unused indices, exactly as the frozen Exp022 audit.

For lifecycle presence, a candidate is **direct** iff at least one greedy
accepted boundary match has both left and right BPM direct. Other accepted
matches that miss one or both BPM sides are reported but do not disqualify that
candidate. Separately report
`full_candidate_all_matches_direct = accepted_match_count > 0 and
direct_match_count == accepted_match_count`; this stricter diagnostic does not
control the lifecycle bucket.

Bucket precedence is selected, eligible-not-selected, retained-ineligible,
generated-pruned, then not-generated, based on existence of direct candidates
at each frozen stage.

## Secondary Metric

- counts and SHAs for raw proposals, unique `(fingerprint, source)`
  occurrences, duplicate multiplicities, boundaries, retained records, hook
  calls, runner rows, freeze rows, and shadow-equal rows;
- direct accepted, non-direct accepted, unmatched, and all-matches-direct
  boundary counts per candidate and row, including zero-direct candidates via
  the packed all-source-occurrence audit metric described above;
- direct fingerprints by pre-cap, retained, eligible, and selected stage;
- unique-fingerprint row aggregate counters, duplicate-fingerprint metric
  consistency checks, row candidate-metric tuple SHA, and row-level sums named
  `predicted_boundary_total`, `accepted_direct_boundary_total`,
  `accepted_non_direct_boundary_total`, `unmatched_predicted_boundary_total`,
  and `unmatched_weak_boundary_total`;
- direct fingerprints by pre-cap, retained, eligible, and selected stage;
- best direct generation/source/family ranks and source families;
- NPY bytes, manifest bytes, total bytes, write time, inference time, and peak
  harness RSS in explicit bytes, reported separately from product runtime.

## Verify Command / Evaluation Procedure

No command is authorized yet. After independent card acceptance, the harness
must expose a synthetic-only self-check such as:

```sh
uv run --extra mps --group dev python \
  /private/tmp/timing_v3_exp025_bounded_precap_jump20_inventory.py \
  --self-check-only
```

It must cover: the exact 120-byte dtype/offsets; `allow_pickle=False`; all
missing/present finite-mask cases; 0--3 boundaries; acceptance of section
counts `1..4` and rejection of `>4` / four boundaries; an `80,001`-unique-
occurrence all-three-boundary streamed stress shard; two independent writes
with identical bytes/SHA; mmap round-trip; aggregate duplicate/rank/tuple-SHA
round-trip; atomic-replace failure injection; exact retained-order equality;
byte-bound arithmetic; `u2` codebook and `u4` rank/count overflow rejection;
bucket precedence; significant-boundary filtering; existential directness plus
separate all-matches-direct; packed all-candidate audit metrics for
non-direct-only, unmatched-only, mixed direct/non-direct, all-direct, and
zero-predicted candidates; proof that packed audit metrics remain under the
non-source `25 MiB` cap; float32 tolerance-edge invalidation; pre-oracle access
rejection; hook restoration; fresh-process exact behavior-projection equality;
and terminal resource snapshots on success and every injected invalid/exception
path.

It must additionally run a synthetic 20-row lifecycle fixture with the frozen
worst-case occurrence-count vector (the 11 exact guard counts, the three exact
durable counts, and six `40,000` rows), producing exactly 20 row manifests,
hooked runner/freeze rows, shadow runner/freeze rows, one root summary, and one
terminal manifest within the declared total bound. Failure injection is
required immediately before and after a row atomic replace, before and after
root-summary closure, and immediately before oracle unlock. Every injected
path must retain a self-consistent durable-row count, terminal resource
snapshot, and invalid run manifest, and none may unlock oracle access.

Only after that self-check passes and an independent harness review approves
the exact harness may the owner authorize one real run. The real sequence is:

1. verify frozen inputs and all new-path absence;
2. resolve exactly 20 identities and durably write identity/sanitized snapshot;
3. durably snapshot all 20 resource sets before any cache/audio/mel read;
4. run the hook and freeze 20/20 source shards, hooked runner rows, and hooked
   inference rows;
5. restore the hook, launch a fresh-interpreter unchanged shadow worker, and
   freeze 20/20 shadow runner/inference rows plus exact
   `behavior_projection_v1` equality;
6. durably finalize/hash the source root and terminal resource snapshot;
7. lock all inference resource access;
8. only then stream pilot80, retaining only the exact 20 weak rows, and write
   the post-freeze audit and terminal run manifest.

Any count below 20 at steps 4--6 is invalid and must not unlock step 8.

## Guard Check

Before implementation/run, retain Exp023's frozen source guards, including:

- `tempo_track.py`
  `fc4153a6310a4db233e1fbd29e87a57775eff924e4904e925773d172e0d7de85`;
- Exp021 timing test
  `d543ec827a893fb1dddc3513edbe6df2396138ace5d96a58c679002dfeee3ae5`;
- `exp013_pilot.py`
  `70891775233cf0c66b0d948689cf8a7d3505c57192ef0d5beed81f3f22f1b3bb`;
- `curve_metrics.py`
  `1a70a9c0e8f965b9c7a9de74bc1c99711c95b938abda900e871f4fce0f316c2a`;
- `exp004_metrics.py`
  `f88366562d35645fa4264cdb8710d2ea4781c6615a2ea23c2c3c87b0524dc21e`;
- `analytic_curve.py`
  `766762aafdf0a3e643b3689c7ec659d14a0dd002befd2607a410fc368962c4d3`;
- pilot80
  `cdb5e2af87d99c8af3bbff71a0985bb490fdeed169d10cb0a64b17a8ca0296d7`.

The new harness itself must be SHA-frozen after review. Source/family/candidate,
ranking, retention, scorer, selector, fallback, evaluator, schema, and version
hashes must remain unchanged.

## Resource And Failure-Durability Guard

Before the first inference/evaluation resource read, persist and fsync one
canonical integrity snapshot covering the exact 20 resolved BeatThis-cache,
raw-audio, and mel-cache paths, with existence, byte count, and SHA-256;
missing paths are explicit. Computing this snapshot is the only allowed
pre-inference byte read of those resources: it may hash bytes but may not parse
content, derive features, branch on content, or expose any value to inference.
The snapshot itself must be durable before an inference loader becomes
reachable.

Wrap the entire run in `try/finally`. On success, storage invalidation, row
exception, runtime stop, or oracle rejection, persist and fsync the terminal
snapshot and run manifest before returning/raising. File data and parent
directories are fsynced around atomic replacement. Pre and terminal states
must be exact; missing mel paths must remain missing. No cache generation,
audio/mel writes, network, or new identity access is allowed.

## Exp023 Preservation Guard

Preserve every Exp023 artifact byte and keep its absent post-freeze audit
absent. In particular, never write any `timing_v3_exp023_*` path, never append
the three partial source rows, and never reuse an Exp023 temporary snapshot.
Preflight and terminal manifests must record the frozen Exp023 SHAs from its
invalid result log and assert no change. Any mismatch kills Exp025.

## Qualitative Check

Inspect only the final source-family/lifecycle table. Do not listen to audio or
inspect new beatmaps/renders. The check asks whether one lifecycle stage or a
source-derived family split is dominant enough for one later mutation card.

## Positive Signal

- deterministic source shards and complete hooked/shadow equality for 20/20;
- all resource, source, storage, and oracle-order guards pass;
- local 0/1 are retained-ineligible and local 4 / Exp022 14 / `6d03...` is
  selected;
- exactly 20 lifecycle rows reveal a dominant stage or bounded source-family
  split.

## Negative Signal

The bounded encoding cannot preserve complete signatures or unchanged
behavior, or the final classifications are too diffuse to choose one bounded
next mechanism.

## Kill Criteria

Kill before oracle access on any independent-review absence, frozen SHA or
identity mismatch, old/new path collision, count other than 20, stable/new
identity access, source/evaluator mutation, hook call count other than one per
row, returned retained identity/order mismatch, clean-process behavior-shadow
difference, unique-occurrence/section/retained or byte-bound breach, codebook or
rank/count overflow, non-deterministic write, partial runner/freeze set, early
oracle access, resource mutation, float32 tolerance-edge ambiguity, or failure
to persist terminal invalid evidence.

Also kill if selected sanity is tested as local ordinal 14 rather than local
ordinal 4, or if lifecycle directness incorrectly requires every accepted
match to be direct.

## Expected Failure Modes

- a source string/codebook or retained record is not bounded as assumed;
- a candidate exceeds four sections;
- a float32 side BPM lies too close to the directness threshold;
- atomic row persistence fails after retention but before delegate return;
- instrumentation changes object identity or hooked output;
- an invalid path exits before terminal snapshots are durable.

## Confounders

- These 20 rows are exposed and Exp022 is invalid for advancement; results are
  diagnostic only.
- The weak comparator is not musical truth.
- Existential boundary directness localizes lifecycle availability; the
  separate all-matches-direct metric captures stricter full-candidate quality.
- Encoding speed is harness performance, not product timing performance.

## Expected Runtime / Runtime Budget

- synthetic self-check target under 30 seconds, hard stop 60 seconds;
- real diagnostic expected under 10 minutes, hard total 15 minutes;
- any hooked or shadow inference invocation over 45 seconds, or p90 over 25
  seconds per invocation, stops before oracle;
- diagnostic I/O and shadow time are reported separately.

The real diagnostic uses one compute hard deadline, not independent per-stage
budgets: `deadline = monotonic_run_start + 900 seconds`. The hooked
`run_exp013_pilot` call and later potentially long main-thread phases,
including post-freeze audit, must run inside a fail-closed timer context using
`signal.setitimer(ITIMER_REAL, remaining_seconds)`. The signal handler raises a
dedicated deadline `BaseException` that is not caught by ordinary
`except Exception` row-level handlers, and the context restores the previous
handler and timer on exit. If the harness is not running on the main thread, it
must fail closed before starting real work rather than running without the
preemptive timer. The fresh shadow subprocess receives only the remaining
deadline budget; `subprocess.TimeoutExpired` is converted to the same deadline
BaseException. Any deadline timeout is an invalid diagnostic result and must
flow through the outer `except BaseException` path that writes terminal
resource snapshots, SOURCE_ROOT scan, output/freeze row counts, and run
manifest before reraising. Self-check must prove hooked sleep timeout escapes a
synthetic `try/except Exception` runner, shadow remaining-time timeout, and
handler/timer restoration without sleeping more than 0.2 seconds.

## Result Interpretation Plan

- Positive: write one new mutation card against the dominant lifecycle stage.
- Negative: kill this encoding and keep Exp023/025 invalid evidence; do not
  widen data.
- Ambiguous: plan one narrower source-derived-family diagnostic without tuning
  thresholds on these outcomes.
- Human owner decides the next algorithmic direction; no result here advances
  a dataset gate.

## Result Log Template

- Experiment / date / harness SHA / source SHAs:
- Independent card review / harness review:
- Identity count and local-to-Exp022 mapping SHA:
- Exp023 preservation checks:
- Pre / terminal resource snapshot SHAs:
- Dtype itemsize/offset check and double-write SHAs:
- Raw-proposal / unique-occurrence / retained counts and byte-bound calculation:
- Source shards / hook calls / hooked runner / hooked freeze counts:
- Shadow runner / shadow freeze / exact-equality counts:
- Oracle unlock ordering and exact20 stream counts:
- Float32 error margins / ambiguous decisions:
- Lifecycle counts and ordered local-ordinal table:
- Direct / non-direct / all-matches-direct boundary metrics:
- Runtime and storage totals:
- Invalid/hard-failure path durability checks:
- Guards / kill criteria / interpretation / next step / owner decision:

## Authoritative Result Log

- Experiment/date: Exp025 real exact20 diagnostic, frozen on 2026-08-14.
- Result status: `complete`, valid diagnostic, stopped before mutation or
  broader rows.
- Run-bound card SHA-256 before this result-log edit:
  `80ab5aa6a2410e9202fcf8fa0592b056bcf072f4475907ffafa0ee75849152e2`.
- Harness SHA-256:
  `e9985edc14583990ca537795fdf01e209cc0ef81a06706b728952adc3271f9a3`.
- Independent reviews: card, synthetic self-check, harness, run plan, and
  post-run integrity audit passed. The real harness was executed exactly once;
  this result freeze did not rerun it.
- Run manifest:
  `artifacts/reports/timing/timing_v3_exp025_jump20_run_manifest_v1.json`,
  SHA-256
  `52d09b1e3b445ff84f7078d0d2686919645f8960d34e3661e85c40c81ab3dfd6`,
  `status=complete`, `reason=null`.

### Authoritative artifact SHAs

| # | artifact | SHA-256 |
| ---: | --- | --- |
| 1 | `artifacts/reports/timing/timing_v3_exp025_jump20_identity_v1.json` | `ab54f293336fb189d2aff4235bcfe1c890ae2e39764ea73155c9d9df4a888b76` |
| 2 | `/private/tmp/timing_v3_exp025_jump20_execution_snapshot_v1.jsonl` | `88e85a5a5001042638380e87a3f5c83a655c79f61de250cf4c844c5c189abfed` |
| 3 | `artifacts/reports/timing/timing_v3_exp025_jump20_resource_pre_v1.json` | `0c07a2035cf8d4bfd222a68772a873d64f6f0f7e982f6ac64b72e1d1bbf0a658` |
| 4 | `artifacts/reports/timing/timing_v3_exp025_jump20_source_inventory_v1/manifest.json` | `cfb90ce8a66bc78566965e5b2b3a67dee17077206d62d177201c78f7e73d67ec` |
| 5 | `artifacts/reports/timing/timing_v3_exp025_jump20_source_summary_v1.json` | `5a9e5b848e37d0c11b370be2beef7b8bafeba3883224e8b3e26c577ea18a2b08` |
| 6 | `artifacts/reports/timing/timing_v3_exp025_jump20_hooked_runner_v1.jsonl` | `35597385287b3fa1d7694a8d285a55f93a9059c289c6c4fa0b268feea5d9725f` |
| 7 | `artifacts/reports/timing/timing_v3_exp025_jump20_hooked_runner_summary_v1.json` | `3570d6574d049b82d57d1425efa8c91fd1e91971bf7e7221a4c7b0fb99fbed1c` |
| 8 | `artifacts/reports/timing/timing_v3_exp025_jump20_hooked_freeze_v1.jsonl` | `85bb8dae998aa50db9e06c3534ae8e130c37a2791c0c3786ae58f297e06faa87` |
| 9 | `artifacts/reports/timing/timing_v3_exp025_jump20_shadow_runner_v1.jsonl` | `ef1043429f2311d459984e89674bee72066e1f18598f309d627917a45210f791` |
| 10 | `artifacts/reports/timing/timing_v3_exp025_jump20_shadow_runner_summary_v1.json` | `ed04692acd67a1fe0b078da5d7e9f48fb0db370497a8dbcd6f05494c01cd085c` |
| 11 | `artifacts/reports/timing/timing_v3_exp025_jump20_shadow_freeze_v1.jsonl` | `a657f511541492eb0c963ade99f3183c835ead42b9a6fed32e578c9ab2a97bab` |
| 12 | `artifacts/reports/timing/timing_v3_exp025_jump20_behavior_shadow_v1.json` | `0f59f38dd960054af78ab00d02fa9988b46344411cfa55be766d8240d3c46da7` |
| 13 | `artifacts/reports/timing/timing_v3_exp025_jump20_resource_terminal_v1.json` | `0c07a2035cf8d4bfd222a68772a873d64f6f0f7e982f6ac64b72e1d1bbf0a658` |
| 14 | `artifacts/reports/timing/timing_v3_exp025_jump20_postfreeze_audit_v1.json` | `5d16ef252bcf55ce2fa32242af3331c2720a70283e41f79390c35b51f0024c30` |
| 15 | `artifacts/reports/timing/timing_v3_exp025_jump20_run_manifest_v1.json` | `52d09b1e3b445ff84f7078d0d2686919645f8960d34e3661e85c40c81ab3dfd6` |

The source inventory schema file is additionally fixed at SHA-256
`8213bcfee3ea9b3dd66f63a70645b16f3c7d27661080b2996f7a0b214cc06552`.

### Integrity, behavior, and preservation checks

- Identity/source rows: `20/20`.
- Hook calls/source shards/hooked runner/hooked freeze rows: `20/20`.
- Shadow runner/shadow freeze rows: `20/20`.
- Exact `behavior_projection_v1` equality: `20/20`.
- Source occurrence totals: `943,299` unique occurrences, `943,299`
  proposals, `880` retained records.
- Exp023 preservation: all frozen Exp023 artifact SHAs unchanged; the absent
  Exp023 post-freeze audit remains absent.
- Resource snapshots: pre and terminal snapshots are byte-identical, both
  SHA-256
  `0c07a2035cf8d4bfd222a68772a873d64f6f0f7e982f6ac64b72e1d1bbf0a658`.
- Oracle unlock ordering: post-freeze audit ran only after complete source,
  hooked, shadow, behavior, source-root, and terminal-resource freezes.
- Packed secondary metric audit: all 20 row blobs decode from base64; decoded
  bytes match their SHA-256, dtype descriptor, `itemsize=6`, occurrence NPY
  SHA/count, source record order, duplicate-fingerprint consistency, row
  candidate-metric tuple SHA, and row aggregate counters.

### Runtime and storage

- Total wall runtime: `318.93819441701635 s`, below the `900 s` hard deadline.
- Hooked runtime total/p50/p90/max:
  `163.33505979101756 / 8.178463603995624 / 11.368546958288066 /
  14.726815083005931 s`.
- Shadow runtime total/p50/p90/max:
  `144.01662916698842 / 7.405641021498013 / 10.090174158592713 /
  13.260920665983576 s`.
- Harness guard bytes before writing the final run manifest: source
  `113,756,456`, other `11,820,520`, total `125,576,976`.
- Final bytes including the run manifest: source `113,756,456`, other
  `11,835,567`, total `125,592,023`.
- Storage stayed under the source `125 MiB`, non-source `25 MiB`, and total
  `150 MiB` caps.

### Lifecycle result

Bucket counts are:

- `not_generated`: `12`
- `generated_pruned`: `5`
- `retained_ineligible`: `2`
- `eligible_not_selected`: `0`
- `selected`: `1`

Ordered local-ordinal table:

| local ordinal | lifecycle bucket |
| ---: | --- |
| 0 | `retained_ineligible` |
| 1 | `retained_ineligible` |
| 2 | `not_generated` |
| 3 | `not_generated` |
| 4 | `selected` |
| 5 | `not_generated` |
| 6 | `generated_pruned` |
| 7 | `not_generated` |
| 8 | `not_generated` |
| 9 | `generated_pruned` |
| 10 | `generated_pruned` |
| 11 | `not_generated` |
| 12 | `generated_pruned` |
| 13 | `not_generated` |
| 14 | `not_generated` |
| 15 | `generated_pruned` |
| 16 | `not_generated` |
| 17 | `not_generated` |
| 18 | `not_generated` |
| 19 | `not_generated` |

Direct source occurrence counts were nonzero only for local ordinals `0`
(`172`), `1` (`21`), `4` (`116`), `6` (`15`), `9` (`56`), `10` (`36`), `12`
(`10`), and `15` (`336`); all other rows had zero direct source occurrences.

### Interpretation and next step

Exp025 successfully repairs Exp023's observability failure without changing
Timing-v3 behavior. The diagnostic conclusion is not a product-quality pass and
does not advance a dataset gate: on these already-exposed jump20 rows, direct
lifecycle availability is mostly missing before retention (`12/20`
not-generated) or lost pre-cap (`5/20` generated-pruned), with only one direct
candidate selected. The run is therefore frozen as valid diagnostic evidence
only.

The next algorithmic action, if any, must be a new Experiment Card before any
mutation or broader-row access. The nominal-BPM nice-number prior remains
scoped to stable, digitally produced constant segments as a soft
proposal/ranking prior; it is not direct boundary evidence, not a weak-oracle
substitute, and not authorized here for jump-boundary acceptance.

## Execution State

- Card complete: yes
- Independent card review passed: yes.
- Harness review and synthetic self-check passed: yes.
- Real execution: completed exactly once under the approved harness/run plan.
- Post-run integrity audit: passed; no reason found to mark the result invalid.
- Further real execution under this card: no.
- Closed loop complete: yes
- Remaining ambiguity: algorithmic next step only; it requires a new card.

## Next-Loop Action

- Stop before mutation or broader rows.
- If the owner chooses to continue, draft one lifecycle-stage mutation card
  using the frozen Exp025 result as diagnostic evidence only.

## Novelty Notes

- Closest analogies: fixed-width trace logs and survivor provenance.
- Novelty layer: none.
- Representation novelty vs engineering variation: engineering variation only.
