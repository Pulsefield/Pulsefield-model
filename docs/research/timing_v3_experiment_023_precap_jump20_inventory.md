# Timing v3 Experiment 023: Pre-cap jump20 inventory classification

Status: completed / invalid; diagnostic artifacts frozen; no lifecycle
classification result; do not overwrite, complete in-place, or reinterpret
the existing Exp023 artifacts

## Mode

- Mode: planner
- Route: TEST diagnostic
- Source idea: use only frozen Exp022 invalid-but-diagnostic jump rows to
  localize direct `+/-1000 ms` candidate failures before any new algorithm
  mutation.
- Acceptance source: this card authorizes source-only harness implementation.
  It does not authorize real execution until an independent review approves the
  harness and run plan.
- Source snapshot / evidence grade: local frozen Exp022 docs and artifacts are
  diagnostic evidence only because Exp022 failed its preregistered runtime
  integrity gate.
- Frozen result: the single Exp023 real run produced incomplete
  instrumentation artifacts and no post-freeze lifecycle audit. It is a
  harness/instrumentation failure, not evidence for generation, retention,
  eligibility, or arbitration.

## Hypothesis

For the already-exposed high/medium, non-ambiguous jump20 slice, the poor
Exp022 boundary result can be explained by where direct `+/-1000 ms` candidates
fall out of the current lifecycle: not generated, generated before the cap but
pruned, retained but production-ineligible, eligible but not selected, or
selected. A source-only pre-cap inventory frozen before oracle access can
separate proposal-generation failures from cap/retention, eligibility, and
arbitration failures without changing the algorithm.

## Root Objective

Classify each of the 20 frozen Exp022 jump rows into exactly one lifecycle
bucket for direct `+/-1000 ms` candidate availability:

1. `not_generated`
2. `generated_pruned`
3. `retained_ineligible`
4. `eligible_not_selected`
5. `selected`

This is a diagnostic inventory. It must not change candidate generation,
retention, scoring, compatibility, eligibility, selector ranking, thresholds,
caps, fallback, or metrics.

## Goal Decomposition

- Subgoal 1: Freeze a source-only pre-cap and post-retention candidate
  inventory for exactly the 20 Exp022 jump rows before any weak oracle read.
- Subgoal 2: Preserve the current candidate lifecycle unchanged by delegating
  to the original cap/retention function exactly once per row and returning
  its result unchanged.
- Subgoal 3: After the source-only inventory and inference freeze are durable,
  read the already-exposed weak comparator only for post-freeze classification
  and assign each row one lifecycle bucket.

## Candidate Variants

- Variant A: classify only retained Exp022 candidates.
  - Reject. This cannot distinguish not-generated from generated-pruned, which
    is the dominant unknown for the 17 unresolved jump rows.
- Variant B: capture every raw proposal expansion in full.
  - Reject. Exp020 observed `27,946` pre-cap proposals on one row; writing full
    duplicate expansions across 20 rows is unnecessary and creates avoidable
    disk/runtime risk.
- Variant C: source-only pre-cap hook with deduplicated compact inventory, then
  post-freeze oracle classification.
  - Select. It answers the lifecycle question while preserving the production
    code path and bounding storage.
- Variant D: mutate retention or eligibility while collecting the inventory.
  - Reject. Any algorithm mutation would confound diagnosis and violate the
    Exp022 stop rule.

## Local Verification Matrix

- Variant A fails if any row has no retained direct candidate, because it cannot
  tell absence from pruning.
- Variant B fails the storage guard if it serializes duplicate full proposal
  expansions instead of unique curve/source records.
- Variant C passes local verification only if synthetic rows prove pre-oracle
  freeze ordering, exact original-retention delegation, bounded deduplication,
  row identity guards, and deterministic post-freeze classification.
- Variant D fails immediately because this card forbids behavior changes.

## Selected Variant

- Selected: Variant C, a temporary diagnostic harness with a pre-cap capture
  hook after proposal construction and before family retention, plus a
  post-freeze classifier.
- Rejected: any scoring, cap, selector, compatibility, candidate-generation, or
  evaluator mutation.
- Why this is the smallest useful test: the existing Exp022 output says only
  `3/20` post-cap jump rows have any fixed `+/-1000 ms` direct candidate. The
  missing information is whether the other `17` rows lacked direct candidates
  at proposal construction or lost them at the cap.

## Selection Pressure

- Primary pressure: one exact lifecycle bucket per jump row, with a count table
  over the 20-row slice.
- Guard pressure: no oracle fields, weak boundaries, BPM truth, row title,
  beatmap path, map metadata, or `.osu` data may enter the frozen source-only
  inventory or inference path.
- Runtime pressure: diagnostic overhead must remain bounded and reported
  separately from product runtime.
- Kill pressure: if the harness cannot prove unchanged cap/retention
  delegation and pre-oracle durability, do not run real rows.

## Research Question

Does the current Exp021/Exp022 source fail the exposed jump20 rows primarily
because direct candidates are not generated, because they are generated then
pruned by the cap, because retained candidates remain ineligible, or because
eligible candidates lose arbitration?

## Closest Analogies / Novelty Layer

- Closest analogies: compiler pipeline provenance, beam-search survivor
  inventories, preregistered post-hoc error attribution.
- Relevant taxonomy bucket: diagnostic mechanism localization, not an
  algorithmic improvement.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering observability
  over an existing proposal lifecycle.

## Minimal Change

Create a temporary diagnostic harness only. The harness may wrap the existing
candidate generator and temporarily intercept `_retain_jump_proposals_by_family`
inside a `try/finally` scope. It must not edit package source or tests for the
real run.

The capture point is after jump proposal construction has produced
`raw_run_proposals + tuple(proposals)` and immediately before the unchanged
call to `_retain_jump_proposals_by_family`. The capturing delegate must:

1. receive the original proposal tuple and config;
2. call the original retention function exactly once with those exact objects;
3. record the returned retained batch without modification;
4. durably write the row source-only inventory before returning; and
5. return the original retained batch object unchanged.

## Files Likely to Change

Harness implementation:

- a temporary script under `/private/tmp`, for example
  `/private/tmp/timing_v3_exp023_jump20_precap_inventory.py`;
- no package source, test, config, or docs changes during real execution.

This card itself is the only repository file added before review.

## Read-Only Context Files

- `docs/research/timing_v3_experiment_022_pareto_retention_pilot42_replay.md`
- `docs/research/timing_v3_problem_log.md`
- `docs/research/timing_v3_experiment_020_precap_short_aba_inventory.md`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- `src/pulsefield_model/timing/evaluation/exp013_pilot.py`

## Dataset Slice

Exactly the 20 jump-candidate rows from Exp022's high/medium-confidence,
non-ambiguous pilot42 slice. The slice identity must be derived from the
frozen Exp022 identity artifact and verified against the Exp022 card:

- Exp022 identity artifact
  `artifacts/reports/timing/timing_v3_exp022_pilot42_identity_v1.json`,
  SHA-256
  `6b315460900d7a569c0e3523b0de0b4f1c902b398c93f1f7bf10763a06d1c4f6`;
- Exp022 output
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_v1.jsonl`,
  SHA-256
  `df3cb61bc8aec41284f27cf2c75d0281046d043216db83196617fbd06989e173`;
- Exp022 summary
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_summary_v1.json`,
  SHA-256
  `00e956ab4fdf66e5cca75cf359a910c40132675957b3906e6174fbf213afb057`;
- Exp022 durable inference freeze
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_freeze_v1.jsonl`,
  SHA-256
  `bbefbc76aad2469ab4af02c7ab4fe6fda805d370dc9d9cc48aea05eef9188988`;
- Exp022 fixed-1-second audit
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_audit_v1.json`,
  SHA-256
  `2b2e8cada2768827e1d78254ebfabb3cde9689b0ca95662e40d4fcfce8424d77`.

The harness may parse the Exp022 identity/order artifact to select the 20 jump
rows, but it must not infer on stable rows, load stable BeatThis caches, load
stable audio, or load stable mel caches. Stable evidence is limited to frozen
Exp022 docs and aggregate artifact identity checks.

Do not open holdout100-v2, structure-manifest6, broad500, full5050,
ramp/dense rows, or any additional real identity.

## Authoritative Outputs And Schemas

All Exp023 output paths are fixed before harness implementation. The real
harness preflight must refuse to run if any of these paths already exists, and
all bytes below, including the `/private/tmp` execution snapshot, count toward
the `200 MiB` total artifact guard:

- identity20 manifest:
  `artifacts/reports/timing/timing_v3_exp023_jump20_identity_v1.json`;
- sanitized execution snapshot:
  `/private/tmp/timing_v3_exp023_jump20_execution_snapshot_v1.jsonl`;
- source-only pre-cap inventory:
  `artifacts/reports/timing/timing_v3_exp023_jump20_source_inventory_v1.jsonl`;
- source-only inventory summary:
  `artifacts/reports/timing/timing_v3_exp023_jump20_source_summary_v1.json`;
- runner result:
  `artifacts/reports/timing/timing_v3_exp023_jump20_runner_v1.jsonl`;
- runner summary:
  `artifacts/reports/timing/timing_v3_exp023_jump20_runner_summary_v1.json`;
- durable inference freeze:
  `artifacts/reports/timing/timing_v3_exp023_jump20_inference_freeze_v1.jsonl`;
- post-freeze classification/audit:
  `artifacts/reports/timing/timing_v3_exp023_jump20_postfreeze_audit_v1.json`.

Schema requirements:

- `identity20` JSON is canonical JSON with exact root keys:
  `schema`, `experiment`, `source_exp022_shas`, `routing`, `identities`, and
  `identities_sha256`. `identities` contains exactly 20 records in ascending
  Exp022 `execution_index` order. Each record contains only
  `execution_index`, `ordinal`, `exp022_row_index`, `stratum`,
  `cache_key_sha256`, `source_output_sha256`, `source_freeze_sha256`, and
  `audio_path_sha256`.
- sanitized execution snapshot is canonical JSONL with exactly 20 lines, in the
  same order as `identity20.identities`. Each line has exact root keys
  `ordinal`, `cache_key_sha256`, `audio_path_sha256`, `resolved_audio_path`,
  `source`, and `label`. `source` has exact keys `cache_audio_key` and
  `cache_duration_seconds`. `label` is routing-only and has exact keys
  `stratum`, `confidence`, and `ambiguous`; the values must be the frozen
  identity/output values `jump_candidate`, `high|medium`, and `false`. These
  nested names are retained only because the existing runner validates that
  shape; the candidate generator never receives them. The snapshot contains no
  representative path, redline, beatmap, BPM truth, boundary truth, title,
  artist, map metadata, weak metric, or post-freeze audit field.
- source inventory JSONL is canonical JSONL with exactly 20 row records, one
  row per sanitized snapshot line. Each row contains only facts available at
  the retention hook: a source-only row header, compact `curve_table`, compact
  `occurrence_table`, the retained fingerprint set and retained order,
  retention delegate call count, and row source-inventory SHA. It must not
  contain eligible indices/fingerprints, selected index/fingerprint, lane,
  fallback, or any later inference fact.
- source summary JSON contains aggregate source-only counts, artifact byte
  sizes, row inventory SHAs, hook call counts, and all pre-oracle guard results.
- runner result JSONL and runner summary JSON are the unchanged runner outputs
  over the sanitized20 snapshot. Eligibility, selection, lane, and fallback
  are read only from this later runner result or its durable inference freeze;
  they are never backfilled into the pre-return source inventory. The runner
  outputs must not include added oracle fields.
- durable inference freeze JSONL contains exactly 20 rows, persisted before any
  weak-oracle load for the same row, with selected fingerprint, selected curve
  digest, candidate inventory digest/counts, seam report, and frozen inference
  SHA.
- post-freeze audit JSON is the only Exp023 artifact allowed to contain weak
  comparator-derived fields. It joins source inventory, runner result, durable
  freeze, and the sanctioned exact20-only post-freeze weak rows to produce the
  lifecycle classification.

Preflight must reject overwrites for every fixed path above, schema-version
mismatches, missing required keys, extra root keys in fixed-schema JSON
artifacts, non-canonical JSON, non-finite numbers, line-count mismatches, and
any path outside the fixed list.

## Sanctioned Row Routing

Exp023 must not derive real-row routing from pilot80 row scans, title/artist
metadata, sorted cache-key manifests, or row order guesses. The only authority
for the raw 20-row route is the frozen Exp022 identity artifact:

1. Verify all five Exp022 artifact SHAs listed in the dataset slice before
   opening row content.
2. Read only
   `artifacts/reports/timing/timing_v3_exp022_pilot42_identity_v1.json`.
   Its root must contain `identities` with exactly 42 records and must not
   contain a root-level `execution_index`. Each identity record must contain
   its own `execution_index`. Select exactly the 20 records whose `stratum` is
   the frozen Exp022 jump stratum `jump_candidate`; record only each selected
   record's `execution_index`, `ordinal`, and `cache_key_sha256` as the
   authority set. Do not use or trust the documented
   `sorted_cache_key_sha256_manifest` ordering field.
3. Join Exp022 output and durable freeze only by those 20
   `record.execution_index` values.
   For each joined row, assert exact agreement for `row_index`, `stratum`,
   `cache_key_sha256`, selected fingerprint, frozen inference SHA, and resolved
   audio path digest. A missing field, duplicate ordinal, mismatched cache SHA,
   mismatched selected fingerprint, mismatched frozen SHA, or mismatched audio
   path is a hard preflight failure.
4. The raw `cache_audio_key`, `cache_duration_seconds`, `resolved_audio_path`,
   `stratum`, and `confidence` used to build the sanitized20 snapshot may come
   only from the successfully joined Exp022 output row; `ambiguous=false` must
   come from the matching identity authority record. They may not be read from
   pilot80, the filesystem layout, map metadata, or any inferred convention.
5. Construct the sanitized20 snapshot from the joined Exp022 output rows and
   the identity authority set, ordered by each selected identity record's
   `execution_index`. The snapshot is the only input file passed to the runner
   for real execution.
6. Do not parse pilot80 stable row content before or during inference. If the
   post-freeze classifier needs representative beatmap or weak-comparator
   fields, it may stream
   `artifacts/reports/timing/timing_v3_pilot_rows_80_v1.jsonl` only after all
   source inventory, runner result, and durable freeze bytes are complete,
   hashed, and closed. That stream must filter by the exact 20 authorized
   `cache_key_sha256` values, discard every non-matching row immediately, never
   retain stable rows, and write only the matched 20 weak rows into the
   post-freeze audit join state. The lifecycle classifier may use no oracle
   rows except those exact 20 streamed matches.

Any alternative row-routing path requires a new accepted card.

## Baseline / Comparator

Baseline is the frozen Exp022 post-cap diagnostic:

- all 20 jump rows were `v3_accepted`;
- 17 selected jumps and 3 selected constants;
- selected jump exact hits remained `0/20`;
- among selected jumps, only three rows had nonzero fixed `+/-1000 ms`
  boundary recall;
- only `3/20` post-cap rows are currently known to have any fixed `+/-1000 ms`
  direct candidate;
- in the frozen jump20 order, row ordinals `0` and `1` have a correct direct
  candidate retained but production-ineligible;
- row ordinal `14` selected a correct direct candidate;
- the remaining `17` rows need pre-cap classification.

These known post-cap facts are diagnostic consistency checks only. The harness
must not branch inference on row ordinal, expected outcome, weak truth, or
target BPM/boundary values.

## Primary Metric

For each of the 20 jump rows, exactly one lifecycle classification. A
`direct candidate` is defined only in the post-freeze exact20-only audit: the
candidate must satisfy both the fixed `+/-1000 ms` boundary match requirement
and direct left/right BPM tolerance for each matched jump boundary. The BPM
tolerance is `max(1 BPM, 1%)` on both the left and right side of every matched
boundary. A candidate with a boundary match but a direct left/right BPM miss is
not direct. A candidate with direct BPM agreement but no fixed-window boundary
match is not direct.

- `not_generated`: no pre-cap proposal is a direct `+/-1000 ms` candidate.
- `generated_pruned`: at least one direct candidate exists pre-cap, but no
  direct candidate survives retention.
- `retained_ineligible`: at least one direct candidate survives retention, but
  no direct candidate is in `production_selection.eligible_candidate_indices`.
- `eligible_not_selected`: at least one direct candidate is eligible, but the
  selected fingerprint is not a direct candidate.
- `selected`: the selected fingerprint is a direct candidate.

Report the count in each bucket and the ordered 20-row classification table.

## Secondary Metric

For each row:

- row ordinal in the frozen Exp022 jump order;
- cache-key SHA-256 and source identity digest;
- selected Exp022 status, lane, source/family, fingerprint, and candidate
  index;
- number of unique pre-cap curve fingerprints and unique `(fingerprint,
  source)` records;
- number of direct candidate fingerprints pre-cap, retained, eligible, and
  selected;
- best direct candidate by pre-cap generation rank and by retained rank;
- generation source, generation score, source-only ranks, duplicate count,
  section count, boundary times, and section BPMs for matching direct
  candidates;
- fixed `+/-1000 ms` boundary precision/recall and direct left/right tempo-pair
  correctness for direct candidates, computed only after source freeze.

## Verify Command / Evaluation Procedure

No real-row execution command is authorized by this card until an independent
review approves the harness and run plan.

Before real execution, the temporary harness must support:

```sh
uv run --extra mps --group dev python /private/tmp/timing_v3_exp023_jump20_precap_inventory.py --self-check-only
```

The self-check must use synthetic in-memory rows only and cover:

- exact Exp022 SHA/identity guard rejection;
- row-count and stratum rejection unless exactly 20 jump rows are selected;
- fixed authoritative output path overwrite rejection for all Exp023 paths;
- fixed output schema rejection for missing keys, extra keys, line-count
  mismatches, non-canonical JSON, and non-finite numbers;
- Exp022 identity-to-output/freeze join rejection for row index, stratum, cache
  SHA, selected fingerprint, frozen SHA, and audio path mismatches;
- Exp022 identity shape rejection unless root `identities` has length 42, every
  record has `execution_index`, and no root-level `execution_index` exists;
- sanitized20 snapshot construction only from joined Exp022 output fields;
- rejection of any pilot80 parse before source inventory, runner result, and
  durable inference freeze are fully closed and hashed;
- forbidden stable cache/audio/mel access rejection;
- no holdout/broad/full path access;
- pre-cap hook placement after proposal construction;
- original retention delegate call count exactly one per synthetic row;
- returned retained batch identity unchanged;
- `try/finally` restoration after success and injected failure;
- source-only inventory write before any oracle loader call;
- pre-return source inventory contains pre-cap/retained facts only and rejects
  any eligible/selected/lane/fallback field; those later facts must appear only
  after the candidate generator returns and before post-freeze classification;
- post-freeze classifier bucket precedence;
- post-freeze lifecycle classification rejection unless every direct candidate
  satisfies both fixed `+/-1000 ms` boundary matching and direct left/right BPM
  tolerance `max(1 BPM, 1%)` using only exact20 streamed oracle rows;
- bounded deduplication by curve fingerprint and source;
- refusal to overwrite any authoritative output path.

Only after independent review and a passing self-check may the owner authorize
the real 20-row diagnostic run.

## Guard Check

Frozen source/input guards must match before real execution:

- `src/pulsefield_model/timing/v3/tempo_track.py` SHA-256
  `fc4153a6310a4db233e1fbd29e87a57775eff924e4904e925773d172e0d7de85`;
- `tests/timing/test_timing_v3_exp021_tempo_track.py` SHA-256
  `d543ec827a893fb1dddc3513edbe6df2396138ace5d96a58c679002dfeee3ae5`;
- `src/pulsefield_model/timing/evaluation/exp013_pilot.py` SHA-256
  `70891775233cf0c66b0d948689cf8a7d3505c57192ef0d5beed81f3f22f1b3bb`;
- `src/pulsefield_model/timing/evaluation/curve_metrics.py` SHA-256
  `1a70a9c0e8f965b9c7a9de74bc1c99711c95b938abda900e871f4fce0f316c2a`;
- `src/pulsefield_model/timing/evaluation/exp004_metrics.py` SHA-256
  `f88366562d35645fa4264cdb8710d2ea4781c6615a2ea23c2c3c87b0524dc21e`;
- `src/pulsefield_model/timing/v3/analytic_curve.py` SHA-256
  `766762aafdf0a3e643b3689c7ec659d14a0dd002befd2607a410fc368962c4d3`;
- pilot80 input
  `artifacts/reports/timing/timing_v3_pilot_rows_80_v1.jsonl`, SHA-256
  `cdb5e2af87d99c8af3bbff71a0985bb490fdeed169d10cb0a64b17a8ca0296d7`;
- paired v2 baseline
  `artifacts/reports/timing/timing_v3_v2_baseline_pilot80_v1.jsonl`,
  SHA-256
  `5d9bb3c50f4173b4bde60cdd1dd30a152565bd89934f2e6cb2407dd8910075a7`.

The real harness must snapshot the 20 jump rows' BeatThis cache, raw audio,
and mel-cache paths before and after execution. Existing files must be
byte-identical, missing mel paths must remain missing, and no cache generation
or network access is authorized.

## Bounded Storage Strategy

Do not serialize full duplicate proposal expansions. The frozen source-only
inventory must be compact and deterministic:

- one row-level header with cache-key SHA, source identity digest, source
  versions, proposal counts, retention counts, and SHA of the row inventory
  payload;
- one `curve_table` entry per unique canonical curve fingerprint, with compact
  source-only curve summary: curve class, section count, section beat spans,
  BPMs, boundary times, start/end time, and seam maximum;
- one `occurrence_table` entry per unique `(fingerprint, source)`, with
  duplicate count, best generation score, min/max/first generation ranks,
  source family, collapse BPM, and retained rank if retained;
- no weak labels, weak BPMs, weak boundaries, `.osu` paths, map metadata,
  titles, artists, or post-freeze evaluation metrics in the source inventory.

Hard storage guards:

- per-row unique `(fingerprint, source)` records must not exceed `40,000`;
- total frozen source inventory must not exceed `150 MiB`;
- total Exp023 artifacts, including the `/private/tmp` sanitized execution
  snapshot, must not exceed `200 MiB`;
- JSON serialization must be canonical: sorted keys, compact separators,
  finite numbers only.

If any guard is exceeded, freeze the partial source-only summary, classify the
run as invalid diagnostic, and do not continue to oracle classification.

## Qualitative Check

Inspect only the final post-freeze classification tables, not audio renders or
new `.osu` material. The qualitative check is whether the dominant bucket is
specific enough to motivate one next card: proposal generation, cap/retention,
eligibility, or arbitration.

## Positive Signal

The diagnostic is positive if it produces:

- exactly 20 row classifications;
- exact agreement with the known post-cap sanity facts for row ordinals `0`,
  `1`, and `14`;
- no source, cache, retention, inference, or oracle-order guard failure;
- a dominant lifecycle bucket or clear row-family split that can be tested in
  one subsequent Experiment Card.

## Negative Signal

The diagnostic is negative if the source-only inventory cannot be frozen before
oracle access, the hook cannot preserve the retention lifecycle unchanged, or
the classification is diffuse enough that no bounded next mechanism can be
selected without tuning on the same rows.

## Kill Criteria

Kill immediately if:

1. independent review has not approved real execution;
2. any frozen source/input/artifact SHA mismatches;
3. the resolved slice is not exactly 20 unique jump-candidate rows from Exp022;
4. Exp022 identity root lacks `identities` length 42, has a root-level
   `execution_index`, or any identity record lacks `execution_index`;
5. Exp022 identity `.identities[*].execution_index` is not the sole authority
   for ordering and joining the 20 ordinal/cache-SHA pairs;
6. Exp022 output/freeze joins fail any row-index, stratum, cache-SHA, selected
   fingerprint, frozen-SHA, or audio-path assertion;
7. sanitized20 snapshot fields are taken from pilot80, filesystem convention,
   or any source other than the joined Exp022 output rows;
8. any pilot80 row is parsed before source inventory, runner result, and
   durable inference freeze are complete and hashed;
9. any stable row is retained from a post-freeze pilot80 stream, any stable row
   is inferred, or any stable BeatThis/audio/mel cache is loaded;
10. holdout100-v2, structure-manifest6, broad500, full5050, ramp/dense rows, or
   any unapproved identity is opened;
11. weak truth, target BPMs, fixed-audit boundaries, row ordinal expectations,
   or map metadata are read before source-only inventory and inference freeze
   bytes are durable and hashed;
12. post-freeze lifecycle classification reads any oracle row outside the exact
    20 streamed matches;
13. any `direct candidate` classification omits either fixed `+/-1000 ms`
    boundary matching or direct left/right BPM tolerance `max(1 BPM, 1%)`;
14. the wrapper changes proposal order, scores, fingerprints, candidate count,
   retention output, eligibility, selection, fallback, or metrics;
15. the original retention delegate is not called exactly once per row with the
   original tuple/config;
16. any authoritative output path already exists or would be overwritten;
17. any authoritative output schema differs from the fixed schemas above;
18. any cache/audio/mel snapshot mutates;
19. total runtime exceeds `12 minutes`, any row exceeds `45 seconds`, or p90 row
    runtime exceeds `25 seconds`;
20. source inventory or total Exp023 artifacts exceed the storage guards;
21. known post-cap sanity checks fail: only three rows may have post-cap direct
    candidates, ordinals `0` and `1` must classify as `retained_ineligible`,
    and ordinal `14` must classify as `selected`.

## Expected Failure Modes

- Pre-cap candidate count is large enough that naive serialization breaches the
  disk guard.
- A hook placed too early misses raw-run proposals, or a hook placed too late
  cannot distinguish generated-pruned from retained-ineligible.
- Row identity order is confused by Exp022's documented cache-key ordering
  field-name ambiguity.
- Existing Exp022 artifacts contain enough selected-candidate information for
  post-cap checks but not enough weak truth for all pre-cap candidates, forcing
  careful post-freeze access to the already-exposed pilot80 comparator rows.
- Diagnostic runtime remains high because Exp022 source itself exceeded the
  product runtime budget.

## Confounders

- Exp022 is invalid for advancement due runtime; Exp023 may use its rows only
  for diagnostic localization, not quality claims.
- Direct `+/-1000 ms` matching is a weak comparator audit, not ground truth.
- A candidate can match fixed boundaries while still having poor full-song
  phase or BPM coverage.
- Row ordinals are diagnostic order labels; they must not become inference
  branches or tuning keys.
- A generated-pruned majority would identify retention pressure but not yet
  choose a new retention rule.

## Expected Runtime / Runtime Budget

Self-check runtime target: under `10 seconds` on CPU-only synthetic data.

Real diagnostic run, if later approved:

- expected total wall time: under `8 minutes` for 20 rows;
- hard total wall time: `12 minutes`;
- hard per-row wall time: `45 seconds`;
- hard p90 row time: `25 seconds`;
- runtime is diagnostic overhead and must not be reported as product runtime.

## Result Interpretation Plan

- Positive result would suggest: one lifecycle stage dominates the jump20
  boundary failures and can be isolated in the next card.
- Negative result would suggest: the current candidate lifecycle evidence is
  not enough to choose a single next mechanism without additional source-only
  instrumentation.
- Ambiguous result would require: a new planning card that narrows the row
  family or comparator definition without tuning thresholds.
- Human owner decides: whether the next loop targets generation, retention,
  eligibility, or arbitration.
- Next-loop action if positive: draft one mutation card against the dominant
  stage, with no broader data access.
- Next-loop action if negative: stop and improve source-only observability
  before any algorithm change.
- Next-loop action if ambiguous: split by source-family/topology only if the
  split is determined from source inventory and not weak outcome tuning.

## Result Log Template

- Experiment: Timing v3 Experiment 023
- Date:
- Harness path and SHA-256:
- Commit / source SHAs:
- Dataset slice identity SHA:
- Exp022 artifact SHAs verified:
- Exp022 identity root shape and identities count:
- Exp022 identity selected execution_index authority count:
- Exp022 output/freeze join assertions:
- Authoritative output paths:
- Authoritative output schema checks:
- Self-check command / result:
- Real execution approved by independent review: yes | no
- Row count / unique identities:
- Sanitized20 snapshot SHA-256:
- Pilot80 stable rows parsed before freeze:
- Pilot80 post-freeze stream matched / discarded counts:
- Post-freeze exact20-only oracle rows:
- Direct candidate boundary/BPM definition checks:
- Stable cache/audio/mel accesses observed:
- Holdout/broad/full accesses observed:
- Runtime total / p50 / p90 / max:
- Disk usage source inventory / all artifacts:
- Retention delegate call count:
- Rows with source-only inventory frozen before oracle:
- Rows with oracle accessed before freeze:
- Lifecycle classification counts:
- Row ordinal classification table:
- Known post-cap sanity checks:
- Direct candidates pre-cap / retained / eligible / selected:
- Dominant failure stage:
- Guard failures:
- Kill criteria triggered:
- Confounders:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Frozen Invalid Result Log

- Experiment: Timing v3 Experiment 023
- Date: 2026-08-14
- Status: completed / invalid; no lifecycle classification result.
- Harness path and SHA-256:
  `/private/tmp/timing_v3_exp023_jump20_precap_inventory.py`,
  `802405b7ec44061180bc94ab438a94838ee1c696c946b681075994593c7262c7`.
- Source guard: `8` source files, source-guard SHA-256
  `6af4c1b4339dbf4a81c5520c7f26d91e1d73b0070efab0bf46e2fec510695ad2`.
- Exp022 artifact SHAs verified by the identity artifact:
  identity `6b315460900d7a569c0e3523b0de0b4f1c902b398c93f1f7bf10763a06d1c4f6`,
  output `df3cb61bc8aec41284f27cf2c75d0281046d043216db83196617fbd06989e173`,
  summary `00e956ab4fdf66e5cca75cf359a910c40132675957b3906e6174fbf213afb057`,
  freeze `bbefbc76aad2469ab4af02c7ab4fe6fda805d370dc9d9cc48aea05eef9188988`,
  audit `2b2e8cada2768827e1d78254ebfabb3cde9689b0ca95662e40d4fcfce8424d77`.
- Authoritative Exp023 artifacts present and frozen:
  identity20
  `b465d2c9a7b4d5a1d39476b7163cf8ea2c1d6f9a803ed4c341d6e9bcd82e2743`;
  sanitized20 execution snapshot
  `c36c79327ae205efa14d02379faedad83e446ece34b1ebdf06adc294c0d3bd71`;
  source inventory
  `7d36defe0f43c050b5e34c1bf88ad49ef4f006035cd198ed67680fba5d2a99a7`;
  source summary
  `487e3bbe56e68105cdfb5fa09f2fa092ccd71f24f64a088f04e72d9eea95bc49`;
  runner JSONL
  `ba993e37e83ab41934a95ff66ecfdaf52cb58f35c2fc844b032ca37335cdd1a5`;
  runner summary
  `95ab261bf57363eae813099a82f957791378b6eab68046b147e9418977f72125`;
  inference freeze
  `873d35d51784883c71740482cd6236320fe4ffd4a857861f4187eac0caa8c62f`.
- Post-freeze classification/audit artifact:
  `artifacts/reports/timing/timing_v3_exp023_jump20_postfreeze_audit_v1.json`
  was not produced and is absent. No post-freeze oracle lifecycle result exists.
- Row counts: identity `20`, sanitized execution snapshot `20`, runner `20`,
  source inventory `3`, inference freeze `3`, post-freeze audit `0`.
- Runner status counts: `17` hard failures, `3` `v3_accepted`, `0`
  `v2_fallback`. The three accepted rows are ordinals `1`, `3`, and `4`;
  they are not interpretable as lifecycle evidence because the source/freeze
  set is incomplete and no post-freeze audit exists.
- Runtime total / p50 / p90 / max: `179.27615929199965 s` /
  `8.410253436988569 s` / `10.768631237183586 s` /
  `14.111760582978604 s`. Source-summary elapsed time was
  `180.46070041699568 s`.
- Disk usage: source inventory `140,133,509` bytes; artifact-size sum reported
  by source summary `140,454,342` bytes.
- Retention delegate call count: `20` entries, sum `20`.
- Source-instrumentation failure: `17` runner hard failures came from
  instrumentation storage guards, not from lifecycle classification:
  `11` rows exceeded the per-row occurrence-table guard and `6` rows hit
  `source inventory exceeds 150 MiB guard`.
- Partial-summary inconsistency: source summary reports `row_count=9` and a
  `row_inventory_sha256_order` length of `9`, while the persisted source
  inventory has only `3` JSONL rows and the durable inference freeze has only
  `3` rows. Treat this as an append-before-guard inconsistency in the
  instrumentation summary.
- Ordinal contract defect: the identity artifact independently verifies that
  Exp022 `execution_index=14` / cache key
  `6d03cb10fbcfb1372cb1e632828f271da436ada4776b610b3c9db05ca7e7a788`
  maps to Exp023 local ordinal `4`, not local ordinal `14`. The card's
  sanity statement that "row ordinal `14` selected a correct direct candidate"
  used an Exp022 execution-index label as if it were the Exp023 local ordinal.
  Do not use that local-ordinal sanity check for interpretation.
- Interpretation: invalid diagnostic harness run. It does not answer whether
  direct candidates are not generated, pruned, ineligible, eligible-not-
  selected, or selected across jump20.
- No-overwrite / no-reinterpret rule: preserve all existing Exp023 artifacts
  exactly as invalid evidence. Do not overwrite them, fill in the missing
  post-freeze audit at the same path, or reinterpret the `3` accepted rows as
  lifecycle classification evidence. Any repair requires a new accepted card
  and new artifact paths.

## Pre-Execution Gate

- Card complete: yes
- Harness implementation allowed by this card: yes
- Real execution allowed by this card alone: no
- Closed loop complete: yes
- Remaining ambiguity: exact temporary harness code must be independently
  reviewed before any real row is opened.

## Next-Loop Action

- If positive: create one separately accepted mutation card for the dominant
  lifecycle stage.
- If negative: create a source-only observability repair card or stop.
- If ambiguous: create a narrower diagnostic card over source-derived families
  only; do not tune on jump20 outcomes.

## Novelty Notes

- Closest analogies: survivor analysis in bounded search, compiler pass
  instrumentation, preregistered diagnostic audit.
- Novelty layer, if any: none.
- Representation novelty vs engineering variation: engineering variation only.
