# Timing v3 Experiment 024: Collapsed-counterfactual memoization

Status: planned / accepted for implementation, source-only verification,
synthetic before/after benchmarking, and frozen Exp021 mechanism2 replay only;
not accepted for pilot42, holdout, broader, or full-corpus execution

## Mode

- Mode: planner
- Route: TEST, behavior-preserving runtime optimization
- Source idea: score each within-row unique collapsed-curve fingerprint once,
  then project the unchanged score back to every original candidate occurrence.
- Acceptance source: the goal's single-variable experiment loop, the positive
  Exp021 mechanism result, and Exp022's frozen runtime diagnostic.
- Execution authority: this accepted card authorizes the package/test change,
  synthetic fixtures and benchmark, and exactly the two already-exposed Exp021
  mechanism rows after every source-only gate passes. It authorizes no other
  real-data access.

## Hypothesis

The current collapsed-counterfactual path repeatedly invokes the deterministic
independent raw-audio scorer for identical canonical constant curves within a
row. A row-local memo keyed only by the collapsed curve's existing
`fingerprint_sha256` can eliminate duplicate scoring work while reproducing the
baseline `CandidateRawAudioScore` for every original `candidate_index` and
therefore preserving every downstream rank, paired raw gain, eligibility, and
selection bit-for-bit.

This is a runtime hypothesis only. It does not explain or repair the six stable
false jumps in Exp022 and makes no quality or generalization claim.

## Root Objective

Reduce redundant collapsed raw-self scoring without changing any candidate,
curve, cap, evidence value, scorer formula, rank, paired raw gain, selector
input, selected result, fallback, or serialized candidate fingerprint.

## Goal Decomposition

- Subgoal 1: group the function's existing collapsed counterfactual tuple by
  canonical curve fingerprint in first-occurrence order.
- Subgoal 2: call the unchanged independent raw-audio scorer exactly once for
  each within-call unique fingerprint, still as a singleton candidate tuple.
- Subgoal 3: clone that score to every original occurrence while restoring the
  exact original `candidate_index` and input order.
- Subgoal 4: reconstruct the current ranking and top-level unavailable reason
  with the current ordering and fallback semantics unchanged.
- Subgoal 5: prove hard behavior equality against the frozen per-occurrence
  implementation on synthetic fixtures and on the two-row Exp021 mechanism
  replay.
- Subgoal 6: demonstrate a bounded runtime improvement before any real replay.

## Frozen Source and Evidence Boundary

The pre-mutation baseline is:

- `src/pulsefield_model/timing/v3/tempo_track.py` SHA-256
  `fc4153a6310a4db233e1fbd29e87a57775eff924e4904e925773d172e0d7de85`;
- `tests/timing/test_timing_v3_exp021_tempo_track.py` SHA-256
  `d543ec827a893fb1dddc3513edbe6df2396138ace5d96a58c679002dfeee3ae5`;
- frozen related source-only guard: `99 passed`.

The authoritative positive Exp021 mechanism artifacts are read-only:

- output
  `artifacts/reports/timing/timing_v3_exp021_mechanism2_authoritative_v1.jsonl`,
  SHA-256
  `18944665c5d91e4435abf1eddc65bc102c6b4748448eb854560bd0f7aee04178`;
- summary
  `artifacts/reports/timing/timing_v3_exp021_mechanism2_authoritative_summary_v1.json`,
  SHA-256
  `69d1ca49510ffb89af74f798ada82fe63c5c459d49d3840c4ebd2c5cee476f40`;
- durable pre-oracle freeze
  `artifacts/reports/timing/timing_v3_exp021_mechanism2_authoritative_freeze_v1.json`,
  SHA-256
  `67fb12b32f4dc6372bef91efc6b3a4a353707d8988966664255739359c5605d3`;
- post-freeze audit
  `artifacts/reports/timing/timing_v3_exp021_mechanism2_authoritative_audit_v1.json`,
  SHA-256
  `0d99998e9fcacc454089152b750ae937036f98ba9cb660728e5c30c0f96d605d`.

The following Exp022 artifacts are also read-only diagnostic evidence:

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
- durable freeze
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_freeze_v1.jsonl`,
  SHA-256
  `bbefbc76aad2469ab4af02c7ab4fe6fda805d370dc9d9cc48aea05eef9188988`;
- audit
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_audit_v1.json`,
  SHA-256
  `2b2e8cada2768827e1d78254ebfabb3cde9689b0ca95662e40d4fcfce8424d77`.

Exp022 recorded row-runtime p50 `6.9256 s`, p90
`11.724480616295475 s`, maximum `15.854203916009283 s`, and total
`296.56915908301016 s`. A read-only timing decomposition gives inference p90
`11.672594633419067 s` versus evaluation p90
`0.06838673369784373 s`; this localizes the budget problem to inference-side
work but does not prove which inference component dominates it.

As a separate read-only source/output diagnostic, summing within-row counts
over those 42 frozen rows gives `1,866` collapsed counterfactual occurrences
but only `206` within-row unique collapsed fingerprints, with `1..11` unique
fingerprints per row. These values motivate the test, but are not an
implementation constant, threshold, fixture requirement, data-selection rule,
or advancement gate. Exp024 must derive uniqueness solely from its current
function input.

## Candidate Variants

### Variant A: current per-occurrence singleton scoring

Invoke `score_raw_audio_evidence_independent(evidence, (counterfactual,))` for
every original collapsed counterfactual.

Decision: retain as the frozen behavior baseline and benchmark comparator.

### Variant B: row-local fingerprint memoization with exact projection

In first-occurrence order, score each unique collapsed fingerprint once as the
same singleton call used by Variant A. Store the returned score as an immutable
prototype. For every original tuple position, construct the same score record
with only `candidate_index` replaced by that original position, then run the
unchanged ranking construction over the fully projected tuple.

Decision: select.

### Variant C: batch all unique collapsed curves in one scorer call

Decision: reject. It changes scorer call shape and aggregation context in
addition to deduplication, making exact attribution and baseline equivalence
harder to audit.

### Variant D: persistent or cross-row score cache

Decision: reject. It introduces mutable lifetime, cache identity, eviction,
memory, and cross-row coupling that this experiment does not need.

## Local Verification Matrix

- Variant A is correct behaviorally but fails the duplicate-heavy synthetic
  efficiency target because scorer candidate inputs equal original
  occurrences.
- Variant B passes only if duplicate-heavy, all-unique, mixed-multiplicity,
  tied-score, all-unavailable, and empty-input fixtures are exactly equal to
  Variant A in every observable field and the duplicate-heavy scorer input
  count falls by at least half.
- Variant C fails isolation if the collapsed-counterfactual helper's dynamic
  scope makes any non-singleton scorer call or changes a candidate-level score,
  ranking-level unavailable reason, or ordering.  The unchanged outer
  raw-self call over the original candidate tuple is outside this scope.
- Variant D fails isolation if any state survives one function call, if a key
  contains row identity or filename, or if any disk/mel cache is read or
  written for memoization.

## Selected Variant

Execute only Variant B.

The memo is local to one
`_score_collapsed_counterfactuals_independent` invocation and is discarded on
return. Its key is exactly `counterfactual.fingerprint_sha256`. It may retain
the first curve object or first returned score only for the duration of the
call. It must not use candidate index, source family, BPM bins, duration bins,
row identity, filenames, weak labels, or oracle data in the key or branch.

The first scoring call for a fingerprint must remain:

```python
score_raw_audio_evidence_independent(evidence, (counterfactual,))
```

Do not batch unique curves. Do not change raw-audio scorer configuration or
math. Projection must preserve, for every original occurrence:

- `candidate_index` from the original tuple position;
- `fingerprint_sha256`;
- `raw_score`;
- `mean_beat_support`;
- `mean_half_beat_support`;
- `window_contrast_p10`;
- `retained_beat_count`;
- `complete_window_count`;
- `unavailable_reason`;
- `candidate_domain_beat_count`;
- `complete_window_start_beats`.

For counting and call-shape guards, the collapsed-helper dynamic scope starts
when `_score_collapsed_counterfactuals_independent` is entered and ends when
it returns or raises.  Only calls to
`score_raw_audio_evidence_independent` made while that scope is active count
toward this experiment's baseline occurrence total `N`, optimized input
total `U`, scorer-call count, singleton-call guard, or unique-fingerprint
count.  The existing outer full-generator call
`score_raw_audio_evidence_independent(audio_evidence, curves)` that produces
the ordinary raw-self ranking is unchanged, may remain non-singleton, and is
excluded from `N` and `U`.  It remains included in full-generator behavior
equality and wall time.

The projected `candidate_scores` tuple stays in original input order. The
existing ranked key stays exactly
`(-raw_score, fingerprint_sha256, candidate_index)`. The existing empty
`common_beat_indices`, empty ranking-level `complete_window_start_beats`, and
first projected unavailable-reason behavior stay unchanged.

No result-dump schema change is allowed. A provenance-only
`TEMPO_TRACK_VERSION` bump to
`pulsefield_model.timing_v3_tempo_track_exp024_v1` is required; update only the
superseded exact-version assertion in earlier tests. The version-string value
is the sole permitted `TempoTrackResult` behavior-comparison exclusion.
`TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION` must remain exactly
`pulsefield_model.timing_v3_tempo_track_result_dump_exp021_v1`; its existing
schema assertion must not change.

The closed behavior-comparison exclusion paths for that provenance-only change
are exactly:

- `diagnostics.version` when comparing a `TempoTrackResult` object;
- `tempo_track_version` when comparing the direct
  `tempo_track_result_to_dict` payload;
- `frozen_inference.tempo_track.tempo_track_version` when comparing an Exp024
  mechanism result row with its frozen Exp021 row.

No other diagnostics, schema, metadata, source-guard, artifact, or
version-bearing path may be excluded. Harness-only provenance records are
validated against explicit expected values outside the behavior comparator,
not hidden by additional exclusions.

## Selection Pressure

- Primary pressure: hard, zero-tolerance behavior equality.
- Secondary pressure: fewer collapsed scorer candidate inputs and lower full
  synthetic `generate_timing_candidates` runtime.
- Guard pressure: preserve the complete original candidate index space and
  deterministic ordering even when many occurrences share one fingerprint.
- Kill pressure: any quality-policy, candidate, evidence, cap, selector,
  fallback, or evaluator mutation ends the experiment.

## Research Question

Can within-row canonical collapsed-curve equivalence remove redundant raw
self-scoring work while leaving the complete Timing-v3 result behavior
bit-identical?

## Closest Analogies / Novelty Layer

- Closest analogies: common-subexpression elimination, hash-consing, and
  per-request memoization of a pure function.
- Relevant taxonomy bucket: behavior-preserving inference optimization.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation
  only; canonical curve fingerprints and scorer semantics already exist.

## Minimal Change

Change only `_score_collapsed_counterfactuals_independent` so identical
collapsed fingerprints are scored once per call and their immutable score
fields are projected back to original occurrences. Add focused equality,
call-count, and benchmark coverage. Do not refactor the scorer or surrounding
candidate-generation/selection pipeline.

## Files Likely to Change

Implementation and tests, after this card is accepted:

- `src/pulsefield_model/timing/v3/tempo_track.py`;
- new `tests/timing/test_timing_v3_exp024_tempo_track.py`;
- `tests/timing/test_timing_v3_exp021_tempo_track.py`, only if needed to replace
  its superseded exact-current-version assertion;
- temporary source-only benchmark and mechanism harnesses under
  `/private/tmp`;
- this card's result log after authorized execution.

No config, candidate, evidence, evaluator, schema, cache-provider, or selector
file may change.

## Read-Only Context Files

- `docs/research/timing_v3_experiment_021_pareto_short_aba_retention.md`
- `docs/research/timing_v3_experiment_022_pareto_retention_pilot42_replay.md`
- `docs/research/timing_v3_problem_log.md`
- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `tests/timing/test_timing_v3_exp014_tempo_track.py`
- the specifically named Exp021 and Exp022 artifacts and no broad artifact
  scan

## Dataset Slice

Stage 1 uses synthetic, deterministic in-memory curves and raw-audio evidence
only.

Stage 2, only after every Stage-1 gate passes, is exactly the two already
exposed Exp021 mechanism rows:

- stable: `dataset/0/2300685/audio.mp3`;
- short ABA: `dataset/0/618173/audio.mp3`.

No third real row is authorized. In particular, do not infer on or load
audio/BeatThis/mel data for Exp022 pilot42, its jump20 subset, pilot80,
structure-manifest6, repair80, holdout100-v2, broad500, full5050, ramp/dense
rows, or any additional identity. Exp022 artifacts may be read only for the
frozen aggregate/runtime facts listed above; they do not authorize replay of
their rows.

No network access, cache generation, mel regeneration, listening, or external
BPM lookup is authorized. Synthetic tests and the source-only benchmark may
read no `.osu` file.

During the authorized mechanism replay only, the representative `.osu` files
associated with the exact identities `2300685` and `618173` may be resolved
and read, and only after both conditions hold for the matching row:

1. the optimized inference payload is durably written and hashed; and
2. the post-freeze shadow baseline has completed and passed hard equality.

Every other `.osu` identity or path is forbidden.  The two permitted weak
reads are evaluation-only and may not affect inference, shadow scoring,
selection, or any frozen prediction field.

## Baseline / Comparator

The behavior comparator is a test/harness-local copy of the frozen
per-occurrence function from source SHA
`fc4153a6310a4db233e1fbd29e87a57775eff924e4904e925773d172e0d7de85`.
It must preserve the exact baseline singleton scoring call, score cloning,
ranking key, and unavailable-reason construction. Its text/digest must be
recorded in the benchmark report. It is verification-only and must never be
selected by production configuration.

The real mechanism comparator is the frozen Exp021 output plus a post-freeze
shadow call of that baseline function over the exact captured evidence and
collapsed tuple from each Exp024 row. The shadow result is audit-only and may
not feed production selection.

Exp021 anchors are:

- both rows `v3_accepted`, no fallback or hard failure, seam maximum `0.0 ms`;
- stable selects constant `200 BPM`, direct coverage `1.0`;
- `618173` selects fingerprint
  `e7e86f7c6828b3089c45b0dbef4e03ea70db5a9b8db3d4a70c7f8d924c58bfe9`
  from `paired_unmerged_boundary`;
- its exact curve is
  `175.1933640630715 -> 143.96419309635093 -> 175.1933640630715 BPM` at
  `54764.95117233091 ms` and `58932.65417608017 ms`;
- mechanism p90 runtime was `3.4802112580102404 s`, with stable/jump row
  runtimes `3.5332770830136724 s` and `3.0026188329793513 s`.

## Hard Behavior Equality Contract

Equality uses no numeric tolerance. A recursive audit encoder must preserve
ordered container type and order, encode every float by its IEEE-754 binary64
bytes (for example `struct.pack(">d", value).hex()`), and hash the resulting
canonical payload. Ordinary approximate comparisons are insufficient.

At function scope, baseline and optimized outputs must have:

- the same input `RawAudioEvidence` object;
- byte-identical values for every `CandidateRawAudioScore` field listed above;
- identical `candidate_scores` length and original-index order;
- identical `ranked_scores` membership and exact order;
- identical `common_beat_indices`, `complete_window_start_beats`, and
  ranking-level `unavailable_reason`.

At full-generator scope, baseline and optimized results must have identical:

- candidate count, order, curves, sections, diagnostics, sources, generation
  scores, cap reason, and all candidate fingerprints;
- raw and raw-self score fields and ranks;
- every collapsed score field captured by the verification wrapper;
- every `paired_raw_gain_by_candidate` index, raw-gain float, collapsed score,
  collapsed fingerprint, and tuple order;
- eligible indices, lane, raw-run diagnostic, fallback/status, selected
  candidate index, selected fingerprint, selected curve, and seams;
- serialized result payload fields other than the exact provenance version
  value declared above and harness-only timing/run-identity fields.

The harness must use a closed explicit exclusion-path allowlist. Wildcard
exclusions such as all diagnostics, all metadata, or all floats are forbidden.
Any newly differing unlisted path is a hard failure.

## Primary Metric

Hard behavior equality passes on every source-only fixture, every benchmark
repetition, and both mechanism rows.

For the duplicate-heavy synthetic full-generator benchmark, let:

- `N_i` be the candidate-input count across singleton independent scorer
  calls made inside measured pair `i`'s frozen baseline collapsed-helper
  dynamic scope;
- `U_i` be the corresponding count inside that pair's optimized
  collapsed-helper dynamic scope;
- `N = sum(N_i)` and `U = sum(U_i)` over measured pairs.

For every measured pair, `N_i` must equal its original collapsed occurrence
count and `U_i` must equal its within-call unique collapsed fingerprint
count. Across measured repetitions, the gate is
`U == sum(per_invocation_unique_fingerprint_count)` and
`U <= floor(N / 2)`. Scorer-call count and scorer candidate-input count are
reported separately even though every in-scope call must be singleton. The
unchanged outer raw-self batch call is reported, if desired, only as an
out-of-scope diagnostic and is never added to `N` or `U`. The implementation
must not special-case the fixture's counts.

## Secondary Metric

On the same duplicate-heavy synthetic fixture, speed attribution comes only
from same-process alternating paired baseline/optimized measurements over the
same prebuilt evidence and counterfactual objects.  For measured pair `i`,
define positive improvement deltas as:

- `collapsed_delta_i = baseline_collapsed_i - optimized_collapsed_i`;
- `full_delta_i = baseline_full_i - optimized_full_i`.

The performance gate requires:

- median `collapsed_delta_i > 0`;
- median `full_delta_i > 0`;
- optimized full-generator time wins a strict majority of measured pairs;
- optimized summed full-generator wall time is strictly below baseline.

Raw baseline/optimized p50/p90/max and ratios remain reported, but unpaired
comparison with the frozen Exp021 or Exp022 run is not speed attribution.
Per-pair AB/BA order, both deltas, collapsed/full win counts, absolute seconds,
relative ratios, scorer call count, scorer candidate-input count, original
occurrence count, and unique fingerprint count are all reported.

All-unique input is a guard, not the optimization target: it must preserve
behavior and report overhead separately. Production candidate count must never
be described as reduced; only redundant scorer inputs may decrease.

## Synthetic Test and Benchmark Matrix

1. Duplicate-heavy full-generator fixture
   - Use deterministic in-memory 10-minute audio evidence and source-only
     candidate generation.
   - Ensure at least two distinct collapsed fingerprints, each repeated at
     non-contiguous original indices, and an overall unique ratio at most
     one-half.
   - Do not use the Exp022 values `1,866` or `206` as fixture constants.
2. Mixed multiplicity and tied ranks
   - Reuse fingerprints with multiplicities one, two, and greater than two.
   - Force equal available raw scores across distinct fingerprints and assert
     the existing fingerprint/index tie order exactly.
3. Field-complete score projection
   - Inject a deterministic counting scorer whose prototypes exercise every
     `CandidateRawAudioScore` field and distinct complete-window tuples.
   - Assert each duplicate receives all prototype bits unchanged and only its
     original `candidate_index` differs.
4. All unavailable
   - Exercise at least two unavailable reasons and duplicated first
     occurrences; assert empty ranked scores and exact baseline top-level
     unavailable reason.
5. All unique
   - Assert optimized scorer inputs equal baseline, call order follows original
     order, and full behavior remains identical.
6. Empty input
   - Assert zero scorer calls and exact baseline empty ranking behavior.
7. Permutation and determinism
   - Repeat nontrivial input permutations and identical replays; projection
     follows each original order while deterministic content/rank rules remain
     exact.
8. Full downstream selection
   - Run constant, accepted paired-jump, fallback, and raw-unavailable
     synthetic cases; assert all raw gains, ranks, eligibility, selection,
     fingerprints, seams, and result-dump behavior fields are equal.

## Verify Command / Evaluation Procedure

Card creation runs none of these commands. After implementation, execute in
this order and stop at the first failure.

1. Focused Exp024 tests:

```sh
uv run --extra mps --group dev pytest -q tests/timing/test_timing_v3_exp024_tempo_track.py
```

2. Re-run the frozen related suite, with only the superseded version assertion
   updated, and require the original `99 passed` tests plus all new Exp024
   tests to pass:

```sh
uv run --extra mps --group dev pytest -q \
  tests/timing/test_timing_v3_analytic_curve.py \
  tests/timing/test_timing_v3_audio_evidence.py \
  tests/timing/test_timing_v3_tempo_track.py \
  tests/timing/test_timing_v3_exp014_tempo_track.py \
  tests/timing/test_timing_v3_exp017_tempo_track.py \
  tests/timing/test_timing_v3_exp018_tempo_track.py \
  tests/timing/test_timing_v3_exp019_tempo_track.py \
  tests/timing/test_timing_v3_exp021_tempo_track.py \
  tests/timing/test_timing_v3_exp024_tempo_track.py
```

3. Run the source-only benchmark, bounded to ten minutes total:

```sh
uv run --extra mps --group dev python /private/tmp/timing_v3_exp024_collapsed_counterfactual_benchmark.py --max-seconds 600
```

The benchmark must build evidence before timing, warm both paths equally, use
at least five measured paired repetitions, alternate `AB` and `BA` order, use
the same evidence/counterfactual objects, restore every monkeypatch in
`try/finally`, and report each pair rather than only an aggregate.  It must
report positive-is-better collapsed/full paired deltas and win counts and
apply the strict-majority rule to the full-generator pairs. The hard behavior
digest is checked after every paired repetition. The entire command, including
warmup, must finish in `<600 s`.

4. Only if steps 1-3 pass, run the mechanism harness self-check on synthetic
   rows:

```sh
uv run --extra mps --group dev python /private/tmp/timing_v3_exp024_mechanism2_replay.py --self-check-only
```

5. Only if the self-check passes and authoritative Exp024 paths do not exist,
   run exactly the two authorized mechanism rows:

```sh
uv run --extra mps --group dev python /private/tmp/timing_v3_exp024_mechanism2_replay.py --run-authoritative
```

No command for pilot42, jump20, holdout, broad500, or full5050 is authorized.

## Guard Check

Before implementation, verify the frozen source/test/artifact hashes listed in
this card. After implementation, record the new source/test/harness hashes and
refuse execution if they change between synthetic verification and mechanism
replay.

The mechanism harness self-check must prove:

- exact two-row identity and order association;
- rejection of any third or substituted identity;
- source/artifact SHA mismatch rejection;
- output no-overwrite behavior;
- no weak oracle read before the authoritative inference payload is durable
  and hashed;
- post-freeze shadow-baseline audit only, with no result fed back to inference;
- exact behavior encoder and closed exclusion-path allowlist;
- counting-wrapper accuracy and restoration on success/injected failure,
  including a sentinel proof that the unchanged outer non-singleton raw-self
  batch is excluded while every collapsed-helper singleton is counted;
- exact aggregate denominators;
- cache/audio/mel snapshot comparison on success/injected failure;
- no network or cache-generation path.

For the real replay, snapshot the two BeatThis cache files, raw audio files,
and mel paths before and after. Existing resources must be byte-identical;
missing mel paths must remain missing. Do not create, refresh, touch, or delete
any cache.

The authoritative Exp024 paths are new and fixed:

- result
  `artifacts/reports/timing/timing_v3_exp024_mechanism2_authoritative_v1.jsonl`;
- runner summary
  `artifacts/reports/timing/timing_v3_exp024_mechanism2_authoritative_summary_v1.json`;
- durable pre-oracle freeze
  `artifacts/reports/timing/timing_v3_exp024_mechanism2_authoritative_freeze_v1.json`;
- post-freeze equality/evaluation audit
  `artifacts/reports/timing/timing_v3_exp024_mechanism2_authoritative_audit_v1.json`.

Refuse if any path exists. Never overwrite or repair an authoritative artifact
in place.

## Mechanism2 Replay Gate

The optimized production call must be timed without the shadow baseline audit.
After its inference payload is durably frozen, run the audit-only baseline over
the captured evidence/counterfactual tuple and check hard equality before any
weak comparator read.

Both rows must:

- be `v3_accepted`, with zero fallback and hard failure;
- preserve candidate count, cap reason, candidate order, every source and
  fingerprint, all score/rank/raw-gain fields, eligibility, selection, and
  maximum seam `0.0 ms` exactly;
- match the frozen Exp021 behavior payload under the closed exclusion allowlist;
- match the same-run shadow baseline's complete collapsed ranking bit-for-bit;
- preserve cache/audio/mel snapshots.

Stable `2300685` must remain the exact selected constant `200 BPM` result.
Short-ABA `618173` must retain and select the exact Exp021 fingerprint, source,
three BPMs, and two boundaries listed above. Its fixed `+/-1000 ms`
precision/recall must remain `1.0/1.0`; the unchanged strict metrics must also
be reported without reinterpretation.

Runtime must report optimized-only row p50/p90/max and total, shadow-audit time
separately, and scorer occurrence/unique counts separately. The original
mechanism p90 `3.4802112580102404 s` and the preregistered `3.85 s` margin
are report-only historical references, not speed-attribution or acceptance
gates. Hard kill only if either optimized row reaches `>=5.0 s`. Runtime is
not part of bit equality; all claims that Exp024 is faster must come from the
same-process alternating paired synthetic benchmark.

If this gate passes, freeze the result and stop. Pilot42 still requires a new
accepted card.

## Qualitative Check

No listening or visual/audio inspection is needed. Inspect only the synthetic
benchmark report and exact equality diff. A successful report should show the
same original-index behavior payload with fewer collapsed scorer inputs and a
lower full-generator runtime.

## Positive Signal

Positive only if all of the following hold:

- all source-only tests pass, including the frozen 99-test guard;
- every baseline/optimized behavior digest matches;
- duplicate-heavy synthetic scorer inputs fall by at least half;
- paired collapsed and full-generator median deltas are positive, optimized
  full-generator time wins a strict majority of pairs, and summed optimized
  full-generator runtime improves within the ten-minute benchmark;
- mechanism2 preserves both Exp021 outcomes and every hard behavior field;
- both optimized mechanism rows remain `<5.0 s`; the `3.85 s` p90 reference
  is reported without gating, and cache/integrity/oracle-order guards pass.

## Negative Signal

Negative if hard equality holds but the synthetic scorer-input or same-process
paired runtime gate does not improve. Freeze that result and do not spend
real-data budget on the mechanism replay.

Any behavior difference is a hard failure, not an acceptable speed/quality
trade-off. Do not tune candidate, evidence, rank, or selector behavior to
recover it under Exp024.

## Kill Criteria

Kill immediately if:

1. the baseline source/test/artifact identity does not match this card;
2. any change extends beyond row-local collapsed-score memoization,
   provenance, and focused tests/harnesses;
3. candidate generation, search breadth, candidate count/order, curve,
   fingerprint, source, quota, cap, or cap reason changes;
4. evidence extraction/config, scorer formula/config, selector,
   compatibility, eligibility, fallback, evaluator, metric, or result schema
   changes;
5. the optimized collapsed-helper dynamic scope makes a non-singleton scorer
   call, any in-scope call is omitted from `N/U`, the unchanged outer
   raw-self batch is included in `N/U`, or a cache survives the function
   invocation;
6. any score field, ordering field, rank, raw gain, selected value, seam, or
   behavior digest differs from baseline;
7. duplicate-heavy optimized scorer inputs exceed `floor(N / 2)`;
8. synthetic paired collapsed/full median delta is not positive, optimized
   full-generator time does not win a strict majority of measured pairs, or
   summed optimized full-generator runtime does not improve;
9. the source-only benchmark reaches `600 s`;
10. synthetic/source guards fail before real replay;
11. any real identity other than `2300685` or `618173` is resolved, opened, or
    inferred;
12. any pilot42/jump20, structure, holdout, repair, broad, full, ramp/dense, or
    additional real audio/cache/mel data is accessed;
13. either permitted representative weak file is read before its matching
    durable optimized inference freeze and successful shadow-baseline
    equality, any other `.osu` is resolved or read, or weak truth enters any
    inference branch;
14. an authoritative path already exists or any artifact would be overwritten;
15. any source, cache, audio, or mel snapshot mutates;
16. either mechanism behavior differs from Exp021 or either optimized row
    reaches `>=5.0 s`; mechanism p90 relative to `3.85 s` is report-only.

## Expected Failure Modes

- Memoizing the returned score object without cloning original indices can
  collapse candidate identity or create duplicate index zero.
- Rebuilding scores incompletely can omit domain/window fields while leaving
  headline `raw_score` unchanged.
- Ranking only unique prototypes can remove duplicate candidate occurrences
  or change tie order.
- Grouping by rounded BPM or curve object identity can miss canonical
  duplicates or merge nonidentical curves.
- Batching unique curves can accidentally change scorer aggregation semantics.
- Benchmark setup/evidence extraction can dominate timing and hide the target
  path unless it is performed outside the timed region.
- A version or harness runtime field can create an expected artifact diff; the
  closed allowlist must isolate it without masking any behavior field.

## Confounders

- Exp022 failed runtime integrity and is diagnostic only; its `1,866/206`
  observation predicts opportunity but cannot serve as a generalization gate.
- Exp022's total row runtime includes other inference work, so synthetic
  before/after timing is required to attribute an improvement to this change.
- Warm caches, GC, MPS scheduling, and AB order can bias small timing samples;
  paired alternating repetitions and full per-repeat reporting are required.
- Identical fingerprints are assumed to identify identical canonical curves;
  that existing fingerprint contract is not changed or revalidated here.
- Mechanism2 is only a behavior regression gate, not evidence of pilot-level
  speed or quality.

## Expected Runtime / Runtime Budget

- Focused/source-only tests: expected under `3 minutes` total.
- Synthetic benchmark: hard wall time `<10 minutes`, with no real data.
- Mechanism self-check: expected under `15 seconds`.
- Authorized two-row replay: expected optimized inference under `10 seconds`
  total; hard per-row limit `<5.0 seconds`; shadow audit reported separately.
- No pilot42 or larger runtime budget exists under this card.

## Result Interpretation Plan

- Positive result would suggest: retain Exp024 as a transparent runtime
  optimization while preserving Exp021's algorithmic identity; any broader
  replay still needs its own accepted card.
- Negative performance with exact behavior would suggest: remove/decline the
  memoization and pursue a separately carded inference bottleneck.
- A mechanism p90 above the report-only `3.85 s` reference but with both rows
  below `5.0 s` does not reverse a positive same-process paired synthetic
  result.
- Behavior mismatch would suggest: the scorer is not pure over canonical
  collapsed curve identity or projection is incorrect; stop and diagnose on
  synthetic inputs only.
- Ambiguous timing would require: repeat or improve only the source-only
  benchmark methodology under a new card, without opening real rows.
- Human owner decides: whether a later card may reopen an exposed pilot after
  Exp024 passes; Exp024 itself cannot do so.

## Result Log Template

- Experiment: Timing v3 Experiment 024
- Date:
- Baseline source/test/artifact SHAs verified:
- Implemented source/test/harness SHAs:
- Changed files:
- Frozen 99-test guard result:
- New Exp024 test result:
- Benchmark baseline-helper digest:
- Benchmark hardware/software identity:
- Benchmark fixture identity/digest:
- Benchmark warmups / repetitions / AB order:
- Baseline/optimized behavior digests per repetition:
- In-scope baseline/optimized scorer calls and candidate inputs:
- Outer raw-self batch exclusion sentinel:
- Original collapsed occurrences `N` / optimized scorer inputs `U` /
  per-invocation unique counts / reduction:
- Collapsed paired deltas / baseline wins / optimized wins / ties:
- Full-generator paired deltas / baseline wins / optimized wins / ties:
- Collapsed-scoring baseline/optimized p50/p90/max and ratio:
- Full-generator baseline/optimized p50/p90/max, summed time, and ratios:
- All-unique overhead:
- Benchmark total wall time:
- Mechanism self-check result:
- Real replay reached: yes | no
- Authorized row identities/count:
- Pre-oracle freeze status:
- Shadow-baseline equality digests:
- Exp021 behavior equality diff/exclusion paths:
- Stable selected result/fingerprint/runtime:
- Short-ABA selected result/fingerprint/runtime:
- Mechanism p50/p90/max/total optimized runtime:
- Mechanism p90 delta from `3.4802112580102404 s` and `3.85 s`
  report-only references:
- Shadow audit runtime:
- Scorer occurrence/unique counts by mechanism row:
- Permitted representative weak paths / post-freeze and post-shadow read order:
- Seam/fallback/hard-failure totals:
- Cache/audio/mel snapshot result:
- Unauthorized data accesses:
- Artifact paths/SHAs:
- Guard failures / kill criteria:
- Classification:
- Next step:

## Pre-Execution Gate

- Card complete: yes
- Implementation allowed: yes
- Synthetic/unit execution allowed: yes
- Synthetic before/after benchmark allowed: yes
- Exp021 mechanism2 replay allowed after source-only PASS: yes
- Exp022 pilot42 or jump20 replay allowed: no
- Holdout, repair80, structure, broad500, or full5050 allowed: no
- Closed loop complete: yes
- Remaining ambiguity: none requiring real-data expansion; benchmark and
  mechanism results must be recorded after execution.

## Next-Loop Action

- If positive: freeze the optimization and stop. Draft a separate accepted
  card before any exposed pilot or broader evaluation.
- If performance-negative but behavior-equal: freeze the negative and remove
  or decline the optimization; do not run mechanism2.
- If behavior-negative: diagnose and repair only on synthetic fixtures under a
  new card.
- If integrity-invalid: preserve artifacts for audit, repair the harness under
  a new card, and do not reuse the invalid result for advancement.

## Novelty Notes

- Closest analogies: per-request memoization and common-subexpression
  elimination.
- Novelty layer, if any: none.
- Representation novelty vs engineering variation: engineering variation.
