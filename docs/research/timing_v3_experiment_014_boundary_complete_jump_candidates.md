# Timing v3 Experiment 014: Boundary-complete jump candidates with paired raw gain

Status: negative at authoritative post-cleanup mechanism gate; pilot42 and protected layers sealed

## Mode

- Mode: planner
- Route: TEST
- Source idea: Experiment 013 recovered the exposed short-ABA mechanism row, but
  failed the formal already-exposed pilot42 because most rows with raw tempo
  runs had no compatible short-ABA proposal.  Post-result inspection of the
  already-exposed jump rows shows that the weak comparator structures include
  single persistent jumps, long ABA sections, and multi-step piecewise-constant
  changes, not only 2--8 s ABA excursions.
- Acceptance source, if any: goal objective; Timing v3 task definition;
  Experiment 010, 011, 012, and 013 negative results; TV3-041 through TV3-045.
- Source snapshot / evidence grade: strong local evidence from Exp013 source
  gates, the two already-exposed mechanism rows, and the formal pilot42 result.
  All pilot42 evidence is already exposed development evidence.  Protected
  holdout100-v2, broad500, and full5050 remain sealed.

## Hypothesis

Timing v3 can improve jump recall without threshold tuning if candidate
generation covers the main Phase-1 piecewise-constant structures and selection
uses a paired, candidate-local raw-audio gain test.  A nonconstant curve should
challenge the best constant only when its own raw self-score is strictly higher
than a full-duration collapsed constant comparator for the same base tempo.

## Root Objective

Repair the Exp013 failure mode by separating two issues:

1. candidate presence for single, long-ABA, and small multi-step jumps; and
2. stable specificity when raw local tempo observations fire on texture or
   alias artifacts.

The product contract is unchanged: accepted v3 output is a phase-continuous
constant/jump grid over one absolute beat axis, while insufficient or invalid
evidence returns explicit `v2_fallback`.  Ramp production remains excluded.

## Goal Decomposition

- Subgoal 1: preserve the existing constant path as an always-competing safety
  candidate.
- Subgoal 2: generate bounded Phase-1 jump candidates for:
  - single persistent `A -> B`;
  - short or long `A -> B -> A` ABA, from 2 s through 60 s;
  - at most four-section piecewise-constant multi-step paths.
- Subgoal 3: keep one global absolute beat axis per candidate, with integer
  section boundaries and exact phase continuity.  Candidate alternatives may
  have different terminal `end_beat` values because a persistent tempo change
  alters the number of beats needed to cover the audio.
- Subgoal 4: score every curve on its own physical beat coverage, then compare
  each nonconstant only to its own full-duration collapsed constant comparator.
- Subgoal 5: prevent raw-run threshold tuning on pilot42.  The production
  challenge rule is strict paired gain sign, not an arbitrary positive margin.
- Subgoal 6: repair measurement reporting needed for this experiment without
  changing inference behavior.

## Candidate Variants

- Variant A: tune Exp013 raw-run thresholds or overlap gates.
- Variant B: keep Exp013 short-ABA candidates, but let raw runs generate the
  missing short boundaries directly.
- Variant C: selected boundary-complete bounded candidate family.  Use the top
  raw local tempo runs as source-only proposals for single persistent jumps,
  2--60 s ABA, and up to four-section multi-step paths.  BeatThis may provide
  boundary/local support ranks and missing local corroboration, but it does not
  decide that a raw run exists.
- Variant D: replace the generator with a full low-rate Viterbi/DP over local
  tempo observations.

## Local Verification Matrix

- Variant A:
  - Check: review Exp013 pilot42 failure attribution.
  - Fail condition: it changes thresholds on exposed pilot42 rather than
    solving candidate absence.
- Variant B:
  - Check: compare against the already-exposed jump structure audit.
  - Fail condition: it still cannot represent single persistent, long-ABA, or
    multi-step rows.
- Variant C:
  - Synthetic checks:
    1. stable direct 200 BPM;
    2. stable half/double or 230.77 BPM fill artifact;
    3. short `175 -> 143 -> 175` ABA;
    4. long 30 s ABA;
    5. persistent down jump;
    6. persistent up jump;
    7. progressive three-stage jump;
    8. false raw-run stable texture change;
    9. weak evidence / raw unavailable fallback;
    10. candidate cap and candidate-domain coverage.
  - Real mechanism checks, only after synthetic passes:
    1. exposed stable mechanism row `dataset/0/2300685/audio.mp3`;
    2. exposed short-ABA mechanism row `dataset/0/618173/audio.mp3`;
    3. fixed already-exposed structure-manifest subset:
       - stable false-jump regression:
         `dataset/0/813270/audio.mp3`;
       - persistent down:
         `dataset/0/882486/audio.mp3`;
       - persistent up:
         `dataset/0/440089/192.MP3`;
       - long ABA:
         `dataset/0/1113833/audio.mp3`;
       - progressive down:
         `dataset/0/2080593/audio.mp3`;
       - multi-step jump:
         `dataset/0/863309/audio110.mp3`.
  - Pilot check: the same already-exposed pilot42 only if all earlier gates
    pass.
- Variant D:
  - Check: implementation scope and observability review.
  - Fail condition: it changes generator family, scoring objective, and search
    semantics at once, making a negative result hard to attribute.

## Selected Variant

- Selected: Variant C.
- Rejected:
  - Variant A is label-tuning on exposed data and is prohibited.
  - Variant B is too narrow; the exposed jump row structures are not mostly
    short ABA.
  - Variant D is a plausible later family but too broad for the next smallest
    defensible test.
- Why this is the smallest useful test: it changes candidate presence and the
  class-comparison rule while preserving raw feature extraction, weak-evidence
  separation, fallback contract, Phase-1 constant/jump output, and protected
  data boundaries.

## Selection Pressure

- Primary pressure: recover more exposed jump rows by making structurally
  compatible candidates present before ranking.
- Specificity pressure: prevent stable false jumps by requiring a nonconstant
  curve to beat its own full-duration collapsed constant comparator under the
  same raw-audio scorer.
- Guard pressure: no accepted curve may violate a single absolute beat axis,
  integer boundaries, phase continuity, seam continuity, candidate cap,
  inference/evaluation separation, or ramp exclusion.
- Runtime pressure: keep candidate count at or below 64 and p90 row runtime on
  pilot42 below five seconds with existing feature caches.
- Kill pressure: fail fast at synthetic or mechanism stage if paired raw gain
  does not reject stable false-run fixtures or does not admit representative
  persistent/long/multi-step jump candidates.

## Research Question

Does a bounded boundary-complete constant/jump candidate family plus strict
paired raw-gain selection outperform the Exp013 short-ABA raw-run selector on
already-exposed pilot42 without opening protected data?

## Closest Analogies / Novelty Layer

- Closest analogies: change-point candidate enumeration, source-separated
  model selection, paired likelihood-ratio/sign tests, sparse piecewise-constant
  tempo tracking, and bounded beam search.
- Relevant taxonomy bucket: deterministic inference and evaluation workflow,
  not model training.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: the representation remains
  the Timing v3 analytic constant/jump section graph.  This is an engineering
  mutation of candidate generation and selection.

## Minimal Change

Implement Exp014 behind the existing disabled-by-default Timing v3 research
path:

1. Reserve at least one constant candidate and keep constants production
   eligible whenever raw evidence is scoreable.
2. Allow candidate alternatives to have candidate-local terminal beat domains.
   All sections within one candidate still use one absolute beat axis and
   integer contiguous section intervals.
3. From source-only raw local tempo observations, retain the top `K=4`
   deviation runs by the Exp013 dominant-run ordering.  Existing Exp013
   canonicalization, smoothing, and run-detection parameters are inherited and
   not tuned in this card.
4. For each retained raw run and base hypothesis, generate bounded jump
   families:
   - persistent jump at the run start;
   - persistent jump at the run end;
   - ABA spanning the run, with valid middle duration from 2 s through 60 s;
   - adjacent-run chains up to four constant sections when two or three
     retained runs imply ordered local tempo states.
5. Snap section boundaries to the nearest integer beat on the evolving
   candidate curve; never reset phase at a boundary.
6. Use BeatThis local/boundary support only as a secondary ordinal rank or to
   fill a missing local support diagnostic.  BeatThis is not the run-presence
   authority and BeatThis chunk seams are not section boundaries.
7. For every nonconstant candidate, construct a collapsed full-duration
   constant comparator with the same origin and base tempo.  Its `end_beat` is
   chosen independently to cover the audio; it is not forced to match the
   nonconstant's terminal beat count.
8. Compute `raw_self_score(candidate)` and
   `raw_self_score(collapsed_constant)` over each curve's own retained physical
   beat coverage.  The existing raw retained/domain guard still applies:
   `>= 0.90` and at least `16` retained beats.
9. A nonconstant is allowed to challenge constants only when:
   `raw_self_score(nonconstant) > raw_self_score(collapsed_constant)`.
   There is no positive margin threshold.
10. Among positive-gain structures, rank by:
    1. ordinal raw paired-gain rank;
    2. generalized BeatThis support rank;
    3. lower section count;
    4. higher raw self-score;
    5. canonical fingerprint.
11. Compare the best positive-gain structure to the best constant by the paired
    gain sign.  If no structure has positive paired gain, select the best
    constant rather than falling back solely because a raw run existed.
12. Return `v2_fallback` only for raw-score unavailability, no valid constant,
    candidate-cap/integrity failure, or explicit schema/continuity failure.
13. Keep diagnostic ramps excluded from production.  Ramp rows can be reported
    only as audit diagnostics with production ramp accuracy `null`.

Measurement-only prerequisite before formal pilot42:

- The runner must report candidate-local domain coverage, collapsed-comparator
  score and fingerprint, paired gain sign, absolute endpoint drift magnitude,
  candidate-cap reason, and full fallback reason.  These fields are diagnostic
  only and cannot affect the selected prediction.

## Files Likely to Change

- `src/pulsefield_model/timing/v3/tempo_track.py`
- `src/pulsefield_model/timing/evaluation/exp013_pilot.py` or a new Exp014
  evaluation runner if keeping Exp013 immutable is cleaner
- `tests/timing/test_timing_v3_tempo_track.py`
- `tests/timing/test_timing_v3_exp013_pilot.py` or a new Exp014 runner test
- `docs/research/timing_v3_problem_log.md`
- this Experiment Card, with result log appended after execution

Read-only unless a conformance bug is found:

- `src/pulsefield_model/timing/v3/analytic_curve.py`
- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `src/pulsefield_model/timing/evaluation/curve_metrics.py`

## Read-Only Context Files

- `/Users/l/.codex/attachments/97bd173a-3590-4524-8b1f-f7a90c5e0223/goal-objective.md`
- `AGENTS.md`
- `README.md`
- `docs/research/timing_v3_task_definition.md`
- `docs/research/timing_v3_problem_log.md`
- `docs/research/timing_v3_experiment_010_real_audio_short_jump.md`
- `docs/research/timing_v3_experiment_011_candidate_local_audio_gain.md`
- `docs/research/timing_v3_experiment_012_ordinal_production_selector.md`
- `docs/research/timing_v3_experiment_013_raw_run_ordinal_selector.md`
- `src/pulsefield_model/timing/v3/analytic_curve.py`
- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- `src/pulsefield_model/timing/evaluation/curve_metrics.py`
- `src/pulsefield_model/timing/evaluation/exp013_pilot.py`
- relevant Timing v3 tests under `tests/timing/`

## Dataset Slice

Execution order is fixed:

1. source-owned unit/synthetic fixtures only;
2. already exposed mechanism rows:
   - `dataset/0/2300685/audio.mp3`;
   - `dataset/0/618173/audio.mp3`;
3. already exposed fixed structure-manifest subset:
   - `dataset/0/813270/audio.mp3`;
   - `dataset/0/882486/audio.mp3`;
   - `dataset/0/440089/192.MP3`;
   - `dataset/0/1113833/audio.mp3`;
   - `dataset/0/2080593/audio.mp3`;
   - `dataset/0/863309/audio110.mp3`;
4. if all earlier gates pass, the same already-exposed pilot42 from Exp013.

No holdout100-v2, broad500, full5050, network catalog data, manual listening,
or ramp-specific production dataset is authorized by this card.

## Baseline / Comparator

- Baseline selector: Experiment 013 raw-run ordinal production selector.
- Baseline pilot42 result:
  - `42` rows;
  - `17/42` v3 accepted;
  - `25/42` v2 fallback;
  - fallback rate `59.52%`;
  - `0` hard failures;
  - row p90 runtime `3.58 s`;
  - maximum seam error `0.0 ms`;
  - stable: `13/22` accepted, `9/22` fallback, one accepted false jump;
  - jump: `4/20` accepted, `16/20` fallback, only two accepted jump curves,
    zero accepted exact jump hits.
- Product comparator: current timing v2 remains the fallback comparator.
- Weak comparator: `.osu` redlines/object evidence are read only after Exp014
  selected candidate fingerprints and product statuses are frozen.

## Primary Metric

Pilot42 product and jump-structure recovery, evaluated only after prediction
freeze:

- `hard_failure = 0`;
- maximum seam error `<= 5 ms`;
- stable accepted false-jump rows `<= 1/22`;
- stable fallback rows `<= 4/22`;
- at least `4/20` jump rows accepted as weak-compatible jump structures;
- among predicted jump rows with weak boundaries, mean nearest-boundary error
  at `<= 1000 ms` improves over Exp013.

These gates are a development-screening bar for this exposed pilot, not a
promotion bar and not a fresh-holdout claim.

## Secondary Metric

- product status counts and fallback reasons;
- candidate-family presence by row: constant, persistent, ABA, and multi-step;
- candidate count and cap-pruning reason;
- raw self-score, collapsed constant score, paired gain sign, retained beat
  count, candidate-domain beat count, and retained/domain ratio;
- generalized BeatThis support rank and tie-break reason;
- selected section count and unsupported short-section count;
- weak phase mean/p50/p90/max in milliseconds and beats;
- direct and alias-aware BPM coverage/error;
- endpoint drift magnitude, max-prefix drift, and 30/60 s local drift;
- jump boundary precision/recall and nearest-boundary time error;
- runtime p50/p90/max and feature-cache/decode source;
- stable false-run diagnostics.

## Verify Command / Evaluation Procedure

1. Run focused unit/synthetic tests for:
   - candidate-local domain support;
   - shared axis within a candidate but different end beats across alternatives;
   - persistent up/down proposals;
   - short and long ABA proposals;
   - at most four-section multi-step proposals;
   - collapsed full-duration constant comparators;
   - strict positive paired raw gain;
   - stable false-run rejection;
   - constant always competing;
   - raw unavailable fallback;
   - ramp exclusion;
   - candidate cap `<= 64`;
   - diagnostic-only reporting fields.
2. Run the two mechanism rows and freeze product status and selected
   fingerprints before weak evaluation.
3. Run the fixed structure-manifest subset only if mechanism rows pass.
4. Run pilot42 only if every prior gate passes, with the same source/config and
   no threshold, feature, metric, split, or fallback-route change.
5. Stop before protected data unless every frozen Exp014 pilot42 gate passes
   and a separate next-stage card is created.

Expected focused command shape:

```text
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_tempo_track.py \
  tests/timing/test_timing_v3_exp013_pilot.py
```

The exact command may use a new Exp014 test file if the runner is separated
from Exp013.

## Guard Check

- no `.osu`, metadata BPM, network BPM, manual listening, or weak comparator
  access before inference fingerprints are frozen;
- no protected holdout100-v2, broad500, or full5050 access;
- no threshold tuning on pilot42;
- no raw feature extractor change;
- no BeatThis cache change, shift change, or BeatThis rerun;
- no use of BeatThis chunk seams as section boundaries;
- production output remains constant/jump only;
- ramp remains diagnostic-only;
- all accepted candidates have one absolute beat axis, integer contiguous
  section intervals, and seam error `<= 5 ms`;
- candidate alternatives may differ in terminal `end_beat`, but sections inside
  each accepted candidate must be internally contiguous;
- candidate cap remains `<= 64`;
- raw retained/domain guard remains `>= 0.90` and retained beats `>= 16`;
- paired gain uses strict sign, not a tuned margin;
- runner reporting repairs are diagnostic-only.

## Qualitative Check

Inspect frozen per-row diagnostics after prediction freeze:

- `2300685` should remain constant and show no positive-gain false structure.
- `618173` should retain a short ABA or equivalent compatible jump.
- `813270` should reject the Exp013 false jump unless its paired raw gain is
  strictly positive and the weak post-freeze audit supports it.
- Persistent and long-ABA manifest rows should contain at least one structurally
  compatible candidate before ranking.
- Any v3 constant accepted on a weak jump row must be counted as a missed jump
  in pure-v3 quality, not hidden as product safety.
- Fallback must be rare and must have precise reasons.

## Positive Signal

- synthetic matrix passes without changing feature extraction or thresholds;
- mechanism rows pass;
- fixed structure-manifest subset has compatible candidates for persistent,
  long-ABA, and multi-step rows;
- pilot42 reduces Exp013's fallback rate while not exceeding the stable
  false-jump guard;
- at least four exposed jump rows are accepted as weak-compatible jump
  structures;
- seams remain exact and runtime stays within budget.

## Negative Signal

- stable false-run fixtures get positive paired gain and accept false jumps;
- persistent, long-ABA, or multi-step synthetic fixtures fail candidate
  presence;
- the collapsed constant comparator is not full-duration or is forced to share
  an invalid end beat with a persistent-jump candidate;
- candidate-local scoring censors candidate tails through a cross-candidate
  common domain;
- pilot42 improvement depends on changing thresholds after exposure;
- fallback remains near Exp013 levels because candidate presence is still low;
- jump rows are accepted mainly as constants and do not improve jump structure
  metrics;
- runtime or candidate cap fails.

## Kill Criteria

Kill this card if any of these occur:

- the synthetic matrix cannot pass with the selected candidate family;
- the two mechanism rows cannot pass without threshold, feature, or evaluator
  changes;
- the fixed structure-manifest subset lacks candidates for persistent, long,
  or multi-step structures;
- pilot42 fails the primary metric gates;
- any inference/evaluation separation, protected-data, continuity, candidate
  cap, raw coverage, production-scope, or hard-failure guard fails.

## Expected Failure Modes

- Raw local tempo runs may still be caused by texture/fill changes rather than
  tempo changes.
- Candidate cap pressure may remove a rare but correct multi-step candidate.
- Paired raw gain sign may be too weak to reject some stable false positives,
  or too strict on low-energy true jumps.
- BeatThis support rank may reorder positive-gain structures incorrectly.
- Weak `.osu` comparators may be aliased, editorial, or sparse; a true audio
  improvement can still look weak-negative on some rows.
- Persistent changes near the end of a song may have too little post-change
  coverage for a stable score.
- Multi-step rows may actually be ramp-like audit material, which Phase 1 must
  not convert into production ramps.

## Confounders

- Pilot42 is already exposed and cannot support fresh performance claims.
- The fixed structure manifest is selected using already-exposed weak
  structure labels, so it is a mechanism check rather than an unbiased metric.
- Current raw features were designed as a deterministic verifier, not a full
  tempo tracker.
- BeatThis cache provenance still cannot justify seam-specific claims.
- The task objective forbids using `.osu` labels for inference even when they
  clearly identify a jump shape.
- Exp013 runner summary endpoint drift was signed; Exp014 reports absolute
  magnitude to avoid repeating that reporting confusion.

## Expected Runtime / Runtime Budget

- Unit/synthetic tests: under 15 seconds.
- Two mechanism rows with existing caches: under 30 seconds.
- Structure-manifest subset: under 2 minutes.
- Pilot42 with existing raw caches: under 10 minutes.
- Hard stop: any row above 180 seconds, p90 row runtime above 5 seconds on
  pilot42 with existing caches, or repeated hard/integrity failure.

## Result Interpretation Plan

- Positive result would suggest: candidate presence, not raw feature capacity,
  was the main Exp013 jump bottleneck, and paired raw gain is a usable
  stable-specificity guard for the next sealed-stage proposal.
- Negative result would suggest: either the raw verifier is insufficient for
  production selection or this bounded enumeration still misses the needed
  section model; do not open protected data.
- Ambiguous result would require: separating candidate presence from selection
  by freezing a candidate-oracle coverage audit on exposed identities only.
- Human owner decides: whether a positive pilot42 is strong enough to create a
  next-stage card for the next authorized data layer.
- Next-loop action if positive: append result, freeze source/config/exposure,
  and create a separate next-stage card; do not immediately open holdout in the
  same card.
- Next-loop action if negative: record TV3 result, `MUTATE` or `KILL`, and
  keep protected data sealed.
- Next-loop action if ambiguous: write a smaller diagnostic card that isolates
  candidate-presence, paired-gain selection, and weak-comparator mismatch.

## Result Log Template

- Experiment: Timing v3 Experiment 014 boundary-complete jump candidates with
  paired raw gain
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Source/config/provenance:
- Runtime:
- Product status counts:
- Primary metric value:
- Secondary metric value:
- Verify command / result:
- Guard command / result:
- Qualitative observations:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected variant:
- Candidate variants rejected before execution:
- Local verification outcomes:
- Selection pressure observed:
- Candidate-family coverage:
- Paired raw-gain diagnostics:
- Fallback reasons:
- Exposure manifest:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Invalidated Pre-Cleanup Result Log: Source/Synthetic and Exposed Mechanism2

This run was produced by a concurrent, incomplete Exp014 implementation before
the source-cleanup and counterfactual fixes recorded below.  Its source state is
not the authoritative Exp014 implementation, its artifact SHA differs from the
post-cleanup run, and it must not be used for algorithm selection or promotion.
It is retained only as an exposure/audit record.

- Experiment: Timing v3 Experiment 014 boundary-complete jump candidates with
  paired raw gain.
- Date: 2026-08-14 Asia/Shanghai.
- Dataset slice:
  - source/unit/synthetic fixtures;
  - two already-exposed mechanism rows:
    `dataset/0/2300685/audio.mp3` and `dataset/0/618173/audio.mp3`.
- Source/config/provenance:
  - tempo-track source SHA-256:
    `fb1ff93445ff16f2a15249b54d130201bbdf78d1cff21d66a10c5483ce358b58`;
  - Exp014 runner source SHA-256:
    `c5d5c65a2d92a11373bd4b00cec569fab763d2a2d6633daeaf11dd9d43fe9f3b`;
  - Exp013 shared runner source SHA-256:
    `70891775233cf0c66b0d948689cf8a7d3505c57192ef0d5beed81f3f22f1b3bb`;
  - tempo-track test SHA-256:
    `cadeae2ff99015ffe63a367b2f5285bf0925701da16ac0a0d9abb077df1198f0`;
  - pilot runner test SHA-256:
    `428cb31437bbfede532c8de671e8e02e0abec91da25348650a23a3c82642a105`.
- Verify command / result:

```text
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_analytic_curve.py \
  tests/timing/test_timing_v3_audio_evidence.py \
  tests/timing/test_timing_v3_curve_metrics.py \
  tests/timing/test_timing_v3_real_audio_pilot.py \
  tests/timing/test_timing_v3_exp013_pilot.py \
  tests/timing/test_timing_v3_tempo_track.py
```

Result: `94 passed in 6.64 s`.

- Mechanism command:

```text
.venv/bin/python -m pulsefield_model.timing.evaluation.exp014_pilot \
  --pilot-jsonl artifacts/reports/timing/timing_v3_pilot_rows_80_v1.jsonl \
  --baseline-v2-jsonl artifacts/reports/timing/timing_v3_v2_baseline_pilot80_v1.jsonl \
  --output-jsonl artifacts/reports/timing/timing_v3_exp014_mechanism2_v1.jsonl \
  --summary-json artifacts/reports/timing/timing_v3_exp014_mechanism2_v1_summary.json \
  --repo-root /Users/l/projects/Pulsefield-model \
  --cache-audio-key '{"audio_mtime_ns":1779024740872522853,"audio_path":"/Users/l/projects/Pulsefield-model/dataset/0/2300685/audio.mp3","audio_size":1388820,"version":1}' \
  --cache-audio-key '{"audio_mtime_ns":1779024695431570386,"audio_path":"/Users/l/projects/Pulsefield-model/dataset/0/618173/audio.mp3","audio_size":968567,"version":1}'
```

- Mechanism artifacts:
  - JSONL SHA-256:
    `2b7079ab7c969cab71bf0c020a594bd83a1240c212f8494c97bed551d353bb30`;
  - summary SHA-256:
    `82dcf22992433d5c8001c8b57fbc3a8b3226618b92af21e8d0b36c51717efde2`.
- Mechanism result:
  - `2/2` `v3_accepted`, `0` fallback, `0` hard failure;
  - maximum seam error `0.0 ms`;
  - row p50/p90/max runtime `2.40/2.50/2.53 s`;
  - stable probe `2300685` selected constant `200 BPM`;
  - short-ABA probe `618173` selected a three-section jump
    `175.193 -> 146.621 -> 175.193 BPM`, with boundaries
    `56477.345 ms` and `58932.654 ms`;
  - `618173` weak phase mean/p90 was `22.99/41.37 ms`; weak exact jump hit
    remained `false`.
- Interpretation: source/synthetic and mechanism2 passed only as a mechanism
  screen.  The short-jump row recovered a local short ABA shape, but not an
  exact weak-comparator jump.

## Invalidated Pre-Cleanup Result Log: Fixed Exposed Structure-Manifest6

This six-row run followed the invalidated pre-cleanup mechanism result above.
Those already-exposed identities were accessed, but the run is not valid
decision evidence for the post-cleanup implementation.  The authoritative
post-cleanup mechanism gate below failed, so no structure-manifest6 rerun was
authorized or performed after cleanup.

- Experiment: Timing v3 Experiment 014 boundary-complete jump candidates with
  paired raw gain.
- Date: 2026-08-14 Asia/Shanghai.
- Dataset slice: fixed already-exposed structure-manifest subset:
  - stable false-jump regression: `dataset/0/813270/audio.mp3`;
  - persistent down: `dataset/0/882486/audio.mp3`;
  - persistent up: `dataset/0/440089/192.MP3`;
  - long ABA: `dataset/0/1113833/audio.mp3`;
  - progressive down: `dataset/0/2080593/audio.mp3`;
  - multi-step jump: `dataset/0/863309/audio110.mp3`.
- Command:

```text
.venv/bin/python -m pulsefield_model.timing.evaluation.exp014_pilot \
  --pilot-jsonl artifacts/reports/timing/timing_v3_pilot_rows_80_v1.jsonl \
  --baseline-v2-jsonl artifacts/reports/timing/timing_v3_v2_baseline_pilot80_v1.jsonl \
  --output-jsonl artifacts/reports/timing/timing_v3_exp014_structure_manifest6_v1.jsonl \
  --summary-json artifacts/reports/timing/timing_v3_exp014_structure_manifest6_v1_summary.json \
  --repo-root /Users/l/projects/Pulsefield-model \
  --cache-audio-key ... # six fixed exposed cache_audio_keys listed above
```

- Output artifacts:
  - JSONL SHA-256:
    `d207a53a88df7ce52ad5aec312fa22f38c2466495857ec1f82269c2efc3a8153`;
  - summary SHA-256:
    `459bc8534bae4294c210d2ac4e6e3ea35e51335d67a4021801c0911c2d73d847`.
- Aggregate result:
  - `6/6` `v3_accepted`, `0` fallback, `0` hard failure;
  - maximum seam error `0.0 ms`;
  - row p50/p90/max runtime `4.25/6.56/6.69 s`, exceeding the five-second
    p90 target before pilot42;
  - accepted weak phase mean/p90 aggregate `53.67/109.10 ms`;
  - accepted weak alias BPM MAE aggregate `8.89 BPM`.
- Row-level observations:
  - `813270` is a stable control but selected a two-section jump
    `199.928 -> 180.853 BPM`; this is a stable false jump regression.
  - `882486` is the persistent-down manifest row but selected a constant
    `103.014 BPM`, so the persistent structure was not recovered.
  - `440089` selected a short ABA `149.997 -> 85.123 -> 149.997 BPM`, not the
    expected persistent-up structure.
  - `1113833` selected a long ABA-like jump, but weak exact jump hit remained
    false.
  - `2080593` selected an extreme `180.125 -> 80.0 -> 180.125 BPM` short jump,
    not a progressive down structure.
  - `863309` selected `131.774 -> 230.769 -> 131.774 BPM`, with large
    weak alias error (`40.55 BPM`) and phase p90 `153.34 ms`.
- Positive signal observed:
  - candidate generation now emits persistent, ABA, and multi-step families;
  - all selected curves remain phase-continuous with exact seams;
  - the hard-failure count stayed zero.
- Negative signal observed:
  - the fixed structure manifest failed the pre-pilot qualitative gates:
    stable false-jump rejection failed, persistent structure selection failed,
    and progressive/multi-step rows were collapsed to implausible ABA choices;
  - row runtime already exceeded the five-second p90 target on this six-row
    exposed subset;
  - weak exact jump hits remained false on the selected jump rows.
- Kill criteria triggered: yes.  The card requires every prior gate to pass
  before pilot42.  Structure-manifest6 is negative, so pilot42,
  holdout100-v2, broad500, and full5050 remain sealed.
- Interpretation: `NEGATIVE` / `MUTATE`.  Exp014 solved candidate-family
  presence but not structure selection.  Strict paired raw-gain sign is too
  permissive for stable false jumps and too myopic for persistent/progressive
  structure choice.
- Recommended next step: create a new card that changes the structure-selection
  mechanism, likely by using a source-only dynamic path or explicit
  structure-family arbitration.  Do not tune Exp014 thresholds on pilot42.

## Result Log: Actual Mechanism2 After Source Cleanup

- Experiment: Timing v3 Experiment 014 boundary-complete jump candidates with
  paired raw gain.
- Date: 2026-08-14 Asia/Shanghai.
- Dataset slice: two already-exposed mechanism rows only:
  - stable probe: `dataset/0/2300685/audio.mp3`;
  - short-ABA probe: `dataset/0/618173/audio.mp3`.
- Source/config/provenance:
  - synthetic/related guard: `95 passed`;
  - output JSONL:
    `artifacts/reports/timing/timing_v3_exp014_mechanism2_v1.jsonl`;
  - output JSONL SHA-256:
    `f8d3ced1e7bfb2d97e932ded4c20bbad045268a1301102753c4cc268d501229b`;
  - summary:
    `artifacts/reports/timing/timing_v3_exp014_mechanism2_summary_v1.json`;
  - summary SHA-256:
    `862c6c788cbbd2ba3251493d5975eeeb13fd63b371c489e3a00e734c50c2530e`.
- Mechanism result:
  - `2/2` `v3_accepted`;
  - maximum seam error `0.0 ms`;
  - row p90 runtime `2.486 s`;
  - total runtime `4.766 s`.
- Row-level observations:
  - `2300685` selected constant `200 BPM`; weak exact hit was `true`, direct
    coverage was `1.0`, and weak phase p90 was `31 ms`.
  - `618173` selected `raw_run_persistent_a_to_b_start`, with sections
    `175 BPM @ 276 ms -> 158.426 BPM @ 54104.57 ms`.
  - `618173` weak truth is
    `175 BPM @ 267 ms -> 143 BPM @ 55124 ms -> 175 BPM @ 59296 ms`.
  - `618173` predicted one boundary while the weak comparator has two;
    matched boundary count was `0`, recall was `0`, weak exact hit was
    `false`, direct coverage was `0.8941`, and weak phase p90 was `15.25 ms`.
- Positive signal observed:
  - the stable mechanism row stayed in the constant lane;
  - selected curves retained exact seams;
  - row runtime stayed under the mechanism-gate target.
- Negative signal observed:
  - the short-ABA mechanism row lost the prior near-`143 BPM` / `4.17 s`
    short-ABA recall and selected a wrong persistent jump;
  - paired raw gain was positive but too weakly specific
    (`+0.00783`) to prevent the wrong family.
- Kill criteria triggered: yes.  The mechanism gate failed, so
  structure-manifest6, pilot42, holdout100-v2, broad500, and full5050 were not
  accessed under this actual run.
- Suspected root cause: internal `maximum_jump_candidates_44` pruning plus
  candidate-family imbalance removed the near-`143 BPM` short-ABA candidate
  before selection; paired gain then promoted the wrong persistent A->B
  candidate.
- Interpretation: `NEGATIVE` / `MUTATE`.  Candidate-family generation needs a
  cap-allocation fix before any additional data layer.
- Recommended next step: create a separate mutation with family-stratified cap
  reservation only.  Do not tune thresholds or open later data layers under
  Exp014.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this result: no further execution under Exp014.
  Structure-manifest6, pilot42, and later data layers are not authorized
  because the actual mechanism gate was negative.
- Closed loop complete: yes
- Remaining ambiguity: whether a dynamic structure path can recover persistent
  and progressive rows without introducing stable false jumps.

## Next-Loop Action

- If positive: write the result log and create the next authorized data-layer
  card.
- If negative: kill or mutate the candidate/selection family in a new card.
- If ambiguous: create a focused candidate-presence versus selector diagnostic
  card on exposed data only.

## Novelty Notes

- Closest analogies: sparse change-point enumeration, paired model-comparison
  tests, and bounded tempo-state search.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this remains an
  engineering variation on Pulsefield's phase-continuous beat-index timing
  representation.
