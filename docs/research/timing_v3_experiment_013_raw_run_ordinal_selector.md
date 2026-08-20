# Timing v3 Experiment 013: Raw-run ordinal production selector

Status: negative at formal exposed pilot42; holdout sealed

## Mode

- Mode: planner
- Route: TEST
- Source idea: Experiment 012 failed before pilot execution because run-gate
  authority was assigned to BeatThis local observations.  On the already
  exposed `618173` mechanism row, BeatThis had no run while raw audio showed a
  coherent negative run.
- Acceptance source, if any: goal objective, Experiment 010 `MUTATE` rule,
  Experiment 011 negative result, and Experiment 012 negative result.
- Source snapshot / evidence grade: strong mechanism evidence from already
  exposed `2300685` / `618173` probes plus an informal pre-freeze diagnostic on
  eight already exposed shortest cache-hit pilot80 rows.  The diagnostic is not
  a formal pilot80 result; its exact identities and identity-set hash are
  recorded below, and the remaining pilot42 slice is not yet formally run.

## Hypothesis

Raw-audio local observations should decide tempo-run presence, direction, and
time-overlap.  BeatThis should remain only an ordinal within-lane support rank.
With stricter alias canonicalization, deviation, overlap, and middle-tempo
compatibility gates, this should recover the exposed short jump without adding
stable false jumps.

## Root Objective

Repair Timing v3 Phase 1 production selection for constant and short-jump rows
while preserving the product contract: accepted v3 outputs are
phase-continuous constant/jump curves on one absolute integer beat axis;
insufficient evidence returns explicit `v2_fallback`; ramps remain diagnostic
and cannot count as production success.

## Goal Decomposition

- Subgoal 1: let raw audio, not BeatThis, determine whether a local tempo run
  exists.
- Subgoal 2: avoid cross-alias raw-score censorship by scoring each candidate
  over its own physical beat coverage.
- Subgoal 3: keep stable specificity by using the constant lane only when no
  raw deviation run survives, and explicit v2 fallback when a raw run survives
  but candidate generation provides no compatible jump.
- Subgoal 4: admit a production jump only when raw-run direction, time overlap,
  and middle-tempo ratio are all compatible.
- Subgoal 5: preserve Timing v3 guards: no `.osu` inference, no metadata/manual
  inference, no local phase reset, integer section boundaries, seam continuity
  at or below 5 ms, at most 64 candidates, production constant/jump only, and
  zero hard failures.

## Candidate Variants

- Variant A: rerun Experiment 012 with a lower BeatThis run threshold.
- Variant B: use raw-run gating with the initial Exp013 v1 rule:
  primary-alias normalization, three-point weighted median, deviation
  `max(6 BPM, 5%)`, and time-overlap only.
- Variant C: use raw-run gating with the stricter Exp013 v2 rule:
  small-rational alias canonicalization, three-point weighted median, deviation
  `max(8 BPM, 5%)`, one-second run-span expansion, at least 500 ms candidate/run
  overlap, and middle-tempo/run ratio at most 1.15.
- Variant D: remove run gating and rank all constants/jumps by raw self-rank
  plus BeatThis support rank.

## Local Verification Matrix

- Variant A:
  - Local check: replay the exposed `618173` observation sequence.
  - Fail condition: BeatThis still lacks a coherent three-point run, or lowering
    thresholds admits low-strength noise.
- Variant B:
  - Local check: informal diagnostic on eight already exposed shortest
    cache-hit pilot80 rows.
  - Fail condition: stable rows or jump rows without compatible candidates are
    routed into the jump lane.
- Variant C:
  - Local check order:
    1. synthetic constant 200 BPM;
    2. synthetic short `175 -> 143 -> 175` ABA jump;
    3. synthetic time-linear ramp as diagnostic-only non-production output;
    4. exposed stable/direct-alias probe `dataset/0/2300685/audio.mp3`;
    5. exposed short-jump probe `dataset/0/618173/audio.mp3`;
    6. formal already exposed pilot80 high/medium non-ambiguous stable/jump
       rows, expected count 42, only after all earlier gates pass.
  - Pass condition: stable/no-compatible-run rows select constant or fallback;
    the exposed short-jump row selects a production three-section jump retaining
    the short middle excursion; ramps remain production-excluded.
- Variant D:
  - Local check: synthetic constant and exposed `2300685`.
  - Fail condition: stable rows can accept false jump candidates because no run
    gate constrains the jump lane.

## Selected Variant

- Selected: Variant C.
- Rejected:
  - Variant A preserves the incorrect BeatThis evidence role.
  - Variant B is too permissive in the already exposed eight-row diagnostic.
  - Variant D is under-gated and likely trades recall for stable false positives.
- Why this is the smallest useful test: it changes only selector evidence
  routing and eligibility.  Candidate generation, analytic representation, raw
  feature extraction, evaluator metrics, split policy, fallback contract, ramp
  policy, and protected-data boundaries stay fixed.

## Selection Pressure

- Primary pressure: recover top-1 production short-jump selection when a viable
  candidate is present and raw observations show a compatible run.
- Specificity pressure: use the constant lane only in the absence of a
  surviving raw run; when a run survives but no compatible jump exists, return
  explicit v2 fallback rather than forcing either a jump or a v3 constant.
- Guard pressure: no accepted candidate may violate phase continuity, integer
  beat boundaries, seam continuity, candidate cap, raw availability,
  inference/evaluation separation, or Phase 1 production scope.
- Runtime pressure: selector overhead must stay small relative to candidate
  generation and raw self-scoring; the 10-minute algorithmic target remains
  under 5 seconds excluding one-time audio decode/mel cache creation.
- Kill pressure: if this rule cannot pass synthetic gates and both exposed
  mechanism probes without changing candidates, features, evaluator, split, or
  thresholds, stop and mutate or kill before formal pilot42 expansion.

## Research Question

Can a stricter raw-run-gated ordinal selector safely choose between constant and
paired short-jump Timing v3 candidates when BeatThis misses the local run but
raw audio shows it?

## Closest Analogies / Novelty Layer

- Closest analogies: gated change-point model selection, onset-strength
  tempo-run detection, ordinal rank aggregation, and non-parametric model
  selection under separate evidence roles.
- Relevant taxonomy bucket: bounded deterministic candidate selection, not model
  training.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: representation remains the
  existing analytic constant/jump curve; this experiment changes only production
  selection.

## Minimal Change

Implement an Experiment 013 selector behind a disabled-by-default Timing v3
path.  The frozen v2 rule is:

1. Keep the Experiment 010 candidate generator unchanged.  Candidate count
   remains at most 64.
2. Keep the raw-audio feature family unchanged: deterministic 16 kHz audio,
   80-bin log-mel, 10 ms hop, four-band positive spectral flux,
   beat-vs-half-beat support, and robust local aggregation.
3. Build raw local observations from the existing generator observation surface.
   Inherit the generator's existing `minimum_local_strength=0.04`.
4. Canonicalize each raw local BPM with small rational multipliers
   `p / q`, where `p, q in {1, 2, 3, 4, 5}`:
   - if a multiplier maps the observation within 1.25% of the primary base BPM,
     map the observation back to the primary alias family;
   - otherwise retain the original observation BPM; do not pull a genuine
     deviation toward a non-matching rational transform.
5. Smooth canonicalized raw BPMs with a three-point weighted median.
6. Mark a raw deviation when
   `abs(smoothed_bpm - primary_bpm) >= max(8 BPM, 0.05 * primary_bpm)`.
7. A raw run survives only when it has the same deviation direction for at least
   three observations and adjacent observation gaps no larger than
   `1.5 * observation_hop`.
8. Expand the surviving run span by one second from the first and last
   observation centers.
9. If multiple raw deviation runs survive, choose the dominant run by:
   1. larger `abs(weighted_median_delta_bpm)`;
   2. longer expanded duration;
   3. larger summed local strength;
   4. earlier expanded start time.
10. Compute `raw_self_score(c)` for each candidate over that candidate's own
    scoreable physical beat times.  Do not intersect beat indices across
    unrelated tempo aliases.
11. Apply the frozen raw-score coverage guard:
    - `retained_beat_count / candidate_domain_beat_count >= 0.90`;
    - `retained_beat_count >= 16`.
    Terminal span or endpoint-duration diagnostics may be reported, but tight
    endpoint equality cannot be the sole production guard.
12. Run production eligibility before ranking:
    - constant candidates are production-eligible in the constant lane;
    - jump candidates are production-eligible only if a compatible raw run
      survives;
    - diagnostic ramp candidates and other non-constant/non-jump structures are
      never production-eligible in Phase 1;
    - if raw self-scoring or raw observations are unavailable, return explicit
      `v2_fallback`.
13. A jump candidate is compatible only when:
    - it has exactly three constant sections;
    - outer sections are alias-consistent with the primary base family;
    - the middle section deviation has the same sign as the selected raw run;
    - the middle interval overlaps the expanded raw-run span by at least
      500 ms;
    - the middle tempo ratio to the raw-run representative tempo is at most
      1.15 after alias canonicalization.
14. If a raw run exists but no compatible jump candidate exists, return
    explicit `v2_fallback`.  Only a row with no surviving raw run uses the
    constant lane.
15. If a compatible jump lane exists, rank eligible jumps by
    `raw_self_rank + beatthis_aba_support_delta_rank`.
16. Rank constant-lane candidates by raw self-rank.
17. Tie-break first by higher raw self-score and then by canonical candidate
    fingerprint.

## Pre-Freeze Diagnostic Evidence

An informal diagnostic on eight already exposed shortest cache-hit pilot80 rows
was used to reject Variant B and freeze Variant C before formal pilot42
execution:

- Exp013 v1 was wrong on this diagnostic slice.
- Exp013 v2 routed all 4 stable rows to constant.
- Of 4 jump rows, only `618173` had a compatible generated jump candidate; v2
  selected candidate index 33 for that row.
- The other 3 jump rows had no compatible jump candidate.  The exploratory
  prototype routed them to constant; the production rule is frozen more
  conservatively to explicit `v2_fallback` rather than accepting a v3 constant
  against contradictory raw-run evidence.
- This diagnostic is not a formal aggregate result.  The exact eight identities
  and their identity-set hash below form its exposure manifest; the remaining
  pilot42 slice has not been formally run.

The eight exposed identities are:

- stable: `dataset/0/2300685/audio.mp3`, `dataset/0/1670474/audio.mp3`,
  `dataset/0/1265173/audio.mp3`, `dataset/0/905071/audio.mp3`;
- jump: `dataset/0/618173/audio.mp3`, `dataset/0/882486/audio.mp3`,
  `dataset/0/349774/KakushintekiMetamaruphose!.mp3`,
  `dataset/0/1564921/audio.mp3`.

Canonical JSON of their sorted cache-key SHA-256 identities hashes to
`44360a6f3054a88f20a42689be2aaf58a5a3f654e91765728f80dd3f338e650b`.
The exploratory prototype SHA-256 is
`337112983b84ef43030517cac5d71f6820fa713e80e98a6656fdef78cc7cd5be`;
its eight-row run took `9.82 s` with existing BeatThis and mel caches.

The exposed pilot80 source JSONL SHA-256 is
`cdb5e2af87d99c8af3bbff71a0985bb490fdeed169d10cb0a64b17a8ca0296d7`.

## Frozen Source Snapshot Before Real Mechanism Execution

The source/unit/synthetic gate passed before either formal mechanism row was
re-executed.  Reviews then caught three card/integrity defects: raw ranks were
inherited from the all-candidate diagnostic order rather than recomputed inside
the eligible lane; middle/run tempo compatibility had not applied the frozen
primary-family alias snap; and raw-evidence arrays were not defensively copied.
These were card-conformance and evidence-integrity fixes, not threshold or
feature changes.  The selector tests and mechanism gate were restarted before
any weak comparator access.  The final frozen identities are:

- analytic curve source:
  `766762aafdf0a3e643b3689c7ec659d14a0dd002befd2607a410fc368962c4d3`;
- raw-audio evidence source:
  `06f04ef90e5ba7f8048b9f9a030377f53456a2b8a99f6ed51aadca18cdf700b6`;
- tempo candidate/selector source:
  `832558ac2fc033538fd347bf2fd8b4cd045421f6b923f5a88a7603f1d671a96b`;
- curve metrics/evaluator source:
  `1a70a9c0e8f965b9c7a9de74bc1c99711c95b938abda900e871f4fce0f316c2a`;
- Exp013 inference-first pilot runner source:
  `107ed3808d976039cdb176a5e4c3f4b3a249c294c917b6ee2dcef814c0d2a8e3`;
- analytic curve test:
  `6034fc0ad7d6ac1a3222a59b3b37dbd2bd10d2c96ae8d0f3b554be8d0d7c5889`;
- raw-audio evidence test:
  `4e9b5d41b233602ed7360153c7be023d2f8114aa057aae6ee389c6492e745eb0`;
- curve metrics/evaluator test:
  `621f84d6ef7929dfe1ef4087a24bf613267eaa11b98b78632941318ff005a6ef`;
- legacy real-audio pilot test:
  `2340d9b613c12e2da66430162823424c5e17890ce3dbe8bfbd1140fe9eaab05d`;
- tempo candidate/selector test:
  `ba086466e51e98dffec7f4426f730860a9b9956a807520000ea7535f14a732e9`;
- Exp013 pilot runner test:
  `3336dd710a9db2d39b9746bfe3f3bdd763f67d86a1a4718615292372f60186ae`.

The paired pilot80 timing-v2 baseline JSONL SHA-256 is
`5d9bb3c50f4173b4bde60cdd1dd30a152565bd89934f2e6cb2407dd8910075a7`.

The pre-real-data verification command was:

```text
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_analytic_curve.py \
  tests/timing/test_timing_v3_audio_evidence.py \
  tests/timing/test_timing_v3_curve_metrics.py \
  tests/timing/test_timing_v3_real_audio_pilot.py \
  tests/timing/test_timing_v3_exp013_pilot.py \
  tests/timing/test_timing_v3_tempo_track.py
```

Result after final runner integrity fixes: `88 passed in 3.69 s`; focused compile
and diff whitespace checks passed.  This snapshot is the only implementation
eligible for the two formal exposed mechanism rows and, if they pass, the
formal pilot42 run.

## Files Likely to Change

If this card is executed, expected code changes are limited to:

- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- `src/pulsefield_model/timing/evaluation/exp013_pilot.py`
- `tests/timing/test_timing_v3_audio_evidence.py`
- `tests/timing/test_timing_v3_tempo_track.py`
- `tests/timing/test_timing_v3_exp013_pilot.py`
- a result log under `docs/research/`

This planning step itself adds only:

- `docs/research/timing_v3_experiment_013_raw_run_ordinal_selector.md`

## Read-Only Context Files

- `/Users/l/.codex/attachments/97bd173a-3590-4524-8b1f-f7a90c5e0223/goal-objective.md`
- `AGENTS.md`
- `README.md`
- `docs/research/timing_v3_experiment_010_real_audio_short_jump.md`
- `docs/research/timing_v3_experiment_011_candidate_local_audio_gain.md`
- `docs/research/timing_v3_experiment_012_ordinal_production_selector.md`
- `src/pulsefield_model/timing/v3/analytic_curve.py`
- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- `src/pulsefield_model/timing/evaluation/curve_metrics.py`
- existing Timing v3 tests under `tests/timing/`

## Dataset Slice

Execution order is fixed:

1. synthetic/unit fixtures only;
2. already exposed stable/direct-alias mechanism probe:
   `dataset/0/2300685/audio.mp3`;
3. already exposed short-jump mechanism probe:
   `dataset/0/618173/audio.mp3`;
4. if all gates pass, already exposed pilot80 high/medium non-ambiguous
   stable/jump rows, expected count 42.

Protected holdout100-v2, broad500, and full5050 are sealed for this card.

## Baseline / Comparator

- Baseline selector: Experiment 012 ordinal production selector, which failed
  pre-execution because BeatThis run evidence was assigned run-gate authority.
- Earlier negative comparator: Experiment 011 candidate-local raw-gain formula,
  which failed before pilot80 because constants and jumps were scored on unlike
  numeric scales.
- Exp013 local negative comparator: initial raw-run v1 gate from this card,
  rejected on the already exposed eight-row diagnostic.
- Candidate-generation comparator: Experiment 010 generator and candidate list
  stay fixed.
- Product comparator: timing v2 remains the fallback comparator; fallback is
  reported separately and is not counted as v3 success.
- Weak evaluation comparator: `.osu` redlines and object-derived evidence are
  read only after selected candidate fingerprints and product statuses are
  frozen.

## Primary Metric

Mechanism-gate top-1 product correctness:

- synthetic constant: accepted production candidate is constant/direct-tempo, or
  explicit fallback if raw evidence is intentionally unavailable;
- synthetic short ABA jump: accepted production candidate is a phase-continuous
  three-section jump with the middle excursion retained;
- exposed `2300685`: no production false jump;
- exposed `618173`: production jump retains the short middle section and its two
  weak-comparator boundaries are within 750 ms after prediction freeze.

## Secondary Metric

- direct BPM coverage and alias-aware BPM coverage;
- stable false-boundary song rate and false-boundary count;
- jump boundary precision/recall at 500 ms and 1,000 ms;
- direct left/right tempo-pair accuracy for matched jump boundaries;
- signed initial phase, mean phase, p90 phase, endpoint drift, and max-prefix
  drift;
- raw-run presence/direction/time span, raw observation count, representative
  raw-run tempo, and dominant-run tie-break reason;
- selected candidate rank before and after Exp013 selection;
- raw self-score rank, BeatThis `_aba_support_delta` rank, tie-break reason, and
  fallback reason;
- compatibility diagnostics: overlap milliseconds and middle/run tempo ratio;
- coverage diagnostics: retained beat count, candidate domain beat count,
  retained/domain ratio, and optional terminal span;
- candidate count, selector runtime, raw scoring runtime, and end-to-end
  generation plus selection runtime;
- production status counts: `v3_accepted`, `v2_fallback`, `hard_failure`;
- diagnostic ramp rows reported with ramp accuracy as `null`.

## Verify Command / Evaluation Procedure

First run focused unit/synthetic tests covering:

- small-rational alias canonicalization with `p, q in {1, 2, 3, 4, 5}`;
- 1.25% primary-alias snap rule;
- three-point weighted-median smoothing;
- `max(8 BPM, 5%)` deviation detection;
- same-direction three-observation run detection;
- `1.5 * observation_hop` run-gap rejection;
- one-second run-span expansion;
- dominant raw-run ordering;
- candidate-local raw self-scoring;
- retained/domain coverage guard at `>= 0.90` and `>= 16` beats;
- compatible jump gate: three constant sections, outer alias consistency,
  same-sign middle deviation, overlap `>= 500 ms`, middle/run tempo ratio
  `<= 1.15`;
- raw-run/no-compatible-jump explicit fallback;
- BeatThis `_aba_support_delta` ordinal rank;
- tie-breaking by higher raw self-score and fingerprint;
- raw-unavailable fallback;
- ramp exclusion from production.

Then run a mechanism evaluation or equivalent script that freezes selected
fingerprints and product statuses before reading weak comparators for:

- `dataset/0/2300685/audio.mp3`
- `dataset/0/618173/audio.mp3`

Only after both mechanism probes pass, run the already exposed pilot80
high/medium non-ambiguous stable/jump 42-row slice with the same source/config
identity and without threshold changes.

Do not run protected holdout100-v2, broad500, or full5050 under this card.

## Guard Check

- no `.osu`, metadata BPM, catalog BPM, manual listening, or weak comparator
  access before selected candidate fingerprints and product statuses are frozen;
- no candidate generator change;
- no raw feature extractor change;
- no evaluator metric, truth-policy, or split change;
- production output remains constant/jump only;
- ramp remains diagnostic-only with production accuracy reported as `null`;
- raw unavailable produces explicit `v2_fallback`;
- no local phase reset and no non-integer section boundary;
- serialized seam discontinuity is at most 5 ms;
- candidate count remains at most 64;
- hard failures are zero in local gates;
- raw-score coverage guard is frozen at retained/domain `>= 0.90` and retained
  beat count `>= 16`;
- selector p90 overhead on pilot80 remains below 5 seconds, excluding one-time
  feature-cache creation;
- exact eight-row pre-freeze diagnostic identities and their identity-set hash
  are recorded in this card before formal execution proceeds;
- no protected holdout100-v2, broad500, or full5050 access.

## Qualitative Check

Inspect frozen per-row and per-candidate tables for the two exposed probes and
the eight-row diagnostic manifest after prediction freeze:

- `2300685` should show no compatible raw-run/jump pair and should route to the
  constant lane rather than accept false short jumps.
- `618173` should show the known negative raw run and should route to the
  paired-jump lane.
- Eligible `618173` jump candidates should show outer alias consistency, same
  middle direction, at least 500 ms run overlap, middle/run tempo ratio at most
  1.15, valid retained/domain coverage, and ordinal rank competitiveness.
- BeatThis `_aba_support_delta` may reorder candidates inside the jump lane but
  must not decide whether a raw run exists.
- Half/double aliases must not truncate the raw-score comparison domain for
  direct-tempo candidates.

## Positive Signal

- all synthetic tests pass;
- exposed `2300685` selects production constant or explicit fallback without a
  false production jump;
- exposed `618173` selects production candidate index 33 or an equivalently
  compatible production short-jump candidate retaining the 2--8 second
  excursion;
- pilot80 high/medium non-ambiguous stable/jump slice improves jump-boundary
  recall or direct left/right tempo-pair accuracy over Experiment 012's frozen
  behavior without increasing stable false-boundary song rate;
- product status includes zero `hard_failure`;
- raw unavailable paths produce `v2_fallback` rather than accepted v3 output.

## Negative Signal

- no-compatible-run stable rows accept false production jumps;
- exposed `618173` contains a viable compatible jump candidate but Exp013 still
  falls back or selects constant;
- the selector depends on changing `minimum_local_strength=0.04`,
  rational-canonicalization scope, 1.25% snap threshold, deviation threshold
  `max(8 BPM, 5%)`, run length, gap rule, one-second expansion, 500 ms overlap,
  1.15 tempo-ratio gate, coverage threshold, or ordinal rank formula after
  pilot42 exposure;
- BeatThis evidence again becomes the run-presence authority;
- raw self-score unavailability is silently converted into a v3 candidate;
- ramp candidates enter production selection;
- runtime exceeds the guard or candidate count exceeds 64.

## Kill Criteria

Kill this selector mutation if any of these occur:

- it cannot pass both exposed mechanism probes without changing candidates,
  raw features, evaluator, split, or frozen thresholds;
- two independent local mutations of raw-run compatibility gates still either
  add stable false jumps or miss the exposed short-jump probe;
- formal pilot80 high/medium non-ambiguous stable/jump slice shows no jump value
  improvement while increasing stable false-boundary song rate;
- any inference/evaluation separation, raw-unavailable fallback, phase
  continuity, seam, coverage, candidate-count, production-scope, or
  protected-data guard fails.

## Expected Failure Modes

- Raw observations may fire on percussion fills or section texture changes that
  are not true tempo jumps.
- Rational alias canonicalization may over-snap complex ratios or under-snap
  true musical aliases outside `p, q <= 5`.
- Weak rhythmic material may make raw self-ranks flat, causing fallback rather
  than accepted v3.
- BeatThis `_aba_support_delta` may be noisy inside the lane and choose the
  wrong eligible jump among several raw-supported candidates.
- The raw-run/no-compatible-jump fallback may be conservative and under-accept
  subtle jumps.
- Time-overlap or middle/run tempo-ratio gating may reject a valid jump if
  candidate boundaries or raw-run centers are shifted.
- Existing candidate generation may omit the right candidate on pilot80 rows;
  this card does not change proposal coverage.

## Expected Runtime / Runtime Budget

- Unit/synthetic tests: under 10 seconds.
- Two exposed mechanism probes with existing feature caches: under 30 seconds.
- Pilot80 high/medium non-ambiguous stable/jump 42-row slice with existing raw
  caches: under 10 minutes.
- Hard stop: any single row exceeding 180 seconds, p90 selector overhead above
  5 seconds, or repeated hard/integrity failure.

## Confounders

- Weak `.osu` redlines can be editorial, sparse, aliased, or inconsistent with
  audio; they are evaluation comparators, not inference truth.
- The eight-row diagnostic is already exposed development evidence and cannot
  be claimed as fresh aggregate performance.
- The exposed `618173` probe is a mechanism row, not a performance estimate.
- Raw local observations and BeatThis local observations may disagree because
  they are different evidence families, not because one is globally true.
- Local strength values are feature-dependent; the `0.04` floor is inherited
  from the generator and must not be optimized on pilot42.
- Pilot80 high/medium stable/jump rows are already exposed development data and
  cannot support fresh-holdout claims.

## Frozen Provenance Requirements

Before any formal pilot42 run, record:

- source hash for this Experiment Card;
- git diff/source identity for `audio_evidence.py`, `tempo_track.py`, and
  relevant tests;
- raw feature extractor version/config identity;
- candidate generator source/config identity;
- selector config identity, including inherited `minimum_local_strength=0.04`,
  small-rational canonicalization `p, q <= 5`, 1.25% primary snap, deviation
  threshold `max(8 BPM, 5%)`, three-point weighted median, three-observation run
  requirement, `1.5 * observation_hop` gap rule, one-second run-span expansion,
  dominant run ordering, compatibility gates, coverage guard
  `retained/domain >= 0.90` and `retained_beat_count >= 16`, lane ranking
  formula, BeatThis `_aba_support_delta` rank definition, tie-break order, raw
  unavailable fallback, and ramp exclusion;
- exposure manifest listing synthetic fixtures, `2300685`, `618173`, the exact
  eight-row pre-freeze diagnostic identities, and then pilot80 high/medium
  non-ambiguous stable/jump identities if the run reaches that stage;
- per-row prediction fingerprint frozen before weak comparator access.

## Result Interpretation Plan

- Positive result would suggest: candidate representation and stricter raw-run
  evidence routing are sufficient for current short-jump rows when viable
  candidates exist.
- Negative result would suggest: either raw-run compatibility gates are not
  reliable enough for stable-vs-jump selection, or candidate generation/ranking
  inside the jump lane is the limiting factor.
- Ambiguous result would require: separating raw-run detection failure,
  candidate-presence failure, and lane-rank failure in a new smaller card
  without opening protected holdout100-v2.
- Human owner decides: whether a positive formal pilot42 result is strong enough
  to freeze and request the next sealed stage under the goal objective.
- Next-loop action if positive: write an Experiment 013 result log, freeze
  source/config/provenance, and proceed only to the next authorized data layer.
- Next-loop action if negative: `KILL` or `MUTATE` this selector in a new card;
  do not tune on protected data.
- Next-loop action if ambiguous: add a focused diagnostic card separating
  raw-run detection, candidate presence, and lane ranking.

## Result Log Template

- Experiment: Timing v3 Experiment 013 raw-run ordinal production selector
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
- Frozen thresholds:
- Coverage diagnostics:
- Compatibility diagnostics:
- Exposure manifest:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Result Log: Formal Exposed Mechanism2

- Experiment: Timing v3 Experiment 013 raw-run ordinal production selector
- Date: 2026-08-14 Asia/Shanghai
- Dataset slice: two already exposed mechanism rows from
  `artifacts/reports/timing/timing_v3_pilot_rows_80_v1.jsonl`:
  `dataset/0/2300685/audio.mp3` and `dataset/0/618173/audio.mp3`.
- Baseline / comparator: post-frozen representative redline weak oracle and
  paired timing-v2 pilot80 baseline JSONL; v2 metrics were read only after the
  Exp013 frozen inference SHA for each row.
- Source/config/provenance: frozen source snapshot in this card; input pilot
  JSONL SHA-256
  `cdb5e2af87d99c8af3bbff71a0985bb490fdeed169d10cb0a64b17a8ca0296d7`;
  paired v2 baseline JSONL SHA-256
  `5d9bb3c50f4173b4bde60cdd1dd30a152565bd89934f2e6cb2407dd8910075a7`.
- Command:

```text
.venv/bin/python - <<'PY'
from pathlib import Path
from pulsefield_model.timing.evaluation.exp013_pilot import run_exp013_pilot
# explicit cache_audio_key filter for 2300685 and 618173 from pilot80 rows
run_exp013_pilot(
    pilot_jsonl_path=Path('artifacts/reports/timing/timing_v3_pilot_rows_80_v1.jsonl'),
    baseline_v2_jsonl_path=Path('artifacts/reports/timing/timing_v3_v2_baseline_pilot80_v1.jsonl'),
    output_jsonl_path=Path('artifacts/reports/timing/timing_v3_exp013_mechanism2_v1.jsonl'),
    summary_json_path=Path('artifacts/reports/timing/timing_v3_exp013_mechanism2_v1_summary.json'),
    explicit_cache_audio_keys=[...],
)
PY
```

- Output artifacts:
  `artifacts/reports/timing/timing_v3_exp013_mechanism2_v1.jsonl` and
  `artifacts/reports/timing/timing_v3_exp013_mechanism2_v1_summary.json`.
- Runtime: total `1.889 s`; row p90 `1.086 s`; max row `1.122 s`.
- Product status counts: `v3_accepted=2`, `v2_fallback=0`,
  `hard_failure=0`; maximum seam error `0.0 ms`.
- Stable mechanism row `2300685`: selected constant lane, candidate index `0`,
  fingerprint
  `01ec70fb9a990185df753d09559e1f3c97a87b98f6fe0af82bc0c09116ffffc5`;
  frozen inference SHA
  `02dd799c836a486bf4726c11310911bd6fe23269d64f47cb41c7944c03c5bf8e`;
  weak phase mean/p90/max all `31.0 ms`; direct and alias BPM coverage `1.0`;
  constant exact hit `true`; paired v2 baseline phase p90 `120.0 ms`.
- Short-jump mechanism row `618173`: selected paired-jump lane, candidate index
  `33`, fingerprint
  `9c71e8edd4b4d7bf488b43b09ffff9ba85e12929b02848fc2f387e9a8d2f398a`;
  frozen inference SHA
  `85b01958f6b9a1bd826797c5044127ada63abfb31081461488cfde21098efd89`;
  raw run direction `down`, median BPM `158.426`, weighted median delta
  `-16.574 BPM`, expanded span `53012.5..58012.5 ms`, four observations;
  weak phase mean `20.37 ms`, p90 `40.93 ms`, max `57.83 ms`; alias BPM
  coverage `0.988`; endpoint drift `85.64 ms`; max-prefix drift `99.57 ms`;
  weak boundary precision/recall `0.0`; paired v2 baseline phase p90 `9.0 ms`.
- Positive signal observed: the frozen selector accepted the stable probe as a
  constant and recovered the short-jump probe as the intended three-section
  jump without seam error or fallback.
- Negative/risk signal observed: on `618173`, the accepted v3 jump improves the
  intended mechanism but does not beat paired v2 on weak phase p90, and the
  weak redline boundary matcher records no exact boundary hit.  This does not
  fail the mechanism gate, but it raises the probability that pilot42 will fail
  the value gate.
- Interpretation: mechanism2 is positive for source/candidate/selector
  correctness and authorizes the already exposed pilot42 run under the frozen
  rule.  It is not a production performance result and does not authorize
  holdout100-v2, broad500, or full5050.
- Recommended next step: run the formal already exposed pilot80 high/medium
  non-ambiguous stable/jump slice (expected 42 rows) with the same frozen
  source/config and record a separate result section.

## Result Log: Formal Exposed Pilot42

- Experiment: Timing v3 Experiment 013 raw-run ordinal production selector.
- Date: 2026-08-14 Asia/Shanghai.
- Dataset slice: all high/medium-confidence, non-ambiguous `stable` and
  `jump_candidate` rows in the already exposed pilot80 source: `42` unique
  audio identities (`22` stable, `20` jump).  No holdout100-v2, broad500, or
  full5050 row was opened.
- Frozen inputs: pilot JSONL SHA-256
  `cdb5e2af87d99c8af3bbff71a0985bb490fdeed169d10cb0a64b17a8ca0296d7`;
  paired timing-v2 baseline JSONL SHA-256
  `5d9bb3c50f4173b4bde60cdd1dd30a152565bd89934f2e6cb2407dd8910075a7`;
  source/config hashes are the frozen snapshot above.
- Command:

```text
.venv/bin/python -m pulsefield_model.timing.evaluation.exp013_pilot \
  --pilot-jsonl artifacts/reports/timing/timing_v3_pilot_rows_80_v1.jsonl \
  --baseline-v2-jsonl artifacts/reports/timing/timing_v3_v2_baseline_pilot80_v1.jsonl \
  --output-jsonl artifacts/reports/timing/timing_v3_exp013_pilot42_v1.jsonl \
  --summary-json artifacts/reports/timing/timing_v3_exp013_pilot42_v1_summary.json \
  --repo-root /Users/l/projects/Pulsefield-model
```

- Output artifact identities: result JSONL SHA-256
  `574a2affc87ac555636a8ebd261ea5bd88ea2346bdb7f29dafd59e93c60f22b0`;
  summary JSON SHA-256
  `a27b68e4b06c546bcedb40babb29cdd3ba70f798243be04899867adb856dd2fc`.
- Integrity and runtime: `hard_failure=0`; maximum accepted seam error
  `0.0 ms`; total wall time `100.02 s`; per-row p50 `1.97 s`, p90 `3.58 s`,
  max `8.70 s`.  Mel source was existing cache for `31/42` and in-memory decode
  for `11/42`, so row runtime includes missing-cache decode on those rows.
- Product coverage: `v3_accepted=17/42` (`40.48%`), `v2_fallback=25/42`
  (`59.52%`).  Every fallback reason was
  `no_production_eligible_jump_for_raw_run`.
- Stable stratum: accepted `13/22`, fallback `9/22`.  Of the accepted rows,
  `12` were constant and one was a false jump.  Weak class was correct on
  `12/13`; frozen constant-exact hit was `11/13`.  Accepted-stable mean phase
  was `30.53 ms`, mean p90 phase `40.27 ms`, and mean direct BPM coverage
  `0.9967`.  This good conditional quality does not compensate for `9/22`
  false raw-run fallbacks.
- Jump stratum: accepted only `4/20`, fallback `16/20`.  Two accepted rows were
  constants and two were jumps; weak class was correct on `2/4`, and frozen
  jump-exact hit was `0/4`.  Among accepted jumps as a whole, boundary recall
  averaged `0.0625` and boundary precision over rows with predicted boundaries
  averaged `0.25`; mean p90 phase was `118.69 ms`, direct BPM coverage
  `0.2470`, and alias BPM coverage `0.4865`.
- Conditional paired signal: on the six accepted stable rows with usable paired
  timing-v2 metrics, Exp013 improved mean and p90 phase and direct BPM coverage.
  On the two accepted jump rows with paired timing-v2 metrics, Exp013 regressed
  mean phase by about `36%`, p90 phase by about `34%`, and direct BPM coverage
  by `0.216`.  These small paired denominators are diagnostic only.
- Reporting correction: the auto-summary's endpoint-drift mean is signed and
  must not be read as magnitude.  Post-hoc audit uses absolute drift; this does
  not affect any frozen inference result or the negative decision.
- Positive signal observed: source/inference separation held, the mechanism
  row remained recovered, accepted stable constants were usually strong, all
  seams were exact, runtime met the five-second p90 selector target, and no row
  failed execution.
- Negative signal observed: product fallback was `59.52%`, jump acceptance was
  only `20%`, `9/22` stable controls were spuriously treated as having a raw
  deviation run, one accepted stable row became a false jump, and no accepted
  jump met the frozen exact criterion.
- Kill criteria triggered: yes.  The formal pilot42 shows no general jump value
  improvement and introduces unacceptable stable false-run/fallback behavior.
  This is a negative result under the card; it cannot authorize protected
  holdout100-v2, broad500, full5050, or default-path integration.
- Failure attribution: Exp013 solved one candidate-ranking mechanism but not
  general section inference.  The dominant current limits are (a) raw
  autocorrelation-run false positives on stable material and (b) absence of a
  compatible short-jump proposal on most exposed jump rows.  Threshold tuning
  on these 42 identities is prohibited; the next attempt requires a new card
  with a mechanism-distinct proposal/evidence design and a restarted synthetic
  gate.
- Interpretation: `NEGATIVE` / `MUTATE`.  Timing v3 source is now a tested
  research module, not a promoted production algorithm.
- Recommended next step: write a new card that separates candidate-presence
  failure from raw-run specificity using synthetic counterexamples and only
  already exposed identities.  Keep the protected data layers sealed.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes, for source/synthetic tests, the
  two already exposed mechanism probes, the exact already exposed eight-row
  diagnostic manifest, and already exposed pilot80 high/medium non-ambiguous
  stable/jump rows only
- Closed loop complete: yes
- Remaining ambiguity: whether the stricter raw-run compatibility gate is too
  conservative on pilot42 jump rows; this must be measured under the frozen
  rule, not tuned on protected data.

## Next-Loop Action

- If positive: implement or keep the selector disabled-by-default, write the
  result log, and move only to the next authorized data layer.
- If negative: kill or mutate this selector family in a new Experiment Card.
- If ambiguous: add a focused diagnostic card that separates raw-run detection,
  candidate absence, and lane-rank failure.
