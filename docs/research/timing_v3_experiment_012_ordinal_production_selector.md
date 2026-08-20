# Timing v3 Experiment 012: Ordinal production selector

Status: negative at pre-execution mechanism replay

## Mode

- Mode: planner
- Route: TEST
- Source idea: Experiment 011 showed that mixing absolute constant raw score
  with jump-vs-collapse gain is dimensionally incoherent. Keep Experiment 010
  candidates and raw features fixed, but replace the production selector with a
  gated ordinal rule that never compares unlike numeric scales.
- Acceptance source, if any: goal objective, Experiment 010 `MUTATE` rule, and
  Experiment 011 negative result.
- Source snapshot / evidence grade: strong local mechanism evidence from
  synthetic/source inspection plus the already exposed `2300685` stable probe
  and `618173` short-jump probe; no pilot80 batch, protected holdout, broad500,
  or full5050 evidence used to choose this rule.

## Hypothesis

A production-only ordinal selector can choose between constant and short-jump
Timing v3 candidates without score-scale leakage by:

1. scoring every candidate over its own full raw-audio coverage;
2. using BeatThis local deviation runs only as directional and time-overlap
   gates;
3. ranking eligible candidates by ordinal ranks rather than mixed absolute
   scores; and
4. explicitly falling back to timing v2 when raw evidence or production
   eligibility is insufficient.

## Root Objective

Repair the Timing v3 top-1 selection failure while preserving the Phase 1
product contract: accepted production outputs are phase-continuous constant or
jump sections on one absolute integer beat axis; evidence gaps produce explicit
`v2_fallback`; diagnostic ramps cannot be counted as production success.

## Goal Decomposition

- Subgoal 1: remove cross-alias common-domain truncation by using
  candidate-local raw self-scores over each candidate's own physical beat times.
- Subgoal 2: prevent stable songs from gaining false short sections by allowing
  production jumps only when BeatThis local observations contain a coherent
  same-direction deviation run.
- Subgoal 3: prevent jump songs from collapsing to constants by requiring
  eligible jump candidates to direction-match and time-overlap the detected ABA
  run, then ranking only within that lane.
- Subgoal 4: preserve product safety: raw unavailable means fallback, not
  candidate zero; ramp and non-constant/non-jump candidates remain diagnostics;
  no weak comparator or `.osu` data enters inference.

## Candidate Variants

- Variant A: retune Experiment 011's scalar formula so jump gain receives a
  large multiplier.
- Variant B: keep the Experiment 009/010 common-domain scorer and add an
  endpoint-duration filter.
- Variant C: build a continuous weighted score from raw self-score, BeatThis
  deviation magnitude, duration, local strength, and overlap.
- Variant D: use a gated ordinal production selector:
  - raw candidate self-rank is computed independently per candidate;
  - BeatThis ABA support is an ordinal rank within the active lane;
  - no-run rows use the constant lane;
  - rows with an eligible deviation run use the paired jump
    direction/time-overlap lane;
  - lane score is `raw_self_rank + beatthis_aba_support_rank`;
  - ties break by higher raw self-score, then canonical candidate fingerprint.

## Local Verification Matrix

- Variant A:
  - Local check: compare exposed `618173` direct short-jump candidate against
    stable constants using only pre-pilot evidence.
  - Fail condition: any coefficient large enough to fix `618173` creates an
    arbitrary cross-class scale choice rather than a defensible selector.
- Variant B:
  - Local check: synthetic constant, synthetic short ABA jump, exposed
    `2300685`, and exposed `618173`.
  - Fail condition: aliases can still define or shrink the common scoring
    domain, or the repair depends on a tight terminal endpoint threshold rather
    than coverage evidence.
- Variant C:
  - Local check: inspect whether continuous strength/duration/overlap weights
    can be frozen before pilot80 without tuning on real aggregate metrics.
  - Fail condition: the rule requires choosing fusion weights not justified by
    source behavior or the two exposed probes.
- Variant D:
  - Local check order:
    1. synthetic constant 200 BPM;
    2. synthetic short ABA jump;
    3. synthetic time-linear ramp as diagnostic-only non-production output;
    4. exposed stable probe `dataset/0/2300685/audio.mp3`;
    5. exposed short-jump probe `dataset/0/618173/audio.mp3`;
    6. if and only if all prior gates pass, already exposed pilot80
       high/medium stable/jump rows, expected count 42.
  - Pass condition: constant probe selects a production constant or explicit
    fallback without false jump; short-jump probe selects a production jump whose
    middle section retains the exposed 2--8 second excursion; ramp remains
    diagnostic and cannot produce a production ramp success.

## Selected Variant

- Selected: Variant D, gated ordinal production selector.
- Rejected:
  - Variant A repeats Experiment 011's scale mismatch and invites coefficient
    tuning.
  - Variant B can reduce one exposed alias failure but does not repair the
    scoring objective.
  - Variant C creates too many continuous weights for a pre-holdout freeze.
- Why this is the smallest useful test: it changes only production selection.
  Candidate generation, raw feature extraction, analytic curve representation,
  evaluator metrics, pilot split, fallback contract, and ramp diagnostic policy
  stay fixed.

## Selection Pressure

- Primary pressure: improve top-1 production selection when a viable constant or
  short-jump candidate is already present.
- Guard pressure: no accepted candidate may violate phase continuity, integer
  beat section boundaries, seam serialization continuity, candidate-count cap,
  raw-evidence availability, or inference/evaluation separation.
- Runtime pressure: selector overhead must remain small relative to existing
  candidate generation and raw scoring; the 10-minute algorithmic target remains
  under 5 seconds excluding one-time audio decode/mel cache creation.
- Kill pressure: if the ordinal selector cannot pass synthetic gates plus both
  exposed mechanism probes without changing candidates, features, evaluator, or
  thresholds after pilot80 exposure, stop and mutate.

## Research Question

Can a lane-gated ordinal selector turn the existing Timing v3 candidates and raw
audio self-scores into a safer production decision than scalar reranking, while
preserving stable specificity and short-jump recall?

## Closest Analogies / Novelty Layer

- Closest analogies: gated model selection, non-parametric rank aggregation,
  change-point proposal filtering by direction/time overlap, and onset-strength
  beat-grid fitness.
- Relevant taxonomy bucket: bounded deterministic candidate selection, not model
  training.
- Novelty layer, if any: no novelty claim; this is engineering variation in a
  Timing v3 production selector.
- Representation novelty vs engineering variation: representation remains the
  existing analytic constant/jump curve; this card changes only selection.

## Minimal Change

Implement an Experiment 012 selector behind a disabled-by-default Timing v3
path. The frozen behavior is:

1. Keep the Experiment 010 candidate generator unchanged. Candidate count remains
   at most 64.
2. Keep the raw-audio feature family unchanged: deterministic 16 kHz audio,
   80-bin log-mel, 10 ms hop, four-band positive spectral flux, beat-vs-half-beat
   support, and robust local aggregation.
3. Compute `raw_self_score(c)` for each candidate over that candidate's own
   scoreable physical beat times. Do not intersect beat indices across unrelated
   tempo aliases.
4. Run production eligibility before ranking:
   - constant candidates are production-eligible only in the no-run constant
     lane;
   - jump candidates are production-eligible only in the paired jump lane;
   - diagnostic ramp candidates and other non-constant/non-jump structures are
     never production-eligible in Phase 1;
   - if raw self-scoring is unavailable, return explicit `v2_fallback`.
5. Derive BeatThis local observation runs using the existing generator local
   observations:
   - normalize observations to the primary base alias family before measuring
     deviation;
   - smooth observation BPMs with a three-point weighted median;
   - mark deviation when `abs(smoothed_bpm - base_bpm) >= max(6 BPM,
     0.05 * base_bpm)`;
   - require same deviation direction for at least three observations;
   - require adjacent observation gaps no larger than `1.5 * observation_hop`;
   - inherit the generator's existing `minimum_local_strength=0.04`; do not add
     a new `strength>=0.15` threshold.
6. If multiple deviation runs are present, choose the dominant run by:
   1. larger `abs(weighted_median_delta_bpm)`;
   2. longer duration;
   3. larger summed local strength;
   4. earlier start time.
   Do not add continuous fusion weights.
7. Apply a terminal/retained coverage guard before ranking:
   - prefer retained scoreable beat coverage and scoreable audio span coverage;
   - do not rely on a tight endpoint-time equality check alone;
   - reject candidates whose valid raw-score support is too sparse to compare
     against other candidates in the same lane.
8. If no deviation run survives, use the constant lane:
   - rank production-eligible constant candidates by raw self-rank and constant
     BeatThis support rank;
   - ties break by higher raw self-score, then canonical fingerprint;
   - if no constant candidate is eligible, return `v2_fallback`.
9. If a deviation run survives, use the paired jump lane:
   - eligible jumps must have exactly three constant sections;
   - outer sections must be alias-consistent with the primary base family;
   - the middle section must deviate in the same direction as the selected run;
   - the middle interval must time-overlap the selected run;
   - rank within the lane by
     `raw_self_rank + beatthis_aba_support_rank`;
   - ties break by higher raw self-score, then canonical fingerprint;
   - if no jump candidate is eligible, return `v2_fallback` rather than forcing
     a constant through a contradictory run.

The `minimum_local_strength=0.04` decision is frozen after inspecting only the
already exposed `618173` probe, where relevant raw-observation strengths were
approximately `0.04--0.104`; it is frozen before any pilot80 aggregate run.

## Files Likely to Change

If this card is executed, expected code changes are limited to:

- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- `tests/timing/test_timing_v3_audio_evidence.py`
- `tests/timing/test_timing_v3_tempo_track.py`
- a result log under `docs/research/`

This planning step itself adds only:

- `docs/research/timing_v3_experiment_012_ordinal_production_selector.md`

## Read-Only Context Files

- `/Users/l/.codex/attachments/97bd173a-3590-4524-8b1f-f7a90c5e0223/goal-objective.md`
- `AGENTS.md`
- `README.md`
- `docs/research/timing_v3_experiment_010_real_audio_short_jump.md`
- `docs/research/timing_v3_experiment_011_candidate_local_audio_gain.md`
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
4. if all gates pass, already exposed pilot80 high/medium stable/jump rows,
   expected count 42.

Protected holdout100-v2, broad500, and full5050 are not opened by this card.

## Baseline / Comparator

- Baseline ranking: Experiment 010 generator plus Experiment 009/010
  common-domain raw scorer.
- Negative comparator: Experiment 011 candidate-local raw-gain formula, which
  failed before pilot80 because constants and jumps were scored on unlike
  numeric scales.
- Product comparator: timing v2 remains the fallback comparator; fallback is not
  counted as v3 success.
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
- selected candidate rank before/after ordinal selection;
- raw self-score rank, BeatThis ABA support rank, tie-break reason, and fallback
  reason;
- candidate count, selector runtime, raw scoring runtime, and end-to-end
  generation plus selection runtime;
- production status counts: `v3_accepted`, `v2_fallback`, `hard_failure`;
- diagnostic ramp rows reported with ramp accuracy as `null`.

## Verify Command / Evaluation Procedure

First run focused unit/synthetic tests covering raw self-score availability,
lane eligibility, coverage guard behavior, no-run constant selection, paired
jump direction/time-overlap selection, tie-breaking, fallback, and ramp
exclusion.

Then run a mechanism evaluation or equivalent script that freezes selected
fingerprints before reading weak comparators for:

- `dataset/0/2300685/audio.mp3`
- `dataset/0/618173/audio.mp3`

Only after both mechanism probes pass, run the already exposed pilot80
high/medium stable/jump 42-row slice with the same source/config identity and
without threshold changes.

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
- selector p90 overhead on pilot80 remains below 5 seconds, excluding one-time
  feature-cache creation;
- no protected holdout100-v2, broad500, or full5050 access.

## Qualitative Check

Inspect the frozen per-candidate table for the two exposed probes after
prediction freeze:

- `2300685` should show no coherent deviation run and should route to the
  constant lane rather than accept false short jumps.
- `618173` should show a coherent same-direction deviation run with enough local
  observations under the inherited `minimum_local_strength=0.04`, and eligible
  jump candidates should be ranked within the paired jump lane.
- Half/double aliases must not truncate the raw-score comparison domain for
  direct-tempo candidates.

## Positive Signal

- all synthetic tests pass;
- exposed `2300685` selects production constant or explicit fallback without
  false jump;
- exposed `618173` selects a production short-jump candidate retaining the
  2--8 second excursion;
- pilot80 high/medium stable/jump slice improves jump-boundary recall or direct
  left/right tempo-pair accuracy over the Experiment 010 selector without
  increasing stable false-boundary song rate;
- product status includes zero `hard_failure`.

## Negative Signal

- no-run stable rows accept false production jumps;
- exposed `618173` contains a viable jump candidate but ordinal selection still
  falls back or selects constant;
- the selector depends on changing the inherited `minimum_local_strength=0.04`
  after pilot80 exposure;
- raw self-score unavailability is silently converted into a v3 candidate;
- ramp candidates enter production selection;
- runtime exceeds the guard or candidate count exceeds 64.

## Kill Criteria

Kill this selector family if any of these occur:

- it cannot pass both exposed mechanism probes without changing candidates,
  raw features, evaluator, split, or frozen thresholds;
- two independent local mutations of the ordinal lane gates still either add
  stable false jumps or miss the exposed short-jump probe;
- pilot80 high/medium stable/jump slice shows no jump value improvement while
  increasing stable false-boundary song rate;
- any inference/evaluation separation, raw-unavailable fallback, phase
  continuity, seam, candidate-count, or protected-data guard fails.

## Expected Failure Modes

- BeatThis local observations may miss real short jumps when raw audio supports
  them but framewise tempo estimates do not form a three-point same-direction
  run.
- Weak rhythmic material may make raw self-ranks flat, causing fallback rather
  than accepted v3.
- The no-run constant lane may be conservative and under-accept subtle jumps.
- Time-overlap gating may reject a valid jump if candidate boundaries are
  shifted relative to BeatThis observation windows.
- Existing candidate generation may omit the right candidate on pilot80 rows;
  this card does not change proposal coverage.

## Confounders

- Weak `.osu` redlines can be editorial, sparse, aliased, or inconsistent with
  audio; they are evaluation comparators, not inference truth.
- The exposed `618173` probe is a mechanism row, not a performance estimate.
- Local strength values are feature-dependent; the `0.04` floor is inherited
  from the generator, not newly optimized for pilot80.
- Pilot80 high/medium stable/jump rows are already exposed development data and
  cannot support fresh-holdout claims.

## Expected Runtime / Runtime Budget

- Unit/synthetic tests: under 10 seconds.
- Two exposed mechanism probes with existing feature caches: under 30 seconds.
- Pilot80 high/medium stable/jump 42-row slice with existing raw caches: under
  10 minutes.
- Hard stop: any single row exceeding 180 seconds, p90 selector overhead above
  5 seconds, or repeated hard/integrity failure.

## Frozen Provenance Requirements

Before any pilot80 run, record:

- source hash for this Experiment Card;
- git diff/source identity for `audio_evidence.py`, `tempo_track.py`, and
  relevant tests;
- raw feature extractor version/config identity;
- candidate generator source/config identity;
- selector config identity, including the inherited
  `minimum_local_strength=0.04`, deviation threshold `max(6 BPM, 5%)`,
  three-observation run requirement, `1.5 * observation_hop` gap rule, dominant
  run ordering, lane ranking formula, tie-break order, and ramp exclusion;
- exposure manifest listing synthetic fixtures, `2300685`, `618173`, and then
  the pilot80 high/medium stable/jump identities if the run reaches that stage;
- per-row prediction fingerprint frozen before weak comparator access.

## Result Interpretation Plan

- Positive result would suggest: the main remaining Experiment 010/011 issue is
  production selection, not analytic representation, for stable and short-jump
  rows where a viable candidate exists.
- Negative result would suggest: the selector still lacks reliable evidence for
  stable-vs-jump decisions, or candidate generation is the limiting factor.
- Ambiguous result would require: separating candidate-presence failure from
  selector-ranking failure in a new card without opening protected holdout100-v2.
- Human owner decides: whether a positive pilot80 result is strong enough to
  freeze and request the next sealed stage under the goal objective.
- Next-loop action if positive: write an Experiment 012 result log, freeze
  source/config/provenance, then proceed only to the next authorized stage.
- Next-loop action if negative: `KILL` or `MUTATE` this selector in a new card;
  do not tune on protected data.
- Next-loop action if ambiguous: create a smaller diagnostic card focused on
  candidate presence vs selector rank.

## Result Log Template

- Experiment: Timing v3 Experiment 012 ordinal production selector
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
- Exposure manifest:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes, for source/synthetic tests, the
  two already exposed mechanism probes, and already exposed pilot80 high/medium
  stable/jump rows only
- Closed loop complete: yes
- Remaining ambiguity: whether the ordinal gates are too conservative on
  pilot80; this must be measured under the frozen rule, not tuned on protected
  data.

## Next-Loop Action

- If positive: implement or keep the selector disabled-by-default, write the
  result log, and move only to the next authorized data layer.
- If negative: kill or mutate this selector family in a new Experiment Card.
- If ambiguous: add a focused diagnostic card that separates candidate absence
  from selector failure.

## Result

- Date: 2026-08-13
- Scope: source/synthetic inspection and the already exposed `618173`
  mechanism probe only; no pilot80 batch or protected data was opened.
- Frozen-rule replay: after the specified three-point weighted-median smoothing
  and `max(6 BPM, 5%)` deviation threshold, the BeatThis observations form no
  same-direction run of three points.  The raw-audio observations do form a
  four-point negative run from approximately `54.012--57.012 s`, with smoothed
  values near `161.7, 158.4, 157.9, 157.9 BPM` against the `175 BPM` base.
- Consequence: the card's BeatThis-run gate would route the known exposed
  short-jump mechanism row into the no-run constant lane and therefore fail its
  own mandatory mechanism gate before the ordinal ranking is exercised.
- Interpretation: negative.  No selector source was changed under this card.
  The evidence roles, rather than thresholds or weights, were specified
  incorrectly.
- Next action: `MUTATE` in a new card.  Use raw-audio local observations for
  run presence/direction/time-overlap and retain BeatThis only for relative ABA
  support ranking inside the eligible jump lane.  All other candidates,
  features, thresholds, evaluation policy, and protected-data boundaries stay
  unchanged.

## Novelty Notes

- Closest analogies: ordinal rank aggregation, gated change-point model
  selection, and onset-strength beat-grid scoring.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: no representation novelty;
  this is a production-selection engineering variation over the existing Timing
  v3 analytic candidate representation.
