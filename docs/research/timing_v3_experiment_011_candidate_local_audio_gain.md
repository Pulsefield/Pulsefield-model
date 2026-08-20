# Timing v3 Experiment 011: Candidate-local raw-audio gain reranker

Status: negative at pre-execution local verification

## Mode

- Mode: planner
- Route: TEST
- Source idea: Experiment 010 recalls a near-correct short-jump candidate, but
  the frozen common-domain raw scorer can rank a shorter or aliased candidate
  above it because half-tempo candidates shrink the shared comparison domain.
- Acceptance source, if any: goal objective and Experiment 010 `MUTATE` rule:
  if the correct candidate is present but not selected, change only the
  reranking family in a new card.
- Source snapshot / evidence grade: strong local evidence from synthetic tests
  plus already exposed mechanism probes; no protected holdout evidence.

## Hypothesis

Scoring each timing candidate over its own full physical beat coverage, then
ranking jump candidates by raw-audio structural gain over their deterministic
outer-tempo constant collapse, will select the correct direct-tempo stable or
short-jump candidate without changing the Experiment 010 candidate generator.

## Root Objective

Repair the Experiment 010 ranking failure while preserving the Timing v3 Phase
1 contract: one global integer beat axis, phase-continuous constant/jump
sections, deterministic raw-audio verification, and explicit v2 fallback when
evidence is insufficient.

## Goal Decomposition

- Subgoal 1: remove the cross-candidate common-domain artifact that lets a
  half-tempo candidate shorten the raw-audio comparison to the first half of a
  song.
- Subgoal 2: prefer a short excursion only when it improves local raw-audio
  evidence relative to collapsing that candidate back to its outer tempo.
- Subgoal 3: keep stable songs stable by making inserted short jumps pay for
  themselves against a constant-collapse comparator.
- Subgoal 4: preserve all Timing v3 guards: no local phase reset, no non-integer
  boundary, no `.osu` or metadata inference, at most 64 candidates, and seam
  continuity at or below 5 ms.

## Candidate Variants

- Variant A: keep the Experiment 009/010 common-domain raw score and only adjust
  ranking weights.
- Variant B: score each candidate independently over all of its valid physical
  beat times and rank by absolute local raw score only.
- Variant C: score each candidate independently, then rank by candidate-local
  structural gain over a deterministic outer-tempo constant collapse, with the
  existing generation score used only as a small tie-breaker.
- Variant D: add only an endpoint-duration guard before the current common-domain
  scorer.

## Local Verification Matrix

- Variant A:
  - Verify on the already exposed `dataset/0/618173/audio.mp3` short-jump probe.
  - Fail if half-tempo or otherwise slow candidates still reduce the retained
    domain enough that the selected candidate ignores the 4.17-second excursion.
- Variant B:
  - Verify on synthetic constant 200 BPM, synthetic short ABA jump, and the
    exposed stable probe `dataset/0/2300685/audio.mp3`.
  - Fail if false short jumps outrank the stable direct-tempo candidate because
    absolute onset support is high but the jump adds no comparative value.
- Variant C:
  - Verify on synthetic constant 200 BPM, synthetic short ABA jump, exposed
    stable `dataset/0/2300685/audio.mp3`, and exposed short-jump
    `dataset/0/618173/audio.mp3`.
  - Pass only if the stable probe remains constant/direct-tempo and the
    short-jump probe selects a direct-tempo candidate whose middle excursion is
    retained.
- Variant D:
  - Verify on the same two exposed probes.
  - Fail if it fixes the short-jump row only by filtering known-bad aliases while
    leaving the common-domain scorer unable to compare candidate-local evidence.

## Selected Variant

- Selected: Variant C.
- Rejected:
  - Variant A is the known failure mode.
  - Variant B lacks a stable-song penalty for unnecessary section insertion.
  - Variant D is too narrow: it can remove some aliases but does not repair the
    scoring objective.
- Why this is the smallest useful test: it changes only reranking. Candidate
  generation, analytic representation, evaluator metrics, pilot split, and
  fallback/product gates stay fixed.

## Selection Pressure

- Primary pressure: improve top-1 selection when the correct candidate is
  already present.
- Guard pressure: reject any candidate that breaks phase continuity, integer
  beat sections, max 64 candidates, inference/evaluation separation, or seam
  serialization continuity.
- Runtime pressure: keep candidate generation plus reranking under 5 seconds
  for a 10-minute recording, excluding one-time audio decode/mel cache creation.
- Kill pressure: if the selected variant cannot pass both exposed mechanism
  probes without changing generation or truth/evaluator rules, stop and mutate
  before pilot80.

## Research Question

Can candidate-local raw-audio gain select the correct constant/jump candidate
more reliably than a cross-candidate common-domain raw score, while preserving
stable-song specificity?

## Closest Analogies / Novelty Layer

- Closest analogies: model selection by held-in candidate-local likelihood,
  contrastive scoring against a nested simpler model, onset-strength beat-grid
  fitness, and change-point selection by gain over a no-change baseline.
- Relevant taxonomy bucket: bounded local verification and candidate selection,
  not model training.
- Novelty layer, if any: engineering variation in deterministic reranking for a
  phase-continuous Timing v3 representation.
- Representation novelty vs engineering variation: the representation remains
  the existing analytic constant/jump curve; this experiment only changes the
  ranking objective.

## Minimal Change

Implement a candidate-local scorer used only inside the Timing v3 Experiment
010 candidate rerank path:

1. For each candidate `c`, collect beat-event support over that candidate's own
   valid physical beat times in the raw-audio feature span. Do not intersect beat
   indices across unrelated tempo aliases.
2. Compute `raw_full(c)` with the same deterministic raw-audio feature family
   already used by Experiment 009/010: 16 kHz audio, 80-bin log-mel, 10 ms hop,
   four-band positive spectral flux, beat-vs-half-beat support, and robust
   16-beat-window aggregation.
3. For each non-constant candidate, construct `collapse(c)`: a single constant
   curve with the same first beat/time origin and beat domain as `c`, using the
   weighted median BPM of the first and last sections when both outer sections
   exist, otherwise the longest-section BPM.
4. Rank by the frozen local formula:

   ```text
   candidate_gain(c) = raw_full(c) - raw_full(collapse(c))
   structure_penalty(c) = 0.004 * max(0, section_count(c) - 1)
   generation_tiebreak(c) = 0.02 * clipped_zscore(generation_score(c))
   final_score(c) = candidate_gain(c) - structure_penalty(c) + generation_tiebreak(c)
   ```

   Constant candidates use `candidate_gain(c) = raw_full(c)` and
   `structure_penalty(c) = 0`.
5. Freeze these numbers before pilot80. If local verification shows the formula
   is incoherent, return `MUTATE`; do not tune coefficients on pilot80.

## Files Likely to Change

If this card is executed, expected code changes are limited to:

- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- `tests/timing/test_timing_v3_audio_evidence.py`
- `tests/timing/test_timing_v3_tempo_track.py`
- a result file for this experiment under `docs/research/`

This planning step itself adds only:

- `docs/research/timing_v3_experiment_011_candidate_local_audio_gain.md`

## Read-Only Context Files

- `/Users/l/.codex/attachments/97bd173a-3590-4524-8b1f-f7a90c5e0223/goal-objective.md`
- `docs/research/timing_v3_experiment_010_real_audio_short_jump.md`
- `src/pulsefield_model/timing/v3/analytic_curve.py`
- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- `src/pulsefield_model/timing/evaluation/curve_metrics.py`
- existing Timing v3 tests under `tests/timing/`

## Dataset Slice

Local execution order:

1. synthetic/unit fixtures only;
2. already exposed stable/direct-alias mechanism probe:
   `dataset/0/2300685/audio.mp3`;
3. already exposed short-jump mechanism probe:
   `dataset/0/618173/audio.mp3`;
4. if and only if all local gates pass, already exposed pilot80 stable/jump
   rows, prioritizing high/medium non-ambiguous rows.

Protected holdout100-v2, broad500, and full5050 are not opened by this card.

## Baseline / Comparator

- Baseline ranking: Experiment 010 candidate generator plus the frozen
  Experiment 009 common-domain raw-audio score.
- Candidate-presence comparator: the best near-truth candidate already present
  in the Experiment 010 candidate list for `dataset/0/618173/audio.mp3`.
- Product comparator: timing v2 remains the fallback comparator; this card does
  not change fallback routing.
- Weak evaluation comparator: `.osu` redlines and object-derived evidence are
  read only after each prediction fingerprint is frozen.

## Primary Metric

Mechanism-gate top-1 correctness:

- stable probe: selected candidate is direct-tempo constant with no false
  boundary;
- short-jump probe: selected candidate retains the 2--8 second middle section,
  middle BPM is direct-tempo, and both weak-comparator boundaries are within
  750 ms after prediction freeze.

## Secondary Metric

- direct BPM coverage and alias-aware BPM coverage;
- stable false-boundary song rate and false-boundary count;
- jump boundary precision/recall at 500 ms and 1,000 ms;
- direct left/right tempo-pair accuracy for matched boundaries;
- signed initial phase, mean phase, p90 phase, endpoint drift, and max-prefix
  drift;
- selected candidate rank before/after rerank;
- candidate count, rerank runtime, and end-to-end generation plus rerank runtime.

## Verify Command / Evaluation Procedure

First run the focused unit/synthetic tests for the changed scorer and tempo
track path. Then run a mechanism script or equivalent test that freezes the
selected candidate fingerprint before loading weak comparators for:

- `dataset/0/2300685/audio.mp3`
- `dataset/0/618173/audio.mp3`

Only after both pass, run the already exposed pilot80 stable/jump slice with
the same source and formula. Do not run protected holdout100-v2.

## Guard Check

- no `.osu`, metadata BPM, catalog BPM, manual listening, or weak comparator
  access before selected candidate fingerprints are frozen;
- no new candidate generator family;
- no evaluator metric or truth-policy change;
- no local phase reset;
- every section boundary remains an integer beat;
- serialized seam discontinuity is at most 5 ms;
- candidate count remains at most 64;
- hard failures are zero in local gates;
- p90 extra rerank latency on pilot80 remains below 5 seconds, excluding
  one-time feature-cache creation.

## Qualitative Check

Inspect the per-candidate score table for both exposed probes after prediction
freeze:

- the stable probe should show false-jump candidates losing to their own
  constant collapse;
- the short-jump probe should show the retained excursion gaining over the
  outer-tempo collapse;
- half-tempo candidates must no longer define or truncate the comparison domain
  for direct-tempo candidates.

## Positive Signal

- synthetic constant and synthetic short-jump tests pass;
- exposed stable probe selects direct constant timing;
- exposed short-jump probe selects or ties for top-1 a direct-tempo short
  excursion candidate with both weak-comparator boundaries within 750 ms;
- pilot80 stable/jump rows improve jump-boundary recall or left/right direct
  tempo-pair accuracy without increasing stable false-boundary song rate.

## Negative Signal

- the correct short-jump candidate is present but still not top-ranked;
- stable rows gain extra false short sections;
- candidate-local raw score overfits onset density and ignores tempo structure;
- runtime exceeds the 5-second additional-latency guard;
- any phase-continuity or inference/evaluation separation guard fails.

## Kill Criteria

Kill this reranking family if either of these occurs:

- it cannot pass both exposed mechanism probes without changing candidate
  generation, evaluator definitions, or coefficients after pilot80 exposure;
- two independent local mutations of the candidate-local gain formula still
  fail to distinguish the retained short excursion from its outer-tempo
  collapse.

## Expected Failure Modes

- `collapse(c)` may be too weak if the outer sections are noisy or not truly the
  same tempo family.
- Per-candidate full coverage may favor denser direct-tempo grids on noisy audio
  unless beat-vs-half-beat support remains discriminative.
- A real tempo change near the end of a recording may be penalized if coverage
  or collapse construction implicitly assumes a stable outro.
- Generation-score tie-breaking may preserve a bad BeatThis prior when raw gain
  is nearly flat.
- Pilot80 may show good mechanism behavior on exposed probes but ambiguous weak
  oracle results on low-confidence rows.

## Confounders

- Weak `.osu` redlines may be sparse, aliased, or editorial rather than exact
  timing truth.
- The exposed short-jump row is a single mechanism probe; passing it does not
  prove broad jump performance.
- Raw-audio onset flux can be weak on sustained, noisy, or highly syncopated
  music.
- Existing candidate generation may still omit the correct candidate on other
  rows; this card only handles selection when a viable candidate is present.

## Expected Runtime / Runtime Budget

- Synthetic/unit tests: under 5 seconds.
- Two exposed mechanism probes with existing feature caches: under 30 seconds
  total.
- Pilot80 stable/jump slice, if reached: under 10 minutes with cached raw-audio
  features.
- Hard stop: any single row exceeding 180 seconds or p90 extra rerank latency
  exceeding 5 seconds on pilot80.

## Result Interpretation Plan

- Positive result would suggest: the main Experiment 010 failure is ranking, not
  representation or candidate generation, for the exposed short-jump mechanism.
- Negative result would suggest: candidate-local raw-audio gain is not enough;
  either raw evidence is too weak or the candidate set requires a different
  proposal objective.
- Ambiguous result would require: a new card that separates stable false-boundary
  control from jump recall, without opening holdout100-v2.
- Human owner decides: whether a positive pilot80 result is worth freezing for
  fresh holdout100-v2.
- Next-loop action if positive: write a result log, freeze the formula/source
  hash, and request/enter the next allowed stage under the goal objective.
- Next-loop action if negative: `KILL` or `MUTATE` this reranker without changing
  data layers.
- Next-loop action if ambiguous: create a smaller reranker or candidate-quality
  diagnostic card.

## Result Log Template

- Experiment: Timing v3 Experiment 011 candidate-local raw-audio gain reranker
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
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
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes, for synthetic fixtures, the two
  exposed mechanism probes, and already exposed pilot80 only
- Closed loop complete: yes
- Remaining ambiguity: whether the fixed formula is strong enough for pilot80;
  that must be measured, not adjusted on protected data.

## Next-Loop Action

- If positive: implement a disabled-by-default rerank path under the Experiment
  011 source/config identity, write the result log, then consider the next
  authorized data layer.
- If negative: kill this reranking family or mutate the scoring formula in a new
  card before any further data access.
- If ambiguous: keep holdout sealed and create a smaller diagnostic card focused
  on either stable false-boundary control or short-jump selection.

## Result

- Date: 2026-08-13
- Dataset slice: synthetic/source inspection plus the already exposed
  `2300685` stable and `618173` short-jump mechanism probes; no pilot80 batch or
  protected holdout was opened for this formula.
- Local verification outcome: `raw_full(c)` is roughly `0.8--1.0`, while the
  non-constant `candidate_gain(c)` is roughly `0.01`.  The frozen formula
  therefore compares unlike quantities: every constant receives an absolute
  raw score, while every jump receives only a small difference score.  This
  structurally forces the selector toward a constant independently of whether
  a real excursion is present.
- Representative exposed values: on `618173`, the recalled near-comparator
  candidate is `143.964 BPM` for `4.168 s`, with `raw_full=0.819719` and
  `candidate_gain=0.012379`; those values cannot outrank a constant scored on
  the absolute scale under the frozen formula.  On `2300685`, the best false
  jump has negative local gain (`-0.005733`), confirming that the contrast is
  useful as evidence but not that the proposed cross-class formula is valid.
- Interpretation: negative.  Variant C as frozen is dimensionally incoherent
  for constant-versus-jump selection.  Coefficients must not be retuned on
  pilot80; this card stops before implementation.
- Kill criteria triggered: the selected variant cannot pass both exposed
  mechanism probes without changing its frozen scoring formula.
- Recommended next step: `MUTATE` in a new Experiment Card to an ordinal,
  evidence-gated production selector.  Keep candidate generation and feature
  extraction fixed; exclude diagnostic ramps from production selection; make
  raw-ranking unavailability an explicit fallback rather than candidate zero.

## Novelty Notes

- Closest analogies: contrastive model selection against a nested no-change
  baseline and onset-strength beat-grid scoring.
- Novelty layer, if any: no research novelty claim; this is a bounded
  deterministic scoring repair.
- Representation novelty vs engineering variation: no representation change;
  this is engineering variation in candidate selection.
