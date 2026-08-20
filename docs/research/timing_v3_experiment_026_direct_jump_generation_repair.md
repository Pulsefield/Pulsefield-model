# Timing v3 Experiment 026: Direct jump generation repair

Status: accepted / execute immediately / behavior improvement only

## Mode

- Mode: planner.
- Route: TEST.
- Acceptance: the human owner authorized algorithm edits and immediate real-data experiments without another approval or review step.
- Evidence: Exp022 exposed pilot42 and Exp025 exact20 lifecycle inventory.

## Hypothesis

The current jump generator often has useful boundary times and useful tempo
modes, but it does not combine them: paired boundaries mostly create ABA
curves, while persistent A-to-B curves require a raw-only run. Recombining
source-local BeatThis/raw tempo modes on both sides of an existing single
boundary, and in the left/middle/right regions of an existing boundary pair,
will create direct phase-continuous candidates for rows that currently have no
direct candidate. Requiring an independent BeatThis boundary anchor before a
jump may defeat the best constant will also remove the current raw-only stable
false jumps.

## Root Objective

Increase real Timing-v3 Phase-1 jump quality now. This experiment does not
optimize tracing, memoization, defensive infrastructure, or documentation.

## Goal Decomposition

- Generate direct segment-tempo combinations for the 12/20 Exp025 rows whose direct candidate was absent before retention.
- Preserve stable classification while allowing the new candidate to survive the unchanged global candidate budget and reach the existing selector.
- Validate first on exposed pilot42, then immediately on the weak-label 5,050 development corpus when the pilot quality gate passes.

## Candidate Variants

- A — selected: boundary-conditioned regional-mode recombination. Around each
  existing single boundary form phase-continuous L/R persistent candidates;
  around each existing boundary pair form L/M/R candidates. Use the strongest
  local BeatThis and raw-audio tempo modes in each region.
- B — rejected: propagate long-ABA support and reorder long retention. Exp025 shows this can affect only the five generated-but-pruned rows, not the 12 not-generated rows that dominate failure.
- C — mutate into A: a bounded local-mode change seed is needed when the
  existing boundary list misses the same regional contrast. It is not a free
  raw-run widening: at most 16 regional contrasts are considered, and each
  keeps only its midpoint and the strongest existing boundary within
  `+/-3000 ms`.

## Local Verification Matrix

- A passes a focused synthetic ABC/ABA fixture only if it generates a candidate whose two boundaries are within `1000 ms` of the source boundary seeds and whose adjacent tempos equal the injected regional modes, while remaining phase-continuous with `0 ms` seams.
- A passes a stable fixture only if no output class or fallback route changes.
- A must keep jump candidates `<=44`, total candidates `<=64`, and deterministic
  fingerprints under repeated calls.
- B fails the coverage check by construction on every Exp025 `not_generated` row. C fails the majority-mechanism check from Exp025.

## Selected Variant

Implement A only:

1. Start with existing boundary and pair seeds. Add at most 16 deterministic
   regional-mode contrast locations; for each, retain only its midpoint and
   the strongest existing boundary within `+/-3000 ms`. This is part of the
   same observation-conditioned lane, not an unbounded boundary search.
2. For a single seed `t`, collect left/right regional modes without crossing
   the seed, using a bounded shoulder of three existing local windows. For each pair
   `(t1, t2)`, collect observations by center time in the left shoulder, middle
   interval, and right shoulder. A shoulder is one existing
   `local_window_seconds` wide.
3. In each region retain at most one strongest alias-normalized BeatThis mode
   and one strongest alias-normalized raw-audio mode. Expand only their octave
   aliases in `60..300 BPM`; this is lane-local and does not widen the existing
   exhaustive autocorrelation/search range. The left pool may also use the
   existing base; a missing right pool falls back to that base. Do not use
   metadata, `.osu`, weak labels, row identity, or nice-number terms.
4. Estimate the left origin with the existing phase estimator and snap the
   first boundary to its nearest integer beat. A single-boundary candidate
   starts its right tempo at that shared beat. For a pair, choose the positive
   integer middle-beat count nearest `(t2-t1) * middle_bpm / 60000` and start
   the right tempo at that shared beat. Reject a combination when an induced
   boundary is more than `1000 ms` from its source seed. In this bounded lane,
   reject only adjacent tempos closer than `max(1 BPM, 0.5%)`; the current
   global `minimum_jump_bpm` and raw-run deviation thresholds remain unchanged
   because they erase real 2--4 BPM jumps.
5. Score with existing physical support gain versus the collapsed constant, plus existing pair rank and regional observation support. No pilot-derived weight or threshold is allowed.
6. Deduplicate by curve fingerprint and keep only the best six recombined
   proposals before normal jump retention. They use the existing overflow
   quota, `maximum_jump_candidates=44`, and `maximum_candidates=64`; no cap or
   family quota increases.
7. A recombined candidate may defeat the best constant only when its raw-audio
   self-score gain is positive and its BeatThis physical-support gain is
   positive. Existing raw-only `raw_run_*` proposals cannot promote a jump
   without an independently anchored BeatThis boundary. This removes the
   same-source proposal/selection loop that caused all six Exp022 stable false
   jumps; it does not alter constant ranking.

This is the smallest variant that can turn `not_generated` into generated and
retained while bounding runtime and stable displacement.

## Selection Pressure

- Primary: exposed jump fixed-`+/-1000 ms` recall and direct adjacent-tempo
  matches.
- Guard: at least `20/22` exposed stable rows remain selected constant.
- Runtime: report p50/p90/max/total before scaling; p90 must be `<=15 s`.
- Kill: no material jump gain, stable below `20/22`, or p90 above `15 s`.

## Closest Analogies / Novelty Layer

- Closest analogies: change-point-conditioned segment decoding, local-mode
  cross-product proposal generation, and bounded beam proposal injection.
- Evidence strength: strong local lifecycle evidence; weak-label evaluation is
  useful development evidence, not musical truth.
- Novelty: none claimed. This is generation engineering inside the existing
  phase-continuous representation.

## Minimal Change / Files Likely to Change

- `src/pulsefield_model/timing/v3/tempo_track.py`
- `tests/timing/test_timing_v3_exp026_tempo_track.py`
- temporary runner
  `/private/tmp/timing_v3_exp026_direct_jump_generation_repair.py`
- this card's result log after execution
- Read-only context: Exp022/Exp025 cards, current evaluator code, and existing v2 full5050 results.

Do not implement Exp024 memoization or any unrelated cleanup in this experiment.

## Dataset Slice

1. Focused source/synthetic tests.
2. The same exposed 42 rows used by Exp022: 22 stable and 20 jump-candidate.
3. On a pilot pass, immediately evaluate all 5,050 canonical audio groups with
   the existing weak-label inventory. This is an exposed development corpus;
   report results as weak-label agreement, not unbiased generalization.

Only these result files are required:

- `artifacts/reports/timing/timing_v3_exp026_pilot42_v1.jsonl`
- `artifacts/reports/timing/timing_v3_exp026_pilot42_v1_summary.json`
- `artifacts/reports/timing/timing_v3_exp026_full5050_v1.jsonl`
- `artifacts/reports/timing/timing_v3_exp026_full5050_v1_summary.json`

No extra freeze, inventory trace, lifecycle dump, or review artifact is needed.

## Baseline / Comparator

- Exp022: stable class exact `16/22`; jump mean fixed-`+/-1000 ms` recall
  `0.12941`; three of 20 jump rows had nonzero recall; direct tempo-pair match
  count `2`; row runtime p90 `11.724 s`.
- Exp025 exact20 lifecycle: `12 not_generated`, `5 generated_pruned`,
  `2 retained_ineligible`, `1 selected`.
- Full-corpus context: existing v2 full5050 JSONL/summary. Do not rerun v2.

## Metrics

- Primary pilot metric: mean fixed-`+/-1000 ms` boundary recall over all 20 jump
  rows; unmatched and non-jump outputs contribute zero.
- Pilot guard metric: selected-constant count over all 22 stable rows.
- Secondary: jump selected count, rows with nonzero recall, direct adjacent-
  tempo-pair matches, direct/alias BPM coverage, phase mean/p90, initial offset
  within `70 ms`, fallback/failure counts, seams, and runtime p50/p90/max/total.
- Full5050: the same metrics audio-first by weak stable/jump stratum, plus tail
  distributions and exact denominators. Ramp remains audit-only and cannot
  enter a Phase-1 production claim.

## Guards, Qualitative Check, Failure Modes, and Confounders

- Guard: source-only inference, `0 ms` seams, unchanged fallback, and candidate caps `44/64`.
- Qualitative check: compare regional modes and selected curves for the three largest recall gains and three largest regressions.
- Positive signal: the complete pilot gate passes; negative signal: jump recall gain is below `0.10` or stable is below `20/22`.
- Kill criteria: any negative signal, hard failure, nonzero seam, metadata/weak-label inference access, or runtime p90 above `15 s`.
- Expected failures: absent pair seeds, alias-wrong local modes, the new candidate being pruned, or existing eligibility rejecting it.
- Confounders: pilot42 is already exposed; weak redlines are noisy; BeatThis and raw observations share the same audio.

## Verify Procedure

```bash
.venv/bin/pytest -q tests/timing/test_timing_v3_exp026_tempo_track.py
.venv/bin/python /private/tmp/timing_v3_exp026_direct_jump_generation_repair.py --stage pilot42 --output-jsonl artifacts/reports/timing/timing_v3_exp026_pilot42_v1.jsonl --summary-json artifacts/reports/timing/timing_v3_exp026_pilot42_v1_summary.json
.venv/bin/python /private/tmp/timing_v3_exp026_direct_jump_generation_repair.py --stage full5050 --output-jsonl artifacts/reports/timing/timing_v3_exp026_full5050_v1.jsonl --summary-json artifacts/reports/timing/timing_v3_exp026_full5050_v1_summary.json
```

Run the third command immediately, without another card or approval, only when
the pilot gate below passes. The 5,050 runner may resume from its JSONL; it may
not create additional audit products.

## Pilot Decision Gate

Pass only if all hold:

- exactly `42 = 22 stable + 20 jump` results, zero hard failures, and maximum
  seam `0 ms`;
- stable selected-constant count `>=20/22`;
- jump mean fixed recall `>=0.25` and at least `0.10` above `0.12941`;
- at least six jump rows have nonzero fixed recall and direct tempo-pair matches
  exceed the baseline count `2`;
- runtime p50/p90/max/total are reported before the 5,050 command, with p90
  `<=15 s`.

Failure kills this variant before 5,050. Do not tune it repeatedly on pilot42.

## Full5050 Interpretation / Final Targets

- Report stable weak-class accuracy, jump weak-class accuracy, fixed boundary
  recall, direct tempo-pair rate, phase p90, offset-within-`70 ms`, fallback,
  failures, and runtime using every eligible row.
- The Phase-1 target is stable `>=99%`, jump `>=80%`, phase p90 `<=70 ms`, and
  production runtime `<5 s` per row. Missing a target is a measured next
  mechanism, not permission to hide denominators.
- Expected runtime: pilot under 15 minutes; full5050 approximately 12 hours,
  hard stop at 18 hours with completed JSONL rows preserved.

## Result Interpretation Plan

- Pilot positive: run full5050 immediately.
- Pilot negative: KILL A and use row-level errors to choose a different
  generation mechanism; do not work on acceleration first.
- Pilot positive/full5050 below target: keep the generation gain and make the
  next behavior card target the measured selector/eligibility bottleneck.
- Full5050 at target: integrate the candidate and then optimize runtime without
  changing behavior.

## Result Log Template

- Date / source commit:
- Selected variant: A
- Tests:
- Pilot rows / stable / jump:
- Stable selected constant:
- Jump fixed recall / nonzero rows / direct tempo-pair matches:
- Phase / offset / fallback / failure / seam:
- Runtime p50 / p90 / max / total:
- Pilot gate: pass | kill
- Full5050 rows / comparator-eligible rows:
- Full5050 stable / jump / phase / offset / runtime:
- Interpretation / next behavior mutation:

## Pre-Execution Gate

- Card complete: yes.
- Code and tests may be changed now: yes.
- Pilot42 and conditional full5050 execution allowed now: yes.
- Additional approval or independent review required: no.
- Nice-number prior: constant-lane soft ranking only; forbidden in jump
  generation, retention, eligibility, and evaluation.
