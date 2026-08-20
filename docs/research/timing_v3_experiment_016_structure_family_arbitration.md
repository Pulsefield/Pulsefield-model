# Timing v3 Experiment 016: Source-only structure-family arbitration

Status: superseded before execution by corrected Exp014 mechanism evidence (TV3-049)

This card was drafted from an invalidated concurrent pre-cleanup Exp014 run.
The authoritative post-cleanup mechanism gate failed earlier at candidate
retention: internal jump-cap family imbalance removed the known short-ABA
candidate.  No Experiment 016 code or execution is authorized; the next
mutation is separately numbered and changes retention only.

## Mode

- Mode: planner
- Route: TEST
- Source idea: Experiment 014 fixed boundary-complete candidate presence, but
  the selector still chose the wrong structure families.  On the fixed exposed
  structure-manifest6, the stable control selected a false jump, persistent rows
  collapsed to constant or ABA, and progressive/multi-step rows collapsed to
  implausible ABA.  The next smallest mutation is therefore not another
  threshold or candidate-family change; it is structure-family arbitration.
- Acceptance source, if any: goal objective; Timing v3 task definition;
  Experiment 013 negative pilot42 result; Experiment 014 negative
  structure-manifest6 result; TV3-045 through TV3-047.
- Source snapshot / evidence grade: strong local evidence from already-exposed
  Exp013 pilot42 and Exp014 mechanism/structure-manifest results.  Protected
  holdout100-v2, broad500, and full5050 remain sealed.

## Hypothesis

Timing v3 can reject stable false jumps and recover more real jump structures if
candidate ranking first checks whether a candidate's family semantics are
supported by the source-only raw observation path.

Exp014 already makes persistent, ABA, and multi-step candidates present.  Its
failure is that a positive whole-candidate raw self-score can promote the wrong
family.  A nonconstant candidate should challenge the best constant only when
its raw observations explain the candidate's family role: persistent edge
change, bounded ABA return, or ordered multi-step path.

## Root Objective

Change exactly one behavior-affecting variable: the selector's structure-family
arbitration.  Keep Exp014 raw features, BeatThis cache use, candidate generator,
candidate cap, collapsed-constant comparator, v2 fallback contract, and
evaluation separation unchanged.

## Goal Decomposition

- Subgoal 1: preserve constant as an always-competing production candidate.
- Subgoal 2: keep Exp014's candidate family:
  - constant;
  - single persistent `A -> B`;
  - short or long `A -> B -> A` ABA from 2 s through 60 s;
  - at most four-section piecewise-constant multi-step paths.
- Subgoal 3: compute a source-only family-support ledger for every nonconstant
  candidate from the existing raw local tempo observations and their existing
  Exp014 compatibility rules.
- Subgoal 4: require a nonconstant to satisfy both:
  - strict positive candidate-local raw gain over its own full-duration
    collapsed constant counterfactual; and
  - family-semantic support for its declared structure class.
- Subgoal 5: prevent wrong-family acceptance.  If a row has raw evidence but no
  family-supported production candidate, select the best constant when
  production-usable, otherwise return explicit `v2_fallback`.
- Subgoal 6: keep pilot42 and protected data sealed until source/synthetic,
  mechanism2, and structure-manifest6 gates pass under one frozen selector.

## Candidate Variants

- Variant A: tune Exp014 raw-gain, BeatThis-support, overlap, or raw-run
  thresholds.
- Variant B: impose a fixed family priority order, for example persistent over
  ABA over chain, once any family-compatible candidate exists.
- Variant C: selected source-only family-support arbitration.  Add a
  deterministic ledger that validates whether each candidate's structure family
  is explained by the ordered raw-run path, then rank only fully supported
  nonconstant candidates against constant.
- Variant D: replace Exp014 with a full low-rate dynamic program over tempo
  states and boundaries.

## Local Verification Matrix

- Variant A:
  - Check: compare against Exp013 and Exp014 failure attribution.
  - Fail condition: it uses exposed labels to choose margins or thresholds
    rather than addressing the wrong-family selector failure.
- Variant B:
  - Check: synthetic matrix containing both true persistent and true short ABA.
  - Fail condition: a hard priority rescues one family by suppressing another
    valid family.
- Variant C:
  - Synthetic/source checks:
    1. stable direct constant;
    2. stable false-run tail/control texture;
    3. stable 2x or `230.77 BPM` fill/alias artifact;
    4. short ABA;
    5. long 30 s ABA;
    6. persistent down with edge-supported suffix;
    7. persistent up with edge-supported prefix or suffix;
    8. progressive three-stage path with two ordered raw runs;
    9. multi-step up/down chain with three retained raw runs;
    10. weak evidence / raw unavailable fallback;
    11. contradictory raw runs with no coherent family path;
    12. candidate cap and candidate-local beat-domain coverage.
  - Exposed real checks, only after synthetic/source passes:
    1. already-exposed mechanism stable row
       `dataset/0/2300685/audio.mp3`;
    2. already-exposed mechanism short-ABA row
       `dataset/0/618173/audio.mp3`;
    3. fixed already-exposed structure-manifest6:
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
  - Pilot check: the already-exposed Exp013 pilot42 only if all earlier gates
    pass with no selector mutation.
- Variant D:
  - Check: scope and attribution review.
  - Fail condition: it changes candidate search, scoring objective, and selector
    semantics at the same time, making a fast negative result hard to interpret.

## Selected Variant

- Selected: Variant C.
- Rejected:
  - Variant A is exposed-data threshold tuning and is prohibited.
  - Variant B is not structurally defensible; wrong priority can convert a true
    ABA into persistent or a true persistent change into ABA.
  - Variant D is plausible if Variant C fails, but it is broader than necessary
    for the current evidence.
- Why this is the smallest useful test: Exp014 already emits the needed
  structure families.  The smallest next test is to change only how emitted
  families compete.

## Selection Pressure

- Primary pressure: reduce wrong-family accepted jumps on already-exposed
  structure controls.
- Guard pressure: do not reduce constant safety, break phase continuity, change
  candidate domains, alter raw features, use `.osu` or metadata for inference,
  or open protected data.
- Runtime pressure: keep the existing candidate cap at or below 64; the ledger
  must be linear in retained raw runs and emitted candidates, with no material
  runtime regression against the Exp014 structure-manifest6 baseline.
- Kill pressure: kill before pilot42 if stable false-jump rejection or
  persistent/progressive broad-family recovery still fails on source/synthetic
  or structure-manifest6.

## Research Question

Does source-only structure-family arbitration improve Exp014's wrong-family
selection failure without changing candidate generation, raw feature
extraction, or exposed-data thresholds?

## Closest Analogies / Novelty Layer

- Closest analogies: structured change-point model selection, finite-state
  tempo-path validation, observation-to-segment alignment, family-specific
  likelihood-ratio guards, and conservative fallback arbitration.
- Relevant taxonomy bucket: deterministic inference and selection workflow, not
  model training.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: the Timing v3 output remains
  a phase-continuous piecewise-constant beat grid.  This card is an engineering
  mutation of selector semantics.

## Minimal Change

Implement the selected selector mutation behind a new Timing v3 research
version:

1. Keep Exp014 candidate generation exactly in scope:
   - constants are reserved and always compete;
   - persistent, ABA, and at-most-four-section candidates remain available;
   - all candidates keep one global absolute beat axis internally;
   - candidate-local terminal `end_beat` remains allowed.
2. Keep Exp014 source signals:
   - shift-0 BeatThis cache;
   - existing deterministic raw-audio local tempo observations;
   - existing raw-run extraction and compatibility thresholds;
   - no `.osu`, metadata, network BPM, or manual listening input.
3. Add a `structure_family` diagnostic for every nonconstant candidate:
   - `persistent`;
   - `aba`;
   - `chain`;
   - `unsupported`;
   - `not_phase1_jump`.
4. Add a family-support ledger built only from retained raw runs:
   - each non-primary section must map to at least one compatible raw run under
     the existing Exp014 section/run compatibility rule;
   - a persistent candidate is family-supported only when the changed section is
     edge-open in candidate physical time and its compatible raw run also
     supports that same edge role;
   - an ABA candidate is family-supported only when the middle section is
     interior and the raw evidence supports both entering and leaving the
     excursion role; a single interior deviation run may support ABA only when
     its expanded interval covers both candidate boundaries;
   - a chain candidate is family-supported only when ordered retained raw runs
     map monotonically to the candidate's ordered non-primary sections without
     requiring an unobserved return to the primary tempo;
   - candidates that explain the same raw run as mutually incompatible family
     roles receive `unsupported` for production arbitration.
5. Keep the Exp014 collapsed-constant comparator:
   - the collapsed comparator represents the no-change counterfactual with the
     same global origin and primary/outer BPM;
   - its integer `end_beat` is chosen independently to cover the audio;
   - score comparison remains over physical in-audio coverage.
6. Production eligibility rule:
   - constants are eligible when production raw self-scoring is available;
   - nonconstants are eligible only when raw self-score and collapsed raw score
     are production-usable, raw self-score is strictly greater than collapsed
     raw score, and `structure_family_support == full`;
   - no arbitrary numeric gain margin is added.
7. Production ranking rule:
   - if no nonconstant is eligible, select the best constant;
   - among eligible nonconstants, rank by:
     1. ordinal paired raw-gain rank;
     2. generalized BeatThis physical support rank;
     3. lower section count inside the same broad family;
     4. candidate fingerprint.
   - do not impose a fixed priority between persistent, ABA, and chain
     families.  Family semantics are a production eligibility gate and a
     diagnostic, not a cross-family hard priority.
8. Do not change:
   - raw-run thresholds;
   - candidate cap;
   - raw feature extraction;
   - BeatThis support calculation;
   - v2 fallback routing;
   - weak comparator metrics;
   - Exp015 nice-number prior.

## Files Likely to Change

If this card is executed:

- `src/pulsefield_model/timing/v3/tempo_track.py`
- `src/pulsefield_model/timing/evaluation/exp013_pilot.py`
- a new or extended `src/pulsefield_model/timing/evaluation/exp016_pilot.py`
- `tests/timing/test_timing_v3_tempo_track.py`
- `tests/timing/test_timing_v3_exp014_tempo_track.py`
- `tests/timing/test_timing_v3_exp013_pilot.py`
- this card's result log
- `docs/research/timing_v3_problem_log.md`

This planning step changes only this card and the problem log.

## Read-Only Context Files

- `/Users/l/.codex/attachments/97bd173a-3590-4524-8b1f-f7a90c5e0223/goal-objective.md`
- `AGENTS.md`
- `README.md`
- `docs/research/timing_v3_task_definition.md`
- `docs/research/timing_v3_problem_log.md`
- `docs/research/timing_v3_experiment_013_raw_run_ordinal_selector.md`
- `docs/research/timing_v3_experiment_014_boundary_complete_jump_candidates.md`
- `docs/research/timing_v3_experiment_015_nominal_bpm_nice_number_prior.md`
- `src/pulsefield_model/timing/v3/tempo_track.py`
- current Timing v3 tests

## Dataset Slice

Card creation reads no new real data and runs no experiment.

If later executed, use this order:

1. source/unit/synthetic fixtures only;
2. already-exposed mechanism2:
   - `dataset/0/2300685/audio.mp3`;
   - `dataset/0/618173/audio.mp3`;
3. fixed already-exposed structure-manifest6 listed above;
4. already-exposed Exp013 pilot42 only if every earlier gate passes.

Do not open holdout100-v2, broad500, full5050, network catalog data, manual
listening, or any new `.osu` comparator evidence under this card.

## Baseline / Comparator

- Primary baseline: Exp014 selector and runner.
- Baseline source hashes recorded in Exp014:
  - tempo-track source SHA-256:
    `fb1ff93445ff16f2a15249b54d130201bbdf78d1cff21d66a10c5483ce358b58`;
  - Exp014 runner source SHA-256:
    `c5d5c65a2d92a11373bd4b00cec569fab763d2a2d6633daeaf11dd9d43fe9f3b`.
- Mechanism2 baseline artifacts:
  - JSONL SHA-256:
    `2b7079ab7c969cab71bf0c020a594bd83a1240c212f8494c97bed551d353bb30`;
  - summary SHA-256:
    `82dcf22992433d5c8001c8b57fbc3a8b3226618b92af21e8d0b36c51717efde2`.
- Structure-manifest6 baseline artifacts:
  - JSONL SHA-256:
    `d207a53a88df7ce52ad5aec312fa22f38c2466495857ec1f82269c2efc3a8153`;
  - summary SHA-256:
    `459bc8534bae4294c210d2ac4e6e3ea35e51335d67a4021801c0911c2d73d847`.
- Product fallback comparator: current timing v2.
- Weak comparators: `.osu` redlines/object placement remain evaluation-only and
  cannot influence prediction.

## Primary Metric

Pre-pilot structure-manifest6 gate:

- stable false-jump control `813270` must select constant or explicit
  `v2_fallback`, not an accepted nonconstant;
- no jump manifest row may be accepted as a contradictory broad family:
  persistent rows cannot be accepted as ABA, ABA rows cannot be accepted as
  persistent, and progressive/multi-step rows cannot be accepted as short ABA;
- at least four of the five jump manifest rows must be selected as weak-compatible
  broad-family nonconstant structures;
- maximum seam error must be at most `5 ms`;
- hard failures must be zero.

## Secondary Metric

If the pre-pilot gate passes, run pilot42 and report:

- accepted stable false-jump count;
- jump-row weak-compatible broad-family success count;
- jump exact-hit count where weak boundary comparators are available;
- pure-v3 accepted count and fallback reasons;
- phase mean/p50/p90/max;
- endpoint and max-prefix drift;
- alias-aware local BPM error;
- section count and unsupported short-section rate;
- p50/p90/max row runtime and relative runtime change versus Exp014;
- family-support diagnostics and rejection reasons.

## Verify Command / Evaluation Procedure

Planning-only card creation:

```text
git diff --check
```

If executed later, use a new Exp016 runner or an Exp014-compatible runner with
frozen Exp016 provenance:

```text
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_analytic_curve.py \
  tests/timing/test_timing_v3_audio_evidence.py \
  tests/timing/test_timing_v3_curve_metrics.py \
  tests/timing/test_timing_v3_real_audio_pilot.py \
  tests/timing/test_timing_v3_exp013_pilot.py \
  tests/timing/test_timing_v3_tempo_track.py \
  tests/timing/test_timing_v3_exp014_tempo_track.py
```

Then run only the frozen data slices in the dataset order above.  Record exact
commands, row identities, source hashes, input hashes, output hashes, and
summary hashes before interpreting results.

## Guard Check

- No protected holdout100-v2, broad500, or full5050 access.
- No threshold tuning on pilot42.
- No candidate-family expansion beyond Exp014.
- No ramp production output.
- No `.osu`, metadata, network, or manual evidence in inference.
- Constant always competes.
- Candidate cap remains at or below 64.
- Candidate-local beat domains remain scoreable over physical in-audio
  coverage.
- Collapsed constants remain full-duration no-change counterfactuals with their
  own terminal beat domain.
- Every accepted curve must have exact phase-continuous seams and serialize
  with maximum seam error at most `5 ms`.
- Hard failures must be zero.

## Qualitative Check

For every synthetic and exposed real row, inspect the source-only diagnostics,
not the audio:

- selected structure family;
- family-support ledger entries;
- raw runs used and rejected;
- paired raw gain against collapsed constant;
- BeatThis support rank;
- fallback or rejection reason;
- whether the selected family is broad-compatible with the row's already-known
  exposed diagnostic role.

## Positive Signal

- Synthetic matrix passes.
- Mechanism2 still selects `2300685` as constant and recovers the `618173`
  short-ABA mechanism without new thresholds.
- Structure-manifest6 passes the primary gate.
- Pilot42, if reached, improves Exp013's jump-family success without increasing
  stable false jumps beyond the frozen guard.
- Runtime stays within budget and seam/hard-failure guards remain clean.

## Negative Signal

- Stable false jump persists on `813270`.
- Persistent rows still collapse to constant or ABA.
- Progressive/multi-step rows still collapse to short ABA.
- The selector needs new numeric thresholds chosen after viewing exposed rows.
- Pilot42 improvement is only from more aggressive acceptance rather than
  broad-family correctness.

## Kill Criteria

Kill Exp016 before pilot42 if any of these occur:

- source/synthetic matrix fails a structure-family invariant;
- mechanism2 regresses stable constant or short-ABA recovery;
- structure-manifest6 has any stable accepted nonconstant;
- fewer than four of five jump manifest rows are broad-family compatible;
- any jump manifest row is accepted as a contradictory broad family;
- p90 row runtime on structure-manifest6 exceeds the Exp014
  structure-manifest6 baseline by more than 10% or exceeds eight seconds;
- hard failure or seam guard fails;
- implementation requires changing raw-run thresholds, candidate generation,
  or weak comparator semantics.

Kill after pilot42 if reached and:

- hard failures are nonzero;
- stable false jumps increase relative to Exp013/Exp014 exposed baselines;
- jump-family success is not materially better than Exp013's two accepted jump
  classifications;
- gains depend on post-hoc row inspection or pilot42 threshold tuning.

## Expected Failure Modes

- Source-only raw observations may not contain enough information to separate
  stable texture changes from true persistent jumps.
- A real persistent change near the track edge can be observationally similar
  to a short tail fill.
- Exp014's retained top `K=4` raw runs may omit the run needed for a true
  chain.
- Weak comparators may describe mapper redlines rather than exact recording
  truth, so a broad-family mismatch can remain ambiguous for some rows.
- Family arbitration may reduce false positives by increasing fallback, without
  improving pure-v3 value.
- Runtime may exceed budget if ledger diagnostics rescore candidates
  redundantly.

## Confounders

- The fixed structure-manifest6 is already exposed and cannot justify holdout
  claims.
- Weak labels may be wrong or incomplete for exact jump boundary timing.
- Exp014 candidate generation can still fail to emit a correct boundary even if
  the family selector is improved.
- Raw-audio local tempo observations are deterministic but not truth.
- BeatThis support is chunk-conditioned activation/ranking evidence, not
  independent beat probability.

## Expected Runtime / Runtime Budget

- Card creation: documentation only.
- Synthetic/source tests if executed: under one minute.
- Mechanism2 if executed: under one minute with existing caches.
- Structure-manifest6 if executed: under two minutes.
- Pilot42 if executed after gates pass: under ten minutes.
- Per-row p90 target before pilot42 promotion: no more than 10% regression
  versus Exp014 on the fixed structure-manifest6 and no more than eight seconds
  on either structure-manifest6 or pilot42.
- Hard row timeout remains the existing runner timeout.

## Result Interpretation Plan

- Positive result would suggest: Exp014's main blocker was selector
  wrong-family arbitration, and source-only structure roles are sufficient for a
  next exposed-pilot gate.
- Negative result would suggest: current raw/BeatThis observations cannot
  reliably distinguish structure families with local selector logic; the next
  mutation should be a more explicit state-path search or a kill of this
  candidate line.
- Ambiguous result would require: record which families remain inseparable and
  decide whether a measurement-only source audit or dynamic-path card is
  justified.
- Human owner decides: whether a positive pilot42 is strong enough to create a
  next-stage card.  This card itself does not authorize holdout, broad, or
  full5050.
- Next-loop action if positive: write result log and freeze a next-stage
  exposed repair/no-new-data review card.
- Next-loop action if negative: close Exp016 and create a new card for dynamic
  path search or kill the jump-selector branch.
- Next-loop action if ambiguous: stop, record ambiguity, and request owner
  decision before opening any new data layer.

## Result Log Template

- Experiment:
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
- Code execution allowed after this card: yes, but only for Variant C and only
  inside the files/slices listed above.
- Closed loop complete: yes
- Remaining ambiguity: source-only family support may still be insufficient to
  distinguish stable tail texture from true persistent edge changes.  The fast
  kill gate is structure-manifest6.

## Next-Loop Action

- If positive: record result, keep protected data sealed, and create the next
  authorized stage card.
- If negative: close Exp016 and mutate to a dynamic state-path card or kill the
  current raw-run selector branch.
- If ambiguous: stop and record the exact ambiguity; do not tune thresholds or
  open protected data.

## Novelty Notes

- Closest analogies: structured change-point family arbitration,
  observation-to-segment alignment, and conservative production fallback.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation inside
  the existing Timing v3 constant/jump representation.
