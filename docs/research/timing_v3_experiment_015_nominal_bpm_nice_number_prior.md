# Timing v3 Experiment 015: Nominal-BPM nice-number soft prior

Status: planned / not executed

## Mode

- Mode: planner
- Route: TEST
- Source idea: stable, digitally produced music often has an intended nominal
  BPM that is a simple human-facing number.  A soft nice-number prior may help
  rank otherwise close constant-tempo candidates, especially when raw/BeatThis
  evidence distinguishes phase but leaves small BPM jitter.
- Acceptance source, if any: human-owner research idea plus the Timing v3 goal
  contract.  This card does not authorize code execution, real-data access, or
  changes to Exp014.
- Source snapshot / evidence grade: hypothesis only.  No new real data,
  holdout, broad, full5050, network evidence, manual listening, or `.osu`
  evidence is read for this card.

## Hypothesis

For source-stable constant rows, a preregistered soft prior toward nominal nice
BPM values can improve constant-BPM candidate ranking without
hurting phase continuity, alias handling, off-grid true BPMs, acoustic/live
material, jump detection, or fallback safety.

The prior is not truth.  It cannot hard-snap BPM, create labels, decide
constant versus jump/ramp structure, or override the evidence role separation.
The original digital-production motivation remains a hypothesis; unless a
separate audio-only proxy is preregistered, execution eligibility is called
`source-stable constant`, not `digital`.

## Root Objective

Test one bounded variable: whether a soft nominal-BPM prior improves stable
constant-candidate selection in Timing v3 Phase 1 while preserving the existing
constant/jump product contract and sealed-data ordering.

## Goal Decomposition

- Subgoal 1: define the nice-number candidate set and distance function before
  any run.
- Subgoal 2: restrict the prior to the source-only source-stable constant
  ranking lane after the prior-disabled structure gate has already admitted that
  lane.
- Subgoal 3: make the prior soft-only: no hard snap, no truth policy, no
  structure-class decision, and no `.osu`/metadata/manual inference input.
- Subgoal 4: include negative controls for off-grid true BPM, acoustic/live
  timing, tempo aliases, and phase invariance.
- Subgoal 5: use an ablation gate so any effect is attributable to the prior,
  not a simultaneous proposal, selector, feature, or evaluator change.

## Candidate Variants

- Variant A: KILL hard snapping.  Replace evidence BPM with the nearest nice
  number whenever close.
- Variant B: proposal-only nice candidates.  Add nearest nice-BPM constant
  candidates but keep ranking unchanged.
- Variant C: selected ranking-only soft prior.  Do not add candidates.  Apply a
  frozen prior term only inside a preregistered source-evidence-equivalent
  bucket of existing constant candidates in the source-only source-stable lane.
- Variant D: global nice prior across constant, jump, and ramp decisions.

## Local Verification Matrix

- Variant A:
  - Check: off-grid stable fixture at `173.37 BPM`.
  - Fail condition: output is forced to `173`, `175`, or another nice value
    despite worse source evidence.
- Variant B:
  - Check: candidate-cap and attribution review.
  - Fail condition: improvement cannot be separated from candidate presence, or
    nice proposals displace jump candidates under the cap.
- Variant C:
  - Checks:
    1. source-stable synthetic exact-nice fixture `180.0 BPM`;
    2. source-stable synthetic near-nice fixture `179.82 BPM`;
    3. stable off-grid fixture `173.37 BPM`;
    4. acoustic/live fixture with small nonstationary drift;
    5. alias fixture `90/180 BPM`;
    6. phase-shift invariance fixture;
    7. jump fixture with a nice outer BPM and non-nice middle BPM;
    8. ramp-audit fixture;
    9. candidate-cap/runtime fixture.
  - Pass condition: only source-stable constant ranking inside an
    evidence-equivalent bucket changes, the synthetic suite contains at least
    one non-no-op efficacy fixture, and the prior does not change structure
    class, phase, beat axis, fallback routing, candidate set, or ramp/jump
    decisions.
- Variant D:
  - Check: jump/ramp structure-control fixtures.
  - Fail condition: nice-number pressure suppresses a true nonconstant section
    or converts ramp audit material into a Phase-1 production claim.

## Selected Variant

- Selected: Variant C.
- Rejected:
  - Variant A violates soft-only behavior and would be label leakage by prior.
  - Variant B changes proposal coverage and ranking attribution at the same
    time.
  - Variant D lets a stable prior interfere with structure inference.
- Why this is the smallest useful test: it changes exactly one ranking
  variable in one lane while leaving candidates, raw features, BeatThis cache,
  metrics, fallback routing, and section representation unchanged.

## Selection Pressure

- Primary pressure: improve source-stable constant BPM selection when
  candidates are source-evidence-equivalent under a preregistered synthetic
  tie-band.
- Guard pressure: never alter phase, beat-axis continuity, structure class,
  jump/ramp decisions, fallback routing, or inference/evaluation separation.
- Runtime pressure: no candidate-count increase and negligible ranking
  overhead.
- Kill pressure: any off-grid, acoustic/live, alias, jump, ramp, cap, or phase
  regression kills this prior before real-data execution.

## Research Question

Can a preregistered nice-number soft prior improve source-stable constant-BPM
ranking without becoming a hidden hard snap, no-op, or structure-class oracle?

## Closest Analogies / Novelty Layer

- Closest analogies: musical tempo priors, quantized parameter priors, MAP
  ranking with weak human-authored nominal-value bias, and ablation-controlled
  model-selection priors.
- Relevant taxonomy bucket: deterministic ranking prior, not model training.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: the Timing v3
  phase-continuous constant/jump representation is unchanged.  This is an
  engineering prior inside the stable constant-candidate ranker.

## Minimal Change

If this card is later executed, implement only the selected ranking-only prior:

1. Define the nice-number set:
   - tier 0: multiples of `10 BPM` in `[80, 240]`;
   - tier 1: multiples of `5 BPM` in `[80, 240]` that are not tier 0;
   - tier 2: integer BPM values in `[80, 240]` that are not tier 0 or tier 1.
2. Define distance for candidate BPM `b`:
   - `nice_distance_cents(b) = min_n 1200 * abs(log2(b / n))`,
     where `n` ranges over the tiered nice-number set;
   - ties choose lower tier number first, then smaller absolute BPM distance,
     then smaller `n`.
3. Before real-data execution, preregister a source/synthetic-only
   source-evidence equivalence rule:
   - compute the no-prior source evidence score and baseline source rank for
     each existing constant candidate;
   - choose any score tolerance, bucket rule, or tie-band only from source/unit
     and synthetic fixtures, not from exposed or protected real rows;
   - candidates outside the same source-evidence-equivalent bucket cannot be
     reordered by the nice prior;
   - if a safe equivalence rule cannot be preregistered without real-data
     tuning, this card becomes ambiguous or killed before implementation.
4. Define the soft prior as an ordinal term only inside one
   source-evidence-equivalent bucket:
   - source-stable constant candidates are ranked by
     `(source_evidence_bucket, nice_tier, nice_distance_cents,
     baseline_source_rank, fingerprint)`;
   - `source_evidence_bucket` stays first, so source-non-equivalent candidates
     cannot be reordered by the prior;
   - `baseline_source_rank` is retained only as the deterministic tie-break
     after nice terms inside an already-equivalent bucket.
5. Do not add candidates, remove candidates, or change candidate BPM values.
6. Do not hard snap any BPM to the nearest nice value.
7. Do not use the prior for jump/ramp proposal generation, raw-run detection,
   boundary choice, class selection, weak-label interpretation, or fallback
   routing.
8. The prior is eligible only when the frozen prior-disabled source-only
   structure gate has already admitted the source-stable constant lane before
   nice-prior ranking:
   - prior-enabled and prior-disabled runs must have identical structure class,
     candidate set, candidate BPM values, fallback route, and lane eligibility
     before the nice term is considered;
   - the prior is never allowed to create, suppress, or reinterpret
     production-eligible jump/ramp evidence;
   - no metadata, `.osu`, network, title, artist, catalog BPM, manual listening,
     or evaluation label evidence is used to decide this lane.
9. If source-stable eligibility or source-evidence equivalence is unavailable or
   ambiguous, the prior is disabled and the baseline ranking is used.

## Files Likely to Change

If later executed:

- `src/pulsefield_model/timing/v3/tempo_track.py`
- a focused Timing v3 test file under `tests/timing/`
- this card's result log

This planning step adds only:

- `docs/research/timing_v3_experiment_015_nominal_bpm_nice_number_prior.md`

## Read-Only Context Files

- `/Users/l/.codex/attachments/97bd173a-3590-4524-8b1f-f7a90c5e0223/goal-objective.md`
- `AGENTS.md`
- `README.md`
- `docs/research/timing_v3_task_definition.md`
- `docs/research/timing_v3_problem_log.md`
- current Timing v3 candidate/ranking source and tests, if this card is later
  executed

## Dataset Slice

This card creation reads no new real data and runs no experiment.

If later executed, the required order is:

1. source/unit/synthetic fixtures only;
2. already-exposed stable diagnostic rows only if synthetic passes;
3. no protected holdout100-v2, broad500, or full5050 under this card.

## Baseline / Comparator

- Baseline: the active Timing v3 constant-candidate ranking before the nice
  prior.
- Ablation comparator: identical source/config with the prior disabled.
- Product fallback comparator: current timing v2 remains the fallback path.
- Weak comparator: `.osu` redlines/object placement are evaluation only and are
  not read before prediction fingerprints are frozen in any later execution.

## Primary Metric

Synthetic ablation result, split into safety and efficacy:

- Safety gate: nice-prior enabled does not change the selected BPM on
  off-grid, acoustic, alias, jump, ramp, or phase-invariance controls unless the
  prior-disabled source evidence already places the changed candidates in the
  same preregistered source-evidence-equivalent bucket.
- Efficacy gate: at least one source-stable synthetic fixture has identical
  candidate set, candidate BPM values, structure class, phase diagnostics,
  fallback route, and beat axis in prior-enabled and prior-disabled runs, but
  selects the nominal nice BPM only when the prior is enabled.
- Preservation-only outcomes pass safety but do not pass efficacy; they are
  interpreted as ambiguous or negative, not positive.

## Secondary Metric

- selected BPM and nearest nice target;
- nice tier and `nice_distance_cents`;
- source evidence bucket and baseline source rank before and after prior;
- phase mean/p90 invariance;
- endpoint and max-prefix drift;
- alias-aware BPM error;
- selected structure class;
- fallback reason;
- candidate count and ranking runtime;
- ablation delta with prior enabled versus disabled.

## Verify Command / Evaluation Procedure

No command is run during card creation.

If later executed, run a focused synthetic suite that includes:

- source-stable exact-nice stable BPM;
- source-stable near-nice stable BPM;
- source-stable source-equivalent non-no-op fixture where the no-prior selected
  candidate is non-nice and a nice candidate exists in the same preregistered
  evidence bucket;
- off-grid true stable BPM;
- acoustic/live nonstationary stable-like BPM;
- half/double alias interaction;
- fixed phase-shift invariance;
- jump row with nice outer BPM and non-nice middle BPM;
- ramp-audit row;
- candidate-cap/runtime guard.

Only after synthetic pass may an already-exposed stable diagnostic slice be
used, and only with a prior-disabled ablation run.

## Guard Check

- no code execution or real-data access during this planning step;
- no hard snapping;
- no candidate BPM mutation;
- no candidate addition/removal in selected Variant C;
- no source-evidence-equivalence tolerance chosen from real exposed or
  protected rows;
- no no-op or preservation-only result counted as efficacy;
- no use as truth policy;
- no use in jump/ramp structure detection;
- no `.osu`, metadata, network, title/artist, or manual-listening inference;
- no phase, seam, or beat-axis change;
- no fallback-routing change;
- no protected data access;
- no candidate cap increase;
- no promotion claim from exposed data.

## Qualitative Check

Inspect the ablation diagnostics after any later synthetic execution:

- nice prior should appear only as an extra rank term inside an
  evidence-equivalent bucket of source-stable constants.
- off-grid and acoustic/live fixtures should document why the prior did not
  override source evidence.
- alias fixture should show that `90` and `180` are not resolved by nice-number
  preference alone.
- jump/ramp fixtures should show prior disabled outside the stable constant
  lane.

## Positive Signal

- at least one source-stable, source-equivalent, non-no-op efficacy fixture
  selects the nominal nice BPM only with the prior enabled;
- synthetic exact-nice and near-nice stable safety cases remain unchanged or
  improve without candidate-set, structure, phase, fallback, or beat-axis
  changes;
- off-grid true BPM remains off-grid when evidence supports it;
- acoustic/live fixture is not over-quantized;
- alias fixture is unchanged unless evidence rank already resolves the alias;
- jump and ramp controls are unchanged;
- phase, seam, fallback, cap, and runtime guards pass.

## Negative Signal

- selected BPM is hard-snapped to a nice value;
- off-grid or acoustic/live fixture is pulled to a nice number against
  evidence;
- prior changes jump/ramp class or boundary behavior;
- prior resolves alias without independent evidence;
- candidate cap or runtime changes;
- prior-enabled and prior-disabled runs cannot be compared cleanly.
- prior-enabled output differs only because a unique source evidence rank was
  treated as a tie without a preregistered source-equivalence rule;
- exact-nice and near-nice fixtures only preserve baseline behavior and no
  non-no-op efficacy fixture changes selection.

## Kill Criteria

Kill this idea if any of these occur:

- any synthetic guard fails;
- the prior changes non-stable structure decisions;
- the prior mutates candidate BPM rather than ranking candidates;
- off-grid or acoustic/live controls regress;
- alias controls regress;
- integer tier 2 pulls an off-grid or acoustic/live fixture toward an integer
  BPM outside the preregistered source-evidence-equivalent bucket;
- candidate cap or runtime changes materially;
- ablation cannot isolate the prior from another implementation change.

## Expected Failure Modes

- Real digitally produced songs may use intentional non-integer or off-grid BPM.
- BeatThis/raw evidence jitter may be larger than the nice-prior effect.
- A nice-number prior can hide alias mistakes if allowed outside a strict lane.
- Acoustic/live timing can be stable enough to look digital but should not be
  quantized.
- Integer BPM may be too broad to help unless source evidence is already close.
- If baseline source evidence ranks are unique and no safe equivalence bucket is
  defined, the selected ranking tuple may be a no-op.

## Confounders

- `.osu` redlines often contain mapper-chosen nominal values and cannot serve
  as inference truth.
- Metadata/catalog BPM is excluded from inference and may itself be rounded.
- Exposed development rows cannot support fresh-holdout claims.
- Nice-number preference is culturally/tooling dependent and not audio truth.

## Expected Runtime / Runtime Budget

- Card creation: no experiment.
- Later synthetic suite: under 10 seconds.
- Later exposed stable ablation, if authorized after synthetic pass: under the
  active card's row-runtime budget with no candidate-count increase.

## Result Interpretation Plan

- Positive result would suggest: a ranking-only nominal-BPM prior has a real
  non-no-op effect inside source-equivalent source-stable constants and may be
  worth testing on an already-exposed stable slice, still disabled outside the
  stable lane.
- Negative result would suggest: nice-number bias is unsafe or too weak for
  Timing v3 Phase 1 and should be killed before any real-data expansion.
- Ambiguous result would require: separating source-stable lane detection or
  source-evidence-equivalence definition from the prior itself in a new smaller
  card.
- Human owner decides: whether a synthetic-positive prior is worth scheduling
  after the currently active Timing v3 candidate work.
- Next-loop action if positive: write an execution card or append a result log
  only after explicit scheduling.
- Next-loop action if negative: record `KILL`; do not carry nice-number bias
  into production ranking.
- Next-loop action if ambiguous: create a lane-detection-only diagnostic card.

## Result Log Template

- Experiment: Timing v3 Experiment 015 nominal-BPM nice-number soft prior
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
- Prior-disabled ablation:
- Prior-enabled ablation:
- Source-evidence equivalence rule:
- Non-no-op efficacy fixture outcome:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: no; this card is a planning artifact
  only until explicitly scheduled.
- Closed loop complete: yes
- Remaining ambiguity: the source-only source-stable eligibility rule and
  source-evidence-equivalence bucket may need their own diagnostic card if
  either proves harder than the prior itself.

## Next-Loop Action

- If positive: later, after explicit scheduling, run synthetic ablation only.
- If negative: kill the prior.
- If ambiguous: split lane eligibility or source-evidence equivalence from the
  ranking prior.

## Novelty Notes

- Closest analogies: tempo priors, quantized parameter priors, and MAP-style
  ranking with explicit ablation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is an engineering
  prior over existing constant candidates, not a Timing v3 representation
  change.
