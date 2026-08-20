# Timing v3 Experiment 018: Closed-ABA coarse raw-run compatibility

Status: stopped / negative

## Authoritative result

Exp018 is frozen as a two-row authoritative negative. No additional rows are
authorized or reported under this card.

- Output:
  `artifacts/reports/timing/timing_v3_exp018_mechanism2_authoritative_v1.jsonl`
  SHA-256
  `9b9d9d437f417af172c3c0a78e8c0283bde3e09d63f95c774e420c533361b840`.
- Summary:
  `artifacts/reports/timing/timing_v3_exp018_mechanism2_authoritative_summary_v1.json`
  SHA-256
  `c04187ff5a90112193312444a8e138a3ff21b86be7e93bffb0c40d6aec49cf9a`.
- Version: `exp018_v1`.
- Scope: exactly the two exposed mechanism rows, with `2/2` accepted.
- Seam max: `0.0 ms`.
- Mechanism p90 row runtime: `3.382936271 s`.

Stable row `2300685` remained accepted as the correct constant `200 BPM`
result. It retained weak exact/direct hit `true`, direct coverage `1.0`, seam
max `0.0 ms`, and runtime `3.428545 s`.

Short-ABA row `618173` selected candidate `20`, source
`paired_unmerged_boundary`, with closed sections
`175.193364 @ 310.834 ms -> 146.621034 @ 56477.345 ms -> 175.193364 @
58932.654 ms`. This is the desired structural shape improvement over Exp017's
persistent false selection. However, post-freeze weak evaluation still reported
`2` predicted boundaries, `2` weak boundaries, `0` matched boundaries, boundary
recall `0.0`, weak exact `false`, direct/alias coverage `0.930874`, phase
mean/p90 `22.9855 ms / 41.3712 ms`, endpoint drift `428.496 ms`, and runtime
`2.972458 s`.

Decision: stop Exp018 as negative. The compatibility relaxation worked enough
to select the retained closed-ABA shape, but it failed the frozen mechanism
requirement that weak boundary recall improve above `0.0`. Do not tune the
selector, change boundary localization, or run structure-manifest6, pilot42,
holdout100-v2, broad500, full5050, or any additional real row under this card.

## Experiment card

### Objective

Test whether the post-Exp017 short-ABA mechanism failure is caused by treating a
coarse 6-second raw-audio tempo observation as an exact boundary locator. Change
only compatibility for already-generated, phase-continuous closed A-to-B-to-A
candidates. Keep candidate generation, retention, evidence extraction, scores,
ranking, thresholds, caps, fallback policy, and metrics unchanged.

### Source state and evidence boundary

- Baseline source:
  - `src/pulsefield_model/timing/v3/tempo_track.py` SHA-256
    `36a793e743ffbd6fed0b951e807e6bb91c579a4b86e8368ea7b99146cce868c6`
  - `tests/timing/test_timing_v3_exp017_tempo_track.py` SHA-256
    `95fa4bd47e0236767390a9e64676b3676f21bef5c4f63dfa0c54cf1bfdc3527d`
- Authoritative negative input:
  - `artifacts/reports/timing/timing_v3_exp017_mechanism2_authoritative_v1.jsonl`
  - SHA-256
    `4d6b470a586f57f9343789267f35bcc9f8b8db722d192d7c9ea7cbe3d4e29db8`
  - summary:
    `artifacts/reports/timing/timing_v3_exp017_mechanism2_authoritative_summary_v1.json`
  - summary SHA-256
    `9db1e546a3fb40dc589e58f58b88a8093ff26ab837aaf757302885d67fd15752`
- Allowed real evidence remains exactly the two already-exposed mechanism rows:
  - stable `2300685`;
  - short-ABA `618173`.
- This card does not authorize structure-manifest6, pilot42, holdout100-v2,
  broad500, full5050, or any additional real row or artifact.
- Weak truth is post-freeze evaluation only. It must not enter inference,
  compatibility, ranking, thresholds, or row branching.

### Background facts

Exp017 passed source-only retention guards but failed both authoritative mechanism
gates:

- stable `2300685` remained the correct constant `200 BPM` result;
- `618173` still selected the old persistent false structure,
  `175 -> 158.426 BPM`, source `raw_run_persistent_a_to_b_start`;
- that selection predicted one boundary, matched none, and had weak boundary
  recall `0.0`;
- authoritative two-row p90 runtime was `3.485 s`: below the Phase 1 hard
  `<5 s` limit, but above Exp017's stricter `3.0 s` mechanism gate.

Exp017 did retain a source-supported closed ABA candidate:

- candidate index `20`;
- source `paired_unmerged_boundary`;
- sections `175.193 -> 146.621 -> 175.193 BPM`;
- middle interval `56477.345 -> 58932.654 ms`;
- positive paired raw gain `+0.0080614`;
- exact phase-continuous return to the outer tempo;
- not eligible under the current compatibility check.

The coarse raw observation for the same row spans approximately six seconds and
produces a down-tempo run. The current `_section_matches_raw_run` check first
requires direction, tempo proximity, and at least `500 ms` overlap, then also
requires the closed middle section's start and end to lie within an anchor
tolerance around the run centers. This uses a coarse observation both as a
presence gate and as an exact boundary estimator. Candidate `20` passes the
coarse direction/tempo/overlap evidence but fails the exact start-anchor test.

### Hypothesis

For a closed A-to-B-to-A proposal whose outer sections are primary-tempo
consistent, raw audio should gate the presence and direction of a local tempo
excursion, while the candidate generator and BeatThis evidence localize its
boundaries. If the redundant exact raw-run boundary-anchor requirement is the
blocking mechanism, removing only that requirement for closed ABA candidates
will make a retained paired ABA eligible and selected on `618173`, while stable
`2300685` remains constant.

### Non-goals

- No new candidates and no retention changes.
- No score or ranking-order changes.
- No raw feature/window/hop changes.
- No tempo-ratio, direction, or minimum-overlap threshold changes.
- No compatibility relaxation for persistent A-to-B or B-to-A candidates.
- No hard-coded BPM, duration, row id, filename, or weak boundary values.
- No ramp production eligibility.
- No cap, fallback, metric, or runner changes.

## Candidate variants

### Variant A: current exact boundary-anchor compatibility

Use the coarse raw run as both presence evidence and an exact boundary locator.

Decision: reject as the frozen Exp017 negative baseline.

### Variant B: remove all raw-run compatibility checks

Let any positive-gain jump compete regardless of raw direction, tempo, or time.

Decision: reject. This removes the independent structural guard and risks stable
false positives.

### Variant C: coarse compatibility for every jump topology

Keep only direction, tempo ratio, and overlap for closed and persistent jumps.

Decision: reject. A persistent change must still be supported at the relevant
song edge; coarse overlap alone cannot establish persistence.

### Variant D: coarse compatibility only for closed ABA

For a bounded non-primary middle section bracketed by primary-consistent outer
sections, retain the existing direction, canonical tempo-ratio, and minimum
overlap checks, but compare the two middle-section edges with the existing
coarse-window `edge_tolerance_ms` instead of the tighter
`anchor_tolerance_ms`. Keep the current exact edge/anchor logic for persistent
and all other structures.

Decision: execute only Variant D.

## Selected mutation

Change only `_section_matches_raw_run` in
`src/pulsefield_model/timing/v3/tempo_track.py`.

Bump `TEMPO_TRACK_VERSION` and the result-dump schema from Exp017 v1 to Exp018
v1 as a provenance-only change. No serialized field shape or runner behavior may
change. Remove the now-superseded Exp017 test that asserts the process-wide
current version can never advance beyond Exp017; all Exp017 retention behavior
guards remain unchanged, and the new Exp018 test owns the exact current-version
assertion.

After the existing checks pass for:

1. raw-run direction versus section tempo delta;
2. primary-canonical tempo distance within the existing
   `raw_run_jump_max_tempo_ratio`;
3. at least the existing `raw_run_jump_minimum_overlap_ms`; and
4. valid scoreable candidate/audio coverage;

detect a closed ABA middle section algorithmically:

- the candidate has exactly three constant sections;
- the matched section is the middle section;
- both neighboring outer sections are primary-alias-consistent;
- the two outer sections have exactly equal BPM values, matching the existing
  closed-ABA proposal construction rather than merely sharing a rational alias;
- the middle section touches neither candidate edge; and
- the curve is already phase-continuous on one integer beat axis.

For only that topology, use the already-computed `edge_tolerance_ms` for the
subsequent raw-run start/end anchor tests. The current 6-second local window and
1-second run expansion make this approximately `4.0 s` on the exposed mechanism,
instead of the current approximately `1.17 s` tight anchor. Do not introduce a
new numeric threshold. Every other topology continues through the existing
`anchor_tolerance_ms` and song-edge logic unchanged.

The mutation must not inspect source strings, weak labels, row metadata, BPM
nice-number status, or oracle timing.

## Test plan

No real inference may run until focused and related source guards pass.

### Source-only guards

1. Closed ABA coarse acceptance
   - Construct a three-section primary-to-nonprimary-to-primary curve.
   - The middle section has correct direction, allowed tempo ratio, and at least
     `500 ms` overlap with a coarse raw run.
   - Place one or both middle boundaries outside the existing exact anchor
     tolerance.
   - Assert compatibility succeeds under `edge_tolerance_ms`.
   - Assert it still fails if either boundary also exceeds
     `edge_tolerance_ms`.
2. Closed ABA evidence guards
   - Assert opposite direction fails.
   - Assert tempo ratio outside the existing threshold fails.
   - Assert overlap below the existing threshold fails.
3. Persistent strictness
   - Construct a two-section persistent jump that would pass coarse overlap but
     fails the existing boundary/song-edge evidence.
   - Assert it remains incompatible.
4. Topology strictness
   - Assert a three-section curve whose outer sections are not both
     primary-consistent does not use the relaxation.
   - Assert a three-section curve whose outer sections are individually
     primary-alias-consistent but have unequal BPM values does not use the
     relaxation.
   - Assert multi-step and ramp curves do not use the relaxation.
5. Selector isolation
   - Freeze candidate curves and raw/collapsed scores in a synthetic fixture.
   - Assert the only eligibility-set change is the closed ABA candidate.
   - Assert ranking remains the existing paired-raw-gain-first order.
6. Existing regression suite
   - Exp014, Exp017, base tempo-track, analytic-curve, and raw-audio-evidence
     focused tests must remain green.
   - Total candidates remain `<=64`, jumps `<=44`, ramps `<=8` diagnostic-only.
   - Assert both tempo-track provenance constants identify Exp018 v1.

### Mechanism-only gate

Run exactly the same two exposed rows and no others.

Stable `2300685` must:

- remain `v3_accepted`;
- select the same constant `200 BPM` curve;
- retain weak exact hit, direct BPM coverage `1.0`, and seam max `0.0 ms`.

Short-ABA `618173` must:

- retain and select a three-section closed ABA candidate;
- not select `raw_run_persistent_a_to_b_start` or another persistent topology;
- improve weak boundary recall above `0.0` in post-freeze evaluation;
- preserve seam max `0.0 ms`;
- report direct and alias BPM coverage, phase mean/p90, both boundary errors, and
  exact-hit status even if exact hit remains false.

If the mechanism passes, stop and freeze the result. A separate accepted card is
required before any additional real row.

## Metrics and runtime

Primary:

- closed ABA becomes eligible in the synthetic isolation fixture;
- persistent false structure remains subject to strict edge evidence;
- `2300685` stays constant;
- `618173` selects closed ABA and boundary recall improves above `0.0`.

Secondary:

- selected source/fingerprint/sections;
- eligible candidate inventory;
- paired raw gain and generalized BeatThis support rank;
- direct and alias BPM coverage;
- boundary precision/recall and signed/absolute errors at the frozen evaluator
  tolerance;
- phase mean/p90, endpoint drift, seam max;
- per-row and p90 runtime.

Runtime mutation is selector-only and must not create feature or scoring calls.
Use Exp017 authoritative p90 `3.485 s` as the local reference:

- target p90 `<=3.85 s` (`+10.5%` tolerance for two-row noise);
- hard kill at `>=5.0 s`.

## Kill criteria

Kill immediately if:

1. any source-only guard fails;
2. a stable/no-run fixture becomes jump-selected;
3. persistent candidates receive the closed-ABA relaxation;
4. direction, tempo-ratio, or minimum-overlap guards are weakened;
5. candidate generation, retention, ranking, evidence extraction, caps, or
   fallback changes are required;
6. any inference branch uses weak truth, row id, filename, or exposed target
   BPM/boundary values;
7. either real mechanism row fails, is missing, or any third row is accessed;
8. `2300685` regresses;
9. `618173` remains persistent-selected or boundary recall stays `0.0`;
10. seam max exceeds `0.0 ms` beyond serialization tolerance;
11. p90 runtime reaches `>=5.0 s`.

If candidate `20` becomes eligible but a persistent candidate still wins under
the frozen ranking, freeze Exp018 as negative. Do not change ranking in this
card; draft a separate arbitration experiment.

## Expected interpretation

Positive:

- the raw-run anchor was over-constraining closed ABA boundary localization;
- coarse raw evidence plus independently proposed boundaries is sufficient for
  the exposed mechanism;
- broader structure evidence still requires a new card.

Negative with candidate still ineligible:

- another compatibility guard is responsible; inspect it without changing
  ranking.

Negative with candidate eligible but persistent selected:

- compatibility is fixed, but arbitration remains wrong; isolate ranking in a
  subsequent card.

## Pre-mortem

- Accidentally widening every three-section curve rather than only a
  primary-bracketed middle excursion.
- Weakening persistent song-edge checks through shared control flow.
- Treating raw overlap as a new boundary estimator rather than a coarse gate.
- Changing rank order while adding a compatibility test.
- Letting a source string or exposed BPM identify the special case.
- Using the weak comparator before frozen inference.
- Broadening real evaluation after a two-row pass without a new card.
