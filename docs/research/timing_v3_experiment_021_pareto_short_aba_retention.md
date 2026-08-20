# Timing v3 Experiment 021: Pareto-front short-ABA retention

Status: completed / mechanism pass; stopped before broader rows

## Experiment card

### Objective

Test one retention-order variable: within the already-reserved fourteen
`short_aba_paired_boundary` slots, preserve the source-only non-dominated
trade-offs between the unchanged blended proposal score and the already-carried
pure BeatThis `aba_support_delta` before filling remaining slots by the current
pure-support order.

Do not change proposal construction, search breadth, evidence features, either
score, compatibility, production arbitration, family quotas, caps, fallback,
or curve representation.

### Source state and evidence boundary

- Baseline source:
  - `src/pulsefield_model/timing/v3/tempo_track.py` SHA-256
    `5348cc4f55f700736e25b6243f5c299d320cf6d8a8209511e20f9790899ab7ac`;
  - `tests/timing/test_timing_v3_exp019_tempo_track.py` SHA-256
    `798eae548505f9fd3d6b0b9b6885ea483426ee86e79bc73be912d61976555cbf`.
- Frozen Exp020 diagnostic:
  - pre-cap inventory
    `artifacts/reports/timing/timing_v3_exp020_618173_precap_short_inventory_v1.json`,
    SHA-256
    `c78cd263389d496c11bff5baf93963fe0dc54e360953001c048f8543960d168f`;
  - post-freeze audit
    `artifacts/reports/timing/timing_v3_exp020_618173_precap_short_audit_v1.json`,
    SHA-256
    `5df0afae4c8fb76e94bf52df3b5ad96f0309890d5660867d5a64cb70fa6b6b62`;
  - one-row unchanged-runner output
    `artifacts/reports/timing/timing_v3_exp020_618173_runner_v1.jsonl`,
    SHA-256
    `a18b9f7d0e8e520be959bb98f93998eca30378a461a1980cafe0d6653e5d2c01`.
- Allowed real evidence is exactly the two already-exposed mechanism rows:
  stable `2300685` and short-ABA `618173`.
- Do not access structure-manifest6, pilot42, holdout100-v2, broad500,
  full5050, or any additional real row/artifact under this card.
- Weak truth is post-freeze evaluation only. It cannot enter retention,
  ordering, branching, scores, thresholds, or inference payloads.

### Background facts

Exp020 captured the complete source-only short-family inventory before the
unchanged Exp019 retention delegate:

- `27,946` input jump proposals;
- `1,958` unique short-ABA fingerprints;
- seven post-freeze goal-compatible proposals before retention;
- zero goal-compatible proposals retained and zero selected.

The strongest goal-compatible proposal, fingerprint
`e7e86f7c6828b3089c45b0dbef4e03ea70db5a9b8db3d4a70c7f8d924c58bfe9`,
was generated from `paired_unmerged_boundary`. It had unchanged blended rank
`22`, pure-support rank `50`, and support delta `-0.061745`. Post-freeze only,
it described `175.193 -> 143.964 -> 175.193 BPM` over
`54764.951 -> 58932.654 ms`, passed direct BPM, and matched both exposed weak
boundaries under the frozen fixed `+/-1000 ms` audit.

The same frozen source-only inventory has exactly three first-front proposals
under the two maximization objectives `(proposal.score, aba_support_delta)`:

1. blended/support ranks `(1, 56)`;
2. the compromise above at `(22, 50)`;
3. the Exp019 selected support extreme at `(62, 1)`.

Thus the required candidate is not merely a low-quality tail item. Pure-support
sorting discards a non-dominated trade-off and selects the least disruptive,
near-primary change. This motivates a Pareto reservation, not a row-specific
BPM, duration, boundary, source, or nice-number rule.

### Metric alignment note

Retain the unchanged strict weak-boundary matcher and additionally report the
already-frozen deterministic greedy one-to-one fixed `+/-1000 ms` audit after
the selected fingerprint and inference payload are frozen. The fixed audit
sorts candidate pairs by
`(absolute_error_ms, predicted_time_ms, weak_time_ms, predicted_index,
weak_index)` and greedily accepts unmatched indices. It is evaluation-only.

### Hypothesis

If the current failure is the scalarization of two legitimate source evidence
objectives, reserving their first non-dominated front before current
pure-support fill will retain the useful compromise without increasing any cap.
The unchanged Exp018/Exp019 compatibility and paired-raw-gain arbitration will
then select a correctly localized closed ABA, while stable material remains on
the constant lane.

### Non-goals

- No quota or total-cap change.
- No new proposal, seed, search branch, or score.
- No direction, BPM, duration, source, filename, row-id, or nice-number bins.
- No weighted sum, tuned coefficient, oracle-aware tie break, or label-derived
  threshold.
- No selector, compatibility, raw-gain, BeatThis-support, feature, or fallback
  change.
- No ramp production eligibility.
- No broader or protected real evaluation.

## Candidate variants

### Variant A: current pure-support order

Keep Exp019 unchanged.

Decision: reject as the frozen negative baseline.

### Variant B: increase the short quota or global cap

Decision: reject. It changes resource allocation and is unnecessary because
the useful proposal is already a source-only non-dominated point.

### Variant C: hand-authored tempo, duration, direction, or nice-number bins

Decision: reject. It adds policy surface adjacent to the exposed row and mixes
the separate stable-only Exp015 hypothesis into jump retention.

### Variant D: retune a linear combination of blended and support scores

Decision: reject. A new coefficient is scale-dependent and would be tuned on a
single exposed mechanism.

### Variant E: first Pareto front, then current pure-support fill

Decision: execute only Variant E.

## Selected mutation

In `src/pulsefield_model/timing/v3/tempo_track.py`, for only the first-pass
reservation of `short_aba_paired_boundary`:

1. Keep the existing fingerprint dedupe and the unchanged proposal objects.
2. Treat only proposals with both finite `score` and finite
   `aba_support_delta` as Pareto inputs. A proposal with a non-finite value on
   either objective does not enter the front and falls back to its unchanged
   position in the current pure-support fill order.
3. A finite proposal `p` dominates finite proposal `q` iff:
   - `p.score >= q.score`;
   - `p.aba_support_delta >= q.aba_support_delta`; and
   - at least one comparison is strict.
4. Compute the first non-dominated front deterministically from exact stored
   float values. Do not round, normalize, or introduce an epsilon.
5. Place all members of that first front before dominated short proposals.
6. Order members inside the front with the existing
   `_short_aba_support_retention_order_key`; this freezes behavior if a future
   front alone exceeds the fourteen-slot quota.
7. Fill remaining short slots with the current pure-support order, skipping
   already-retained fingerprints. Proposals with missing/non-finite support
   retain their current position behind finite support.
8. All other family passes and the global backfill keep the existing global
   blended order exactly.
9. Keep quotas `14/10/8/6/6`, jump cap `44`, ramp cap `8`, and total cap `64`.
10. Bump tempo-track and result-dump provenance to Exp021 v1. Do not change the
    serialized product field shape.

This is a reservation-order mutation only. Pareto membership must not be
serialized as a product score or used by production arbitration.

## Test plan

No real inference may run until source-only guards pass.

1. Three-point trade-off frontier
   - Build more than fourteen valid short paired-ABA proposals.
   - Include a blended-score extreme, a pure-support extreme, and a compromise
     that each are mutually non-dominated.
   - Include dominated proposals that pure-support ordering would otherwise use.
   - Assert all three frontier fingerprints survive and the dominated items do
     not displace them.
2. Exact dominance and ties
   - Assert equality in both objectives does not create unstable duplication;
     one strict objective plus one equal objective dominates.
   - Assert no epsilon or rounded comparison is used.
3. Missing/non-finite objectives
   - Assert support `None`, NaN, and infinities, plus score NaN, `+inf`, and
     `-inf`, do not enter the frontier and fall back to the unchanged
     pure-support fill order.
4. Front overflow
   - Create more than fourteen mutually non-dominated proposals.
   - Assert the hard quota remains fourteen and the frozen existing support key
     deterministically chooses the prefix.
5. Non-short and backfill isolation
   - Freeze persistent, long, multi-step, overflow, and global-backfill
     fingerprints/order; assert byte-identical behavior to Exp019 fixtures.
6. Dedupe, determinism, caps, and product guards
   - Duplicate fingerprints consume one slot.
   - For one duplicate fingerprint, make the high-blended representative and
     high-support representative disagree; assert the existing pre-family
     blended-score dedupe still chooses the high-blended representative before
     Pareto membership is computed.
   - Repeated input permutations produce identical retained fingerprints.
   - constants remain available; jumps `<=44`, ramps `<=8`, total `<=64`;
     ramps remain diagnostic-only; no Pareto field is serialized.
7. Provenance
   - Exact Exp021 tempo-track and result-dump version assertions.
8. Related regression
   - analytic curve, raw evidence, base tempo track, Exp014, Exp017, Exp018,
     and Exp019 focused suites remain green after updating only the superseded
     exact-current-version assertion.

## Mechanism-only gate

Run exactly stable `2300685` and short-ABA `618173` through the same sanitized,
oracle-blind inference path used by the recent authoritative mechanism runs.
Freeze output bytes and selected fingerprints before weak evaluation.

Stable must:

- stay `v3_accepted` constant `200 BPM`;
- retain direct coverage `1.0`, weak constant exact `true`, and seam max
  `0.0 ms`;
- show no fallback or candidate-cap regression.

Short ABA must:

- retain and select a three-section paired/virtual closed ABA, not a persistent
  or raw-run-only topology;
- have equal, primary-consistent outer tempos;
- achieve full-song direct BPM coverage `>=0.95`;
- emit exactly two significant boundaries;
- achieve fixed `+/-1000 ms` boundary precision `1.0` and recall `1.0`;
- report the unchanged strict weak-boundary metrics even if stricter than the
  product goal;
- keep phase p90 `<=70 ms` and seam max `0.0 ms`.

Runtime:

- target two-row p90 `<=3.85 s`;
- hard kill at `>=5.0 s`.

If the mechanism passes, freeze the result and stop. A new accepted card is
required before any additional real row.

## Kill criteria

Kill immediately if:

1. any source-only guard fails;
2. any non-short family or global backfill changes;
3. generation, search breadth, score formulas, quota, or cap changes;
4. selector, compatibility, raw evidence, fallback, or evaluator semantics
   change;
5. inference uses weak truth, row id, filename, exposed BPM, or exposed
   boundaries;
6. any third real row or unauthorized artifact is accessed;
7. stable `2300685` regresses;
8. `618173` is not a selected closed paired/virtual ABA;
9. fixed-1-second boundary precision or recall is below `1.0`;
10. direct BPM coverage is below `0.95`, phase p90 exceeds `70 ms`, or seam max
    exceeds serialization tolerance;
11. p90 runtime reaches `>=5.0 s`.

Do not repair arbitration under this card if the intended proposal is retained
but not selected. Freeze the negative and isolate the next variable.

## Expected interpretation

- Pass: scalar pure-support pruning was the missing mechanism; freeze Exp021
  and request a new card for any broader exposed pilot.
- Candidate retained but not selected: stop and isolate arbitration.
- Candidate still pruned: stop; first-front implementation or evidence identity
  is inconsistent with Exp020.
- Stable regression: stop; the retention mutation is not safely isolated.

## Authoritative result log

### Execution identity and scope

- Mode: executor.
- The accepted Experiment Card existed before implementation and execution.
- Tempo-track source SHA-256:
  `fc4153a6310a4db233e1fbd29e87a57775eff924e4904e925773d172e0d7de85`.
- Exp021 test SHA-256:
  `d543ec827a893fb1dddc3513edbe6df2396138ace5d96a58c679002dfeee3ae5`.
- Related source-only verification: `99 passed`.
- Real-data scope remained exactly stable `2300685` and short-ABA `618173`.
  No third row, structure-manifest6, pilot42, holdout100-v2, broad500,
  full5050, or protected evaluation was accessed under Exp021.

Authoritative artifacts:

- output
  `artifacts/reports/timing/timing_v3_exp021_mechanism2_authoritative_v1.jsonl`,
  SHA-256
  `18944665c5d91e4435abf1eddc65bc102c6b4748448eb854560bd0f7aee04178`;
- summary
  `artifacts/reports/timing/timing_v3_exp021_mechanism2_authoritative_summary_v1.json`,
  SHA-256
  `69d1ca49510ffb89af74f798ada82fe63c5c459d49d3840c4ebd2c5cee476f40`;
- pre-oracle freeze manifest
  `artifacts/reports/timing/timing_v3_exp021_mechanism2_authoritative_freeze_v1.json`,
  SHA-256
  `67fb12b32f4dc6372bef91efc6b3a4a353707d8988966664255739359c5605d3`;
- post-freeze audit
  `artifacts/reports/timing/timing_v3_exp021_mechanism2_authoritative_audit_v1.json`,
  SHA-256
  `0d99998e9fcacc454089152b750ae937036f98ba9cb660728e5c30c0f96d605d`.

The freeze manifest records exactly two candidate-generator calls, unchanged
cache snapshots before and after inference, and zero pre-freeze oracle calls.
Both selected fingerprints and frozen inference payloads were committed before
the audit's two post-freeze oracle reads.

### Result

Both rows were `v3_accepted`; there were no fallbacks or hard failures, and
maximum seam error was `0.0 ms`.

Stable `2300685` stayed on the constant lane:

- selected constant `200 BPM`;
- direct BPM coverage `1.0` and weak constant exact `true`;
- phase p90 `31.00000000000591 ms`;
- maximum seam error `0.0 ms`;
- row runtime `3.5332770830136724 s`.

Short-ABA `618173` retained and selected fingerprint
`e7e86f7c6828b3089c45b0dbef4e03ea70db5a9b8db3d4a70c7f8d924c58bfe9`
from `paired_unmerged_boundary`. Its phase-continuous three-section curve was:

- `175.1933640630715 -> 143.96419309635093 -> 175.1933640630715 BPM`;
- boundaries at `54764.95117233091 ms` and `58932.65417608017 ms`;
- direct and alias BPM coverage `0.9880358923230309`;
- phase p90 `40.9291993514992 ms`;
- endpoint relative drift `85.63851241740865 ms`;
- maximum seam error `0.0 ms`;
- row runtime `3.0026188329793513 s`.

The frozen fixed `+/-1000 ms` greedy audit matched both boundaries with
absolute errors `359.0488276690885 ms` and `363.34582391982985 ms`, giving
precision `1.0` and recall `1.0`. The unchanged strict matcher reported
matched count `0`, precision `0.0`, and recall `0.0`; this is retained as an
honest stricter diagnostic and does not replace the goal-aligned fixed-window
gate.

Two-row p90 runtime was `3.4802112580102404 s`, below the `3.85 s` target;
the maximum runtime was `3.5332770830136724 s`, below the `5.0 s` hard limit.
All mechanism gates passed.

### Closed-loop outcome

- Selected variant passed local and mechanism verification: yes.
- Positive signal: reserving the first exact Pareto front recovered the
  source-evidence compromise that pure-support ordering had pruned, and the
  unchanged compatibility/arbitration selected it.
- Stable guard, candidate caps, fallback behavior, phase, seam, boundary,
  direct-coverage, and runtime gates all passed.
- Kill criteria triggered: none.
- Classification: positive mechanism result, not a generalization result.
- Decision: keep and freeze Exp021, then stop. Any evaluation on additional
  real rows requires a new accepted Experiment Card.
- Next-loop action: `TEST` only through a separately frozen broader exposed
  pilot; do not broaden under Exp021.

This result supports the narrow claim that scalar pure-support pruning was the
missing mechanism on the exposed short-ABA row. It does not establish corpus
coverage, broad stable/jump discrimination, protected-set performance, or ramp
production readiness.
