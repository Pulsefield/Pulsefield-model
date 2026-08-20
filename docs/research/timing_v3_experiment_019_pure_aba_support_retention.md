# Timing v3 Experiment 019: Pure-ABA-support short-family retention

Status: stopped / negative

## Experiment card

### Objective

Test one retention-order variable: within the already-reserved
`short_aba_paired_boundary` slots, prefer candidates with stronger pure
BeatThis ABA support before the blended generation score. Do not change search
breadth, family quotas, selector eligibility, raw evidence, candidate scores,
ranking after retention, caps, fallback, or curve representation.

### Source state and evidence boundary

- Baseline source:
  - `src/pulsefield_model/timing/v3/tempo_track.py` SHA-256
    `ff03b42a40e70c935cade460b3bcc89f3c3f1d6f1e610e88d35d924548e87e06`
  - `tests/timing/test_timing_v3_exp017_tempo_track.py` SHA-256
    `3821bde19dbcc2160e03838770712b20c0f28dcf90c7e5934ba2b9c4dc4ead01`
  - `tests/timing/test_timing_v3_exp018_tempo_track.py` SHA-256
    `833e6371c7c115e0247d045575d9c4077d7361447890b643e8a126011e4abf12`
- Authoritative Exp018 negative:
  - `artifacts/reports/timing/timing_v3_exp018_mechanism2_authoritative_v1.jsonl`
  - SHA-256
    `9b9d9d437f417af172c3c0a78e8c0283bde3e09d63f95c774e420c533361b840`
  - summary
    `artifacts/reports/timing/timing_v3_exp018_mechanism2_authoritative_summary_v1.json`
  - summary SHA-256
    `c04187ff5a90112193312444a8e138a3ff21b86be7e93bffb0c40d6aec49cf9a`
- Allowed real evidence is exactly the same two already-exposed mechanism rows:
  stable `2300685` and short-ABA `618173`.
- Do not access structure-manifest6, pilot42, holdout100-v2, broad500,
  full5050, or any additional real row/artifact under this card.
- Weak truth is post-freeze evaluation only and cannot enter inference,
  retention, ordering, thresholds, or row branching.

### Background facts

Exp018 fixed compatibility enough to select a closed ABA instead of the old
persistent false structure:

- stable `2300685` remained exact constant `200 BPM`;
- `618173` selected retained candidate `20`, source
  `paired_unmerged_boundary`;
- selected structure:
  `175.193 -> 146.621 -> 175.193 BPM`;
- selected middle interval:
  `56477.345 -> 58932.654 ms`;
- two boundaries were emitted with exact `0.0 ms` seams;
- direct/alias BPM coverage improved to `0.930874`;
- strict weak boundary recall remained `0.0`, so Exp018 stopped.

The current short-family reservation still orders by the blended proposal score:

`aba_support_delta + boundary-rank bonus + local-observation bonus + preferred-BPM bonus`.

This mixture can let correlated boundary-rank pressure displace a candidate with
better direct beat-versus-half-beat ABA support. `_aba_support_delta` is already
computed for every paired/virtual ABA spec before retention, but it is discarded
after being blended into the aggregate score.

An earlier already-exposed diagnostic found a wider-pool closed ABA near
`175.193 -> 143.964 -> 175.193 BPM`, about `4.168 s`, whose pure ABA support was
stronger than retained competitors. This fact motivates the mechanism but must
not define any BPM, duration, boundary, or row-specific rule.

### Metric alignment note

The current `weak_oracle_boundary_recall` matcher uses a tolerance of one half
of the smallest adjacent beat period, capped at `2 s`. Around 143–175 BPM this
is only about `0.17–0.21 s`, materially stricter than the goal contract's
transition-boundary target of `<=1 s`.

Exp018 remains negative under its preregistered strict gate. Exp019 must report
both metrics after the prediction fingerprint is frozen:

1. the unchanged strict weak matcher; and
2. a transparent greedy one-to-one fixed `+/-1000 ms` boundary audit matching
   the goal contract.

The fixed audit reuses the existing significant-boundary extraction. It forms
all predicted/weak pairs with absolute time error `<=1000 ms`, sorts pairs by
`(absolute_error_ms, predicted_time_ms, weak_time_ms, predicted_index,
weak_index)`, greedily accepts unmatched indices, and reports matched/predicted
precision plus matched/weak recall. This rule is frozen before Exp019 inference.

The fixed-1-second audit is evaluation-only. It cannot affect candidate
retention or selection. No evaluator threshold is changed in production.

### Hypothesis

If short-family pruning is now the limiting mechanism, retaining the fourteen
reserved short ABA proposals by pure `_aba_support_delta` will preserve an
evidence-strong, better-localized ABA candidate. The unchanged Exp018 selector
will then choose it by its paired raw gain, while stable material remains on the
constant lane and runtime remains under the Phase 1 budget.

### Non-goals

- No quota or total-cap changes.
- No new proposals or search breadth.
- No selector, compatibility, raw gain, BeatThis support formula, feature, or
  fallback change.
- No direction, BPM, duration, nice-number, filename, or row-specific bins.
- No use of weak boundaries or BPMs before frozen inference.
- No ramp production eligibility.
- No protected or broader real evaluation.

## Candidate variants

### Variant A: current blended-score ordering

Keep the Exp018 retention order.

Decision: reject as the frozen negative baseline.

### Variant B: increase short-family quota or global cap

Reserve more than fourteen short ABA candidates or raise the 44/64 caps.

Decision: reject. This changes resource allocation and does not isolate which
evidence should own short-family retention.

### Variant C: direction, tempo-ratio, or duration bins

Subdivide short candidates into hand-authored bins.

Decision: reject. It adds exposed-row-adjacent policy surface and can waste
slots even when direct structural evidence is available.

### Variant D: pure-ABA-support first-pass ordering

Carry the already-computed `_aba_support_delta` into internal proposal state and
use it only for the first-pass ordering of reserved
`short_aba_paired_boundary` slots.

Decision: execute only Variant D.

## Selected mutation

In `src/pulsefield_model/timing/v3/tempo_track.py`:

1. Add internal optional field `aba_support_delta: float | None = None` to
   `_CurveProposal`.
2. When `_jump_proposals` constructs a paired/virtual closed ABA whose middle
   duration is in the frozen short interval `[2.0 s, 8.0 s]`, store the
   already-computed finite `support_delta` in that field without changing the
   existing blended `score`.
3. Paired/virtual ABA longer than `8.0 s`, raw-run persistent, raw-run ABA,
   long, multi-step, overflow, constant, and ramp proposals keep
   `aba_support_delta = None`.
4. Keep the existing fingerprint dedupe and global order.
5. For only the first-pass reservation of family
   `short_aba_paired_boundary`, use this deterministic key:
   - finite support values before `None`;
   - descending `aba_support_delta`;
   - descending unchanged blended `score`;
   - source string;
   - canonical curve fingerprint;
   - canonical boundary-time tuple.
6. All other family first passes and global backfill keep the existing
   `_jump_retention_order_key` exactly.
7. Keep quotas `14/10/8/6/6`, jump cap `44`, ramp cap `8`, and total cap `64`.
8. Bump tempo-track and result-dump provenance to Exp019 v1 without changing
   serialized field shape or runner behavior.
9. Remove the superseded process-wide Exp018 exact-current-version assertion;
   Exp018 behavior/compatibility guards remain unchanged, and the new Exp019
   test owns the exact current provenance assertion.

The optional support field is internal retention state only. It must not be
serialized as a product score, used by the selector, or exposed as truth.

## Test plan

No real inference may run until source-only guards pass.

1. Pure support beats blended score within short quota
   - Create more than fourteen valid short paired ABA proposals.
   - Target has lower blended score but the strongest finite
     `aba_support_delta`.
   - Assert it survives the short-family first pass.
2. Missing support sorts behind finite support
   - A valid short proposal with `None` support cannot displace a finite-support
     proposal solely through blended score during the reserved pass.
   - Assert paired/virtual long ABA, raw-run ABA, overflow, constants, and ramps
     retain `aba_support_delta is None`.
3. Non-short isolation
   - Freeze persistent, long, multi-step, and overflow proposal inventories.
   - Assert their retained order/fingerprints are unchanged.
4. Backfill isolation
   - Assert global backfill still follows the unchanged blended global order.
5. Dedupe and determinism
   - Duplicate fingerprints consume one slot.
   - Repeated runs return identical fingerprints and family counts.
6. Cap and product guards
   - constants preserved;
   - jumps `<=44`, ramps `<=8`, total `<=64`;
   - ramps remain diagnostic-only;
   - selector ranking and Exp018 compatibility behavior tests remain unchanged.
7. Provenance
   - exact Exp019 tempo-track and result-dump version assertions.
8. Related regression
   - analytic curve, raw evidence, base tempo track, Exp014, Exp017, and Exp018
     focused suites remain green.

## Mechanism-only gate

Run exactly stable `2300685` and short-ABA `618173`.

Stable must:

- stay `v3_accepted` constant `200 BPM`;
- retain direct coverage `1.0`, weak constant exact `true`, seam max `0.0 ms`;
- show no candidate-cap or fallback regression.

Short ABA must:

- retain and select a three-section paired/virtual closed ABA, not a persistent
  or raw-run-only topology;
- have outer tempos equal and primary-consistent;
- achieve full-song direct BPM coverage `>=0.95`;
- emit exactly two significant boundaries;
- achieve fixed `+/-1000 ms` boundary precision `1.0` and recall `1.0` in the
  post-freeze goal-aligned audit;
- report the unchanged strict weak boundary metrics even if they remain zero;
- keep phase p90 `<=70 ms` and seam max `0.0 ms`.

Runtime:

- target two-row p90 `<=3.85 s`;
- hard kill at `>=5.0 s`.

If the mechanism passes, freeze the result and stop. A new accepted card is
required before any additional real row.

## Kill criteria

Kill immediately if:

1. any source-only guard fails;
2. support ordering affects a non-short family or global backfill;
3. candidate generation/search breadth or any cap changes;
4. selector, compatibility, scoring formulas, raw evidence, or fallback changes;
5. optional support state is serialized into the product contract;
6. inference uses weak truth, row id, filename, or exposed BPM/boundary values;
7. any third real row or unauthorized artifact is accessed;
8. stable `2300685` regresses;
9. `618173` is not a selected closed paired/virtual ABA;
10. fixed-1-second boundary precision or recall is below `1.0`;
11. direct BPM coverage is below `0.95` or phase p90 exceeds `70 ms`;
12. seam max exceeds serialization tolerance;
13. p90 runtime reaches `>=5.0 s`.

Do not alter selection or boundary thresholds if the evidence-strong candidate
is retained but not selected. Freeze the negative and isolate the next variable.

## Expected interpretation

Positive:

- blended boundary-rank pressure was crowding out the best direct ABA support;
- existing selector/compatibility can use the recovered candidate;
- broader generalization still requires a separate card.

Negative with target-like candidate still absent:

- pure support alone does not solve retention or the proposal is no longer in
  the generated pool; inspect candidate generation in a separate card.

Negative with candidate retained but not selected:

- retention is fixed and arbitration is the remaining mechanism.

Negative with fixed-1-second success but strict recall zero:

- record both honestly; this passes the goal-aligned mechanism but does not
  claim sub-beat boundary precision.

## Pre-mortem

- Accidentally replacing the blended generation score instead of carrying a
  separate retention-only field.
- Applying pure support to persistent or long/multi candidates.
- Re-sorting global backfill and silently changing every family.
- Letting `None` sort ahead of finite support.
- Using the exposed 143 BPM or 4.17-second values to define a bin.
- Reading weak truth before frozen inference.
- Treating a two-row mechanism pass as a generalization result.

## Authoritative result log

Authoritative output:

- `artifacts/reports/timing/timing_v3_exp019_mechanism2_authoritative_v1.jsonl`
- SHA-256
  `c23fcd39d1cba1e95b2b69e66abb88eff39848dc6c6dc0e1eedb8db094bcf86c`
- summary
  `artifacts/reports/timing/timing_v3_exp019_mechanism2_authoritative_summary_v1.json`
- summary SHA-256
  `8c9ab3d78ee3b9ca2f03462e4c6f89b85a6b3ebd748a52a056df5c7c167e665d`

Scope remained exactly the two already-exposed mechanism rows: stable
`2300685` and short-ABA `618173`. No structure-manifest6, pilot42,
holdout100-v2, broad500, full5050, protected, or broader row was evaluated
under Exp019.

Two rows ran, both were accepted, seam max remained `0.0 ms`, p90 row runtime
was `3.595412642 s`, and total runtime was `6.73549 s`.

Stable row `2300685` stayed on the constant lane:

- selected constant `200 BPM`;
- weak constant exact `true`;
- direct coverage `1.0`;
- runtime `3.652552 s`.

Short-ABA row `618173` selected retained candidate index `18` from
`virtual_right_beatthis`, a closed ABA:

- `175.193364 -> 170.151563 -> 175.193364 BPM`;
- first boundary `56477.345 ms`;
- second boundary `60356.240 ms`;
- paired raw gain `+0.0119691`;
- direct/alias coverage `0.913260`;
- phase p90 `40.8448 ms`;
- endpoint drift `717.623 ms`.

The selected topology was still a closed paired/virtual ABA, but the candidate
failed the mechanism gate. Direct/alias coverage remained below the `0.95`
requirement, endpoint drift worsened, and both boundary audits missed the
already-exposed weak boundaries at `55124 ms` and `59296 ms`.

Strict weak-boundary evaluation stayed at matched `0`, recall `0.0`. The frozen
goal-aligned `+/-1000 ms` audit also matched `0/2`: the nearest errors were
`1353.345 ms` and `1060.240 ms`, both outside the fixed window.

Decision: stop Exp019 as negative. Do not tune selector/retention further under
this card, and do not add broader rows. Pure support ordering rewarded
minimal-deviation near-base candidates, i.e. the least destructive closed ABA
near the existing tempo, while the target-like near-`143 BPM` candidate was
absent from the retained inventory. Candidate presence, pair-seed construction,
or proposal generation must be audited before any more selector or retention
tuning.
