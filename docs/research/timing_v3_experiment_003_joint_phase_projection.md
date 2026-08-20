# Timing v3 Experiment 003: Joint Anchor/BPM Phase Projection

## Mode and route

- Mode: critic
- Route: TEST
- Trigger: Experiment 002 accepted the beat-index representation but Family B
  missed the frozen mean-phase guard (`1.11043 > 1.10`). Its p90, fallback,
  seam, serialization, and section-count guards passed.
- Scope: mutate only the constant/jump adapter already declared as Family C.
  Do not add ramps, confidence weights, new BeatThis inference, `.osu` inputs,
  or long-track decomposition in this experiment.

## Core question

Can a parameter-free global compromise between moving source change anchors
and changing preceding BPMs remove v2 seams without the section-wide phase
regression caused by preserving every right anchor?

The candidate consumes only the v2 grid produced from the one cached BeatThis
prediction and its cache coverage. `.osu` redlines, objects, labels, metadata,
titles, and network evidence remain evaluation-only.

## Source and candidate coordinates

For source v2 segments `i = 0..m-1`, define:

```text
o_i       source offset (ms)
p_i       source beat length (ms)
delta_i   o_(i+1) - o_i                    for i < m-1
N_i^0     max(1, floor(delta_i / p_i + 0.5))
r_i(N)    delta_i - N_i * p_i
```

The half-up rule is unchanged; Python banker's rounding is forbidden.

For an integer vector `N`, Family C solves projected change anchors `tau_i`:

```text
tau_0       = o_0
p_hat_i     = (tau_(i+1) - tau_i) / N_i    for i < m-1
p_hat_(m-1) = p_(m-1)
d_i         = tau_i - o_i
d_0         = 0
```

The last source BPM remains fixed. The first source anchor remains beat zero;
the v3 schema extends its first lattice backward as required for coverage.

## Frozen source-displacement objective

The objective is a duration-weighted, same-beat displacement surrogate in
source-beat units. It is not described as exact physical-time phase error.

For each finite source interval:

```text
a_i = d_i
b_i = tau_(i+1) - (o_i + N_i*p_i)
    = d_(i+1) + r_i(N_i)

J_i = delta_i / (3*p_i^2) * (a_i^2 + a_i*b_i + b_i^2)
```

The residual in `b_i` is mandatory. If `b_i` were only the next-anchor
displacement, anchor-preserving Family B would incorrectly receive zero cost.

The fixed-final-BPM tail cost is:

```text
H      = max(0, coverage_end_ms - max(o_(m-1), coverage_start_ms))
J_tail = H * (d_(m-1) / p_(m-1))^2
```

If the source beat-zero anchor occurs after coverage start, the backward first
lattice also changes when `p_hat_0 != p_0`. Freeze:

```text
L        = max(0, o_0 - coverage_start_ms)
J_prefix = L^3 / (3*N_0^2*p_0^4) * (d_1 + r_0)^2
```

For a one-section grid, `J_prefix = 0` and the source grid is returned exactly.

The total is:

```text
J(N,d) = J_prefix + sum_i J_i + J_tail
source_surrogate_rms_beats = sqrt(J / coverage_duration_ms)
```

For fixed `N`, this is a convex chain quadratic with `d_0=0`. The source code
must assemble its tridiagonal system from the quadratic terms, solve in
float64 with a deterministic Thomas solver, and verify:

```text
||A*d - b||_inf
  <= 1e-10 * max(1, ||A||_inf*||d||_inf + ||b||_inf)
```

Nonfinite coefficients, a nonpositive pivot, or a failed normalized residual
is an explicit projection failure.

## Frozen candidates

Two Family C variants are evaluated:

- C0: solve once with `N = N^0`.
- C1: start from C0 and perform the deterministic nearby-count search below.

C1 is the primary hypothesis; C0 is the complexity ablation. Family A and B
remain whole-grid controls. They are never mixed boundary-by-boundary with C.
The fitter is not switched to C until this card's held-out gate accepts a
candidate.

## Deterministic integer search for C1

For interval `i`, the candidate set is exactly:

```text
{N_i^0 - 1, N_i^0, N_i^0 + 1} intersect positive integers
```

Candidates are deduplicated and visited in ascending order. Do not enumerate
the Cartesian product.

1. Start with `N = N^0` and its feasible C0 solution.
2. Visit boundaries from left to right.
3. At one boundary, try every alternate count and re-solve the entire chain.
4. Accept only a strictly better feasible whole-grid candidate.
5. Repeat complete sweeps until one sweep changes no count.
6. The cap is `m + 1` complete sweeps. Reaching the cap after a changing sweep
   is `search_not_converged`, not a silently accepted result.

Objective equality tolerance is:

```text
max(1e-12, 16 * max(ulp(left_J), ulp(right_J)))
```

Within tolerance, tie-break by:

1. smaller maximum anchor displacement in adjacent-local beats;
2. fewer counts different from `N^0`;
3. smaller `sum(abs(N_i - N_i^0))`; and
4. lexicographically smaller integer vector.

The result records every tried candidate count, sweep, accepted change,
objective, feasibility reason, and final replay fingerprint.

## Hard feasibility guards

Retain all Experiment 002 schema and coverage invariants, and freeze:

- every source BPM and source interval must be finite and in the 20-1000 BPM
  source guard;
- every `N_i >= 1`, every `tau_(i+1) > tau_i`, and every projected BPM is in
  20-1000;
- relative projected BPM adjustment is at most 5%:

  ```text
  abs(p_i / p_hat_i - 1) <= 0.05
  ```

- every interior anchor retains the same nearest beat on both adjacent source
  lattices, with a `1e-12` beat numerical allowance:

  ```text
  abs(tau_i - o_i) / p_i <= 0.5 + 1e-12
  abs(tau_i - (o_(i-1) + N_(i-1)*p_(i-1))) / p_(i-1)
      <= 0.5 + 1e-12
  ```

- section count cannot exceed the source count;
- the final source BPM is unchanged;
- the derived grid covers the exact cache support and has strict prefix time;
- JSON round-trip stays within `1e-6 ms` or eight ULPs; and
- all diagnostics are finite and JSON-safe.

Any violation yields `grid=None` and a specific tagged v2 fallback reason.
Exact v3 seams never erase the original residual or source-anchor movement.

## Source-only diagnostics

For C0, C1, A, and B, sample the source grid and candidate at the existing
20 ms evaluator hop without any oracle. Report wrapped source-relative phase
mean/RMS/p90/max in beats, source-relative endpoint and prefix drift, BPM
error, and active-section disagreement near moved boundaries.

The quadratic supplies only a surrogate dominance claim. No C candidate is
claimed to improve true `.osu` phase from its source-only score. Oracle metrics
remain the acceptance gate, and moving a boundary is separately audited.

## Frozen data progression

### Repair set

Run the unchanged 80-audio Experiment 002 pilot first. It is an exposed repair
and regression set, not a new holdout. No parameter, weight, candidate set,
guard, or tie-break may change after observing it.

### New 100-audio holdout

The original v1 holdout was invalidated before its formal run. During evaluator
plumbing, a worker ran `--limit 3` and saw aggregate output for the first three
v1 rows. The worker confirmed that no algorithm, objective, statistical
definition, threshold, or candidate was changed from those values, and the
formal evaluator never ran the remaining 97 rows. Nevertheless, all three
rows were `ramp_audit`, so changing only the random seed could not produce five
unexposed ramp rows from the seven-row post-pilot pool. The v1 manifest and its
derived files remain immutable audit evidence but are forbidden as a held-out
gate.

The protocol-v2 replacement is frozen before any formal C metric is inspected.
Create an audio-disjoint manifest from `timing_v3_labels_v1.jsonl`, excluding:

- all 80 pilot cache audio keys; and
- the three exposed v1 cache audio keys stored independently in
  `artifacts/reports/timing/timing_v3_exp003_protocol_exclusion_v1.json`, file
  SHA-256
  `120ff805ccbb925ca338045cd3bc40df7f973ac4eab9899e3959ae26d77ee5c7`.

The two exclusion sets must be present in the frozen labels, mutually
disjoint, persisted with independent source provenance, and replayed exactly.
Rank within each quota by:

```text
sha256("timing-v3-exp003-holdout100-v2\0" + cache_audio_key)
```

Use exclusive priority `ramp -> ambiguous -> long -> dense -> jump -> stable`
and these quotas:

| Quota | Count | Definition |
| --- | ---: | --- |
| stable | 40 | `label.stratum == stable` |
| jump | 25 | `label.stratum == jump_candidate` |
| dense | 10 | `label.stratum == dense` |
| ramp audit | 4 | every remaining `label.stratum == ramp_candidate` |
| long | 11 | `source.long_track == true` after higher priorities |
| anomaly | 10 | `label.stratum == ambiguous` after ramp priority |

Four unexposed ramp candidates remain, so the removed ramp slot is reassigned
before selection to the long quota. This is an identity-only protocol repair,
not an outcome-driven backfill rule. If any revised quota is unavailable, stop
rather than backfill another class. Store the manifest, complete source
objects, both exclusion hashes, quota assignment, and selection rank. The
broad-500 manifest remains forbidden until the protocol-v2 holdout gate passes.

The protocol-v2 identity artifacts were materialized after the evaluator and
split verifier were frozen, without running Family C:

- manifest
  `artifacts/reports/timing/timing_v3_exp003_holdout100_v2_manifest.json`,
  file SHA-256
  `87fc944f22abaf39ae5762dca57ec4153840b33a86839b17b2104fcd4211b5c4`;
- manifest fingerprint
  `7ae093565b18876e55c057fffddf710306c8f7dc0473d686cfaa3c2c0983d400`;
- selected full label rows SHA-256
  `d109a064ee2c72aa07d3a6091f5b20bf7b74c8703a7980bad9ba2b503071c0b7`;
- byte-preserving frozen-v2 baseline subset SHA-256
  `3b0151c6ff745335131318a777a13a7e06629314f3f6ffa5257ea88e27bf60f5`;
- pilot exclusion-set SHA-256
  `3a2504bbe9a0d632c4cbffc8fe1de17123e3a34f3c972c149711fb21348304a0`;
- protocol exclusion-set SHA-256
  `cec724e02837371cba4934c28a11b3fb52c3c28a5bbcff43bd5fd44bea559b60`.

The quotas are exactly `4/10/11/10/25/40` in priority order, with zero overlap
against either exclusion set. The audited baseline subset contains 100 fitted
rows, 92 comparison-eligible audio groups, eight comparator-unavailable groups,
and 261 successful map comparisons. These are denominator identities, not
Family C outcomes.

The formal-run freeze, recorded after repair80 and before opening holdout100,
is:

- joint projection source SHA-256
  `9fde5804a193d9858b1e1a97d4be561f8f2f0158fc0b9e4c8e04a741fc6a7ccf`;
- source-only comparator SHA-256
  `db7434462be79d2bc52049be0179f51bd8ce2df0348b94f8ac49c6334fcb7c2e`;
- projection evaluator SHA-256
  `d7bca7629049186f5a08991f772a551ba24af4db6337b12865556f8bd0ae2326`;
- split verifier SHA-256
  `b09655524c514b6bfd912ec9ca62b8fd5f00e52e99ae5a525833ef2b322907f9`;
- projection-config fingerprint
  `80906a2b3f081dcc223ba2258e2d8c5db67ac3a96a41488713cffc85551b1c88`;
- evaluator-behavior fingerprint
  `f08300b3b1a407ecf1c04d0e9fd7935817bcaf56464f160f0506d5014b7a94d9`.

### 500 and full corpus

If the 100 passes unchanged, the 500 manifest contains those 100 plus the 400
lowest ranks under seed `timing-v3-exp003-broad500-v2`, excluding both the
original 80 and the protocol-exposed three, and deduplicating by cache audio
key. No quota is applied to the added 400. Broad replay must recompute the
deterministic holdout from the exact label and exclusion provenance and require
full manifest equality.

The final gate evaluates all 5,050 audio groups, including the repair, exposed,
and holdout groups, after source/config hashes are frozen. Report a sensitivity
row excluding the three protocol-exposed groups, but do not remove them from
the required 5,050 denominator. Any algorithm or metric-definition change
restarts with a new card and a new audio-disjoint holdout.

## Required metrics and selection guards

Report C0 and C1 separately, plus v2/A/B controls.

Existing aggregate guards on every comparison-eligible matched population:

- candidate/v2 mean phase ratio <= 1.10;
- candidate/v2 p90 phase ratio <= 1.15;
- projection fallback <= 5% of every projection-evaluable population;
- serialized maximum seam <= 5 ms, expected near machine precision;
- no section-count increase; and
- no new cache/fit failure.

Before ratios are interpreted at any stage, the evaluator must match the
audited baseline artifact exactly: `projection_evaluable_audio_count` equals
the number of successful stored v2 fits, `comparison_eligible_audio_count`
equals the number of successful stored comparators, every eligible row is in
the paired denominator, and the remaining rows are accounted for as explicit
comparator-unavailable or baseline-unusable failures. A denominator mismatch
is a failed run, never a metric exclusion.

Additional Experiment 003 guards:

- stable, jump, and long holdout strata each have mean phase ratio <= 1.15 and
  p90 ratio <= 1.25 when at least five comparable audio groups exist;
- alias-normalized endpoint drift mean and p90, maximum-prefix drift mean and
  p90, and drift-slope mean and p90 are each <= 1.25 times matched v2;
- all anchor displacements satisfy the half-beat construction guard;
- solver normalized residual passes on every successful projection;
- deterministic replay produces byte-identical mathematical grid and integer
  search fingerprints; and
- runtime, source-surrogate metrics, BPM adjustment, boundary displacement,
  changed-count rate, original residuals, fallback reasons, and per-stratum
  outcomes are all present.

Dense, ramp-audit, and anomaly strata are mandatory robustness reports but are
not treated as ramp truth. For any such stratum with at least five comparable
audio groups, mean and p90 phase ratios above `1.50` are a catastrophic
regression and a review stop even though they are not ramp precision/recall
claims.

Stage decisions are frozen as follows:

- repair80 must pass the aggregate, fallback, seam, section-count, solver, and
  replay guards before the replacement holdout is opened;
- holdout100 must additionally pass the stable, jump, long, and drift guards;
- broad500 repeats every applicable holdout guard on its exact matched
  population and on stable/jump/long strata with at least five comparable
  groups; any failure blocks promotion and requires a new card, without
  changing C0/C1 under this card;
- full5050 repeats the same aggregate, drift, fallback, seam, section-count,
  solver, replay, and eligible-stratum guards. It must have 5,050
  projection-evaluable rows and 5,026 comparison-eligible rows when replaying
  the frozen full baseline hashes recorded by Experiment 001. A mismatch or
  guard failure rejects production promotion and requires a new card;
- protocol-exposed-three sensitivity is reported at full5050. It cannot rescue
  a failed all-5,050 guard and may only turn an otherwise passing outcome into
  `ambiguous` if exclusion changes a headline pass/fail conclusion.

## Interpretation and next-loop rule

- Positive at holdout: one C candidate passes the exposed-80 repair set and the
  protocol-v2 holdout unchanged. Freeze its exact source/config hashes and run
  broad500; do not update the production `TimingV3Fitter` yet.
- Positive final: the same candidate passes broad500 and full5050 unchanged,
  including denominator and sensitivity checks. Only then may the production
  fitter select it.
- Negative: both C candidates fail a headline gate. Kill the adapter mutation
  and proceed directly to a BeatThis-supported global jump path; do not add
  confidence weights.
- Ambiguous: aggregate passes but jump/long, cumulative-drift, robustness, or
  protocol-exclusion sensitivity guards fail. Audit only the failing strata
  and create a new card before mutation. A broad500/full5050 failure can never
  be overridden by the earlier holdout pass.
- Ramp representation and detection remain blocked until the accepted
  constant/jump path is frozen.
