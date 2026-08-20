# Timing v3 Experiment 017: Family-stratified candidate retention

Status: stopped / negative

## Authoritative result

Exp017 is stopped at the two-row mechanism gate. It did not run
structure-manifest6, pilot42, holdout100-v2, broad500, full5050, or any
additional protected/full evaluation.

Authoritative artifacts:

- Output:
  `artifacts/reports/timing/timing_v3_exp017_mechanism2_authoritative_v1.jsonl`
  SHA-256
  `4d6b470a586f57f9343789267f35bcc9f8b8db722d192d7c9ea7cbe3d4e29db8`
- Summary:
  `artifacts/reports/timing/timing_v3_exp017_mechanism2_authoritative_summary_v1.json`
  SHA-256
  `9db1e546a3fb40dc589e58f58b88a8093ff26ab837aaf757302885d67fd15752`
- Any earlier partial Exp017 artifact remains nonauthoritative audit evidence
  only and must not overwrite these authoritative v1 artifacts.

Mechanism result:

- Stable row `2300685` was accepted and stayed on the constant path:
  - selected source: `global_constant`
  - selected sections: `200 BPM`
  - weak constant exact: `true`
  - row runtime: `3.542446292 s`
- Jump row `618173` was accepted but still selected the old persistent failure:
  - selected fingerprint: `c97b48d7a793ff8fe55801f42c3a425aa977e5285bd660aaa9c1e8f78705f7ee`
  - selected source: `raw_run_persistent_a_to_b_start`
  - selected sections: `175 BPM @ 276 ms -> 158.426 BPM @ 54104.57 ms`
    through the end
  - weak jump exact: `false`
  - predicted boundaries: `1`
  - weak boundaries: `2`
  - boundary recall: `0.0`
  - row runtime: `2.967137250 s`
- Candidate `20` on row `618173` is the short ABA audit candidate that
  retention was meant to preserve:
  - source: `paired_unmerged_boundary`
  - sections: `175.193 -> 146.621 -> 175.193 BPM`
  - middle span: `56477.345 ms -> 58932.654 ms`
  - raw gain: `+0.008061438`
  - retained in the candidate list but not production-eligible and not selected

Runtime result:

- Mechanism p90 row runtime was `3.484915388 s`.
- This fails the Exp017 runtime gate because it exceeds both the `3.0 s` gate
  and the `+20%` gate versus the Exp014 mechanism p90 of `2.486 s`.
- The hard `< 5.0 s` runtime kill was not triggered.

Decision: `KILL`. Family-stratified retention alone is insufficient. The
plausible paired-boundary short ABA candidate is present, but compatibility /
selection excludes it and still favors the persistent one-boundary candidate.
Do not tune selector thresholds, change compatibility, or advance to
structure/pilot/protected runs under Exp017. Any next selector-arbitration card
must use post-cleanup Exp014/Exp017 evidence, not the invalidated pre-cleanup
Exp016 basis.

## Experiment card

### Objective

Restore the short ABA candidate path that was exposed by the post-cleanup Exp014 mechanism result, without changing the Timing v3 selector, raw evidence, collapsed comparator, metrics, inference inputs, or runtime caps.

This card supersedes Exp016 for this branch. `timing_v3_experiment_016_structure_family_arbitration.md` was planned against invalidated pre-cleanup structure-manifest evidence and was not executed. Exp017 is narrower: it tests whether deterministic candidate retention under the existing caps is enough before considering selector or structure-family arbitration work.

### Source state and evidence boundary

- Baseline source read for this card:
  - `src/pulsefield_model/timing/v3/tempo_track.py` SHA-256 `33c0a7cd64ac75ba4b6213fcbb2875ecd27a229e57053b03f9b6b55019c68cce`
  - `tests/timing/test_timing_v3_exp014_tempo_track.py` SHA-256 `896b634838893df4a44d2fd600a21fd945f436b1cfde4f6bb85631f987784b2d`
  - `docs/research/timing_v3_experiment_014_boundary_complete_jump_candidates.md` SHA-256 `e450a38b147b8dac0f98369e274d9f7826dc029a90e052cb77146ee548c0a67e`
  - `docs/research/timing_v3_experiment_016_structure_family_arbitration.md` SHA-256 `1bff3a9fa0c0be824dca0bedcc39a271a068f33ab28c36d9917a3af601d6be50`
- Current source caps remain:
  - `maximum_base_hypotheses = 12`
  - `maximum_jump_candidates = 44`
  - `maximum_ramp_candidates = 8`
  - `maximum_candidates = 64`
- Allowed real evidence is only the already-exposed Exp014 post-cleanup mechanism result for the two rows named in that result:
  - stable row `2300685`
  - short-ABA row `618173`
- Do not access structure-manifest6, pilot42, holdout100-v2, broad500, full5050, or any additional real row/artifact under this card.
- Weak truth and row labels are evaluation-only after predictions are frozen. They must not enter candidate generation, retention, scoring, selection, feature thresholds, or per-row branching.

### Background facts

Exp014's actual post-cleanup mechanism gate killed the boundary-complete candidate extension before broader pilots:

- Synthetic and related source guards passed.
- Mechanism output:
  - `artifacts/reports/timing/timing_v3_exp014_mechanism2_v1.jsonl`
  - SHA-256 `f8d3ced1e7bfb2d97e932ded4c20bbad045268a1301102753c4cc268d501229b`
- Summary:
  - `artifacts/reports/timing/timing_v3_exp014_mechanism2_v1_summary.json`
- Both rows were accepted with seam max `0.0 ms`.
- p90 row runtime was `2.486 s`; total wall runtime was `4.766 s`.
- Stable row `2300685` selected constant `200 BPM` and matched weak truth exactly.
- Short-ABA row `618173` selected the wrong one-boundary persistent candidate:
  - selected source: `raw_run_persistent_a_to_b_start`
  - selected sections: `175 BPM @ 276 ms -> 158.426 BPM @ 54104.57 ms`
  - exposed weak structure: `175 @ 267 -> 143 @ 55124 -> 175 @ 59296`
  - predicted boundaries: `1`
  - weak boundaries: `2`
  - matched boundaries: `0`
  - boundary recall: `0.0`
  - exact match: `false`
  - direct coverage: `0.8941`
  - weak phase p90: `15.25 ms`
- The paired-boundary evidence gain was positive (`+0.00783`), so the observed failure is not a lack of local paired-boundary evidence.
- The exposed root-cause candidate is retention: internal `maximum_jump_candidates = 44` pruning and family imbalance can remove the near short-ABA candidate before the unchanged selector sees it.

### Hypothesis

If the Exp014 failure is caused by candidate-family retention rather than scoring semantics, then deterministic family-stratified retention under the existing caps will:

1. preserve constant candidates for stable material;
2. preserve at least one source-supported short ABA paired-boundary candidate under jump-candidate pressure;
3. allow the unchanged selector to choose a short ABA candidate for `618173`; and
4. keep `2300685` on the constant path with no stable regression.

### Non-goals

- No selector changes.
- No raw audio evidence changes.
- No collapsed comparator changes.
- No metric or runner changes.
- No fallback-policy changes.
- No ramp production eligibility.
- No cap increase above `maximum_candidates = 64`.
- No increase above `maximum_jump_candidates = 44`.
- No row-specific or weak-label-specific heuristics.
- No threshold tuning to the exposed `143 BPM` or `4.17 s` mechanism.

## Candidate variants

### Variant A: current global jump truncation

Keep the current retention behavior.

Expected result: fail. This is the Exp014 post-cleanup mechanism behavior: raw-run persistent and other family candidates can occupy the retained jump slots before the short paired-boundary ABA candidate is retained.

Decision: reject as known-negative baseline.

### Variant B: raise global caps

Increase `maximum_jump_candidates` or `maximum_candidates`.

Expected result: possibly restores presence, but confounds runtime and violates Phase 1 constraints.

Decision: reject. Existing caps must hold.

### Variant C: family-stratified retention under existing caps

Retain jump candidates by deterministic family quotas, then backfill unused quota from the same global deterministic order. This is the selected mutation.

Decision: execute only Variant C.

### Variant D: structure-family arbitration / selector preference

Change selector arbitration so short ABA can beat persistent candidates.

Expected result: may be required later if candidate presence is restored but selection still fails.

Decision: reject for this card. Exp016 planned this broader direction from invalidated pre-cleanup evidence. Exp017 must first isolate retention.

## Selected mutation

Change only the candidate retention step in `src/pulsefield_model/timing/v3/tempo_track.py`.

The intended implementation point is after jump proposal construction and dedupe inputs are available, but before the final `maximum_jump_candidates` truncation and before `_bounded_proposals` combines constants, jumps, and diagnostic ramps.

The mutation must:

1. classify each jump proposal into exactly one algorithmic family;
2. retain proposals by fixed family quota;
3. backfill unused quota deterministically;
4. preserve constants before jumps/ramps;
5. keep ramp candidates diagnostic-only and capped at `8`;
6. keep total candidates capped at `64`; and
7. keep the global phase-continuous integer beat axis semantics unchanged.

### Frozen family taxonomy

Use algorithmic family derived from candidate structure/source, not weak truth:

1. `short_aba_paired_boundary`
   - exactly three constant sections;
   - two transition boundaries;
   - middle-section duration in `[2.0 s, 8.0 s]`;
   - generated from paired-boundary / virtual-pair source candidates, not from weak labels.
2. `persistent`
   - exactly two constant sections;
   - one transition boundary;
   - persistent A-to-B or B-to-A raw-run family.
3. `long_aba`
   - exactly three constant sections;
   - two transition boundaries;
   - middle-section duration in `(8.0 s, 60.0 s]`;
   - may come from raw-run ABA or paired-boundary ABA source candidates.
4. `multi_step`
   - multiple step changes, capped at four step changes;
   - existing global phase-continuous integer beat axis semantics;
   - raw-run chain / multi-jump source family.
5. `overflow`
   - any valid Phase 1 jump proposal not classified above.

If a proposal matches more than one family, use the first matching family in the order above. This prevents short paired-boundary ABA candidates from being reclassified behind persistent or long-family pressure.

### Frozen quotas

The jump cap remains exactly `44`.

| Family | Reserved slots |
| --- | ---: |
| `short_aba_paired_boundary` | 14 |
| `persistent` | 10 |
| `long_aba` | 8 |
| `multi_step` | 6 |
| `overflow` | 6 |
| Total | 44 |

Rules:

- Constants are always preserved first, up to the existing `maximum_base_hypotheses = 12`.
- Diagnostic ramps remain capped by `maximum_ramp_candidates = 8`.
- Constants + retained jumps + diagnostic ramps must never exceed `maximum_candidates = 64`.
- Unused family quota is not transferred immediately to an adjacent family. After all family first passes complete, fill remaining jump slots from all unretained proposals using the frozen global deterministic order.
- Backfill must not evict any already-retained family-quota candidate.
- Duplicate curve fingerprints are retained only once. If duplicate fingerprints appear across families, the first retained proposal wins.
- The implementation must emit or expose enough diagnostics in tests to count retained proposals by family and detect pruning reason without changing production metrics.

### Deterministic ordering

Within each family and in global backfill, use a deterministic key based only on source-produced proposal attributes:

1. descending existing proposal score;
2. source string;
3. canonical curve fingerprint;
4. boundary times in milliseconds, rounded only by the existing fingerprint/canonicalization mechanism.

Do not use Python object identity, dictionary iteration accident, filesystem order, weak label values, row id, filename, or mutable insertion order as a tie-breaker.

## Test plan

No inference may run until source/unit guards pass.

### Source-only guards

Add or update unit tests only after this card is approved for execution.

Required tests:

1. `test_exp017_constants_are_always_preserved`
   - Construct more than enough non-constant jump/ramp proposals to create cap pressure.
   - Assert every generated constant candidate remains retained.
   - Assert total candidates `<= 64`.
2. `test_exp017_short_aba_retained_under_family_pressure`
   - Synthetic fixture must create more than `44` jump candidates with persistent, long ABA, multi-step, overflow, and at least one source-supported short paired-boundary ABA candidate.
   - Assert retained jump count `<= 44`.
   - Assert retained `short_aba_paired_boundary` count `>= 1`.
   - Assert the short ABA candidate survives even when raw-run persistent candidates score above it.
3. `test_exp017_family_quotas_and_backfill_are_deterministic`
   - Run retention twice on identical synthetic proposal inventory.
   - Assert retained fingerprints and family counts are identical.
   - Assert unused family quota is filled only after all family first-pass reservations.
4. `test_exp017_ramps_remain_diagnostic_only`
   - Assert ramp candidate count `<= 8`.
   - Assert ramps do not become production-selected timing sections.
5. `test_exp017_no_cap_regression`
   - Assert `maximum_candidates <= 64`.
   - Assert `maximum_jump_candidates <= 44`.
   - Assert selected implementation does not require cap reconfiguration.

### Mechanism-only gate

After source-only guards pass, run only the already-exposed two-row mechanism gate from Exp014:

1. Stable mechanism row `2300685`
   - Must remain accepted.
   - Must remain selected as constant timing.
   - Must not regress weak exactness, direct coverage, seam max, or phase error versus the Exp014 post-cleanup mechanism result.
2. Short-ABA mechanism row `618173`
   - Post-freeze audit must show that candidate inventory contains a retained near short-ABA paired-boundary candidate matching the already-exposed mechanism family.
   - The selected candidate must be short ABA, not the previous persistent `raw_run_persistent_a_to_b_start` failure.
   - Boundary recall must improve from `0.0`.
   - Exactness must improve from `false` or be explicitly reported as still false with the retained candidate selected.

This card does not authorize structure-manifest6, pilot42, holdout100-v2, broad500, full5050, or any additional real-row access. If both mechanism rows pass, freeze an Exp017 result and draft the next experiment card before broader evaluation.

## Metrics

### Primary metrics

- Synthetic family-pressure fixture:
  - constants retained: `100%` of generated constants up to existing cap;
  - total candidates: `<= 64`;
  - retained jumps: `<= 44`;
  - retained ramps: `<= 8`;
  - retained short ABA under pressure: `>= 1`;
  - deterministic retained fingerprint set across repeated runs.
- Mechanism gate:
  - `2300685`: accepted constant path with no stable regression.
  - `618173`: retained short ABA present and selected.

### Secondary metrics

- Retained candidate count by family.
- Pruned candidate count by family.
- Backfill count.
- Selected source and selected family.
- Seam max in milliseconds.
- Phase p90 in milliseconds.
- Row runtime p90.

### Runtime budget

The retention mutation is sorting/truncation only. It must not create new audio-evidence calls or increase candidate-generation search breadth.

Kill if mechanism-gate p90 row runtime exceeds either:

- `3.0 s`, or
- `+20%` versus the Exp014 post-cleanup mechanism p90 of `2.486 s`,

whichever is more permissive for the local environment. In all cases p90 must remain `< 5.0 s`.

## Kill criteria

Kill immediately if any of these occur:

1. Any source-only guard fails.
2. Total candidates exceed `64`.
3. Retained jumps exceed `44`.
4. Constants are no longer always preserved.
5. Ramps become production eligible.
6. Retention depends on weak truth, row id, filename, or the exposed `143 BPM` / `4.17 s` values.
7. Mechanism gate accesses any row other than `2300685` and `618173`.
8. `618173` does not retain the already-exposed near short-ABA paired-boundary candidate in post-freeze audit.
9. `618173` retains short ABA but still selects the previous persistent failure source.
10. `2300685` regresses from stable constant behavior.
11. Runtime p90 reaches `>= 5.0 s`.
12. Any change is required outside candidate retention, tests, and experiment result documentation.

If kill criterion 8 is hit, the retention implementation is wrong or the family taxonomy is incomplete. Do not tune selector thresholds.

If kill criterion 9 is hit while candidate presence is confirmed, retention is not sufficient. Freeze a negative Exp017 result and consider a new selector-arbitration card using only the exposed mechanism evidence.

## Expected interpretation

Positive result:

- Exp014's short-ABA mechanism failure was caused by candidate retention/cap family imbalance.
- Proceed to a separate card for broader sealed evaluation only after documenting Exp017.

Negative result with no retained short ABA:

- The family classifier or quota placement failed to preserve the target family under cap pressure.
- Fixing selector behavior would be premature.

Negative result with retained short ABA but persistent still selected:

- Candidate retention is not enough.
- A future selector-arbitration experiment may be justified, but it must be planned from post-cleanup Exp014/Exp017 evidence, not the invalidated pre-cleanup Exp016 basis.

## Pre-mortem

Likely implementation traps:

- Classifying raw-run ABA before short paired-boundary ABA and recreating the same crowd-out problem.
- Applying quotas before fingerprint dedupe, causing duplicates to consume reserved slots.
- Letting backfill evict quota-retained short ABA candidates.
- Accidentally increasing cap values to make tests pass.
- Using weak labels or exposed row facts to define candidate thresholds.
- Preserving family counts while changing selector input order in a way that changes constant stability.
- Counting diagnostic ramps in a way that lets them become production timing choices.
- Making retention nondeterministic through object identity or dictionary iteration.

Required discipline:

- Freeze predictions before weak-truth comparison.
- Keep protected data sealed.
- Report candidate inventory by algorithmic family.
- Treat this as a retention-only isolation test.
