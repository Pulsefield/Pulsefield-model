# Timing v3 Experiment 022: Frozen Exp021 pilot42 replay

Status: completed / invalid for advancement (runtime); diagnostic negative; stopped before tuning or broader rows

## Experiment card

### Objective

Measure whether the frozen Exp021 Timing-v3 source generalizes beyond its
two-row mechanism gate on the already-exposed high/medium-confidence,
non-ambiguous stable/jump pilot42. This is a no-mutation replay: do not change
source, config, candidates, scores, thresholds, selector, caps, evaluator, or
fallback.

### Frozen source and evidence boundary

- `src/pulsefield_model/timing/v3/tempo_track.py` SHA-256
  `fc4153a6310a4db233e1fbd29e87a57775eff924e4904e925773d172e0d7de85`.
- `tests/timing/test_timing_v3_exp021_tempo_track.py` SHA-256
  `d543ec827a893fb1dddc3513edbe6df2396138ace5d96a58c679002dfeee3ae5`.
- Inference/evaluation support sources:
  - `src/pulsefield_model/timing/evaluation/exp013_pilot.py` SHA-256
    `70891775233cf0c66b0d948689cf8a7d3505c57192ef0d5beed81f3f22f1b3bb`;
  - `src/pulsefield_model/timing/evaluation/curve_metrics.py` SHA-256
    `1a70a9c0e8f965b9c7a9de74bc1c99711c95b938abda900e871f4fce0f316c2a`;
  - `src/pulsefield_model/timing/evaluation/exp004_metrics.py` SHA-256
    `f88366562d35645fa4264cdb8710d2ea4781c6615a2ea23c2c3c87b0524dc21e`;
  - `src/pulsefield_model/timing/v3/analytic_curve.py` SHA-256
    `766762aafdf0a3e643b3689c7ec659d14a0dd002befd2607a410fc368962c4d3`.
- Exp021 source-only related guard: `99 passed`.
- Authoritative Exp021 mechanism artifacts:
  - output JSONL SHA-256
    `18944665c5d91e4435abf1eddc65bc102c6b4748448eb854560bd0f7aee04178`;
  - summary SHA-256
    `69d1ca49510ffb89af74f798ada82fe63c5c459d49d3840c4ebd2c5cee476f40`;
  - inference freeze SHA-256
    `67fb12b32f4dc6372bef91efc6b3a4a353707d8988966664255739359c5605d3`;
  - post-freeze audit SHA-256
    `0d99998e9fcacc454089152b750ae937036f98ba9cb660728e5c30c0f96d605d`.
- Input is only
  `artifacts/reports/timing/timing_v3_pilot_rows_80_v1.jsonl`, SHA-256
  `cdb5e2af87d99c8af3bbff71a0985bb490fdeed169d10cb0a64b17a8ca0296d7`.
- Paired v2 baseline is only
  `artifacts/reports/timing/timing_v3_v2_baseline_pilot80_v1.jsonl`, SHA-256
  `5d9bb3c50f4173b4bde60cdd1dd30a152565bd89934f2e6cb2407dd8910075a7`.
- Deterministically select every row with stratum in
  `{stable, jump_candidate}`, confidence in `{high, medium}`, and
  `ambiguous=false`. Expected identity is exactly `42` unique audio rows:
  `22` stable and `20` jump.
- These rows and pilot80 were already exposed by Exp013. Do not open
  holdout100-v2, structure-manifest6, broad500, full5050, ramp/dense rows, or
  any other real identity.

### Frozen comparator

The negative Exp013 pilot42 baseline remains:

- output SHA-256
  `574a2affc87ac555636a8ebd261ea5bd88ea2346bdb7f29dafd59e93c60f22b0`;
- summary SHA-256
  `a27b68e4b06c546bcedb40babb29cdd3ba70f798243be04899867adb856dd2fc`;
- `17/42` accepted and `25/42` fallback;
- stable accepted/fallback `13/22` and `9/22`, with one false jump;
- jump accepted/fallback `4/20` and `16/20`, only two selected jumps, no exact
  jump hit;
- p90 row runtime `3.58 s`, hard failures `0`, seam maximum `0.0 ms`.

Exp013 is a descriptive frozen comparator only. No row-level Exp013 outcome may
enter Exp022 inference or branching.

### Hypothesis

Exp014 added persistent, long-ABA, and bounded multi-step proposals, while
Exp017-Exp021 repaired cap allocation, closed-ABA compatibility, and the
short-family scalarization failure. With the whole current source frozen, more
exposed jump rows should now reach a defensible v3 decision without losing the
strong stable behavior or runtime budget.

### Non-goals

- No tuning, retraining, code/config change, threshold sweep, or ablation.
- No nice-number prior; Exp015 remains separate and stable-only.
- No ramp claim or ramp production eligibility.
- No protected/broader/full-corpus access.
- No promotion into a default production path from this development slice.

## Execution protocol

Use one temporary, source-audited harness:
`/private/tmp/timing_v3_exp022_pilot42.py`. Before any real execution it must
provide `--self-check-only` coverage with synthetic rows for manifest identity,
forbidden pre-freeze oracle access, per-row observer durability, overwrite
rejection, cache snapshot mismatch, fixed-boundary matching, and exact
aggregate denominators.

The authoritative paths are fixed before execution:

- identity manifest:
  `artifacts/reports/timing/timing_v3_exp022_pilot42_identity_v1.json`;
- result:
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_v1.jsonl`;
- runner summary:
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_summary_v1.json`;
- durable pre-oracle freeze:
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_freeze_v1.jsonl`;
- fixed-1-second audit and advancement aggregate:
  `artifacts/reports/timing/timing_v3_exp022_pilot42_authoritative_audit_v1.json`.

The harness must:

1. Verify every frozen source/input/artifact SHA and refuse if any authoritative
   path above already exists.
2. Parse the already-exposed pilot80 identity file without loading audio/cache,
   resolve exactly 42 qualifying rows, require `22/20` strata and unique cache
   keys, write their sorted cache-key SHA manifest, and create a physical
   42-line execution snapshot. Parsing the other pilot80 routing rows is
   allowed; loading their cache/audio or inferring on them is not.
3. Bind a frozen Exp022 `_PilotRunMetadata` and call `run_exp013_pilot`
   programmatically on only that 42-line snapshot with the paired v2 baseline.
   Do not use the Exp013 CLI defaults.
4. Install a `frozen_payload_observer`. On every callback, validate the Exp021
   tempo-track version/schema and atomically persist the growing freeze JSONL
   before returning control to the runner's weak-oracle stage. Each freeze row
   contains cache-key SHA, product status, selected fingerprint/curve digest,
   candidate inventory digest/counts, seam report, and frozen-inference SHA.
5. Wrap the weak-oracle loader and baseline access audit so any call before the
   matching durable observer record is a hard failure. The inference payload
   exposed to the generator remains identity, duration, audio path, BeatThis
   cache, and raw-audio evidence only.
6. Resolve each of the 42 expected BeatThis paths and mel-cache paths before
   inference. Snapshot existence, size, mtime, and SHA for existing files;
   record missing mel paths as missing. Repeat after execution and abort if an
   existing file changes or a missing path appears. Record raw audio as read
   only. No network or cache generation is authorized.
7. After the complete result/summary bytes are written and hashed, compute the
   separate fixed `+/-1000 ms` audit. Reuse significant boundary extraction;
   form all pairs within tolerance, sort by `(absolute_error_ms,
   predicted_time_ms, weak_time_ms, predicted_index, weak_index)`, and greedily
   accept unmatched indices. The audit must read only already-frozen output and
   the same 42 source rows.
8. Stop after exactly 42 rows and the frozen aggregate. Do not repair failures
   in the same run. Preserve partial evidence on failure; never overwrite or
   delete it under this card.

## Required reporting

Report overall and separately for stable/jump:

- v3 accepted, v2 fallback, hard failure, and every fallback reason;
- selected constant/jump topology and candidate source/family;
- weak class exactness and constant/jump exact-hit fields;
- direct and alias BPM coverage, local BPM error, phase mean/p90/max, endpoint
  and maximum-prefix absolute drift;
- strict boundary precision/recall and fixed-1-second precision/recall;
- for matched jump boundaries, direct left/right tempo-pair correctness;
- stable false-jump count and false-boundary/song rate;
- candidate count/cap reason, maximum seam, row p50/p90/max, total runtime;
- paired changes versus frozen Exp013 and available v2, with denominators.

Ramp accuracy remains `null` everywhere.

## Decision gates

Execution integrity requires:

- exactly `42` rows (`22/20`) and unique identities;
- hard failures `0`;
- maximum seam `0.0 ms` within serialization tolerance;
- p90 row runtime `<5.0 s` and max runtime reported;
- no cache mutation, pre-freeze oracle call, or source/input SHA mismatch.

Advance to a separately carded broader exposed slice only if all of these hold:

- total v3 acceptance is greater than Exp013's `17/42`;
- stable v3 acceptance is at least `13/22`, accepted-stable class accuracy is
  at least `95%`, and stable false jumps are at most one; the denominator is
  every accepted stable-stratum row, every such row must have post-freeze
  metrics, and metric-unavailable rows count as inaccurate and fail this gate;
- jump v3 acceptance is at least `8/20` and at least six rows select a jump;
- at least four jump rows have nonzero fixed-1-second boundary recall;
- among accepted rows with weak class `jump` and selected class `jump`, the
  denominator is at least six, every row has post-freeze metrics, and mean
  fixed-1-second boundary recall (unmatched rows contribute zero) is at least
  `0.25`;
- across all accepted weak-jump rows, every row has post-freeze metrics, mean
  direct BPM coverage is greater than Exp013's `0.2470`, and mean phase p90 is
  less than Exp013's `118.69 ms`; report the exact denominator and do not drop
  low-quality rows.

These are development advancement gates, not the final goal claim. The final
targets remain stable `99%`, jump `80%`, ramp `70%`, offset/phase `70 ms`, and
the product runtime contract.

If any execution-integrity condition fails, classify the run invalid. If the
run is valid but any advancement condition fails, freeze it as negative and
isolate the dominant mechanism in a new card. Never tune on pilot42 under
Exp022.

## Kill criteria

Kill before or during execution if source/config/evaluator identity changes;
the resolved slice is not exactly `42/22/20`; any unapproved row is opened;
weak truth reaches inference; cache state changes; any threshold, score, cap,
or selector is changed; or a result path already exists.

## Expected interpretation

- Pass: Exp021 generalizes enough to justify a new, still-exposed broader
  evaluation card; it does not authorize holdout/full5050.
- Valid negative: freeze the aggregate and use source-only/post-freeze evidence
  to isolate one next mechanism.
- Invalid: discard no bytes, document the integrity failure, and do not infer
  algorithm quality.

## Authoritative result log

### Execution identity

- Mode: executor; the accepted card and all decision gates above were frozen
  before execution.
- Date: 2026-08-14.
- Temporary harness SHA-256:
  `a718c841593ca23f168fd729a27731330f0786983f5e332de353df5114e1330a`.
- The six frozen source/evaluator SHAs, pilot80 input SHA, v2 baseline SHA,
  Exp013 comparator SHAs, and four Exp021 mechanism artifact SHAs all matched
  the values declared in this card.
- The execution snapshot SHA-256 was
  `33f4a5808fc6cfe588955ee216d8a974d6c0e9c7e29e0f6e2d1d72da7ccb586b`.
  Its result, freeze, audit, and identity orders agree exactly and contain 42
  unique identities: 22 stable and 20 jump-candidate rows.
- No algorithm, config, evaluator, threshold, candidate rule, selector, or cap
  changed. No inference was rerun while freezing this result into docs.

The five authoritative artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| `timing_v3_exp022_pilot42_identity_v1.json` | `6b315460900d7a569c0e3523b0de0b4f1c902b398c93f1f7bf10763a06d1c4f6` |
| `timing_v3_exp022_pilot42_authoritative_v1.jsonl` | `df3cb61bc8aec41284f27cf2c75d0281046d043216db83196617fbd06989e173` |
| `timing_v3_exp022_pilot42_authoritative_summary_v1.json` | `00e956ab4fdf66e5cca75cf359a910c40132675957b3906e6174fbf213afb057` |
| `timing_v3_exp022_pilot42_authoritative_freeze_v1.jsonl` | `bbefbc76aad2469ab4af02c7ab4fe6fda805d370dc9d9cc48aea05eef9188988` |
| `timing_v3_exp022_pilot42_authoritative_audit_v1.json` | `2b2e8cada2768827e1d78254ebfabb3cde9689b0ca95662e40d4fcfce8424d77` |

### Outcome

All 42 rows returned `v3_accepted`; there were no v2 fallbacks, fallback
reasons, or hard failures. This increases acceptance by 25 rows over Exp013
(`42/42` versus `17/42`), but the failed runtime integrity gate makes the run
invalid for advancement and prevents a formal generalization classification.
As diagnostic evidence, stable acceptance rose from 13 to 22 while false jumps
rose from one to six. Jump acceptance rose from 4 to 20 and selected jumps
from 2 to 17, but exact jump hits remained `0/20`.

| Frozen metric | Stable, n=22 | Jump candidate, n=20 |
| --- | ---: | ---: |
| Accepted / fallback / hard failure | 22 / 0 / 0 | 20 / 0 / 0 |
| Selected constant / jump | 16 / 6 | 3 / 17 |
| Weak-class exact | 16/22 (72.73%) | 17/20 (85.00%) |
| Constant / jump exact hit | 12/22 / n/a | n/a / 0/20 |
| Mean direct / alias BPM coverage | 0.722884 / 0.903079 | 0.495591 / 0.722803 |
| Mean local alias-BPM MAE | 5.0196 BPM | 9.3780 BPM |
| Mean phase mean / p90 / max | 44.017 / 72.063 / 87.073 ms | 72.807 / 140.318 / 202.823 ms |
| Mean endpoint absolute / maximum-prefix absolute drift | 37,078.490 / 37,092.581 ms | 40,941.527 / 41,281.497 ms |

The stable lane produced six false-jump songs and 11 false boundaries: a
false-jump song rate of `6/22 = 27.27%` and `11/22 = 0.5` false boundaries per
stable song. Five came from `raw_run_aba` and one from
`raw_run_persistent_a_to_b_start`.

Among the 17 jump-candidate rows that selected a jump, strict boundary
precision/recall averaged `0.05882/0.05882` with one matched boundary. The
fixed `+/-1000 ms` audit averaged `0.17647/0.12941`, matched four boundaries,
and found only three rows with nonzero recall; two matched boundaries also had
the correct direct left/right tempo pair. Selected jump sources were
`raw_run_aba` 10, `raw_run_persistent_a_to_b_start` 5,
`paired_unmerged_boundary` 1, and `virtual_right_raw_audio` 1. The three
constant selections used `global_constant`.

Every row reported candidate-cap reason `maximum_jump_candidates_44`.
Candidate counts were exactly 56 on stable rows and ranged from 56 to 59
(mean 56.9) on jump rows. Maximum serialization seam was `0.0 ms`. Row runtime
was p50 `6.9256 s`, p90 `11.7245 s`, and maximum `15.8542 s`; total runtime was
`296.5692 s`.

### Paired comparators

Against the frozen Exp013 aggregate, jump mean direct BPM coverage improved by
`+0.24859` (`0.49559` versus `0.2470`), while jump mean phase p90 worsened by
`+21.6282 ms` (`140.3182` versus `118.69 ms`). Runtime p90 worsened by
`+8.1445 s` (`11.7245` versus `3.58 s`).

Paired v2 curve metrics were available for only 9 stable and 4 jump rows, so
these deltas are descriptive rather than slice-wide. Current minus v2 was:

| Stratum | Direct / alias BPM coverage | Phase mean / p90 | Signed endpoint / maximum-prefix absolute drift |
| --- | ---: | ---: | ---: |
| Stable, n=9 | -0.116946 / -0.085533 | +13.375 / +26.697 ms | -56,838.945 / +53,930.403 ms |
| Jump, n=4 | -0.192097 / -0.282192 | +23.190 / +37.074 ms | +19,666.500 / -23,465.617 ms |

### Integrity and decision gates

The resource guard captured 42 audio, 42 BeatThis, and 42 mel paths. All 126
before/after snapshots were byte-for-byte identical; 115 resources existed
and the 11 missing mel paths remained missing. The durable freeze contains 42
unique rows. There were 42 guarded weak-oracle events, one baseline-file read,
and 84 baseline row lookups; all row lookups had a matching durable freeze and
`oracle_guard_failures` was empty. Ramp accuracy remained `null`.

The identity field `sorted_cache_key_sha256_manifest` has a naming/contract
ambiguity. The harness first sorted rows by the original cache-key string and
then projected each key to its SHA-256; the stored SHA strings are therefore
in cache-key order, not SHA-string lexicographic order. Identity uniqueness and
all result/freeze/audit orders still agree, so this is non-destructive artifact
contract debt rather than evidence of row substitution. Do not rerun, reorder,
rewrite, or overwrite the Exp022 artifacts to repair the field name. Any later
card or harness must state the ordering semantics explicitly.

The execution-integrity aggregate failed only
`runtime_p90_under_5s`; exact identity/strata, zero hard failures, zero seam,
cache immutability, durable freeze, and oracle guards passed. The advancement
aggregate failed exactly:

- `stable_class_accuracy_at_least_0_95` (`0.72727` observed);
- `stable_false_jumps_at_most_1` (6 observed);
- `fixed_nonzero_recall_at_least_4` (3 observed);
- `mean_fixed_recall_at_least_0_25` (`0.12941` observed over 17 selected
  weak-jump rows);
- `jump_phase_p90_below_exp013_118_69ms` (`140.3182 ms` observed).

Acceptance, metrics completeness, jump-selection count, and jump direct-BPM
coverage passed their advancement thresholds, but those passes do not rescue
the failed specificity, boundary, phase, and runtime gates.

### Closed-loop interpretation

Exp021 remains a positive two-row mechanism result: the first-Pareto-front
reservation recovers the known short-ABA compromise. Exp022 shows that the
same frozen source cannot advance from pilot42 because row-runtime p90 exceeds
the preregistered integrity limit. The frozen statistics are diagnostically
negative: the run converts every prior fallback into an accepted result, but
also over-selects change on stable songs, poorly localizes most jump
boundaries, and worsens phase versus Exp013.

Classification: completed but invalid for advancement because of runtime;
diagnostic evidence only, not a formal generalization result. Route:
`MUTATE`. Stop this card before all tuning, ablation, broader exposed rows,
holdout100-v2, broad500, or full5050. The next separately accepted card may
isolate one dominant mechanism from the already-frozen diagnostic rows, but it
must not treat Exp022 as valid generalization evidence.
