# Timing v3 Experiment 020: Pre-cap short-ABA inventory audit

Status: completed / diagnostic root cause classified

## Experiment card

### Objective

Locate the current `618173` failure before another algorithm mutation. Capture
the complete source-produced short paired/virtual ABA proposal inventory at the
input to family retention, freeze and hash that inventory before weak truth is
loaded, then perform post-freeze diagnostic comparison. Determine whether the
goal-compatible candidate is absent at proposal generation, present but pruned,
or retained but not selected.

### Evidence boundary

- Source baseline is Exp019 v1.
- Authoritative Exp019 two-row output:
  `artifacts/reports/timing/timing_v3_exp019_mechanism2_authoritative_v1.jsonl`,
  SHA-256
  `c23fcd39d1cba1e95b2b69e66abb88eff39848dc6c6dc0e1eedb8db094bcf86c`.
- This diagnostic may read only the already-exposed jump row `618173`, its
  existing BeatThis cache, its existing/raw mel path, the exact authoritative
  Exp019 row after the inventory bytes are frozen, and its already-exposed
  representative weak oracle after that freeze.
- Do not read stable controls, structure-manifest6, pilot42, holdout100-v2,
  broad500, full5050, or any additional row/artifact.
- No product source, tests, config, metrics, cache, or model state may change.

### Background

Exp019 changed only short-family retention to pure ABA support. It selected a
closed `175.193 -> 170.152 -> 175.193 BPM` virtual-right candidate. This is a
near-base minimal excursion, not the exposed short slowdown. Both the strict
matcher and fixed `+/-1000 ms` goal audit matched `0/2` boundaries. Direct BPM
coverage fell to `0.9133`.

The retained Exp019 artifact contains no near-143 BPM / approximately 4-second
candidate. It is therefore unsafe to keep tuning selection or retained-family
order without observing the pre-cap proposal pool.

### Hypotheses

1. Proposal-present/pruned: a goal-compatible closed ABA exists before
   retention but ranks below the fourteen short slots.
2. Pair/proposal-absent: no goal-compatible candidate reaches retention because
   pair-seed retention or proposal construction omits its boundary combination.
3. Retained/not-selected: a compatible proposal is retained but loses the
   unchanged selector.

This is a diagnostic audit, not an algorithm comparison.

## Frozen procedure

Use a temporary diagnostic runner under `/private/tmp`; do not modify package
source.

1. Before calling the runner, create a new physical one-line sanitized JSONL
   from already-exposed identity constants only. It contains:
   - `resolved_audio_path`;
   - `source.cache_audio_key` and `source.cache_duration_seconds`;
   - the minimum `label.stratum`, `label.confidence`, and `label.ambiguous`
     routing fields required by the existing runner;
   - no `representative_redline_grid`, maps, redlines, beatmap path, BPM truth,
     boundary truth, title, artist, or other weak payload.
2. Preflight raw line count `==1`, exact cache key/path/duration, and absence of
   forbidden keys before calling the runner. Record the sanitized input SHA-256
   and physical line count in the audit summary.
3. Call the existing exposed pilot runner on only that sanitized one-row JSONL,
   with a supplied `candidate_generator` wrapper. Do not pass pilot80 or any
   multi-row file to the runner.
4. In the wrapper, temporarily replace `_retain_jump_proposals_by_family` with
   a capturing delegate that:
   - receives the exact pre-retention proposal tuple;
   - constructs a deterministic inventory only for proposals classified as
     `short_aba_paired_boundary`;
   - then calls the original retention function unchanged.
5. Inventory fields are source-only:
   - canonical fingerprint;
   - source;
   - blended generation score;
   - optional pure ABA support delta;
   - outer and middle BPM;
   - middle start/end/duration;
   - beat indices and canonical boundary times;
   - duplicate count and deterministic blended/support ranks.
6. Sort inventory records by canonical fingerprint for serialization. Use
   canonical JSON with sorted keys and compact separators.
7. Compute inventory bytes and SHA-256 inside the candidate-generator wrapper,
   before it returns and before the runner can access representative redlines or
   `.osu` truth. Hold the frozen bytes in memory.
8. Let the unchanged runner finish and freeze the selected Exp019 inference.
   Because the sanitized row has no representative payload, its built-in weak
   evaluation must be unavailable and cannot load an `.osu` oracle.
9. Atomically write the already-frozen inventory bytes and its SHA-bearing
   summary artifact.
10. Only after those writes, read the exact authoritative Exp019 `618173` row,
    compare the selected fingerprint, obtain its already-exposed representative
    beatmap path, load the weak grid, and write a separate post-freeze audit.
    Do not mutate the frozen inventory records.

### Post-freeze diagnostic audit

For every unique short proposal, report without changing inference:

- outer/middle direct BPM error;
- greedy fixed `+/-1000 ms` two-boundary precision/recall using the Exp019
  frozen rule;
- whether it meets direct BPM tolerance `max(1 BPM, 1%)` on all three sections;
- whether it is present in the final retained candidate list;
- whether it is production-eligible/selected;
- its blended-score rank and pure-support rank within the short family.

Weak fields must live only in the post-freeze audit payload, not the frozen
source inventory.

## Outputs

- `artifacts/reports/timing/timing_v3_exp020_618173_precap_short_inventory_v1.json`
- `artifacts/reports/timing/timing_v3_exp020_618173_precap_short_inventory_summary_v1.json`
- `artifacts/reports/timing/timing_v3_exp020_618173_precap_short_audit_v1.json`
- a narrow one-row runner output/summary with `exp020` in the filenames.

All output paths must be new and noncolliding. Preserve any pre-existing file.

## Guards

- Exact input row count: `1`, path ends with `/618173/audio.mp3`.
- Source version remains Exp019 v1 before and after the run.
- Sanitized runner input has exactly one physical line and no representative,
  redline, map, beatmap, BPM, or boundary truth fields.
- Capturing delegate calls the original retention implementation exactly once
  with the original proposal tuple/config and returns its result unchanged.
- Product selected fingerprint must equal the selected fingerprint in the
  existing authoritative Exp019 `618173` row; no second ordinary rerun is
  required.
- Frozen inventory contains no weak labels, weak BPMs, `.osu` path, row stratum,
  title, artist, map metadata, or evaluation metrics.
- Frozen inventory SHA is computed before weak loading.
- No source/test/config edit and no additional real row access.
- Runtime is reported but is not a product benchmark because diagnostic
  serialization is extra work.
- Before real execution, the temporary script must run source-only self-checks
  for one-row preflight rejection, forbidden-key rejection, capture freeze
  ordering, `try/finally` hook restoration, original delegate call count, and
  unchanged returned batch identity.

## Interpretation / stop rule

- If a goal-compatible proposal exists pre-cap but not retained, the next card
  may isolate proposal retention using its source-only ranks.
- If it does not exist pre-cap, stop retention/selector tuning; the next card
  must isolate pair-seed or proposal construction. Exp020 observes only
  proposals that reach the retention hook; pair-seed absence is inferred, not
  directly captured.
- If it is retained but not selected, the next card may isolate arbitration.
- Do not implement any fix in Exp020.
- Stop after the one-row audit and document the classified root cause.

## Kill criteria

Kill if the wrapper changes returned candidates, source version, selected
fingerprint, cache state, proposal scores, or order; if inventory bytes are not
frozen before weak loading; if any extra row is accessed; or if any product
source/test/config file is edited.

## Authoritative result log

Exp020 completed as the frozen one-row diagnostic. It accessed only the
already-exposed `618173` row and did not modify product source, tests, config,
metrics, cache, or model state. The source version remained
`pulsefield_model.timing_v3_tempo_track_exp019_v1`; the capturing delegate
called the original retention implementation exactly once and returned the
unchanged batch. The runner selected the same fingerprint as authoritative
Exp019, and its built-in weak metrics remained unavailable until the separate
post-freeze audit.

Core artifacts:

- frozen pre-cap inventory
  `artifacts/reports/timing/timing_v3_exp020_618173_precap_short_inventory_v1.json`,
  SHA-256
  `c78cd263389d496c11bff5baf93963fe0dc54e360953001c048f8543960d168f`;
- post-freeze audit
  `artifacts/reports/timing/timing_v3_exp020_618173_precap_short_audit_v1.json`,
  SHA-256
  `5df0afae4c8fb76e94bf52df3b5ad96f0309890d5660867d5a64cb70fa6b6b62`;
- one-row runner output
  `artifacts/reports/timing/timing_v3_exp020_618173_runner_v1.jsonl`,
  SHA-256
  `a18b9f7d0e8e520be959bb98f93998eca30378a461a1980cafe0d6653e5d2c01`.

The companion inventory summary has SHA-256
`c425be7d898bf38409dcb11550f0f146ffa020a38e0908d490de0a7b94fc0192`;
the runner summary has SHA-256
`ca378f22ac0a8c7bc1c18f789b1b6a0135936dcd8588541a8c5098c6fdba6231`.
The sanitized one-line input had SHA-256
`8427f1c8b7858046deb72fd52d32eadb84fdfac0a4f4ca7b3702051ad0e3c56b`,
and the authoritative Exp019 artifact guard matched its frozen SHA-256
`c23fcd39d1cba1e95b2b69e66abb88eff39848dc6c6dc0e1eedb8db094bcf86c`.

The capture observed `27,946` proposals at the input to family retention and
`1,958` unique short-ABA fingerprints. Post-freeze evaluation classified the
root cause as `goal_compatible_present_but_pruned`: seven goal-compatible
proposals existed before retention, zero survived into the retained list, and
zero were selected (`7/0/0`). The unchanged retention returned `44` total jump
candidates.

The strongest goal-compatible source proposal for the intended mechanism had
fingerprint
`e7e86f7c6828b3089c45b0dbef4e03ea70db5a9b8db3d4a70c7f8d924c58bfe9`
from `paired_unmerged_boundary`. It described a closed
`175.193 -> 143.964 -> 175.193 BPM` ABA with middle interval
`54764.951 -> 58932.654 ms` (about `4.168 s`). Its source-only blended rank was
`22`, pure-support rank was `50`, and ABA support delta was `-0.061745`. In the
post-freeze audit it passed all three direct-BPM tolerances and matched both
already-exposed weak boundaries within the fixed `+/-1000 ms` rule, giving
precision `1.0` and recall `1.0`, but it was neither retained nor eligible.

The one-row diagnostic runtime was `3.4299525 s` (`3.4309005 s` total). This is
reported only as diagnostic cost, as preregistered, not as a product benchmark.

Decision: Exp020 succeeds as a diagnostic and stops. Proposal construction is
not the missing mechanism on this row; the compatible proposal exists before
the cap. The failure is localized to short-family retention, where both the
current blended and pure-support orders place the goal-compatible candidate
outside the fourteen reserved short slots. Any next mutation must be a
separately accepted, source-only retention proposal. Do not change arbitration,
compatibility, scoring, search breadth, caps, or access broader rows under
Exp020.
