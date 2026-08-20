# Timing v3 Experiment 001 Result: Evaluation Foundation

Date: 2026-08-11

Decision: accept the evaluation foundation for stable-tempo and jump work;
keep ramp labels audit-only. Proceed to Experiment 002 for the beat-index schema
and a phase-continuous constant/jump projection. Do not tune a ramp detector to
the current weak ramp queue.

## Frozen inputs and outputs

- Canonical inventory: 5,050 audio groups, 14,689 mania maps, and 5,050 valid
  shift-0 float32 BeatThis caches at 50 Hz.
- Inventory JSONL SHA-256:
  `09cc6107b3db60c9ec924b328c027587525112f76ee95d4ed40ee95542aa7acb`.
- Label JSONL SHA-256:
  `cce8287a8270e49913d37ade2f8b1e2ec75d4724573d67dca1be0d8e1be6ddd8`.
- Deterministic 80-audio pilot: 20 stable, 20 jump, 10 ramp-audit,
  10 dense, 10 long, and 10 anomaly rows, with no repeated audio.
- BeatThis cache identity: `beatthis_frame_predictions_v2`, fingerprint
  `483efcdce06c1fd1`, checkpoint `final0`, `shift_ms=0.0`.
- Durable artifacts:
  `artifacts/reports/timing/timing_v3_inventory_v1.*`,
  `timing_v3_labels_v1.*`, `timing_v3_pilot_80_v1.json`,
  `timing_v3_pilot_rows_80_v1.jsonl`, and
  `timing_v3_v2_baseline_pilot80_v1*`.

The fitter saw only the cached BeatThis frame prediction. `.osu` timing,
objects, and local API metadata were read by the evaluation layer only.

## Full-corpus label result

| Audio stratum | Count | Confidence / ambiguity |
| --- | ---: | --- |
| stable | 4,169 | 2,670 high, 1,426 medium, 73 low/ambiguous |
| jump candidate | 232 | 133 high, 94 medium, 5 low/ambiguous |
| dense | 606 | all low and audit-only |
| ramp candidate | 17 | all low, ambiguous, and audit-only |
| ambiguous | 26 | 24 have no valid comparator; 2 remain conflicting |

The parser isolated 72 invalid-redline maps affecting 54 audio groups. These
are comparator limitations, not BeatThis or fitter failures.

Object placement gave a strong but narrower result than initially hoped:

- 14,566 of 14,617 valid maps supported the redline phase grid after requiring
  at least eight start objects over at least four seconds;
- only 3 maps resolved a tempo alias when the rational subdivision and alias
  families were both allowed;
- cross-difficulty scoring supported phase in 9,592/9,627 comparisons but
  resolved the alias in only one.

Therefore object placement is useful evidence for phase and redline
consistency, but is structurally unable to choose musical half/double-time in
most maps. The label schema now exposes `grid_supported` and `alias_resolved`
separately.

Local osu! API scalar BPM agreed alias-aware with the representative redline
for all sources in 4,185 audio groups, partially agreed in 812, disagreed in
27, lacked a valid reference in 24, and was missing in 2. This is a useful
source-stamped cross-check, but it is map-derived and remains correlated with
the redlines.

## Reproduced v2 pilot

The unchanged v2 fitter completed for all 80 audio files. Seventy-seven had at
least one valid redline comparator, producing 190 successful difficulty-level
comparisons. Three anomaly rows had a successful cache load and fit but no
valid `.osu` redline; they are excluded only from oracle-dependent metrics.

Audio-group-weighted headline metrics over the 77 comparable rows were:

| Metric | Mean | p50 | p90 | Max |
| --- | ---: | ---: | ---: | ---: |
| phase error (ms) | 47.596 | 43.308 | 74.369 | 169.088 |
| alias-aware local BPM MAE | 2.238 | 0.423 | 7.123 | 22.500 |
| predicted section count | 3.987 | 2 | 9.4 | 20 |
| alias-normalized endpoint drift (ms) | 5,848.381 | 521.725 | 15,056.167 | 122,874.955 |
| alias-normalized drift slope (ms/min) | 1,172.084 | 114.407 | 1,861.351 | 29,996.250 |
| v2 boundary discontinuity (ms) | 89.285 | 30.661 | 280.267 | 357.000 |

The large cumulative and boundary values demonstrate why mean circular phase
error alone is not an adequate guard. They are now frozen comparator values for
the phase-continuous v3 experiments rather than treated as acceptable targets.

With four local worker processes, per-audio fit time was 2.821 seconds on
average, 6.936 seconds p90, and 16.409 seconds maximum; the wall-clock pilot
completed in about 85 seconds. The runner is fingerprinted, bounded-in-flight,
checkpointed, deterministic in output order, and does not silently reuse a
changed cache or fitter.

The frozen pilot strata also expose where the current module is weak:

| Pilot stratum | Comparable audio | Mean phase ms | Alias BPM MAE | Mean endpoint drift ms | Mean v2 seam ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| stable | 20 | 41.667 | 0.591 | 5,182.228 | 37.808 |
| jump | 20 | 48.129 | 3.478 | 3,748.271 | 110.877 |
| dense | 10 | 56.005 | 1.325 | 3,238.320 | 86.850 |
| ramp audit | 10 | 51.778 | 3.093 | 3,097.824 | 139.229 |
| long | 10 | 53.239 | 4.674 | 20,832.942 | 158.878 |
| anomaly | 7 of 10 | 36.966 | 0.005 | 3.505 | 7.383 |

The anomaly metrics describe only the seven rows with a valid comparator; all
ten loaded and fitted successfully. Long tracks have the largest accumulation
and boundary-seam burden, so decomposition experiments must not be selected
from whole-pilot averages alone.

## Reproduced v2 full corpus

The same frozen cache and fitter were then run over all 5,050 audio groups.
Every cache loaded and every v2 fit completed. The 24 audio-level failures are
exclusively comparator-unavailable rows whose 36 referenced maps all have
invalid red timing; another 36 invalid maps occur beside at least one usable
difficulty. Thus 5,050 rows are inference-valid, 5,026 are `.osu`-comparison
eligible, and 14,617 difficulty comparisons are valid. These denominators are
not interchangeable.

Frozen full-corpus artifacts are:

- `artifacts/reports/timing/timing_v3_v2_baseline_full5050_v1.jsonl`,
  SHA-256 `e31089b0aa5688e6cdad9b11f53efb36a7be7147552d76418ae4133eff1239b3`;
- `artifacts/reports/timing/timing_v3_v2_baseline_full5050_v1_summary.json`,
  SHA-256 `109446fc64cb37e9e8e8ab4cd2a0e51b8d179056822febbd2f41b49a649651b8`;
- regenerated pilot JSONL SHA-256
  `5d9bb3c50f4173b4bde60cdd1dd30a152565bd89934f2e6cb2407dd8910075a7`;
  its summary now correctly records 77 rather than 80 comparison-eligible
  audio groups.

Audio-group-weighted metrics over the 5,026 comparison-eligible rows are:

| Metric | Mean | p50 | p90 | Max |
| --- | ---: | ---: | ---: | ---: |
| phase error (ms) | 46.566 | 43.000 | 74.039 | 235.000 |
| alias-aware local BPM MAE | 1.662 | 0.002 | 3.361 | 129.028 |
| predicted section count | 2.812 | 1 | 7 | 20 |
| alias-normalized endpoint drift (ms) | 2,237.011 | 7.104 | 4,865.174 | 158,845.495 |
| alias-normalized drift slope (ms/min) | 601.507 | 1.696 | 934.446 | 39,905.113 |
| raw v2 maximum boundary seam (ms) | 66.037 | 0 | 216.686 | 1,339.429 |

The full distribution is easier than the deliberately hard pilot: 4,169
audio groups are stable, while the pilot overweights jump, dense, ramp-audit,
long, and anomaly cases. The full run therefore supplies scale and tail-risk
evidence; the stratified pilot remains the repair set. Stable rows have phase
p50/p90 `39.9/72.2 ms` and alias endpoint-drift p50/p90 `0.36/671 ms`, while
dense, jump, and ramp-audit rows have roughly `1.9-2.1 s` median alias endpoint
drift and `295-313 ms` p90 alias-normalized boundary seams.

Four workers completed the full run in 3,031.6 seconds wall time. Per-audio
fit time was 2.373 seconds mean, 4.217 seconds p90, and 28.889 seconds maximum;
no row approached the 180-second timeout. This removes cache and fitter
availability as confounders for the later 100 -> 500 -> 5,050 progression.

## What failed and what changed

The first full label attempt ran for about eight minutes inside the scalar
aliases x subdivisions x events loop without writing a checkpoint. It was
stopped. Active-grid lookup and residual statistics were then vectorized with
NumPy and checked against a scalar reference. A 100-audio benchmark completed
in 6.4 seconds; the complete 5,050-audio calculation then completed in about
355 seconds with a checkpoint every 100 rows. A zero-compute resume of all
5,050 rows completed in 4.4 seconds.

Review also found and fixed stale resume reuse, missing-audio aborts, swallowed
timeouts, dataset-root path escapes, a one-tap object-support false positive,
and full-file rewrites after every baseline row. Details are retained in
`timing_v3_problem_log.md` and regression tests.

## Ramp audit finding

The 17 ramp candidates satisfy only a permissive monotonic-redline heuristic.
Spot checks show that this bucket mixes real ramp-shaped runs with sparse
jumps and alias changes. For example, one unanimous three-point map changes
147 -> 123 -> 61.5 BPM across roughly 133 seconds; another contains a 120 ->
180 BPM jump followed by a short 184 -> 210 BPM run. Neither should be used as
an unquestioned whole-song ramp label.

This is an intended ambiguous outcome: the queue is useful for manual and
activation-overlay audit, but it does not yet support headline ramp
precision/recall. Ramp work must use section-level evidence and synthetic
truth before any parameters are selected.

## Experiment-card decision

- Inventory determinism and 5,050-group coverage: pass.
- Audio-group isolation and zero inference leakage: pass.
- Useful stable and jump strata: pass; 227 jump rows are high or medium
  confidence, above the 30-row kill threshold.
- Object evidence improves phase corroboration: pass.
- Object evidence resolves tempo aliases: negative; role narrowed explicitly.
- Nonempty ramp audit queue: pass (17), but credible manual ramp count remains
  unresolved, so no ramp headline claim is allowed.
- Reproducible v2 pilot with attributable failures: pass.

Overall interpretation: positive for evaluation infrastructure and the
constant/jump path, ambiguous for ramps. The next loop tests only the new
phase-continuous representation and a minimal constant/jump projection.
