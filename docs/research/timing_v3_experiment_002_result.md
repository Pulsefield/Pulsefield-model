# Timing v3 Experiment 002 Result: Phase-Continuous Constant/Jump Projection

Date: 2026-08-11

Decision: accept the Timing v3 constant-section representation and reject
Family B as the selected production projection. Family B makes every seam
exact and passes the fallback, section-count, serialization, and aggregate p90
guards, but its paired mean phase ratio is `1.11043`, above the frozen `1.10`
limit. Follow the preregistered Family C mutation; do not relax the 5% BPM
adjustment guard or tune on the exposed 80-audio pilot.

## Delivered representation and inference boundary

Experiment 002 adds an immutable global integer beat axis:

```text
beat [a,b) bpm q
```

Each grid stores one beat/time origin and contiguous half-open sections. The
origin may lie inside the first section, which preserves the first v2 anchor as
beat zero while allowing negative beats to extend backward over cache time
zero. Every section boundary time is derived by one prefix scan; adjacent
independent time anchors are not serialized as authorities.

The schema validates finite 20-1000 BPM sections, strict beat and time order,
cache coverage, lookups, conversion to the unchanged v2 grid, and JSON
round-trip seams. `TimingV3Fitter` consumes one `FrameTimingPrediction`, calls
the unchanged v2 fitter exactly once, and reads no `.osu`, object, title,
metadata, or network evidence.

Two whole-grid projections were frozen:

- Family A keeps source BPMs and moves later anchors; it is the control.
- Family B keeps every source anchor and changes each preceding BPM so the
  anchor interval contains the half-up-rounded integer beat count. A section
  exceeding 5% relative BPM adjustment fails closed to a tagged v2 fallback.

Both keep the final source BPM and extend the final beat boundary only far
enough to cover the exclusive cache end.

## Frozen inputs and durable outputs

- Baseline input: `timing_v3_v2_baseline_pilot80_v1.jsonl`, 80 audio groups,
  SHA-256 `5d9bb3c50f4173b4bde60cdd1dd30a152565bd89934f2e6cb2407dd8910075a7`.
- Projection config: 20-1000 BPM, 5% maximum Family B adjustment, cache
  coverage beginning at 0 ms, canonical BPM band 80-160.
- Per-audio output:
  `artifacts/reports/timing/timing_v3_projection_pilot80_experiment002_v1.jsonl`,
  SHA-256 `8fde0a09a7ff50023aed4510f310814e737fe408eb650defdf656bf29de1031f`.
- Aggregate output:
  `artifacts/reports/timing/timing_v3_projection_pilot80_experiment002_v1_summary.json`,
  SHA-256 `db555b86658e5128942032bceb39d7064538af2146075ae9b9135e304bbecbea`.
- Summary schema: `pulsefield_model.timing_v3_projection_evaluation_summary_v2`;
  behavior fingerprint:
  `fe7ed36b8e2550f5dec2449677f6af9b3ca34056d0166f3a090bf4f1db411c9a`.

The evaluator reconstructs only stored v2 and oracle segments from the durable
baseline. It has no cache loader, model, fitter, or `.osu` parser path.

## Pilot result

All 80 stored fits were projection-evaluable. Seventy-seven audio groups had at
least one usable stored `.osu` comparator, producing the same 190 paired
difficulty comparisons as the v2 baseline. The other three remain comparator
unavailable but still count in projection and fallback rates.

| Candidate | Projection success | Fallback | Mean phase ms | Mean ratio vs v2 | p90 phase ms | p90 ratio vs v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 baseline | 80/80 fits | n/a | 47.596 | 1.00000 | 74.369 | 1.00000 |
| Family A control | 80/80 | 0 | 56.522 | 1.18753 | 97.638 | 1.31288 |
| Family B | 78/80 | 2 | 52.852 | 1.11043 | 84.182 | 1.13195 |

The two Family B failures are anomaly rows without a valid redline comparator.
One requires a `-7.63%` BPM correction in a low-BPM interval; the other would
require `+53.61%`. They correctly remain tagged fallbacks instead of being
reported as phase-continuous v3 grids. The successful B projections have a
maximum absolute correction of `4.048%`; the corpus-duration-weighted absolute
correction is `0.170%`.

The regression is not explained by large BPM corrections. A long section can
receive a sub-percent period change while its phase deviation grows from zero
at the left anchor toward the old seam residual at the right anchor. The
frozen strata make that failure visible:

| Pilot stratum | Comparable audio | B/v2 mean ratio | B/v2 p90 ratio | Interpretation |
| --- | ---: | ---: | ---: | --- |
| stable | 20 | 0.986 | 1.055 | acceptable |
| jump | 20 | 1.091 | 1.239 | mean passes locally; tail regression |
| dense | 10 | 1.179 | 1.298 | robustness failure |
| ramp audit | 10 | 1.334 | 1.576 | robustness failure; not ramp truth |
| long | 10 | 1.088 | 1.141 | inside aggregate guards but accumulation remains high |
| anomaly | 7 of 10 | 1.032 | 1.033 | two additional non-comparable fallbacks |

Family A is worse overall and is rejected as expected. Family B's stable slice
shows that exact continuity is benign when source seams are already small, but
dense and ramp-shaped source grids show that preserving every right anchor can
move too much phase inside the preceding section.

## Continuity, section count, and accumulation

Across all 80 source fits, the largest v2 seam is `997.143 ms`; it is an
out-of-distribution anomaly at about 21 BPM. On the 77 comparable rows the
largest source seam is `428 ms`. Successful Family B grids reduce the maximum
reported boundary seam to `3.48e-11 ms`, and JSON binary64 round-trip changes
no derived boundary at the recorded precision. Section-count delta is exactly
zero for every successful A and B projection.

Alias-normalized endpoint drift remains a separate warning:

| Metric | v2 | Family B | Change |
| --- | ---: | ---: | ---: |
| mean absolute endpoint drift | 5,848.381 ms | 6,664.100 ms | +13.9% |
| p90 absolute endpoint drift | 15,056.167 ms | 16,467.856 ms | +9.4% |
| mean raw absolute endpoint drift | 52,690.459 ms | 52,694.944 ms | approximately unchanged |

The raw values remain dominated by tempo aliases; the normalized increase is
consistent with the failed mean-phase guard. Exact seams therefore do not by
themselves establish a safe projection.

## Gate decision

| Frozen guard | Result | Decision |
| --- | --- | --- |
| mean phase no more than 10% worse | `+11.04%` | fail |
| p90 phase no more than 15% worse | `+13.19%` | pass |
| serialized boundary discontinuity <= 5 ms | `3.48e-11 ms`; round-trip delta 0 | pass |
| no section-count increase | maximum delta 0 | pass |
| projection fallback <= 5% | 2/80 = 2.5% | pass |
| no new cache/fit failure | 80 stored fits evaluated | pass |
| cumulative drift reported | raw and alias-normalized metrics retained | pass with concern |

Overall result: negative for Family B selection, positive for the v3 schema and
evaluation machinery. This is the preregistered mutation condition, not an
invitation to change the frozen thresholds.

## Problems found and resolved during the loop

- The first schema draft conflated the serialized first beat with the source
  beat-zero anchor. The origin is now allowed inside the first section, so both
  facts are represented by one mathematical authority.
- The first evaluator smoke conflated projection success with comparator
  availability and hid two fallbacks. Projection-evaluable and
  comparison-eligible denominators are now separate.
- Extreme positive beat lengths could overflow BPM/count math. Every source BPM
  and interval is now prevalidated; failures carry a source section index and
  finite JSON-safe diagnostics.
- Review added destructive path-collision guards, implementation source hashes
  to resume provenance, paired-intersection headline ratios, per-comparison
  malformed-oracle isolation, strict baseline-stage eligibility, and nonzero
  CLI status for evaluator failures. The final manifest is regenerated after
  these protections.

## Next-loop rule

Experiment 003 tests the already declared Family C: a parameter-free global
quadratic compromise that jointly moves change anchors and adjusts preceding
BPMs while directly penalizing section-wide source-lattice displacement. It
uses the same single cached BeatThis-derived v2 result, keeps `.osu` out of
inference, retains 20-1000 BPM and 5% guards, and adds a half-beat anchor
identity guard. The exposed 80 remains a repair/regression set; a new
audio-disjoint deterministic 100-group holdout is required before 500 and full
5,050 progression.
