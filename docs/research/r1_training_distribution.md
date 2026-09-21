# Training distribution of the bounded R1 corpus

The completed 5M-onset training plan spends 80.0803% of its exposures on
2–6-star source charts and only 1.39394% above 6 stars. Its remaining 18.52576%
is below 2 stars. These measurements weaken a simple explanation that generated
high difficulty comes from predominantly high-difficulty training data. They do
not identify the cause of generation errors or establish that sampling is optimal.

This census describes the corpus used by
[bounded typed continuation](bounded_typed_continuation.md). It makes no model,
objective, sampler or decoder change and assigns no new semantic labels.

## Population and measurements

The population contains all 11,563 eligible TRAIN charts in 3,169 admitted song
groups. It has 10,735,674 post-seed required onsets. The completed plan consumes
5,000,000 onset exposures through 26,462 draws, covering 3,459,305 distinct
onsets in 9,533 charts and all 3,169 groups. The seed is the complete first row
prefix reaching at least 30 heads; suffix statistics exclude its heads while
retaining its history for subsequent action-clock measurements.

Each original source SHA, admitted metadata SHA and row-cache SHA is verified.
Original note objects and reconstructed cached objects agree as full multisets.
Stars use the pinned local mania 20241007 calculator, native 4K, no mods and rate
1. Original file order is preserved when calling its legacy sort; cached row
order is not substituted. No audio content or TEST source payload is opened.
A source's star rating does not establish a generated output's difficulty.

The four weightings answer different questions:

- **Charts:** each admitted chart has equal weight.
- **Expected draw:** a group is uniform, then a chart within that group is uniform.
- **Available onsets:** each distinct post-seed source onset has equal weight.
- **Consumed onsets:** each actual training exposure has equal weight, including
  repetitions and shortened windows at plan milestones.

| Source stars | Charts | Chart share | Expected draw share | Available onset share | Consumed onset share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Below 2 | 2,576 | 22.278% | 19.627% | 8.532% | 18.526% |
| [2,3) | 2,878 | 24.890% | 23.008% | 16.368% | 23.360% |
| [3,4) | 2,711 | 23.445% | 23.644% | 23.023% | 24.154% |
| [4,5) | 2,140 | 18.507% | 22.533% | 28.324% | 22.841% |
| [5,6] | 956 | 8.268% | 9.641% | 17.640% | 9.725% |
| (6,8] | 289 | 2.499% | 1.479% | 5.855% | 1.363% |
| Above 8 | 13 | 0.112% | 0.068% | 0.257% | 0.031% |

The source range is 0.669818–8.777979 stars. Group/chart sampling gives
5–6-star charts substantially less exposure than their available-onset mass.
This is a property of the stated sampler, not a source-quality judgment or
proof that rebalancing improves generation.

## LN amount, chord size and seed representativeness

The available suffixes contain 2,482,778 LN heads among 15,728,453 total heads
(15.7853%). Exact consumed windows contain 1,295,558 LN heads among 7,214,828
heads (17.9569%). Computing consumed counts from whole-chart averages would
ignore the actual crop distribution. Among the 5M consumed onset rows,
3,129,691 are single heads, 1,546,151 doubles, 303,797 triples and 20,361 quads.

LN-rich charts are a small part of exposure when richness is defined only by
amount: 5.83064% of exposures come from charts with at least half their suffix
heads as LNs, and 1.27144% from charts with at least three quarters. These are
quantity thresholds, not Beatmap Lens LN-presence or salience labels. Conversely,
840 charts have no suffix LN heads and receive 6.04162% of exposures.

For each chart, divide its suffix into four consecutive equal-required-onset
phases using boundaries `floor(N * b / 4)`. Measure LN heads divided by all heads
in each phase, then average the absolute difference from the original seed's
LN-head fraction. Empty phases in the shortest charts are omitted from this
per-chart mean. The chart-mean discrepancy is 0.145142, or 14.51 percentage
points; its median is 9.75 points and its 90th percentile is 35.18 points. Under
expected-draw weighting the mean is 15.87 points; weighting each chart by its
actual consumed onsets gives 16.01 points.

A descriptive follow-up substitutes each chart's full-suffix LN fraction for
its seed fraction in that calculation. The chart-mean discrepancy is 4.61 points.
This arithmetic reference uses future source information; it is neither a
seed-only predictor nor a generated chart. It illustrates the difference between
preserving the introduction and supplying a whole-chart quantity request.
Neither operation guarantees the independent press/release relationships or
LN/TAP organization required by the semantic evaluation.

## Action clocks and annotation coverage

At every suffix head, compare its time with the preceding head and preceding
release in the same lane. Count strict intervals below each threshold separately,
and count a head satisfying either predicate once in the union. The clocks
include history before the sampled window; they do not restart at crop boundaries.

| Population | Below 20 ms: head/head, release/head, union | Below 40 ms: head/head, release/head, union |
| --- | --- | --- |
| All available suffixes | 0, 9, 9 | 132, 1,450, 1,582 |
| Exact 5M consumed exposures | 0, 2, 2 | 19, 327, 346 |

The consumed below-40-ms union rate is 0.0692 heads per 1,000 required onsets.
The presence of such events in source charts reinforces that the thresholds
are diagnostics, not universal legality or playability cutoffs. This TRAIN
population is not matched to the difficulty/LN-enriched generation evaluation,
so a direct rate ratio would not isolate a model effect.

Only the `source_sha256` column of the pinned publication's human table is read.
Its 592 raw records join to 466 records on 149 admitted TRAIN charts, covering
1.2886% of the training charts. Of those, 409 records on 140 charts fall within
2–6 stars; the remaining 57 records on nine charts are above 6 stars. No joined
human record is below 2 stars. These are source/record coverage counts, not
positive labels, effective concept cells, label balance or supervision weights.
No label, rationale or evidence columns are read, and no classifier is evaluated.

The sparse coverage limits what can be inferred about corpus-wide semantic
supervision. Counts and source stars do not replace the frozen Beatmap Lens
Foundation and human examples used to inspect generated organization.

## Verification and provenance

The source census completes in 783.14 seconds on one Apple M5 CPU thread with
Python 3.10.20, NumPy 1.26.4 and Torch 2.11.0, launched with the explicit `mps`
extra. Peak sampled RSS is 531.94 MB and footprint 443.84 MB, with zero swap
growth. A separate physical-row replay verifies every chart's seed, phase,
clock and exposure count, using direct per-onset draw multiplicities rather
than the census's window prefix sums. It also checks all aggregate weightings
and source-only annotation joins. That recount takes 405.99 seconds, peaking at
537.69 MB RSS and 437.86 MB footprint with zero swap growth. RSS and footprint
overlap and must not be added. It checks accounting independently, not a second
implementation of the star formula. The complete evidence owner uses 45.78 MB.

Runtime source: `342fa15294670d9e790d94c7c604f2c525cc3057`.
Evidence owner: `artifacts/bounded-typed-continuation/training-distribution-20260920-v1/`.
The population is a complete census of the pinned plan, not a random estimate;
no sampling confidence interval is assigned to its descriptive totals.

- Plan SHA: `3a8ebdc688ba1a94084f2826af525615ead940eb04a5fef99ebc117bcf666128`.
- Calculator SHA: `3faaed183a3ef7f62b6ea58f0b0b156cc707c94b760717f78794224a0fb722e9`.
- Publication manifest SHA: `92ecf080737ec2e38a2508b0730a672e09edd45d48f46a2d5777cbb70cefcbcf`.
- Human table SHA: `4ba27584fb3fe4db7b8f8d051d20debe70c0af6b9be91603cbb283c65f6f2b70`.
- Readout SHA: `c4f93fe8560b2e791a0ef66c662e97613682e08bf70009203777689e58501bab`.
- Per-chart SHA: `bd77030a9452bb31228bf530c78f0cc567eee4c86d95f5c96734d092fd282905`.
- Independent audit SHA: `c5076a25249e87d46c391071c1287502c2b688f934bc3cd9cb2c32f06b6125a1`.
