# Agent Note: Does the active training distribution match the playable 2–6-star goal?

Note ID: 2026-09-20-r1-training-distribution
Status: proposed
Kind: investigation
Created: 2026-09-20
Updated: 2026-09-20
Product revision: 342fa15294670d9e790d94c7c604f2c525cc3057
Scope: TRAIN-only difficulty, LN/type, action-clock and exposure census for the current bounded R1 plan
Related: 2026-09-20-r1-state-choice-audit; 2026-09-20-r1-persistent-seed

## Question and authority

The generation goal requires stable playable outputs at 2–6 stars, while the
active sampler chooses a song group and then a chart uniformly without a star
or requested-profile field. The 28-chart development evaluation deliberately
uses 2–6-star sources; that selection does not establish the training population.
Measure the actual training distribution before choosing a data change, explicit
intent condition or larger model. The active goal authorizes this bounded audit
and local commits. No new training, filter, inferred semantic labels or accepted
research Card follows from this investigation.

## Frozen inputs and procedure

Use all 11,563 eligible TRAIN sources and 3,169 groups from the completed 5M plan,
SHA `3a8ebdc688ba1a94084f2826af525615ead940eb04a5fef99ebc117bcf666128`.
The inherited catalog SHA is
`e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28`,
and split SHA is
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.
Verify every source and admitted cache digest. Read original .osu objects for
star calculation, preserving their ordering, and check their full object
multiset against the lossless cached rows. Use the existing pinned 1x mania
calculator, SHA
`3faaed183a3ef7f62b6ea58f0b0b156cc707c94b760717f78794224a0fb722e9`.
No audio content or TEST source/annotation payload is used.

Report star bands below2, [2,3), [3,4), [4,5), [5,6], (6,8], and above8 under
four weightings: chart count, expected group-then-chart draw probability,
available suffix onset mass, and actual consumed 5M onset exposures. Report LN
head fraction, chord-head multiplicity, four equal-onset phase fractions,
seed-to-phase mismatch and strict20/40 ms head/head and release/head counts.
The clock predicate is a diagnostic, not a source-quality filter. Counts and
LN amounts are not Beatmap Lens semantic labels. Aggregate actual draw LN/head
exposure from the exact source windows rather than a whole-chart approximation.

For annotation coverage only, project the source_sha256 column of the pinned
publication's human table. Join only admitted TRAIN identities to computed
stars. This counts source/record coverage, not effective concept cells or
positive/negative judgments. Do not read rationale, evidence or label columns.
Publication manifest SHA:
`92ecf080737ec2e38a2508b0730a672e09edd45d48f46a2d5777cbb70cefcbcf`;
human parquet SHA:
`4ba27584fb3fe4db7b8f8d051d20debe70c0af6b9be91603cbb283c65f6f2b70`.

Preflight the first eight source SHAs and four largest-row sources, deduplicated,
for exact cache/object agreement and runtime feasibility within120 seconds.
Freeze the driver digest after that check, before computing population summaries.
Run serial CPU1 with Python3.10 and the explicit mps extra, at most2,400 seconds,
2GiB RSS/footprint,128MiB swap growth and128MiB fresh outputs. Stop on a digest,
object/row, nonfinite statistic, accounting or resource failure; preserve partial
output and do not silently omit a chart. Output owner:
`artifacts/bounded-typed-continuation/training-distribution-20260920-v1/`.

## Interpretation and next design boundary

A large exposure share outside 2–6 stars would support separating target intent
from corpus breadth or testing a scoped sampler; a small share would weaken a
simple wrong-difficulty-corpus explanation. Seed-to-phase mismatch measures the
limits of treating an introduction as a whole-chart profile, without proving a
particular conditioning design. Dense annotation coverage would make direct
style supervision more plausible; sparse coverage would require keeping its
role separate from corpus-scale generative learning. None of these measurements
is a causal estimate of a training change or a final quality assessment.

After the census, choose a bounded model experiment with one declared question.
Explicit source-derived profile controls would change the stage's inference
contract and must be labeled as supplied requests, not hidden R1 information.
A larger model or new sampling policy cannot inherit a quality claim from this
read-only census. The ultimate goal and the independent Beatmap Lens assessment
remain unchanged.

## Result

Complete source census and independent physical-row recount; descriptive interpretation is REFINE. No model-quality or adoption claim follows.


## Execution freeze

The first eight lexicographic source SHAs and four largest-row sources give
12 distinct preflight charts. Original object multisets, admitted cache bytes,
seed rules and onset counts agree for all12. Synthetic checks cover strict20/40
boundaries, seed history in suffix clocks, exact exposure windows and star-band
endpoints. The preflight completed in14.583925 seconds, with maximum sampled
RSS434,110,464 bytes, footprint352,961,688 bytes and zero swap growth.

The clean product remains `342fa15294670d9e790d94c7c604f2c525cc3057`.
Under the declared output owner, the frozen driver `census.py` SHA is
`d2486a81a93c08ab94808db32079affbffca291ea750ef3c0218d2543dd556c0`;
`preflight-v1/receipt.json` SHA is
`99f19d5f40c945d54c4bca8ec12f94ac739c2ab94d98a6ba5fe1c88afeedb83b`;
`freeze.json` SHA is
`ede0a5257f825a65ad255f799e4f4a401baf229afb15f72af239932c2de27017`.
The full census runs with `uv run --offline --python 3.10 --extra mps python
artifacts/bounded-typed-continuation/training-distribution-20260920-v1/census.py`.
The128MiB output limit includes preflight and full-census files. Failed output
is retained; a rerun needs a fresh declared destination. No population aggregate
was computed before this freeze. The freeze-writing command succeeded; its
subsequent console-only resource summary failed on a startup journal entry
without a footprint field and was corrected without changing the freeze bytes.
Acceptance remains none; execution follows the active goal's separate authority.


## Population receipt and independent recount freeze

The complete source census finished all 11,563 charts in 783.143477 seconds.
All original/cache comparisons and exact 5M exposure-accounting guards passed.
`results-v1/readout.json` SHA is
`c4f93fe8560b2e791a0ef66c662e97613682e08bf70009203777689e58501bab`;
`results-v1/charts.jsonl` SHA is
`bd77030a9452bb31228bf530c78f0cc567eee4c86d95f5c96734d092fd282905`.
Maximum sampled RSS is 531,939,328 bytes, footprint 443,843,856 bytes and swap
growth zero. The output owner occupies 45,763,873 bytes before independent audit.

Before interpreting the aggregates, independently replay every cached physical
row without `SourceChart` or the census helper. Direct per-onset draw multiplicity
weights must recover the exact window totals and each milestone interval. Check
seed boundaries, all four phase counts, strict clocks, band membership, group
probabilities, annotation source-only coverage and all aggregate weightings.
This verifies counting, not a second implementation of the pinned star formula.
The serial CPU1 audit uses the same 2,400-second, 2GiB process, 128MiB swap-growth
and 128MiB whole-owner output limits. Its driver `check_result.py` SHA is
`b6db95d2af74dfec3a8d525e3654fb67a578356658e6c27391096210340c039f`;
`audit-freeze.json` SHA is
`9a0a8cdb8841735ae6719c87a18b3e83070ff3c6bc69a634fe8af4a9fd02ec20`.
Run with the same Python3.10/mps-extra command, replacing `census.py` with
`check_result.py`. The audit uses a fresh `independent-audit-v1/` directory.


## Complete result and interpretation

The independent recount passes all 11,563 charts in 405.994565 seconds. Its
receipt `independent-audit-v1/audit.json` SHA is
`c5076a25249e87d46c391071c1287502c2b688f934bc3cd9cb2c32f06b6125a1`.
Maximum sampled RSS is 537,690,112 bytes, footprint 437,863,672 bytes and swap
growth zero; the complete owner uses 45,777,799 bytes. This independently checks
counting and weighting, not a second star-rating implementation.

Of the actual 5M onset exposures, 4,004,015 (80.0803%) are on 2–6-star sources,
926,288 (18.52576%) below 2, and 69,697 (1.39394%) above 6. Thus excessive
high-star training mass is not a strong simple explanation for over-difficult
generation. The 5–6 band still receives only 9.72486% of exposure despite
17.63990% of available onset mass, so targeted rebalancing remains distinct
from broadly removing hard charts. The census reproduces 3,459,305 unique
onsets, 9,533 exposed charts and all 3,169 groups.

Consumed windows contain 1,295,558 LN heads among 7,214,828 heads (17.9569%);
source suffixes contain 2,482,778 among 15,728,453 (15.7853%). The full four-phase
seed-to-suffix LN-fraction discrepancy has chart mean 0.145142, expected-draw
mean 0.158651 and actual-exposure-weighted chart mean 0.160111. Empty phases in
very short suffixes are omitted, as fixed in the frozen driver. A descriptive
follow-up gives median 0.097536 and 90th percentile 0.351811. Replacing the
seed fraction with the full-suffix fraction gives chart-mean phase discrepancy
0.046123; this is a future-informed arithmetic reference, not a generator or
an R1-permitted estimate. Charts with suffix LN fraction at least 0.5 receive
5.83064% of exposures; at least 0.75 receive 1.27144%. None is a style label.

Available suffixes have 132 head/head and 1,450 release/head events below40ms,
union 1,582. Consumed windows have 19 and 327, union 346 (0.0692/1000 required
onsets). Below20ms the source/consumed unions are 9/2, all release/head. These
source events are not rejected or converted into a universal demand threshold.
The population differs from the enriched generation screen, preventing a causal
claim from a direct train/generated rate ratio.

Source-only human-table projection joins 466 raw records on 149 TRAIN charts
(1.2886% of admitted charts), including 409 records on 140 charts within 2–6.
No label/evidence/rationale column was read. This does not recompute effective
concept cells or label balance, and supplies no new semantic training label.

No chart was skipped, no population gate changed, and all declared runtime and
resource limits passed. The descriptive follow-ups use the completed per-chart
census and are not prospective decision gates. Durable findings are published
in `docs/research/r1_training_distribution.md` with a link from the bounded
continuation owner, product commit `7fdeb7a282923670947514da4db65453a6c4d8be`.
Those two Markdown files pass scoped numeric/digest/link checks and diff checks;
no behavior or model tests are claimed for this documentation-only change.

Disposition: REFINE. Long-form structural stability now has priority; cross-difficulty
control is secondary. The unexecuted global star/LN-request draft is withdrawn
before its first commit or implementation. Preserve this census as evidence,
without starting its previously considered control ablation. The next owner is
`2026-09-20-r1-longform-structure`. The census does not identify a unique cause
of collapse or native traps. Acceptance remains none and status remains proposed.
