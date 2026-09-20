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

Pending source/object preflight and complete TRAIN census.


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
