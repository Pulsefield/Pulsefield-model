# Agent Note: Preserve long-form structure beyond the local R1 context

Note ID: 2026-09-20-r1-longform-structure
Status: proposed
Kind: research
Created: 2026-09-20
Updated: 2026-09-20
Product revision: 7fdeb7a282923670947514da4db65453a6c4d8be
Scope: Long-form structural stability and avoidance of diversity collapse; difficulty control deferred
Related: 2026-09-20-r1-training-distribution; 2026-09-20-r1-persistent-seed; 2026-09-20-r1-state-choice-audit

## Priority and evidence

The human owner prioritizes sustained long-form structure, specifically avoiding
pattern-diversity collapse, and defers cross-difficulty stability. Minimize
ablation branches and broad review overhead. Playability has priority over speed
and parameter count; scale the backbone or add modules when the evidence calls
for it, subject to measured Mac resource feasibility. The active goal authorizes direct
architecture changes, bounded training and local commits; no Note acceptance,
remote push, semantic-label promotion or final-quality claim is implied.

The current predictor's learned rolling content has a strict 511-token range.
Exact clocks/counts and the original seed persist, but learned representations
of generated passages outside that range do not. Complete-prefix replacement
improves later LN behavior and then re-drifts; persistent seed conditioning alone
fails its primary long-form gate. Native traces also contain concentrated repeats
and LN occupancy traps. These observations establish a relevant representation
limit and history dependence, not proof that finite memory is their only cause.

## Experiment Card: r1-longform-memory-v1

Revision 2, proposed, acceptance none. One candidate is compared with existing
5M outputs. There is no new zero arm or requested-star experiment.

**Question and mechanism.** Give the local predictor a learned record of earlier
generated passages: a small causal GRU over every committed physical row, with
retrievable landmarks every64 required head rows. The existing TCN handles
local action detail; attention to all completed landmarks can retrieve older
organization after it leaves the local window. The new memory reads only actual
committed rows and their permitted features. It introduces no audio, source
suffix oracle, external style request or repetition/duration rule.

[MusicVAE](https://arxiv.org/abs/1803.05428) is a close analogue for separating
local realization from longer-scale sequence organization. Here the mechanism
is causal memory and retrieval over committed 4K history, not a learned future
latent plan or a VAE. This adaptation does not inherit musical or gameplay
quality evidence from the paper. A future planner, backbone scaling and
on-policy recovery training remain alternatives if richer past memory does not
prevent collapse; they are not bundled into this first candidate.

**Baseline and data.** Start from clean product
`7fdeb7a282923670947514da4db65453a6c4d8be`. Parent: persistent-seed observed5M,
training source `a64ac0a0ed5b1ec6b7af5ab3b571cd122b30a5c4`, checkpoint SHA
`f5c272f1eaa3e79a488e9e11ef3bf16c9c6015275167e5c2bc98861816daf491`.
Keep its seed module, learned parameters, Adam state, counters and RNG. Extend
plan `3a8ebdc688ba1a94084f2826af525615ead940eb04a5fef99ebc117bcf666128`
with5.125M and6M milestones, preserving every old draw and source. No sampling
or optimizer change: CPU1, seed172, draws471, batch4/micro2, AdamW0.0003,
decay0.01, clipping1, inherited warmup. Train one additional1M-onset segment.
The parent has phase LN MAE0.133155 and below40ms union rate2.954803 per1000
required onsets on the28-group development cohort; these are diagnostics,
not long-form quality certificates.

**Implementation.** Add optional `long_memory=landmarks`, default none, with
memory width256 and stride64 required onsets. A shared-hand causal GRU encodes
raw permitted row features from the true beginning. Store its outputs at stride
boundaries. Readout attends only to landmarks strictly preceding the current
query, then adds a zero-initialized residual to existing hand vectors. Preserve
mirror symmetry and modular dimensions. Native execution updates this memory
on emitted physical rows; skipped candidates do not become content. Training
recomputes full-prefix memory under current weights instead of persisting stale
learned states across updates. Durable recovery stores raw history and rebuilds
memory, binding it to the model, external condition and committed row journal.
Legacy none-mode behavior and identities remain unchanged. Required changes
include typed/Hydra configuration, data projection, native/recovery owners,
checkpoint migration, focused invariant tests and concise operator documentation.

**Verification and bounds.** Before training, commit the clean intervention and
freeze input/config/driver digests. Check causal masking, full-prefix versus
cropped-query agreement, dense/native predictions, mirror behavior, expired-local
history sensitivity, recovery and legacy compatibility. Verify all inherited
weights/Adam state and initial logits. Benchmark real ordinary and largest-row
training windows before the run. Preflight at most180 seconds; new training
wall time at most3,600 seconds; process RSS/footprint6GiB, swap growth128MiB,
whole-owner outputs2GiB. Retain the parent's14,400-second cumulative lineage
limit. Stop on digest, mechanics, nonfinite, accounting or resource failures.
No silent overwrite or scientific resume; a same-plan interruption may use its
last audited durable checkpoint in a fresh segment within these bounds.

**Focused evaluation.** Reuse source indices3/9/15/19/21/23/26/27 from the fixed
28-group cohort (condition SHA
`7c757dbe90fcdaecb667af38e6b17b98a192595542d6aa62fcb742ea9e543bc7`).
Generate complete charts at seeds17/23,16 outputs total, native temperature1;
existing observed5M outputs at those identities provide the comparison.
Evaluation bound1,800 seconds and the same resource limits. Require complete
mechanics and exact export/reparse. Compare early/middle/late behavior and
rolling128-required-onset trajectories of LN amount, head-mask diversity, lane
concentration and repeated motifs. Use those trajectories only as locators:
legitimate jacks, repeated themes and purposeful sparse sections are not collapse,
and high entropy alone is not good structure. Inspect fixed early/middle/late
8-second contexts at seed17 plus the two worst sustained concentration/type-loss
contexts per arm, with entering/exit context and exact action witnesses. Inspect
seed23 wherever its locators reveal a worse sustained departure. Use the frozen
Beatmap Lens Foundation and human gold to judge independent LN organization,
TAP integration, coherent development and concerning self-repetition. Skip arm
masking; clearly identify this as an unblinded agent development review.

**Decision.** The practical gate is an improvement in the identified sustained
collapse/degeneration cases with no newly concerning long-form failure in this
fixed screen, retained independent LN/TAP organization, and no mechanical
failure or new below10ms regression. Quantity/diversity counts cannot pass that
gate by themselves. Report all unresolved cases and short-restart clusters;
remaining concerning failures prevent a final quality claim. Difficulty metrics
remain descriptive, not the selection objective. Extra training and architecture
are intentionally bundled, so the result cannot isolate their causal shares.
A successful screen justifies broader use/testing; failure directs the next
change toward planning or native-state recovery rather than more seed ablations.
An exploratory result has disposition REFINE and acceptance none.

Output owner: `artifacts/bounded-typed-continuation/longform-memory-20260920-v1/`.
Exact commands and generated identities will be frozen after implementation,
before training. No TEST payload, new semantic training labels or external
computation is needed.

## Result

Pending implementation and resource preflight. The global star/LN-request
proposal was an uncommitted draft and has no run, code or checkpoint to retain.

Revision2 increases the memory width before implementation and records the human
priority of quality over parameter/speed optimization. No run or implementation
used revision1. The inherited local width is a starting point, not a quality cap.


## Implementation and execution freeze

Clean implementation: `ed1b9b945d4aa7dc01f5e70dba55031dd1292bc6`, branch
`codex/r1-longform-memory`. The implemented256-wide GRU/landmark branch adds
446,848 parameters, giving2,777,232 with the inherited observed-seed model.
The base-to-intervention diff implements the declared full-history mechanism,
its typed configuration/projection, causal readout, native raw recovery and
journal binding, migration, tests and owning documentation. No quantity request,
sampling policy, decoder rule or additional training arm was added.

Selected CPU/MPS verification passes: the initial new-memory owner had7 passes;
the subsequent memory/fork/native/seed/training selection had66 passes. After
adding explicit full-history work counters and package coverage, the affected
memory/fork/smoke/package selection has21 passes plus22 package subtests, and
the final training/native selection has40 passes. These counts overlap and
must not be summed. An earlier complete bounded suite had201 passes and one
legacy-hash fixture failure: the fixture constructed its historical config with
new optional memory fields. Removing those fields from the historical fixture
restored the intended old-config test; its equality assertion was retained and
passed. No behavior failure was bypassed or excluded.

Real-data preflight finishes in12.690461 seconds. All inherited parameter values
and Adam moments are exact; median-chart initial NLL and common gradients are
exact;128 native initial decisions and RNG are exact against the parent, with
at least one completed landmark. Full-prefix forward/backward is finite for the
median770-row source and the largest26,976-row source. Their elapsed checks are
1.811036 and7.917173 seconds; the median check also includes native comparison.
Maximum sampled RSS is1,750,089,728 bytes, footprint1,620,609,184 bytes and swap
growth zero. These counters overlap. This is feasibility/equivalence evidence,
not generated-quality evidence.

Under the declared output owner:

- `prepare.py` SHA: `13bacc0e0800e8c30c36f59e120f909ebb75017277884eb3a8339db5e0d4bcbb`.
- `preflight.json` SHA: `c894807510f9e6afaecc80b205c044156b8ec781ef0ad37ee22c35c3856dbce0`.
- Extended plan SHA: `08e1a0980864aeaa994012b673bb4e7369913987fda2bd118c8474c6c298afc1`.
- `train-config.json` SHA: `d49ba6932101fbb90ddf1bd595ba8142a9d352e592004fca3d5d058e63296826`.
- `train.py` SHA: `ee3ba491cb2c04a12408ae4c3e8ba66058cdb8d066df291e9f676937e7616029`.
- `execution-freeze.json` SHA: `fdf763a230497ea1536de4eec2b9743186840632a74b59470bd0bc7d1eb27ee5`.

Run `uv run --offline --python 3.10 --extra mps python
artifacts/bounded-typed-continuation/longform-memory-20260920-v1/train.py`.
It runs one candidate serially, pausing at5.125M and resuming the same immutable
plan to6M in a fresh segment. The original parent is untouched. An outer
3,600-second alarm supplements the inherited cumulative training guard; caught
failures preserve the last durable boundary and record the failure. Card revision2
and its resource/quality scope remain unchanged; acceptance remains none.


## Focused baseline witnesses and evaluation procedure

The running candidate has crossed its5.125M checkpoint and continues the same
plan. No pilot quality selection or new intervention was introduced. Before
new generation, the evaluation driver recomputes all16 selected parent outputs'
clock counts and four-phase LN fractions exactly. Baseline-locator SHA:
`45a5b9f2f6bffc006055b471be197e2f46354a24f4907470827e918c390e01a3`.

Canonical Beatmap Lens source/generated contexts were inspected for the two
largest parent runs of consecutive head rows containing one lane:

- Source19,seed23: lane3 appears on77 consecutive required onsets over
  103,565–112,458ms. Head-row gaps range107–429ms, median107ms; there is no
  continuing hold and no rest of at least1,000ms. The generated context routes
  nearly all activity into that lane while the reference distributes the same
  rhythm across changing attack groups. Agent playability review: concerning
  sustained concentration, without a universal prohibition on jack passages.
- Source03,seed23: lane3 appears on54 consecutive required onsets over
  86,797–95,842ms. A2,133ms rest separates91,944 from94,077ms, so this is not
  nine seconds of uninterrupted physical load. Minimum/median head-row gaps
  are73/147ms; no continuing holds force that lane choice. The surrounding
  generated context also has extended concentration in another lane. Review:
  concerning routing degeneration, with the rest explicitly preserved.

All24 canonical pages across the two source/generated comparisons were viewed
through their complete reading montages, with exact action packets checked.
These are unblinded agent observations; formal style labels remain unreviewed
and source arrangements are not human gold. Both runs fit inside511 physical
tokens, so finite context length alone cannot explain these specific failures.
The candidate changes access to historical organization; if concentration remains,
planning/native-state recovery is more relevant than assuming that longer memory
by itself is sufficient. No short-gap threshold would locate these two failures.
Render-manifest SHA:
`302233e4faa605b0d04a25a4e92ce8d7d8d1d60755f62666cb0a953a7b395a43`;
review SHA `85799fe478d8589c95bec55af3122a2bb36a1d01bf753d26b39b35d7de9f07da`.

`evaluate.py` SHA is
`84d5ea653986f6e8147678df4b5c07a8e44d6d4497940e5cfda7f76918c244c6`.
`evaluation-procedure.json` SHA is
`1e68b5aee4a8845a48f9ac92eef86c892e068f6657f8c9200add7d6d45d19837`.
The procedure fixes the declared eight sources and seeds17/23,128-onset
nonoverlapping locator windows, the full matched parent/source references and
the1,800-second/2GiB-output bounds. A final execution freeze adds the completed
training receipt and checkpoint SHA before generation. Head-word entropy,
concentration, repeated motifs and LN amount locate structural inspection; they
are not optimization targets, calibrated style scores or acceptance by themselves.


## Training complete; generation freeze

The one candidate reaches6M with1,000,000 new onset exposures and1,317 updates
(final update7,936) in1,877.209920 seconds. Full-prefix work totals3,463,182
physical rows over those updates. Complete plan replay exactly reproduces
3,915,051 unique onsets across10,061 charts and3,169 groups. Weights are finite;
the memory output projection norm is0.447032, rather than its initial zero.
Maximum sampled RSS is2,534,981,632 bytes and footprint1,919,322,824 bytes,
with zero swap growth. This is one candidate with bundled extra fitting and
architecture; there is no new control-training arm or causal ablation claim.

Final checkpoint at `memory-6000k/checkpoint.pt` SHA:
`b44c9c83df84f2d9acd039edf64c1fc40377d2fc9ae0d25e9997ef548ae8876f`.
Training receipt SHA:
`67c7c6f58dac87bfb82f5da0bf3a7413ca55c2e974d44bd3de799ebf0d0764a9`.
Coverage/finiteness/resource audit SHA:
`b66b5c39c5a5269db721d4dfbe48a3297d72835b9a40b549cb9e424827904db5`;
its driver SHA is
`03b730b717a855f58b158819049b81bc040cd62b30d7fddd1cb61f46f7962280`.

The fixed16-output generation now binds that completed checkpoint through
`evaluation-freeze.json` SHA
`37c4ba01d991a471496ec0b905fb050100339cea51f41f57b32bea037ce10860`.
Run the same Python3.10/mps-extra command with `evaluate.py`. Driver, cohort,
seeds and limits match the frozen procedure. Long-form structure, native
repetition and LN/TAP organization remain the quality decision; stars remain
secondary diagnostics. No generation-quality conclusion is yet established.


## Long-memory result: REFINE

The fixed 16 complete generations finish in 275.199619 seconds. All exports,
reparses and mechanics pass; no below-10ms event appears. Below-40ms union
counts improve from 152 to 69, but the long-form quality gate fails. Candidate
19/seed17 has 143 quadruple TAP rows among 152 consecutive head rows containing
column 2 over 376458–394672ms. All 152 pre-states have no continuing holds;
there is no rest of at least 1000ms. Candidate 23/seed23 instead has 78 column-0
singles in a 79-head-row episode over 316175–324800ms, with three other lanes
occupied for 69 rows. The first is unforced chord collapse; the second persists
through a source transition from a preceding single-column episode into flow.

The source and candidate reading montages for both contexts were viewed in
full, using the pinned Beatmap Lens renderer and exact actions. These are
unblinded agent playability observations, not new human labels. Jack presence
is not itself a failure; the concern is sustained degeneration of the complete
passage. Source arrangements are reference examples, not human gold. A shorter
54-to-7 lane run on source03/seed23 does not offset these new failures. More
past memory plus fitting is insufficient in this candidate; the test does not
isolate memory from the additional 1M training exposures.

Readout SHA: `89479813c81ad1dff64f13631d2be1b3cf0fe32f1310f42a87448225fe91e581`.
Candidate render manifest SHA:
`54c0cf9a191f06651be73f78d630b85a19df785c9a4f9c5ad7f6efe399246af9`.
Candidate review SHA:
`782235dad1915f34655088d2ab05ab755cb0fec0199b14a915bc4ba5f53ae2fa`.
All live under the declared longform-memory artifact owner. Deviation: after
these decisive failures, stop the broader fixed-context semantic acceptance
review to honor the owner's reduced experiment/review scope. Those remaining
contexts are unreviewed. No adoption or final playability claim follows.

## Experiment Card: r1-native-recovery-v1

Revision 1, proposed, acceptance none. Execution is separately authorized by
the active goal. This is the active next Card; the memory Card's result remains
above. Start at clean product `ed1b9b945d4aa7dc01f5e70dba55031dd1292bc6`
with the completed 6M checkpoint pinned above. Preserve architecture, all
weights, Adam state, ordinary source sampler, support and native temperature.

**Mechanism.** Add a small complement-set likelihood term on the parent's own
TRAIN-generated prefixes. Ordinary teacher-forced CE continues. Penalize
continuing a sustained concentrated attack group, or continuing every long
blocking hold when that prevents escape; do not invent a unique correct next
action. Score the probability mass of legal alternatives using logsumexp.
[Unlikelihood training](https://arxiv.org/abs/1908.04319) is the primary analogue
for training against self-generated repetition. This is a 4K action-family
adaptation, not a novelty claim or inherited evidence of playability. Extra
memory alone is rejected as sufficient; explicit future planning remains an
alternative if learning escape from native states fails.

**Pool and heuristic.** Deterministically select 32 distinct TRAIN groups from
pinned census charts with 1500–6000 suffix onsets and at least 180 seconds,
ordered by SHA256 of `953:source_sha256`; no VAL or TEST payload is admitted.
Use generation seed43 and the fixed 6M policy. Replay the original R/H timing,
including empty R candidates, and only committed generated rows. A locator
requires 32 preceding head rows without a 1000ms rest; a core contains each
lane present on at least 28 of those rows. The source's corresponding 32-row
window must have maximum lane participation at most 24. These are conservative
machine-negative preferences, not human judgments or universal Jack rules.
At H, alternatives either omit at least one core lane or release a blocking
lane held since the start of that window. At R, alternatives release at least
one such blocking lane. Keep a query only when both families have legal mass;
never modify inference support. Space candidates by at least eight head rows,
cap each chart at 32 retained queries selected evenly through the chart. Stop
if fewer than 16 queries across four TRAIN groups are available; do not silently
relax thresholds. Freeze the complete pool and generation provenance before
training. The source comparison is training supervision only.

**Training.** Append 250k ordinary source-onset exposures to the unchanged 6M
plan prefix, with one final 6.25M checkpoint. At each update add two separately
recomputed native-prefix queries, mean-weight 0.25, selected deterministically
with seed954 and group-balanced sampling. All original optimizer settings,
batch4/micro2, CPU1 and seeds remain unchanged. Full generated histories and
original seeds are recomputed under current weights; no stale learned bank is
trained through. Keep the original runtime/resource guards. New pool generation
at most1800 seconds, training at most1800 seconds, evaluation at most1800
seconds; RSS/footprint6GiB, swap growth128MiB, total owner2GiB. Fail on provenance,
mechanics, nonfinite loss or resource violations. No automatic extra arms.

**Decision.** Reuse the fixed 16 development outputs, compare against both the
existing 5M parent and 6M memory candidate. Primary practical gate: neither of
the newly inspected collapse mechanisms persists in a comparable sustained
form, and no new equally concerning long-form episode emerges. Locators rank
review only; lower repetition alone does not establish quality. Check complete
chart mechanics, no new below-10ms events, and inspect the worst repetitions
plus LN/TAP contexts against Beatmap Lens foundation and relevant human gold.
Reject indiscriminate randomization, removal of authored Jack behavior or loss
of LN/TAP coordination. Cross-difficulty stability remains deferred. A positive
result supports further focused playability review; negative or ambiguous
results are REFINE, with no claim of causal isolation from extra CE fitting.

**Implementation and handoff.** Add native-prefix batch preparation, reusable
batch action log-probabilities, pinned TRAIN-only recovery pool, explicit typed
training objective configuration and same-architecture objective fork support.
Verify dense/native agreement including skipped R slots, source/suffix plan
visibility, finite complement gradients, legal/nonempty alternatives, pool
split/digest rejection, default checkpoint compatibility, and runner consumption.
Commit source and freeze driver/config/inputs before each model-backed run.
Use fresh output owner `artifacts/bounded-typed-continuation/native-recovery-20260920-v1/`
in the longform-memory worktree. No implicit overwrite or unmeasured resume.
