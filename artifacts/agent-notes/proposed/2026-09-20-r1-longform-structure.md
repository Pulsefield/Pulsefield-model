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


### Native-recovery implementation and pool repair

Clean implementation: `fe6983004b69189888f4901c6541bcb0ed0606f1`.
The focused existing data/train/fork selection passes 41 tests; eight new
recovery checks cover CPU/MPS dense/native predictions, skipped R candidates,
current-placeholder noninterference, stable complement gradients, legal hold
escape, pinned TRAIN admission, same-architecture fork/strict-resume equality,
Hydra projection and real runner consumption. The final recovery/fork/package
selection passes 18 tests and 22 package subtests; counts overlap. Diff checks
pass. Only the declared objective/projection/configuration and documentation
changed; native support, policy architecture and sampling remain unchanged.

Pool generation v1 stops after 13 complete trajectories because its artifact
script incorrectly required the last physical replay row itself to be terminal.
The existing contract permits an unused final R candidate when all LNs are
closed. Driver SHA:
`8d4107c065f0f1faebb15e412fc8196e8eb85435f1bd84a077fe4f52ecce63fb`.
Its partial outputs and failure receipt are retained. A conservative 440-second
wall upper bound includes debugging delay. Fresh pool-v2 restarts the identical
32 selections and seed43 policy, using schedule completion plus empty occupancy;
its remaining limit is 1360 seconds, so total charged harvest remains within
1800 seconds. This is a completion-assertion repair, not a changed preference
threshold, cohort or policy. The implementation source is fe69830; the policy
checkpoint and its training source remain the pinned 6M/ed1b9b9 parent. The new
freeze pins the complete repaired driver before its first draw. No training
will run unless the fixed minimum query/group gate passes.


### Native pool and training freeze

Pool-v2 completes all32 trajectories in269.439678 seconds:80 eligible queries
from8 TRAIN groups. Its manifest SHA is
`f712c597eefb8269f61a540c3459aad284c247b0ce3e61d9d8cc6fd5af4abbc3`.
The first13 trajectory files exactly equal the pre-repair outputs. Total
conservatively charged harvest is below710 seconds. No threshold or selection
was relaxed. All queries are H decisions with no blocking-hold mask: this pool
trains escape from unforced concentration, not demonstrated recovery from LN
starvation. The latter remains an independent evaluation failure condition.

Real-data preflight takes8.235387 seconds. All inherited weights/Adam values are
exact. The longest selected native prefix is1818 physical rows; its dense/native
logits agree, complement loss is2.945007 and backward is finite. That full query
check takes1.805457 seconds. Preflight driver SHA:
`a77b1792645be114515451d98a0f4f82ff685824c5c0bdb1ab6d40e0320e00e2`.
The complete hashed configuration, plan, pool, preflight and training driver are
bound in `native-recovery-20260920-v1/execution-freeze.json` before training.
Run `uv run --offline --python 3.10 --extra mps python
artifacts/bounded-typed-continuation/native-recovery-20260920-v1/train.py`.
The one candidate keeps all declared settings and stops at6.25M, with the
1800-second new-training guard and inherited lineage/resource limits.

The focused generation driver has SHA
`e6b32c9b92417cbddf5550b04b50fd94215812ff4f3f419b55569ac1c443823b`.
It reuses all16 fixed source/seed pairs and verifies both existing5M and6M
row journals before comparison. Final checkpoint and completed training receipt
will be pinned before generation. No exploratory pilot selection is added.
Beatmap Lens human gold images03f7e3 (LN prominent),2af746 (Stream prominent with
LN/TAP) andb8e955 (Stream absent with repeated changing chords) were refreshed.
Their style judgments distinguish organization from counts; the last example
is not a judgment that repeated chords are universally unplayable.


### Native recovery training complete; fixed generation begins

The single6.25M candidate completes250,000 new source onsets in333 updates,
with666 native preference queries, in729.789842 seconds. Final update is8269;
coverage is4,022,343 unique source onsets across10,176 charts and3,169 groups.
Peak sampled RSS is1,772,355,584 bytes, footprint1,105,135,872 bytes, swap growth
zero. Complete plan/coverage replay, journal boundary and query accounting agree;
all model weights are finite. Mean recovery loss across the first/last32 updates
is0.328905/0.004369 on the sampled training pool, which is not generated-quality
evidence or an independent generalization measurement.

Checkpoint `recovery-6250k/checkpoint.pt` SHA:
`fee32588f4e353e04b33d6299494dc023a4902e9e848e357c8058c18d89f2592`.
Training receipt SHA:
`8736e8ba8a409b69296741bf9e1f15abf8f3e7cc070026c98f125ca6c9559507`.
Audit driver SHA:
`dbb094874b7354c0b7ec086f30c43d1242f1cda90f794db35c74d3c6b5c8fbc4`.
`evaluation-freeze.json` pins the finished training receipt, audit, final model
and predeclared driver before all16 complete generations. It retains the same
source/seed pairs, mechanics checks, two existing model references and1800-second
bound. No decision is made from the training-loss decline.

The approved slow-Jack calibration was also rendered and viewed with its full
9400–16300ms context, preserving the human-positive400ms recurrence and its
entry/exit. Gold manifest SHA:
`25add75e70e59fddd7fe0d51651705daff18c617f780d868d1e1cefced4e8b50`.
It is a TRAIN calibration example, not independent model validation.


## Native-recovery result: REFINE; retention guard failed

All16 fixed outputs finish in281.510863 seconds, with legal mechanics, successful
export/reparse and zero below-10ms events. Pooled below-40ms diagnostic counts
are152 at5M,69 for6M memory and3 for6.25M recovery. The maximum number of
consecutive required H rows containing a fixed lane changes77 ->152 ->11.
The two historic failed regions no longer contain the quadruple wall or the
three-held-lane lock:19/seed17 distributes150 of152 attacks as singles across
all four lanes, plus two chords;23/seed23 distributes all79 as singles after its
preceding held lane releases. These are real improvements in those witnesses.

The LN retention guard nevertheless fails. On source09/seed17 at
97684.5–105684.5ms, the6M/candidate counts are79/5 LN heads,35/0 H rows with
at least two held lanes, and22/0 independent release rows (including R-only
rows). On source21/seed17 at220097–228097ms, corresponding counts are73/36,
19/0 and36/3. Full canonical source/candidate montages show the first becoming
TAP flow with isolated short LNs, and the second retaining short-LN/TAP sequences
while greatly weakening overlapping independent control. This is not a judgment
from LF alone. The refreshed human LN/Stream gold distinguishes these relations
from merely having green notes. The sources are references, not human labels
for generated outputs.

A further exact head-word period1–16 locator finds bounded outer-column Trills:
64 attacks over4.922 seconds on27/seed17 and47 over5.250 seconds on19/seed23.
Their source/candidate contexts were viewed in full. These are simpler than the
source, but Trill repetition is not automatically poor playability. They remain
important diversity checks, especially since a same-lane-run count misses them.
The fixed historic candidate montages, two LN source/candidate comparisons and
two cycle source/candidate comparisons were reviewed. Rendered LN15/LN27 and
remaining longest-lane witnesses are unreviewed; acceptance review stops at the
clear retention failure under the user's reduced-overhead priority.

Readout SHA: `9df7ac1609f16c6c2c4353db736fff40f35ea476398d635ebed547ea8b3b9097`.
Semantic review plus exact LN action recount SHA:
`e9f8d53eb436f9083dde7de9dd8cc0bdc2fb94a3dfd8ab69bd7c381380866be2`.
Historic/LN/cycle render manifest SHAs:
`635b6039d0609554bbdf67186f083071af3420fbd6a7228cefd472cd242d3139`,
`237a0f50c6af507b4c50d6e520767d391e9f847957b32f0a521c1da7ba474248`,
`a478ba54a666e927a6eae1536162e42350d1e9c0ab9c86d5d73131f11ae24c06`.
No new human annotation is created. Full-parameter recovery is not adopted.
Ordinary CE fitting and the new objective are bundled; their separate causal
contributions to simplification are not established. Product conclusions are
published in clean docs-only descendant
`004c98da74de45f80c8ae18626f939b8f339b37a`.

## Experiment Card: r1-routing-recovery-v1

Revision1, proposed, acceptance none; active next Card. The user's goal separately
authorizes implementation/runs and modular scaling. Preserve the failed full-
parameter candidate and return to the6M parent for this one targeted correction.
No run or code for this Card exists yet.

**Mechanism and branch.** Restrict the native-recovery update to a new nonlinear
head-mask residual while freezing the entire inherited policy. For head-mask
G and full legal action A, use score(A)=base_score(A)+route_score(G(A)). The
constant shift within each head-mask family preserves the base conditional
P(A | G,state), including TAP/LN typing and simultaneous release choices.
At R-only candidates add exactly zero, retaining the original release policy
for the same state. This is a structural preservation constraint rather than
a smaller loss weight. It cannot guarantee identical future LN structure after
routing changes the sampled states; native quality must still be checked.

The closest learning analogue remains unlikelihood training; the new component
is ordinary conditional-family reweighting, not a novelty claim. Full-policy
recovery removed self-repetition but lost LN organization. A restricted learned
routing module is therefore the next direct candidate. A future section planner
remains an alternative if restricted recovery merely moves collapse into fixed
cycles or fails to retain LN coordination. Do not start a broad ablation suite.

**Baseline and module.** Implement from clean `004c98da74de45f80c8ae18626f939b8f339b37a`,
a docs-only descendant of the executed fe69830 source. Use the unchanged6M
checkpoint `b44c9c83df84f2d9acd039edf64c1fc40377d2fc9ae0d25e9997ef548ae8876f`,
training source ed1b9b9, and its complete6M plan. Add an optional R1-only head
routing module, default none. A512-wide MLP reads the pair of inherited hand
vectors and produces16 binary head-mask scores. Average its output with the
mirrored output under swapped hands and reversed lane bits, preserving mirror
equivariance. Zero-initialize the last projection for exact initial behavior.
This adds roughly140k learned parameters; scale further only if this mechanism
needs capacity. Keep all inherited weights frozen, including memory, seed,
TAP/LN/action and release computation. Preserve their existing Adam state;
only new route parameters receive gradients and optimizer updates.

**Data/objective/procedure.** Reuse the exact80-query,8-group TRAIN pool SHA
`f712c597eefb8269f61a540c3459aad284c247b0ce3e61d9d8cc6fd5af4abbc3`.
Do not reharvest or train on VAL witnesses. Fork the same6M parent and append
250k ordinary source onsets using the exact already-frozen6.25M plan
`37aa23798a4184e3990daf791033a46dbf790bb1acfda29001a5b645555a565a`.
Use the same CE plus mean0.25 complement objective, two native queries/update,
seed954, CPU1, batch4/micro2, parent optimizer settings and ordinary corpus seed.
The CE also updates only the new routing module. Preserve inherited RNG/counters;
restore/fork identity explicitly admits the new module and trainable scope.
Default checkpoints, no-module identity, Hydra projection and recovery remain
compatible. All raw histories are re-encoded under the fixed inherited policy.

**Checks and bounds.** Before training, verify exact initial logits/native draws;
conditional action distributions within every legal head mask; exact R-only
logits; nonzero route gradients with every inherited gradient absent; unchanged
inherited weights/Adam values after real updates; mirror symmetry; native/dense
agreement; and strict resume. Commit the intervention and pin driver/config/input
digests. Preflight180 seconds, training1800 seconds, evaluation1800 seconds,
RSS/footprint6GiB, swap growth128MiB, fresh owner2GiB. Output owner:
`artifacts/bounded-typed-continuation/routing-recovery-20260920-v1/`
in the longform-memory worktree. No silent overwrite or unmeasured resume.

**Decision.** Generate the same16 complete development cases with seeds17/23.
Compare against existing6M and full-parameter6.25M outputs, with no new control
training. Require the historic unforced collapse improvement to survive, no
new similarly concerning long-form degeneration, and retention of meaningful
LN/TAP overlapping independent control in the fixed09/21 contexts. Reuse the
Beatmap Lens foundation and refreshed human gold. Count/entropy/period and LF
metrics locate review rather than certify playability. Mechanics/export must
pass and no below-10ms event may appear. Fail fast on lost coordination, new
sustained fixed cycles or resource/accounting violations. Difficulty calibration
remains secondary. Positive results justify focused fresh-development verification;
negative or ambiguous results remain REFINE. No automatic adoption follows.


### Routing implementation and real-data preflight freeze

Clean intervention: `8cf31e8d177fab28060ce92be4f5e92f8f8585f3`.
The module, explicit routing-only trainable scope, fork/identity handling,
packaged configuration and conditional-preservation documentation implement
the proposed Card without changing inference support or temperature. Six new
routing tests cover CPU/MPS initial equivalence, nonzero-score conditional and
R-only preservation, mirror symmetry, frozen gradients/weights, native/dense
agreement, raw-state recovery, inherited Adam equality, real runner consumption
and exact resumed updates. Routing/memory/seed selection passes23 tests.

The broader fork/train/native-export/smoke/package selection initially has63
passes,22 package subtests and one legacy-fixture failure: the extended tiny
model helper used a nondefault dormant routing width in its old-format snapshot.
The helper now keeps the historical default while routing-enabled cases remain
small. Earlier legacy digest fixtures also explicitly exclude the new fields
when constructing the historical schema; their exact hash assertions remain.
Final affected seed/native-export/smoke checks pass36 tests. Counts overlap;
no behavior failure is hidden or skipped. Diff checks pass.

`routing-recovery-20260920-v1/preflight-freeze.json` pins the clean source,
preflight driver,6M checkpoint and unchanged native pool/6.25M plan before
real-data execution. The180-second preflight checks128 exact initial native
draws and logits, then applies one real source/native optimizer update and
verifies every inherited weight and Adam value unchanged, nonzero new-module
gradients, conditional-family preservation and exact R-only logits.


### Routing preflight passes; one continuation is frozen

Real-data preflight completes in7.539603 seconds. The model grows from2,777,232
to2,917,008 parameters; only139,776 routing parameters train. All128 initial
native logits, sampled decisions and RNG states match the6M parent exactly.
One actual640-onset source batch plus two native queries completes in1.203953
seconds. Its gradient norm is0.457191; every inherited gradient is absent and
all inherited weights/Adam values remain exact after the optimizer step. The
maximum observed conditional-family log-probability difference is9.54e-7,
consistent with float32 normalization; R-only logits remain exact. This proves
the mechanism at fixed inputs, not long-form quality.

Preflight driver SHA:
`55b8c11090ebeaef62b441293a0ebd3ca23ff81e73ef3fb25311628711c8f666`.
The fresh routing owner contains `execution-freeze.json`, pinning the completed
preflight, source8cf31e8, full typed configuration, unchanged shared plan/pool,
training driver, audit driver and focused evaluation driver before training.
Run `uv run --offline --python 3.10 --extra mps python
artifacts/bounded-typed-continuation/routing-recovery-20260920-v1/train.py`.
It makes the declared single250k-onset continuation to6.25M with routing-only
updates, unchanged optimizer settings and the1800-second new-runtime bound.
The final audit will compare every inherited parameter and Adam value with the
pinned6M parent; no control-training arm or new negative pool is added.


### Routing continuation completes; native comparison is frozen

Training completes in299.934137 seconds with333 updates,250k new source onsets
and666 native recovery queries. Final coverage matches the full-parameter run:
4,022,343 unique source onsets,10,176 charts and3,169 groups. The audit verifies
every inherited model tensor and Adam state tensor remains bit-identical to the
6M parent. The route output projection norm is1.168693. First/last32-update
sampled recovery losses are1.514103/0.114654; these are training-pool values,
not evidence of native playability. Maximum sampled RSS955,990,016 bytes,
footprint779,699,808 bytes, swap growth zero. The declared guards all hold.

Final checkpoint SHA:
`dff1727bcb634666992dc4c836ee507c02f31917b1a32f2f1f1d5329eaf5f080`.
Training receipt SHA:
`e30075119f9d18a512ac7ab548c836576a277618d29cb4290836fc5420949b22`.
Audit driver SHA:
`4512c15921ac6cd8ce9311165fdddad9c6f0146833c6976ec9260f38fc604ae0`.
The completed checkpoint, receipt, audit and predeclared generation driver are
now bound by `routing-recovery-20260920-v1/evaluation-freeze.json` before all16
fixed complete-chart generations. The driver independently recounts pinned5M,
6M and full-parameter-recovery rows before comparing the new candidate. Retention
counts use the same09/21 contexts for both seeds; canonical visual judgment and
periodic-cycle inspection remain necessary. No adoption or quality conclusion
is made from frozen-parameter checks alone.

### Completed routing evaluation and handoff recovery

The completed 16-output evaluation takes 328.793489 seconds. Mechanics and
export/reparse pass; no below-10ms event appears. Maximum consecutive H containing
one fixed lane falls from 152 for memory6M to 22 for routing6.25M. Below-40ms
diagnostic events are 69 and 68 respectively, which is not a meaningful measured
improvement. The declared native sampler remains temperature one. Readout SHA:
`ddae474b166e426dcd4c29943e0f3b2e97da95e19f347a5e3589bcd0e20f574d`.

Canonical source/candidate images recorded in Note ID
2026-09-20-r1-longform-handoff show improved historic quadruple and held-lane
collapse contexts. LN retention is mixed: source21/seed17 at 220097–228097ms
has 72 LN heads, 20 H with at least two held lanes, and 34 independent close rows,
close to memory6M's 73/19/36. Source09/seed17 at 97684.5–105684.5ms is mostly TAP,
18/1/1 versus memory6M's 79/35/22; seed23 has brief meaningful LN passages, 30/10/7.
These are agent visual observations and exact action counts, not human labels.
Late source27 is LN-heavy relative to the source. Residual static allocations
and fixed-group exchanges need episode context; Trill presence is not collapse.
Retention recount SHA:
`309e78601bc505a522ebf0aa1cbcf8e3b03f06dc7ef782cf3ffb2b5137149a5f`.
Disposition remains REFINE. No candidate is accepted as playable.

## Experiment Card: r1-routing-fresh-cohort-v1

Revision1, proposed; acceptance none. The active goal independently authorizes
this exploratory evaluation and local commits. It continues the first priority
of long-form structure and diversity; requested-star optimization stays secondary.

**Question.** Does the unchanged routing6.25M candidate sustain organization on
previously unused validation musical groups, and do its failures point toward
release policy or larger-scale planning? This is a generalization screen of one
fixed checkpoint, not a causal ablation or an independent final test set.

**Frozen source and inputs.** Product `8cf31e8d177fab28060ce92be4f5e92f8f8585f3`;
checkpoint SHA `dff1727bcb634666992dc4c836ee507c02f31917b1a32f2f1f1d5329eaf5f080`.
Source-only census SHA
`48ba7f52dd0dd507eece8dc993b941f0d26b6d69766e152f41ec8088e9add062`
and previous28 conditions SHA
`7c757dbe90fcdaecb667af38e6b17b98a192595542d6aa62fcb742ea9e543bc7`
provide selection metadata. Exclude their 48 previous groups plus the28 selected
groups; close exclusions across catalog and original allocation group IDs.
This excludes76 catalog groups and14 represented original allocation groups.
No TEST payload is opened.

Choose two new groups per source band [2,3), [3,4), [4,5), [5,6], with suffix
duration at least180seconds. Seed20260920 samples groups uniformly, then charts
uniformly within a group. In each band select one with at least16 source
independent-LN-start witnesses and one ordinary long chart; the latter may also
contain LNs. Every selected group is distinct under both available identities.
The witness threshold selects evidence; it is not a semantic label. Selected
source SHA prefixes in order are dd580ab2be42, 18bad88a6007, 3d80203596f1,
5d452f8134f8, f2e23b5e0b78, a2f9ae5c5069, d34487d31198, 3352f5ed768a.
Suffix lengths are187.106–302.388seconds. Exact identities, paths, source row
hashes and fixed early/middle/late8-second scopes are in conditions SHA
`3e9a4cd57b576612eacee66025652f38cba11b8a8f4a2a6ad95d5d16a0759a43`.

**Procedure and bounds.** Generate all16 complete suffixes, seeds17/23, CPU1,
native temperature1. No new training, support bans, LN caps or routing edits.
Keep complete original seeds and supplied R/H timing; source suffix actions
are evaluation references only. Bound the run to1800seconds, each generation
to600seconds, process RSS/footprint6GiB, swap growth128MiB, owner output2GiB.
Use the fresh owner
`artifacts/bounded-typed-continuation/routing-fresh-20260920-v1/` in the longform
worktree; no overwrite or scientific resume. Run `uv run --offline --python 3.10
--extra mps python artifacts/bounded-typed-continuation/routing-fresh-20260920-v1/evaluate.py`.
Freeze SHA `19024335a07db1073ff8504abe93430883fc880484d227e3a2ced3d9cf2c27a7`
pins checkpoint, selection, drivers and source before generation.

**Readout and decision.** Require complete legal mechanics and exact export/reparse.
Inspect fixed early/middle/late contexts, the source-selected LN witness, and
worst repeated-lane, periodic-cycle, held-state and type-loss contexts using the
frozen Beatmap Lens Foundation and relevant High human examples. Begin with seed17;
inspect seed23 wherever the locators indicate worse or inconsistent organization.
Counts, LN fractions and star ratings locate and describe outcomes; they cannot
pass playability. One clear sustained degeneration or lost independent LN/TAP
relationship defeats a readiness claim and permits focused failure inspection
instead of an exhaustive review of already-failed candidates. A below10ms event
requires an exact source/condition/native witness and prevents an unchecked pass.
Mechanical, nonfinite, identity or resource failure stops the run. Positive
evidence would justify wider validation, not difficulty certification or adoption.
Raw results and images remain in the artifact owner. Append results here and
publish reusable conclusions in curated product documentation.

### Fresh cohort result: persistent held-state failure

All16 generations finish in184.943946seconds. Mechanics and exact export/reparse
pass; no below10ms diagnostic event appears. Candidate below40ms events total242
versus26 in the source references counted once per seed. Most are in sources06/07;
source06 generates6.835/7.213stars from a5.863star source. Difficulty is descriptive
and remains unresolved. Readout SHA:
`fae14c90363c793ed7895e9d2edd159a862afa878722829eb365ea3c1d1d2350`.

Canonical rendered review of source01/seed17 confirms three persistent holds
confining26 consecutive H to lane1 over154651–163818ms. Holds began at147985,
149318 and153318ms. A denser passage starts near160318ms while these holds persist.
This differs from the human slow-Jack example's deliberate local repeated attacks:
the generated passage has no changing independent hold/release roles and cannot
route attacks elsewhere before releasing. Source-reference organization varies
across columns. Gold presence is not an automatic quality judgment; the persistent
allocation through the density change is the concern.

Native-state recomputation preserves the external schedule and every generated
prefix row. All26 states have15 legal full actions but the sole head mask2.
Release probability is initially0.814, then declines; at161651–163651ms its range
is0.0221–0.0419. The final163818ms action releases lanes0/3. A constant head-mask
score cannot change any of these conditional release choices. This is a concrete
limit of the trained routing intervention, not proof that all long holds are bad.
The exact audit is `routing-fresh-20260920-v1/locked-state-audit.json`.

Also inspected: the four source-selected LN cores for indices00/02/04/06 and the
source06/seed17 short-restart source/candidate context. The candidates retain
short-LN flow, varying lengths and some staggered independent releases;04 is a
clearer rich LN/TAP guard than00/02. Source06 is substantially denser in LN
articulation than its source. These observations are calibrated against existing
High human LN/Stream positive and LN negative examples. They do not establish
complete long-form quality. Fixed phase montages, other-seed and other source
dimensions remain unreviewed after the clear held-state failure. Disposition
REFINE; no model-ready claim.

## Experiment Card: r1-release-native-pool-v1

Revision1, proposed, acceptance none. The active goal authorizes this bounded
TRAIN-only preparation. Baseline is clean8cf31e8 and unchanged routing6.25M SHA
`dff1727bcb634666992dc4c836ee507c02f31917b1a32f2f1f1d5329eaf5f080`.
Before changing the model, determine whether its own TRAIN continuations contain
enough analogous persistent-held states for a focused learned release correction.

Use the existing TRAIN census SHA
`bd77030a9452bb31228bf530c78f0cc567eee4c86d95f5c96734d092fd282905`
and6.25M plan SHA
`37aa23798a4184e3990daf791033a46dbf790bb1acfda29001a5b645555a565a`.
Choose32 distinct groups by SHA sorting with prefix957, suffix duration>=180s
and256–3000 suffix H; seed47 generates each complete native trajectory on CPU1.
At a state with exactly three held lanes, require the same three holds to precede
all previous12 H, whose generated head mask is the sole other lane. Select only
when the corresponding TRAIN source H use at least3 masks and no lane in more
than9of12. The candidate preference is any legal action releasing a blocking
lane. At H all legal heads share the free lane, so the existing complement-mask
API already represents that preference. At R it selects release directly.
Keep at most one query per4 H and32 evenly spaced queries per chart.

This is source-contrasted heuristic negative supervision, not human annotation
or a universal prohibition of triple holds, long LNs or Jacks. Source actions
select TRAIN examples but never become native predictor inputs. Require at least
16 queries across4 groups before considering adapter training. Fewer examples
means insufficient evidence; do not silently relax selection or add VAL examples.
Bounds:1800s, RSS/footprint6GiB, swap growth128MiB, fresh owner2GiB. Stop on
mechanics, identity, nonfinite or resource failure. No overwrite or scientific
resume. `release-recovery-20260920-v1/harvest.py` freezes its digest, exact sources
and settings before generation; its command uses offline Python3.10 with mps extra.

If the pool is adequate, the selected follow-up branch is a zero-initialized,
mirror-equivariant release-mask residual with all inherited weights and Adam
states frozen. A common correction per release mask preserves conditional
head routing and TAP/LN kinds within that mask at a fixed state. It can change
release timing without directly refitting the learned head/type distribution;
trajectory retention still needs evaluation. Full-policy recovery already lost
LN structure. A planner is the alternative if this focused correction fails or
only displaces degeneration. The learning analogue is
[unlikelihood training](https://arxiv.org/abs/1908.04319); conditional family
reweighting is an adaptation, not a novelty or quality claim. Freeze a separate
training procedure and guards only after inspecting the actual TRAIN pool.

### Pool driver correction before evidence collection

The first harvest fails on chart5 after four completed outputs because its final
assertion tests `replay.is_complete`, which refers to a materialized terminal
row. R1 may skip an unused final R with all lanes closed. The driver mistakenly
reintroduced the original pool-v1 assertion already repaired in the earlier
native-recovery harvest-v2. This is a driver error, not a generated mechanics
failure or usable pool. Preserve the partial pool-v1 and failure receipt.

Card revision2 changes only this completion check and fresh output identity:
`harvest-v2.py`, `harvest-v2-freeze.json`, and `pool-v2/`. It requires schedule
completion, all occupancy closed, exact H count, and independent `verify_complete`
over the complete generated physical history. All32 sources, seeds, selection
criteria and limits are unchanged. The four complete prior trajectories can be
compared byte-for-byte; none of the partial run is used for training. This
exploratory correction remains proposed with acceptance none.

### Ordinary TRAIN harvest supplies no release examples

The corrected32-chart seed47 run finishes in286.247803seconds, with all schedules
complete, all holds closed and independent mechanics checks passing. The earlier
four complete trajectories are byte-identical. There are zero selected queries
and zero eligible groups. A separate recount finds that the longest unchanged
three-hold state spans only2 H anywhere in these outputs. Manifest SHA:
`0f37e0d853a7718288a7f06d0c937339c8c8387df29d33460d2fa0597e9f2831`.
This is insufficient-native-failures, not support for release-adapter training.
No model change or training follows from this pool. The fresh VAL witness is
a real conditional-routing limitation but these32 TRAIN generations do not
establish a widespread defect. LN anchors and Jacks can be valid; final semantic
judgment still requires their larger context and cannot come from12-H selection.

The completed fresh-cohort results are curated in docs-only descendant
`c1cd14c81d2058299144f4e41ec574853e5e0280`. This commit occurred while the
already-imported baseline harvest ran; its sole change is explanatory Markdown,
with no executable or data difference from the pinned8cf31e8 source.

## Experiment Card: r1-release-transition-pool-v1

Revision1, proposed, acceptance none. Test the more specific hypothesis that
sparse-to-dense transitions expose persistent allocation states. This is one
targeted TRAIN data probe, not adapter training or a model comparison.

Use the unchanged routing6.25M checkpoint and plan, from clean docs descendant
c1cd14c. Exclude the previous32 TRAIN groups. From existing TRAIN census and pinned
row caches, admit2–6star sources lasting180–600s with256–4000 suffix H. Source-only
H timing must contain a position whose preceding12 intervals span6–30s and following
12 span at most3s, with ratio>=4 and no preceding interval>2.5s. This excludes a
long empty break as the only explanation for sparsity. Rank charts by maximum
ratio, qualifying-position count and SHA; select32 distinct groups. The resulting
296 eligible charts cover176 groups. Selected ratios are8.51–19.08. Exact source
identities and witnesses are frozen in selection SHA
`3ea6dc066521cfab1efac1b8c69155008d6d13c94531f675d29ee4adb1be0666`.
The earlier `selection.json` is an unused source-only draft that admitted long
empty breaks; no generation used it.

Generate each whole chart once with seed17, CPU1, native temperature1. Keep the
same prospective native preference rule: three unchanged holds through12 H,
single remaining head lane, contrasted with at least3 source masks and no source
lane in more than9of12, at most one query per4 H and32 per chart. Use original
external R/H and complete seeds; no suffix actions enter prediction. At least
16 queries across4 groups are still required before considering adapter training.
Do not relax the rule when this probe is negative. Recount stationary holds even
when there are no eligible preferences, to separate rare state visitation from
source-contrast filtering. Any positive pool must be inspected in full contexts
before a training Card is frozen.

Runtime1800s, RSS/footprint6GiB, swap growth128MiB, output2GiB. Stop on mechanics,
nonfinite, digest or resource failure. Fresh owner is
`artifacts/bounded-typed-continuation/release-transition-20260920-v1/`; no overwrite
or scientific resume. `harvest.py` pins itself and selection before execution.
Positive evidence would support the release-recovery branch; a second negative
would leave that branch unsupported and favor broader structural planning/quality
assessment over increasingly narrow hold-duration rules. No training or model
readiness is authorized by the result alone; the user goal provides separate
execution authority for an explicitly recorded subsequent intervention.

### Transition probe result and bounded pool extension

The32 transition-selected TRAIN charts finish in520.027087seconds. There are19
selected states, all from one group, sourcece0f1f7310f9 (Aqua Regia / Crystalline).
Thus the pool is still insufficient for training under the unchanged4-group
criterion. Manifest SHA:
`397f9f76bc15a2c435e3bf7052ac40de5488a952742de8796272510510c1c52c`.
The full20–41s source/candidate Lens montages show two long-held allocations:
three occupied lanes leave a single attack lane, with especially concentrated
re-attacks after31s. Releases restore varied flow near37s. These examples occur
earlier than the source's maximum-ratio timing witness at186253ms; selection is
an enrichment proxy, not proof of transition causality. The observation supports
a real failure family but does not establish broad prevalence.

The six existing gold examples were reprojected from six unchanged canonical
TRAIN/VAL source documents. Their scopes, labels, High confidence and original
comments match the earlier refresh exactly. Current projection SHA:
`647bfd027aa7a780ff09cc25fcc56be25d3f7032b9ce3d5e628aaa042f0111ad`.
No human record, label or confidence changed.

Transition-pool Card revision2 keeps the selection and query rules, checkpoint,
seed17 and all resource limits except wall time unchanged. It expands the fixed
ordered cohort from its first32 to all176 eligible groups, then stops after a
complete chart as soon as the aggregate pool has16 queries across4 groups, or
at3600seconds. Reuse the32 completed trajectories by exact byte copy into a fresh
owner; do not regenerate them. Remaining sources are sampled only if needed,
in the same prospective ratio/count/SHA order. This is bounded collection of
rare native recovery data, not an unbiased incidence estimate. No criterion is
relaxed and no validation example becomes training data. Fresh owner:
`artifacts/bounded-typed-continuation/release-transition-expanded-20260920-v1/`.
The preparation asserts its first32 entries equal the earlier fixed selection.
The driver freezes the full ordered selection and its own digest before execution.

## Release-mask implementation scope

The user goal independently authorizes implementing a bounded candidate while
native data collection proceeds. Use the isolated `Pulsefield-model-release-recovery`
worktree, branch `codex/r1-release-recovery`, at clean baseline
`c1cd14c81d2058299144f4e41ec574853e5e0280`. The collector stays in the unchanged
longform worktree. Training remains conditional on an adequate inspected pool
and a separately frozen procedure; no training is authorized by this scope record.

Add an optional R1 release-mask residual of width512. It gives one common score
shift to actions sharing the four-lane close mask and averages mirrored hand
orders. A zero final layer preserves initial predictions. With the inherited
policy frozen, conditional head routing and TAP/LN choices within a release
mask remain unchanged for a fixed state. Queries with no held lane receive
exactly zero correction. It changes neither support nor temperature and imposes
no duration or repetition rule. Preserve all inherited model tensors, Adam
moments, source-plan prefix, counters and RNG in an explicit release-only fork.
The fork may replace only the pinned recovery-pool identity while scalar recovery
settings remain unchanged. Ordinary resume remains exact and configuration-bound.

Implement typed/Hydra fields, parameter/digest compatibility, fork/trainable scope,
and focused CPU/MPS checks of exact initial predictions, conditional preservation,
no-hold identity, mirror symmetry, native/dense/recovery agreement and inherited
weight/Adam preservation. Use existing joint scoring, raw-state recovery and
pool owners. Record actual parameter count and real-data preflight before training.
This structural restriction addresses the failed full-policy LN-retention result,
but it cannot guarantee unchanged whole trajectories; complete-chart quality
and independent LN/TAP guards remain mandatory.

### Release implementation and adequate pool

Clean implementation: `1e5f5da8d7e1fe54646934e4d512ff8e113bef93` in the release-recovery
worktree. Six release-specific CPU/MPS tests pass. The affected routing, recovery,
fork, native export, seed/memory, smoke-Hydra and package selection passes70 tests
and22 package subtests. `--cfg job` exposes the new module and release-only scope;
this is configuration inspection, separate from the runner-consumption tests.
The outgoing diff and default model/legacy checkpoint identities were checked.
No remote push occurred.

The expanded collector stops exactly at its first adequate complete-chart prefix:
103 distinct TRAIN groups, with32 trajectories reused byte-for-byte and71 newly
generated in701.429509seconds. Four groups supply72 queries (19/4/21/28); every
query is H, with three persistent blocking lanes. Manifest SHA:
`6c2c66482f6fd72686364b43b1e04c0840a8f078461b981a670d21b2195da7de`.
Canonical source/candidate contexts covering every query cluster were inspected:
ce0f1f7310f9 at23298–36765ms, 053e66bd0d06 at146496–153049ms,
8f524a899141 at176024–196739ms, and7d18c54387e3 at151147–173315ms.
The latter two show prolonged stationary holds redirecting successive dense/sparse
episodes into one remaining lane; occasional lane swaps do not restore varied
organization. Sources have varying routing and, in053e66bd0d06, articulated LN
releases. These are machine recovery preferences, not semantic training labels.
The stopping rule makes this an enriched training pool, not a prevalence estimate.

## Experiment Card: r1-release-recovery-v1

Revision1, proposed, acceptance none. The active goal separately authorizes this
one exploratory continuation and its complete native comparison.

**Question and intervention.** Can release-mask-only learning escape native
persistent-hold states while retaining the parent's independent LN/TAP structure?
Use the implemented512-wide residual, freezing all2,917,008 inherited parameters
and their Adam states. Only139,776 added parameters train, for3,056,784 total.
The mechanism preserves conditional complete actions within each release set
and exact no-held predictions at a fixed state. It does not preserve whole
trajectories. The analogue remains unlikelihood training; this is constrained
conditional-family adaptation rather than a new novelty claim.

**Frozen inputs and procedure.** Parent routing6.25M checkpoint SHA
`dff1727bcb634666992dc4c836ee507c02f31917b1a32f2f1f1d5329eaf5f080`, training source8cf31e8,
complete plan37aa23798a41. Intervention source1e5f5da. The exact pool above is
TRAIN-only, with original external timing and raw generated prefixes. Extend
the existing draw stream to6.375M and6.5M milestones; new plan SHA
`14e1b05f7a212e83d97e0558dab8fb84b534716ba0d5b1362d5a6d0c5d7356b1`
preserves every old draw and adds1335 draws totaling250k source onsets. Keep
CPU1, model seed172, batch4/micro2, LR0.0003, decay0.01, clip1 and inherited
warmup/RNG. Source CE plus mean0.25 native complement loss uses two queries per
update, uniform by group then query, recovery seed954. No extra training arm,
temperature change, release ban or LN-duration rule.

**Preflight and bounds.** Before training, validate128 initial native draws and
logits, exact inherited weights and Adam before/after one real source/native
optimizer update, changed release residual, within-release conditional preservation
and exact no-held logits. Preflight freeze SHA
`c3a86a34d1e4908804095b6f001dbae94e34b594f2e925c7ebaf15b381e15624`
pins source, driver, pool and plan. Bound preflight180s, new training1800s and
evaluation1800s, process RSS/footprint6GiB, swap growth128MiB, fresh owner2GiB.
The parent's charged6873.82s retains the14400s lineage training bound. Stop on
mechanics, nonfinite, identity, accounting or resource failure. Preserve failed
outputs; no silent overwrite or scientific resume.

**Evaluation and decision.** Generate all32 complete VAL continuations: the earlier
eight development groups at indices3/9/15/19/21/23/26/27 plus the eight fresh groups,
seeds17/23. Reuse the exact recorded routing6.25M outputs; no baseline rerun. Use
native temperature1, original seeds and supplied R/H. Require exact mechanics and
export/reparse, no below10ms diagnostic regression, improvement of the fresh01/17
held-state episode without a similarly concerning new persistent allocation,
and retention of meaningful independent LN/TAP organization in old21 at
220097–228097ms and fresh04 at52881–60881ms, both seeds. Inspect other worst held,
repetition, cycle and type-loss contexts and the long old27 late passage against
the frozen Foundation and refreshed human gold. Counts locate evidence rather
than certify quality; an isolated Jack or Trill is not automatically collapse.
Generated stars remain descriptive. Failure of LN retention, a displaced collapse
or unresolved concerning context prevents readiness. A positive development result
justifies further validation, not final goal completion or automatic adoption.

Output owner in the release worktree:
`artifacts/bounded-typed-continuation/release-recovery-20260920-v1/`.
The preflight writes the complete typed train configuration. Freeze its digests,
training/audit/evaluation drivers and checkpoint after preflight before execution.

### Release real-data preflight passes

Preflight completes in5.797547seconds. One actual768-onset batch plus two native
queries takes0.440288seconds; gradient norm0.957650. Every inherited parameter
and Adam value remains bit-identical after the update, and only release parameters
receive gradients. The128 initial native logits/draws/RNG match exactly. Maximum
within-release conditional log-probability difference after the real update is
9.54e-7; no-held logits remain exact. Counts are2,917,008 parent,3,056,784 candidate
and139,776 trainable parameters. This verifies the mechanism, not generated quality.
The execution freeze pins typed configuration, plan, preflight and training,
audit and32-chart evaluation drivers before the declared single continuation.
The audit separates new recovery queries from the inherited head-routing query
counter; recovery losses from different pools are not treated as one quality metric.

### Release continuation completed; complete-chart comparison frozen

The declared250k-onset continuation reaches6.5M in148.164666seconds, ending at
update8603. Final checkpoint SHA:
`0a9c87afa43caa9d3647fe3315d79b3a048cc71261571c599c85cf49361dd1ec`.
Coverage reaches4,121,620 unique source onsets,10,257 charts and3,169 groups.
The final audit verifies every inherited model and Adam tensor remains exact,
finite weights, exact exposure/coverage accounting and the durable training log.
Its native-query counts separate the new segment from inherited routing training.
The checkpoint and completed training/audit receipts now bind the predeclared
32-case native generation driver in `evaluation-freeze.json`. Quality remains
unreviewed pending complete outputs and matched Lens inspection.

### Release6.5M passes the scoped long-form and LN-retention screen

All32 complete VAL outputs finish in293.156722seconds on CPU1; independent
mechanics and exact export/reparse pass. Readout SHA:
`d354a939829e0877ccbfa7815bfb312521b91c93692eca8f88c6033651aff0d5`.
Generation sampled peak RSS734,330,880bytes, zero swap growth. Training used334
new updates and668 recovery queries, sampled peak RSS1,850,736,640bytes and zero
swap growth. All inherited model/Adam values remain exact.

Maximum consecutive H under the same three unchanged holds falls26->4; maximum
consecutive H containing a fixed lane falls26->14. Fresh01/seed17 now changes held
lanes and releases through the density rise, then develops multi-column TAP/chord
flow. The old26/seed23 late three-hold run also disappears (18->2 maximum H for
that chart). The worst remaining14-H witness is a bounded2.17s Jack with varying
entry/exit. Exact head-word cycles of31H/5.25s and21H/2.14s are recognizable
bounded Trill figures; they are not treated as collapse just for repeating.

Matched Lens views of old21 (Luster,220097–228097ms) and fresh04 (Gloomy Flash,
52881–60881ms), both seeds, retain independent LN starts/holds/releases, varied
lengths and TAP integration. Independent close-row counts change34/31->37/37 and
36/32->28/47 respectively. Actual views, not those counts alone, establish the
retention observation. Candidate views for all23 main scopes and4 extra scopes
were inspected, including fixed early/middle/late samples, two longest lane/held
witnesses, two cycle contexts and two short-gap clusters. Late passages around
713–721s and882–890s retain organization. Not every second was manually reviewed.

The scoped long-form recovery and LN-retention development gate passes. Overall
goal/readiness remains incomplete; disposition REFINE. Below40ms diagnostic
events increase310->366 over90,678 required onsets, with no below10ms event.
The largest inspected2s cluster has20 release-to-head gaps of37–38ms versus17
baseline diagnostic events. It is rhythmic short-LN rearticulation, not a new
prolonged allocation collapse, but its execution cost and near6star suitability
remain unresolved. Generated stars span2.137–6.717; fresh06 produces6.649/6.717
from a5.863star source. No requested-difficulty calibration or player trial exists.

Semantic review SHA:
`2b7be2eb28d855f558e16cc676476f13ad4839662d91de326352273a22fa0bf6`.
Current High human LN/Stream records and the frozen slow-Jack example were used;
the slow-Jack example has unrecorded confidence, not High. No human record changes
or semantic-label promotion occurred. Use release6.5M as the working candidate
for further verification, retaining routing6.25M and all earlier evidence.

## Experiment Card: r1-release-unseen-confirmation-v1

Revision1, proposed, acceptance none. Confirm the fixed release6.5M candidate on
unused VAL musical groups before another model change. This is an exploratory
generalization check, not training or a final independent TEST evaluation.

Baseline and execution source is clean docs-only descendant
`dfdbc75776cf6e2ded0c2f65f62aa7f8fed94c74`; candidate checkpoint remains
`0a9c87afa43caa9d3647fe3315d79b3a048cc71261571c599c85cf49361dd1ec`.
Use existing source-only census48ba7f52dd0d, excluding the original48 groups,
the28-group screen, the recent8 groups and the six consulted canonical gold
sources' groups. Close exclusions across catalog and original allocation IDs:
88 catalog and18 represented original groups are excluded. No TEST payload.

Seed20260921 selects two distinct groups in each source band2–3/3–4/4–5/5–6,
suffix duration>=180s: one with at least16 independent-LN-start witnesses and
one ordinary long chart, sampled uniformly by group then chart. The witness is
selection metadata, not a semantic label. Exact source and condition identities
are frozen in SHA
`087121241840944676dca12a3b22eb5bca63c5f304753be13ef3e507f24ffeb4`.
Selected SHA prefixes are ef4f8522961c,819b6796594b,2e817c1fb39d,2b93d4068d4c,
4e20e7aad3a8,1a290721dfe8,35b11030b1e2,c8d03a3e7626. Suffixes last189–351s.

Generate all16 complete charts at seeds17/23, CPU1, native temperature1, with
the supplied complete seeds and original R/H. No new training or support changes.
Fixed early/middle/late8s scopes are selected before output. Inspect them and
source-selected independent-LN cores, plus worst held/repeated/cycle and short-gap
contexts, using the frozen Foundation and current gold. Absence of a counter
threshold violation does not establish quality. Require legal complete mechanics
and exact export/reparse; a new below10ms diagnostic must receive an exact witness
and prevents an unchecked pass. Sustained allocation/diversity collapse or loss
of meaningful LN/TAP relations prevents readiness. Difficulty and short-gap burden
remain explicit limitations even if long-form confirmation is positive.

Bounds:1800s total,600s per generation, RSS/footprint6GiB, swap growth128MiB,
fresh owner2GiB. Stop on mechanics, nonfinite, identity or resource failure.
No overwrite, source reselection after viewing outputs, or scientific resume.
Owner: `artifacts/bounded-typed-continuation/release-confirmation-20260920-v1/`
in the release worktree. `evaluation-freeze.json` pins selection, preparation,
driver, checkpoint and source before running `evaluate.py` with offline Python3.10
and the mps dependency extra. A positive result supports continued use of this
working candidate while addressing the remaining full-goal requirements; it does
not change Note lifecycle or establish final2–6star quality.
