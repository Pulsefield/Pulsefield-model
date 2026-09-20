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
