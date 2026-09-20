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
ablation branches and broad review overhead. The active goal authorizes direct
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

Revision 1, proposed, acceptance none. One candidate is compared with existing
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
memory width64 and stride64 required onsets. A shared-hand causal GRU encodes
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
