# Agent Note: Can persistent original-seed conditioning stabilize R1 generation?

Note ID: 2026-09-20-r1-persistent-seed
Status: proposed
Kind: research
Created: 2026-09-20
Updated: 2026-09-20
Product revision: cefe3deb97c6e7510c3bbde6f7d679648ee5b06c
Scope: R1 persistent seed representation, matched continuation and native development evaluation
Related: 2026-09-20-r1-prefix-recovery; 2026-09-20-r1-action-consequence

## Question, alternatives and evidence

The current 511-token encoder eventually loses the supplied original seed from
learned history. Whole-prefix replacement causally changes subsequent type
balance, but also changes occupancy, clocks and counts. Its 66.82% early error
reduction on eight selected groups does not establish which of those changes
matters, nor prove that the original seed specifies later structure.

Persistent permitted information is one testable branch. Optimization averaging
or annealing could instead reduce a drifting local prior; explicit desired
difficulty and LN structure could supply information absent from the seed.
Larger history/capacity could retain motifs and evolving form. These remain live
alternatives and are not bundled into this intervention. In particular, a sparse
intro can be a poor guide to a later LN body. Source type fractions are a
development diagnostic, not the only valid continuation of a condition.

The closest architectural analogue is global conditioning in
[WaveNet, section 2.5](https://arxiv.org/html/1609.03499v2): a persistent vector
conditions each autoregressive decision. Here the vector is a shared encoder's
representation of permitted complete seed objects, not a speaker identity or
audio. Conditioning is added at readout, not every convolutional layer.
[MusicVAE](https://arxiv.org/abs/1803.05428) is a contrasting hierarchical latent
approach to long-term music structure; this test has no latent conductor or
future-sequence encoder. It is an adaptation of global conditioning, not a new
generative objective or a claimed new architecture.

## Experiment Card: r1-persistent-seed-v1

Card ID: r1-persistent-seed-v1. Revision: 1. Accepted revision: none.
The active user goal separately authorizes bounded implementation, execution and
local commits. This Note stays proposed; execution is exploratory and cannot
produce an accepted or SUPPORTED adoption decision.

### Fixed comparison

Baseline source is the clean product revision above, whose runtime is the
behavior-equivalent support-batched descendant of the consequence comparison.
Use its none4.5M checkpoint, SHA
`2c452a954b57f0b412a0806dead0adffe11b227a22955f4fb9920d0fdf03d288`, trained at
`4a98c0230549baa19ad860c3a13caf301e0f1f71`. Parent plan SHA is
`767138e58c5288593be284b29a9333544e048468efb8d3f3aae2e42ece4c0fc6`.
It pins 11,563 TRAIN charts and 3,169 groups. Extend every old draw unchanged to
milestones 4.625M and 5M H exposures; sampler seed 471, training seed 172, effective
batch 4/microbatch 2, AdamW .0003/.01, clip 1, native temperature 1 stay unchanged.
Each of three branches adds 500k H with identical draws and inherited optimizer
and RNG state. No annotation, validation or generated judgment becomes a label.

The causal intervention makes the original seed persistently available. Add
R1-only `seed_context=none|zero|observed`, independent of the default-disabled
consequence residual. Re-encode the original complete seed with the existing
shared finite temporal encoder under current weights; masked-mean pool all seed
token outputs into two canonical hand vectors. Feed a shared
`Linear(2h,h), GELU, Linear(h,h,bias=False)` residual on current hand readout
concatenated with its seed vector. Zero-initialize the final projection.
`zero` uses the same residual and parameters but a zero seed vector; `none`
continues the parent unchanged. The two new branches add 49,280 parameters at
h=128; equal size does not imply equal effective input capacity.

There are no new external inputs, future suffix actions/endpoints, gold-derived
labels, objective terms, difficulty targets, sampler restrictions or penalties.
During generation the fixed seed representation is computed once. Durable
recovery stores original raw seed facts and rebuilds learned values, separately
from the unchanged bounded rolling history. Required edits cover the owning
model/data/runtime/fork/config code, its tests and curated contract. No unrelated
cleanup or scientific change belongs in this comparison.

### Measurement and decision

Reuse the 28 development groups in condition manifest SHA
`7c757dbe90fcdaecb667af38e6b17b98a192595542d6aa62fcb742ea9e543bc7`; TEST remains
unread. Generate every full chart with seeds 17/19/23 and score complete source
suffix likelihood. Primary diagnostic: partition each source's post-seed H into
four consecutive equal-count bins using floor(N*b/4), compute absolute generated
minus source LN-head-fraction error in each bin, average four bins and three
seeds within a group, then 28 groups equally. Parent none4.5M phase MAE is
.2552939643, with group-bootstrap90% interval [.1938338937,.3231275318]. Its four
bin errors are [.168861374,.274578132,.284595219,.293141133]. Frozen parent
reference SHA: `1b391217b47d749303c0b20b90fb404cfe24556a5d464962c9cf28f4e39d333e`.

Require observed to improve by at least20% against each concurrently continued
control and each paired whole-group90% interval upper bound to be below zero
(10,000 bootstrap draws, seed1223). Do not select among intermediate checkpoints
using this metric. Report individual bins, source-star bands and all seeds.
Require macro source-suffix NLL no more than .05 worse than either control;
short-interval union rate per1,000 H no higher than max(1.15*control,control+.5)
for either control. Mechanics, finite values and exact export/reparse must pass
every output. Record actual generated star bands/range and every new <10ms
release/head or head/head witness; no numeric proportion gate implies quality.

After numeric evaluation, inspect fixed seed17 scopes: the same12 LN cores,
12 ordinary early/middle/late scopes and4 long-chart late scopes used in the
consequence comparison. Include the worst new action-risk context per branch.
Use Beatmap Lens V2 and confirmed gold, full endpoints and time-proportional
renderings. Mask variants and seal judgments before revealing assignments.
Require no loss of independently organized LN presence in any concurrently
prominent control LN core and report long-scope changes separately. A new tight
restart concern or failed numeric guard prevents a quality-improvement claim.
These are agent judgments, not new human labels or independent gold.

### Execution, bounds and interpretations

Implement in a new clean descendant worktree and commit before corpus execution.
Verify zero-function/Adam preservation, no suffix leakage, mirror equivariance,
source crop/dense and native/recovery agreement, old default snapshot behavior,
actual Hydra consumption and CPU/MPS gradients. Old seed information may persist;
unrelated nonseed history older than the finite field may not.

Use packaged training/generation boundaries via artifact orchestration with
`uv run --offline --python 3.10 --extra mps python` on this M5, Torch2.11, CPU1.
Pin final clean source, extended-plan, config and driver digests before training.
Fresh artifact owner: `artifacts/bounded-typed-continuation/persistent-seed-20260920-v1/`.
Parent reference is already immutable there. Use distinct mode/milestone output
directories; never overwrite. Resume only from a verified durable boundary with
all measured and discarded time charged, or stop and record the failure.

Bounds: 90 minutes total added training, 90 minutes evaluation, 6GiB RSS/footprint,
128MiB swap growth, 6GiB outputs, no external compute or new dataset downloads.
Stop on mechanical/parity/nonfinite/resource failure. The4.625M boundary is an
engineering checkpoint, not a quality-selection opportunity. If the complete
comparison exceeds a bound, retain all evidence and return REFINE without
silently enlarging it. Read-only primary literature is permitted.

A positive result supports further testing of persistent permitted conditioning
on a new development cohort, not final adoption. Failure against zero weakens
the information-specific claim; worse LN-rich passages suggest seed mismatch.
Failure after500k exposure cannot rule out training from scratch or other global
conditions. Shared-encoder gradients, intro representativeness, source-proportion
nonuniqueness and reused development groups remain explicit confounders. Neither
success nor failure settles requested2–6-star control or the full Pulsefield task.

## Next lifecycle condition

Keep the Note proposed. Record implementation, execution and evaluation evidence
as append-only results; any changed protected field increments the Card revision.
Human acceptance of an exact revision remains separate from autonomous execution.
