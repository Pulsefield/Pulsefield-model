# Agent Note: Continue from release6.5M long-form recovery

Note ID: 2026-09-20-r1-release-recovery-handoff
Status: proposed
Kind: process
Created: 2026-09-20
Updated: 2026-09-20
Product revision: dfdbc75776cf6e2ded0c2f65f62aa7f8fed94c74
Scope: Active playable-model goal; release6.5M candidate, completed evidence and next verification
Related: 2026-09-20-r1-longform-structure; 2026-09-20-r1-longform-handoff

## Current first goal and authority

The active user goal remains incomplete. First priority is sustained long-form
structure, avoiding diversity collapse and retaining independent LN organically
combined with TAP. Cross-difficulty2–6star stability is secondary to that immediate
priority. This stage is audio-free and receives the original complete seed plus
R/H timing skeleton. The user authorizes plan/architecture changes, local
implementation, bounded runs and meaningful local commits. Do not request those
permissions again. No remote push, human-label modification or Note acceptance
is implied. Use Beatmap Lens Foundation and human gold before readiness claims.

One targeted release correction now passes its scoped long-form/LN-retention
development screen. It is the working candidate for further verification, not
a final playable-quality acceptance. Do not mark the global goal complete.

## Repository and runtime state

Continue in sibling `Pulsefield-model-release-recovery`, branch
`codex/r1-release-recovery`, clean at the product revision above. Implementation
and training source is `1e5f5da8d7e1fe54646934e4d512ff8e113bef93`; the current
descendant only documents results. Product owner is
`docs/research/bounded_typed_continuation.md` and
`src/pulsefield_model/research/bounded_typed_continuation/`.

`Pulsefield-model-longform-memory` stays clean at docs commit
`c1cd14c81d2058299144f4e41ec574853e5e0280`; it owns the parent checkpoints,
fresh-cohort preparation and TRAIN native-pool artifacts. The original
`Pulsefield-model` worktree on `witness-style-probe` retains its pre-existing
dirty M0–M4 work and was not edited. Notes remain in the registered orphan
`agent-notes` worktree. No training, generation or rendering job remains running.

## Candidate and completed work

The inherited routing6.25M model has2,917,008 parameters. A new512-wide,
mirror-equivariant release-mask MLP adds139,776 parameters, for3,056,784 total.
It reweights the16 sets of closing lanes. With all inherited weights frozen,
conditional complete actions within a fixed release mask remain unchanged for
the same input state, including head routing and TAP/LN kinds. No-held queries
receive exactly zero correction. This preserves a conditional distribution,
not future trajectories. No support ban, LN-duration cap or temperature change
is used. `model.release_routing=residual`, `model.release_hidden=512` and
`trainable=release` are fully projected through the packaged configs and runner.
An explicit release fork can replace the pool identity while retaining scalar
recovery/optimizer settings and the exact old source-plan prefix.

The ordinary32-group TRAIN probe yielded zero eligible states; the targeted
sparse-to-dense32-group probe yielded19 from one group. Keeping the same rule,
the fixed ordered cohort was extended until its first adequate prefix,103 groups.
The first32 targeted trajectories were copied exactly;71 more were generated.
Four groups supply72 native H queries (19/4/21/28); none are R-only queries.
Their full query-cluster contexts were inspected. They show persistent three-held
allocations continuing through denser passages; sources have changing routing.
These are machine preferences, not human semantic labels. The collection stopping
rule does not support a prevalence estimate. TEST payloads stayed untouched.

Release-only training adds250k ordinary source onsets and668 native queries in
334 updates, reaching6.5M in148.164666seconds on CPU1. Peak sampled RSS1.85GB,
swap growth zero. Every inherited model and Adam tensor remains bit-identical.
Preflight verifies128 exact initial native decisions and a real optimizer update;
within-release conditional error is at most9.54e-7. Six new CPU/MPS tests plus
70 affected tests and22 package subtests pass. No source change invalidated them;
subsequent product edits are documentation-only.

## Exact evidence for resumption

Let R be `artifacts/bounded-typed-continuation/release-recovery-20260920-v1/`
in the release worktree:

- Candidate `R/release-6500k/checkpoint.pt`, SHA
  `0a9c87afa43caa9d3647fe3315d79b3a048cc71261571c599c85cf49361dd1ec`.
- Training receipt `R/training-result.json`, SHA
  `44dfd0c5147e0e6b793a4c5cf9fd447ff79067594a61fb878b335e897d1c5d4e`.
  `R/training-audit.json` checks inherited tensors, exposure and resource accounting.
- Complete32-output comparison `R/evaluation-v1/readout.json`, SHA
  `d354a939829e0877ccbfa7815bfb312521b91c93692eca8f88c6033651aff0d5`.
- Scoped semantic review `R/semantic-review.json`, SHA
  `2b7be2eb28d855f558e16cc676476f13ad4839662d91de326352273a22fa0bf6`.
  `review-v1/` contains23 main candidate scopes; `review-extras-v1/` contains two
  cycle and two short-gap scopes. All27 candidate contexts were inspected.
- `R/retention-readout.json` contains exact paired LN relations;
  `R/cycle-locators.json` contains full-chart head-word repeat locators.
- New6.5M plan `R/plan.json`, SHA
  `14e1b05f7a212e83d97e0558dab8fb84b534716ba0d5b1362d5a6d0c5d7356b1`.
  `train-config.json`, `execution-freeze.json` and `evaluation-freeze.json` bind
  the actual configuration/drivers. Their old pinned drivers expect source1e5f5da;
  use an explicit matching checkout to reproduce them, or freeze a new clean
  docs-descendant run. Do not edit an old freeze to hide a source mismatch.

In the longform worktree, the parent is
`routing-recovery-20260920-v1/routing-6250k/checkpoint.pt`, SHA
`dff1727bcb634666992dc4c836ee507c02f31917b1a32f2f1f1d5329eaf5f080`.
The new TRAIN pool is
`release-transition-expanded-20260920-v1/pool-v1/manifest.json`, SHA
`6c2c66482f6fd72686364b43b1e04c0840a8f078461b981a670d21b2195da7de`.
`pool-review.json` and `review-pool-v1/` preserve its agent inspection.

Generated assets are local evidence, not source authority. Do not scan artifacts
broadly or rerun completed work to rediscover its results.

## Quality result and limits

On16 VAL groups, seeds17/23, all32 complete outputs pass independent mechanics
and exact export/reparse. They contain90,678 required suffix onsets and include
17.50-minute and14.20-minute continuations. Generation takes293.16seconds on
CPU1, peak sampled RSS734MB, swap growth zero.

- Longest unchanged three-hold allocation:26->4 H. Fresh01/17's26-H confinement
  through the density increase becomes changing holds/releases and multi-column
  TAP/chord flow. Old26/23's18-H allocation also disappears (chart maximum18->2).
- Longest fixed-lane attack run:26->14 H. The remaining14-H/2.17s episode is an
  articulated Jack. The largest exact31-H/5.25s cycle is a bounded left/right-pair
  Trill; neither is automatically collapse.
- Independent LN/TAP retention passes in Luster at220097–228097ms and Gloomy Flash
  at52881–60881ms, both seeds. Independent close counts34/31->37/37 and36/32->28/47
  corroborate the viewed staggered starts/releases, varied durations and TAP flow.
- Inspected phases and late passages around713–721s and882–890s retain changing
  organization. Not every second was manually reviewed.
- Below40ms diagnostic events increase310->366; none are below10ms. The largest
  inspected2s cluster has20 release-to-head gaps of37–38ms (baseline17 diagnostic
  events). It is rhythmic short-LN rearticulation, not a long allocation collapse,
  but its execution cost remains unresolved.
- Generated stars2.137–6.717. The5.863star source produces6.649/6.717. This does
  not establish requested-difficulty calibration or full2–6star quality stability.

The review is unblinded agent development evidence, disposition REFINE. No player
trial, final independent acceptance or new human labels exist. Preserve the
improvement without treating a diagnostic threshold as a universal quality rule.

## Gold and next work

Use installed `mania-pattern-judgment` and the frozen Foundation
`../beatmap-lens/annotation/foundations/15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97.json`.
Six selected TRAIN/VAL canonical source documents were re-read through the Lens
workflow. The existing six gold examples' labels, scopes, High confidence and
original comments remain unchanged. Fresh projection, extracted examples and
audit are under the longform worktree's `routing-fresh-20260920-v1/current-gold-*`.
The slow-Jack Foundation example has unrecorded confidence, not High.

Next, preserve this exact candidate and confirm its long-form result on a small
unused VAL musical-group cohort before another training change. No new selection,
Card or run has been frozen yet. Source-only census metadata remains in sibling
`Pulsefield-model-bounded-typed`,
`difficulty-ln-longform-20260920-v1/{census.json,screen-conditions.json}`. Exclude
the original48 groups, its28-group screen and the new8 groups in longform
`routing-fresh-20260920-v1/conditions.json` (84 total catalog groups before any
additional closure). Validate both catalog and original annotation group IDs;
also exclude consulted gold groups if claiming unseen semantic context. A useful
small continuation is8 new groups, two per2–3/3–4/4–5/5–6 band, duration>=180s,
balancing independent-LN and ordinary long sources, with fixed seeds before generation.

If that confirmation holds, address high-density articulation and2–6star stability
while retaining this candidate's long-form/LN relationships. Do not restart failed
memory-only, seed-only, action-consequence or full-policy recovery ablations without
new evidence. No next architecture is selected. Do not introduce unconditional
anti-Jack rules, LN caps or release penalties from the366 count alone.

Use `uv run --offline --python 3.10 --extra mps python ...`; tests add `--group dev`,
Lens rendering adds `--extra render`. Actual training/generation use CPU1.
Preserve original external timing during native replay, keep TEST untouched and
machine preferences separate from annotations. Note lifecycle remains proposed;
no remote publication has been authorized.
