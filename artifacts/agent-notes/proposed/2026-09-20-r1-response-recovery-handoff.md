# Agent Note: Continue from response6.75M structural recovery

Note ID: 2026-09-20-r1-response-recovery-handoff
Status: proposed
Kind: process
Created: 2026-09-20
Updated: 2026-09-20
Product revision: db0a9b8b1a24d481e18bbb40e1a1266bf885756b
Scope: Active playable-model goal; completed response continuation and next independent group check
Related: 2026-09-20-r1-longform-structure; 2026-09-20-r1-release-recovery-handoff

## First goal and authority

The full user goal remains active and incomplete. The first priority is long-form
structure without diversity collapse, preserving independent LN organically mixed
with TAP. Quality across2–6stars remains part of the full goal. This stage is
without audio inputs and retains the original complete seed plus R/H timing.
The user authorizes plan/model changes, bounded local runs and meaningful commits.
No new permission is needed for those actions. No remote push, human-label edits
or Note acceptance is implied. Keep TEST payloads untouched in these experiments.

Response6.75M improves sharp local burden while passing the scoped long-form and
LN-retention development screen. It is the working candidate for further fixed
policy verification, not final playable-quality acceptance. Do not mark the goal
complete from diagnostic counts. Source-star matching is not itself playability;
do not silently make exact source reconstruction the user's requirement.

## Workspace state

Continue in `/Users/l/projects/Pulsefield-model-row-response`, branch
`codex/r1-row-response`, clean at the product revision above. Training/evaluation
implementation is9da67258d76c8359964d81aafa33ed177c2de6fd; the descendant only adds
results documentation. Product owners remain
`src/pulsefield_model/research/bounded_typed_continuation/` and
`docs/research/bounded_typed_continuation.md`.

The release worktree stays clean at dfdbc75776cf6e2ded0c2f65f62aa7f8fed94c74 and
owns the parent checkpoint, confirmation cohort, audits and new TRAIN pool.
The longform worktree owns earlier routing/memory runs and first fresh cohort.
Original `Pulsefield-model` on `witness-style-probe` retains its pre-existing dirty
M0–M4 work and was not edited. Notes live on the registered orphan `agent-notes`
worktree. No training, generation, rendering or audit job remains running.
No subagent work was used. No remote publication occurred.

## Intervention and evidence

A current-row plus next-two-H response bound showed that128 of154 improvable
choices near residual under30ms events could improve while preserving both head
and LN counts. Release-only changes could improve52. The helper's optimistic
future uses one TAP per H and earliest unknown-LN releases, preserving fixed seed
ends. This is not an equal-style future or a demand definition.

The implementation reuses RowConsequence with `model.row_consequence=frontier2`,
adding second-H timing to its existing action/state features. Width32 adds27,648
parameters, total3,084,432. `trainable=consequence` freezes all3,056,784 inherited
parameters. `source_kl_weight=1` anchors source-state probabilities to the frozen
parent branch. Native preferences minimize head-count change, LN-count change,
response cost, then action Hamming distance. Their loss is conditional on actual
and preferred composition families, so probability sent elsewhere does not directly
satisfy it. Every preference is recomputed on exact native state. Adam migration
maps inherited identities by parameter name despite inserted residual parameters.
Inference support and temperature1 remain unchanged; no LN cap or Jack ban exists.

Native TRAIN harvest selected32 distinct source groups, eight in each4–5/5–6star
band crossed with LN fraction below/at-least0.1, uniformly by group then chart.
It completed in172.83s, admitting92 queries from nine groups,81 with unchanged
head/LN counts and two R-only queries. Corresponding source intervals have no
under30ms head. All nine largest-improvement contexts were inspected with source
counterparts. These are machine preferences, not annotations. An opportunity
census run during collection did not alter selection or the admitted pool.

One250k-onset continuation reaches6.75M in235.354203s,328 updates,656 native draws.
Sampled peak RSS1,375,289,344bytes and zero swap growth. Initial128 native logits,
draws/RNG, all inherited weights and named Adam states remain exact. All92 admitted
preferences recompute exactly in production. Coverage is4,219,995 unique source
onsets,10,338 charts and3,169 groups. Source KL averages0.003546 per onset.

Selected local checks cover response/exhaustive future equivalence, mirror/support,
conditional-loss gradients, source KL, interleaved Adam migration, strict resume,
native generation/raw restoration, Hydra projection and packaged configs on CPU
and MPS. The affected66-test owner selection's sole failure was a new synthetic
fixture, fixed and covered by its8-test rerun; other passing tests remain valid.
Generation/packaging passes47 tests and22 subtests. No CUDA or full legacy suite
claim. Documentation-only descendant changes do not invalidate these checks.

## Exact artifacts

Let R be `artifacts/bounded-typed-continuation/row-response-recovery-20260920-v1/`
in the row-response worktree.

- `R/response-6750k/checkpoint.pt`:195f1b0109696302addc1aa62bf896ca399d2a1656c15ba4fe27b4a47c8171ce.
- `R/plan.json`:a613d4b77839e181c6474ba01e9e1a7d275efe35322608ac25c3e24f9853dc00.
  All6.5M parent draws are unchanged;1,312 draws extend to6.625M and6.75M.
- `R/training-result.json`:5a472d4a48f2c3b182d317bb8399dd0bc57baa17c0d796d11585f7a617b3b9a8.
  `training-audit.json` verifies exposure, coverage, frozen tensors and resource accounting.
- `R/evaluation-v1/readout.json`:9d364a3a994f9928c43e8d9d5156742bc180407e46c365c27698487f1c401586.
- `R/state-choice-v1/readout.json`:44bf9f418a44a946c0297a4ecc514334ac30a6b95260a43ba7b3580eeb672daa.
- `R/semantic-review.json`:781e908bbe028a309c4d0d4e519b3f7aa629bfac97d34baab8d1f1de43c304c5.
- `R/comparison-summary.json`:e50b7f39fdfd99e6f003ea60d998d9267599519a743ba48d3a98a935b76f3742.
- `R/retention-readout.json`:ed85d0c879cf8de21d1b5304917d77a0726e87daa9e6843f7fb0646e133d582a.
- `review-v1` has43 candidate contexts; `review-extras-v1` two cycles;
  `review-remaining-v1` the fifth minor short-gap regression;
  `review-longest-v1` four17.50min-chart phases. All50 candidate contexts and
  eleven parent counterparts were viewed. Do not repeat them to rediscover results.

In the release worktree, parent is
`release-recovery-20260920-v1/release-6500k/checkpoint.pt`, SHA
0a9c87afa43caa9d3647fe3315d79b3a048cc71261571c599c85cf49361dd1ec.
New TRAIN pool is `row-response-native-20260920-v1/pool-v1/manifest.json`, SHA
76912db12f9bcf6e0035d296bb1cfc35bb436835cf591f0010ab89727fbb4f19.
The16-output release confirmation is `release-confirmation-20260920-v1/`:
conditions SHA087121241840944676dca12a3b22eb5bca63c5f304753be13ef3e507f24ffeb4,
readout SHAd3abdb77a108cbf40c6b3e6beec867389e5522ee3970b336d335b19c3d2fe41f,
semantic review SHA1c9fcb92b9d51ce43674a04a807076da207ba7e87df993452477344174e222eb.

Saved experiment drivers pin their execution source9da6725 or the prior source.
For new runs on the docs descendant, freeze a new owner/source identity instead
of editing old freezes. Generated assets are local evidence and may be absent in
a fresh clone. Do not broadly scan artifacts.

## Completed quality judgment

All48 outputs from24 VAL groups at seeds17/23 pass exact mechanics and osu! reparse,
covering120,416 required H, including17.50min and14.20min continuations. Generation
takes460.92s on CPU1, sampled peak RSS613MB/footprint912MB, zero swap growth.
Under20/30/40ms head counts fall45->13,160->70,442->178. None below10ms. No paired
chart increases at20 or30ms. Every one of five charts increasing at40ms was viewed;
minimum gaps30–38ms are localized execution costs, not sustained collapse.
At30ms forced heads71->13 and same-cardinality excess77->52; at40ms forced223->43.

The longest unchanged three-hold allocation falls4->2H. A fixed-lane run grows
14->18H but is a bounded2.357s chord-Jack figure with changing groups. The largest
head-mask cycles76H/5.357s and71H/5.468s are bounded traversal and L/R-pair figures
with entry/exit. All phase views retain organization, including long-chart late
scopes882–890s and713–721s. Not every second was visually reviewed.

LN retention is positive from actual relationships, with a visible reduction in
concurrent pressure and more TAP at Gloomy Flash. Independent close rows change
Luster37/37->34/30 and Gloomy Flash28/47->23/16. Four source-selected confirmation
LN cores remain independently organized. The new counter of H with two continuing
holds excludes lanes released on that H and is not the older H-held>=2 counter.

Stars2.142–6.128. Fresh06 source5.863 changes6.649/6.717->6.011/5.758, but earlier27
source5.089 rises5.463->6.128 atseed17; earlier19 source3.143 gives4.235/4.384.
Reduced sharp counts and this aggregate range do not establish2–6star consistency.
Residual local burden, changed chord/hold pressure and stronger repeated figures
remain limits. Disposition REFINE; no player trial or final independent acceptance.

## Gold and next work

Use the installed mania-pattern-judgment skill and frozen Foundation
15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97.json in
`../beatmap-lens/annotation/foundations/` (file-byte SHAb1aea3cbdfe9102e1657d01acfae3f36729467d0a8675b6272ba4f0b17c743ab).
Current gold projection in the longform worktree's
`routing-fresh-20260920-v1/current-gold-feedback-v2.json` has SHA
647bfd027aa7a780ff09cc25fcc56be25d3f7032b9ce3d5e628aaa042f0111ad.
Four High examples were revisited:03f7e300 LN prominent,9f2c08a LN absent,
2af74649 Stream prominent with LN, and b8e95528 Stream absent with changing dense
chords/quads. The latter is not pure L/R Trill. Slow-Jack calibration has unrecorded
confidence. Gold calibrates pattern-presence judgment, not a numeric quality score.
No label, comment, confidence or authority changed.

Next preserve response6.75M and freeze a small unused VAL group confirmation across
source2–3/3–4/4–5/5–6, balancing ordinary and independent-LN long sources. Exclude
original48 groups,28-group screen, routing-fresh8, release-confirmation8 and gold
identities; close exclusions across catalog and original allocation IDs. Reuse
source-only census/preparation logic from the release confirmation and add its
selected groups to exclusions. Freeze new selection and native seeds before runs.
Use fixed phase, LN and whole-chart failure locators and judge with Lens/gold.
No new cohort/Card or architecture has been selected yet. If new evidence demands
another learner, choose one bounded intervention; avoid broad ablation restarts,
hard anti-Jack/short-gap rules or treating zero diagnostic counts as the objective.

Commands: `uv run --offline --python 3.10 --extra mps python ...`; pytest adds
`--group dev`, Lens rendering adds `--extra render`. Actual runs use CPU1.
