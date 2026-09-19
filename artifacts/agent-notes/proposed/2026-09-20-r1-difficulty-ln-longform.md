# Agent Note: Does 4M R1 remain playable across difficulty and LN forms?

Note ID: 2026-09-20-r1-difficulty-ln-longform
Status: proposed
Kind: research
Created: 2026-09-20
Updated: 2026-09-20
Product revision: 23a91b04cafaedb57a9f61dc82ce532adcfda805
Scope: Fixed 4M R1 evaluation on additional VAL groups stratified by 2–6-star source difficulty, LN form and continuation length
Related: 2026-09-19-r1-additional-exposure; 2026-09-19-native-inference-and-difficulty-coverage

## Question and current evidence

The paired 2M-to-4M learning comparison improves both model initializations'
likelihood and selected burden failures. All sixteen 4M semantic scopes are
locally plausible, with independent LN control retained in both LN-rich cores.
Additional learning also reduces LN prevalence and some coordination prominence.
The reused development groups and single-seed semantic review do not establish
the owner's long-form, 2–6-star or varied LN/tap requirements.

The next discriminator changes the evaluation population while keeping the
models and decoding fixed. The analogue is the existing complete-suffix native
screen; this is a stratified coverage extension, not a new model or learning
algorithm. It can reject the claim that the current candidate already meets the
broader practical requirements before another training intervention is chosen.

## Experiment Card: r1-difficulty-ln-longform-v1

Card ID: r1-difficulty-ln-longform-v1. Revision: 1. Accepted revision: none.
The user authorizes autonomous bounded implementation, evaluation and meaningful
local commits toward the overall goal. This exploratory Card stays proposed;
execution does not accept the Note or establish final model adoption.

### Fixed candidates and comparison

Execution source is the clean product revision above. Its difference from the
evaluated `50dda55040f51a7afc9a13994b762f953fe3064d` is documentation only.
Primary candidate: R1 initialization 172 at 4M, checkpoint SHA
`ed4ad7dcec30fb2c6f13ee39908bb34c45b06b41799cdf30cfe410d96efdac28`.
Comparator: R1 initialization 171 at 4M, checkpoint SHA
`6f940f0e2c710b7a8c7f365d5703ce0190ad707b449d23c4a87aec78a81651af`.
Both are the completed `seed{171,172}-4000k/checkpoint.pt` artifacts owned by
`r1-exposure-20260919-v1`. Initialization 172 is selected prospectively here
because its development NLL is lower and the inspected LN-rich core retains
prominent coordination. Preserve 171 as a complete fixed comparator; do not
substitute it after observing these outputs.

Keep all model parameters, width128/levels8/expansion4, 511-token learned context,
exact state, R/H/complete-object seed, native temperature-one sampling and
candidate support unchanged. No retraining, type ratio, star target, minimum
LN length, reattack filter or post-generation repair is introduced. Generation
seeds are 17, 19 and 23 for every source and model. TEST remains unread.

On the previous 24 groups, 172/171 group-mean NLL is 2.001587/2.028668. Their
group/seed-mean below-40-ms same-column pair rates per 1000 supplied suffix
onsets are .013006/.019530, maximum runs two. These describe a different source
population and are not absolute targets for the new strata. The primary question
is coverage and structural stability, not winning a cross-population NLL race.

### Source-only selection

Use the same pinned catalog, split and lossless admitted cache. Catalog SHA:
`e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28`.
Split SHA:
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.
Exclude the complete union of the two prior 24-group screen manifests, SHAs
`f0ead07f41111c0413cae9b3f27b3a20c2abbedc2efea1b94f5c2b2707b808eb`
and `4200b2c27b8d21ba617889d75add008d3a90efff4cb194b805b5c0ca0d5ac5dd`.
These are additional groups relative to the bounded model's screens, not a claim
that every earlier Pulsefield study left them untouched.

Eligibility: VAL, at least 128 post-seed onsets and at least 64 seconds after the
seed. Rate original digest-verified 4K sources with the pinned local 20241007
calculator, no mods, rate 1, including the seed. Its source SHA is
`3faaed183a3ef7f62b6ea58f0b0b156cc707c94b760717f78794224a0fb722e9`.
Keep bands [2,3), [3,4), [4,5), [5,6]. For each ascending band, select three
groups uniformly without replacement, then one eligible chart uniformly within
each group, using NumPy RNG seed 1201 and sorted group/source identities.
Preserve global group uniqueness across bands.

After all twelve ordinary selections, select one additional distinct group per
band and descriptor in this order, ranking descending and breaking ties by
source SHA:

1. Independent LN starts: number of suffix-born LNs starting while another LN
   is active and ending at a different time from that existing hold.
2. Short-LN/tap adjacency: number of suffix-born LNs of at most 250 ms with a TAP
   head within 500 ms of their start. These durations locate candidate episodes;
   they do not define a semantic label or good/bad articulation.
3. Long-LN/tap interaction: number of TAP heads strictly inside suffix-born LNs
   lasting at least 2000 ms, counted once per tap regardless of simultaneous
   anchors. This locates sustained interaction, not independent coordination.
4. Longest continuation: elapsed milliseconds after the seed.

The three LN descriptors must have positive support; a missing stratum stops
selection as insufficient coverage without replacement by another criterion.
This yields 28 globally distinct groups, seven per band. Freeze the complete
census, selected identities, source ratings/descriptors and condition manifest
before any model score or generation. Preserve every generated output.

### Measurements and decision rule

Score both models on all 28 complete suffixes with 128-onset chunks and candidate
budget8192; generate all 168 native continuations. Report every source and output
star rating separately. Summarize group-balanced NLL, rapid pairs per1000 supplied
suffix onsets, maximum rapid run, chords, LN durations, independent overlapping
holds, taps during long holds, and early/middle/late development. Reuse existing
diagnostic definitions and state any new descriptive measurement explicitly.

Mechanical guard: every generation completes, verifies exact task support,
exports and reparses. Numerical screens: primary group-mean NLL is no more than
comparator+.05 separately in each source band; its rapid-pair rate is at most
.8 per1000 supplied suffix onsets in every band and no rapid run exceeds four.
These are screening tolerances, not calibrated human limits. Report uncertainty
using2000 paired group-bootstrap resamples per band, RNG1202, without pretending
the seven deliberately heterogeneous groups are a population prevalence sample.
Report generated-band coverage separately; a practical 2–6-star claim requires
at least three distinct groups producing outputs in each band. Crossing the
source band's edge is not itself a quality failure.

Preselect semantic coverage before outputs are observed:

- For the first ordinary source in each band, inspect the primary model under
  all three generation seeds at early/middle/late 16-second cores. Centers are
  at1/6,1/2,5/6 of the post-seed physical duration, clamped so the full16seconds
  remain inside the suffix. Add two-second entry/exit context with entering holds.
- For the longest-continuation selection in each band, inspect those same three
  phases at seed17, plus its complete 10-second-bin development timeline.
- For each of the twelve LN-form selections, seed17, inspect a16-second core
  centered on the source's strongest eight-second bin for its selecting
  descriptor (ties choose earliest), with the same flanks. Inspect the source
  counterpart and exact full LN endpoints as context. Source descriptors are
  locators, not transferred labels.

Use the frozen V2 Foundation and High human gold comparisons at source scope and
rate. Review all local risk locators in the selected cores; additionally inspect
the primary output with the longest rapid run, smallest same-lane head interval,
and largest three-seed star span in each band. Deduplicate overlapping cases.
These output-selected diagnostics remain distinct from representative evidence.
Other pattern dimensions may be unreviewed; inspected uncertainty stays unresolved.

A numerical pass is insufficient. All preselected primary scopes must be
plausible, with no confirmed drift into repetitive allocation or disappearance
of required LN/tap expression at later phases. The collection must demonstrate
at least two distinct source groups for each requested form: sustained anchors
with taps, independently articulated multi-LN control, short/fragmented held
organization, and transitions between held and tap-led passages. No fixed LN
percentage or source copying is required. Insufficient generated-band coverage,
missing expressive modes, failed numerical guards or unresolved burden prevents
an overall quality claim and directs a focused next probe. A scoped positive
result supports a practical candidate on the observed conditions, not universal
playability or human preference.

### Reproduction, bounds and failure handling

Fresh artifact owner:
`artifacts/bounded-typed-continuation/difficulty-ln-longform-20260920-v1/`.
Run its frozen `prepare_conditions.py`, verify its manifest and driver digests,
then run `evaluate.py` using `uv run --offline --python 3.10 --extra mps python`.
Freeze the evaluation driver after source-only preparation and before launch.
No scientific field changes during that engineering handoff. A required field
change increments this Card before dependent execution.

Environment: Apple M5/24GiB, Python3.10.20, Torch2.11.0, CPU1. Selection has a
five-minute bound and64MiB output allowance. Evaluation has a3600-second bound,
6GiB process RSS/physical footprint,2GiB minimum available memory,128MiB maximum
swap growth,3GiB output and1GiB disk reserve. Individual generation may take at
most180seconds. Runtime and resource guards apply throughout. No network data
or additional training is needed.

Fresh directories only. Stop on changed hashes, invalid source allocation,
nonfinite likelihood, mechanics/export failure, resource limit or elapsed bound.
Keep the last durable snapshots and partial journals; a stopped process is not
automatically retried or its output replaced. Semantics and any residual failure
remain in this owner. All previous checkpoints, outputs and judgments remain
unchanged. Source-only selection and new diagnostics do not accept a Note or
complete the overall goal.

## Source-only selection and execution handoff

The frozen selector completes in57.051seconds, within its five-minute bound.
It finds1032 eligible 2–6-star charts in368 remaining VAL groups and selects
exactly28 distinct groups, seven per band, disjoint from all48 prior groups.
Every LN stratum has positive source support. No checkpoint was scored and no
generation was performed during selection.

The longest remaining selections span270.775,435.000,595.102 and852.187seconds
after the seed in ascending source bands. The5–6-star long-LN/tap selection is
even longer,1050.064seconds; it is already excluded when selecting the subsequent
longest-remaining stratum. Thus complete generation covers nearly17.5minutes
without changing the source-only selection rule. Numerical source ratings and
these physical durations do not establish output difficulty or long-form quality.

All artifacts are in the owner above:

| Artifact | SHA-256 |
| --- | --- |
| `prepare_conditions.py` | `29cfbf839a4495090afec2f2cd0ac0fb4e983e1e8fd3a1c472ea2e73bd8589dc` |
| `screen-conditions.json` | `7c757dbe90fcdaecb667af38e6b17b98a192595542d6aa62fcb742ea9e543bc7` |
| `evaluate.py` | `4867bbda0c1c37f96762d671202f8edc787c2486bed4938d7cbb34fb21a4695c` |
| `evaluation-plan.json` | `14575b73295f1fba371c014d0c7f939239802a4448d07f85e9bf72805a0837e1` |
| `preflight.json` | `eb72d49c8d0c2324b2ab40187e705d0c1f684989eceb0c5105b18da03873daed` |

Source-only preparation verifies raw source, admitted cache and portable condition
identities. An independent preflight checks clean execution source, group/band
counts, exclusions and phase bounds. Both checkpoint hashes match their completed
parents. A synthetic action sequence verifies independent-start counts,
short-LN/tap adjacency, unique taps inside long holds, integrated occupied-lane
time and exact pre-row occupancy in the new descriptive locator. These are
instrumentation checks, not model-quality evidence. The initially written
pre-row occupancy locator was corrected before evaluation-plan freezing to
capture all lanes before any same-row action; no model output was read.

The evaluation runner uses the packaged generator and existing suffix likelihood
without model changes. It first evaluates172, then171, retaining all outputs,
per-chart diagnostics, full10-second timelines and continuous resource records.
The readout is published only on complete success; a failure records its stage
and leaves partial artifacts. Primary/comparator identities, the28-group cohort,
the168 outputs and all Card thresholds are unchanged. The next action is the
exact frozen evaluation command; semantic results remain pending.

## Execution and prospective review preparation

The frozen evaluation was launched from the clean product source above using
the declared command. Execution session54323 initially owned uv PID46922 and
Python PID46928. At132.585seconds the primary172 model had completed24 outputs
through ordinary source index7. Those processes were observed alive; this is a
progress observation, not final completion or a maximum resource measurement.
`evaluation.log`, per-output durable results and `evaluation-v1/resources.jsonl`
own the live evidence. Verify the existing process/session before deciding that
an observation timeout means termination; do not launch a second evaluator.

The behavior-neutral `render_preselected.py` has SHA
`64c769168c615eec3112ed760281bdbb60140b61726d3cd3b5104b0ab1572c1e`.
It renders only the Card's first ordinary sources, longest-remaining sources
and twelve LN-form sources, using completed primary output receipts. Ordinary
sources use all three seeds and phases; longest sources use seed17 and phases;
LN-form sources use seed17 at the frozen source-selected core and include a
source counterpart. It verifies the frozen condition, canonical renderer,
Foundation and chart-reader hashes. Full original pages, exact action packets
and derived reading montages retain source times and complete LN endpoints.
Each new case owns a fresh directory; rerendering over an existing case fails.

Rendering these preselected completed outputs may overlap the unchanged
evaluation. It cannot change models, random draws, selected scopes or the later
numerical decision. Output-selected risk scopes remain pending until the complete
readout makes their deterministic extrema available. New cases start unreviewed;
rendering itself is not a semantic judgment.

## Partial semantic evidence and rendering correction

Fourteen primary judgments are saved in `judgments-v1`: all nine phase/seed
combinations for ordinary source0, the three seed17 phases for ordinary source3,
and the source-selected LN cores12 and20. This accounts for112 generated and16
source pages actually inspected. Other rendered cases remain unreviewed.
`review-progress-01.json` freezes these individual judgment digests.

Source0's three outputs rate2.213074,2.493926 and2.522272 stars. All nine inspected
cores remain locally plausible, with distributed moving taps, changing accents
and isolated/sequential short-held articulation. No independent multi-LN control
is claimed in those cores. LN prevalence varies across seeds/phases, but the
inspected tap organization does not degenerate into confined-lane repetition.
Source3 seed17 rates3.047723 stars; its three phases also retain plausible moving
tap organization and bounded held punctuation. These are two sources, not a
cross-band population conclusion.

The2.622067-star generated LN core12 is substantially simpler than its source.
Repeated staggered-start/shared-release pairs give weak supporting coordination
at medium confidence; they do not reproduce the source's sustained independent
release organization. In contrast, the4.249286-star generated core20 has clear
prominent independent LN control: a column1 anchor56221-57589 accompanies
changing0/2/3 holds with distinct endpoints, then taps2 return before the anchor
closes. Later a column3 anchor69246-70273 supports a tap-led contrast before
layered holds resume. This preserves short holds, sustained roles and tap/LN
interaction rather than treating LN count as the semantic target. Both cores
are locally plausible; the first remains a weaker-form judgment, not proof of
equivalent source style.

Rendering source15's late scope exposed a numeric pagination edge: a20-second
floating interval subtracts to slightly more than20000, but its reconstructed
ninth-page start equals its end. The unchanged canonical renderer then divides
by a zero page span. Evaluation and model outputs are unaffected. The caller
now omits only pages whose reconstructed start is greater than or equal to the
requested end; it changes no times, positive-width page, note, scope or glyph.
The original driver is retained as `render_preselected_before_empty_page_fix.py`.
The corrected driver SHA is
`7ed6390986a6c79a2ef26c3a11d8d6ea4e8c074d09ecabebd9682b30d24a16fb`.
The partial failed directory is preserved, and the same late scope renders
successfully in the fresh `source15-seed17-late-recovery1` directory. No judgment
had been made on that scope and no scientific Card field changed.

The evaluator continues unchanged. At844.899seconds, primary172 had completed
81 outputs, including all three1050.064-second continuations. The comparator
and final quantitative readout remain pending. Several5–6-star source conditions
produce4-star outputs; source-band coverage must therefore remain separate from
the required generated-band coverage. Do not infer a numerical or overall pass
from the current partial semantic positives.

## Complete primary generation and declared risk follow-up

All84 primary outputs completed at1014.150seconds; the fixed171 comparator is
running serially. Primary generated-band coverage is13/13/9/2 distinct groups
in ascending2–6-star bands, with29/28/21/3 outputs respectively. Groups may
appear in more than one generated band. The required three groups in5–6 stars
are missing, so this collection cannot establish the full target coverage even
if its other guards pass. It is not valid to substitute source ratings for this
negative observation or to filter/resample outputs until the requirement passes.

The declared shortest-pair locators are79ms (source0,seed19),73ms
(source3,seed19),23ms (source21,seed17) and37ms (source9,seed19) in ascending
source bands. The latter two have three other lanes occupied immediately before
the later attack. This replicates the kind of burden concern that motivated the
earlier investigation, while leaving the actual local organization to inspection.
No whole-chart judgment is inferred from these scalars.

`inspect_critical_risks.py`, SHA
`6ad65240af6a8abb8c4090bcddd67aff397c3a8889f6d218f779664abe6f75e5`,
implements the declared per-band minimum-pair inspection and an exploratory
read-only probability diagnosis. It freezes selection across all84 completed
primary receipts (ties: source SHA, then generation seed), reconstructs the
eight preceding and two following candidate decisions from each generated
history, and checks each chosen log probability against the original journal.
The existing native probability-probe functions are reused at their pinned SHA.
Source-history values are descriptive controls, not an isolated causal change.

The diagnostic uses the same fixed172 checkpoint, draws no new output, applies
no masks/penalties and changes no model source. It renders an8-second risk core
with2-second flanks for both generated and source charts, preserving complete
endpoints. CPU1,240seconds,128MiB diagnostic output and existing resource bounds
apply. Its fresh owner is `critical-risks-v1`; the rendering receipts remain
under unique `inspection-v1/risk-min-*` identities. This additional diagnostic
does not alter the Card's numerical or semantic decision criteria.
