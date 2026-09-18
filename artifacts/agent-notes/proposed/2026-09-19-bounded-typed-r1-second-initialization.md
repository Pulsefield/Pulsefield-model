# Agent Note: Does R1 retain its generation behavior under a second initialization?

Note ID: 2026-09-19-bounded-typed-r1-second-initialization
Status: proposed
Kind: research
Created: 2026-09-19
Updated: 2026-09-19
Product revision: 15d27db0b0723d7f606b1429c901d26c1f61e5ac
Scope: R1 initialization sensitivity on the fixed corpus plan and development groups
Related: 2026-09-18-bounded-typed-time-three-arm-comparison

## Question and evidence

R1 is a competitive, cheaper alternative to O1: it receives the same typed timing
conditions and complete seed objects, but chooses new LN releases row by row.
At 2M source-onset exposures, initialization 171 has complete-suffix pooled NLL
1.915833909, group macro NLL 1.987622, 27.651% generated LN heads, 28 same-lane
attack pairs below 40 ms, and a maximum such attack run of two. All 72 generations
pass mechanics and export/reparse. These are 24 development groups with three
generation seeds; they are not independent confirmation groups.

The fixed LN-rich 16-second core has independently coordinated LN activity,
present/prominent at High confidence using the frozen Foundation and human
comparisons. The fixed dense core has no LN coordination. These two judgments do
not establish whole-output playability. Another initialization can expose a lucky
checkpoint or training trajectory before spending more effort on this candidate.

## Experiment Card: bounded-typed-r1-initialization-172-v1

### Identity and authority

- Revision: 1. Owning Note: this Note ID. Accepted revision: none.
- Exploratory implementation and execution use the standing user instruction to
  optimize and experiment autonomously toward playable continuation. The Note
  remains proposed; execution is not lifecycle acceptance or adoption.

### Hypothesis and decision

The same R1 learning setup can reproduce useful LN organization and low rapid
attack burden after a fresh model initialization. If it does, retain R1 for
larger-context quality review and unused-group confirmation. If it does not,
treat initialization/checkpoint sensitivity as a shared training problem instead
of declaring rowwise factorization stable from one run.

The closest comparison is the existing matched three-arm experiment, not a new
representation claim. Alternative explanations include incomplete training,
uncontrolled intent/mode selection and generated-history feedback. A second
initialization isolates initialization sensitivity under fixed source sampling;
it cannot uniquely distinguish those mechanisms or estimate a seed population.

### Fixed comparison

- Baseline source: `1693d62ffaca04b2a6127d8e3a72d1988adf441f`, clean.
  Execution source: the clean Product revision above; its availability changes
  are inactive for R1 with `endpoint_availability=none`.
- Baseline checkpoint:
  `artifacts/bounded-typed-continuation/corpus-20260918-v1/r1-cpu-2000k/checkpoint.pt`,
  SHA `302fae523cf1fe36afff463fe28f20739a2eb40cab5064d0941a588d8f70c564`.
- Immutable plan: `corpus-20260918-v1/plan.json`, SHA
  `a6e727d0bbfa414e7d20c742a18d84e8786f155b91629d285e7abf04348cd82f`;
  sampling seed 471, 11563 eligible TRAIN charts in 3169 groups, fixed draws,
  128/256 onset targets and 250k/1M/2M exposure milestones.
- Screen: `corpus-20260918-v1/screen-conditions.json`, SHA
  `f0ead07f41111c0413cae9b3f27b3a20c2abbedc2efea1b94f5c2b2707b808eb`;
  12 ordinary and 12 preselected stress VAL groups, generation seeds 17/19/23.
- Baseline summary SHA
  `262979f1deb1069f0ff9b401f5c2d83b55b79f0de4051b770b57d6ac64e828f3`.
- Intervention: model initialization 172 instead of 171, trained from scratch.
  Preserve model, AdamW, LR .0003, warmup 32768 onsets, batch4/microbatch2,
  candidate budget8192, CPU one thread and all source draws. No forked weights.
- Behavior-neutral instrumentation: artifact drivers, provenance and paired
  summary. No product edits or altered decoding rules are planned.

### Evidence and decision rule

At 2M, require all mechanical/export/reparse checks and selected raw-state
recoveries to pass. For a candidate that merits further quality evaluation:

1. Primary burden gate: group-balanced mean of each generation's below-40-ms
   same-lane adjacent attack pairs per1000 suffix heads is at most0.8, with no
   generated run longer than four attacks. Report paired group-bootstrap
   differences to seed171, 10000 resamples, seed671; this engineering threshold
   is not a universal playability limit or a significance claim.
2. Complete-suffix group macro NLL is no more than baseline +0.05. Count every
   candidate decision, including releases, normalized by common source onsets.
3. Collection LN share remains between half and twice the baseline27.651%.
   This is a gross mode-loss guard, not a per-chart LN quota or quality label.
4. The same fixed LN-rich core, source0447fb187bc3 seed17, still contains
   independent LN coordination under the Foundation; merely overlapping holds
   is insufficient. Inspect the matched dense core too, without requiring LN.

Report all gates separately. Passing them advances review, not adoption. Failure
or a burden improvement accompanied by lost organization supports REFINE.
Uncertain semantic judgments remain unresolved. Evaluate all three exposure
milestones to describe the trajectory, without replacing the preselected 2M
primary checkpoint with a favorable intermediate result.

Before a broad quality conclusion, inspect preselected ordinary indices0/3/6/9
at their 16-second cores and indices0/6 at their 64-second contexts, seed17,
paired against seed171 with masked method identities and tie/both-inadequate
allowed. These qualitative comparisons remain descriptive. Independent groups
and broader human judgment are still needed for confirmation.

### Procedure and bounds

Fresh root: `artifacts/bounded-typed-continuation/r1-initialization-20260919-v1/`.
Pin each driver digest before running. First execute `preflight.py`: verify the
source, plan, checkpoint and screen hashes, load the old R1 model on current
code, and reproduce complete native rows and suffix NLL sums within1e-5 for
indices0/12/16 at generation seed17. Audit scientific configuration equivalence
except model_seed and the inactive default availability field. No outputs are
used to select model_seed172.

Then run `uv run --offline --python 3.10 --extra mps --group dev python
artifacts/bounded-typed-continuation/r1-initialization-20260919-v1/train.py`.
The driver calls the packaged Hydra composition and `run_training`, from scratch
to2M, retaining milestone checkpoints. Python3.10.20, Torch2.11.0, NumPy1.26.4,
Apple M5/24GiB, CPU one thread. Four hours cumulative training maximum, 6GiB
footprint/RSS, minimum2GiB available memory, at most128MiB swap growth, training
output512MiB, evaluation2GiB and one hour per screen. No external network work.
Stop on any guard, source/configuration mismatch, nonfinite loss or mismatch in
preflight. Never overwrite runs; exact source/configuration resume may use a
fresh directory if needed, charging all parent compute. A failed preflight must
be resolved and recorded before starting training.

One CPU interpolation evaluation may overlap this one-thread training. Record
the overlap; wall-clock throughput is then confounded and is not a causal speed
comparison with the earlier solo run. Model/data comparisons retain the same
exposure ledger. Verify paired coverage and factor counts at completion.

## Next lifecycle condition

Append reproducible measurements and scoped Foundation judgments. The Note stays
proposed until explicit human acceptance; two seeds on reused development groups
cannot complete the overall playable-generation goal.

## Pinned preflight driver

`r1-initialization-20260919-v1/preflight.py` SHA
`434688b90b8b2f60a8b6187a68fda76d367cef5f52647309440f97e37389d699`.
Read-only preparation caught and corrected two driver-inspection errors before
this driver was written: an attempted JSON dump of tensor buffers and the Hydra
enum spelling `r1` instead of `R1`. Neither created a run or modified a checkpoint.

## Preflight result and pinned training driver

Preflight completed in15.451s with exactly matching head/endpoint NLL sums and
all native rows at indices0/12/16 (620/1696/4403 physical rows). Mechanical checks
pass, and the scientific configuration matches after the declared initialization
change and inactive default field. `preflight.json` SHA
`5803d32717be8e3b0dbc45f4b50555e3978a13ac41756c2eb7d8d6023405c164`.
`train.py` SHA
`ba5709fc7639ff27e2b76d2ac5afa0eeaa1701750bef68ee99b6978e0bff6cf7`.
The driver checks this preflight identity before training and compares all source
exposure/coverage ledgers against the original three training segments afterward.

## Pinned analysis and semantic preparation

`readout.py` SHA
`3b758eac0a34915926e325131562f2d99d88984ace22c9405f28ffae5ae04f8e`
compares all three declared milestones with initialization171. Its shared
row-recount/statistics helper is the interpolation `readout.py`, SHA
`e1becc82254144700ea22b3c01adfb3abe4472cd1b0dd72345c46905c574a81e`.
`render_primary.py` SHA
`7961864a2a47158d869474202fcc6e884195acab32cb842aafb8b79971ce8b2f`
prepares the fixed LN-rich and dense2M cores.

`render_blind_ordinary.py` SHA
`33790710dc8e215d0d8fb88e786fa595d868abda8bdae2daca819b22eb9ef52b`
prepares the declared ordinary indices0/3/6/9 locally and0/6 at64s, with complete
entry/exit contexts and paginated actions. Each pair's A/B initialization mapping
is randomly shuffled and saved separately. This randomization only masks display
identity; it does not select sources, checkpoints, generation seeds or outcomes.
The mapping is to remain unread until scoped machine judgments are recorded.
Masked machine review is not a human blind-preference experiment.

The frozen Foundation byte digest was reverified as
`b1aea3cbdfe9102e1657d01acfae3f36729467d0a8675b6272ba4f0b17c743ab`.
Canonical human-example queries for Jack/Trill initially returned no matches with
a key-count facet because the cached raw packet lacks materialized sourceFacts.
Removing that facet recovered the known High comparisons. A missing query result
is not evidence that a semantic pattern or human example is absent. Source4K
identity is still checked by parsing before using those comparisons.

## Training result and native-screen pin

Initialization172 completes2M exposures and2645 updates in1616.286s of charged
training time. Source ledgers match initialization171 exactly across every update,
including physical/prefix/padded rows and context spans. Coverage matches:
1697938 distinct onsets,6315 charts,3078 groups. Peak physical footprint921831032
bytes and zero swap growth. The declared interpolation screen overlapped; brief
readout/rendering work also ran alongside training. These timings do not establish
a causal speed change. The product worktree stayed clean at its pinned revision.

`training-comparison.json` SHA
`d721a0d1aa68b1e4f0b1cdd230d484913d2c8921b7238c9256685bbf156147ce`.
Milestone checkpoint SHAs for250k/1M/2M:
`7174793e0b6d4bddd3408b57460dc2b7646cc93dbd4aaeab134dff069c31f7c0`,
`09c9029efae9cf4618e53aeb4b0acd94e10f95df9af7e66a600a3f931507d191`,
`9be0adc00cea85058b59f0a2446884d43fc40cd412188ee123e866df6369d83a`.
`development_screen.py` SHA
`e29439dd9ed2aea55d568e46c4c9ea3cbada98c5287cc0057d3647bcb87c9eba`
is pinned before execution. It verifies the training report and milestone
digests, then scores and natively samples the declared24 conditions at each
milestone. Its task is R1, initialization172, availability none; source sampling,
suffix scoring, support, generation seeds and export/reparse match the baseline.

### Additional human calibration

`human-motion-calibration-v1/manifest.json` SHA
`79d1a53380e3b027dde0ef12ba27da21e8c3531839d7d687f26856f97f8998b9`;
`review.json` SHA
`10cfb254f801a215b14af4e01e2fc9069fd083a4de779adfd5c20be420afa8d2`.
All11 context images were inspected at1x. Source4K identity, exact bytes, the
canonical human documents and cached public example identities were verified.
No human records were changed and no human rationale was invented.

- Source871955…105169–109343: High Jack absent and Trill absent, IDs
  `human-cd6e12362ea63698869fd567` and `human-48f59fc3f96b602b5f3cc79e`.
  Changing leads, single/pair attacks and short-LN articulation contrast with a
  fixed exchange or recurrence organization. These absences coexist with its
  previously inspected prominent Stream human judgment.
- Sourceecc496…148072–153786: High Jack prominent and Trill supporting, IDs
  `human-30aca5c52b2e159f6a2ecf69` and `human-955bf3e14af9e97c36db37c0`.
  Repeated quads dominate, while an articulated pair exchange occurs at
  150500/150571/150643/150714/150785ms between columns[2,3] and[0,1]. The following
  quad at150928 marks its exit. This is a scoped comparator, not a five-row rule.
- Source98357f…87509–93156: High Jack supporting and Trill prominent, IDs
  `human-c54a1981c3fea4fd4f1d8684` and `human-c31acb592d831d60ea0902b9`.
  Its sustained left-pair/right-pair exchange sits inside a broader mixed scope
  with a localized LN passage and moving attacks. The whole-scope Jack judgment
  must not be transferred onto pure disjoint A/B alone.

Generated Jack/Trill judgments remain unreviewed until the new samples are read.

## Native-screen result: primary guards pass, trajectories differ

The screen completes216 generations and72 whole-suffix scores in697.909s. Every
mechanical/export/reparse check and the three selected exact raw-state recoveries
passes. Peak footprint230114240 bytes, zero swap growth. The training, evaluation
and rendering processes are all terminal; no live job remains from this experiment.
Raw readout SHA
`c957b31053872019f6e8ac74fb1e229529a0ed558030e6db1a73a3d9e4e96c1a`;
`r1-initialization-20260919-v1/readout.json` SHA
`48a7cece3ffd7c9ade1c81023cfa9857a55941116bfc421182c465369b66c02d`.

| Exposure | Initialization | Pooled suffix NLL | LN head share | Group mean rapid pairs/1000 heads | Quad rows |
| --- | --- | --- | --- | --- | --- |
| 250k | 171 | 2.268465 | 9.582% | 2.497277 | 507 |
| 250k | 172 | 2.247107 | 8.188% | 2.266637 | 612 |
| 1M | 171 | 2.054385 | 8.072% | .353136 | 19354 |
| 1M | 172 | 2.210655 | 1.516% | .042441 | 2 |
| 2M | 171 | 1.915834 | 27.651% | .175522 | 104 |
| 2M | 172 | 1.894703 | 51.323% | .338994 | 365 |

At the predeclared2M endpoint, initialization172 has54 rapid pairs and maximum
run2, versus28 and2 for171. The paired group-mean rate difference is+.163472,
95% group-bootstrap interval[-.038340,.529225]. Its macro suffix NLL is1.965769
versus1.987622, difference-.021853, interval[-.034605,-.009672]. These intervals
describe group variability conditional on these two fixed initializations; they
are not uncertainty over a population of model initializations. All four numeric
primary guards pass, including the broad half-to-twice-baseline LN-share guard.

The early250k values are similar, but the1M trajectories have different generated
modes: initialization171 has many quads while172 is overwhelmingly single-note,
with very few LNs. Identical source draws alone do not determine the earlier quad
outcome. At2M, low rapid burden is reproduced, but the aggregate LN style remains
initialization-sensitive. This difference is not by itself proof that one output
style is invalid under the shared timing/seed condition.

The immutable eligible plan contains10735674 post-seed source onsets. Its2M run
covers1697938 distinct ones, about15.8%. The definition was checked in `corpus.py`:
these are suffix onsets, not total physical rows. A preliminary read incorrectly
treated the plan's sources mapping as a list and failed before any mutation;
the corrected mapping sum supplies this count. Coverage does not prove that more
training will help, but incomplete learning remains a live explanation. The
sampler is group-uniform, then chart-uniform, so this fraction is not an epoch
count or a claim of uniform onset sampling.

### Primary semantic guard

All16 context images of the fixed LN-rich/dense2M cores were viewed. Manifest SHA
`6a81820ec391a3ac49d22569b126982099cfc2121d46f1967aaf302bfd5f4a39`;
`inspection-2000k-v1/review.json` SHA
`60cd380e90929920790de39e7ccf47ec34a3bbe0df42e281a68d976968cf5395`.
The frozen Foundation and eight canonical human documents were revalidated.

The LN-rich core has242 LN heads,19 taps and three entering holds. LN coordination
is present/prominent, High: distinct hold roles persist through the whole episode.
For example, columns0/3 start19030 and end19405/19555, while column2 survives to
19180; columns1/2 enter19330, and subsequent releases/restarts preserve different
surviving layers. This is independent control rather than just parallel bars.

The dense core has297 taps and five short LNs, no entering hold. Coordination is
absent, High. Its only two-lane overlap is columns3/2 at106056–106169 and
106112–106169, sharing their release. It then has one held lane plus a tap, and
successive nonoverlapping short holds. This articulation does not establish an
independently evolving multicolumn layer. The primary semantic retention guard
passes; other tags in these two stress cores remain unreviewed.

### Masked ordinary local review

Public packet `ordinary-blind-v1/manifest.json` SHA
`d975fa52536bdea1ba3d7e6fa6023ea92f556a3c57a57943985008ee6d17af0e`.
All64 paired context images for the four preselected16s cores were viewed before
their judgments were recorded. The private initialization mapping remains unread.
Each pair is locally plausible with no observed forced rapid-repeat breakdown;
the descriptive machine preference is a tie in all four. This is neither a human
win-rate experiment nor evidence of statistical equivalence.

- Index0: both have at most one simultaneous held lane in the core, so LN
  coordination is absent, High. One version has a chain of single-lane LN handoffs,
  the other mainly taps; both sustain a sparse moving-column pulse. Stream remains
  unresolved because the calibrated comparisons do not settle this slower sparse
  episode. Judgment SHA
  `87e2a3bc1bc423584d0db0b55ac245983810d6c82077522cd62806733b53aa31`.
- Index3: both show supporting Stream and supporting LN coordination, High. A
  mixes TAP/chord/hold figures; B sustains a legato line with a definite independent
  hold island. Their different articulation does not by itself decide quality.
  Judgment SHA
  `d3c638a39d51098475a1559a309311a4e550dcf67652223cdf18a966b403b39b`.
- Index6: both show prominent LN coordination and supporting Stream, High.
  Independent starts/releases persist through the dense-to-broader supplied timing
  transition. Short LNs down to36ms were retained and inspected. A is more steadily
  LN-led; B has more TAP/LN contrast. Judgment SHA
  `b5a10fa698bbdc70cd2a3aa99df0548f9a4eac03cfbee82a876e2c9a2a7c3436`.
- Index9: A is a flowing TAP-led version with one isolated LN anchor. B has
  independent LN layers and a definite local column1 recurrence across six
  changing-chord attacks311693–312550ms. Both receive prominent Stream, High;
  A has LN/Jack absent and B has both supporting, High. More texture alone is not
  scored as a quality win. Judgment SHA
  `fb2f68d78d81ab887a40fb04f06bc7b3a05815ae2f9b2a53cbdb71187f1b8c0f`.

These machine judgments distinguish actual event relations from the supplied
timing skeleton: the model is not credited with creating the external gaps.
Uninspected tags remain unreviewed. The two64s pairs still need full inspection;
do not read `private-mapping.json` until their masked judgments are also recorded.

### Current evaluation boundary

All predeclared primary numeric and scoped semantic guards pass, making R1 worth
further quality evaluation. Evidence remains exploratory under this proposed
Card. Larger-context review, unused-group confirmation and a durable practical
train/inference handoff remain open before a final quality decision. Preserve the
full objective; local plausibility and mechanical validity do not complete it.

### Wider masked review and subsequent unmasking

Both preselected 64-second pairs are now fully inspected: 56 images for index 0
and 54 for index 6, including the shorter final pages. Their judgments were saved
before opening the private mapping. All six judgments and their referenced packet
files were then hash-verified. The frozen record
`ordinary-blind-v1/judgments-before-unmask.json` has SHA
`2b75c8df5ce5d2ed7e5de09d97a68c1d28da6c7f5742dd9039a894584419a0e3`.
The private mapping retains its predeclared SHA
`71f367499052ed0f58d425b2215732949bafcd2be50640444a472737242e064e`.
No original judgment was rewritten after the identities were revealed.

Index 0, 44609.750–108609.750 ms, gives a moderate preference to A. Its related
hold handoffs, independent anchors and TAP passages return across the wider
scope, producing clearer connected variation. B maintains a readable sparse
single-note pulse and remains a plausible simple arrangement. A has supporting
LN coordination, High: column 1 holds 46061–47023 while column 2 holds
46350–46638 and column 0 enters at 46734; comparable independent roles return
near 105–107 s. B never has two simultaneously held lanes, so coordination is
absent, High. Sparse Stream remains unresolved for both. The preference concerns
the placement and return of figures, not A's 67 versus B's 15 LN heads or the
presence of an extra tag. The earlier local tie remains intact. Judgment SHA
`1e9da5afec78e7213db2785c90cc89967de052fa230bfe67f3c4c40cdb8b34b3`.

Index 6, 31275–95275 ms, remains a moderate tie. Both have prominent independent
LN coordination and supporting Stream, High. A gives longer TAP/LN contrasts:
the staggered holds around 38–55 s return after the moving TAP passage around
62–70 s. B develops a more consistently LN-led texture, with independent
release roles and longer anchors across the sparse 57–62 s passage. Both carry
their organization through the true ending. For example, A's column 2 hold
45017–45421 overlaps distinct entries on columns 0 and 1, while column 3 taps
twice and then starts its own hold. B's columns 0/1 start together at 47039 but
release at 47186/47113; successive entries preserve distinct surviving layers.
The two arrangements have 435 TAP/433 LN heads and 207 TAP/754 LN heads,
respectively. Their shortest LNs are 36 and 30 ms; no duration cutoff was added.
This inspection is not a player-specific difficulty estimate or physical playtest.
Judgment SHA
`98d11ca59bdacc64a06fb55d43a24c0a7f53b47babbad57bd1db50b033dfe895`.

After unmasking, A is initialization 172 in ordinary 0 and 3, and in the local
ordinary 6 pair; A is initialization 171 in ordinary 9 and the wider ordinary 6
pair. The only preference is therefore for initialization 172 in wider ordinary
0. The result is five ties, one limited preference for 172, and no scope judged
both inadequate. `ordinary-blind-v1/unmasked-readout.json` SHA
`ba279b219fbde2ad97a56005117102392bd9e4f7a56357aff5b4f6ded96cc6f1`
records the identities, frozen judgment digests and scope boundaries.

The complete masked review inspected 174 images across four distinct development
groups. The two wider scopes overlap their local counterparts: six scopes are not
six independent trials. Aggregate metrics were known before the display labels
were masked, and all comparisons used generation seed 17. This remains a
descriptive machine review calibrated to human examples, not a double-blind human
preference experiment or an estimate of initialization-population uncertainty.
External timing gaps and density transitions receive no credit as learned rhythm
composition. Other uninspected tags remain unreviewed.

### Evaluation and next decision

Evaluation: REFINE toward confirmation. The scoped initialization experiment is
complete under its exploratory authority. The primary numeric gates, fixed
stress semantics and preselected local/wider inspections jointly justify keeping
both 2M R1 checkpoints as practical candidates. They do not establish one
initialization's general superiority or the full goal's completion.

Freeze these two candidates for a separate, prospectively defined confirmation
on VAL groups excluded from the bounded experiment's development screen. Report
ordinary and stress strata separately, preserve native sampling, and include
preselected local and wider semantic inspection. This can reject the inference
that the observed organization extends beyond the reused development cases.
Inspect and complete a reusable train/inference handoff alongside that work.
There is no evidence-based need to start another training run merely to choose
between the two valid styles. Further O1 training or averaging remains a separate
question; the interpolation mode-loss result is unchanged.

All experiment processes are terminal. The product worktree remains clean at
`15d27db0b0723d7f606b1429c901d26c1f61e5ac`. This evidence append is local; the Note
remains proposed and the overall playable-generation goal remains active.
