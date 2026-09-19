# Agent Note: Does additional R1 learning improve playable continuation?

Note ID: 2026-09-19-r1-additional-exposure
Status: proposed
Kind: research
Created: 2026-09-19
Updated: 2026-09-20
Product revision: 50dda55040f51a7afc9a13994b762f953fe3064d
Scope: LN-mode interpretation and matched R1 continuation from 2M to 4M onset exposures
Related: 2026-09-18-bounded-typed-time-three-arm-comparison; 2026-09-19-native-inference-and-difficulty-coverage; 2026-09-19-bounded-typed-r1-burden-mechanism

## Findings that constrain the next experiment

The largest seed-dependent star spread in the frozen 2–6-star source collection
does not by itself demonstrate broken LN organization. Source index 21,
`6450d6ff083550e03a3aec467e219c3bc6d56c21190c5a73cc31b4e1ad6413ef`,
has source rating 3.719836. With R1 initialization 172 at 2M, generation seeds
17/19/23 yield 90.604%/0.626%/75.425% suffix LN heads and whole-chart ratings
6.020917/4.138147/5.798457. The initial 20 physical rows contain 31 heads, including
one LN head, and end at 2415 ms; the candidate skeleton ends at 149961 ms.

The complete 10-second-bin timeline shows seed 17 already reaching 44.4% LN heads
in the first post-seed bin and 84.2% in 10–20 seconds. Seed 23 develops a mixed
opening before mostly LN later; seed 19 stays mostly TAP. All three have only
65 physical rows before 10000 ms, so the entire supplied seed still fits in the
511-token context. Context truncation cannot be the sole cause of this early
separation. This does not establish that the model effectively uses all retained
information, or isolate the cause of subsequent behavior.

The unmasked, retrospectively selected diagnosis fixes three 6-second cores:
8000–14000, 56000–62000 and 130000–136000 ms, with 2-second entry/exit flanks.
All 48 canonical source/generated images and 10 human-calibration images were
actually inspected, plus the whole-suffix timeline. These are one source and
three overlapping-with-context phase samples, not independent quality votes or
full-chart certification. The Foundation, three High human comparisons and
actual source times/endpoints are preserved in the frozen review artifact.

Definite independent LN control remains present/prominent in all three seed-17
cores and the middle/late seed-23 cores. For example:

- Seed 17 early: column 0 holds 7688–9506 while columns 1/2/3 enter and release
  at different times, including 8597–8961, 8779–9143 and 8961–9143. Held anchors,
  shorter exchanges and taps recur through the core.
- Seed 17 middle: column 3 holds 56870–57506 while columns 0/1 pulse together
  and column 2 articulates 56961–57052 and 57143–57234. Later the anchor changes
  to column 0 at 57779–58234 with independently ending surrounding holds.
- Seed 17 late: columns 0/3 hold 131264–131627 and 131324–131627 while columns
  1/2 exchange 60/61-ms short articulations. Groupings continue to change.
- Seed 23 middle: a column-0 hold at 59961–60506 overlaps column 3 ending at
  60143, column 2 at 60052–60143, then column 1 at 60143–60415. Taps occur within
  the surrounding held roles, alongside paired and single short holds.

The early seed-23 LN-coordination judgment stays unresolved at medium confidence:
the core mainly contains sequential single holds with taps. Its first late
two-hold overlap begins at 13870, close to the 14000 boundary; clearer independent
releases/starts occur in exit context after 14052. Do not transfer that exit
context's prominence into the early core. LN/tap integration itself is visible.
All seed-19 cores and the source cores lack independent simultaneous multi-LN
organization; their tap flow and occasional single holds remain locally plausible.
Other Foundation tags were not formally rated in this diagnosis.

The relevant human contrasts are `human-03f7e300cf02f58f3dcbba66` (High prominent,
staggered independent holds), `human-9f2c08a0592fa13087c98d5e` (High absent,
synchronized paired holds) and `human-2208bdfda699add6503ca166` (High supporting,
a bounded LN episode within tap organization). Only the first has a substantive
human comment in this frozen projection. Descriptions of the others are machine
source observations, not invented human rationale. A first filtered retrieval
returned no cards because the stored projection lacks source-fact facets;
tag/confidence retrieval and explicit full-record lookup recovered the examples.
No missing-query result was treated as absence.

This evidence preserves the high-LN alternatives as plausible local expression.
It does not settle whole-chart development, long-gap transitions, player burden
or appropriate difficulty. The earlier harmful 26/35-ms reattacks remain unresolved
in other sources. A type-ratio penalty or short-LN prohibition would remove some
demonstrated organization without isolating those failures.

Diagnostic artifacts under `artifacts/bounded-typed-continuation/ln-mode-20260919-v1/`:

- `plan.json`: `6b7e8786d490a8ad6702ae600b4ac5b84ab93a7fb973b3a6f9b2fa7c983f207c`.
- `inspection-v1/manifest.json`: `58eda7a599e3df77e2944b300d0047e3a4504cf998b870dafba27fcd5d5410b7`.
- `inspection-v1/review.json`: `f92c4ee107b940965b9944da091ad222143d6cf2e0d01f47830a36b6f13f7b17`.
- Frozen human projection: `9ed249b8d7190d31ecbb57dccfba7d91baf88916e425933ffd7b6e5a26839930`.
- Foundation bytes: `b1aea3cbdfe9102e1657d01acfae3f36729467d0a8675b6272ba4f0b17c743ab`.
- Canonical renderer: `28ca3b220fe74f742b495b173719e2e94842c3a6a9840df96a57170af5e66e9d`.

Rendering/measurement took 2.598 seconds under a 240-second/128-MiB-output bound.
No training or generation occurred in that diagnosis. Acceptance/Card: none.
Evaluation: REFINE; scalar style differences must be separated from quality loss.

## Experiment Card: r1-exposure-2m-to-4m

### Identity and authority

Card ID: r1-exposure-2m-to-4m. Revision: 1. Owning Note: this Note.
Acceptance: none. The user explicitly authorizes autonomous model/plan revision
and bounded experiments toward the full goal; execution is exploratory under
that authority. Neither execution nor a passing screen accepts the Card or
establishes overall completion.

### Question, alternatives and analogue

Does continuing the same two R1 models to twice the source-onset exposure improve
the learned distribution and native quality without suppressing useful LN modes?
The selected branch is inadequate training/coverage at 2M. At that checkpoint,
the draw ledger covers about 1.698M unique source onsets out of roughly 10.735M
eligible TRAIN onsets. Shared exposure, finite-context encoding and full optimizer
state make a direct learning-curve test possible without changing the task.

The generic analogue is the distinction between model capacity and training
volume investigated by [Hoffmann et al.](https://arxiv.org/abs/2203.15556).
Their Transformer language-model scales, token units and compute-optimal ratios
do not transfer numerically to mania. This is an ordinary fixed-architecture
learning-curve experiment, with no representation or algorithm novelty claim.

Live alternatives are underidentified style conditioning, generated-history
feedback and an insufficient representation/objective for future occupancy
consequences. The existing high-probability harmful predecessor choices defeat
rare sampling tails as a complete explanation. The new visual evidence defeats
excess LN fraction alone as a sufficient quality-failure criterion. A flat or
worse held-out learning curve would reduce the case for simply extending this
training setup; it would not by itself identify which alternative is correct.

### Fixed comparison

Execution source is clean `50dda55040f51a7afc9a13994b762f953fe3064d`. Parent 171
was trained at `1693d62ffaca04b2a6127d8e3a72d1988adf441f`, parent 172 at
`15d27db0b0723d7f606b1429c901d26c1f61e5ac`. The former source transition added an
O1-only optional endpoint residual and fork support; R1 functions were previously
verified equivalent. The latter-to-execution diff adds the portable wrapper and
extracts the existing source-row projection, leaving cached training, R1 model,
features and loss unchanged. Native equivalence and focused fork tests pass.

Parent checkpoint SHA-256:

- 171: `302fae523cf1fe36afff463fe28f20739a2eb40cab5064d0941a588d8f70c564`.
- 172: `9be0adc00cea85058b59f0a2446884d43fc40cd412188ee123e866df6369d83a`.

The one intervention is additional exposure from 2M to 4M for both initializations.
Architecture, R/H/seed task, legality, 511-row learned context, input features,
native decoder, optimizer and its moments, RNG state, group/chart sampler,
128/256-onset horizons, .125 seed-stratum probability and data allocation remain
unchanged. The shared plan preserves all 10573 old draws exactly and extends to
21179 draws over the same 11563 TRAIN charts/3169 groups. No type-ratio policy,
duration restriction, loss reweighting or difficulty input is added.

Training uses CPU1, hidden128, levels8, expansion4, coupling rank16, no endpoint
availability residual, batch4/microbatch2, candidate budget8192, AdamW learning
rate .0003 after the existing 32768-onset warmup, weight decay .01 and clipping1.
The learning rate does not decay in this runner. Fork restores complete parent
Adam/RNG/coverage state and charges prior compute, rather than restarting the
schedule or treating this as fresh training.

Evaluation reuses the fixed 24 additional VAL groups, with paired generation
seeds17/19/23 at native temperature one. Both model initializations are analyzed
separately. TEST remains unread. Source difficulty bands use the unchanged pinned
20241007 4K/no-mod/rate1 audit. These reused development groups are not independent
confirmation or final evidence of generalization.

### Metrics and decision rule

Primary: for each initialization, reduce the mean of 24 per-chart complete-suffix
NLL/onset values by at least 2%. A paired group bootstrap of the mean 4M-minus-2M
difference, 2000 resamples with RNG seed1189, must have a 95% percentile interval
strictly below zero. Source-band means and pooled values are also reported.

| Initialization | Baseline group-macro NLL | Rapid-pair rate per 1000 supplied suffix onsets | Maximum rapid run |
| --- | --- | --- | --- |
| 171 | 2.128666791174 | 0.062647637983 | 2 |
| 172 | 2.073766490466 | 0.293406866115 | 3 |

The rapid rate here is the mean over 24 groups and three seeds of each output's
below-40-ms same-lane pair count divided by that source's post-seed onset count,
times1000. Its denominator differs from earlier head-normalized summaries; both
2M baselines are recomputed with this exact definition. This is a diagnostic
regression guard, not a universal physical playability cutoff.

Guards for each initialization: all mechanical/export/reparse checks pass;
rapid rate increases by no more than .05; no generated rapid run exceeds three;
no source band [2,3), [3,4), [4,5), [5,6] has mean NLL worse by more than .05 nats.
Report complete star, chord and LN distributions; star variation and LN ratio
alone do not fail quality. These are prospective decision tolerances for this
learning screen, not claimed population or human-performance thresholds.

If primary and numerical guards pass, make masked 2M-versus-4M comparisons for
the original six 16-second core scopes (indices0/3/6/9/12/16) and two 64-second
wide scopes (indices1/2), separately for initializations171/172, generation seed17.
Use the original frozen scope/context boundaries and Foundation/human calibration.
Require both wide scopes plausible, no inadequate core, and preserved definite
LN coordination at index12 and moving-flow organization in ordinary scopes.
Freeze judgments before opening the new pair identities; previous labels remain
unchanged. Index21's phase views remain a separate diagnostic, not an extra vote.

Passing supports continued investigation of additional learning, with difficulty
stratification and LN-form coverage still required for the full goal. Numerical
pass with semantic regression, only one initialization improving, a failed guard
or an ambiguous comparison gives REFINE. There is no automatic run beyond4M.

### Reproduction, bounds and preflight

All new assets live under `artifacts/bounded-typed-continuation/r1-exposure-20260919-v1/`.
The exact projected configs and CLI argument arrays are frozen before launch.

- Original plan: `a6e727d0bbfa414e7d20c742a18d84e8786f155b91629d285e7abf04348cd82f`.
- Extended `plan.json`: `5621297fcf8988d0aa3ae2f35f134af94bba2d21ce68e224dda1d70fb160538c`.
- `commands.json`: `119bcaa88986465102060bb845ec7deec65ec9b45586765a0add9c17a4c50916`.
- `seed171-config.json`: `70bcc562f5530137503375639fd165d30cf6fdf4dfe5bb020149053da6a309c5`.
- `seed172-config.json`: `0d51254a12c5b38cb8d356917f6c32a285f01c11dde87f1aaa93b6337c47fc37`.
- `evaluation-plan.json`: `f02def35cc69fc17495de65cc41eb3ec8d10261bb74765332cf217882a9aae4a`.
- `preflight.json`: `29d90a367171412887d69ef6067989322687fabdd521f959f6cae4886df9426a`.
- `train_pair.py`: `0692064d6a4d6ff7c50d2f067f8482317ab81cb717e1958f4b90d8ff27ceaad9`.
- Frozen VAL conditions: `4200b2c27b8d21ba617889d75add008d3a90efff4cb194b805b5c0ca0d5ac5dd`.
- Baseline readout: `b46ff31e677ec370dfc5487da3b56896ad7fd27baaff23a2e8c2af04cb733765`.
- Difficulty audit: `557707316060149029b7ff93cce8be1ef3deb5a8a8e3e0aa0ea903a4e87dbf6e`.

Launch from the clean implementation worktree:

```sh
uv run --offline --python 3.10 --extra mps python \
  artifacts/bounded-typed-continuation/r1-exposure-20260919-v1/train_pair.py
```

The controller runs171 then172 serially through packaged `train_hydra`, retaining
separate stdout/stderr logs and fresh `seed171-4000k`/`seed172-4000k` directories.
Environment: Apple M5/24GiB, Python3.10.20, Torch2.11.0, CPU1, explicit mps extra.
No training network access or accelerator training occurs. Each arm retains its
14400-second cumulative budget, including parent compute1501.429/1616.286seconds,
6GiB RSS/footprint limit, 2GiB minimum available memory, 128MiB maximum swap growth,
512MiB output limit, 128MiB checkpoint limit and 1GiB disk reserve. Checkpoint every
32 updates and at4M. Evaluation is separately bounded to1800seconds/2GiB output.

Stop on any pin, finite-gradient, source/row, resource or state failure. Preserve
the last durable checkpoint and finalize/audit its runtime ledger before any
explicit fresh-directory recovery. Never overwrite a parent or restart merely
because observation timed out. A controller timeout kills the child and records
that an external runtime audit is required. A failed/incomplete first arm stops
the serial controller before the second arm.

Preflight verifies both parent checkpoints/ledgers, unchanged scientific config
and exact old draw prefix. The first command-construction attempt used serialized
`r1` where Hydra requires enum name `R1`; it failed before any training directory
was created. That failure is recorded, the plan remains unchanged, and corrected
command projection passes. The focused fork owner has two passing tests.

No training has launched at this Note revision. Results, evaluation and decision
remain pending. No Note acceptance, model adoption or goal completion is inferred.

## Result Log: Launch and evaluation preparation

The prelaunch Card and diagnostic were committed at Notes revision
`57dd75570b98e915a315f64f414f7d2c2431a833`; acceptance remains none. The exact
frozen controller command was then launched from unchanged clean product source
`50dda55040f51a7afc9a13994b762f953fe3064d`. Execution session53813 owns controller
PID49232, which started the171 command through uv PID49235 and Python PID49236.
Those processes were explicitly observed alive; at the recorded check the
Python child used99.1% CPU and had committed exposure3293657/update4362. The
latest resource sample had RSS818905088, physical footprint566347384, available
memory9048342528 bytes and zero swap growth. Stderr was empty. These are live
observations, not final maxima or completed-training evidence. The172 arm is
queued serially and has not yet started.

`active.json` locates the current process, but its contents alone never prove
liveness. Poll the existing execution session or verify those PIDs before
deciding whether the job ended. Do not relaunch because a progress observation
times out. The controller retains stdout/stderr and each run's resource/training
journals; successful terminal completion produces `training-pair.json`.

The frozen evaluation procedure is implemented in `evaluate_pair.py`, SHA
`9bac6614fdc9b5033290237774886435670b544e913d55fc986dd5395473d4d5`.
It requires both completed4M ledgers and their exact final checkpoint digests
before creating its fresh `evaluation-v1` output. It prepares standalone
conditions, checks them against the original cache, scores full suffixes and
uses the packaged generation runner for all144 outputs. It verifies source
summaries against the original readout, rates every new export, and computes
the declared group-paired bootstrap and regression guards. A synthetic arithmetic
check confirms zero change for identical baseline inputs and detection of a
3% NLL decrease; this is analysis-code validation, not4M evidence. No evaluation
has run on an intermediate checkpoint.

After both training arms complete, run:

```sh
uv run --offline --python 3.10 --extra mps python \
  artifacts/bounded-typed-continuation/r1-exposure-20260919-v1/evaluate_pair.py
```

Training results, quality evaluation and the research decision remain pending.
The continuation goal stays active, with long-form, cross-difficulty and varied
LN/tap quality requirements intact. All commits remain local.

## Result Log: Both 4M continuations and numerical evaluation complete

Both serial arms completed at the frozen 4M endpoint under unchanged clean source
`50dda55040f51a7afc9a13994b762f953fe3064d`. The original execution session ended
successfully. Each segment performs 2652 updates and exactly 2M additional onset
exposures. The complete 5297-update trajectory covers 2940384 distinct onsets,
8799 charts and 3167 groups. An independent union of the plan's target intervals
matches the checkpoint's coverage bitmaps; both segment exposure ledgers match
each other and every declared draw/checkpoint boundary. Parent results remain
unchanged. No protected Card field changed.

| Initialization | Added training seconds | Cumulative charged seconds | Maximum sampled RSS / footprint, bytes | Swap growth |
| --- | --- | --- | --- | --- |
| 171 | 1465.917 | 2967.346 | 1091321856 / 874595960 | 0 |
| 172 | 1494.112 | 3110.397 | 1154039808 / 928925328 | 0 |

Final checkpoint SHA-256:

- 171: `6f940f0e2c710b7a8c7f365d5703ce0190ad707b449d23c4a87aec78a81651af`.
- 172: `ed4ad7dcec30fb2c6f13ee39908bb34c45b06b41799cdf30cfe410d96efdac28`.

Final result SHA-256, respectively:
`de832a35c4ae54ebd3a93a7b8df03fb79de5ed3b936a938d76c79c9fabf03bfa` and
`7cf3bcfde4634e952a9149c57119329c84ff4171b364b030f0a5461d3aed06ce`.
`training-pair.json` is
`e121343b541f6743d53ae96b757887412884855e1fd231ff7dbb3a6ba7005a1d`;
`audit_training_pair.py` is
`317869096cae87775af909825e687231592d9f560243fed3e3b14a0397cbdeec`;
`training-audit.json` is
`08e168ffd34ff69594fbb7eb9b3be0fec1796e8c0ffcdcf9bad2abeb2b447e25`.

The exact evaluation command completed all 48 suffix likelihoods and 144 native
generations in 420.350 seconds. Every output passes independent mechanics and
exact export/reparse. Sampled maximum RSS/footprint are 1141719040/1082542432 bytes,
with zero swap growth. No output was resampled or filtered.

| Initialization | Group-macro NLL, 2M → 4M | Reduction | Paired group-bootstrap 95% interval of 4M−2M | Pooled NLL, 2M → 4M |
| --- | --- | --- | --- | --- |
| 171 | 2.128667 → 2.028668 | 4.698% | [−0.151322, −0.057800] | 1.970716 → 1.883906 |
| 172 | 2.073766 → 2.001587 | 3.481% | [−0.099802, −0.043235] | 1.920651 → 1.863659 |

All declared numerical gates pass for both initializations. The rapid-pair rate
per 1000 supplied source onsets falls .062648→.019530 for 171 and
.293407→.013006 for 172; total below-40-ms same-lane pairs fall 10→3 and 43→2.
The longest run is two for each 4M model. These counts are locators, not a new
universal playability threshold. Source-band macro NLL differences are all
negative, so none approaches the +.05-nat regression bound:

| Source band | 171 mean NLL difference | 172 mean NLL difference |
| --- | --- | --- |
| [2,3) | −0.052956 | −0.090726 |
| [3,4) | −0.069022 | −0.031474 |
| [4,5) | −0.070577 | −0.038944 |
| [5,6] | −0.102774 | −0.062215 |

The type tradeoff needs semantic evaluation. Suffix LN-head fractions fall
25.342%→11.857% and 53.857%→10.597%. All 72 outputs per 4M initialization contain
suffix-born LNs. However, outputs containing a suffix LN at least 2 seconds long
fall 48→21 and 34→23; at least 4 seconds, 19→6 and 15→9. The maximum remains
20646 ms for both 4M models. These physical-duration locators exclude seed-born
holds and do not define the semantic meaning of a long LN. They establish that
long output objects still occur, not that their organization is good.

Source-history head-conditional LN Brier scores improve .051674→.048965 and
.052080→.048562, while negative log probability assigned to observed LN lane
actions rises .956356→.964375 and .840921→.998330. Better aggregate calibration
and lower overall NLL therefore coexist with a more conservative LN distribution.
These source-history diagnostics do not identify the free-running mechanism.
Do not call the lower rapid rate an unqualified quality win before checking
the preserved range of LN/tap expression.

Numerical artifacts under this experiment owner:

- `evaluation-v1/readout.json`:
  `ab80ec8e351a3d537d1b387c37cdf41af7d76f9e9aa18544c7ac1102477007de`.
- `evaluation-v1/comparison.json`:
  `bd1c1a612f0c6ca63788f661a8fae784fc5c4f7cc1a2936a287e81d4c7f61f69`.
- `evaluation-v1/distribution-summary.json`:
  `39991ea34aea0e53f4b72780fdb84b4bff453ac754b0e4252695ed07869bb9b9`.
- `review-scope-difficulty.json`:
  `317202e7f51a6e6f51ce32233f7669dbdd09206f3c0e0204dc9f26901c0dd0d9`.

The existing eight semantic scopes do span all four 2–6-star source bands:
indices1/9 in [2,3), index3 in [3,4), indices6/12 in [4,5), and index2 in [5,6].
Index0 is below2 and index16 above6. This is a retrospective coverage description;
their original selection used ordinary/stress descriptors. Some bands have one
group and only generation seed17 has these semantic scopes, so this is not an
independent cross-difficulty stability confirmation.

## Masked quality comparison in progress

The numerical pass activates the Card's original 16 paired comparisons. The new
packet preserves every original scope/context boundary and compares 2M versus
4M within each initialization at generation seed17. Presentation RNG seed1197
only assigns A/B labels; it changes no sample, scope or decision rule. Real output
hashes and stage names are hidden from public images/actions. Prior 2M scopes
have already been inspected, so familiarity can reveal identity: this is a
masked machine comparison, not a blinded new human study.

The packet contains 416 canonical pages. The private mapping remains **unopened**.
Freeze all 16 judgments before opening it. Public action-derived risk locators
identify short same-lane pairs and short LNs without exposure-stage information;
the thresholds locate review events and do not assign quality labels.

Packet identities:

- `render_masked_comparison.py`:
  `6ad025945656e14a007b86cfd1fe1796b5a76b8c6acdd9097fb0820ae60ea42a`.
- `masked-inspection-v1/manifest.json`:
  `1ceaaec56d6af5e0f65e5059d984e32eba9caa00be18ca2d951f0d9bb0f14731`.
- Private mapping digest, without opening its content:
  `153758d5ebad9f9a1868979bdd3dad0ea4cc1a35ee78466bbb2ce355b9b67ba6`.
- `masked-inspection-v1/risk-locators.json`:
  `375b44ccaeb4b6dbb306c7ddea20a09ae1eb118de4d5ed25e941b6e8575a1b87`.

Two LN-rich pairs are fully inspected: **32 of 416 pages**, with exact action
witnesses checked. No other new packet pages have yet been reviewed.

- `seed171-ln-rich-12-core_16s`: A and B are locally plausible. A has prominent
  repeated anchor/independent-articulation organization; B has definite supporting
  coordination in a more sequential-handoff/tap arrangement. Limited expressive
  preference for A; greater LN density itself is not the rationale. Judgment SHA
  `f5b3efdf97de1fce3823be36b78c024e650e06e8e1c33ff740beb6ef32752d17`.
- `seed172-ln-rich-12-core_16s`: both variants are plausible with prominent
  independent LN control; A emphasizes continuous interlocking holds and B more
  tap/LN contrast. No decisive preference. Judgment SHA
  `80ca26dfcf9111fa8da92cacdf25b6f4d987771810d6f94450cd4cb26b64c2e8`.

These judgments are saved under `masked-inspection-v1/masked-judgments/` and the
coverage ledger is `masked-inspection-v1/review-progress.json`. Do not infer
either A/B stage or the full qualitative gate from this partial review. Fourteen
pairs remain, including all four wide64s comparisons. Continue those reviews,
freeze the complete judgment set, then open the mapping and evaluate the Card.
No further training is authorized by this Card beyond its completed4M endpoint.
The standing overall research authority persists, but a new intervention needs
its own explicit recorded comparison.

All training, numerical evaluation and rendering processes are terminal. Product
source remains clean at the same revision; Notes stay proposed and commits local.
Overall playability, long-range/cross-difficulty stability and the full LN/tap
goal remain unproven. Research decision and model adoption remain pending.

## Result Log: Complete masked judgments frozen before reveal

The complete packet now has 16 paired judgments over all 416 canonical pages.
The existing six judgments (both LN-rich cores and all four wide contexts) were
recovered from the local coverage ledger and preserved byte-for-byte. The four
wide judgments had been completed after the preceding Notes commit. Their
digests agree with that ledger. Ten remaining pairs were inspected using all
eight time-proportional plot panels per variant in reading montages, with exact
event times, LN endpoints and witness source lines checked in masked actions.
The dense 172 risk page and its paired held passage were additionally inspected
at full resolution. No scope, sample, mapping or numerical guard changed.

The six frozen High-confidence human comparisons were reopened with their full
review-context images. Stream prominent/supporting/absent and LN
prominent/supporting/absent examples remain separate from the machine judgments.
Only the LN-prominent example supplies a substantive human rationale in this
projection. A further short-comment search for staggered/coincident relationships
returned the same prominent example; query misses were not used as absence.

The completed judgments retain three burden-unresolved variants: the previously
reviewed 35-ms events in the wide scopes, and the dense 172 A event with a 26-ms
column-0 repeat at 43422/43448 while other columns are held. All other variant
quality judgments are plausible; none is marked inadequate. The dense 172 B
variant's marginal LN-coordination presence is unresolved: its staggered two-hold
passage ends together without further interaction while both are held. Its
general quality remains plausible. This label uncertainty does not affect the
separately judged definite coordination in the LN-rich core.

There are 12 ties and four limited preferences: 171 LN-rich A, 171 wide02 B,
172 dense16 B and 172 wide02 A. Their exposure-stage identities remain unknown
at this commit. These are scoped machine preferences, not independent human
votes, physical playtests or whole-chart quality certification. Timing/rhythm
and density supplied by R/H are not credited as learned composition.

`masked-inspection-v1/masked-judgments-seal.json` freezes all judgment hashes
with SHA-256
`a2023a2c8f6579d989fba1f684eb609d9d04470a4971d5be0c70b9b5bf83a6be`.
The seal records the unchanged packet, calibration and reading-montage manifests.
Every referenced page/action digest, witness line, scope/context and complete
page coverage was checked before sealing. The six inherited judgments remain
unchanged. `review-progress.json` now records 416/416 pages and 16/16 cases.

The private stage mapping is still unopened. The next step is to verify its
previously recorded digest, reveal the identities against this seal, and apply
the original Card's qualitative gate without revising any judgment. Overall
completion, model adoption and Note acceptance remain unestablished. The human
owner reiterated local commits for meaningful progress on 2026-09-20; no remote
publication was requested.
