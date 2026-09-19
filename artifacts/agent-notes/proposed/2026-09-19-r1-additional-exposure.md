# Agent Note: Does additional R1 learning improve playable continuation?

Note ID: 2026-09-19-r1-additional-exposure
Status: proposed
Kind: research
Created: 2026-09-19
Updated: 2026-09-19
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
