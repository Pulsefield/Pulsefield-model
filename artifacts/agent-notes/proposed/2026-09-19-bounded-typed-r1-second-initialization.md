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
