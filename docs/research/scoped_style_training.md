# Scoped style model and paired training

`ensomi_model.research.scoped_style_modeling` implements the encoder,
assessment readout, teacher-forced evidence selector, and paired trainer for
[Evidence-supervised scoped style modeling](scoped_style_witness_generation.md).
The frozen study retains SHA-256
`197ae4c5de62d4f7207200c6892562650a47e3dbc0ea80640ac86776e7ea9bdd`.
These are research components, not a Ensomi V3 reference architecture.

## Inputs and model

Consume the verified artifacts described in
[data preparation](scoped_style_data_preparation.md). Training checks readiness,
dataset revision, publication manifest, method, cohort bytes, frozen split
assignments, and each loaded chart's hash. Evidence identities are checked against
the replayed candidates before applying an availability mask. Missing evidence
and explicit empty selections preserve the same assessment cohort. Older
preparation summaries can omit the specification hash; the run records that field
as unavailable and separately verifies the current frozen study document.

`tensors.py` separates chart inputs, concept queries, assessment targets, and
selector inputs. It converts milliseconds to signed `log1p` seconds, preserves
availability flags, and orders lane roles as left outer/inner and right
outer/inner. Relation descriptors retain every merged relationship and role pair.
Source-line IDs only resolve exact object history and never become embeddings.

`model.py` implements the study's default dimensions. Shared lane projections
and a packed hand BiGRU feed one sparse relation attention block with relation
biases and values. The assessment branch symmetrizes its ordered hand-pair MLP,
then processes both section boundaries and every in-scope source event using a
packed section BiGRU. The summary uses its forward/backward terminal states,
the mean over actual in-scope events, transformed duration, and an empty-event
flag. Context-only events never enter that mean. Packing prevents recurrent
padding updates in both BiGRUs.

The selector has separate concept/assessment/mask embeddings and separate
per-hand memory. It scores all 16 joint masks through shared unary and symmetric
interaction heads, masks impossible choices, and advances both hands from their
prior states. Forced skips advance memory and replay history but add zero NLL;
padded steps retain the previous state. Entering-LN selections record the original
object without inventing a boundary attack. Exact history carries observed
previous-attack selection, the most recent selected in-scope attack time, and
selected active occupation.

`ScopedStyleModel.assessment(chart, concepts)` accepts no supervision or selector
state. The encoder is the only shared trainable component. `beta=0` skips the
selector entirely. At positive beta, training averages assessment NLL plus
beta times each record's eligible, decision-normalized evidence NLL over the full
batch. Missing evidence does not change the assessment denominator.

## Paired training

Run from the repository root with a fresh output directory:

```bash
uv run --extra mps python -m ensomi_model.research.scoped_style_modeling.train_hydra \
  output_dir=artifacts/scoped-style-modeling/paired-17-new
```

Use `--extra cuda device=cuda` on NVIDIA Linux. `device=cpu` is available for
bounded correctness checks while retaining the platform's explicit Torch extra.
The packaged preset is `configs/hydra/scoped_style_train.yaml` under
`src/ensomi_model/`; `TrainConfig` and `ModelConfig` own accepted fields.
`--help` works without importing Torch. Unknown fields, including nested model
fields added with Hydra's `+` syntax, fail projection.

The default pair uses seed 17, beta 0 and 0.1, batch size 16, AdamW at
`learning_rate=0.0003`, `weight_decay=0.0001`, and `gradient_cap=1`. One epoch
draws the machine training cohort's record count, uniformly sampling concept,
then group, then record. Each epoch's full sampled stream is recorded before
its updates. Both arms receive the same batch, matching shared initialization,
and a deterministic per-update random stream. Selector construction cannot
change shared weights or sampling. No class reweighting or augmentation is applied.

`max_epochs=30` and `patience=5` implement the frozen initial stopping rule.
Checkpoints minimize machine validation assessment NLL, averaging records within
group, then groups within concept, then concepts. The selector's likelihood never
selects a checkpoint. Presence and positive-strength metrics are available as
validation guards; changing beta remains an explicit experiment revision.

`arm_budget_seconds=1200` charges shared batch preparation to each active arm,
plus that arm's training and checkpoint-selection validation time. A paired
update finishes before budget checking, so a slow auxiliary update cannot leave
its control with additional updates. Validation and a final paired batch can
overshoot the allocation; overshoot and truncation are explicit. Per-arm early
stopping can end at different epochs. A wall-clock truncation is not evidence of
equal convergence: use pilot throughput to choose a common feasible bound before
interpreting the comparison. `max_updates` supplies an optional common update
cap for development checks. The command does not resume existing directories.

After inspecting pilot feasibility, paired repeats use `seeds=[29,43]` with the
same fixed configuration and split. Starting the pilot or repeats is an actual
training run; the unit tests do not execute them.

### Overnight allocation on Apple Silicon

The packaged `scoped_style_overnight` preset runs seeds 17, 29, and 43 sequentially
with `arm_budget_seconds=4800`, batch size 16, the same 30-epoch bound and
five-epoch patience, and `memory_log_every=20`. The six arm/seed budgets total
eight hours of charged time. This is an increased execution allocation; model,
loss, optimizer, sampling, split, and assessment checkpoint rules are unchanged.

Run from the repository root with a fresh output directory:

```bash
caffeinate -i uv run --extra mps python -m ensomi_model.research.scoped_style_modeling.train_hydra \
  --config-name scoped_style_overnight \
  output_dir=artifacts/scoped-style-modeling/overnight-new
```

`caffeinate -i` prevents idle system sleep for the command's lifetime and allows
the display to sleep. Keep the Mac connected to power and leave the lid open.
The allocation is not an eight-hour wall-clock deadline: shared preparation is
charged to both arms, early stopping can finish sooner, and final diagnostics
and paired-boundary overshoot add time. The process stops after the configured
seeds; it does not repeat completed runs to fill the night. Interrupted runs
retain already written checkpoints but have no resume interface.

Each seed's `memory.jsonl` records process-wide counters after initialization,
every 20 paired batches, after epoch validation, and after final diagnostics.
MPS active and driver allocation are current byte counters; process peak RSS is
a high-water mark. They overlap and must not be summed. Driver allocation
includes allocator caches; an increase alone is not proof of a leak. Logging
does not clear caches, enforce a memory limit, or establish sustained-run memory
safety. The default training preset leaves sampling disabled;
`memory_log_every=0` also disables it in the overnight preset.

## Outputs and evaluation

Each run writes resolved Hydra configuration, typed runtime configuration,
source/config/test/lock snapshots and hashes, Git revision, command, device,
dataset/cohort/split identities, and preparation support. Each arm records beta,
training/inference parameter counts, update/draw counts, loss logs, elapsed time,
and unweighted and weighted encoder gradient norms at the configured interval.
Gradient logging is diagnostic; it does not balance losses automatically.

`best.pt` contains the selected full model and checkpoint identity.
`assessment.pt` contains only the encoder and assessment parameters and can load
strictly into `ScopedStyleModel(config, auxiliary=False)`. No training resume
interface is provided. Failed runs write `failure.json` and stop the pair.

Validation outputs separate machine and human layers and preserve cell/group,
original assessment, origin, and provenance. Metrics include three-way NLL,
confusion counts and rates, presence NLL/balanced accuracy, and conditional
supporting/prominent NLL/balanced accuracy on **all reference-positive cells**,
including predictions of absent. Missing class support produces unavailable
balanced accuracy rather than a fabricated zero. Evidence outputs contain
reference-conditioned normalized and sequence NLL, with eligibility counts.
Final validation diagnostics have separately recorded runtime.

The trainer never evaluates test charts. Formal held-out paired bootstrap
reporting and fixed relationship-case inspection belong to the subsequent
comparison stage. Beam search, ancestral sampling, count costs, and history
ablations are not implemented and are not required for teacher-forced training.

## Correctness checks

```bash
uv run --extra mps --group dev python -m pytest -q \
  tests/research/scoped_style_modeling tests/test_package_layout.py
```

Model tests cover padding at recurrence/attention/readout and selector memory,
mirror probabilities and sequence scores, valid-mask support, missing/empty
targets, branch gradient paths, target/history isolation, selector removal,
simultaneous memory updates, and the beta-zero AdamW update with dropout. CPU
updates must be bitwise identical. Accelerator checks also compare logits and
gradients, then allow `atol=5e-6, rtol=1e-5` on updated parameters: repeated
identical MPS baselines exhibit small reduction differences even with PyTorch
deterministic algorithms enabled. A synthetic tiny slice checks that both
declared losses can overfit.
Trainer tests exercise paired update caps, budget truncation, validation
selection, separate origins, strict inference export, artifact identity,
configuration projection, and the Torch-free help path. These checks establish
implementation behavior; they do not establish held-out pattern accuracy or a
benefit from evidence supervision.
