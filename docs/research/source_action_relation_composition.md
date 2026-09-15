# Source-action relation composition and semantic reuse

The [packaged experiment](../../src/pulsefield_model/configs/hydra/source_action_composition.yaml)
compares two orders of the same local and relation operators. It evaluates action
prediction, frozen human assessment and controlled structural readouts. The
[runner](../../src/pulsefield_model/research/source_action_modeling/experiment.py)
executes every stage on the local corpus within explicit resource bounds.

## What information paths can establish

An information path supplies an inductive bias: it changes which computations
are available at a given depth and how easily their parameters can be shared.
It does not identify the intended semantic variables. For an invertible change
of representation $T$, an encoder $E$ and readout $D$ can be replaced by
$T\circ E$ and $D\circ T^{-1}$ without changing their predictions, whenever the
model families can implement those maps. Restricted readout families can favor
some coordinates, but action NLL still provides no named Trill or Jack axis.

[Locatello et al.](https://proceedings.mlr.press/v97/locatello19a.html) establish
non-identifiability without suitable assumptions in unsupervised disentanglement.
That theorem is not a direct characterization of this masked-action model; the
relevant lesson is to specify the assumptions and external evidence that would
identify a representation. [Graph networks](https://arxiv.org/abs/1806.01261)
motivate shared relation computations and combinatorial generalization. They
do not establish alignment with rhythm-game semantics.

The desired concepts are also overlapping, rather than statistically independent
generative factors:

| Query | Reusable evidence that could support it | What the evidence omits |
| --- | --- | --- |
| Trill | Repeated two-lane alternation, recurrence after an intervening action, timing and persistence | Human judgment of its extent and prominence amid other actions |
| Jack | Same-lane recurrence with elapsed time, repeated accumulation and chord context | A universal speed threshold or player difficulty |
| Stream | Distribution of attacks, returns, chord sizes and persistent organization over an interval | A definition based on one lag statistic |
| Hand transitions | Ordered own/other-hand and outer/inner role changes, with occupancy and timing | A player's chosen fingering or movement strategy |
| Player-facing demand | Structural evidence combined with player, task, rate and performance observations | Identification from chart likelihood or source frequency alone |

It is reasonable for these queries to reuse the same intermediate evidence.
Neither orthogonal embeddings nor one storage slot per style is required.
The operational question here is whether inexpensive readouts can reuse learned
relations under controlled changes and unfamiliar combinations. A positive result
would justify further causal testing, not the naming of latent slots as demand.

## One intervention: compose after retrieval

Both arms use the `combined` time/local policy: full-row nonlinear encoding,
physical and relative gap coordinates, smooth time bases, source-event adjacency,
action/time-conditioned kernels and direct source access. Visible-prefix state
is not added to the local kernels. Their schedules are:

| Arm | Computation | Readable storage |
| --- | --- | --- |
| `serial` | `U → L1 → L2 → L3 → R → H` | `U,S1,S2,S3,S4,H` |
| `interleaved` | `U → L1 → L2 → R → L3 → H` | `U,S1,S2,S3,S4,H` |

`S1` through `S4` denote execution positions. They use the same fixed reader
descriptors in both arms. They do not denote styles, independent factors or
unconditional local supports. In the interleaved arm, `S4` combines states
whose support already includes visible relation neighbors. Synthetic boundaries
retain their preceding state and do not consume a source-event convolution step.

The hypothesis is that a local operator can combine retrieved evidence from
successive anchors: for example, whether a same-lane return is repeatedly
interleaved with another role, or whether returns persist while hand transitions
change. Retrieval before that operator makes this computation direct. The serial
arm still has nonlinear relation attention and a contextual GRU; it may already
learn the same useful distinction. The experiment can reject a practical benefit
from this reorder. It cannot prove that the serial family lacks that function.

Both arms have exactly 340,913 parameters under the preset: 236,516 encoder,
17,484 action reader and 86,913 decoder. Every initial tensor is identical within
a seed when training from scratch. The bank width remains 64; the feedforward
layer is 256, decoder hidden
size 64, action embedding 16 and interaction rank 16. There are no added slots,
retrieval rounds, losses or synthetic training examples. Changed effective
support and optimization conditioning are consequences of the reorder and remain
alternative explanations to semantic factorization.

## Population and training

The [source catalog](../../src/pulsefield_model/research/source_action_modeling/full_corpus.py)
includes every eligible training song group and every distinct, valid beatmap
within it. `train_group_limit: null` applies no training-group cap. Both novel
index groups and annotation-connected training groups participate. Original
annotated source files absent from the index enter through `source_cache`.
Annotation sections do not become extra pretraining sampling units.

Published groups, positive beatmap/set IDs and normalized artist/title metadata
form transitive song components. Any component touching validation or test is
reserved from training; test takes precedence over validation. Novel components
retain the existing identity-hash 80/10/10 split. Every candidate in training and
validation is verified against original bytes, metadata and 4K action replay.
Exact source/arrangement duplicates are counted once, with validation processed
first to prevent training overlap. Invalid replay and fewer than four events
receive explicit rejection records. Same-lane close/tap and close/head
coincidences are rejected under the V3 single-action lane schema. Grouping does
not establish audio deduplication or
content deduplication against unopened test payloads.

Each training draw selects uniformly at five levels: song group, beatmap within
the group, source-window start, feasible target size in `{4,16,64}`, and target
start. A context contains `min(source_events, max_source_rows)` events, with
`max_source_rows: 256`. Its start ranges over the entire original chart; windows
are generated on demand rather than frozen to two locations. Short charts with
at least four events remain eligible. Original objects supply exact occupancy
for long notes entering the interval. A bounded LRU retains 16 parsed sources,
rechecking byte hashes on cache misses; it does not retain every prepared window.

Large difficulty sets receive the same total group probability as small sets.
Each beatmap within a group receives an equal share, independently of annotation
count or chart length. Long charts consequently receive less exposure per event.
The logged probability describes the entire five-stage draw, not a uniform
distribution over corpus rows. Overlapping windows and edge positions produce
different marginal target probabilities. No importance correction is applied:
the objective is the equal-block mean of mean-row NLL under this declared risk.
Each draw receives near, detailed and coarse views with equal weight. Eight
blocks per update produce 24 block/view examples per arm. Selection uses event
counts and positions, without action-derived style heuristics. Hidden-row times
remain supplied; this experiment does not generate event timing.

Prediction validation fixes 64 identity-hash-selected eligible validation groups,
one beatmap and up to two fixed windows per group, then up to two targets per
feasible scale/group. This bounded evaluation subset is shared across arms and
seeds. Human readouts use all eligible human training/validation cells on their
original scopes, independently of this subset and the prediction context cap.
Source actions from annotated training charts participate in pretraining, while
human labels fit only a frozen encoder's concept reader. A section's label is
never transferred to an entire chart or a random window. Machine assessments and
human evidence-note labels do not supervise this action objective.

Sections provide external judgments of organization over a declared interval,
including same-input Jack/Stream distinctions that action likelihood alone does
not identify. Their value here is as a semantic fitting and evaluation target;
this experiment does not measure whether section-focused sampling or supervised
encoder training improves learning. Adding either would change the intervention.

No test source payload is opened. Allocation Parquets contain every candidate's
split and disposition, eligible source hashes and window-start ranges, and fixed
validation windows. Actual training windows and draw probabilities live in
`seed-*/updates.jsonl`. Each seed records unique song/beatmap coverage in
`coverage.json`, its report and sampler checkpoints. Full population membership
does not guarantee every beatmap is drawn before the time endpoint. Local index,
raw files, annotation snapshot, prepared sections and source cache are required
local assets, not packaged sources of truth.

Each seed trains the two arms on identical samples with AdamW, learning rate
`3e-4`, weight decay `1e-4`, gradient cap `1.0` and zero dropout. Training stops at
20,000 updates or a common per-seed time endpoint, with at least 1,000 updates
required. No validation result selects the endpoint. Both seeds, 17 and 29,
complete all evaluation stages; their update counts may differ under the time
budget and must be reported separately.

## Measurements and interpretation

`structure-final.json` contains sequence NLL, mean-row NLL, first/later decoder
positions, scale/duration strata and matched context gains. Its paired group
bootstrap resamples group means of within-block differences. It measures source
group uncertainty, not uncertainty across training seeds. Keep first-position
and total-block results visible when teacher forcing makes later rows easy.

`*-paths.json` holds the existing fixed-information prefix-route diagnostics:
joint-row JS, TV, both KL directions and true-target NLL. The base detailed view
has its summaries removed once, then half its target prefix is revealed to the
encoder. Scope, visible outside facts, skeleton and suffix legality stay fixed.
Only a fixed prefix of the validation manifest is used. This measures information
routing compatibility; agreement with an uninformative prior is not a success.

`*-human.json` fits the same modest concept reader on each trained and matched
untrained encoder, with the same seed and sampled cells. Encoders stay frozen;
human labels never update the action model. `human-order-comparison.json` compares
the two trained arms on identical cells. Preserve per-concept and same-input
Jack/Stream distinctions alongside macro NLL. A successful readout may construct
some semantics itself; it is evidence of accessible information, not a discovered
causal factor.

`*-relations.json` uses a separate synthetic factorial diagnostic. Every example
contains 64 tap rows, equal counts in all four lanes and a constant physical gap.
Three binary generator settings vary run expansion, motif reordering and the
mapping of lane identities to hands. Frozen ridge readouts predict four exact
statistics from each stage's mean and second moment:

- adjacent same-lane recurrence;
- two-step return with a different intervening lane;
- lane agreement at a lag of 16 events;
- adjacent hand transitions.

Even-parity combinations at 72/120/200 ms fit the readouts. Odd-parity combinations
at those same paces test composition transfer while every individual setting and
every pair of settings appeared during fitting. Both parities at 90/160 ms test
pace transfer. These holdouts concern the diagnostic readout's synthetic training
set; they are not proven absent from the real pretraining corpus.

For counterfactuals at held-out paces, one generator setting changes at a time.
The report records the actual change in every statistic, predicted-delta MSE,
direction accuracy where the target changes, and unwanted response where it
stays fixed. A setting may change multiple facts. This is not a pure intervention
on a human concept. The identical protocol runs on matched untrained encoders
and an orderless source-summary control. Ridge strength is fixed at 0.1 and never
selected on held-out labels. Full case identities and generated sequences make
the diagnostic auditable.

Useful evidence requires a joint pattern: repeatable held-out prediction gains,
meaningful context use, human reuse beyond the untrained encoder, and controlled
structural response beyond orderless and untrained controls. Better NLL alone
supports only prediction. Better synthetic readouts alone may reflect the chosen
grammar. Small routing divergence alone supports only agreement. Two seeds are
an initial stability check, not a precise estimate of training variability.

A stronger semantic-factorization claim would need a declared causal model of
the intended concepts, validated edits with known semantic consequences, and
internal interventions tested on held-out examples and downstream predictions.
[Interchange intervention training](https://proceedings.mlr.press/v162/geiger22a.html)
is a relevant analogue when such a causal specification exists. Imposing it here
would introduce supervision and a new objective, so it is outside this order
comparison. Player-facing demand additionally needs player-conditioned outcomes;
none of the current targets identify that quantity.

## Running on a 24 GB Apple Silicon machine

Run from the repository root after installing the existing `mps` dependencies.
Use a fresh output path; an existing directory is rejected rather than reused:

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.source_action_modeling.experiment_hydra \
  output_dir=artifacts/source-action-composition/overnight-01
```

Inspect the composed values with the same command plus `--cfg job`. Hydra's
inspection does not execute the runner's additional semantic validation.
The runner saves both `resolved.yaml` and the validated `runner.json` before data
loading, along with Git identity, working diff and exact source-module copies.
A dirty run remains exploratory even though these copies aid recovery.

The preset caps total wall time at 11 hours, training at four hours per seed,
MPS driver allocation at 8 GiB, process peak RSS at 12 GiB and output at 4 GiB.
RSS and MPS allocations overlap on unified memory; these are separate guards,
not quantities to add as physical ownership. Each seed reserves 1.5 hours for
finalization when choosing its training budget. The reservation is not a promise
that arbitrarily enlarged readout settings fit. Checks run between updates and
evaluation batches; they cannot preempt an individual kernel or file read.

Snapshots are atomically replaced every 500 updates or five minutes. Each seed
keeps `serial-latest.pt`, `interleaved-latest.pt`, final endpoints and fitted
readouts. A stopped run writes `status: incomplete` and raises; `completed` means
all declared seeds and evaluations finished. The CLI does not automatically
resume. Optional `warm_start_dir` imports compatible per-seed, per-arm weights
before initial evaluation; it starts fresh optimizer and sampler state. See
[checkpoint reuse](source_action_checkpoint_reuse.md) for six-action migration,
required file names and the interpretation of inherited training exposure.
The existing trusted-local checkpoint API restores training state on the
same device family; it does not resume the entire evaluation workflow.

A short verification run can lower execution exposure while retaining the full
training population:

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.source_action_modeling.experiment_hydra \
  'seeds=[17]' validation_groups=4 \
  max_updates=6 min_updates=1 readout_steps=3 bootstrap_samples=20 path_pairs=3 \
  training_seconds_per_seed=300 finalization_seconds_per_seed=900 max_seconds=1800 \
  output_dir=artifacts/source-action-composition/verify-01
```

This checks executable stages, not convergence or architectural superiority.
An explicit `train_group_limit=32` additionally reduces the training candidates
for software debugging; it is outside the full-corpus research setting.
Hardware timing and memory evidence belongs with the exact run identity and
exposure. The configured 64-channel bank is fixed; increasing `hand_hidden` is
rejected rather than silently ignoring it. Larger feedforward/decoder settings
remain explicit but require a new resource check and change the study's scope.
