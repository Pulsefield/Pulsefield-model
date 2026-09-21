# Oracle-time continuation: causal data, backbone and sequence training

The `research/oracle_time_continuation` package implements M0–M3 of the
[continuation plan](Pulsefield_oracle_time_causal_continuation_plan.md#9-里程碑与依赖):
verified source rows, a time skeleton, the 30-note seed, exact pre/post-row state,
complete-chart terminal legality, and a trainable causal backbone with bounded
local, relation and temporal memory, window sampling and sequence training.
It also implements joint-row sampling, disk-backed source budgets, durable
training/generation recovery and streamed exports. Corpus training and comparative
generation quality remain M4 work.

The [V3 formulation](../formulation/notation.md) owns lane actions, simultaneous
rows, occupancy and committed-prefix semantics. The source-action package's
strict raw `.osu` parser supplies byte verification and four-action admission.

## Data ownership

| Owner | Contents |
| --- | --- |
| `ContinuationSource` | Verified source/arrangement SHA-256, existing song-group ID and split, complete supervision rows and their skeleton |
| `TimeSkeleton` | Scheduler-owned, strictly increasing union of original attack and release times |
| `ContinuationState` | Skeleton and exact replay; next position is the committed row count |
| `PredictionInput` | Current time, terminal flag, immutable committed replay facts, and up to 16 unlabeled future time offsets |
| `ExactReplayState` | First/last committed row, row/note counts, four open LN start times, lane attack/release clocks and completion status |

[`admit_source()`](../../src/pulsefield_model/research/oracle_time_continuation/data.py)
requires the expected source digest and the caller's existing song-group
assignment. It preserves those identities and does not assign new splits.
Every simultaneous action is merged into one complete row. Release-only rows
are retained. Negative/nonfinite times, unsupported sources, overlapping objects
and same-lane close/attack coincidences are rejected without retiming.

Source targets and the complete skeleton stay outside `PredictionInput`.
`ContinuationState.query(time_lookahead_rows)` projects at most 16 future times
into positive offsets from the current time; the default zero retains the
history-only timing ablation. No future action, lane, source-event type, LN
head/endpoint pairing, section/window coordinate or source identity is exposed.
An open LN contributes its committed head time and occupancy. `is_terminal`
comes only from the true last skeleton row. Changing future actions while
keeping the skeleton fixed leaves earlier predictions unchanged; changing known
future times may affect a model configured to read them.

## Seed and prefix construction

`minimum_seed()` counts each TAP or LN_START as one original hit object. A chord
counts every attacking lane; LN_CLOSE contributes zero. The seed ends at the
first complete row reaching at least 30 notes and includes all earlier release
rows. `SeedSelection` records `seed_note_count`, `seed_row_count` and
`seed_elapsed_ms`, measured from the first source row through the seed's last row.

Charts with fewer than 30 notes receive `fewer-than-30-notes`. Reaching the
threshold on the last row receives `no-target-suffix`. Both are explicitly
ineligible; `prefix_state()` rejects them. A release-only suffix is eligible.

`prefix_state(target_start)` replays every row from the chart's beginning through
the row preceding that target position. Omitting `target_start` chooses the
minimum seed. Later windows keep the original first-row time, clocks, counts and
open LNs. They cannot begin before the minimum seed or leave an empty target.

## Query, commit and time

[`ContinuationState.query()`](../../src/pulsefield_model/research/oracle_time_continuation/engine.py)
returns the pre-row prediction input without consuming a target or changing
state. Repeated calls return equal inputs. A teacher-forcing caller must score
this input before reading and committing the corresponding true row; a decode
caller commits its chosen row through the same `commit()` method.

```python
def teacher_replay(source, predict, score):
    state = source.prefix_state()
    while not state.finished:
        distribution = predict(state.query())
        target = source.targets[state.next_index]
        score(distribution, target)
        state = state.commit(target)
    return state
```

`commit()` returns a new state. It validates the whole row against the same
pre-state, then updates every lane simultaneously. An illegal row, duplicate or
out-of-order time, or a time differing from the next skeleton slot raises
`ContractError`; the previous state remains unchanged.

`query.clocks` computes elapsed milliseconds since the previous row, first row,
each active LN head, and each lane/hand's latest attack and release. Canonical
hands use lanes `(0, 1)` and `(3, 2)`. Missing predecessor times use `None`; a
known elapsed time of zero remains zero. Querying a long silent interval advances
ages without releasing LNs or modifying history.

Exact replay always has known history from true BOS. Unknown rows and padding
cannot be materialized as `CompleteRow`; all-empty rows are rejected.
`prefill()` requires consecutive positions from skeleton index zero, so cropped
prefixes cannot impersonate BOS. Learned-memory truncation and batch padding
do not erase exact replay facts.

The optional `model.time_lookahead_rows` enables a separate skeleton-time MLP.
It encodes ordered future offsets and successive gaps with the same physical
time basis; missing positions near the real end have unavailable clocks.
A shared output is added to both hands' query/content facts. At 16 positions
and width 128 this adds 110,848 parameters and no chart-length-dependent cache.
Its final projection starts at zero, preserving shared initialization and the
initial function of a compatible warm start. The engine obtains timing from
the entire skeleton, including positions beyond a training or compute chunk.

`training.timing_learning_rate` optionally gives this new module a separate
AdamW rate. All groups share warmup, decay and the single global gradient clip;
journals record each actual rate. Supplying a timing rate without a nonzero
lookahead is rejected. This is an experimental conditioning path: future-time
availability does not establish a learned gap response or playability.

`model.clock_readout_hidden` optionally adds a direct physical-clock residual
to the joint head's two hand-pair unary scores. Zero disables it. Each hand sees
18 elapsed clocks in relative lane/hand order, the previous complete row,
occupancy and terminal flag, plus the same permitted future offsets and gaps.
Missing clocks retain their availability channels. This path contains no source
LN endpoint pairing or target actions and adds no persistent learned state.
With width 128 and lookahead 16 it has 152,080 parameters. Its final layer starts
at zero; copying every existing parameter preserves a compatible warm start's
initial distribution. This is a conditioning experiment, not a playability claim.

`training.clock_readout_learning_rate` controls its separate AdamW group and
requires a nonzero readout width. When omitted, the group uses the backbone rate.
The first update gives the zero output layer gradients; subsequent updates can
train its input layer. Checkpoints own this group and its warmup state under the
same exact-resume contract. Enabling the branch requires compatible initialized
weights and cannot reinterpret an older model or optimizer cache as a resume.

## Legal support and terminal closure

`query.legal_actions` enumerates legal nonempty joint rows. On ordinary rows,
EMPTY preserves occupancy, TAP/LN_START require a closed lane, and LN_CLOSE
requires an open lane. A chunk or sample horizon has no closure effect.

At the true final skeleton row, every open lane must LN_CLOSE, and each closed
lane may TAP or EMPTY. New LN_START actions are excluded. This leaves at least
one nonempty legal candidate for every occupancy state. Commit marks completion;
querying an exhausted skeleton is an error. Seed LN endpoints are determined by
the subsequently committed rows, even when they differ from source endpoints.

Replay retains a fixed number of exact facts. Full source rows remain in their
CPU supervision owner. Parsed-source cache budgets, RNG snapshots and durable
output recovery belong to the later runtime stages.

## Learned backbone and state ownership

[`CausalBackbone`](../../src/pulsefield_model/research/oracle_time_continuation/model.py)
uses shared hand operators in canonical outer/inner coordinates. Each hand's
features preserve both ordered roles and the other hand's ordered facts. The
model receives only `PredictionInput` and learned history. The scheduler's
skeleton, supervision rows and source identity remain outside its input.

| Owner | Contract |
| --- | --- |
| `features.py` | Available elapsed clocks, physical time basis, and the last 32 completed positive event gaps; timestamp differences are computed before conversion to network precision |
| `local.py` | Three causal time/action-conditioned layers with dilations 1/2/4; separately readable 3/7/15-row summaries and bounded layer-input buffers |
| `relation.py` | Complete-row nodes selected by all four lane frontiers, deduplicated across attack/release indices and active LN heads |
| `temporal.py` | Shared-weight query/content attention; raw layer-input carry, mean archives, visibility metadata and optional inference K/V |
| `model.py` | History encoders and a shared hand unary plus transpose-symmetric bilateral coupling over the full serialized 256-row table |
| `engine.py` | `ContinuationEngine` schedules pure prediction, atomic commit, content-only prefill and bounded dense teacher forcing |
| `state.py` | `NeuralState` groups the existing execution/exact state with local, relation and temporal carry; `detached()` copies learned carry into owned storage |

The local kernels condition each offset-specific channel map on elapsed time
and both committed endpoint actions. Missing predecessors contribute no edge;
they are never repeated rows or synthetic EMPTY actions. Each summary records
the union of original row IDs and timestamps, so overlapping sub-summaries do
not inflate counts. BOS reads use a shared learned boundary. Present and
truncated local supports have explicit status, count and span. Unknown rows are
rejected by `CompleteRow`; batch padding is excluded from all encoders.

Relation queries do not select a target lane. The default frontier indexes the
last 12 attacks and four releases per lane and reads their unique row IDs.
Active LN heads are also pinned with lane identity and committed start time.
Nodes preserve causal lane/hand predecessor IDs, completed close-to-head links
and elapsed intervals; these are facts at commit, not pointers to prior states.
Payloads are never rewritten when later events arrive. A chord appears once in
attention while retaining every relevant lane/role, rank and pin tag. The
default ordinary-node bound is 64 and the bound including pins is 68.

Temporal queries read only the pre-commit bank. Each content layer reads that
same bank plus its own current layer input. Only after all content layers have
been constructed does the engine archive and evict. The default recent base is
512 rows, with up to 15 additional rows still readable while a group accumulates.
At committed count 528, rows 1–16 become one coarse token atomically and rows
17–528 remain fine. Coarse capacity is 64 tokens, evicted FIFO. Tokens retain
start/end row IDs, times, count, birth and eviction metadata. Compression takes
the mean of raw layer inputs before normalization and K/V projection.

Training reads normalize and project retained raw inputs with current trainable
parameters. Within a chunk, committed content retains its writer graph. At a
declared TBPTT boundary, `state.detached()` cuts every learned path, including
local buffers and pinned descriptors, while preserving exact LN obligations.
The next read still trains normalization and K/V projections. `predict()` never
stores a query representation in memory or changes pace, indices or caches.

Inference mode additionally retains projected K/V for the content bank. It
requires `model.eval()` and disabled gradients. All states, including raw-input
training carry, have a process-local parameter/buffer/device/cache signature.
An optimizer update, weight load or device/dtype change invalidates them and
raises `ContractError`; rebuild by replaying the prefix. This guard is not a
durable checkpoint format or an optimizer-resume implementation.

## Model execution API

`BackboneConfig()` selects width 128, two temporal blocks with four heads,
coupling rank 16, and the memory capacities above. Local dilations, the smooth
8–4096ms time basis and dropout zero are fixed. `max_chunk` defaults to 128 and
cannot exceed 128. Smaller model/memory dimensions are useful for contract tests;
configuration validation alone does not establish a hardware resource envelope.
The default model uses FP32.

```python
from pulsefield_model.research.oracle_time_continuation.engine import ContinuationEngine
from pulsefield_model.research.oracle_time_continuation.model import CausalBackbone, row_index

model = CausalBackbone()
engine = ContinuationEngine(model)
seed = source.minimum_seed()  # source is an admitted ContinuationSource
if not seed.eligible:
    raise ValueError(seed.ineligible_reason)
state = engine.prefill(source.skeleton, source.targets[:seed.seed_row_count])

# The training or generation caller chooses the row after reading the distribution.
distribution = engine.predict(state)
target = source.targets[state.execution.next_index]
log_probability = distribution.score(target.actions)
state = engine.commit(state, target)
```

`prefill()` replays content only, under `no_grad`, from true BOS. `predict()`
returns a `JointRowDistribution` in serialized lane order; illegal and all-empty
entries are `-inf`. The joint head accepts typed pre-row encodings only.
`commit()` validates through the M0 owner before constructing a private new
learned state. A rejected row cannot partially advance either exact or learned
history. Committing a generated row uses the same path and makes its actions
visible to every subsequent feature builder.

`teacher_force(state, rows)` accepts at most `max_chunk` consecutive complete
rows. Local and relation inputs are constructed in causal order. Temporal layers
then evaluate rows in parallel using explicit query/content visibility masks
over a bounded union of carried and newly born tokens. Every query gets its own
archive birth/eviction mask, rather than the bank at the chunk's start or end.
The result contains `[Q,256]` log probabilities and legality, the updated state,
and CPU relation/temporal visibility traces for contract checks. It leaves writer
graphs attached; the caller owns backward and the TBPTT cut. Chunk boundaries
are computational and do not change predictions or terminal flags.

`teacher_force_batch(states, row_sequences)` dispatches independent per-chart
dense chunks, then pads their outputs to `[B,Q,256]`. It is a ragged batch API,
not a fused cross-chart attention kernel. A separate validity mask identifies
real queries; padded log-probability cells are zero and padded legality is false.
An empty sequence preserves its state. The sequence objective selects valid
cells and uses the complete effective batch's fixed denominator.

For a differentiable chunk, the scoring primitive is:

```python
import torch

rows = source.targets[state.execution.next_index:state.execution.next_index + model.config.max_chunk]
result = engine.teacher_force(state, rows)
targets = torch.tensor([row_index(row.actions) for row in rows], device=result.log_probs.device)
sequence_cost = -result.log_probs.gather(1, targets[:, None]).sum()
# SequenceTrainer supplies the effective-batch denominator before backward.
state = result.state
```

`temporal_expansion` sets temporal width to `hidden * temporal_expansion`.
When larger than one, a shared hand-wise input projection maps query and content
frontiers to that width. Local and relation representations retain `hidden`;
the joint head consumes the temporal width. `temporal_bias_hidden` separately
bounds the pairwise time-bias MLP. Its default `null` preserves the original
equal-width architecture. None of these settings changes visible events.

Persistent learned payload counts are bounded by the configured capacities.
Caller-retained states, logits or attached graphs can still accumulate: discard
consumed outputs and detach carry at TBPTT boundaries. `SequenceTrainer` owns
this optimizer/window lifecycle. Disk-backed source caching and streaming
export are owned by the runtime; the training API alone does not establish
full-model long-run memory or generation quality.

## Training-window population

[`WindowSampler`](../../src/pulsefield_model/research/oracle_time_continuation/windows.py)
uses `WindowSamplingPolicy`, independently of any future decode policy. It
first filters sources by the existing split and complete-seed eligibility,
then draws uniformly by song group, eligible chart, feasible context stratum,
start event and horizon. Duplicate source identities and song-group split
conflicts are errors. Sources are sorted by identity for reproducible draws;
the sampler owns a separate Python RNG.

Context strata count rows between the original minimum seed and the target:
0–63, 64–511 and 512 or more. Starts are stored as ranges. The three increasing
positive horizons default to 1/4/16 seconds, with equal probability at every
eligible start. A target contains all rows in `[start_time, start_time+horizon)`,
including its first row and any release-only rows, and ends at the true chart
end when necessary. Neither dense targets nor short remaining duration remove
a horizon from the draw population.

`TrainingWindow` records the chart, seed, zero-based start/exclusive stop,
stratum, horizon and population counts. Its `probability` is the complete draw
path probability. Different horizons can yield the same clipped interval;
`sampler.target_probability(window)` sums those paths. `sampler.window(sha,
start, horizon_index)` describes a particular path without advancing the RNG.
Window metadata and source targets stay outside learned features.

`gap_sampling_probability` optionally mixes this base population with time-only
gap strata. Its default is zero, which retains the original RNG sequence and
does not scan gap locations. With probability `p>0`, the sampler chooses a
feasible gap band, song group, chart, gap boundary and eligible start uniformly
at each level. The default bands are `[2,8)`, `[8,32)` and `[32,infinity)` seconds;
only boundaries after the complete seed with a following row qualify. Starts
cover at most `gap_context_rows` (default 32) preceding rows and must include
the boundary in the longest configured half-open horizon. The following gap
does not turn a cropped window into a terminal row. All earlier history still
receives full prefill. Neither action values nor annotations select gaps.

This mixture changes the training risk; it does not importance-correct back
to the original population. A base path has probability `(1-p)*q_base`. A gap
path has probability `p/(bands*groups*charts*gaps*starts)`, with every count
conditioned on the earlier choices. Window records include the route, band and
boundary. `target_probability` sums all matching paths across both routes,
including multiple boundaries that produce the same interval. `population.json`
and startup resource logs record available bands, groups, charts and gaps.
Validation windows remain a separately pinned population.

`draw_batch(count)` defaults to independent draws. With `windows_per_chart>1`,
it shares the route and one group/chart draw within each cohort, plus the band
for a gap cohort. It draws remaining choices independently, then sorts that
cohort by start. The expected batch-average risk of the configured mixture is
unchanged, but windows are correlated. Recorded path probabilities describe
individual draws before sorting, not position-specific order statistics.

## Sequence objective and updates

[`sequence_cost()`](../../src/pulsefield_model/research/oracle_time_continuation/objective.py)
scores valid joint rows using the sum of their negative log probabilities,
divided by `effective_batch_size * normalization_rows`. The reference scale
defaults to 128 and remains fixed for an entire run; it is never a window,
chunk, active-batch or token-count mean. Ragged padding is excluded before
scoring. Illegal truth and nonfinite or unnormalized legal probabilities fail
explicitly.

The same distribution supplies three exact `logsumexp` marginals: press count,
the complete ordered left/right outer/inner press configuration, and all four
pre/post lane occupancy transitions. Their fixed weights are 1/3. A finite
nonnegative `lambda_struct` controls their contribution under the same complete
denominator; zero selects sequence likelihood alone. Reports keep each group's
unweighted code length, weighted loss and weighted logit-gradient L2 norm.
The latter is computed analytically with respect to pre-softmax row logits,
not model parameters; it does not measure learned long-range organization.

[`SequenceTrainer`](../../src/pulsefield_model/research/oracle_time_continuation/training.py)
accepts exactly `effective_batch_size` train windows per update. Each microbatch
replays prefixes from true BOS with the current model, under `no_grad` and
without a prediction head or query stream. `reuse_prefixes=true` permits at most
two same-update chart states; later starts advance that exact state, while a
rewind replays from BOS. All caches are discarded before the optimizer step.
Reports separate logical `prefill_rows` from `computed_prefill_rows`.
Targets are teacher-forced in chunks
of at most 128 rows. Each chunk is backpropagated once, all learned carry is
detached into owned storage, and consumed outputs are released. Current-chunk
writers and current historical-read projections remain trainable. A short
tail has the same denominator as every preceding chunk.

All windows finish before one gradient clip and one AdamW step. The next update
rebuilds prefix states under the new parameters. Reports include unclipped
gradient norm and cumulative clipping frequency. A failure before the step
clears accumulated gradients and propagates the error; the runner does not
attempt partial-graph or optimizer recovery.
`warmup_updates=W` sets update k's LR to the base LR times `min(1,k/W)` for
one-based k; zero disables warmup. Recovery restores the completed-update count
and therefore the same subsequent learning-rate sequence.

```python
from pulsefield_model.research.oracle_time_continuation.objective import ObjectiveConfig
from pulsefield_model.research.oracle_time_continuation.training import SequenceTrainer
from pulsefield_model.research.oracle_time_continuation.training_config import TrainingConfig
from pulsefield_model.research.oracle_time_continuation.windows import WindowSampler

sampler = WindowSampler(sources)  # admitted sources with existing group/split identities
trainer = SequenceTrainer(model, TrainingConfig(), ObjectiveConfig(lambda_struct=0.3))
report = trainer.update(sampler.draw_batch(trainer.config.effective_batch_size))
```

## Local training entrypoint

The packaged preset
[`oracle_time_train.yaml`](../../src/pulsefield_model/configs/hydra/oracle_time_train.yaml)
is the process configuration owner. The Hydra boundary rejects unknown fields
and projects typed model, sampling, objective and training settings into the
runner. Inspect defaults with:

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.oracle_time_continuation.train_hydra --cfg job
```

Training requires the pinned existing split-manifest digest and either an
explicit source list or a pinned admitted catalog. The manifest is verified before source loading;
selected identities must already belong to train. No held-out payload is read,
and this entrypoint neither downloads sources nor assigns groups or splits.
For two sources from the
[pinned real-input table](source_action_stage1_verification.md#real-input-provenance-and-bounds),
choose a fresh output directory:

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.oracle_time_continuation.train_hydra \
  split_sha256=15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a \
  'source_sha256=[000662977cf314075da22600d2fbabbf140edc15791a5fc5dc473f2f64a58923,0123b75a850ebf24136fffe0f975679aa4da8dc46db42e66799a1f494bafaf6e]' \
  output_dir=artifacts/oracle-time-continuation/m2-example \
  updates=2 windows.seed=2 objective.lambda_struct=0.3
```

The default FP32 model runs on CPU; select `device=mps` explicitly for MPS. On NVIDIA Linux,
replace the environment extra with `--extra cuda` and set `device=cuda`.
`model_seed` initializes weights independently of
`windows.seed`. The output directory must be absent or empty. Before training,
the runner writes resolved Hydra settings, the complete typed runtime config
and the eligible/excluded source population. It streams draw records to
`windows.jsonl` and update metrics to `updates.jsonl`, including prefix
notes/rows/span, recent/coarse coverage, horizon and actual target span,
terminal inclusion, prefill time and cumulative supervised rows.

Successful runs write `weights.pt` for generation and `checkpoint.pt` for training
recovery. The latter is saved before the first draw and after every complete
checkpoint interval, and at the requested final update. The base preset uses
`checkpoint_every_updates=1`; the Mac capacity preset uses 25. It includes
AdamW moments, sampler and Torch RNGs, accumulated exposure
counts, model/runtime identity, and both journal boundaries. Resume the same run
with `resume=true updates=<new-total>`. Only the requested total updates and
output location may differ; optimizer, data, model and runtime changes require
a fresh run. Updates beyond the durable boundary are discarded and their draws
are replayed, including completed updates in the unsaved interval. A failed
save leaves the preceding checkpoint intact. For a new optimizer run,
`initial_weights` plus `initial_weights_sha256` loads compatible pinned weights
and records their provenance; optimizer, RNGs and exposure counts start fresh.
An exact resume instead owns its parameters and does not reload that source file.

The 20M capacity preset
[`oracle_time_train_mac.yaml`](../../src/pulsefield_model/configs/hydra/oracle_time_train_mac.yaml)
has 19,976,776 parameters: temporal width 512, six layers, eight attention heads,
64-wide time bias, local/relation width 128, Q=64 and microbatch 1. It uses four
CPU threads, effective batch 8, cohorts of four windows, same-update prefix reuse
and 20 warmup updates. Its LR is3e-5 and weight decay0.01, chosen from the
paired measurements in the [validation report](oracle_time_m3_validation.md).
Inspect it with `--config-name oracle_time_train_mac --cfg job`.

`oracle_time_train_mac_large.yaml` increases only temporal width to 1,024,
for 76,838,984 parameters before optional timing conditioning. It uses a 1 GiB
checkpoint cap, 6 GiB driver/8 GiB RSS guards, at least 2 GiB available memory,
and publication every 100 updates. Its corresponding generation preset is
`oracle_time_generate_mac_large`; the larger load cap covers its FP32 weights,
and its 1,024-row publication interval offsets the wider continuation state.
Both are operational capacity presets, not evidence of improved arrangements.

`widen.widen_temporal()` can double an explicitly projected temporal module
from CPU FP32 weights while leaving local/relation/facts unchanged. It duplicates
hidden and FF channels, rescales Q/K within each head for the changed SDPA width,
and divides outgoing weights across duplicate inputs. Optional zero-sum split
noise breaks gradient symmetry while preserving the initial function to floating
point tolerance. This adapts the function-preserving transfer idea in
[Net2Net](https://arxiv.org/abs/1511.05641); the attention-specific transformation
is checked independently here. It returns a new model without mutating source
weights or global RNG. All learned caches must be replayed and optimizer state
must start fresh. Its seeded initialization and source digest belong in the
converted weights' provenance; it is not exact training resume.

For the existing catalog, a fresh run can use:

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.oracle_time_continuation.train_hydra \
  --config-name oracle_time_train_mac \
  split_sha256=15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a \
  catalog_path=artifacts/oracle-time-review/20260915-adfb1ee/catalog.json \
  catalog_sha256=e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28 \
  output_dir=artifacts/oracle-time-continuation/mac20m-example \
  updates=100 objective.lambda_struct=0.3
```

The catalog adapter validates the entire inherited metadata allocation before
opening selected train payloads. It preserves song groups and verifies raw SHA,
arrangement SHA and event count during admission. The annotation-only source-list
path remains available when the catalog is absent.

Training and generation now use streaming source admission and disk row arrays.
The canonical parser's object rules, arrangement digest and existing group/split
identities are preserved. SQLite orders source objects and event rows with a
2 MiB page cache; Python retains one object/row and four lane predecessors.
The source byte cap is 64 MiB. Steady-state source handles retain metadata only.
`cache.max_sources=16` and `cache.max_bytes=268435456` jointly bound owned parsed
buffers; an oversized entry uses at most `cache.staging_rows=4096` rows of
separate staging. A device never receives the entire chart array.

## Durable generation

The packaged `oracle_time_generate.yaml` accepts the pinned allocation, one
source SHA, a matching `weights.pt`, a distinct decode seed and a fresh output
directory. `split=validation` is the default; `split=train` is explicit for
resource cases. No test source is opened by this runner.

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.oracle_time_continuation.generate_hydra \
  split_sha256=15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a \
  source_sha256=02322a4a739eeb6d2d39dd59a1d667b019bdc23693d0c926ed75fe99d0412191 \
  weights=artifacts/oracle-time-continuation/m3-train/weights.pt \
  output_dir=artifacts/oracle-time-continuation/m3-validation \
  device=cpu seed=17 decode.top_p=1
```

The accelerator extra installs Torch; `device=cpu` explicitly runs the model on
CPU. Both `decode.top_p=1` and `0.95` sample from the legal complete-row
categorical distribution. Temperature defaults to one, and `beta=0` is required
because no history prior has been fitted. Cutoff ties are retained together.
`decode.mode=greedy` chooses uniformly among tied maxima for a debug comparison;
it does not replace stochastic generation. Every generated record distinguishes
raw model log probability from the renormalized decode probability and records
raw/policy LN-close mass, pre-row hold ages and terminal forced closures.

`Rollout` accepts a skeleton and seed iterator, never a source suffix. It commits
only sampled actions after the seed. Output `rows.jsonl` preserves every source
time and complete action row. `generated.osu` uses a bounded disk sort to emit
start-ordered TAP/LN objects; its endpoints all come from committed LN_CLOSE
rows. Source playback metadata, audio filename and timing/SV points are copied
only into the export header; none enters the model. The difficulty has a new
beatmap identity and does not bundle audio. An independent lane
validator checks every output row and complete closure before export.
`summary.json` reports press counts, occupied duration, same-row repetition,
LN-duration bins, minimum same-lane attack gap and decode truncation diagnostics.
These measurements do not establish human playability or style quality.

For interruption recovery use the same command with `resume=true`. The
checkpoint binds model tensors, runtime/source hashes, cache schema, skeleton,
policy, RNG, next event, exact/local/relation/temporal state, accumulated metrics
and the output byte boundary/digest. Resume validates the durable prefix before
truncating only its uncommitted tail. It never combines an old output prefix
with a new sampler state. Runtime or weight changes require a fresh rollout.
`checkpoint_every_rows=512` controls periodic publication independently of
resource checks. Prefill and the end of each `advance()` call also save; export
uses that already durable completed state without rewriting it.
This interval is part of the resume identity; increasing it trades less writing
for more replay after interruption, without changing sampling probabilities.

On macOS/Linux, each checkpoint, export and admitted source has one owned
`.staging` directory and a POSIX publication lock. A process death releases
the kernel lock; the next publication reclaims only that destination's incomplete
staging. Repeated interrupted saves therefore retain at most one staging payload
per destination, alongside the last valid publication. Concurrent publication
fails without touching the active writer's staging. This lock does not permit
multiple trainers or rollouts to share output journals; use separate run folders.


## Resource and storage limits

The runtime supports local/relation width up to 128, temporal expansion up to
eight, up to six temporal layers and eight heads. Expanded temporal models
require microbatch 1, `max_chunk<=64` and time-bias width at most 64. The original
128-wide, two-layer configuration permits microbatch 2 and Q=128.
Smaller dimensions are permitted, but validation of one
workload does not establish every device/shape combination. Resource checks run
through prefix replay, target forward/backward/update, decode and publication.
They track MPS active/driver bytes, process RSS, system available memory, pressure
and swap growth independently; overlapping counters must not be added.

Default stops are 4 GiB driver, 6 GiB RSS, less than 512 MiB available memory,
critical macOS memory pressure or more than 1 GiB additional system swap.
The MPS allocator ceiling is 8 GiB divided by the device's recommended maximum.
Guard checks are sampled at most 128 rows apart and cannot rule out transient
operator peaks or allocations by other processes. An exception aborts the
operation; recovery relies on the already published checkpoint, not a save
attempt after an allocator OOM. Training rejects a model whose FP32 weights plus two AdamW moments already exceed the checkpoint cap before constructing the optimizer; actual serialization still has a separate byte limit. `release_mps_cache_at_boundaries=true` releases
unused MPS allocator capacity between no-grad prefill and target execution, and
after each complete update and gradient disposal. At generation resource boundaries and checkpoints,
MPS carry is copied into fresh bounded storage before releasing idle allocations:
live MPS operations were observed retaining more allocation than reported tensor
storage, and a fresh copy released it. Values and exact obligations are preserved;
there is no row-level allocator flush or context shortening.

`resources.checkpoint_max_bytes=134217728` bounds each checkpoint, including
serialization overhead. CPU staging clones tensor views into compact storage.
A temporary file is fsynced and atomically replaces the one retained checkpoint;
its previous version is never removed first. Publication reserves the entire
configured staging allowance plus `disk_reserve_bytes=536870912` of free disk.
The capacity training preset raises the checkpoint cap to 512 MiB and saves
every 25 updates to bound write frequency for its roughly 229 MiB checkpoint.
A full-memory 20M generation checkpoint is about 45 MB. The 512-row publication
interval reduces periodic writes relative to the former 128-row interval;
MPS carry ownership is still refreshed at the more frequent resource boundaries.
A training run also retains one weights artifact. Journals and each export have
an explicit 2 GiB limit. SQLite export staging is disk-backed and separately
space-checked. Checkpoint size is independent of chart length; persisted rows
and logs grow with the executed work.

Time features retain exact milliseconds in replay, but network channels use
`seconds / (1 + abs(seconds))`, `asinh(seconds)` and the existing elapsed-time
basis. This removes the linear amplification of chart/hold age on long charts.
The cache and `weights-v2-bounded-time` formats distinguish this representation
from earlier linear-seconds weights. Earlier weights require their original
runtime and cannot silently initialize this generation entrypoint.

See the [M3 resource and parameter report](oracle_time_m3_validation.md) for
measured throughput, storage, stability and remaining output-quality limits.

## Verification

Run the focused contract suite with:

```sh
uv run --offline --extra mps --group dev pytest -q tests/research/oracle_time_continuation
```

The suite exhaustively checks four-lane action legality across all 16 occupancy
states, including terminal support. It also covers threshold chords and short
seeds, release-only suffixes, simultaneous pre/post clocks, missing versus zero
time, invalid sources, immutable queries, changed future LN endpoints/actions,
changed future skeleton times, mirrored replay, prefix/chunk parity, and a
continuation that replaces the seed's original LN endpoints. An import check
keeps the data path independent of model, legacy training/inference, and masked
feature-building modules.

The model tests independently check short-history support unions, causal pace,
time-conditioned kernels, relation deduplication and LN pins, future-action
isolation, simultaneous commit, mirror equivariance, ragged batch/step parity,
all occupancy/terminal supports, and parameter-version rejection. Temporal tests
cover Q=1/17/64/128, n=512/513/527/528, FIFO eviction after 64 coarse tokens,
mean-before-normalization, layer-input and inference-cache parity, and the
separate historical-read and within-chunk writer gradients. The MPS test uses
the default FP32 model and a 128-row differentiable chunk; it is conditional on
MPS availability. On NVIDIA Linux use `--extra cuda`; MPS evidence does not
establish CUDA coverage.

M2 tests enumerate population probabilities, stratum boundaries, half-open
horizons and clipped-path aggregation. Independent enumeration checks all three
marginals across every occupancy and terminal condition, including their logit
gradients. Training checks cover ragged effective batches, microbatch parity,
the 532-row target's five chunks, sequence-cost additivity, within-chunk writer
gradients, detached prefix/carry, parameter-version replay, single clip/step
and gradient disposal on failure. Entry tests cover typed configuration,
unknown-key rejection, split provenance, runtime consumption, deterministic
CPU updates, package resources and the runtime's Hydra import boundary.

[`verify_source()`](../../src/pulsefield_model/research/oracle_time_continuation/verification.py)
checks every pre/post-state against an independent source-side oracle. The oracle
uses bisection over raw attack/release times and LN intervals, including exact
endpoint equality. It also compares selected full-prefix constructions with
continuous replay. A mismatch raises `ContractError`. The returned report keeps
source identity, counts, seed eligibility and checked prefix positions.

For a locally available source from the existing pinned split:

```sh
uv run --offline --group dev python - <<'PY'
import json
from pathlib import Path
from pulsefield_model.research.oracle_time_continuation.verification import verify_source
from pulsefield_model.research.scoped_style_modeling.dataset import canonical_json, digest

root = Path('artifacts/scoped-style-modeling')
split = json.loads((root / 'prepare-v1/split-manifest.json').read_text())
expected = '15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a'
actual = digest(canonical_json({k: v for k, v in split.items() if k != 'sha256'}).encode())
assert actual == split['sha256'] == expected
sha = '000662977cf314075da22600d2fbabbf140edc15791a5fc5dc473f2f64a58923'
assignment = split['sources'][sha]
report = verify_source((root / 'sources' / (sha + '.osu')).read_bytes(), sha,
                       group_id=assignment['group_id'], split=assignment['split'])
print(json.dumps(report, indent=2))
PY
```

Local source files and the split manifest are required; this API performs no
downloads. The check uses CPU replay and does not require an accelerator extra.

On 2026-09-15, complete source replay was checked for the eight source identities
in the [pinned real-input table](source_action_stage1_verification.md#real-input-provenance-and-bounds),
under that split digest. All **16,800 rows** matched in both pre- and post-state,
covering **23,901 hit objects**, **2,018 LNs** and **417 release-only rows**.
The eight minimum seeds contained 30–31 notes over 15–30 rows. Every source had
an eligible suffix and closed occupancy after its final row. These are data and
replay correctness checks; those checks trained or evaluated no model.

### Sequence-training smoke

On 2026-09-17, the two sources in the training command above completed two
updates on MPS with the default FP32 backbone, `model_seed=17`,
`windows.seed=2`, `lambda_struct=0.3`, effective/microbatch size two and Q=128.
The sampling seed was selected to cover all three context strata and horizons
in four draws. Source bytes and the pinned split digest were verified before
training. The windows were:

| Context stratum | Horizon (s) | Prefix rows | Target rows | Backward chunks |
| --- | ---: | ---: | ---: | ---: |
| 0–63 | 1 | 69 | 6 | 1 |
| 512+ | 16 | 644 | 161 | 2 |
| 64–511 | 4 | 413 | 41 | 1 |
| 512+ | 1 | 673 | 13 | 1 |

The run replayed 1,799 prefix rows and supervised 221 target rows. Both updates
had finite losses and gradients, and each of the three weighted marginal
logit-gradient norms was positive. Unclipped parameter gradient norms were
65.31 and 157.86; each update clipped once at the configured norm cap of one.
Update wall time totaled 269.45 seconds, including 212.13 seconds of prefix
replay, measured with device synchronization. The process used an 8 GiB MPS
allocator ceiling. These observations verify the training path across the
chosen histories and durations; they do not measure convergence, generation
quality, long-run memory stability or CUDA behavior.
