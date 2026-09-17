# Oracle-time continuation: causal data, backbone and sequence training

The `research/oracle_time_continuation` package implements M0–M2 of the
[continuation plan](Pulsefield_oracle_time_causal_continuation_plan.md#9-里程碑与依赖):
verified source rows, a time skeleton, the 30-note seed, exact pre/post-row state,
complete-chart terminal legality, and a trainable causal backbone with bounded
local, relation and temporal memory, window sampling and sequence training.
Decode policy, durable generation, disk-backed source budgets and corpus runs
remain M3–M4 work.

The [V3 formulation](../formulation/notation.md) owns lane actions, simultaneous
rows, occupancy and committed-prefix semantics. The source-action package's
strict raw `.osu` parser supplies byte verification and four-action admission.

## Data ownership

| Owner | Contents |
| --- | --- |
| `ContinuationSource` | Verified source/arrangement SHA-256, existing song-group ID and split, complete supervision rows and their skeleton |
| `TimeSkeleton` | Scheduler-owned, strictly increasing union of original attack and release times |
| `ContinuationState` | Skeleton and exact replay; next position is the committed row count |
| `PredictionInput` | Current time, true skeleton terminal flag, and immutable committed replay facts |
| `ExactReplayState` | First/last committed row, row/note counts, four open LN start times, lane attack/release clocks and completion status |

[`admit_source()`](../../src/pulsefield_model/research/oracle_time_continuation/data.py)
requires the expected source digest and the caller's existing song-group
assignment. It preserves those identities and does not assign new splits.
Every simultaneous action is merged into one complete row. Release-only rows
are retained. Negative/nonfinite times, unsupported sources, overlapping objects
and same-lane close/attack coincidences are rejected without retiming.

Source targets and the complete skeleton stay outside `PredictionInput`.
In particular, it contains no original LN close time, remaining duration,
following gap, future event type, section/window coordinates or source identity.
An open LN contributes only its committed head time and occupancy. The scheduler
provides `is_terminal` only at the true last skeleton row.

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

Persistent learned payload counts are bounded by the configured capacities.
Caller-retained states, logits or attached graphs can still accumulate: discard
consumed outputs and detach carry at TBPTT boundaries. `SequenceTrainer` owns
this optimizer/window lifecycle. Disk-backed source caching and streaming
export remain runtime work; the training API alone does not establish
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
replays every prefix from true BOS with the current model, under `no_grad` and
without a prediction head or query stream. Targets are teacher-forced in chunks
of at most 128 rows. Each chunk is backpropagated once, all learned carry is
detached into owned storage, and consumed outputs are released. Current-chunk
writers and current historical-read projections remain trainable. A short
tail has the same denominator as every preceding chunk.

All windows finish before one gradient clip and one AdamW step. The next update
rebuilds prefix states under the new parameters. Reports include unclipped
gradient norm and cumulative clipping frequency. A failure before the step
clears accumulated gradients and propagates the error; the runner does not
attempt partial-graph or optimizer recovery.

```python
from pulsefield_model.research.oracle_time_continuation.objective import ObjectiveConfig
from pulsefield_model.research.oracle_time_continuation.training import SequenceTrainer
from pulsefield_model.research.oracle_time_continuation.training_config import TrainingConfig
from pulsefield_model.research.oracle_time_continuation.windows import WindowSampler

sampler = WindowSampler(sources)  # admitted sources with existing group/split identities
trainer = SequenceTrainer(model, TrainingConfig(), ObjectiveConfig(lambda_struct=0.3))
report = trainer.update(tuple(sampler.draw() for _ in range(trainer.config.effective_batch_size)))
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

Training requires an explicit list of source SHA-256 identities and the pinned
existing split-manifest digest. The manifest is verified before source loading;
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

The default FP32 model runs on MPS; use `device=cpu` for CPU. On NVIDIA Linux,
replace the environment extra with `--extra cuda` and set `device=cuda`.
`model_seed` initializes weights independently of
`windows.seed`. The output directory must be absent or empty. Before training,
the runner writes resolved Hydra settings, the complete typed runtime config
and the eligible/excluded source population. It streams draw records to
`windows.jsonl` and update metrics to `updates.jsonl`, including prefix
notes/rows/span, recent/coarse coverage, horizon and actual target span,
terminal inclusion, prefill time and cumulative supervised rows.

Successful runs write `weights.pt` with model settings and CPU model tensors.
These are weights for initialization, not a durable training-resume checkpoint.
The selected source set still uses M0's in-memory targets, and no disk-backed
LRU, resource guard, durable update recovery or generated-chart export is
provided here. Those contracts remain M3 work; select a bounded local source
set for this entrypoint.

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
