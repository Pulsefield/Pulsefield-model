# Source-action block prediction foundation

The `source_action_modeling` research package implements conditional prediction
of exact four-lane action blocks at supplied event times. It provides the common
observation contract, contextual reference encoder, joint decoder, sampling
policy and update diagnostics needed by the
[representation-access comparison](source_action_representation_directions.md).
It does not implement the [V3 generation contract](../formulation/notation.md).
The [2026-09-14 verification report](source_action_stage1_verification.md)
records the contract checks and bounded real-input observations.

## Information access

[`observe()`](../../src/pulsefield_model/research/source_action_modeling/observation.py)
constructs an immutable `PartialObservation` before any feature or relation
extraction. Complete `PreparedChart` objects remain unchanged for source identity
and target extraction. The partial type contains no source objects, source hashes,
annotation labels, LN identities or complete-chart feature descriptors.

The condition includes the union of real attack and release times in a declared
half-open context. Synthetic scope/context markers occur before source actions
at the same timestamp and cannot be prediction targets. The hidden block is
contiguous in real-event indices, including release-only rows. Its exact attack
group count is recorded for analysis only. The task predicts arrangement at
supplied times; the event count and timing are already known.

| Input | Observable information |
| --- | --- |
| Actions | Tap, LN head and close bits on visible rows; one availability bit distinguishes unknown rows from known lane absence |
| Time | Signed `log1p` seconds for event deltas and declared scope/context coordinates |
| Previous/next lane attack | Elapsed time and availability, recomputed within uninterrupted visible intervals; a known zero differs from unavailable time |
| Occupation | Value and availability before/after each row, propagated only from declared context-entry state and visible actions |
| Skeleton relations | Self, simultaneous hands, adjacent timeline rows and adjacent real events, including across synthetic markers |
| Action relations | First/second attack succession, same-lane recurrence and LN identity only when the required interval is entirely visible |

The caller supplies four bool-or-unknown context-entry occupation values.
`declared_entering_occupancy()` extracts this condition explicitly without
revealing the occupying object's future close time. Hidden actions invalidate
all four lanes' subsequent occupation knowledge. A later visible head or close
can establish new knowledge; occupation after the block is never copied from
complete replay. A hold closing exactly at context start is occupied before
that timestamp's source row. Holds can leave the context without an invented
release or a disclosed future endpoint.

LN ages, durations, remaining times, endpoint visibility flags and complete-chart
occupation relations are omitted. No recurrence or LN identity edge crosses an
unknown row. These rules also apply to features and descriptors on visible rows
outside the target block. Derived partial features are not compatible with old
complete-chart checkpoints.

## Reference predictor and loss

[`ReferenceEncoder`](../../src/pulsefield_model/research/source_action_modeling/model.py)
uses a shared lane projection, a shared-hand bidirectional GRU and the existing
relation-attention computation with the new input dimensions. It returns
`[batch, row, hand, output_dim]`; padded rows are zero. An alternative encoder
can supply this interface and `output_dim` to `SourceActionPredictor` while
retaining the decoder. Source columns map to hand roles as left `(0, 1)` and
right `(3, 2)`, so mirroring exchanges hands without changing outer/inner roles.

The decoder scores one categorical distribution over $6^4=1296$ joint rows.
Each lane has silence, tap, head, close, close-plus-tap and close-plus-head
actions. Each hand's 36 action pairs have learned embeddings; bilateral
interactions couple the two hands. Shared recurrent hand states incorporate
preceding chosen rows. The encoder receives neither targets nor decoder state.
Teacher forcing reads a row only after scoring its distribution.

Legality masks depend on declared entering occupation and the decoded prefix.
A held lane can remain held or close, with an optional simultaneous new tap or
head. An unheld lane can remain absent, tap or start a hold. Unknown occupation
permits the union of those possibilities until the prefix resolves it. The
all-silent joint row is excluded at every real event position. No hidden suffix
or complete post-block state constrains the mask. These are source-action
rules: same-lane close/head coincidences are retained without retiming.

The loss first averages target-row negative log likelihood within each block,
then averages blocks:

$$
\mathcal L = \frac{1}{B}\sum_{b=1}^{B}
\frac{-1}{|I_b|}\sum_{j\in I_b}
\log p(a_j\mid A_{I_b,<j},V_b,\Gamma_b).
$$

Padding contributes no loss and does not update decoder state. This weighting
differs from pooling all target rows. A likelihood improvement can result from
teacher-forced prefix use; it is not sufficient evidence of encoder reuse.
Diagnostics report the first target row separately from later rows.

The default encoder has a 16-dimensional lane projection, 32 GRU units per
direction, four attention heads and a 128-dimensional attention feedforward
layer. The decoder has 32 recurrent units per hand, eight-dimensional action
embeddings and eight-dimensional bilateral interactions. Dropout is zero for
the bounded wiring check. Evaluation commutes with hand exchange; stochastic
dropout during other training runs need not match samplewise under mirroring.
The default model has 58,516 encoder parameters and 23,465 decoder parameters,
81,981 in total.

## Sampling, diagnostics and restoration

[`BlockSampler`](../../src/pulsefield_model/research/source_action_modeling/sampling.py)
samples uniformly over represented groups, then contexts in that group, then
feasible scales from `{4, 16, 64}`, then valid starting event indices. Sampling
uses replacement and never reads action values or labels to choose a block.
Contexts must contain at least four real events. Short contexts redistribute
scale probability uniformly over their feasible sizes. Sampler state includes
the policy identifier, population identity, RNG state and draw position.

[`diagnostics.py`](../../src/pulsefield_model/research/source_action_modeling/diagnostics.py)
captures a fixed mean block log likelihood and its gradient before an actual
optimizer step. It then measures parameter displacement, the resulting response
change, and the gradient-displacement dot product, separately for the disjoint
encoder and decoder parameter owners. Their sum is compared with the actual
change and the linearization residual is reported. Inputs, targets and
non-parameter state must match; both evaluations disable dropout. These terms
do not attribute an update to individual training examples or latent slots.

[`checkpoint.py`](../../src/pulsefield_model/research/source_action_modeling/checkpoint.py)
creates new snapshot files exclusively. Snapshots retain model parameters and
buffers, optimizer state, optional scheduler state, Python/NumPy/PyTorch RNGs,
the active accelerator RNG, training mode, sampling position and update count.
Restoration checks the versioned partial-input contract, model configuration,
encoder/optimizer/scheduler types and sampler population. Exact RNG restoration
requires the same device family. Only trusted local snapshots may be loaded.

CPU continuation tests require bit-exact state and next-update agreement.
MPS state restoration is also bit-exact; probability and next-update comparisons
allow small device-kernel roundoff. Identical MPS modules can produce different
last-bit outputs without checkpoint I/O. Near-zero gradients of softmax-common
biases can amplify that difference in AdamW. Tests therefore also check resumed
gradients and every parameter update against a CPU optimizer supplied with the
original in-memory moments and observed accelerator gradients.

## Bounded local check

[`run_smoke()`](../../src/pulsefield_model/research/source_action_modeling/smoke.py)
is a Python API for software verification. It is not a research-training CLI.
It requires existing local assets from annotation dataset revision
`b22a7a443783e05fee4db4b1d22b8e573ad448ae` and split SHA-256
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.
Missing assets or identity mismatches stop the check; it never downloads data.
The source-group split does not establish comprehensive song/audio deduplication.

Selection takes one context from each of the first eight sorted training groups,
choosing the lowest context hash within each group. Labels, concept availability
and evidence do not affect selection. Original source bytes are SHA-256 verified
and parsed. Each selected review context supplies its first at most 128 real
event positions, with the next event time as the exclusive end when capped.
That bounded interval becomes both task scope and task context. The run records
the original source/scope/context identity, used interval and capped status in
`contexts.json`. Validation/test charts are not read.

Seed 17, logical batch size eight, AdamW learning rate `0.0003`, weight decay
`0.0001` and gradient norm cap `1` are fixed. The check stops at 20 optimizer
updates or 120 seconds including data checks and diagnostics. There are no
unrecorded warmup updates. Fixed evaluation blocks use a separate seed-17 sampler;
updates 1, 2 and 3 also inspect a fixed four-row block from the first context.
Admission reserves measured time for final evaluation and snapshot creation.
An exhausted bound or failed contract is recorded explicitly.

Run from the repository root on Apple Silicon:

```sh
uv run --offline --extra mps --group dev python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
from pulsefield_model.research.source_action_modeling.smoke import run_smoke

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
root = Path('artifacts/scoped-style-modeling')
out = Path('artifacts/source-action-modeling/stage1-smoke') / run_id
result = run_smoke(root / 'prepare-v1', root / 'sources', out, device='mps')
print(out, result['status'], len(result['updates']), result['elapsed_seconds'])
PY
```

Every run directory is fresh. `report.json` records model size, fixed-block
first/later-row losses, sampled exposure, actual parameter movement and the
three update linearizations. `final.pt` retains resumable state when finalization
fits within the bound. The smoke API itself never resumes an existing run.
Generated outputs remain under `artifacts/` and are not repository contracts.

The owning checks and affected complete-chart checks are:

```sh
uv run --offline --extra mps --group dev pytest -q \
  tests/research/source_action_modeling \
  tests/research/scoped_style_modeling/test_replay.py \
  tests/research/scoped_style_modeling/test_model.py
git diff --check
```

## Representation-access comparison

The [composition and access implementation](source_action_stage2.md) adds a
six-level bank, action and concept readers, complete observations and paired
near/detailed/coarse views. It preserves this decoder computation and objective.
Prediction queries are now separate from visibility; the input contract is
`source-action-visibility-v2`, and snapshot schema 2 rejects older snapshots.
The Stage 1 smoke API remains a bounded software check.

The comparison still requires a fixed population, held-out structural metric,
practical gain threshold, semantic regression bounds and execution budget.
Software verification does not establish a representation-access gain, semantic
improvement or demand state. Those questions retain their
[separate research criteria](source_action_representation_directions.md#evidence-of-structure-and-semantic-reuse).
