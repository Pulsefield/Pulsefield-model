# Source-action composition and representation access

The `source_action_modeling` research package provides three trainable predictors
and frozen human assessment probes. They compare a contextual reference encoder
with an encoder that composes local patterns, retrieves related patterns, then
forms global context. The two composed predictors differ only in which retained
levels their readers can access.

This is conditional reconstruction at supplied source-event times. Bidirectional
chart context and style assessment do not implement the
[V3 generation contract](../formulation/notation.md). The architecture is a
research hypothesis; software verification does not establish a structural or
semantic gain.
The [prediction and evidence contract](source_action_objective.md) owns the
training risk and distinguishes reconstruction, context use, semantic reuse
and matched-information path compatibility.
The [time and local-composition comparison](source_action_time_local.md) adds
separate early-representation controls while retaining these baseline arms.

## Computation and support

[`representation.py`](../../src/pulsefield_model/research/source_action_modeling/representation.py)
owns the 64-wide representation bank. Each retained tensor has shape
`[batch, timeline_row, hand, 64]`. Left-hand columns are `(0, 1)` and right-hand
columns are `(3, 2)`, preserving outer/inner order under hand exchange.

| Level | Computation | Action support |
| --- | --- | --- |
| `U` | Shared projection of own/other two-lane actions, availability and row-time/phase features | One complete simultaneous four-lane row |
| `L1` | Residual temporal block, dilation 1 | Three timeline rows |
| `L2` | Residual temporal block over `L1`, dilation 2 | Seven timeline rows |
| `L3` | Residual temporal block over `L2`, dilation 4 | Fifteen timeline rows |
| `R` | Relation attention over `L3` | Union of local supports at retrieved endpoints |
| `H` | Shared-hand bidirectional GRU over `R` and permitted contextual facts | Full supplied context |

Each local block applies per-node LayerNorm, a kernel-width-3 convolution from
64 to 128 channels, a `tanh(value) * sigmoid(gate)` activation, a 64-channel output
projection and a residual connection. All three blocks use stride one and share
their parameters between hands. Padding is zeroed after the source projection,
after affine normalization and between blocks.

The early path reads only the first four lane channels: tap, head, close and
action availability. It excludes propagated occupation, previous/next attacks
and far summaries. The contextual GRU receives `R`, the existing 12-channel
own-hand lane features, row features, and summaries for both hands. It has 32
units per direction. Locality concerns dependence on action values with a fixed
supplied skeleton: time coordinates and explicit scope/context boundaries are
dependencies of even `U`.

Relation attention reuses the partial-observation graph: simultaneous hands,
timeline/event adjacency, first/second observed attack succession, lane
recurrence and visible LN identity. Action-derived relations never bridge an
unknown source row. Release-only rows participate in temporal composition;
synthetic markers carry no source action. `R` can read distant local supports.

`support_report(observation, complete_chart=...)` returns row indices, covered
timeline/source-event counts and elapsed duration for every level and anchor.
Relation support is the union across the two hand outputs. An optional aligned
complete chart supplies realized attack-group counts solely for reporting.
These counts never enter tensors or reader metadata. The bank retains row-time,
availability, section and length alignment through `ObservationTensors`; it
contains no source-hash embedding or annotation label.

## Action and concept queries

[`model.py`](../../src/pulsefield_model/research/source_action_modeling/model.py)
connects the bank to action predictions. For each target, `ActionReader` forms a
query from its `H` state and supplied row-time/position features. One shared-hand
cross-attention block reads row/level states throughout the context, including
visible source/local states surrounding the hidden target. Keys and values share
projections across levels. Fixed sinusoidal level descriptors and signed/unsigned
`log1p` relative seconds plus time coincidence supply metadata. An output
projection, contextual residual and LayerNorm give one 64-wide context per target
and hand.

Target-index gathering occurs before `JointDecoder.forward`. Its `score` and
`advance` operations, 1,296-class joint alphabet, legality rules and recurrent
prefix computation remain common to all predictors. The reader sees no chosen
action, teacher-forced prefix or target value. The decoder scores each row before
reading that row's target. Loss averages row NLL within each block and then
averages blocks, as in the [foundation](source_action_stage1.md#reference-predictor-and-loss).

[`semantic_probe.py`](../../src/pulsefield_model/research/source_action_modeling/semantic_probe.py)
provides a modest `ConceptReader`. It averages hands symmetrically at assessment,
uses concept-and-duration attention over section source-event anchors and
readable levels, and combines that summary with the mean anchor state, concept
embedding, section duration and an empty-section flag. A shared 64-hidden-unit
MLP returns three logits. Each concept has its own absent/supporting/prominent
distribution; multiple concepts can be prominent together. The head contains no
temporal convolution or recurrent composition.

Complete assessment calls `observe_complete`, `collate_observations`, the encoder
and the concept reader. It creates no reconstruction block and runs no decoder.
Empty sections produce finite outputs with zero anchor summaries. Both heads
mask padding and commute with hand exchange in evaluation mode.

## Comparison configurations

`initialize_comparison(config, seed)` initializes all three predictors from
scratch while preserving the caller's CPU RNG stream. It explicitly copies the
common reader, decoder and relation-attention weights. The composed pair has
identical initial parameters and buffers, with distinct access policies.

| Configuration | Encoder | Reader access | Encoder parameters | Action reader | Decoder | Predictor total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `reference_h` | Lane projection → shared-hand BiGRU → relation attention | `H` | 61,612 | 17,484 | 25,313 | 104,409 |
| `composed_h` | `U → L1 → L2 → L3 → R → H` | `H` | 171,652 | 17,484 | 25,313 | 214,449 |
| `composed_all` | Same composed encoder | All six levels | 171,652 | 17,484 | 25,313 | 214,449 |

Counts use default `ModelConfig`: width 64, four attention heads and zero dropout.
The reference adds a 3,096-parameter projection of both hands' far summaries to
its pre-GRU input, giving both families the same total supplied facts. The
composed arms require 32 GRU units per direction. Each concept reader has 19,155
parameters regardless of access; no level has a separate trainable branch.

Equal improvement in both composed arms supports the backbone as a package,
including its larger capacity. Improvement of full access over the same
backbone's `H`-only access supports direct reads. If the contextual reference
learns the same useful structure, its smaller architecture remains a live
alternative. Old style checkpoints are not these trained controls.

## Visibility, targets and paired views

[`observation.py`](../../src/pulsefield_model/research/source_action_modeling/observation.py)
separates target selection from visibility. Targets are a contiguous block of
real source events and must be unavailable; other source rows may also be
unavailable. A complete observation has no prediction queries. Features and
action relations are recomputed from visibility before tensorization.

`ViewPolicy(near_radius=8)` uses a fixed number of timeline rows on each side of
the target block. The default is a software interface default, not a selected
research condition. A comparison record must fix its radius before training.

| View | Visible action arrangement | Far summaries |
| --- | --- | --- |
| Near | The target's surrounding neighborhood | Unavailable |
| Detailed | All non-target source rows | Supplied |
| Coarse | Same action visibility as near | Identical to detailed |

All views retain the same source/boundary timeline and supplied time features.
For each far side and lane, summaries count taps, heads, closes, occupied-before,
known-before, occupied-after, known-after and source rows. Tensorization applies
`log1p` to each count and adds an availability bit. Occupation counts come from
the detailed observation **after target erasure**, never from complete replay.
These are unordered summaries, not ordered lane content or far relation edges.
Summaries enter the contextual stage only. No altered view receives a style label.

`paired_views` takes the near observation's permitted prefix occupation as the
common decoder entry condition. Detailed/coarse inputs therefore cannot improve
NLL merely by changing the legal output set. `paired_batches` checks matching
targets, skeleton, summaries and query conditions. Structural evaluation also
checks that views match their recorded visibility/summary policy.

`PairedBlockSampler` wraps the foundation's uniform group/context/feasible-scale/
start sampler. Each draw exposes all three views of the same 4-, 16- or 64-event
block to every predictor. Its state retains target RNG/position, population and
view policy. Manifests record event start, scale, actual duration, realized attack
count and policy; sampled training blocks also record the exact marginal draw
probability. Update reports identify the loss, sampling and equal-view weights;
update/evaluation reports add exact tensor-and-target hashes.

For cross-mask comparisons, `prefix_path_pair` moves a fixed prefix from decoder
history into encoder observations while preserving the base view's other facts
and common suffix legality. It requires an unsummarized base because far-summary
source ranges are not explicit in the model inputs. It does not call `paired_views` again.
`evaluate_path_consistency` compares complete joint-row distributions and target
costs on the shared suffix, with no added training regularizer. See the
[fixed-information protocol](source_action_objective.md#fixed-information-prefix-routing).

## Population and evaluation APIs

[`comparison.py`](../../src/pulsefield_model/research/source_action_modeling/comparison.py)
provides Python APIs without adding a training CLI or extending the Stage 1
smoke runner. `pretraining_contexts` takes a verified `PreparedCorpus` and an
explicit set of `input_identity` keys. It deduplicates concept records into exact
source/scope/context/rate inputs, rejects held-out source/group identities and
caller-declared related variants, and preserves original scopes and contexts.

The pinned annotation revision is
`b22a7a443783e05fee4db4b1d22b8e573ad448ae`, and the split SHA-256 is
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.
These identities and the original prepared charts must already be available
locally. The frozen annotation adapter supports recorded 1x inputs only and
rejects other rates. Source-group exclusions do not establish comprehensive
song/audio deduplication; additional exclusion identities remain a research
population responsibility.

`train_paired_step(models, optimizers, sampler, blocks=...)` performs one update
per arm using equal block/view exposure, dimensions, device and optimizer
settings. It returns losses, gradient norms, synchronized update times, parameter
counts and a common manifest. The caller owns the update horizon and schedulers.

`evaluate_structure(models, paired, records)` reports every block's row losses,
`sequence_nll`, `mean_row_nll`, first-row NLL and later-row mean NLL. Raw losses
appear beside near-minus-detailed and coarse-minus-detailed gains in both units,
overall and by event scale and actual duration. A separate breakdown retains
every decoder position's group means, without position-specific bootstrap
intervals. It also compares detailed-view costs between arms via
`paired_detailed_sequence_nll` and `paired_detailed_mean_row_nll`. The report
schema is `source-action-structure-v2`; ambiguous block-level `*_nll` fields
are replaced with explicit sequence/mean-row names. Means are
computed within source groups before averaging groups. Confidence intervals
bootstrap group means of paired differences; one group yields no interval.
The default 1,000 bootstrap draws use a private seed-17 RNG. These means describe
the supplied evaluation manifest; they need not estimate the training sampler's
distribution over contexts and feasible scales.

`SemanticCorpus(corpus, issues)` reuses the effective-human confidence/conflict
policy and then explicitly excludes machine fallback. `issues` must come from
the pinned annotation adapter. `fit_readout` samples eligible human training
concepts/groups/cells, freezes the encoder with `no_grad` and evaluation mode,
and updates only the concept reader. It preserves existing encoder gradients,
parameters, buffers and mode. It requires explicit steps, batch size and learning
rate, with training support for all five concepts.

`fit_matched_probes` repeats the same head initialization and sampled cells on a
matched untrained encoder, using the predictor's original initialization seed.
`evaluate_readout` evaluates original human validation inputs with the existing
group-macro three-class NLL, ranking and 0.5 presence operating point. Reports
include Trill false positives/separation, same-input selective Jack/Stream pairs,
LN presence/conditional strength, confidence support and High-confidence Tech.
Missing classes retain unavailable metrics. Assessment does not consume targets
or evidence masks as encoder input.

Before a research run, fix population and exclusions, near radius, one primary
held-out structural metric, practical gain and semantic regression bounds,
update horizon, selection rule and runtime budget in a bounded
[Experiment Card](../../.agents/skills/research-triage/SKILL.md).
Seed 17 is exploratory; seeds 29 and 43 can confirm a fixed comparison. The APIs
do not choose a budget or claim research success from software fixtures.

## Update responses and checkpoints

The existing actual-update block-likelihood linearization now reports disjoint
encoder, action-reader and decoder contributions. It checks fixed inputs,
targets, buffers and architecture/access policy. `capture_semantic_response`
and `finish_semantic_response` reuse that calculation with a fitted, fixed
concept reader. Their scalar response is the mean positive presence margin
minus the mean negative presence margin, where margin is
`logsumexp(supporting, prominent) - absent`. The fixed human batch must contain
both positive and negative targets. Contributions belong only to encoder
parameters, and changes to the fitted head or its policy are rejected.

`save_snapshot(..., readout=head)` optionally stores that fitted head alongside
model, optimizer, scheduler, RNGs, sampler and update position. Loading requires
the same architecture, access, view population/policy and semantic policy. The
input contract is `source-action-visibility-v2` and snapshot schema is 3; older
snapshots are rejected. Both model families record the context-bilinear decoder
and equal-block mean-row loss policies. The Stage 1 Python smoke API and model defaults
remain available. Snapshots are exclusively created and trusted-local-only.

## Software verification and measurements

Run the owning contract checks on Apple Silicon:

```sh
uv run --offline --extra mps --group dev pytest -q \
  tests/research/source_action_modeling \
  tests/research/scoped_style_modeling/test_replay.py \
  tests/research/scoped_style_modeling/test_model.py
git diff --check
```

The initial composition implementation passed 78 tests in 15.49 seconds on
2026-09-14 with PyTorch 2.11.0 and CPU/MPS devices; those measurements preceded
the context-bilinear decoder and prefix-routing diagnostics. CUDA was
unavailable. Software fixtures do not establish held-out structural/semantic
improvement.

The tests cover hidden-target invariance in every view, strict local action
support, equal-count order changes, visible early-state gradients through direct
reads, gradients through composition, hand exchange, learned normalization biases
at padding, both heads, complete/empty scopes, human-only selection, frozen
encoders and fixed-head update responses. CPU continuation requires exact next
paired draw, next dropout update and head restoration. The foundation's
checkpoint tests also exercise MPS restoration with its documented roundoff
accommodation. CUDA requires its own available hardware and `--extra cuda` run.

Parameter counts in the table are reproducible without local datasets:

```sh
uv run --offline --extra mps --group dev python - <<'PY'
from pulsefield_model.research.source_action_modeling.model import initialize_comparison
from pulsefield_model.research.source_action_modeling.comparison import parameter_counts
from pulsefield_model.research.source_action_modeling.semantic_probe import initialize_readout

for name, model in initialize_comparison().items():
    counts = parameter_counts(model)
    print(name, counts, 'total', sum(counts.values()))
print('concept reader', sum(p.numel() for p in initialize_readout('all').parameters()))
PY
```

`evaluate_structure` measures synchronized evaluation time and maximum retained
bank storage at supplied batch/context lengths. Float32 bank storage is
`batch * padded_timeline_rows * 2 * 64 * levels * 4` bytes: one 130-row input uses
66,560 bytes for the reference bank and 399,360 bytes for either composed bank.
These are retained-state bytes, not peak allocator or backward memory. Reader
stacks, attention matrices and autograd activations require additional memory;
measure device peak memory at selected research context lengths before choosing
the run's batch size.
