# Full-audio context and event-history paths

This experiment separates the finite audio receptive field and unrestricted
history influence of the [joint audio model](audio_conditioned_choreography.md).
It retains native integer-millisecond hazards, complete legal action rows,
exact LN state and the R1-derived finite causal history encoder. It is a
research comparison, not an adopted V3 architecture.

## Conditions

The local branch remains the canonical 128-bin Mel input and a 100 Hz,
96-dimensional TCN. An optional global branch reads the **complete song in
training and inference**. Learned depthwise reduction over 50-frame cells
produces approximately 2 Hz audio tokens. Two bidirectional attention layers
use width 128, four heads, feedforward width 512 and zero dropout. Positional
sinusoids use elapsed audio seconds. Partial final cells count only real
frames; padded cells cannot become attention keys.

Token `j` is anchored at `250 + 500*j` milliseconds. These are audio coordinates,
not generated event slots. Coarse features are interpolated onto the fine
frame clock and read at candidate event times. Training crops include the
local TCN halo and receive coarse features from the entire song. Native
generation encodes both branches once and caches the result. Explicit global
projections into row and timing conditions initially contribute zero, then
pass supervised gradients into the global encoder as their weights learn.

The independent timing switch uses:

```text
logit h = audio_hold_base(F, G, hold_state) + 4 * gate * tanh(history_modulation)
```

The base reads audio, occupancy and active-LN ages. It cannot read learned
content history, row/note counts, previous-row age or time since the first row.
The modulation retains all historical conditions. Its gate is one while any
LN is active, zero at genuine BOS, and `exp(-elapsed_ms/1000)` otherwise.
Thus the historical adjustment has magnitude at most `4*gate` in log-odds.
The row head continues to read full history and exact state. No hazard floor,
forced nonterminal event, history deletion or query-boundary closure is added.
Real rests, sustained repetition and LN release organization can falsify this
proposed restriction.

Self-attention is an established relation-computation primitive
([Transformer](https://arxiv.org/abs/1706.03762)); evolving historical influence
has precedents in event modeling
([Neural Hawkes](https://arxiv.org/abs/1612.09328)). Neither source establishes
this hold gate or its playability. This study tests a task-specific combination.

## Interval likelihood

Training selects a song group, one separate arrangement, then an interval
uniformly from a partition of the actual integer audio clock `0..T`.
Intervals contain at most 8000 milliseconds. Each interval sums every
no-event millisecond and every event's hazard plus conditional full-row NLL.
Every query uses its true preceding source history/state. One causal TCN pass
over prehistory and interval rows supplies each preceding encoding. Future
target content cannot influence earlier queries. Source exhaustion, padding
and interval ends are not completion events or LN deadlines.

With `J` intervals, multiply the selected interval's summed NLL by
`1000*J/(T+1)`. Its expectation is full-chart NLL per second. Song groups and
their separate arrangements are uniform; short final intervals have the
correct inclusion weight. Row and survival terms are summed before weighting.
Alternative arrangements are never merged into a union of labels.

Each update contains two song microbatches with two independently selected
arrangement/interval pairs per song. Coarse audio is shared within a song
microbatch. This time-weighted objective differs from the old query mixture,
so the local/fused cell is a newly trained interval control. The four cells
cross local/global audio with original/bounded timing using the same exposure
plan, R1 initialization, TRAIN normalizer and optimizer.

## Execution and evidence

The typed schema and package-local `joint_audio_context.yaml` own settings:

```sh
uv run --extra mps python -m ensomi_model.research.joint_audio_continuation.context_hydra \
  global_audio=true bounded_timing=true run_name=paths-global-bounded-v1
```

Defaults name pinned local research assets that may be absent in a fresh
clone. The runner requires a clean committed checkout and verified corpus.
It freezes an architecture-independent protocol before fitting and rejects
changed sampling/data settings under an existing protocol name. Unique run
directories contain resolved YAML, flat config, identities, exposure/resource
logs and checkpoints. There is no overwrite, implicit resume or best-endpoint
selection. The default final endpoint is update 1200. A separate preflight can
use `updates=32 validation_every=32 validation_songs=2`; its weights must not
initialize the main runs.

Wall-clock and PAUSE/memory/disk guards apply between updates. Bounded stops
retain the last weights and stop reason; exceptions are recorded and raised.
The existing verified generation loader recognizes `joint-audio/context-v1`
checkpoints and preserves old checkpoint defaults. Native generation and
source-free inference reuse the existing scheduler and export protocol.

`rollout(..., on_update=consumer)` can publish each immutable complete row and
its fixed-through audio clock synchronously. LN heads arrive before their later
release rows. Empty updates advance coverage without adding history tokens;
resource stops do not fabricate closure or completion. Consumer time is included
in runtime measurements and consumer exceptions propagate. Source-free audio
inference flushes these updates to `events.jsonl` as generation proceeds, followed
by a stop record. This local research stream is available before whole-song
export; it is not a client transport or a crash-durable acknowledgement protocol.

Training computation uses finite padding buckets to limit first-seen MPS
operator shapes. Local audio, history and timing-query axes use multiples of
128; row-query axes use multiples of 64. Full-song coarse input uses a power
of two in 50-frame cells. Real-frame/history masks and actual target counts
exclude padding from the modeled distribution and exposure totals. Coarse
context perturbations rotate real tokens only. Padding changes execution cost,
not the source clock or interval objective.

Population validation reports importance-weighted NLL per second. BOS is
reported separately using observed interval duration. Global diagnostics can
zero or shift coarse context by half a song while keeping local audio fixed.
Historical query panels use `score_context_queries` with explicit full-song
context; the original crop-only scorer rejects a global model.

Likelihood and perturbation sensitivity are not playability. Native evaluation
must retain empty, early, post-30 silent and capped outputs. Compare activity,
LN occupancy/duration, eligible same-key TAP transitions and complete action
organization. Release-to-head intervals are a different relation. Preserve
valid irregular timing and jack/chordjack opportunities. Lens inspection,
audio listening and player assessment remain separate evidence.

Focused checks cover independent-query likelihood equivalence, interval
weights, causal indices, hold-only base independence, the modulation bound,
padding/crop/full audio agreement, global gradients, shared plans and the
training-to-checkpoint-to-native-generation path. These establish implementation
properties, not musical quality or realtime deadlines.
