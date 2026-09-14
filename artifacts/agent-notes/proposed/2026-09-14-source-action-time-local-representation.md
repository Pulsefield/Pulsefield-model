# Agent Note: Separate source-action time geometry from row and local computation

Note ID: 2026-09-14-source-action-time-local-representation
Status: proposed
Kind: research
Created: 2026-09-14
Updated: 2026-09-14
Product revision: 84b02a3037d26dae301854bbf9cf0717edd907a7 with uncommitted time/local representation implementation and verification changes
Scope: Source-action time coordinates, row interaction, event-local operators, direct source/state access and controlled comparison interfaces
Related: 2026-09-14-source-action-stage2-access-comparison, 2026-09-14-source-action-stage3-pilot

## Question and authority

Do explicit physical/relative time coordinates and action-conditioned event-local
composition make complete source-action relationships easier to learn and reuse,
beyond the existing timeline convolution and single row projection?

The owner requested implementation of the review's second priority after the
decoder/objective work. This authorized the scoped model/control implementation
and software verification. No corpus-training run, exact Experiment Card,
acceptance, lifecycle transition, remote push or product commit is recorded by
this note. Its proposed status is independent of the code's implementation state.

## Baseline and ownership

The baseline product revision contains the context-dependent bilateral decoder,
sequence versus mean-row likelihood contract and fixed-information prefix-route
diagnostics. The selected baseline here is its `composed_all` bank, with the same
sampler, visibility, decoder, relation graph, contextual mixer and readers.

Relevant owners:

- `docs/research/source_action_objective.md`: prediction family and evidence.
- `docs/research/source_action_stage2.md`: original composition and access arms.
- `docs/research/source_action_time_local.md`: implemented early-path contracts,
  mathematical coordinates, support, parameter counts and comparison interfaces.
- `src/pulsefield_model/research/source_action_modeling/time_basis.py`: physical
  coordinates and source-event geometry.
- `src/pulsefield_model/research/source_action_modeling/local_representation.py`:
  source packet, residual row interaction, conditional local kernel and support.
- `src/pulsefield_model/research/source_action_modeling/representation_experiments.py`:
  named controls, shared initialization and paired updates.

The related composition note retains the earlier access hypothesis. The related
pilot note owns its original population and execution plan. Neither is
superseded or changed by these early-path alternatives.

## Closest analogues and provisional novelty

[CKConv, version 3](https://arxiv.org/abs/2102.02611v3) parameterizes sequential
kernels by continuous coordinates and supports irregular sampling. The useful
mechanism is direct time-dependent composition. The adaptation here restricts
support to a few supplied source events and includes observed endpoint action
roles and availability in a compact kernel gate.

[Numerical feature embeddings, version 4](https://arxiv.org/abs/2203.05556v4)
studies piecewise-linear and periodic scalar embeddings before tabular backbones.
It motivates a coordinate comparison independently of network depth. The present
smooth saturating bank is a different basis for exact event-gap inputs, with
separate physical and local-ratio coordinates.

The provisional novelty is the source-visibility/mirror-preserving adaptation and
the controlled combination of these mechanisms. No new general architecture or
empirical superiority is claimed.

## Live branches and separating evidence

| Branch | Mechanism | Supporting observation | Falsifying or ambiguous observation |
| --- | --- | --- | --- |
| Event adjacency | Markers no longer consume a convolution step; releases remain events | Better paired prediction with unchanged parameter count; software marker invariance | No gain or degradation on the fixed distribution; event count still poorly matches useful duration |
| Scalar physical/relative coordinates | Direct gaps and local ratios reduce the work needed to recover timing geometry | Scalar improves over event-only, including tempo/irregularity strata | Improvement restricted to a timing-only prior, without context use or reuse |
| Smooth basis | Multiple time sensitivities improve finite-capacity learning of short intervals | Smooth improves over the same-information scalar arm | No gain, long-gap regression, or improvement explained by added projection capacity |
| Residual row composition | Another nonlinear full-row/time interaction forms useful primitives | Row improves over smooth, including frozen reuse | Only decoder reconstruction improves or extra MLP capacity explains the change |
| Pair-conditioned kernel | Endpoint actions and elapsed time modulate local combinations directly | Time-action gating exceeds constant, action-only and time-only controls | Action-only accounts for the gain, or all extra-gate variants behave alike |
| Row plus kernel | Richer primitives and conditioned composition provide complementary computation | The smooth/row/time/row_time square shows added benefit from their combination | Either component adds no value once the other is present |
| Repeated source access | Raw facts/metadata bypass U compression and content normalization | Combined improves over row_time, with meaningful context use and reuse | No gain or reader/decoder priors account for the entire effect |
| State conditions | Known visible-prefix occupation is locally usable | State improves over combined on hold/release organization | Gain depends only on broadened support; state route dominates without reusable local distinctions |

Each comparison must share its target/view manifest, decoder prefixes, legality,
optimizer settings and reader access. Kernel controls retain the same tensors
and initialization but zero selected inputs; their effective functional capacity
is not thereby equal. Additional parameters in other arms are also confounders.

## Implemented constraints

- Scalar inputs preserve physical seconds and asinh(seconds). Smooth inputs add
  signed/even responses at 8 through 4096 milliseconds in powers of two.
- Local ratios use only adjacent supplied event gaps. They are invariant under
  uniform tempo rescaling while physical channels remain sensitive to speed.
  This is neither attack pace nor BPM, and no player ability prior is calibrated.
- Unknown actions are masked before every feature calculation. Source-event
  order retains hidden and release-only events and excludes synthetic markers.
- U contains current-row action facts and declared time only. Optional occupation
  enters local blocks separately, with availability and visible-prefix provenance.
- Action supports of the event-local layers are at most 3/7/15 source events.
  Time support additionally includes the neighboring gaps used by pace. State
  support conservatively includes the prefix and declared entering occupation.
- Raw source access is an independent projection into each local value/gate
  drive. Learned content retains Pre-Norm gated residual composition.
- Kernel conditioning is output-channel modulation of offset-specific maps,
  with possible sign changes. It is not an arbitrary dense per-pair kernel.
- The original architecture remains the exact baseline. Unchanged tensors are
  copied, new modules have stable initialization per path, and selected arm order
  does not consume or change the caller's CPU RNG stream.
- Existing structural reports, snapshots, actual-update responses and frozen
  semantic probes consume the new policy. Matched untrained probes recreate the
  selected representation and access instead of selecting an old architecture.

## Software verification

The focused tests use synthetic source-exact chart fixtures, not a corpus
performance experiment. The verification command is:

```sh
uv run --offline --extra mps --group dev pytest -q \
  tests/research/source_action_modeling \
  tests/research/scoped_style_modeling/test_replay.py \
  tests/research/scoped_style_modeling/test_model.py
git diff --check
```

On 2026-09-14 the command passed 131 tests in 15.36 seconds. Environment:
Python 3.10.20, PyTorch 2.11.0, macOS arm64, CPU and available MPS. Checks include
exact baseline/common initialization, all three visibility views, hand exchange
of both heads, learned normalization bias at padding, marker insertion,
source/action/time/state support, direct raw access, isolated kernel time/action
controls, gradients, empty timelines, paired updates/evaluation, matched frozen
probes and exact CPU continuation. A same-shape kernel-policy change is rejected
before checkpoint loading mutates model state. Product whitespace checks pass.

CUDA is unavailable. No baseline or intervention corpus metric, trained research
checkpoint, semantic gain, accepted Card or acceptance revision is available.

## Recommendation and next decision

Outcome: REFINE. The candidate computations and separating controls are
implemented; their empirical ranking remains open. Keep the original baseline
and select a bounded subset rather than launch a twelve-arm search by default.
A practical first question is event/scalar/smooth; the computation square and
kernel input controls address subsequent questions if the initial coordinates
warrant them.

Before an inferential corpus comparison, record a clean intervention source OID,
the exact dataset slice and baseline measurements, paired exposures, primary
held-out NLL contrast, context-use and frozen-reuse guards, seeds, stopping and
resource bounds, fresh output owner and thresholds in one Experiment Card.
Reconstruction NLL alone cannot establish reusable local structure. Scalar versus
smooth basis capacity, chart-selection priors, the supplied event-time skeleton
and teacher forcing remain explicit alternative explanations.

Do not infer player demand, calibrated motor thresholds, improved Trill detection
or a need for persistent slots from these software checks. The proposed research
direction has no acceptance or implemented lifecycle transition.
