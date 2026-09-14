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

## Result Log: Concrete untrained forward examples

### Reproduction and scope

- Date: 2026-09-14. The owner requested concrete examples explaining the model
  changes and the scope of valid claims.
- Accepted revision and Experiment Card: none. This is an explanatory forward
  demonstration on synthetic fixtures, with zero optimizer updates or corpus
  evaluations. It is not a result supporting predictive superiority.
- Product HEAD: `84b02a3037d26dae301854bbf9cf0717edd907a7` with the uncommitted
  implementation described above. Tracked-diff SHA-256:
  `09f2e821b62ebb239b9ba3086edb13a7ec963972b52ee23819a64ded43a52868`.
  The result artifact includes hashes of the seven directly used source owners.
- Artifact owner: `artifacts/source-action-modeling/time-local-examples/example-ZjY6Id/`.
  `examples.py` contains all source-exact fixtures, assertions and interventions;
  `results.json` contains the measured values and source identities.
- Script SHA-256:
  `e2be2aff9ca34fcaed783eeebe4b5d7f3aa55cb5eceb669523d5a756e955b5ec`.
- Command: `uv run --offline --extra mps --group dev python
  artifacts/source-action-modeling/time-local-examples/example-ZjY6Id/examples.py`.
- Environment: CPU, one Torch thread, Python 3.10.20, PyTorch 2.11.0. Model seed
  17; the frozen kernel-content tensor uses a private seed 901. All models are
  untrained and evaluated without dropout. Runtime was approximately 0.64 seconds.
  Results were exclusively created in a fresh directory; rerunning requires a
  fresh copy of the script/output location.

### Observations

Hidden-state differences below are maximum absolute channel differences, not
NLL, accuracy, importance or comparable quality scores across architectures.

1. Source events at 0, 100 and 200 ms, with an otherwise inert boundary row
   inserted at 150 ms and unchanged scope/context: original timeline L1/L3
   differences were 0.186062/0.352876. The event and combined branches had exact
   zero differences at every source-anchored U/L1/L2/L3 state. Event-branch R/H
   differences remained 0.206778/0.320954; whole-model invariance is false.
2. Holding all learned local content fixed, doubling physical event gaps and
   independently changing a same-lane endpoint into a same-hand lane switch gave:

   | Kernel condition | Time-change difference | Lane-change difference | Difference of time effects across lane conditions |
   | --- | ---: | ---: | ---: |
   | Constant | 0 | 0 | 0 |
   | Time only | 0.007658 | 0 | 0 |
   | Actions only | 0 | 0.019178 | 0 |
   | Time and actions | 0.006735 | 0.020073 | 0.001357 |

   All four gates shared their parameter tensors and initialization. The
   nonzero interaction shows that timing sensitivity can depend on endpoint
   actions. It supplies no evidence that the untrained preference is useful.
3. With learned local content set to zero, changing a tap into an LN head at
   the same event changed a raw-access block by 0.115891. This isolates the
   direct fact route; it is not a proof of lossless fact preservation.
4. With source events every 100 ms, L3 at 1800 ms had action support 1100–2500
   ms and time support 1000–2600 ms. Changing a tap at 0 ms into a hold lasting
   beyond the context left combined U/L1/L2/L3 unchanged. With state conditions,
   U stayed unchanged while L1/L3 changed by 0.119453/0.237513. Occupation thus
   has a real prefix dependency and cannot be labeled intrinsic local content.
5. Middle-event gap pairs 40/80 and 80/160 ms had identical relative coordinates
   `(-0.405465, 0.287682)`, while 80/80 ms gave `(0, 0)`. Physical coordinates
   still differed between the first two cases. A two-weight linear combination
   of the even 256-ms and 64-ms bases produced a broad interval-band response:
   0 at 0 ms, 0.097851 at 32 ms, 0.447214 at 128 ms, 0.464571 at 256 ms and
   0.046755 at 4096 ms.

### Interpretation

The examples support explicit dependencies and invariances of the implemented
computations. A single affine function of signed log time cannot reproduce the
nonmonotonic band above, but the original network's nonlinear layers could
learn such a response. Basis expansion adds coordinates, not new observations;
its value for finite-budget learning remains unmeasured. The old whole encoder
already read time and full row actions. The additional local gates and raw route
make these dependencies directly available at specified operations.

No trained comparison, new model adoption or Note lifecycle transition follows.
The previously recorded REFINE recommendation and corpus evidence gaps remain.
