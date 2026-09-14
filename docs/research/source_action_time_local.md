# Source-action time coordinates and local composition

The early-representation comparison separates time coordinates, within-row
nonlinearity, event adjacency and direct source access. It retains the
[composition/access baseline](source_action_stage2.md), its six readable levels,
relation attention, contextual mixer, readers and
[prediction objective](source_action_objective.md).

These are implemented research alternatives. Contract tests establish input
isolation, support, symmetry and executable updates; they do not establish
better held-out prediction or semantic reuse. The scalar and smooth alternatives
receive the same supplied information. Additional projection or operator
parameters remain a capacity confound in any empirical comparison.

## Time coordinates

[`time_basis.py`](../../src/pulsefield_model/research/source_action_modeling/time_basis.py)
uses the supplied source-event skeleton, including hidden and release-only
events. Synthetic boundaries do not define an event gap. It retains physical
seconds and a compressed scalar coordinate:

$$
\phi_{\mathrm{scalar}}(d)
=\left(d/1000,\ \operatorname{asinh}(d/1000)\right),
$$

where $d$ is in milliseconds. The smooth alternative appends signed and even
responses at $\tau\in\{8,16,32,64,128,256,512,1024,2048,4096\}$ milliseconds:

$$
\phi_{\tau}(d)
=\left(\frac{d}{\sqrt{d^2+\tau^2}},
        \frac{\tau}{\sqrt{d^2+\tau^2}}\right).
$$

The signed channel has derivative $1/\tau$ at zero, giving short intervals
distinct sensitivity scales. Differences of scale responses can describe
interval bands. There is no hard timing bin or periodic repetition, and the
physical scalar remains distinct after the bounded channels saturate. These
fixed scales specify numerical coordinates, not calibrated motor ability.
The basis changes inputs; loss weighting remains unchanged.

For an event, $g_-$ and $g_+$ are the preceding and following source gaps. Local
pace $p$ is the arithmetic mean of its available positive gaps. Row-time inputs
contain both gap bases, $\log(g_-/p)$, $\log(g_+/p)$ and two availability flags.
Missing gaps have zero features and false availability. Pair conditions contain
$\phi(t_j-t_i)$, $\operatorname{asinh}((t_j-t_i)/p_i)$,
$\log(p_j/p_i)$ and both pace-availability flags. A uniform tempo rescaling
changes the physical coordinates while preserving these ratios.

Pace depends on one adjacent source event on each side, including hidden events
whose times were supplied. It is not an attack statistic. The prepared input
does not expose a musical timing map, so this implementation introduces no BPM
or beat-phase channel. Recovering either from event density would confound
musical tempo with articulation and subdivision. A future timing-map adapter
must declare that additional information and its visibility policy explicitly.

[Numerical feature embeddings](https://arxiv.org/abs/2203.05556) motivate testing
coordinates separately from the backbone; that paper studies piecewise-linear
and periodic embeddings for tabular tasks. The smooth saturating basis here is
a task-specific alternative. Periodic features remain a possible control, but
are not assumed to supply musical phase when their input is elapsed time.

## Source packet and state conditions

[`local_representation.py`](../../src/pulsefield_model/research/source_action_modeling/local_representation.py)
forms a `SourcePacket` with distinct components:

| Component | Contents | Dependency |
| --- | --- | --- |
| `row_facts` | Own and other hand's tap/head/close bits and availability, preserving outer/inner roles | Current four-lane row |
| `row_metadata` | Supplied position, source/section flags and boundary metadata | Current row and declared scope/context; preceding source gap in event modes |
| `time` | Physical gap bases, relative gap ratios and availability | Supplied skeleton within one source event on either side |
| `state_before` | Own and other hand's occupation value and availability | Declared context-entry state and visible prefix |

Unknown action rows remain distinguishable from observed silence. Every packet
is derived after masking. Occupation comes from the existing forward visibility
replay; no complete-chart LN endpoint, duration or next attack is copied.

`U` starts with the baseline full-row projection. Scalar/smooth arms add a
separate linear projection of row-time coordinates. The `row_interaction` switch
adds a `64 → 128 → 64` residual MLP over that result, allowing another nonlinear
interaction of the complete row and time. It introduces no cross-row action
dependency. Exact source facts remain in the observation; no learned projection
is claimed to be lossless.

`source_skip` supplies the unnormalized 27-channel facts/metadata packet directly
to each local block's value/gate drive. It bypasses both the learned `U` encoding
and normalization of the accumulated local content. This tests whether repeated
access to source coordinates aids composition, separately from retaining `U`
in the reader's bank.

`state_condition` adds an independent projection of `state_before` to each local
block. Such a block is local **conditional on the supplied state**. Its output
also depends on the prefix used to derive that state; it is not strictly local
in source actions. `U` excludes occupation in every arm. After-state,
previous/next attack intervals and far summaries still enter through `H` only.

## Event adjacency and conditional kernels

The `event` operator gathers real source rows in supplied order and applies the
same three width-three convolutions, with dilations 1, 2 and 4. The first event
has no preceding event gap. Boundary states remain readable copies of their `U`
metadata, and results scatter back to the original timeline for existing
relations and readers. No target action determines which rows are gathered.

Consequently `L1`, `L2`, and `L3` cover at most 3, 7, and 15 **source events**.
They have no fixed millisecond radius and are not attack-count windows.
Inserting an otherwise inert synthetic row preserves event-anchored `U/L`
states in evaluation mode. This does not assert invariance of `R/H`, the reader,
or an observation policy whose near radius is measured in timeline rows.

The `time` operator keeps those neighborhoods and offset-specific linear maps.
For offset $k$ its 128-channel drive is modulated before the baseline
`tanh(value) * sigmoid(gate)` activation:

$$
m_i=b+\sum_{k\in\{-d,0,d\}}
\left[1+2\tanh G\left(\psi(t_{i+k}-t_i),F_i,F_{i+k}\right)\right]
\odot W_k\operatorname{LN}(z_{i+k}).
$$

Here $F$ is the full own/other action-and-availability packet and $\psi$ includes
physical and relative time. A shared `32`-hidden-unit MLP conditions on both
endpoint rows, so timing sensitivity can differ for lane reuse, hand changes
and press/release patterns. Endpoint co-occurrence does not certify immediate
attack succession across hidden rows. The gate can change sign; it parameterizes
a diagonal modulation of each learned map, not an arbitrary dense kernel for
every pair. Learned content retains per-row LayerNorm and an identity residual.

[CKConv](https://arxiv.org/abs/2102.02611) supplies the closest sequential
analogue for kernels depending on continuous coordinates, including irregular
sampling. This implementation uses a bounded event neighborhood and also
conditions on observed actions and availability. It is an adaptation of that
mechanism, not a CKConv reproduction or evidence that it is optimal here.

`local_support_report(observation, config)` reports action, time and state
dependency bounds for `U/L1/L2/L3`. The time envelope includes the extra adjacent
gaps used by pace. The state envelope conservatively includes the prefix before
each consumed condition. `row_indices` unions action and state dependencies;
`scope_context_condition` and `entering_occupancy_condition` identify supplied
conditions. `action_source_events` and `action_duration_ms` describe the content
support separately from those additional conditions. Existing `support_report`
describes the original timeline baseline and must not be used to label the event
operators. Relation/context support
remains governed by the existing graph and full-context mixer.

## Controlled comparisons

[`representation_experiments.py`](../../src/pulsefield_model/research/source_action_modeling/representation_experiments.py)
provides independent configuration switches and named controls. These names are
experiment configurations, not a new definition of the earlier implementation
stages.

| Arm | Intervention from the named control | Total parameters |
| --- | --- | ---: |
| `baseline` | Exact `composed_all` initialization and computation | 214,449 |
| `event` | `baseline`: source-event adjacency and preceding source gap | 214,449 |
| `scalar` | `event`: physical gap scalars and relative coordinates | 214,961 |
| `smooth` | `scalar`: smooth basis expansion of the same information | 217,521 |
| `row` | `smooth`: residual row interaction | 234,097 |
| `time` | `smooth`: pair-conditioned local kernel | 235,857 |
| `time_kernel` | `time`: zero action/availability inputs to the kernel gate | 235,857 |
| `action_kernel` | `time`: zero physical/relative time inputs to the kernel gate | 235,857 |
| `constant_kernel` | `time`: zero both gate input groups | 235,857 |
| `row_time` | `smooth`: both row interaction and conditioned kernel | 252,433 |
| `combined` | `row_time`: raw source access in every local block | 262,801 |
| `state` | `combined`: visible-prefix occupation conditions | 265,873 |

Counts use default `ModelConfig`. All arms retain the same 17,484-parameter
action reader and 25,313-parameter decoder. The event control has identical
capacity to the baseline. The smooth/row/time/row_time square separates row
composition, temporal conditioning and their combination. Compare combined
against row_time before attributing a gain to raw access; compare state against
combined before attributing a gain to occupation.

The kernel controls keep exactly the same parameter tensors and initialization,
replacing only the selected gate inputs with zeros. They separate direct timing,
endpoint-action conditioning and constant modulation. Time and actions remain
available through `U`; these are interventions on a computation path. Zeroing
inputs also reduces the gate's effective degrees of freedom, so equal tensor
counts do not establish equal functional capacity.

`initialize_representation_comparison(arms, config, seed, access="all")` requires
an explicit selected subset. It copies every unchanged tensor from the baseline,
including the decoder, reader, relation attention and local convolutions. Added
modules have stable seeds per module path, independent of the selected arm
order. Changing a basis dimension necessarily changes that projection's
initialization. The caller's CPU RNG stream is preserved. `access="H"` supplies
the same backbone controls with contextual-only reading.

`train_representation_step` uses the existing paired sampler and update
implementation. Models receive identical blocks, views, target prefixes and
legality, with the existing equal-block mean-row objective. Select the same
optimizer settings and reader access. `evaluate_structure` reports sequence and
mean-row NLL, first/later positions, scale/duration strata, group uncertainty,
parameter counts and retained bank bytes. Snapshots and update diagnostics
include the complete representation policy. `fit_matched_probes` reconstructs
the selected untrained architecture and access before testing frozen reuse.

Use the Python APIs with a declared training population and fixed evaluation
manifest; the existing access pilot continues to select its original three
arms. A small first comparison can use event/scalar/smooth, followed by the
four-arm computation square if the coordinate change warrants further study:

```python
import torch
from pulsefield_model.research.source_action_modeling.representation_experiments import (
    initialize_representation_comparison, representation_arms, train_representation_step,
)

catalog = representation_arms()
models = initialize_representation_comparison(
    {name: catalog[name] for name in ("event", "scalar", "smooth")}, seed=17,
)
models = {name: model.to("mps") for name, model in models.items()}
optimizers = {name: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
              for name, model in models.items()}
# sampler is the declared population's PairedBlockSampler.
record = train_representation_step(models, optimizers, sampler, blocks=4)
```

Fix population, seeds, exposure, stopping budget, held-out metric and semantic
guards before a corpus comparison. A lower reconstruction loss alone does not
establish useful local computation; retain matched-context gains and trained
versus untrained frozen probes. Capacity, decoder prefix use and the fixed event
skeleton remain alternative explanations. Persistent slots, `L → R → L`, new
query formation, calibrated motor priors and loss reweighting are separate
interventions.

## Verification

The focused checks exercise identical hidden inputs across all three views,
hand exchange for both readers, padding with learned normalization bias,
source-event support and marker insertion, distinct action/time/state
dependencies, physical versus ratio response under rescaling, and pair-time
effects with content held fixed. They also check direct source access with `U`
removed, finite gradients, empty event timelines, matched frozen probes and
exact CPU checkpoint continuation.

```sh
uv run --offline --extra mps --group dev pytest -q \
  tests/research/source_action_modeling \
  tests/research/scoped_style_modeling/test_replay.py \
  tests/research/scoped_style_modeling/test_model.py
git diff --check
```

Device-parametrized tests exercise CPU and MPS when available; CUDA requires
available NVIDIA hardware and `--extra cuda`. Exact input equality is required
on every device. MPS forward comparisons allow floating-point reduction
roundoff without weakening input-isolation checks.
