# Canonical gameplay frontier, demand state, and style semantics

Pulsefield uses one fixed canonical gameplay profile to describe how a legal
4K chart changes the gameplay situation over time.

The central chart-derived objects are

\[
x_H(t)
=
\operatorname{Replay}_0(H;t),
\qquad
d_H(t)
=
\operatorname{DemandRoll}_0(H;t),
\]

followed, for section \(W\), by

\[
\left(
\chi_H(W),
d_H(a^-),
d_H|_W
\right)
\xrightarrow{\Psi}
r_H(W)
\xrightarrow{\operatorname{StyleRead}}
z_H(W).
\]

Here:

- \(H\) is the materialized chart and source of truth for chart actions;
- \(x_H(t)\) is exact chart and control state;
- \(d_H(t)\) is a profile-relative representation of the moving gameplay
  frontier;
- \(\chi_H(W)\) is exact section chart evidence;
- \(r_H(W)\) is a section-level action-demand geometry;
- \(z_H(W)\) is the modeled section-style profile;
- \(o_{H,W}^{\mathrm{sec}}\) is a section-level style annotation;
- \(o_H^{\mathrm{map}}\) is an optional community map-level observation.

On this page, gameplay style denotes a versioned Pulsefield vocabulary of
player- and mapper-facing concepts used to describe chart organization, such
as drill, jack, technical patterning, stream topology, and long-note
organization.

The current style module is restricted to concepts whose evidence is determined
by the materialized chart, its causal history, and the fixed canonical gameplay
profile over a declared interval. Its initial scope excludes concepts that
require aligned audio, external assets, provenance, or unobserved mapper intent.

The Pulsefield style vocabulary is not identified with the complete osu!
community-tag catalogue. Compatible community tags may provide weak
observations for selected concepts, but they do not define the coordinates of
the modeled style profile.

The notation extends [notation.md](notation.md).

## 1. One canonical gameplay profile

Pulsefield currently models exactly one gameplay profile, denoted

\[
\pi_0.
\]

The profile fixes:

- the lane-to-hand-role mapping \(\lambda\);
- successful execution of every legal action;
- ideal hit timing;
- the semantics and dynamics of gameplay-demand state;
- left-right symmetry conventions;
- calibration references for named demand dimensions.

There is no player-profile variable in the current formulation.

The current scope does not model:

- player-specific capacity;
- player errors or misses;
- timing noise;
- adaptive fingering;
- alternative physical executions;
- handedness-specific profiles;
- multiplayer interaction;
- physiological fatigue or subjective pain.

The canonical gameplay state is a reproducible chart descriptor under
\(\pi_0\). It is not a claim about the hidden internal state of an observed
person.

## 2. Epistemic layers

The formulation separates the following objects.

| Object | Status | Meaning |
| --- | --- | --- |
| \(H\) | Observed or generated symbolic object | The materialized chart and source of truth for chart actions. |
| \(x_H(t)\) | Exact | Replayable chart-format and exact control facts. |
| \(\chi_H(W)\) | Exact derived view | Exact entry state and materialized chart rows over a section. |
| \(\mathcal F_H(t)\) | Abstract profile-relative object | The current response surface over possible future actions and continuations. |
| \(d_H(t)\) | Model-defined but reproducible under \(\pi_0\) | A finite representation of the moving gameplay frontier. |
| \(r_H(W)\) | Derived section representation | Geometry of actions and demand over a section. |
| \(z_H(W)\) | Modeled semantic profile | Salience of versioned Pulsefield gameplay-style concepts in a section. |
| \(o_{H,W}^{\mathrm{sec}}\) | Partial annotation | Human-confirmed section-level style evidence. |
| \(o_H^{\mathrm{map}}\) | Optional weak external observation | Community-voted map-level evidence for declared compatible concepts. |
| \(d^\star(t)\) | Optional request | Desired gameplay-demand target for generation. |
| \(c^{\mathrm{style}}\) | Optional request | Desired style tendency or mixture for generation. |

Exact chart measurements are deterministic functions of \(H\) and \(x_H\).
They may be expressed in serialized lane coordinates or after the row-wise
hand-role projection \(\operatorname{Act}_\lambda\). They are not another
epistemic layer and are not semantic labels.

A gameplay mechanism is an intervention-grounded regularity in how exact chart
evidence changes frontier responses. It is not an additional hidden state
between exact chart state and demand state.

The hard semantic separations are:

1. Chart legality belongs to exact chart-format state.
2. Canonical hand-role coordinates are a deterministic view of chart rows, not
   a separate execution state or latent action sequence.
3. Gameplay demand belongs to the profile-relative moving frontier.
4. Section style belongs to the geometry of chart actions and demand over time.
5. Section annotations and compatible community tags are sparse semantic
   observations, not exact chart state.
6. Compatible map-level tags are optional weak external observations, not
   labels for every section.
7. Desired style and desired demand are optional generation controls.
8. Realized style and realized demand are consequences of the generated chart.
9. Missing annotation is not automatically a negative annotation.
10. Neither demand state nor style tag is an observed player's hidden state.

## 3. Exact hand-role projection and state

For row \(m_k\), let

\[
a_k
=
\operatorname{Act}_\lambda(m_k)
=
(a_k^L,a_k^R)
\]

be its deterministic canonical hand-role view, as defined in
[notation.md](notation.md#4-lanes-hands-and-canonical-roles).

The operator \(\operatorname{Act}_\lambda\) changes coordinates only. Under the
fixed 4K mapping, \(a_k\) contains no information beyond \(m_k\), does not infer
fingering or physical execution, and is computed from a row whenever hand-role
coordinates are useful. The materialized chart \(H\) remains the sequence-level
source of truth; no separate global execution trace is required.

Write the exact state as

\[
x_H(t)
=
\left(
x_H^{\mathrm{fmt}}(t),
x_H^{\mathrm{ctrl}}(t)
\right).
\]

### Chart-format state

The format state contains every fact required for hard legality:

\[
x_H^{\mathrm{fmt}}(t)
=
\left(
\omega_L(t),
\omega_R(t),
\xi^{\mathrm{fmt}}(t)
\right),
\]

where

\[
\omega_h(t)
=
\left(
\omega_{h,\mathrm{outer}}(t),
\omega_{h,\mathrm{inner}}(t)
\right)
\in
\{0,1\}^2
\]

is role-wise long-note occupancy.

The value \(\xi^{\mathrm{fmt}}\) contains any additional exact format
obligations.

### Exact control state

The exact control component may contain:

\[
x_H^{\mathrm{ctrl}}(t)
=
\left(
x_L^{\mathrm{ctrl}}(t),
x_R^{\mathrm{ctrl}}(t),
x_\times(t)
\right).
\]

Admissible exact summaries include:

- the most recent active-role set of each hand;
- per-role action clocks;
- press, close, or release clocks;
- exact recent role-set sequences;
- exact simultaneity or alternation clocks;
- persistent cross-hand facts;
- exact information required to update the declared demand dynamics.

After a one-hand chord, the most recent active-role state is a set:

\[
A_h^{\mathrm{last}}
\in
\left\{
\varnothing,
\{\mathrm{outer}\},
\{\mathrm{inner}\},
\{\mathrm{outer},\mathrm{inner}\}
\right\}.
\]

It is not a probability distribution over which role was used.

For row \(y_k=(t_k,m_k)\), exact state updates as

\[
x_k^+
=
E_{\mathrm{exact}}
\left(
x_k^-,
y_k
\right).
\]

The row is legal if and only if the chart-format transition is defined:

\[
\operatorname{legal}(x_k^-,y_k)
\iff
E_{\mathrm{fmt}}
\left(
x_k^{\mathrm{fmt},-},
m_k
\right)
\text{ is defined}.
\]

Demand and style cannot alter this predicate.

## 4. The moving gameplay frontier

The generation-relevant gameplay object is not only the current value of one
difficulty scalar. It is the changing response surface over possible future
actions.

Let

\[
b_H(t)
=
\left(
t,
x_H(t),
d_H(t)
\right)
\]

be a canonical gameplay boundary.

For a candidate future action \(a\) executed after delay \(\Delta\), define the
one-action frontier

\[
\mathcal F_H(t;a,\Delta)
=
\operatorname{DemandCost}_0
\left(
a
\text{ at }t+\Delta
\mid
b_H(t)
\right).
\]

The result may contain multiple dimensions such as:

- short-horizon burst demand;
- accumulated strain;
- role-reuse pressure;
- outer-inner transition pressure;
- one-hand chord demand;
- sustained long-note control;
- release demand;
- cross-hand coordination.

The frontier is not restricted to one action. For a legal continuation \(Y\)
over horizon \([t,t+L]\), define the continuation-response operator

\[
\mathfrak C_0
\left(
b_H(t),
Y
\right)
=
\left.
d_{b_H(t)\oplus Y}
\right|_{[t,t+L]}.
\]

This operator answers:

> Starting from the current canonical gameplay situation, how would the demand
> trajectory evolve under this future continuation?

The abstract moving frontier may therefore be viewed as the map

\[
\mathcal F_H(t)
:
Y
\longmapsto
\mathfrak C_0(b_H(t),Y)
\]

over legal future continuations.

This functional object is too large to use directly. Pulsefield therefore uses
\(d_H(t)\) as a finite, interpretable, profile-relative representation of it.

## 5. Gameplay-demand state

For the fixed profile \(\pi_0\), define

\[
d_H(t)
=
\operatorname{DemandRoll}_0(H;t)
\in
\mathcal D_0.
\]

A typical decomposition is

\[
d_H(t)
=
\left(
d_H^L(t),
d_H^R(t),
d_H^\times(t)
\right),
\]

where:

- \(d_H^L\) and \(d_H^R\) contain corresponding hand-local dimensions;
- \(d_H^\times\) contains non-additive cross-hand dimensions when needed.

Named exposed dimensions are nonnegative unless another interpretation is
explicitly declared:

\[
d_{H,c}(t)\ge0.
\]

The dimensions are not required to be independent, additive, or mutually
exclusive.

### Bounded scope of the finite demand state

The abstract frontier may depend on arbitrarily long chart history. The finite
state \(d_H(t)\) is not required to encode every chart-intrinsic fact, every
recognizable pattern, or every distinction useful for generation.

Its current contract is relative to a declared continuation family
\(\mathcal Y_0\) over horizon \(L_0\), response summaries \(\Omega_0\), and
tolerance \(\varepsilon\):

\[
x_H(t)=x_{H'}(t),\quad d_H(t)=d_{H'}(t)
\]

should imply

\[
\sup_{Y\in\mathcal Y_0}
\left\|
\Omega_0\!\left(\mathfrak C_0(b_H(t),Y)\right)
-
\Omega_0\!\left(\mathfrak C_0(b_{H'}(t),Y)\right)
\right\|
\leq\varepsilon.
\]

A chart-derived distinction is added to the demand state only when controlled
continuation probes show that the distinction changes canonical gameplay
response. Improving style recognition alone is not sufficient justification
for adding a demand coordinate.

The complete history remains available to the chart generator. Demand state is
not required to summarize:

- musical motif identity;
- mapper intent;
- section identity;
- every long-range style pattern;
- all information useful for chart generation.

Its sufficiency contract concerns canonical gameplay continuation response.

### Demand state is not merely current burden

Two states may have the same current scalar intensity but different future
frontiers:

\[
i_H(t)=i_{H'}(t)
\]

while

\[
\mathcal F_H(t)\neq\mathcal F_{H'}(t).
\]

For example, the same current intensity may arise from:

- recent repeated use of left outer;
- a one-hand chord;
- sustained long-note occupancy;
- balanced two-hand density.

Those histories may respond differently to the same next row. The demand state
must preserve distinctions needed for such continuation behavior.

## 6. Hybrid demand dynamics

The canonical gameplay process is a hybrid dynamical system:

- rows cause discrete state transitions;
- time between rows causes recovery, decay, sustained forcing, and clock
  advancement.

Use the right-continuous convention

\[
x_k^-
=
x(t_k^-),
\qquad
x_k^+
=
x(t_k),
\]

and

\[
d_k^-
=
d(t_k^-),
\qquad
d_k^+
=
d(t_k).
\]

### Event transition

For legal row \(y_k=(t_k,m_k)\),

\[
x_k^+
=
E_{\mathrm{exact}}(x_k^-,y_k).
\]

Let

\[
a_k
=
\operatorname{Act}_\lambda(m_k).
\]

Demand state updates as

\[
d_k^+
=
R_0
\left(
d_k^-,
x_k^-,
x_k^+,
a_k
\right).
\]

The same row may produce different updates after different histories.

For example, a tap on right inner may differ depending on whether:

- right inner was recently repeated;
- right outer participated in a recent chord;
- right inner currently sustains a long note;
- the opposite hand acts simultaneously;
- the section is inside a dense burst.

### Between-event flow

For

\[
0<\tau<t_{k+1}-t_k,
\]

exact state evolves as

\[
x(t_k+\tau)
=
F_{\mathrm{exact},\tau}(x_k^+),
\]

and demand state evolves as

\[
d(t_k+\tau)
=
F_{0,\tau}
\left(
d_k^+,
x(t_k+\tau)
\right).
\]

The implementation may use:

- analytic decay kernels;
- discrete state-space updates;
- continuous-time ODEs;
- neural state-space dynamics;
- gated recurrent transitions;
- another causal rollout family.

No particular parameterization is fixed.

### Long-note effects

`LN_START` changes exact occupancy. The demand dynamics may use open occupancy
to produce sustained forcing.

`LN_CLOSE` ends occupancy and may produce a release transient.

These demand effects are properties of the fixed gameplay profile and its
validated dynamics. They are not implied by the chart alphabet alone.

### Isolated note responses

For diagnostics, one may define the response to one action under a declared
no-future-event baseline.

Under nonlinear, history-dependent dynamics, isolated responses are not
generally additive:

\[
d_{H_1\oplus H_2}
\neq
d_{H_1}
+
d_{H_2}.
\]

A realized chart trajectory must be obtained by complete sequential rollout.

## 7. Demand dimensions, intensity, and calibration

The final inventory of named demand dimensions is open.

Exact chart measurements may include:

- inter-event intervals and local event rate;
- same-lane and same-role recurrence;
- alternation-run length and lane-span;
- chord cardinality and lane-set transitions;
- snap and rhythmic-position statistics;
- long-note occupancy;
- head, body, and release topology;
- simultaneous and staggered actions.

These quantities are deterministic chart evidence. They may drive demand
dynamics, but they are not demand dimensions by themselves.

Candidate demand-response families include:

- burst accumulation and recovery;
- same-role reuse response;
- outer-inner transition response;
- one-hand chord-control response;
- sustained long-note control;
- release-transition response;
- cross-hand coordination;
- hand-local imbalance and recovery.

The formulation fixes the provenance of exact measurements, not a final feature
inventory or threshold-based definition of named styles.

A natural-language name does not by itself establish a valid demand dimension.

A named dimension must be anchored by controlled interventions. For example,
same-role repetition pressure should respond predictably when:

- the future action is held fixed;
- the previously active role is changed;
- the inter-action interval is varied;
- chord participation is controlled;
- long-note occupancy is controlled;
- the opposite-hand history is controlled.

A local scalar intensity may be defined as

\[
i_H(t)
=
R_{\mathrm{intensity}}
\left(
d_H(t)
\right).
\]

Section or map difficulty requires a separate aggregation:

\[
\operatorname{Diff}
\left(
H;W
\right)
=
\operatorname{Agg}_{t\in W}
i_H(t).
\]

Neither operation is a raw vector norm unless channel units, interactions, and
calibration justify that choice.

### Representation non-uniqueness

The finite representation of the frontier is not automatically unique.

For an invertible transformation \(T\),

\[
\widetilde d=T(d),
\]

one may transform the dynamics and continuation evaluator while preserving the
same frontier responses.

Therefore:

- frontier behavior is the primary operational object;
- a demand coordinate receives a gameplay name only after semantic anchoring;
- an unanchored learned coordinate remains latent;
- style annotations alone must not be used to circularly define a demand
  channel with the same name.

## 8. Hand symmetry and cross-hand interaction

The canonical `(outer, inner)` coordinates make mirror structure explicit.

For the mirror operator \(\mu\), exact replay satisfies

\[
x_{\mu H}(t)
=
\mu x_H(t).
\]

A mirror-equivariant demand specification satisfies

\[
d_{\mu H}(t)
=
\mu d_H(t).
\]

For a mirrored boundary and continuation,

\[
\mathfrak C_0
\left(
\mu b,
\mu Y
\right)
=
\mu
\mathfrak C_0(b,Y).
\]

Mirror equivariance does not imply hand independence.

A hand-local demand update may depend on:

- the same hand's exact and demand state;
- the opposite hand's exact and demand state;
- the complete simultaneous row;
- persistent cross-hand state;
- symmetric messages exchanged between hand representations.

Parameter sharing with symmetric cross-hand communication is a valid
implementation.

Most gameplay-style tags are expected to be mirror invariant:

\[
z_{\mu H}(W)=z_H(W).
\]

If the final vocabulary contains a genuinely directional tag, its explicit tag
transformation must be declared rather than emerging accidentally from
unrelated left- and right-hand semantics.

## 9. Section gameplay geometry

Let

\[
W=[a,b]
\]

be a chart section.

Because gameplay demand at section entry depends on preceding history, a
section cannot always be represented by rows inside \(W\) alone.

Define the exact section chart evidence

\[
\chi_H(W)
=
\left(
x_H(a^-),
H|_W
\right).
\]

Every exact recurrence, alternation, timing, chord, lane-set, or long-note
measurement used by an implementation is a deterministic function of
\(\chi_H(W)\). The rows \(H|_W\) may be used directly or projected row-wise to
canonical hand-role coordinates with \(\operatorname{Act}_\lambda\).

Define the canonical section record

\[
\mathcal G_H(W)
=
\left(
\chi_H(W),
d_H(a^-),
d_H|_W
\right).
\]

Define the section gameplay geometry

\[
r_H(W)
=
\Psi
\left(
\mathcal G_H(W)
\right).
\]

A gameplay mechanism is not an additional representation layer. It names a
validated regularity in how controlled changes to \(\chi_H(W)\), entry state,
or continuation actions change the frontier and demand rollout.

The representation \(r_H(W)\) may summarize:

- which demand dimensions rise;
- the actions that cause them to rise;
- how quickly they rise;
- how long they remain elevated;
- how they recover;
- whether peaks repeat;
- whether mechanisms alternate or overlap;
- which hand and role combinations are used;
- how long-note occupancy interacts with free-role actions;
- how the two hands coordinate;
- how similar demand islands recur through the section.

The operator \(\Psi\) is not fixed to one neural architecture or hand-designed
feature set. Its semantic responsibility is fixed: it represents the temporal
geometry of how concrete chart actions drive the canonical gameplay frontier.

## 10. Section style and style tags

Let the versioned style vocabulary be

\[
\mathcal K_{\mathrm{style}}
=
\{k_1,\ldots,k_K\},
\]

where \(K\) is the versioned size of the curated Pulsefield gameplay-style
vocabulary.

The vocabulary is selected from concepts that players and mappers use to
describe or request chart organization. It is not required to follow osu!
catalogue namespaces.

Style coordinates may overlap, correlate, and share the same exact chart
measurements or demand mechanisms. They are semantic predicates, not an
orthogonal generative basis.

Every coordinate is read directly from section evidence:

\[
z_{H,k}(W)
=
R_k
\left(
r_H(W)
\right).
\]

The formulation does not define one style coordinate as a function of another
style coordinate and does not introduce a first-order versus second-order
concept hierarchy.

For section \(W\), define the modeled style profile

\[
z_H(W)
=
\left(
z_{H,1}(W),
\ldots,
z_{H,K}(W)
\right)
\in
[0,1]^K.
\]

The value \(z_{H,k}(W)\) represents the salience or applicability of style
concept \(k\) in that section.

The coordinates are not required to sum to one:

\[
\sum_{k=1}^{K}
z_{H,k}(W)
\neq 1
\quad\text{in general}.
\]

The vocabulary is therefore multi-label rather than one-of-\(K\).

A section may:

- have one dominant style;
- combine several styles;
- transition between styles;
- weakly express several concepts;
- contain no especially salient named style.

The absence of a strong named tag does not imply the absence of concrete
gameplay organization.

### Style as section-level action-demand geometry

The central style hypothesis is

\[
z_H(W)
=
\operatorname{StyleRead}
\left(
r_H(W)
\right).
\]

In words:

> A style tag names a salient, recurring, player- and mapper-recognizable region
> or predicate in the space of section action-demand geometries.

This does not require the style concepts to form disjoint clusters. Their
regions may overlap.

A tag may depend on:

- demand trajectory morphology;
- action identity;
- role sequence;
- timing organization;
- chord and long-note topology;
- incoming frontier state;
- transitions between demand mechanisms.

Style is therefore not merely the value of \(d_H(t)\) at one time.

More strongly,

\[
z_H(W)
\not\equiv
d_H(t)
\]

and

\[
z_H(W)
\not\equiv
\operatorname{Agg}_{t\in W}d_H(t).
\]

Two sections may have similar aggregate demand and different styles. The same
style may appear at different demand intensities.

### Parallel sources of section evidence

The intended structure has two parallel sources of section evidence:

\[
H
\longrightarrow
\chi_H(W),
\]

and

\[
H
\longrightarrow
\mathcal F_H(t)
\approx
d_H(t).
\]

They meet at the section representation:

\[
\left(
\chi_H(W),
d_H(a^-),
d_H|_W
\right)
\longrightarrow
r_H(W)
\longrightarrow
z_H(W).
\]

Demand is the main profile-relative representation of gameplay continuation
response. It is not a universal information bottleneck between chart
arrangement and style.

Exact chart arrangement reaches the style readout in parallel. A style
distinction that remains in exact chart evidence is not, by itself, evidence
that the demand state is insufficient.

### Namespaced semantics

Demand dimensions and style tags may share natural-language roots, but they are
different formal objects.

For example:

\[
d_{\mathrm{burst}}(t)
\]

may denote a continuous short-horizon demand dimension, while

\[
\texttt{STYLE\_BURST}
\]

denotes a section-level semantic tag.

The style tag may require:

- a sufficiently salient burst-demand island;
- a characteristic rise and recovery shape;
- a minimum duration or repetition condition;
- corresponding concrete note organization.

A high instantaneous value of \(d_{\mathrm{burst}}(t)\) does not by itself make
the whole section `STYLE_BURST`.

Likewise, a style tag must not be used as the sole definition of a demand
channel with the same name. Demand channels require independent continuation
and intervention grounding.

## 11. What should demand explain about style?

Demand-only sufficiency is not a target requirement of the current
formulation.

Some gameplay-style concepts are primarily identified by exact arrangement;
others additionally depend on history-conditioned gameplay response. The
purpose of the demand representation is to expose the latter connection, not
to absorb every chart-intrinsic distinction that is useful for style
recognition.

A strong demand-only hypothesis would be

\[
z_H(W)
\perp
\chi_H(W)
\mid
\left(
d_H(a^-),
d_H|_W
\right).
\]

Equivalently, after observing the incoming and section demand trajectory, exact
chart evidence would add no further information about style.

This may be false.

Some style concepts may retain residual dependence on:

- lane and role grammar;
- repeated symbolic motifs;
- long-note topology;
- rhythmic organization;
- pattern identity;
- arrangement conventions not fully represented by current demand channels.

The more conservative formulation is

\[
z_H(W)
=
\operatorname{StyleRead}
\left(
r_H(W)
\right).
\]

A direct test compares:

1. a style predictor from demand trajectory only;
2. a style predictor from exact chart evidence only;
3. a predictor from exact chart evidence and demand trajectory;
4. matched counterfactual sections with similar demand and different
   arrangement.

If the combined predictor consistently outperforms the demand-only predictor,
this does not by itself indicate that \(d\) is deficient.

The demand state should be revised only when the residual chart distinction also
produces a continuation-response difference under controlled probes.

## 12. Section-level style annotation

For each annotated section \(W\), define

\[
o_{H,W}^{\mathrm{sec}}
=
\left(
o_{H,W,1}^{\mathrm{sec}},
\ldots,
o_{H,W,K}^{\mathrm{sec}}
\right),
\]

with

\[
o_{H,W,k}^{\mathrm{sec}}
\in
\{1,0,?\}.
\]

The values mean:

- \(1\): the annotator confirms style \(k\) is present;
- \(0\): the annotator explicitly confirms style \(k\) is absent;
- \(?\): the label is unobserved or not judged.

Beatmap-lens section annotation provides high-resolution supervision of
\(z_H(W)\).

Section annotations supervise gameplay-style concept salience. Annotators are
not asked to label exact chart measurements such as recurrence counts,
alternation runs, chord cardinality, or long-note occupancy; those quantities
are computed directly from the chart.

The annotation vocabulary is \(\mathcal K_{\mathrm{style}}\), not the complete
community-tag catalogue.

The annotation workflow must declare whether it is:

- **positive-only**: annotators select confirmed present tags;
- **exhaustive**: annotators judge every tag as present or absent;
- **partially exhaustive**: only some tag dimensions receive explicit negative
  judgments.

Unless an explicit negative judgment is recorded,

\[
\text{unmarked}
\neq
\text{absent}.
\]

Section annotations may themselves be noisy or ambiguous. They are semantic
observations of style, not exact chart facts.

An observation model may be written as

\[
o_{H,W}^{\mathrm{sec}}
\sim
p_{\eta,\mathrm{sec}}
\left(
o
\mid
z_H(W)
\right).
\]

## 13. Map-level community style tags

The Pulsefield gameplay-style vocabulary and the osu! community-tag catalogue
are not assumed to be identical.

Let \(\mathcal K_{\mathrm{community}}\) denote the versioned osu! community-tag
catalogue.

Let

\[
\mathcal A
\subseteq
\mathcal K_{\mathrm{community}}
\times
\mathcal K_{\mathrm{style}}
\]

be a versioned set of declared weak-supervision alignments.

Only an aligned community tag may provide weak evidence for a Pulsefield style
concept. The alignment records the expected scope and pooling semantics. No
section-level or map-level claim is made for unaligned community tags.

Let

\[
o_H^{\mathrm{map}}
\]

denote community-voted map-level style evidence.

When vote counts or ratios are available, preserve them as

\[
v_{H,c}^{\mathrm{map}},
\qquad
c\in\mathcal K_{\mathrm{community}}.
\]

A thresholded enum loses information about confidence and disagreement and
should be treated as a derived representation.

Partition a map into sections

\[
W_1,\ldots,W_M.
\]

For an alignment \((c,k)\in\mathcal A\), define map-level salience as

\[
\zeta_{H,c\leftarrow k}^{\mathrm{map}}
=
\operatorname{Pool}_{c,k}
\left(
\{
z_{H,k}(W_m),
|W_m|,
q_m
\}_{m=1}^{M}
\right),
\]

where \(q_m\) may contain section prominence, confidence, or structural
importance.

The pooling operator need not be a mean. An aligned Pulsefield style concept
may support a community map-level tag because it:

- appears in a large fraction of the map;
- persists for a long duration;
- recurs across several sections;
- dominates a prominent climax;
- is unusually salient despite occupying a shorter interval.

Aligned community evidence is modeled as

\[
o_{H,c}^{\mathrm{map}}
\sim
p_{\eta,\mathrm{map}}
\left(
o
\mid
\left\{
\zeta_{H,c\leftarrow k}^{\mathrm{map}}
:
(c,k)\in\mathcal A
\right\}
\right).
\]

An aligned map-level tag does not imply

\[
z_{H,k}(W_m)=1
\qquad
\text{for every }m.
\]

Therefore aligned community map tags must not be copied onto every section as
local ground truth.

The supervision hierarchy is:

\[
\text{declared-compatible community map tags}
\rightarrow
\text{optional weak aggregate supervision},
\]

\[
\text{beatmap-lens section tags}
\rightarrow
\text{strong local supervision}.
\]

Charts without annotations still possess concrete gameplay geometry and
realized style. Their style labels are merely unobserved.

## 14. Learning the meaning of style tokens

A style token is considered semantically learned only when both recognition and
intervention contracts hold.

### Recognition contract

For a section manually confirmed to contain style \(k\), the style recognizer

\[
q_\eta
\left(
z_{H,k}(W)
\mid
\chi_H(W),
d_H(a^-),
d_H|_W
\right)
\]

should assign high salience to \(k\), with calibration appropriate to annotation
uncertainty.

Recognition must be evaluated separately on:

- section-level human annotations;
- declared-compatible map-level community tags after section pooling;
- held-out songs and mappers;
- sections with mixed styles;
- negative and unknown labels.

### Intervention contract

For fixed audio, committed history, and compatible demand conditions, changing
the requested style should change the generated chart distribution in the
corresponding semantic direction:

\[
p_\theta
\left(
Y_W
\mid
X,H_k,c_a^{\mathrm{style}},d_W^\star
\right)
\neq
p_\theta
\left(
Y_W
\mid
X,H_k,c_b^{\mathrm{style}},d_W^\star
\right).
\]

The difference should be recognized as the requested style by held-out human
annotations or an independently validated style recognizer.

It is insufficient for a style token merely to change:

- note count;
- global difficulty;
- decoding entropy;
- one superficial token frequency;
- memorized mapper identity.

A token's meaning is grounded by

\[
\text{recognition from charts}
+
\text{controlled effect on generated charts}.
\]

Explicit style modeling does not require a separate style planner. A model may
generate chart rows directly while learning style through:

- conditional tokens;
- an auxiliary recognition objective;
- contrastive section supervision;
- latent internal features;
- branch reranking;
- another architecture consistent with the same semantics.

## 15. Realized style, requested style, realized demand, and desired demand

The following four objects must remain distinct.

| Object | Meaning |
| --- | --- |
| \(z_H(W)\) | Style realized by a materialized section. |
| \(c_W^{\mathrm{style}}\) | Optional request to favor a style or mixture. |
| \(d_H(t)\) | Demand state induced by a materialized chart. |
| \(d_W^\star(t)\) | Optional desired demand trajectory. |

### Requested style

A style request may be represented as

\[
c_W^{\mathrm{style}}
=
(w_W,\gamma_W),
\]

where

\[
w_W\in[0,1]^K
\]

is a requested multi-tag profile and

\[
\gamma_W\ge0
\]

is adherence strength.

A single requested enum is a one-hot special case.

The coordinates of \(w_W\) need not sum to one when mixed styles are allowed.

The coordinates of a named-style request are semantic predicates, not assumed
independent generative factors. Requesting one named concept does not assert
the absence of correlated concepts or mechanisms.

The vector form \((w_W,\gamma_W)\) is one coarse interface to the abstract style
request space. It does not prevent future structured, exemplar-based, or
mechanic-oriented controls, and those controls need not be added until their
axes are empirically validated.

When no request is supplied,

\[
c_W^{\mathrm{style}}=\varnothing.
\]

This means “allow the model's natural style distribution,” not a separate
`STYLE_NONE` class.

### Desired demand

A desired demand trajectory is

\[
d_W^\star(t),
\qquad
t\in W.
\]

A simpler intensity control is

\[
i_W^\star(t).
\]

When no desired demand is supplied, the generator does not receive a mandatory
demand plan. Every output still induces

\[
d_{H_k\oplus Y_W}(t).
\]

### Four generation cases

Without either control:

\[
Y_W
\sim
p_\theta
\left(
Y_W
\mid
X,H_k
\right).
\]

With style only:

\[
Y_W
\sim
p_\theta
\left(
Y_W
\mid
X,H_k,c_W^{\mathrm{style}}
\right).
\]

With demand only:

\[
Y_W
\sim
p_\theta
\left(
Y_W
\mid
X,H_k,d_W^\star
\right).
\]

With both:

\[
Y_W
\sim
p_\theta
\left(
Y_W
\mid
X,H_k,c_W^{\mathrm{style}},d_W^\star
\right).
\]

In the style-only case, the model chooses a compatible demand trajectory.

In the demand-only case, the model may realize the target through several
different styles.

In the joint case, it searches for a legal continuation in the overlap between
the requested style region and demand target.

The two controls may conflict. Neither is assumed always feasible.

### Abstract reweighting semantics

An optional control interpretation is

\[
p
\left(
Y_W
\mid
X,H_k,c_W^{\mathrm{style}},d_W^\star
\right)
\propto
p_0
\left(
Y_W
\mid
X,H_k
\right)
\exp
\left[
\gamma_W
Q_{\mathrm{style}}
\left(
z_{H_k\oplus Y_W}(W),
w_W
\right)
-
\lambda_d
L_{\mathrm{demand}}
\left(
d_{H_k\oplus Y_W}|_W,
d_W^\star
\right)
\right].
\]

This equation specifies semantics, not a required energy-model
implementation.

### Sampling temperature

Sampling temperature \(\tau\) changes decoding entropy.

It is not:

- style composition;
- style adherence;
- gameplay intensity;
- desired demand.

These interfaces must remain separately named.

## 16. Demand and style in generation

For generation boundary \(g\), define

\[
d_g
=
\operatorname{DemandRoll}_0(H_k;g).
\]

The base semantic target may be written

\[
p_\theta
\left(
Y_W
\mid
X,H_k,\beta_g,
[c_W^{\mathrm{style}}],
[d_W^\star]
\right).
\]

Because \(d_g\) is derived from the committed chart, an implementation may
equivalently use it as a runtime feature:

\[
p_\theta
\left(
Y_W
\mid
X,H_k,\beta_g,d_g,
[c_W^{\mathrm{style}}],
[d_W^\star]
\right).
\]

This does not make demand an independently supplied mandatory input.

For candidate continuation \(Y_W\), branch rollout yields

\[
\widetilde d_Y(t)
=
\operatorname{DemandRoll}_0
\left(
\beta_g,d_g,Y_W;t
\right).
\]

Its branch-local exact section evidence is

\[
\widetilde\chi_Y(W)
=
\left(
x_g,
Y_W
\right),
\]

where \(x_g\) is the exact state at the fixed-through boundary and
\(W=(g,e]\). This is the left-open generation-window counterpart of
\(\chi_H(W)\). Its section geometry is

\[
\widetilde r_Y(W)
=
\Psi
\left(
\left(
\widetilde\chi_Y(W),
d_g,
\widetilde d_Y|_W
\right)
\right),
\]

and its realized style profile is

\[
\widetilde z_Y(W)
=
\operatorname{StyleRead}
\left(
\widetilde r_Y(W)
\right).
\]

These quantities may be used for:

- training supervision;
- branch evaluation;
- requested-style matching;
- requested-demand matching;
- analysis and visualization;
- controlled generation tests.

The generator is not required to first materialize an explicit \(d^\star\) plan
or style plan before producing rows.

## 17. Controlled continuation probes

Because the full frontier cannot be enumerated, define a probe bank

\[
\mathcal Y_{\mathrm{probe}}
=
\{Y_1,\ldots,Y_M\}.
\]

Useful probes include:

- same-role action after controlled intervals;
- outer-inner alternation;
- chord-to-single continuation;
- single-to-chord continuation;
- controlled one-hand density bursts;
- long-note hold with free-role taps;
- long-note release followed by a chord;
- mirrored left and right patterns;
- simultaneous two-hand chords;
- silence and recovery intervals.

For response summaries \(\Omega_i\), define

\[
p_H(t)
=
\left[
\Omega_1
\left(
\mathfrak C_0(b_H(t),Y_1)
\right),
\ldots,
\Omega_M
\left(
\mathfrak C_0(b_H(t),Y_M)
\right)
\right].
\]

Probe responses are used to:

- test whether \(d_H(t)\) preserves relevant frontier information;
- anchor named demand dimensions;
- compare alternative dynamics;
- validate mirror equivariance;
- detect history shortcuts;
- distinguish current intensity from future response;
- define interpretable continuation costs.

The probe vector need not itself be the runtime demand state.

## 18. Evaluation contracts

### Frontier sufficiency

Find histories with matched

\[
x_H(t),
\qquad
d_H(t),
\]

and test them on held-out continuation probes.

Systematic response differences indicate insufficient demand state.

### State-conditioned demand

Hold the current row fixed while varying:

- prior role use;
- inter-event interval;
- chord participation;
- long-note occupancy;
- cross-hand history.

The demand update should reflect the controlled history differences.

### Exact evidence and demand in style

Compare section-style prediction from:

1. \((d_H(a^-),d_H|_W)\) only;
2. \(\chi_H(W)\) only;
3. \((\chi_H(W),d_H(a^-),d_H|_W)\);
4. density, event-count, and global-difficulty baselines.

This measures how exact chart evidence and demand contribute to gameplay-style
recognition.

### Matched-demand style control

Generate with different style requests while matching realized demand
statistics as closely as possible.

If the outputs are still recognized as distinct styles, style control contains
information beyond difficulty or intensity.

### Matched-style demand control

Generate at different desired demand levels while holding requested style
fixed.

This tests whether demand control changes intensity without merely replacing
the section's style.

### Annotation hierarchy

Evaluate:

- section recognition on beatmap-lens labels;
- aligned map-tag prediction after section pooling;
- localization of aligned map-level tags to relevant sections;
- calibration under mixed and missing labels.

### Mirror consistency

For mirrored audio-independent chart conditions,

\[
d_{\mu H}(t)
\approx
\mu d_H(t)
\]

and mirror-invariant style predictions should agree.

## 19. Falsifiable hypotheses

1. **Finite demand state can approximate the moving gameplay frontier.**  
   Histories mapped to the same exact and demand boundary should produce
   matched responses on held-out controlled continuations.

2. **State-conditioned dynamics outperform fixed note kernels.**  
   The same row after different repetition, chord, occupancy, and recovery
   histories should produce distinguishable demand updates when gameplay
   semantics require it.

3. **Gameplay-style concepts are recognizable from exact chart evidence and
   canonical demand rollout.**
   A joint predictor from exact section evidence and demand trajectory should
   outperform event-count, density, and global-difficulty baselines on
   held-out section annotations.

4. **Demand contributes selectively to history-sensitive style concepts.**
   For concepts whose interpretation depends on incoming repetition,
   recovery, occupancy, or coordination state, a joint exact-evidence and
   demand predictor should outperform exact evidence alone. Purely structural
   concepts need not exhibit this gain.

5. **Exact arrangement and demand carry non-identical information.**
   Some style distinctions remain identifiable from recurrence, lane grammar,
   chord topology, timing organization, or long-note structure after demand
   statistics are matched. This is expected and does not by itself imply that
   the frontier representation is insufficient.

6. **Section annotation explains aligned map-level tags.**
   Alignment-specific pooling of section-style salience should predict
   aligned community map tags better than assigning each map tag uniformly to
   all sections.

7. **Style tokens support semantic intervention.**
   Changing a style token should alter held-out style judgments while
   controlling for audio, history, and realized demand.

8. **Demand control and style control are distinguishable.**
   Matched-demand style interventions and matched-style demand interventions
   should both produce their intended independent effects.

9. **Structural hand symmetry improves consistency and sample efficiency.**
   Mirror-equivariant state and demand dynamics should reduce unjustified
   left-right discrepancies without preventing cross-hand interaction.

10. **An explicit desired-demand plan is not required for ordinary
   generation.**  
   A direct generator with derived gameplay state should remain competitive
   when no \(d^\star\) is provided. Failure would justify a stronger explicit
   planning layer.

## 20. Open definitions

- What is the smallest exact control state needed to update demand without
  replaying all history?
- What finite representation of the frontier is sufficient?
- Which demand dimensions are stable, interpretable, and independently
  anchorable?
- Which dynamics best express burst, strain, recovery, long-note occupancy, and
  release?
- Is an explicit cross-hand demand component necessary?
- Which parts of section style are explained by demand trajectory?
- Which symbolic arrangement residuals remain?
- How should section boundaries be defined or annotated?
- How should mixed or transitional section styles be represented?
- What pooling operator best relates section style to aligned community map
  tags?
- Which community vote information should be retained rather than thresholded?
- Does the annotation workflow provide explicit negatives or only positives?
- How should incompatible style and demand controls be detected?
- Which continuation probes best distinguish frontier states?
- How should local demand aggregate into intensity and map difficulty?
- When does explicit desired-demand conditioning improve controllability enough
  to justify its complexity?

## 21. Scope boundary

This page fixes:

- one canonical gameplay profile;
- deterministic canonical hand-role coordinate projection;
- exact chart and control state;
- the moving gameplay frontier as the gameplay-relevant abstract object;
- \(d_H(t)\) as its finite profile-relative representation;
- hybrid event and recovery dynamics;
- mirror-equivariant hand structure;
- joint exact-evidence and demand section geometry;
- the distinction between demand, style, and style tags;
- multi-label section-style semantics;
- declared-compatible weak map-level and strong section-level annotation
  roles;
- realized versus requested style and demand;
- controlled probes and validation contracts.

This page does not fix:

- a concrete neural architecture;
- a final demand dimension inventory;
- numerical units or calibration datasets;
- a particular section encoder;
- a particular style classifier;
- the final enum names;
- the candidate proposal or stable-prefix algorithm;
- player-specific or multiplayer models.

Candidate generation, legal continuation support, and commit semantics belong to
[notation.md](notation.md). Concrete architectures, annotation experiments,
ablation results, and empirical evidence belong in `docs/research/`.
