# Canonical gameplay state

Pulsefield associates every legal chart with a reproducible canonical
gameplay-state trajectory. The trajectory is a research object and a generation
interface, not an assertion about the hidden state of a particular human
player.

This page deliberately separates three epistemic layers:

\[
H
\xrightarrow{\operatorname{Replay}}
x_H(t),
\]

\[
(H,\vartheta_0)
\xrightarrow{\operatorname{Infer}}
\alpha_H(t),
\]

\[
(H,\vartheta_0)
\xrightarrow{\operatorname{DemandRoll}}
d_H(t).
\]

Here:

- \(x_H(t)\) is exactly determined by chart replay;
- \(\alpha_H(t)\) is a canonical belief over ambiguous execution
  interpretations under profile \(\vartheta_0\);
- \(d_H(t)\) is a profile-dependent continuous demand state whose semantics
  require calibration and empirical anchoring.

The complete relation is

\[
\text{legal chart}
\xrightarrow{\operatorname{Roll}_{\vartheta_0}}
\text{canonical gameplay-state trajectory}
\xrightarrow{\Pi_d}
\text{gameplay-demand field}.
\]

The notation extends [generation notation](notation.md).

## 1. Formal scene and epistemic decomposition

For hand \(h\in\{L,R\}\), define

\[
z_h(t)
=
\left(
x_h(t),
\alpha_h(t),
d_h(t)
\right).
\]

The exact hand state is

\[
x_h(t)
=
\left(
\omega_h(t),
\chi_h(t),
c_h(t)
\right),
\]

where:

- \(\omega_h(t)\) is exact long-note occupancy;
- \(\chi_h(t)\) contains selected exact finite-state summaries;
- \(c_h(t)\) contains exact deterministic clocks.

Let \(\mathcal P\) be a profile-owned parity or fingering-state space. The
canonical execution belief is

\[
\alpha_h(t)
\in
\Delta(\mathcal P).
\]

Let

\[
\mathcal C
=
\{c_1,\ldots,c_D\}
\]

be the hand-demand channel set. Continuous hand demand is

\[
d_h(t)
\in
\mathbb R_{\ge 0}^{D}.
\]

The complete state is

\[
z(t)
=
\left(
z_L(t),
z_R(t),
x_\times(t),
d_\times(t)
\right),
\]

where

\[
x_\times(t)
\]

contains selected exact cross-hand facts and

\[
d_\times(t)
\in
\mathbb R_{\ge 0}^{D_\times}
\]

contains continuous cross-hand interaction demand. \(D_\times=0\) is an
admissible model; the usefulness of a nonempty cross-hand demand component is
an empirical question.

For aggregate equations, write

\[
x(t)
=
\left(
x_L(t),
x_R(t),
x_\times(t)
\right),
\]

\[
\alpha(t)
=
\left(
\alpha_L(t),
\alpha_R(t)
\right),
\]

\[
d(t)
=
\left(
d_L(t),
d_R(t),
d_\times(t)
\right),
\]

and

\[
z(t)
=
\left(
x(t),
\alpha(t),
d(t)
\right).
\]

The epistemic status of these components is:

| Component | Status | Meaning |
| --- | --- | --- |
| \(x(t)\) | Exact | Uniquely replayable from committed chart events and elapsed time. |
| \(\alpha(t)\) | Canonical inference | Reproducible under \(\vartheta_0\), but not uniquely implied by the chart and not a record of an actual player's fingers. |
| \(d(t)\) | Latent calibrated state | Reproducible after fixing \(\vartheta_0\), but channel scale and semantics are not identified by chart syntax alone. |

The formulation imposes the following hard separation:

1. Chart legality depends on exact replay state \(x\), never on demand magnitude
   or preferred parity.
2. \(\alpha\) may represent ambiguity, but its update must be reproducible after
   fixing \(\vartheta_0\).
3. \(d\) is part of the same causal rollout used during generation and
   post-chart analysis.
4. Gameplay state need not summarize motif identity, mapper intent, or every
   long-range chart property. The full history \(H_k\) remains a separate
   conditioning object.
5. Any information required to decide hard legality at a branch boundary must
   be exact or exactly derivable; it must not exist only inside an
   unconstrained neural hidden state.

## 2. Worked example

Use the chart prefix

\[
H_4
=
\bigl(
(29.40,1000),
(29.70,0020),
(29.90,0100),
(30.00,0030)
\bigr).
\]

Before the row at \(29.70\), suppose both right-hand lanes are closed:

\[
\omega_R(29.70^-)
=
(0,0),
\]

where right-hand order is `(outer, inner)` and therefore corresponds to lanes
\((4,3)\).

For

\[
y_2=(29.70,0020),
\]

the right-hand action pair is

\[
a_2^R=(0,2).
\]

The exact event transition gives

\[
\omega_R(29.70^+)
=
(0,1).
\]

At \(29.90\),

\[
a_3^R=(0,0),
\]

so the `EMPTY` action on lane 3 preserves the open state:

\[
\omega_R(29.90^+)
=
(0,1).
\]

At \(30.00\),

\[
a_4^R=(0,3),
\]

and the exact transition closes the long note:

\[
\omega_R(30.00^+)
=
(0,0).
\]

Replacing row 4 with `0020` would attempt a second `LN_START` on an open lane.
The exact event transition would be undefined, so the replacement is illegal
regardless of style, difficulty, model score, or player profile.

The continuous demand trajectory is not determined by these occupancy values
alone. Under a fixed canonical profile,

\[
d(29.70^+)
=
R_{d,\vartheta_0}
\left(
d(29.70^-),
x(29.70^-),
\alpha(29.70^-),
y_2
\right).
\]

The tap at \(29.90\) occurs while lane 3 is held. Therefore its demand update
may differ from the counterfactual update produced by the same `0100` row when
no long note is open. The difference is profile-dependent, but after
\(\vartheta_0\) is fixed it must be reproducible.

Now suppose a later left-hand chord is

\[
y_5=(30.25,1100).
\]

Its exact lane occupancy transition is unambiguous. Its continuation parity
need not be. If the canonical profile admits more than one reasonable
post-chord continuation, then

\[
\alpha_L(30.25^+)
\]

is not a point mass. This does not mean the chart is uncertain or illegal. It
means the chart does not uniquely identify one canonical fingering
continuation.

The example therefore contains three different kinds of information:

- lane-3 occupancy is exact;
- post-chord parity may be belief-valued;
- demand magnitude and decay depend on the calibrated profile.

## 3. Exact replay state

Per-hand occupancy is

\[
\omega_h(t)
=
\left(
\omega_{h,\mathrm{outer}}(t),
\omega_{h,\mathrm{inner}}(t)
\right)
\in
\{0,1\}^2.
\]

`LN_START`, `LN_CLOSE`, `TAP`, and `EMPTY` update occupancy according to the
action table in [generation notation](notation.md#5-rows-and-histories).

Other exact state may include:

- last active physical role;
- per-role and per-lane recency clocks;
- the most recent hand-action interval;
- exact same-side run length;
- exact pending obligations;
- exact cross-hand alternation clocks;
- exact summaries needed by a declared transition rule.

The inventory is not “every fact that could be computed from history.” An exact
summary belongs in \(x\) when at least one of the following holds:

1. it is required to decide hard legality;
2. it is required to make the declared canonical update Markov at a branch
   boundary;
3. it is intentionally exposed as an exact control or diagnostic variable.

Other chart statistics may remain derivable from \(H_k\) or represented by a
history model.

Exact cross-hand state \(x_\times\) is required only for persistent facts that
cannot be recovered from the two hand states and the current row. A
simultaneous row does not automatically require stored cross-hand state; it may
be consumed directly by the event transition.

Let

\[
E_{\mathrm{chart}}(x_k^-,y_k)
\]

be the exact chart transition. It is profile-independent. If a lane-state or
other hard chart invariant is violated, the operator is undefined.

This separation is essential:

\[
\text{chart legality}
\neq
\text{canonical player preference}.
\]

A difficult jack, hand imbalance, dense chord, or non-alternating continuation
may be undesirable under one scorer, but it is not illegal unless an explicit
chart-format rule forbids it.

## 4. Canonical execution belief and parity

Let \(p_h^k\in\mathcal P\) denote a profile-defined execution interpretation
immediately after row \(k\). One admissible canonical representation is

\[
\alpha_h^k(p)
=
P_{\vartheta_0}
\left(
p_h^k=p
\mid
H_k
\right),
\qquad
\alpha_h^k
\in
\Delta(\mathcal P).
\]

This probability is defined by the canonical profile. It is not an empirical
claim that an observed human player used state \(p\).

The state space \(\mathcal P\) should be expressive enough to distinguish, when
relevant:

- the recently active physical role;
- the natural role for a possible alternating continuation;
- an active same-side repetition or jack run;
- ambiguity after a two-finger chord;
- role availability forced by long-note occupancy.

The formulation does not yet fix the members of \(\mathcal P\).

The two hand beliefs are updated jointly:

\[
\alpha_k^+
=
U_{\alpha,\vartheta_0}
\left(
\alpha_k^-,
x_k^-,
y_k
\right).
\]

Although \(\alpha_L\) and \(\alpha_R\) are separately readable, the update may
use both hands and the complete row. The formulation does not assume
independent hand inference.

When the continuation state is uniquely determined, \(\alpha_h\) may be a
point mass. A simpler implementation may use a single point state everywhere,
but that is an approximation on ambiguous patterns rather than a consequence
of chart syntax.

\(\alpha\) represents execution ambiguity, not parameter uncertainty. Unknown
or uncertain parameters of \(U_{\alpha,\vartheta_0}\) require a separate
posterior, ensemble, or calibration analysis.

There is also a closure requirement on the state representation. If demand
dynamics depend nonlinearly on latent parity, then a marginal
\((\alpha_h,d_h)\) may fail to preserve enough information for an exact update.
An admissible profile must then enlarge its state, for example by retaining
parity-conditioned demand states

\[
\left\{
\alpha_h(p),
d_h^{(p)}
\right\}_{p\in\mathcal P},
\]

or another sufficient joint representation. Replacing such a mixture by one
mean demand vector is an approximation and must be identified as one.

## 5. Hand symmetry and cross-hand communication

The canonical `(outer, inner)` coordinates make left-right symmetry explicit.
Let \(\mu\) be the mirror operator defined in
[generation notation](notation.md#4-lanes-hands-and-physical-roles).

The current structural prior is that the base hand-update family is
mirror-equivariant. For a profile \(\vartheta\), its mirrored profile
\(\mu\vartheta\), and mirrored state and chart,

\[
\operatorname{Roll}_{\mu\vartheta}
\left(
\mu z_0,
\mu H;
t
\right)
=
\mu
\operatorname{Roll}_{\vartheta}
\left(
z_0,
H;
t
\right).
\]

For a symmetric canonical profile satisfying

\[
\mu\vartheta_0=\vartheta_0,
\]

this becomes

\[
\operatorname{Roll}_{\vartheta_0}
\left(
\mu z_0,
\mu H;
t
\right)
=
\mu
\operatorname{Roll}_{\vartheta_0}
\left(
z_0,
H;
t
\right).
\]

This is a chosen structural prior, not an observed fact about every player.

The prior has two consequences:

1. left and right hands should use the same role semantics and compatible
   update parameters;
2. any handedness asymmetry should enter through an explicit profile variable,
   not through arbitrary unrelated meanings for left-hand and right-hand
   channels.

Mirror equivariance does not imply hand independence. A hand update may depend
on:

- its own exact state and belief;
- the other hand's exact state and belief;
- the complete simultaneous row;
- exact cross-hand clocks;
- symmetric messages exchanged between the two hand representations.

Parameter sharing with symmetric cross-hand message passing is one possible
implementation. The formulation fixes the equivariant semantics, not a
particular neural topology.

## 6. Hybrid transition dynamics

Gameplay state is a hybrid dynamical system. Rows cause discrete transitions;
time between rows advances clocks and continuous dynamics.

For a row \(y_k\) at time \(t_k\), use the right-continuous convention

\[
z_k^-
=
z(t_k^-),
\qquad
z_k^+
=
z(t_k)
=
z(t_k^+).
\]

A bare \(z(t_k)\) therefore includes row \(k\) exactly once.

### Exact event transition

The profile-independent exact transition is

\[
x_k^+
=
E_{\mathrm{chart}}
\left(
x_k^-,
y_k
\right).
\]

If the transition is undefined, the row is illegal.

### Execution-belief transition

For a legal row,

\[
\alpha_k^+
=
U_{\alpha,\vartheta_0}
\left(
\alpha_k^-,
x_k^-,
y_k
\right).
\]

The operator may use the exact post-event state internally because

\[
x_k^+
=
E_{\mathrm{chart}}(x_k^-,y_k)
\]

is already determined.

### Demand event transition

The generic demand reset is

\[
d_k^+
=
R_{d,\vartheta_0}
\left(
d_k^-,
x_k^-,
\alpha_k^-,
y_k
\right)
\in
\mathbb R_{\ge 0}^{2D+D_\times}.
\]

The same row may produce different demand updates in different histories.

An additive event injection is one admissible parameterization:

\[
d_k^+
=
d_k^-
+
J_{\vartheta_0}
\left(
y_k,
x_k^-,
\alpha_k^-,
d_k^-
\right).
\]

It is not part of the canonical definition. A model using this form must ensure

\[
d_k^+\ge 0.
\]

Requiring \(J_{\vartheta_0}\ge 0\) is an additional, narrower prior. The generic
reset map also permits saturation, gating, redistribution across channels, or
a calibrated release transition.

### Between-event flow

Let

\[
\Delta t_k
=
t_{k+1}-t_k.
\]

For \(0<\tau<\Delta t_k\), write

\[
x(t_k+\tau)
=
F_{\mathrm{chart},\tau}(x_k^+),
\]

\[
\alpha(t_k+\tau)
=
F_{\alpha,\vartheta_0,\tau}
\left(
\alpha_k^+,
x_k^+
\right),
\]

\[
d(t_k+\tau)
=
F_{d,\vartheta_0,\tau}
\left(
d_k^+,
x_k^+,
\alpha_k^+
\right).
\]

The exact flow advances deterministic clocks and preserves finite state unless
an explicitly declared time-triggered transition occurs.

The default parity convention may keep \(\alpha\) unchanged between rows.
Allowing time-dependent belief relaxation is an additional profile choice.

One possible continuous demand parameterization is

\[
\frac{d}{dt}d(t)
=
f_{\vartheta_0}
\left(
d(t),
x(t),
\alpha(t)
\right).
\]

Again, the differential equation is a model family, not a formulation-level
necessity.

Demand flow must preserve nonnegativity. It may include:

- recovery or decay between rows;
- occupancy-dependent forcing while a long note remains open;
- interaction between hand-local and cross-hand channels;
- profile-dependent time constants.

`LN_START` changes exact occupancy. A demand model may use that open occupancy
to sustain forcing. `LN_CLOSE` ends the occupancy and may also produce a
calibrated release transient. Sustained forcing and release transients are
chosen demand priors, not consequences of the row alphabet alone.

Hard legality reads only \(x\). Neither \(\alpha\) nor \(d\) may turn an
otherwise legal pattern into an illegal one.

## 7. Canonical rollout and continuation boundaries

The exact pre-chart state \(x_0\) contains:

- closed occupancy on every lane;
- declared finite-state sentinels;
- declared exact clock sentinels.

The canonical player profile \(\vartheta_0\) owns:

- the initial execution belief \(\alpha_0\);
- the initial demand baseline \(d_0\);
- the parity or fingering state space;
- belief-update rules;
- demand channels and cross-hand channel definitions;
- demand event and flow dynamics;
- optional handedness parameters;
- calibrated style, intensity, and difficulty readouts.

The canonical rollout assumes:

- every chart action is executed successfully;
- hit timing is ideal;
- the canonical hand partition is used;
- player errors and misses are not modeled.

For a legal chart \(H\),

\[
z_H^{\mathrm{can}}(t;\vartheta_0)
=
\operatorname{Roll}_{\vartheta_0}
\left(
z_0,
H;
t
\right).
\]

Its exact projection must agree with profile-independent replay:

\[
\Pi_x
z_H^{\mathrm{can}}(t;\vartheta_0)
=
\operatorname{Replay}
\left(
x_0,
H;
t
\right).
\]

Thus changing demand calibration or parity priors cannot change whether the
chart has an open long note.

A timestamped continuation boundary is

\[
\beta_b
=
(t_b,z_b).
\]

Continuation rollout is

\[
\operatorname{Roll}_{\vartheta_0}
\left(
\beta_b,
Y;
t
\right).
\]

During candidate refresh \(r\), revision \(s\),

\[
\beta_{r,s}
=
(g_{r,s},z_{r,s}),
\]

where

\[
z_{r,s}
=
\operatorname{Roll}_{\vartheta_0}
\left(
z_0,
H_{k_{r,s}};
g_{r,s}
\right).
\]

The boundary includes every committed row with time at or before \(g_{r,s}\)
and all silent flow up to \(g_{r,s}\).

Generation and post-chart analysis must share one rollout semantics:

\[
\operatorname{Roll}_{\mathrm{generation}}
=
\operatorname{Roll}_{\mathrm{analysis}}.
\]

Otherwise a chart can be optimized against one definition of demand and
reported under another.

A real player may be represented by another profile \(\vartheta\). The
canonical profile is a reference convention, not a universal human model.

## 8. Canonical gameplay-demand field and readouts

The hand and cross-hand demand channel sets are

\[
\mathcal C
=
\{c_1,\ldots,c_D\},
\]

\[
\mathcal C_\times
=
\{c_1^\times,\ldots,c_{D_\times}^\times\}.
\]

The demand field is the continuous projection

\[
\mathcal D_H(h,c,t;\vartheta_0)
=
[d_h(t)]_c,
\qquad
h\in\{L,R\},
\quad
c\in\mathcal C,
\]

and

\[
\mathcal D_H(\times,c^\times,t;\vartheta_0)
=
[d_\times(t)]_{c^\times},
\qquad
c^\times\in\mathcal C_\times.
\]

The field may be visualized with time on the horizontal axis, demand channel on
the vertical axis, and magnitude as intensity.

\(d(t)\) is an idealized chart-induced control-demand state. It is not, without
an additional observation model, a direct measurement of fatigue, pain,
attention, error probability, or subjective difficulty in a particular human.

The field is not a collection of independent scalar note difficulties. A row
changes state at its event time; subsequent recovery, occupancy-dependent
forcing, and later state-conditioned events determine the realized trajectory.

A per-note response shape may be defined for diagnostics only as a
counterfactual isolated response under a declared no-future-event baseline.
With nonlinear or state-conditioned dynamics, those isolated shapes are not
additive components of the realized chart trajectory.

Candidate hand-demand semantics include:

- same-side repetition;
- outer-inner alternation;
- local density;
- one-hand chord burden;
- sustained long-note control;
- release timing;
- short-term stamina;
- hand speed;
- finger-occupancy pressure.

Candidate cross-hand semantics include:

- coordinated simultaneity;
- hand-balance pressure;
- two-hand chord burden;
- synchronization or alternation demand.

These are proposed channel meanings, not identified canonical dimensions.

A local intensity readout is

\[
i_H(t;\vartheta_0)
=
r_{\mathrm{intensity}}
\left(
d_L(t),
d_R(t),
d_\times(t);
\vartheta_0
\right).
\]

Section or map difficulty requires a separate aggregation:

\[
\operatorname{Diff}
\left(
H;
W,
\vartheta_0
\right)
=
\operatorname{Agg}_{t\in W}
i_H(t;\vartheta_0).
\]

Neither readout is defined as \(\lVert d(t)\rVert\) until channel units,
interactions, and profile calibration justify that operation.

The working style hypothesis is that chart style is partly expressed by the
temporal geometry of the demand field:

- which channels rise;
- whether they rise together or alternate;
- how long they remain elevated;
- how quickly they recover;
- whether the two hands are symmetric;
- how one demand type transitions into another;
- whether similar intensity islands recur.

A map-level tag is only a weak observation of this geometry. For example, a
`jack-heavy` section predicate may take the form

\[
\frac{1}{|W|}
\int_W
\operatorname{Agg}_{h\in\{L,R\}}
\mathcal D_H
\left(
h,
\mathrm{jack},
t;
\vartheta_0
\right)
\,dt
>
\tau,
\]

together with a minimum-duration condition. It does not imply that the whole
chart has that style at every time.

Demand magnitude as intensity, demand composition as style mixture, and
demand-field sufficiency for style are hypotheses rather than definitions.

## 9. Uncertainty and identifiability

Several distinct uncertainties must not be collapsed:

| Object | Uncertainty represented |
| --- | --- |
| \(\alpha(t)\) | Ambiguity among canonical execution or parity interpretations for a fixed chart and fixed profile. |
| A distribution over future charts \(P(Y\mid\cdot)\) | Uncertainty over which legal choreography should be generated. |
| \(\Gamma_r\) | Candidate or musical-opportunity uncertainty, when the chosen representation contains it. |
| A posterior or ensemble over \(\vartheta\) | Parameter and model uncertainty. |
| \(d(t)\) under fixed \(\vartheta_0\) | Not a probability distribution; it is a deterministic latent trajectory unless the profile explicitly propagates a state distribution. |

Continuous demand channels are not identifiable from chart reconstruction or
map-level style tags alone. Without anchors:

- latent rotations may preserve the objective;
- channel scale may be arbitrary;
- multiple event-response systems may generate similar readouts;
- a model may use non-gameplay shortcuts;
- section-level tags may not identify note-level causal responses.

A channel should therefore be treated in one of two ways:

1. **named and anchored**, with explicit semantic constraints and validation; or
2. **free and latent**, without attaching a human-readable gameplay name merely
   from post-hoc inspection.

Possible anchoring sources include:

- an explicitly defined initial demand basis;
- controlled synthetic pattern probes;
- expert section annotations;
- pairwise style or burden comparisons;
- player performance and error data;
- channel-specific monotonicity constraints;
- intervention tests that hold the row type fixed while changing its history.

Control V3 targets or other legacy reconstruction objectives do not by
themselves identify a canonical causal state or a readable demand basis. See
the repository
[legacy code boundary](../../README.md#legacy-code-boundary).

## 10. Falsifiable hypotheses

1. **Exact branch-local state improves legality and continuity.**  
   A generator carrying exact occupancy, clocks, and obligations should reduce
   illegal long-note transitions and cross-window discontinuities relative to
   an otherwise matched history-only model. No reproducible reduction rejects
   the claimed benefit for the tested setting.

2. **Belief-valued parity preserves useful ambiguity.**  
   A belief representation should improve continuation quality or calibrated
   uncertainty on ambiguity-controlled chord cases relative to forced
   single-state parity. No gain supports the simpler representation.

3. **State-conditioned demand is more informative than fixed note kernels.**  
   It should better predict controlled burden comparisons in which the same row
   occurs under different preceding occupancy, recency, or repetition states.
   Failure on those contrasts rejects the extra conditioning for the tested
   channels.

4. **Structural hand symmetry is useful.**  
   Mirror-equivariant hand semantics should improve mirrored consistency and
   sample efficiency without preventing explicitly conditioned handedness.
   No controlled benefit, or systematic damage to asymmetric profiles, argues
   for revising the prior.

5. **Demand trajectories contain section-level information.**  
   Calibrated trajectories should predict held-out section annotations or
   player outcomes better than matched global labels and event-count baselines.
   Otherwise the claimed trajectory information has not been established.

6. **Cross-hand demand may be non-additive.**  
   \(d_\times\) should improve controlled two-hand coordination predictions
   beyond hand-local demand and exact simultaneous-row facts. No controlled
   gain supports setting \(D_\times=0\).

## 11. Open definitions

- What is the smallest exact state sufficient for legality and canonical
  continuation?
- Which exact summaries belong in \(x_h\) and \(x_\times\), and which should
  remain in chart-history memory?
- What parity or fingering state space \(\mathcal P\) is adequate?
- When is a point parity state sufficient, and when is a belief or
  parity-conditioned demand mixture required?
- Which profile variables should express handedness?
- Which demand channels can be semantically anchored?
- Which demand event and flow dynamics are adequate for recovery, long-note
  occupancy, and release?
- Is a nonempty cross-hand demand component necessary?
- How should local intensity aggregate into section and map difficulty?
- Should generation condition on an explicit desired demand trajectory, or
  should demand remain an implicit consequence scored after rollout?