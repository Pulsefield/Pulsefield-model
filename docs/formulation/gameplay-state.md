# Canonical control-load state and gameplay-demand field

Pulsefield associates every legal chart with a reproducible exact control
trace, an operational control-load trajectory, and a declared gameplay-demand
field. These objects are research interfaces for generation and analysis. They
are not, without a separate player observation model, claims about the hidden
physiological or psychological state of a particular human player.

The central factorization is

\[
H
\xrightarrow{\operatorname{Replay/Exec}_\lambda}
\left(x_H(t),u_H\right)
\xrightarrow{\operatorname{LoadRoll}_\kappa}
q_H^\kappa(t)
\xrightarrow{\operatorname{Readout}_\varrho}
\mathcal D_H^{\Theta}(t),
\]

where

\[
\Theta=(\kappa,\varrho).
\]

The fixed lane-role map \(\lambda\) is a canonical system convention rather
than a parameter of \(\Theta\). Within the control-load specification:

- \(\kappa\) owns the operational state space, initial state, event resets,
  and between-event flow;
- \(\varrho\) owns the exposed demand basis, readout map, units, and
  calibration conventions.

A reference specification used by generation is

\[
\Theta_0=(\kappa_0,\varrho_0).
\]

Here:

- \(x_H(t)\) is exact state determined by chart replay and elapsed time;
- \(u_H\) is the exact canonical hand-role action trace induced by the fixed 4K
  role mapping \(\lambda\);
- \(q_H^\kappa(t)\) is model-dependent operational memory sufficient for the
  declared future load dynamics;
- \(\mathcal D_H^\Theta(t)\) is a nonnegative exposed demand readout whose
  named semantics require calibration;
- actual player execution, errors, fatigue, adaptation, and subjective burden
  belong to a later observation model.

The notation extends the
[generation notation and structured decision problem](notation.md).

## 1. Formal scene and epistemic decomposition

For a legal chart \(H\), write the exact state as

\[
x_H(t)
=
\left(
x_H^{\mathrm{fmt}}(t),
x_H^{\mathrm{ctrl}}(t)
\right).
\]

The two exact components have different responsibilities:

- \(x^{\mathrm{fmt}}\) contains chart-format state required for hard legality,
  such as long-note occupancy and pending format obligations;
- \(x^{\mathrm{ctrl}}\) contains exact canonical control-history summaries, such
  as role-wise action clocks and recent active-role sets.

For hand \(h\in\{L,R\}\), one admissible decomposition is

\[
x_h^{\mathrm{fmt}}(t)
=
\omega_h(t),
\]

\[
x_h^{\mathrm{ctrl}}(t)
=
\left(
A_h^{\mathrm{last}}(t),
\chi_h(t),
c_h(t)
\right),
\]

where:

- \(\omega_h(t)\) is exact role-wise long-note occupancy;
- \(A_h^{\mathrm{last}}(t)\subseteq\mathcal R\) is the exact active-role set of
  the most recent hand action;
- \(\chi_h(t)\) contains selected exact finite-state summaries;
- \(c_h(t)\) contains exact deterministic clocks.

Exact cross-hand state is written

\[
x_\times(t),
\]

and the aggregate exact state is

\[
x(t)
=
\left(
x_L(t),x_R(t),x_\times(t)
\right).
\]

For fixed load dynamics \(\kappa\), let

\[
q^\kappa(t)
=
\left(
q_L^\kappa(t),
q_R^\kappa(t),
q_\times^\kappa(t)
\right)
\in\mathcal Q_\kappa
\]

be the operational load state. The state space \(\mathcal Q_\kappa\) is not
required to use named, nonnegative, or uniquely identifiable coordinates.

For fixed readout \(\varrho\), the gameplay-demand field is

\[
\mathcal D_H^\Theta(t)
=
G_\varrho
\left(
x_H(t),q_H^\kappa(t)
\right)
\in
\mathbb R_{\ge0}^{C}.
\]

The epistemic status of the objects is:

| Object | Status | Meaning |
| --- | --- | --- |
| \(H\) | Observed or generated symbolic object | The committed complete-row chart. |
| \(u_H\) | Exact under \(\lambda\) | The deterministic hand-role action trace induced by the chart. |
| \(x_H(t)\) | Exact | Replayable chart-format and canonical-control facts. |
| \(q_H^\kappa(t)\) | Model-defined operational state | Deterministic after fixing \(\kappa\), but not uniquely identified by chart syntax. |
| \(\mathcal D_H^\Theta(t)\) | Declared readout | A chart-induced demand description whose named coordinates require anchoring and calibration. |
| Player response | Unmodeled in the canonical rollout | Requires player identity, execution, and observed outcomes. |

In the base fixed-lane 4K setting, the admissible canonical execution set is a
singleton:

\[
\mathcal E_\lambda(H)
=
\{u_H\}.
\]

There is therefore no base variable \(\alpha(t)\) representing a posterior over
which canonical role executed the chart. A one-hand chord acts on both canonical
roles. Uncertainty appears only if the formulation is later extended to allow
multiple admissible physical executions.

The formulation imposes the following hard separation:

1. Chart legality depends only on \(x^{\mathrm{fmt}}\), never on operational load,
   demand magnitude, or stylistic preference.
2. \(u_H\) and \(x_H\) are exact consequences of \(H\) and the declared
   canonical mapping; they are not learned player beliefs.
3. \(q^\kappa\) is operational memory for future rollout, not automatically a
   human-readable gameplay ontology.
4. Named gameplay semantics belong to the calibrated readout
   \(\mathcal D^\Theta\), not to arbitrary coordinates of \(q^\kappa\).
5. The same \(\kappa\) and \(\varrho\) must be used when a branch is optimized
   and when its demand trajectory is later reported.
6. The full history \(H_k\) remains available alongside finite state; control
   state need not summarize motif identity, mapper intent, or every long-range
   property.
7. Any fact required for hard legality at a continuation boundary must be exact
   or exactly derivable. It must not exist only inside an unconstrained neural
   hidden state.

## 2. Worked example: long-note occupancy and chord continuation

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

Before the row at \(29.70\), suppose both right-hand roles are closed:

\[
\omega_R(29.70^-)
=
(0,0),
\]

where right-hand role order is `(outer, inner)` and corresponds to serialized
lanes \((4,3)\).

For

\[
y_2=(29.70,0020),
\]

the right-hand action pair is

\[
a_2^R=(0,2).
\]

The exact format transition gives

\[
\omega_R(29.70^+)
=
(0,1).
\]

At \(29.90\),

\[
a_3^R=(0,0),
\]

so `EMPTY` on lane 3 preserves the open long note:

\[
\omega_R(29.90^+)
=
(0,1).
\]

At \(30.00\),

\[
a_4^R=(0,3),
\]

and the exact transition closes it:

\[
\omega_R(30.00^+)
=
(0,0).
\]

Replacing row 4 with `0020` would attempt a second `LN_START` on an open lane.
The exact format transition would be undefined, so the replacement is illegal
regardless of music, style, difficulty, candidate score, or load state.

The same row may produce different operational load updates in different
histories. Under fixed \(\kappa\),

\[
q(29.70^+)
=
R_\kappa
\left(
q(29.70^-),
x(29.70^-),
x(29.70^+),
a_2
\right).
\]

The tap at \(29.90\) occurs while right inner remains occupied. Its load update
may therefore differ from the counterfactual update produced by the same
`0100` row when no long note is open.

Now append a left-hand chord:

\[
y_5=(30.25,1100).
\]

Its left-hand action pair and active-role set are exactly

\[
a_5^L=(1,1),
\]

\[
A_5^L
=
\{\mathrm{outer},\mathrm{inner}\}.
\]

Both left-role recency clocks are reset by the chord. The state does not need to
choose whether outer or inner was “really last”; both roles acted.

Suppose the next row is

\[
y_6=(30.37,1000).
\]

This row deterministically reuses left outer after \(0.12\) seconds. Its load
update may depend on:

- left outer's exact recency;
- left inner's simultaneous participation in the preceding chord;
- the fact that the previous left-hand action was a two-role set;
- current operational load and recovery;
- simultaneous or recent right-hand activity.

Several demand mechanisms may be active at once, such as role reuse, chord
recovery, density, and cross-hand coordination. They are not required to be
mutually exclusive states whose weights sum to one.

This example separates four objects:

- long-note legality is exact chart-format state;
- chord participation is exact role-set state;
- future burden depends on operational dynamics \(q^\kappa\);
- named descriptions such as repetition pressure or chord burden are demand
  readouts from \(\mathcal D^\Theta\).

## 3. Exact chart-format and canonical-control state

Per-hand long-note occupancy is

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
action table in [notation.md](notation.md#5-rows-long-notes-and-committed-histories).

The exact state is divided by responsibility:

\[
x^{\mathrm{fmt}}(t)
=
\left(
\omega_L(t),
\omega_R(t),
\xi^{\mathrm{fmt}}(t)
\right),
\]

where \(\xi^{\mathrm{fmt}}\) contains any additional exact format obligations,
and

\[
x^{\mathrm{ctrl}}(t)
=
\left(
x_L^{\mathrm{ctrl}}(t),
x_R^{\mathrm{ctrl}}(t),
x_\times(t)
\right).
\]

Possible exact canonical-control summaries include:

- the most recent active-role set \(A_h^{\mathrm{last}}\);
- per-role and per-lane action clocks;
- separate press, close, and release clocks where declared;
- the most recent hand-action interval;
- exact recent role-set sequences of bounded length;
- exact run lengths under an explicitly declared deterministic definition;
- exact cross-hand simultaneity or alternation clocks;
- exact pending obligations required by a declared transition rule.

A single-valued “last active role” is not sufficient after a two-role chord. The
exact object is a set or equivalent multi-hot representation:

\[
A_h^{\mathrm{last}}
\in
\mathcal P(\mathcal R)
=
\left\{
\varnothing,
\{\mathrm{outer}\},
\{\mathrm{inner}\},
\{\mathrm{outer},\mathrm{inner}\}
\right\}.
\]

An exact summary belongs in \(x\) when at least one of the following holds:

1. it is required to decide hard chart legality;
2. it is required to continue a declared exact transition without replaying the
   entire history;
3. it is intentionally exposed as an exact generation control or diagnostic;
4. it is required to initialize the operational load dynamics at a branch
   boundary without ambiguity.

The inventory is not “every statistic computable from history.” Other chart
facts may remain derivable from \(H_k\), and motif or sectional information may
remain in a separate history model.

Exact cross-hand state \(x_\times\) is required only for persistent facts that
cannot be recovered from the two hand states and the current row. A simultaneous
row does not automatically require stored cross-hand state; it can be consumed
directly by the event transition.

For row \(y_k=(t_k,m_k)\), let

\[
a_k
=
\operatorname{Act}_\lambda(m_k).
\]

The chart-format transition is a partial operator

\[
x_k^{\mathrm{fmt},+}
=
E_{\mathrm{fmt}}
\left(
x_k^{\mathrm{fmt},-},m_k
\right).
\]

If it is undefined, the row is illegal. For a legal row, exact control state
updates deterministically:

\[
x_k^{\mathrm{ctrl},+}
=
E_{\mathrm{ctrl}}
\left(
x_k^{\mathrm{ctrl},-},
x_k^{\mathrm{fmt},-},
x_k^{\mathrm{fmt},+},
a_k,t_k
\right).
\]

The aggregate exact transition is

\[
x_k^+
=
E_{\mathrm{exact}}(x_k^-,y_k).
\]

Hard legality reads only the chart-format projection:

\[
\text{legal}(x_k^-,y_k)
\iff
E_{\mathrm{fmt}}(x_k^{\mathrm{fmt},-},m_k)
\text{ is defined}.
\]

This separation is essential:

\[
\text{chart legality}
\neq
\text{canonical control history}
\neq
\text{gameplay preference}.
\]

A difficult jack, hand imbalance, dense chord, non-alternating continuation, or
high demand may be undesirable under one scorer, but it remains legal unless an
explicit chart-format rule forbids it.

## 4. Operational control-load state

For declared load dynamics \(\kappa\), the operational state is

\[
q^\kappa(t)
\in
\mathcal Q_\kappa.
\]

A finite-dimensional implementation may use

\[
\mathcal Q_\kappa
\subseteq
\mathbb R^{Q},
\]

with a decomposition such as

\[
q^\kappa(t)
=
\left(
q_L^\kappa(t),
q_R^\kappa(t),
q_\times^\kappa(t)
\right).
\]

The decomposition expresses hand-local and cross-hand computational
responsibilities. It does not require every coordinate to have an independent
human-readable meaning.

The role of \(q^\kappa\) is predictive and operational:

> Given the exact boundary state, \(q^\kappa(t)\) should retain the information
> required by the declared model to roll out future control load under legal
> continuations.

It may contain:

- decaying activations;
- role-specific recovery variables;
- short-term accumulation or recovery reservoirs;
- sustained forcing from open long notes;
- release transients;
- saturation or gating variables;
- hand-local interaction memory;
- cross-hand coordination memory;
- learned predictive features constrained by the declared rollout contract.

The formulation does not require

\[
q^\kappa(t)\ge0.
\]

Signed contrasts, gates, phase-like variables, or other internal coordinates
may be necessary. Nonnegativity and named channel semantics are imposed only on
readouts for which those properties are declared.

For fixed \(\kappa\), chart \(H\), and initial state \(q_0^\kappa\), the
operational trajectory is deterministic:

\[
q_H^\kappa(t)
=
\operatorname{LoadRoll}_\kappa
\left(
q_0^\kappa,x_0,H;t
\right).
\]

This determinism is a property of the canonical simulator, not evidence that
\(q^\kappa\) is the unique true state of an ideal or actual player.

Operational state and parameter uncertainty are distinct. A posterior or
ensemble over \(\kappa\) represents model uncertainty. It must not be collapsed
into one coordinate of \(q^\kappa\).

## 5. Hybrid exact and load dynamics

The control-load process is a hybrid dynamical system. Rows cause discrete
event transitions; time between rows advances clocks, occupancy-dependent
forcing, and continuous or discrete recovery dynamics.

For a row at time \(t_k\), use the right-continuous convention

\[
x_k^-
=
x(t_k^-),
\qquad
x_k^+
=
x(t_k)
=
x(t_k^+),
\]

\[
q_k^-
=
q(t_k^-),
\qquad
q_k^+
=
q(t_k)
=
q(t_k^+).
\]

A bare state at \(t_k\) therefore includes row \(k\) exactly once.

### Exact event transition

For a legal row,

\[
x_k^+
=
E_{\mathrm{exact}}(x_k^-,y_k).
\]

The action representation used by load dynamics is exact:

\[
a_k
=
\operatorname{Act}_\lambda(m_k).
\]

### Operational load event transition

The generic load reset is

\[
q_k^+
=
R_\kappa
\left(
q_k^-,
x_k^-,
x_k^+,
a_k
\right).
\]

The same row may therefore produce different load updates after different
histories.

An additive event injection

\[
q_k^+
=
q_k^-
+
J_\kappa
\left(
q_k^-,x_k^-,x_k^+,a_k
\right)
\]

is one possible parameterization, not a formulation-level requirement. The
generic reset permits saturation, redistribution, multiplicative gates,
release-specific transitions, and other nonlinear effects.

### Between-event flow

Let

\[
\Delta t_k=t_{k+1}-t_k.
\]

For \(0<\tau<\Delta t_k\), exact state evolves as

\[
x(t_k+\tau)
=
F_{\mathrm{exact},\tau}(x_k^+),
\]

where deterministic clocks advance and persistent finite state remains fixed
unless an explicitly declared time-triggered transition occurs.

Operational load evolves as

\[
q(t_k+\tau)
=
F_{\kappa,\tau}
\left(
q_k^+,
x(t_k+\tau)
\right).
\]

A continuous-time implementation may use

\[
\frac{d}{dt}q(t)
=
f_\kappa(q(t),x(t)),
\]

but an ODE is not required. Discrete decay, analytic kernels, state-space
updates, neural flows, or other causal semigroups are admissible if they obey
the same boundary semantics.

`LN_START` changes exact occupancy. The load model may use open occupancy to
sustain forcing. `LN_CLOSE` ends that occupancy and may produce a release
transient. These effects are chosen load priors or learned dynamics, not
consequences of the row alphabet alone.

Hard legality never reads \(q\). Operational load cannot turn an otherwise legal
row into an illegal row.

## 6. Hand symmetry and cross-hand communication

The canonical `(outer, inner)` coordinates make left-right symmetry explicit.
Let \(\mu\) be the mirror operator defined in
[notation.md](notation.md#4-lanes-hands-physical-roles-and-exact-actions).

The structural prior is imposed on the model family, not inferred separately
for each hand. For mirrored load dynamics \(\mu\kappa\), mirrored boundary, and
mirrored continuation,

\[
\operatorname{Roll}_{\mu\kappa}
\left(
\mu b^\kappa,
\mu Y;t
\right)
=
\mu
\operatorname{Roll}_\kappa
\left(
b^\kappa,Y;t
\right).
\]

For mirrored readout \(\mu\varrho\),

\[
G_{\mu\varrho}(\mu x,\mu q)
=
\mu G_\varrho(x,q).
\]

A symmetric canonical specification satisfies

\[
\mu\Theta_0=\Theta_0,
\]

and therefore

\[
\mathcal D_{\mu H}^{\Theta_0}(t)
=
\mu\mathcal D_H^{\Theta_0}(t).
\]

This is a chosen structural prior, not a claim that every human player is
left-right symmetric.

Mirror equivariance does not imply hand independence. A hand-local update may
depend on:

- its own exact state and operational state;
- the other hand's exact and operational state;
- the complete simultaneous row;
- exact cross-hand clocks;
- symmetric messages exchanged between the two hand representations.

Parameter sharing with symmetric cross-hand message passing is one admissible
implementation. The formulation fixes equivariant semantics, not a particular
neural topology.

Handedness asymmetry may enter through an explicitly asymmetric reference
specification or, more naturally for empirical prediction, through a separate
player profile. It must not arise from accidentally assigning unrelated
semantics to corresponding left- and right-hand channels.

## 7. Canonical rollout and continuation boundaries

The exact pre-chart state \(x_0\) contains:

- closed long-note occupancy on every lane;
- declared exact finite-state sentinels;
- declared exact clock sentinels.

For each load model \(\kappa\), the operational initial state is

\[
q_0^\kappa.
\]

The canonical rollout assumes:

- every chart action is executed successfully;
- hit timing is ideal;
- the fixed canonical hand-role mapping \(\lambda\) is used;
- player errors, misses, timing noise, and adaptive execution are not modeled.

For legal chart \(H\), define

\[
\left(
x_H(t),q_H^\kappa(t)
\right)
=
\operatorname{Roll}_\kappa
\left(
x_0,q_0^\kappa,H;t
\right).
\]

Its exact projection must agree with load-model-independent replay:

\[
\Pi_x
\operatorname{Roll}_\kappa
\left(
x_0,q_0^\kappa,H;t
\right)
=
\operatorname{Replay}(x_0,H;t).
\]

Changing \(\kappa\) or \(\varrho\) cannot change whether a chart has an open long
note or whether a row is legal.

At a continuation time \(t_b\), define the exact boundary

\[
\beta_b^{\mathrm{exact}}
=
(t_b,x_b),
\]

the load cache

\[
\sigma_b^\kappa
=
(t_b,q_b^\kappa),
\]

and the runtime boundary

\[
b_b^\kappa
=
(t_b,x_b,q_b^\kappa).
\]

For continuation \(Y\), branch-local rollout is

\[
\left(
\widetilde x_Y(t),
\widetilde q_Y^\kappa(t)
\right)
=
\operatorname{Roll}_\kappa
\left(
b_b^\kappa,Y;t
\right).
\]

The boundary includes every committed row at or before \(t_b\) and all silent
flow from the most recent row to \(t_b\).

The exact boundary is canonical after the exact-state schema is fixed. The load
cache is tied to \(\kappa\) and must be versioned. If \(\kappa\) changes, the
cache must be recomputed from a valid earlier checkpoint or from chart history.
The readout \(\varrho\) need not be stored in the boundary because it has no
state of its own in the base formulation.

Generation and post-chart analysis must share one rollout semantics:

\[
\operatorname{Roll}_{\mathrm{generation},\kappa_0}
=
\operatorname{Roll}_{\mathrm{analysis},\kappa_0}.
\]

Otherwise a chart can be optimized against one load definition and reported
under another.

The complete history remains available beside the boundary. Runtime state is a
continuation interface, not a replacement for symbolic history.

## 8. Gameplay-demand field and calibrated readouts

Let

\[
\mathcal C
=
\{c_1,\ldots,c_D\}
\]

be the hand-local demand channel set and

\[
\mathcal C_\times
=
\{c_1^\times,\ldots,c_{D_\times}^\times\}
\]

be the cross-hand channel set. \(D_\times=0\) is admissible; the value of an
explicit cross-hand component is an empirical question.

The readout maps exact and operational state to a nonnegative field:

\[
\mathcal D_H^\Theta(t)
=
G_\varrho
\left(
x_H(t),q_H^\kappa(t)
\right).
\]

For hand-local channels,

\[
\mathcal D_H^\Theta(h,c,t)
=
\left[
G_\varrho(x_H(t),q_H^\kappa(t))
\right]_{h,c},
\qquad
h\in\{L,R\},
\quad
c\in\mathcal C.
\]

For cross-hand channels,

\[
\mathcal D_H^\Theta(\times,c^\times,t)
=
\left[
G_\varrho(x_H(t),q_H^\kappa(t))
\right]_{\times,c^\times}.
\]

The distinction between \(q\) and \(\mathcal D\) is deliberate:

- \(q\) is internal state chosen for causal rollout sufficiency;
- \(\mathcal D\) is an exposed semantic and calibrated projection;
- invertibility between them is neither required nor assumed;
- several \(q\) states may produce the same current readout while differing in
  future response.

Possible hand-demand semantics include:

- role repetition or jack pressure;
- outer-inner alternation speed;
- local density;
- one-hand chord control;
- sustained long-note control;
- release timing;
- short-term stamina demand;
- role-occupancy pressure.

Possible cross-hand semantics include:

- coordinated simultaneity;
- hand-balance pressure;
- two-hand chord burden;
- synchronization or alternation demand.

These are proposed meanings, not automatically identified canonical axes.

The field is not a collection of independent scalar note difficulties. A row
changes state at its event time; recovery, sustained occupancy, nonlinear
interactions, and later state-conditioned events determine the realized
trajectory.

A per-note response shape may be defined for diagnostics as a counterfactual
isolated response under a declared no-future-event baseline. Under nonlinear or
state-conditioned dynamics, isolated responses are not additive components of
the realized chart trajectory.

A local intensity readout is

\[
i_H(t;\Theta)
=
r_{\mathrm{intensity}}
\left(
\mathcal D_H^\Theta(t);\varrho
\right).
\]

Section or map difficulty requires a separate aggregation:

\[
\operatorname{Diff}
\left(
H;W,\Theta
\right)
=
\operatorname{Agg}_{t\in W}
i_H(t;\Theta).
\]

Neither quantity is defined as a raw norm until channel units, interactions,
and calibration justify that operation.

The working style hypothesis is that chart style is partly expressed by the
temporal geometry of the demand field:

- which channels rise;
- whether they rise together or alternate;
- how long they remain elevated;
- how quickly they recover;
- whether the two hands are symmetric;
- how one demand type transitions into another;
- whether similar demand islands recur.

A map-level tag is only a weak observation of this geometry. For example, a
`jack-heavy` section predicate may take the form

\[
\frac{1}{|W|}
\int_W
\operatorname{Agg}_{h\in\{L,R\}}
\mathcal D_H^\Theta(h,\mathrm{jack},t)
\,dt
>
\tau,
\]

together with a minimum-duration condition. It does not imply that the whole
chart has that style at every time.

Demand magnitude as intensity, demand composition as style mixture, and demand
field sufficiency for style are hypotheses rather than definitions.

Most importantly,

\[
\mathcal D_H^\Theta(t)
\neq
\text{an observed player's fatigue, pain, attention, or miss probability}.
\]

Those quantities require an additional player model and observations.

## 9. Realized demand and desired demand are different objects

For a committed chart \(H\),

\[
\mathcal D_H^\Theta(t)
\]

is realized chart-induced demand under the declared canonical specification.
It is a consequence of chart replay and load rollout.

A generation planner may separately provide

\[
\mathcal D^\star(t)
\]

or a target set

\[
\mathfrak T(t)
\subseteq
\mathbb R_{\ge0}^{C}.
\]

These are desired demand plans. They express intended style or intensity and do
not describe the current player or current chart automatically.

For candidate continuation \(Y\), branch rollout produces

\[
\widetilde{\mathcal D}_Y^\Theta(t)
=
\operatorname{Readout}_\varrho
\left(
\operatorname{Roll}_\kappa(b^\kappa,Y;t)
\right).
\]

A target-matching term may compare

\[
\widetilde{\mathcal D}_Y^\Theta
\quad\text{with}\quad
\mathcal D^\star,
\]

but the target cannot override exact legality, candidate support, or long-note
obligations. Target feasibility is part of generation search.

The operational state \(q\), realized field \(\mathcal D_H\), and desired field
\(\mathcal D^\star\) therefore have distinct roles:

| Object | Role |
| --- | --- |
| \(q^\kappa(t)\) | Internal causal memory for future rollout. |
| \(\mathcal D_H^\Theta(t)\) | Demand induced by a concrete chart. |
| \(\mathcal D^\star(t)\) | Optional design target supplied to generation. |

## 10. Continuation-demand operator and predictive state semantics

A current scalar demand value is not enough to characterize continuation
behavior. The generation-relevant object is the response to possible future
legal continuations.

For runtime boundary

\[
b_t^\kappa=(t,x_t,q_t^\kappa)
\]

and legal continuation \(Y\) over horizon \([t,t+L]\), define the
continuation-demand operator

\[
\mathfrak C_\Theta
\left(
b_t^\kappa,Y
\right)
=
\left.
\mathcal D_{b_t^\kappa\oplus Y}^{\Theta}
\right|_{[t,t+L]}.
\]

Here \(b_t^\kappa\oplus Y\) means branch-local continuation from the boundary;
it does not mutate committed history.

The intended sufficiency contract for runtime state is:

\[
(x_t,q_t^\kappa)
=
(x'_t,q_t^{\prime,\kappa})
\]

implies

\[
\mathfrak C_\Theta(b_t^\kappa,Y)
=
\mathfrak C_\Theta(b_t^{\prime,\kappa},Y)
\]

for every continuation \(Y\) legal from both exact boundaries.

Equivalently, define a history relation at time \(t\):

\[
H\sim_{\Theta,t}H'
\]

when

\[
x_H(t)=x_{H'}(t)
\]

and, for every common legal continuation \(Y\),

\[
\mathfrak C_\Theta(b_{H,t}^\kappa,Y)
=
\mathfrak C_\Theta(b_{H',t}^\kappa,Y).
\]

A theoretically minimal continuation state represents the equivalence classes
of \(\sim_{\Theta,t}\). The formulation does not claim that a finite exact
representation is known. A learned or hand-designed \(q^\kappa\) is an
approximation whose adequacy must be tested.

This perspective replaces the question

> Which hidden parity state is the ideal player really in?

with the operational question

> Which information about committed control history is required to predict the
> response to declared future continuations?

For example, the relative preference for a future role can be represented as a
continuation-conditioned value

\[
V_h(\rho,\Delta\mid b_t^\kappa)
=
\operatorname{Cost}
\left(
\text{action on role }\rho\text{ at }t+\Delta
\mid
b_t^\kappa
\right),
\]

rather than as a posterior probability that the current hidden state “is” role
\(\rho\).

### Controlled continuation probes

Because all continuations cannot be enumerated, declare a probe bank

\[
\mathcal Y_{\mathrm{probe}}
=
\{Y_1,\ldots,Y_M\}.
\]

Possible probes include:

- outer-role jack after several controlled intervals;
- outer-inner alternation;
- chord-to-single continuation;
- long-note hold with free-role taps;
- long-note release followed by a chord;
- matched left and right mirror patterns;
- simultaneous two-hand chords;
- one-hand density bursts;
- controlled silence and recovery intervals.

For trajectory summary functionals \(\Psi_i\), define

\[
p_H(t)
=
\left[
\Psi_1
\left(
\mathfrak C_\Theta(b_{H,t}^\kappa,Y_1)
\right),
\ldots,
\Psi_M
\left(
\mathfrak C_\Theta(b_{H,t}^\kappa,Y_M)
\right)
\right].
\]

Probe responses can be used to:

- test whether \(q^\kappa\) preserves relevant future information;
- anchor named demand channels;
- compare alternative dynamics;
- validate mirror equivariance;
- reveal history shortcuts;
- define interpretable continuation costs.

The probe vector need not itself be the runtime state. It is an observable test
interface for state adequacy and readout semantics.

## 11. Optional extension to multiple physical executions

The base Pulsefield setting uses fixed lane-to-role mapping, so

\[
\mathcal E_\lambda(H)=\{u_H\}.
\]

Execution ambiguity should be introduced only when the formulation explicitly
allows alternatives, such as:

- variable fingering for one lane;
- hand crossing;
- alternative long-note holding strategies;
- player-specific role assignment;
- execution errors or adaptive recovery actions.

Let

\[
\mathcal E_\pi(H)
\]

be the admissible execution-trace set under an extended execution convention
\(\pi\). For each trace \(e\in\mathcal E_\pi(H)\), roll out independently:

\[
q_{H,e}^\kappa(t)
=
\operatorname{LoadRoll}_\kappa(H,e;t),
\]

\[
\mathcal D_{H,e}^{\Theta}(t)
=
\operatorname{Readout}_\varrho
\left(
x_{H,e}(t),q_{H,e}^\kappa(t)
\right).
\]

Without calibrated execution weights, preserve a set-valued demand object:

\[
\mathfrak D_H^\Theta
=
\left\{
\mathcal D_{H,e}^{\Theta}
\;\middle|\;
e\in\mathcal E_\pi(H)
\right\}.
\]

If a declared stochastic execution policy provides a measure

\[
\nu_\pi(de\mid H),
\]

its pushforward through complete rollout is

\[
\mathbb Q_H^\Theta
=
\left(
e\mapsto\mathcal D_{H,e}^{\Theta}
\right)_\#
\nu_\pi(de\mid H).
\]

Expectation, worst case, quantiles, or risk-sensitive functionals are applied
after per-execution rollout. There is generally no valid operation of averaging
execution traces before nonlinear rollout. In any representation where an
averaged trace summary \(\overline e\) is defined, one should not assume

\[
\operatorname{Roll}_\kappa(\overline e)
\neq
\int
\operatorname{Roll}_\kappa(e)
\,\nu_\pi(de\mid H).
\]

If execution policy itself depends on accumulating player load, then execution
and response form a coupled player-control process. That extension requires a
true player model and is not part of the canonical chart descriptor.

## 12. Player response is a separate observation layer

The canonical control-load specification describes a chart under a reference
execution convention. It does not identify how one concrete person performs or
feels.

Let

\[
\phi_{\mathrm{player}}
\]

contain player-specific properties such as:

- handedness and role asymmetry;
- speed and coordination capacity;
- stamina and recovery;
- timing variance;
- learned execution strategy;
- adaptation to repeated patterns;
- sensitivity to subjective burden.

Let \(e_{\mathrm{player}}\) be the player's actual execution trace and
\(O_{\mathrm{player}}\) observed outcomes such as hit timing, accuracy, misses,
early long-note releases, or subjective ratings. A future observation model
may take the form

\[
O_{\mathrm{player}}
\sim
p_\eta
\left(
O
\mid
H,
e_{\mathrm{player}},
\mathcal D_{H,e_{\mathrm{player}}}^\Theta,
\phi_{\mathrm{player}}
\right).
\]

A richer model may introduce a player-capacity state with its own dynamics and
feedback into execution policy. That state must not be silently identified with
canonical \(q^\kappa\).

The epistemic boundary is therefore

\[
H
\rightarrow
(x,u_H)
\rightarrow
q^\kappa
\rightarrow
\mathcal D_{H,e_{\mathrm{player}}}^\Theta
\rightarrow
O_{\mathrm{player}}.
\]

Without player observations, Pulsefield can define and evaluate the earlier
layers as reproducible chart descriptors. It cannot claim to have recovered an
actual player's hidden fatigue, pain, attention, or subjective difficulty.

Player data can nevertheless help calibrate \(\kappa\), \(\varrho\), or the
observation model. Calibration does not erase the conceptual separation between
chart-induced demand and player-specific response.

## 13. Identifiability and semantic anchoring

The operational state factorization is not unique. For any suitable invertible
transformation \(T\), define

\[
\widetilde q=T(q).
\]

Dynamics and readout can be transformed accordingly:

\[
\widetilde R
=
T\circ R\circ T^{-1},
\]

\[
\widetilde F
=
T\circ F\circ T^{-1},
\]

\[
\widetilde G
=
G\circ T^{-1}.
\]

The same observable demand field and continuation behavior may result. Chart
reconstruction or one scalar quality score therefore cannot establish that a
particular coordinate system is the unique natural player state.

The formulation responds by assigning different standards to \(q\) and
\(\mathcal D\):

- \(q\) is judged by causal rollout sufficiency, stability, efficiency, and
  predictive tests;
- a named coordinate of \(\mathcal D\) requires semantic anchoring and
  calibration;
- an unanchored learned projection must remain explicitly latent rather than
  receiving a gameplay name from post-hoc inspection.

Without anchors:

- latent rotations or nonlinear reparameterizations may preserve objectives;
- channel scale may be arbitrary;
- several event-response systems may generate similar final scores;
- models may exploit non-gameplay shortcuts;
- section-level tags may fail to identify local causal responses;
- one readout may hide future-response distinctions present in \(q\).

Possible anchors include:

- controlled synthetic continuation probes;
- expert section annotations;
- pairwise burden or style comparisons;
- player timing, error, and miss data;
- channel-specific monotonicity constraints;
- mirror and intervention tests;
- counterfactuals that hold the current row fixed while changing preceding
  occupancy, recency, or repetition state;
- explicitly declared units and calibration references.

A demand channel should be treated as either:

1. **named and anchored**, with explicit constraints and validation; or
2. **free and latent**, without an unsupported human-readable label.

Parameter uncertainty, execution-set uncertainty in an extended model, future
chart uncertainty, and player-response noise are separate objects. They must
not be collapsed into one generic “state uncertainty.”

Legacy chart-reconstruction objectives do not by themselves identify a causal
control-load state or readable demand basis. See the repository
[legacy code boundary](../../README.md#legacy-code-boundary).

## 14. Falsifiable hypotheses

1. **Exact role-set state is sufficient for base chord semantics.**  
   Exact multi-role action history plus operational load should match or
   outperform a belief-valued parity representation on controlled chord
   continuations. If a separate execution mixture consistently improves
   prediction under the fixed lane-role mapping, the exact-state schema or base
   execution assumptions are incomplete.

2. **State-conditioned load is more informative than fixed note kernels.**  
   The model should distinguish controlled cases where the same row occurs after
   different occupancy, recency, chord, or repetition histories. Failure on
   those contrasts rejects the additional state conditioning for the tested
   channels.

3. **Continuation probes expose state insufficiency.**  
   Histories mapped to the same runtime state should produce matched responses
   on held-out probe continuations. Systematic differences imply that the state
   is not sufficient for the declared rollout family.

4. **Structural hand symmetry is useful.**  
   Mirror-equivariant exact semantics, dynamics, and readouts should improve
   mirrored consistency and sample efficiency without preventing explicitly
   modeled player asymmetry. No controlled benefit, or systematic harm, argues
   for revising the prior.

5. **Demand trajectories contain information beyond event counts.**  
   Calibrated trajectories should predict held-out expert comparisons, section
   annotations, or player outcomes better than matched density, event-count,
   and global-tag baselines. Otherwise the claimed trajectory information has
   not been established.

6. **Cross-hand demand may be non-additive.**  
   An explicit \(q_\times\) or \(\mathcal D_\times\) should improve controlled
   two-hand coordination predictions beyond hand-local states and exact
   simultaneous-row facts. No gain supports removing the cross-hand component.

7. **Separating operational state from semantic readout improves robustness.**  
   Models allowed to retain predictive internal state while constraining only
   the exposed demand basis should outperform models forced to make every
   latent coordinate named and nonnegative, without sacrificing readout
   interpretability.

## 15. Open definitions

- What is the smallest exact chart-format state sufficient for legality?
- Which exact canonical-control summaries should be cached in \(x^{\mathrm{ctrl}}\),
  and which should remain derivable from \(H_k\)?
- What dimension and factorization of \(q^\kappa\) are sufficient for held-out
  continuation responses?
- Which event and flow dynamics adequately express recovery, long-note
  occupancy, chord interaction, and release?
- Which demand channels can be semantically anchored, and in what units?
- Which continuation probes best distinguish relevant gameplay histories?
- Is a nonempty cross-hand operational or demand component necessary?
- How should local demand aggregate into intensity, section difficulty, and
  map-level judgments?
- Should generation use an explicit desired demand plan, a target set, or only
  implicit control?
- Under what future extension would multiple physical executions become
  necessary rather than artificial?
- Which player observations are required to separate chart-induced demand from
  player capacity and execution strategy?

## 16. Scope boundary

This page fixes:

- the separation between exact state, operational load, semantic demand, and
  player response;
- deterministic base execution under the fixed 4K role mapping;
- exact role-set semantics for chords;
- hybrid event and between-event rollout contracts;
- exact, derived-load, and runtime continuation boundaries;
- the distinction between realized and desired demand;
- continuation-response semantics for state sufficiency;
- the extension contract for multiple admissible executions;
- identifiability and anchoring requirements for named demand channels.

This page does not fix:

- a concrete exact-state inventory beyond required semantics;
- the architecture or dimension of \(q^\kappa\);
- a particular ODE, state-space model, neural network, or kernel family;
- the final demand channel inventory or calibration dataset;
- a concrete player observation model;
- candidate proposal, structured decoding, or stable-prefix algorithms.

The final group of generation decisions belongs to
[notation.md](notation.md). Concrete architectures, experiments, and evidence
belong in `docs/research/` rather than in the formulation layer.