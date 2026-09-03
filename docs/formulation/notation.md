# Pulsefield generation notation

This page defines the chart language, exact legality, target continuation
distribution, rolling candidate lifecycle, branch isolation, and stable-prefix
commit semantics used by Pulsefield.

The canonical gameplay frontier, demand state, section style semantics, and
style-tag supervision are defined in
[gameplay-state.md](gameplay-state.md).

All materialized row times are absolute audio times in seconds.

The page distinguishes four kinds of statements:

- **Fixed formal invariant**: every conforming implementation must preserve it.
- **Canonical project convention**: Pulsefield currently adopts it; changing it
  changes the formulation rather than only model parameters.
- **Open implementation choice**: alternative architectures may instantiate it
  differently without changing the target problem.
- **Falsifiable working hypothesis**: a research claim that must be tested
  against controlled alternatives.

## 1. Core generation problem

Let \(A\) be the complete source audio and

\[
X=\Phi(A)
\]

its complete-song representation. All of \(X\) is available before chart
generation begins.

Let

\[
H_k=(y_1,\ldots,y_k),
\qquad
y_i=(t_i,m_i)
\]

be the committed materialized chart history. Let \(g\) be the exact time
through which both row and no-row decisions are fixed.

The history \(H_k\) stores only materialized nonempty rows. The absence of
additional rows at or before \(g\) is represented by the fixed-through time
\(g\), not by inserting null rows into \(H_k\).

Let

\[
x_g
=
\operatorname{Replay}_0(H_k;g)
\]

be the exact continuation state induced by the committed chart under the fixed
chart semantics and canonical hand-role mapping. Define the exact boundary

\[
\beta_g=(g,x_g).
\]

For a future window

\[
W=(g,e],
\qquad
g<e\le T,
\]

a continuation is an ordered sequence of complete rows

\[
Y_W
=
\bigl(
(\tau_1,n_1),
\ldots,
(\tau_N,n_N)
\bigr),
\qquad
N\ge0,
\]

with

\[
g<\tau_1<\cdots<\tau_N\le e.
\]

When \(N=0\),

\[
Y_W=().
\]

The target Pulsefield generation problem is

\[
Y_W
\sim
p_\theta
\left(
Y_W
\mid
X,
H_k,
\beta_g,
W,
[c_W^{\mathrm{style}}],
[d_W^\star]
\right),
\]

subject to

\[
H_k\oplus Y_W
\in
\mathcal H_{\mathrm{legal}}.
\]

Here:

- \(c_W^{\mathrm{style}}\) is an optional requested section-style control;
- \(d_W^\star\) is an optional desired gameplay-demand target;
- square brackets indicate that the argument may be absent;
- \(\oplus\) denotes legal ordered chart continuation.

The ordinary uncontrolled case is therefore

\[
Y_W
\sim
p_\theta
\left(
Y_W
\mid
X,H_k,\beta_g,W
\right).
\]

The absence of a style or demand request does not mean that the output has no
style or no demand. Every materialized continuation induces both under the
definitions in [gameplay-state.md](gameplay-state.md).

The final decision object is always the complete row sequence \(Y_W\).
Temporal paths, candidate assignments, planning tokens, demand trajectories,
style profiles, edits, and search branches are auxiliary objects. They do not
become chart history.

### Derived gameplay state

For the fixed canonical gameplay specification, the committed chart also
induces a gameplay-demand state

\[
d_g
=
\operatorname{DemandRoll}_0(H_k;g).
\]

The value \(d_g\) may be recomputed from \(H_k\) or stored as a versioned
runtime cache. It is not an independent user input and does not change the
semantic target distribution: it is already determined by the committed chart
and the declared canonical specification.

The runtime demand state is not a general-purpose embedding of chart history
and is not expected to preserve every arrangement distinction relevant to
style. The committed chart \(H_k\), exact boundary \(\beta_g\), and derived
demand state \(d_g\) remain separate conditioning information.

An implementation may therefore evaluate

\[
p_\theta
\left(
Y_W
\mid
X,H_k,\beta_g,d_g,W,
[c_W^{\mathrm{style}}],
[d_W^\star]
\right)
\]

for efficiency. This is a computational factorization of the same problem, not
a different target problem.

## 2. Fixed lifecycle invariants

Every conforming implementation preserves the following invariants:

1. A committed chart time carries one complete nonempty row.
2. Materialized row times are strictly increasing.
3. Candidate nulls and lane-level `EMPTY` are different symbols.
4. Planning tokens, edits, provisional decisions, and losing branches never
   enter committed history.
5. The committed prefix is immutable.
6. The fixed-through time may advance without adding a row.
7. Row index, candidate-refresh index, and stable-prefix revision index are
   different objects.
8. Exact chart legality depends only on exact chart state.
9. Demand magnitude, style preference, candidate score, or model probability
   cannot make an illegal row legal or a legal row illegal.
10. A complete chart closes every long note.
11. An intermediate boundary may carry open long notes and their future close
    obligations.
12. Committing an `LN_START` commits the open occupancy and future obligation,
    not one particular provisional `LN_CLOSE`.
13. The full committed history remains available alongside any finite runtime
    state.
14. A provisional branch never mutates committed history, fixed no-row
    decisions, exact state, or derived gameplay caches.
15. Committed rows, fixed-through time, and exact boundary state advance
    atomically.

## 3. Audio and absolute time

Let the song duration be \(T\). The audio representation is defined over

\[
t\in[0,T].
\]

The value \(X(t)\) denotes features associated with the neighborhood of absolute
audio time \(t\).

Pulsefield separates audio availability from chart causality:

- future audio is available;
- future chart rows are not available;
- committed history contains only past chart decisions;
- a model must not condition on an uncommitted future chart decision as though
  it were history.

An implementation may internally use beat coordinates, time-shift tokens,
relative offsets, lattices, or segment-local coordinates. These are encoding
choices. A materialized row always has one absolute audio timestamp.

Rows at time \(0\) are legal. An implementation requiring a boundary before the
first possible row may use the formal sentinel \(0^-\). Its numeric audio
position remains \(0\).

## 4. Lanes, hands, and canonical roles

The lane, hand, and within-hand role sets are

\[
\mathcal L=\{1,2,3,4\},
\qquad
\mathcal H=\{L,R\},
\qquad
\mathcal R=\{\mathrm{outer},\mathrm{inner}\}.
\]

Serialized chart rows always use lane order

\[
(1,2,3,4).
\]

The canonical hand-role-to-lane map is

\[
\lambda:
\mathcal H\times\mathcal R
\rightarrow
\mathcal L
\]

with

\[
\begin{aligned}
\lambda(L,\mathrm{outer}) &= 1,
&
\lambda(L,\mathrm{inner}) &= 2,
\\
\lambda(R,\mathrm{inner}) &= 3,
&
\lambda(R,\mathrm{outer}) &= 4.
\end{aligned}
\]

Thus

\[
\mathcal L_L=\{1,2\},
\qquad
\mathcal L_R=\{3,4\}.
\]

For row \(m_k\), define the canonical action pair of hand \(h\) as

\[
a_k^h
=
\left(
a_{k,h,\mathrm{outer}},
a_{k,h,\mathrm{inner}}
\right),
\]

where

\[
a_{k,h,\rho}
=
(m_k)_{\lambda(h,\rho)}.
\]

Therefore

\[
a_k^L
=
\bigl((m_k)_1,(m_k)_2\bigr),
\]

and

\[
a_k^R
=
\bigl((m_k)_4,(m_k)_3\bigr).
\]

Write

\[
a_k
=
\operatorname{Act}_\lambda(m_k)
=
(a_k^L,a_k^R).
\]

The pair \((a_k^L,a_k^R)\) enumerates lanes as \((1,2,4,3)\), not serialized
chart order. It must not be directly concatenated to reconstruct a row.

The exact active-role set of hand \(h\) is

\[
A_k^h
=
\left\{
\rho\in\mathcal R
\;\middle|\;
a_{k,h,\rho}\neq\mathrm{EMPTY}
\right\}.
\]

A one-hand chord activates both roles. It does not create uncertainty over
which role was used.

The operator \(\operatorname{Act}_\lambda\) is a deterministic coordinate
projection of one serialized row. When a sequence-level hand-role view is
useful, extend it pointwise over any chart interval:

\[
\operatorname{Act}_\lambda(H|_W)
=
\bigl(
(t_k,\operatorname{Act}_\lambda(m_k))
\bigr)_{t_k\in W}.
\]

This derived view contains no information beyond \(H|_W\). It is not a separate
execution state, does not infer fingering or physical player actions, and need
not be stored as a global chart object. Serialized chart rows remain the source
of truth.

### Mirror operator

Define the left-right mirror of a row as

\[
\mu(m_1,m_2,m_3,m_4)
=
(m_4,m_3,m_2,m_1).
\]

This exchanges hands while preserving the semantic roles `outer` and `inner`.

For a chart,

\[
\mu H
=
\bigl((t_k,\mu m_k)\bigr)_{k=1}^{|H|}.
\]

The mirror operator extends to exact state, gameplay-demand state, future
continuations, and model outputs as specified in
[gameplay-state.md](gameplay-state.md).

## 5. Row language and long notes

Each lane uses the row-local action alphabet

\[
\mathcal S
=
\{0,1,2,3\}
\equiv
\{
\mathrm{EMPTY},
\mathrm{TAP},
\mathrm{LN\_START},
\mathrm{LN\_CLOSE}
\}.
\]

The codes are

\[
0=\mathrm{EMPTY},
\qquad
1=\mathrm{TAP},
\qquad
2=\mathrm{LN\_START},
\qquad
3=\mathrm{LN\_CLOSE}.
\]

Their lane-occupancy transitions are:

| Code | Action | Valid occupancy transition |
| ---: | --- | --- |
| \(0\) | `EMPTY` | closed \(\to\) closed or open \(\to\) open |
| \(1\) | `TAP` | closed \(\to\) closed |
| \(2\) | `LN_START` | closed \(\to\) open |
| \(3\) | `LN_CLOSE` | open \(\to\) closed |

Every other action-occupancy pair is illegal.

In particular:

- `EMPTY` is a row-local no-op;
- `EMPTY` on an open lane preserves the long note;
- `TAP` cannot occur on an already open lane;
- `LN_START` cannot occur on an already open lane;
- `LN_CLOSE` cannot occur on a closed lane.

A complete materialized row belongs to

\[
\mathcal M
=
\mathcal S^4
\setminus
\{(0,0,0,0)\}.
\]

A row is

\[
y_k=(t_k,m_k),
\qquad
m_k\in\mathcal M.
\]

A chart contains at most one complete row at a time:

\[
t_1<t_2<\cdots<t_k.
\]

## 6. Exact state and chart legality

Let

\[
x_H(t)
=
\operatorname{Replay}_0(H;t)
\]

be the exact state obtained by replaying every committed row at or before \(t\)
and advancing exact deterministic clocks through silent time.

Write

\[
x_H(t)
=
\left(
x_H^{\mathrm{fmt}}(t),
x_H^{\mathrm{ctrl}}(t)
\right).
\]

The chart-format component contains facts required for hard legality, including:

- lane-wise long-note occupancy;
- pending format obligations;
- any additional exact format state required by the chart language.

The canonical-control component may contain exact deterministic summaries used
to continue gameplay rollout, including:

- recent active-role sets;
- per-role action clocks;
- exact release or occupancy clocks;
- exact cross-hand timing facts.

The exact row transition is a partial operator

\[
x_k^+
=
E_{\mathrm{exact}}(x_k^-,y_k).
\]

A row is chart-legal exactly when its chart-format projection is defined:

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

Gameplay demand, model score, style, and musical correspondence do not
participate in this predicate.

The exact boundary at fixed-through time \(g\) is

\[
\beta_g
=
(g,x_g),
\qquad
x_g=x_{H_k}(g).
\]

The boundary includes:

- every committed row at or before \(g\);
- every fixed no-row decision at or before \(g\);
- all exact silent-time evolution from the most recent row to \(g\).

## 7. Legal continuation space

For scene

\[
\Xi_g
=
(X,H_k,\beta_g,W),
\]

define the legal continuation set

\[
\mathcal V^{\mathrm{legal}}(\Xi_g).
\]

A continuation

\[
Y_W
=
\bigl(
(\tau_1,n_1),
\ldots,
(\tau_N,n_N)
\bigr)
\]

belongs to \(\mathcal V^{\mathrm{legal}}(\Xi_g)\) if and only if:

1. every \(\tau_i\in W\);
2. row times are strictly increasing;
3. every \(n_i\in\mathcal M\);
4. the committed prefix \(H_k\) is unchanged;
5. exact replay from \(\beta_g\) accepts every row transition;
6. the resulting exact state records every open long note and future
   obligation;
7. if \(e=T\), every long note is closed by song end.

An intermediate continuation may end with open long notes.

The legal set does not depend on:

- candidate membership;
- style tags;
- requested style;
- gameplay-demand magnitude;
- requested demand;
- alternation or repetition preference;
- hand balance;
- musical quality;
- model likelihood.

These quantities may affect preference among legal continuations, but never
hard legality.

The target model is a distribution supported on the legal set:

\[
p_\theta(Y_W\mid\Xi_g,\ldots)=0
\qquad
\text{for }
Y_W\notin\mathcal V^{\mathrm{legal}}(\Xi_g).
\]

For continuous-time output spaces, a declared base measure, point-process
formulation, or equivalent construction is required. A finite normalized sum
must not be written as though the support were automatically discrete.

## 8. Optional generation controls

Pulsefield supports two conceptually distinct optional controls.

### Style request

A style request is written

\[
c_W^{\mathrm{style}}.
\]

It asks the generator to favor one or more concepts from the versioned
Pulsefield gameplay-style vocabulary defined in gameplay-state.md.

The vocabulary is player- and mapper-facing, chart-intrinsic in the current
scope, multi-label, and not identified with the complete community-tag
catalogue. Its coordinates are not assumed to be orthogonal generative
factors.

The detailed semantics of realized style, requested style, mixtures, and
annotation are defined in
[gameplay-state.md](gameplay-state.md#15-realized-style-requested-style-realized-demand-and-desired-demand).

### Demand target

A desired gameplay-demand trajectory is written

\[
d_W^\star(t),
\qquad
t\in W.
\]

A more general target may be a time-varying admissible set

\[
\mathfrak D_W^\star(t)
\subseteq
\mathcal D_0.
\]

A scalar intensity envelope may be used as a lower-dimensional interface:

\[
i_W^\star(t).
\]

These targets are design requests. They are not the realized demand state of
the committed chart.

### Control semantics

Neither control is mandatory.

A control cannot override:

- chart legality;
- committed history;
- open long-note obligations;
- candidate support in a support-restricted implementation;
- song-end closure requirements.

A requested style or demand trajectory may be infeasible in the current
window. Generation therefore searches for a legal compromise or reports the
control as infeasible; it does not reinterpret an impossible request as a hard
chart command.

Sampling temperature is separate from both controls. It changes decoding
entropy, not gameplay intensity or semantic style.

## 9. Rolling candidate lifecycle

Candidate information is an optional computational layer. It is not the target
chart space.

Let

\[
r=0,1,2,\ldots
\]

index candidate refreshes.

At refresh \(r\):

- \(H_{k_r}\) is the committed history snapshot;
- \(f_r\) is the fixed-through time;
- \(L_r>0\) is the candidate horizon length;
- \(e_r=\min(T,f_r+L_r)\);
- \(W_r=(f_r,e_r]\).

Candidate information is computed as

\[
\Gamma_r
=
G_\phi
\left(
X,H_{k_r};W_r
\right).
\]

The high-recall candidate layer does not receive

\[
c_W^{\mathrm{style}}
\quad\text{or}\quad
d_W^\star.
\]

This is the canonical separation between musical opportunity support and
style-conditioned chart generation.

Style-agnostic does not mean history-blind. The candidate predictor may use
committed history to preserve temporal continuity, account for existing long
notes, and expose useful rhythmic support. It must not use the requested style
or demand target to prematurely remove otherwise plausible opportunities.

The object \(\Gamma_r\) may contain:

- a finite opportunity bank;
- a time lattice;
- point-process or hazard parameters;
- dense rhythmic evidence;
- candidate probabilities or scores;
- path proposals;
- another high-recall temporal support representation.

It need not be calibrated or normalized.

### Frozen snapshot semantics

One candidate snapshot may support several stable-prefix revisions.

Let

\[
s=0,1,2,\ldots
\]

index revisions under refresh \(r\).

Initially,

\[
k_{r,0}=k_r,
\qquad
g_{r,0}=f_r.
\]

At revision \(s\), the current remaining horizon is

\[
W_{r,s}
=
(g_{r,s},e_r].
\]

Generation uses the current committed history and exact boundary:

\[
\Xi_{r,s}
=
\left(
X,
H_{k_{r,s}},
\beta_{r,s},
\Gamma_r,
W_{r,s}
\right),
\]

where

\[
\beta_{r,s}
=
(g_{r,s},x_{r,s}).
\]

During the lifetime of \(\Gamma_r\), a partial commit may change

\[
H_{k_{r,s}},
\qquad
g_{r,s},
\qquad
x_{r,s},
\qquad
d_{r,s},
\]

but it does not change the provenance of \(\Gamma_r\).

Recomputing candidate information from the advanced history creates a new
refresh \(\Gamma_{r+1}\). It is not an implicit mutation of \(\Gamma_r\).

This convention permits expensive candidate computation to be reused while
making candidate staleness measurable.

## 10. Model-reachable support

For active candidate information \(\Gamma_r\), define the model-reachable set

\[
\mathcal V^{\Gamma}_{r,s}
\subseteq
\mathcal V^{\mathrm{legal}}(\Xi_{r,s}).
\]

It contains the legal continuations representable under the current candidate,
proposal, tokenization, and decoding contract.

For a hard opportunity bank, all generated row times must belong to the active
candidate suffix.

For a soft candidate field, \(\Gamma_r\) may affect score without excluding
times, allowing

\[
\mathcal V^{\Gamma}_{r,s}
=
\mathcal V^{\mathrm{legal}}(\Xi_{r,s}).
\]

If a desirable legal continuation is absent from
\(\mathcal V^{\Gamma}_{r,s}\), this is a support failure, not a chart-legality
failure.

The target problem is defined on

\[
\mathcal V^{\mathrm{legal}},
\]

while a concrete implementation may approximate it on

\[
\mathcal V^\Gamma.
\]

This distinction must not be reversed.

For a support-independent quality functional \(\overline S\), the conceptual
support gap is

\[
\Delta_{\mathrm{support}}
=
\sup_{Y\in\mathcal V^{\mathrm{legal}}}
\overline S(Y)
-
\sup_{Y\in\mathcal V^\Gamma}
\overline S(Y).
\]

The first term is generally intractable. Oracle and controlled representation
experiments may nevertheless estimate whether the candidate representation
excludes high-quality continuations.

Pointwise timing recall is necessary but not sufficient to characterize this
downstream gap.

## 11. Opportunity banks, temporal paths, and row materialization

When \(\Gamma_r\) contains a concrete opportunity bank, write

\[
U_r
=
\bigl(
(u_j,e_j)
\bigr)_{j=1}^{M_r},
\]

with

\[
f_r<u_1<\cdots<u_{M_r}\le e_r.
\]

Here \(u_j\) is an absolute candidate time and \(e_j\) is candidate-associated
information.

At revision \(s\), define the active candidate indices

\[
J_{r,s}
=
\{j\mid u_j\in W_{r,s}\}.
\]

A complete candidate-indexed draft is

\[
B_{r,s}
=
(b_j)_{j\in J_{r,s}},
\]

where

\[
b_j\in\mathcal B,
\qquad
\mathcal B
=
\{\bot\}
\uplus
\mathcal M.
\]

The distinguished candidate-null symbol satisfies

\[
\bot\notin\mathcal M.
\]

Its semantics are:

- \(b_j=\bot\): no row exists at candidate time \(u_j\);
- \(b_j\in\mathcal M\): one complete row exists at \(u_j\);
- an `EMPTY` coordinate inside \(b_j\) means only that one lane has no action;
- \((0,0,0,0)\) is never a materialized row.

The temporal selection path is

\[
P_{r,s}
=
(p_j)_{j\in J_{r,s}},
\qquad
p_j
=
\mathbf 1[b_j\neq\bot].
\]

The path selects times but does not determine lane actions.

Candidate-row extraction is

\[
\operatorname{Rows}_{U_r}(B)
=
\left(
(u_j,b_j)
\right)_{
\substack{
j\in J_{r,s}\\
b_j\neq\bot
}
}.
\]

A temporal path \(P\), a complete draft \(B\), and the extracted row sequence
\(Y(B)\) are distinct objects.

### Completion-aware path semantics

For path \(P\), define its legal row-completion set

\[
\mathcal C(P)
=
\left\{
B
\;\middle|\;
\operatorname{sel}(B)=P,
\;
Y(B)\in\mathcal V^\Gamma_{r,s}
\right\}.
\]

If

\[
\mathcal C(P)=\varnothing,
\]

the path has no legal row materialization under the current boundary and
support.

A staged implementation may factorize generation as

\[
p(P,B\mid\cdot)
=
p(P\mid\cdot)
p(B\mid P,\cdot),
\]

but the target object remains the complete row sequence.

A path may be irreversibly frozen before full row materialization only when:

1. it has at least one legal completion;
2. its evaluation accounts for the quality or probability mass of its row
   completions;
3. the approximation error introduced by early freezing is measured.

Otherwise the system has reduced the task to position-only optimization and
may select a musically plausible path with incoherent or impossible
choreography.

## 12. Branch-local gameplay rollout and whole-continuation quality

Every legal continuation induces an exact and gameplay-demand rollout under the
fixed canonical gameplay specification:

\[
\mathcal R_0
\left(
Y_W
\mid
\beta_g,d_g
\right)
=
\left(
\widetilde x_Y(t),
\widetilde d_Y(t)
\right)_{t\in W}.
\]

The exact projection must agree with exact replay:

\[
\widetilde x_Y(t)
=
\operatorname{Replay}_0(\beta_g,Y_W;t).
\]

The demand projection is derived from the same branch:

\[
\widetilde d_Y(t)
=
\operatorname{DemandRoll}_0
\left(
\beta_g,d_g,Y_W;t
\right).
\]

Branch rollout is provisional and cannot mutate committed state.

Let

\[
S_\theta
\left(
Y_W;
\Xi_{r,s},
\mathcal R_0(Y_W),
[c_W^{\mathrm{style}}],
[d_W^\star]
\right)
\]

be a whole-continuation quality score.

Its diagnostic responsibilities may include:

- correspondence with audio and musical structure;
- continuity with committed chart history;
- exact gameplay continuity;
- natural style and motif continuity learned from data;
- the induced demand trajectory;
- optional requested-style compatibility;
- optional desired-demand compatibility.

The formulation does not require an additive decomposition or statistical
independence between these responsibilities.

For a finite reachable set, the score may define

\[
p_\theta
\left(
Y_W
\mid
\Xi_{r,s},
[c_W^{\mathrm{style}}],
[d_W^\star]
\right)
=
\frac{
\mathbf 1[
Y_W\in\mathcal V^\Gamma_{r,s}
]
\exp S_\theta(Y_W)
}{
Z_\theta
},
\]

where

\[
Z_\theta
=
\sum_{Y'\in\mathcal V^\Gamma_{r,s}}
\exp S_\theta(Y').
\]

Exact partition-function computation is not required. Any beam, sampling,
diffusion, dynamic-programming, or refinement approximation must remain
identified as an approximation to the target continuation distribution.

A candidate score, local onset probability, diffusion residual, or path
proposal likelihood is not automatically a calibrated whole-continuation
quality score.

Variable-length scoring must avoid trivial preferences for either silence or
density. Per-event, per-time, and whole-window terms require explicit
normalization or calibration.

## 13. Branch isolation and stable-prefix commit

Every proposal branch starts from the same committed objects:

\[
\left(
H_{k_{r,s}},
\beta_{r,s},
d_{r,s}
\right).
\]

Each branch owns provisional copies of:

\[
\left(
\widetilde H,
\widetilde x,
\widetilde d
\right).
\]

After inference or reranking, the system may select a stable time prefix of one
winning legal branch.

Suppose all row and no-row decisions are fixed through

\[
g'>g_{r,s}.
\]

Let

\[
I=(g_{r,s},g']
\]

and let \(Y_I\) be the rows selected in that interval.

If \(Y_I\) contains \(n\ge0\) rows, then

\[
k_{r,s+1}
=
k_{r,s}+n,
\]

and

\[
H_{k_{r,s+1}}
=
H_{k_{r,s}}
\mathbin{\|}
Y_I,
\]

where \(\mathbin{\|}\) denotes ordered history concatenation.

The exact state advances by replay:

\[
x_{r,s+1}
=
\operatorname{Replay}_0
\left(
\beta_{r,s},
Y_I;
g'
\right),
\]

and

\[
\beta_{r,s+1}
=
(g',x_{r,s+1}).
\]

The derived demand state advances as

\[
d_{r,s+1}
=
\operatorname{DemandRoll}_0
\left(
\beta_{r,s},
d_{r,s},
Y_I;
g'
\right).
\]

The semantic commit is the atomic promotion

\[
\left(
H_{k_{r,s}},
g_{r,s},
x_{r,s}
\right)
\longrightarrow
\left(
H_{k_{r,s+1}},
g',
x_{r,s+1}
\right).
\]

The versioned demand cache may be promoted in the same transaction, but it
remains derived and recomputable.

If \(n=0\), materialized history is unchanged while \(g\), exact clocks, and
demand recovery still advance through silent time.

The commit invariants are:

- losing branches cannot mutate committed rows or state;
- committing `LN_START` commits open occupancy and a future close obligation;
- a provisional close outside the committed interval remains provisional;
- an intermediate boundary may carry open long notes;
- a partial commit under the same refresh continues to use the original
  \(\Gamma_r\);
- a fresh candidate computation creates \(\Gamma_{r+1}\).

## 14. Worked example

Suppose

\[
H_3
=
\bigl(
(29.40,1000),
(29.70,0020),
(29.90,0100)
\bigr),
\]

and

\[
g=29.90.
\]

Lane 3 is open because the row at \(29.70\) used `LN_START` on lane 3 and the
row at \(29.90\) used `EMPTY` on lane 3.

Assume the opportunity bank begins

\[
U
=
\bigl(
(30.00,e_1),
(30.12,e_2),
(30.25,e_3)
\bigr).
\]

Consider path

\[
P=(1,1,0).
\]

It selects rows at \(30.00\) and \(30.12\), but does not specify their actions.

Three drafts are

\[
B^A=(0030,1000,\bot),
\]

\[
B^B=(0020,1000,\bot),
\]

and

\[
B^C=(0001,1000,\bot).
\]

For \(B^A\):

- `0030` legally closes lane 3 at \(30.00\);
- `1000` taps lane 1 at \(30.12\);
- the completion is chart-legal.

For \(B^B\):

- `0020` attempts another `LN_START` on already open lane 3;
- exact replay rejects the first proposed row;
- the completion is chart-illegal.

For \(B^C\):

- `0001` taps lane 4;
- lane 3 remains open;
- the continuation is legal at an intermediate horizon;
- it is illegal at song end unless a later row closes lane 3.

The three drafts share the same temporal path but differ in:

- exact legality;
- long-note obligations;
- per-hand action history;
- induced gameplay-demand trajectory;
- possible section-style interpretation.

A style request or demand target may rank \(B^A\) and \(B^C\) differently, but
it cannot make \(B^B\) legal.

## 15. What is fixed and what remains open

### Fixed by the target problem

- The committed object is a sequence of complete nonempty rows.
- Materialized row identity uses absolute audio time.
- Complete audio is available before chart generation.
- Future chart decisions remain causal.
- Lane serialization and the base hand-role mapping are fixed.
- Chart legality is decided by exact replay.
- Candidate support is distinct from legality.
- Candidate null is distinct from lane-level `EMPTY`.
- The target distribution is defined over legal row continuations.
- Style and desired demand are optional controls.
- Realized demand is derived from a materialized chart.
- A candidate layer does not receive style or desired-demand controls.
- Temporal paths are auxiliary planning objects.
- Branch-local state is isolated.
- Only a selected stable time prefix mutates committed state.
- History, fixed-through time, and exact state advance atomically.

### Current project conventions

- Candidate information is computed in rolling windows.
- One candidate snapshot may support several stable-prefix revisions.
- Candidate information is designed as broad, high-recall musical opportunity
  support.
- Exact and demand rollout use the one fixed canonical gameplay specification.
- Left-right mirror structure is explicit.

### Open implementation choices

- audio representation \(X\);
- candidate representation and training objective;
- refresh trigger and horizon length;
- opportunity bank, lattice, hazard, or continuous-time support;
- direct, autoregressive, diffusion, dynamic-programming, sampling, or
  refinement generation;
- whether timing and row materialization are factorized;
- exact or approximate structured inference;
- score parameterization and variable-length normalization;
- stable-prefix confidence rule;
- internal tokenization;
- whether derived demand state is explicitly supplied to the generator;
- concrete style- and demand-control interfaces.

## 16. Falsifiable hypotheses and open questions

### Falsifiable hypotheses

1. A broad style-agnostic candidate layer can preserve high downstream support
   while leaving style-dependent density and omission decisions to the
   generator.

2. Completion-aware temporal-path evaluation improves choreography quality over
   position-only path ranking.

3. Exact branch-local replay reduces illegal long-note transitions and
   cross-window discontinuities.

4. Retaining structured uncertainty over complete continuations improves
   stable-prefix decisions over equally sized greedy local decoding.

5. Reusing one frozen candidate snapshot across several commits reduces
   computation without unacceptable staleness.

### Open questions

- Which candidate representation best preserves downstream chart quality?
- How should candidate support loss be estimated beyond pointwise timing recall?
- When should a candidate snapshot be refreshed?
- How far may a stable prefix advance without destroying useful revision?
- Which approximation best preserves path and materialization uncertainty?
- When is explicit branch demand rollout useful during generation rather than
  only during training and evaluation?
- Which score normalization prevents silence or density from dominating?
- Should a provisional `LN_CLOSE` ever be locked together with an earlier
  committed `LN_START`?
- How should incompatible style and demand requests be detected and reported?

## 17. Scope boundary

This page fixes:

- chart syntax and absolute-time identity;
- canonical lane and hand-role coordinates;
- exact legality and long-note obligations;
- committed-history and fixed-through semantics;
- the legal continuation target;
- optional style and demand control positions;
- rolling candidate support;
- temporal path and materialization distinctions;
- branch isolation and stable-prefix commit.

This page does not fix:

- the canonical gameplay-demand dimensions;
- moving-frontier semantics;
- section style signatures;
- map-level or section-level style annotations;
- style-tag observation models;
- concrete model architectures or training objectives.

Those gameplay and style concepts belong to
[gameplay-state.md](gameplay-state.md). Experimental evidence and concrete
architectures belong in `docs/research/`.
