# Generation notation and structured decision problem

This page owns the canonical vocabulary and target structured generation
problem for Pulsefield. It defines chart syntax, coordinate conventions,
committed history, rolling candidate snapshots, legal and model-reachable
completion spaces, whole-completion scoring, branch isolation, and stable-prefix
commit semantics.

It does not define the internal dynamics or empirical interpretation of the
control-load state. Those belong to the
[canonical control-load formulation](gameplay-state.md).

All materialized row times are absolute audio times in seconds.

The page distinguishes four kinds of statements:

- **Fixed formal invariant**: every conforming implementation must preserve it.
- **Canonical system convention**: Pulsefield currently adopts it; replacing it
  changes the formulation rather than only model parameters.
- **Open implementation choice**: alternative architectures may instantiate it
  differently without changing the target problem.
- **Falsifiable working hypothesis**: a research claim that must be evaluated
  against controlled alternatives.

## 1. Formal scene and target object

Let \(A\) be the complete source audio and

\[
X=\Phi(A)
\]

its complete-song feature field. Let

\[
H_k=(y_1,\ldots,y_k),
\qquad
y_i=(t_i,m_i)
\]

be the committed materialized chart history.

At candidate refresh \(r\), candidate information is computed from a frozen
history snapshot:

\[
\Gamma_r
=
G_\theta
\left(
X,
H_{k_r};
W_r
\right).
\]

One \(\Gamma_r\) may remain active across several stable-prefix revisions. At
revision \(s\), define the exact committed boundary

\[
\beta^{\mathrm{exact}}_{r,s}
=
\left(
g_{r,s},
x_{r,s}
\right),
\]

where \(g_{r,s}\) is the time through which both row and no-row decisions are
fixed and \(x_{r,s}\) is the exact replay state at that time.

For fixed load dynamics \(\kappa\), define the derived load boundary

\[
\sigma^\kappa_{r,s}
=
\left(
g_{r,s},
q^\kappa_{r,s}
\right)
\]

and the runtime rollout boundary

\[
b^\kappa_{r,s}
=
\left(
g_{r,s},
x_{r,s},
q^\kappa_{r,s}
\right).
\]

The exact generation scene is

\[
\Xi^{\mathrm{exact}}_{r,s}
=
\left(
X,
H_{k_{r,s}},
\beta^{\mathrm{exact}}_{r,s},
\Gamma_r,
W_{r,s}
\right).
\]

A load-aware evaluator additionally receives the derived state:

\[
\Xi^\kappa_{r,s}
=
\left(
X,
H_{k_{r,s}},
b^\kappa_{r,s},
\Gamma_r,
W_{r,s}
\right).
\]

Equivalently, \(\Xi^\kappa_{r,s}\) augments the exact scene with
\(q^\kappa_{r,s}\).

The candidate snapshot \(H_{k_r}\) may be older than the current committed
history \(H_{k_{r,s}}\). Generation always starts from the current history and
current boundary while \(\Gamma_r\) remains the frozen result of its original
refresh.

The final decision object is an ordered future sequence of complete rows:

\[
Y
=
\left(
(\tau_1,m_1),
\ldots,
(\tau_n,m_n)
\right),
\qquad
n\ge 0.
\]

When \(n=0\),

\[
Y=().
\]

The target problem separates three layers:

\[
\mathcal V^\Gamma_{r,s}
\subseteq
\mathcal V^{\mathrm{chart}}_{r,s},
\]

followed by a preference over reachable legal completions:

\[
Y^\star_{r,s}
\in
\arg\max_{Y\in\mathcal V^\Gamma_{r,s}}
S_\psi(Y).
\]

Here:

- \(\mathcal V^{\mathrm{chart}}_{r,s}\) contains chart-legal completions;
- \(\mathcal V^\Gamma_{r,s}\) contains legal completions reachable under the
  active candidate and proposal contract;
- \(S_\psi\) ranks reachable legal completions by musical, historical,
  gameplay, and optional target-matching quality.

The fixed lifecycle invariants are:

1. A committed chart time carries one complete nonempty row.
2. Materialized row times are strictly increasing.
3. Candidate nulls, planning tokens, edits, and losing-branch decisions never
   enter \(H_k\).
4. A fixed-through time may advance without adding a row; row index, refresh
   index, and stable-prefix revision index are different objects.
5. Committed history, fixed-through time, and exact boundary state advance
   atomically.
6. A provisional branch never mutates committed history, exact state, derived
   load state, or fixed no-row decisions.
7. Candidate-level absence and lane-level `EMPTY` are different symbols.
8. Hard chart legality reads exact chart-format state, not candidate score,
   demand magnitude, or stylistic preference.
9. A complete chart closes every long note; an intermediate boundary may carry
   open long notes and their future close obligations.
10. The full committed history remains available alongside any finite runtime
    state; runtime state is not required to summarize motif identity, mapper
    intent, or every long-range chart property.

The canonical system conventions are:

- all of \(X\) is available before chart generation begins;
- lanes are serialized in order \((1,2,3,4)\);
- hands use canonical within-hand coordinates `(outer, inner)`;
- the lane-to-role mapping is fixed in the base 4K setting;
- candidate information is computed in rolling windows and remains frozen
  during the lifetime of one refresh.

The structure of \(\Gamma_r\), the refresh policy, proposal architecture,
search procedure, internal tokenization, load model, and quality model remain
open unless constrained below.

## 2. Worked example

Suppose the committed history is

\[
H_3
=
\bigl(
(29.40,1000),
(29.70,0020),
(29.90,0100)
\bigr)
\]

and

\[
g_{r,s}=29.90.
\]

Lane 3 is open at this boundary because row 2 started a long note and row 3 used
`EMPTY` on lane 3.

Assume the active opportunity bank begins

\[
U_r
=
\bigl(
(30.00,e_1),
(30.12,e_2),
(30.25,e_3),
\ldots
\bigr).
\]

Consider the temporal selection mask

\[
P=(1,1,0).
\]

It selects row times \(30.00\) and \(30.12\), and selects no row at \(30.25\).
The path does not determine the row actions. Three materializations are

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

- `0001` taps lane 4 and leaves lane 3 open;
- the completion is legal at an intermediate horizon;
- it is illegal if the horizon reaches song end and no later row closes lane 3.

Thus one temporal path can have a legal closing materialization, an illegal
materialization, and a legal intermediate materialization with a different
future obligation. Their induced control-load trajectories may also differ.
Timestamp plausibility alone therefore cannot establish final choreography
quality.

If branch \(B^A\) wins but the stable prefix ends at \(30.12\), then

\[
H_3\longrightarrow H_5,
\qquad
g_{r,s+1}=30.12,
\]

while the candidate-null decision at \(30.25\) remains provisional. If the
stable prefix instead ends at \(30.25\), the no-row decision at \(30.25\) also
becomes fixed while the row index remains \(5\).

In either case, generation continues from the advanced history and boundary.
The active \(\Gamma_r\) remains the candidate information computed from its
older refresh snapshot until a new refresh is triggered.

## 3. Audio and absolute time

Let the song duration be \(T\). The audio feature field is defined over

\[
t\in[0,T].
\]

\(X(t)\) denotes features associated with the neighborhood of absolute audio
time \(t\). Future audio is known; future chart rows are not.

An implementation may internally use time-shift tokens, relative positions,
beat coordinates, or segment-local coordinates. These are encoding choices
only. A materialized row always has one absolute timestamp, and planning tokens
never become chart history.

Rows at time \(0\) are allowed. An implementation that requires a boundary
strictly before the first possible row may use the formal sentinel \(0^-\). The
sentinel is used only for boundary ordering; its numeric audio position is still
\(0\).

Audio availability and chart causality are distinct:

- \(X\) may contain complete-song context;
- \(H_k\) contains only committed chart events;
- a candidate or materialization model must not condition on uncommitted future
  chart decisions as though they were history.

## 4. Lanes, hands, physical roles, and exact actions

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

The lane serialization is a fixed chart-language invariant. The hand partition
and physical-role interpretation are canonical gameplay conventions, not a
claim that every human player uses the same fingering.

For row \(m_k\), the canonical action pair of hand \(h\) is

\[
a_k^h
=
\left(
a_{k,h,\mathrm{outer}},
a_{k,h,\mathrm{inner}}
\right),
\qquad
a_{k,h,\rho}
=
(m_k)_{\lambda(h,\rho)}.
\]

Therefore

\[
a_k^L
=
\bigl((m_k)_1,(m_k)_2\bigr),
\qquad
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

The row is reconstructed by

\[
(m_k)_{\lambda(h,\rho)}
=
a_{k,h,\rho}.
\]

The two action pairs must not be concatenated directly: \((a_k^L,a_k^R)\)
enumerates lanes as \((1,2,4,3)\), not serialized order \((1,2,3,4)\).

Define the exact active-role set of hand \(h\) at row \(k\) as

\[
A_k^h
=
\left\{
\rho\in\mathcal R
\;\middle|\;
a_{k,h,\rho}\neq\mathrm{EMPTY}
\right\}.
\]

For a one-hand chord, both roles belong to \(A_k^h\). This is simultaneous exact
action information, not a probability distribution over which role was used.

The canonical event trace of a chart is

\[
u_H
=
\operatorname{Exec}_\lambda(H)
=
\bigl((t_k,a_k)\bigr)_{k=1}^{|H|}.
\]

In the base fixed-lane 4K setting, \(u_H\) is uniquely determined by \(H\) and
\(\lambda\).

Define the left-right mirror of a row as

\[
\mu(m_1,m_2,m_3,m_4)
=
(m_4,m_3,m_2,m_1).
\]

This exchanges hands while preserving semantic roles `outer` and `inner`. For a
chart history,

\[
\mu H_k
=
\bigl((t_i,\mu m_i)\bigr)_{i=1}^k.
\]

The mirror operator extends to exact state, operational load state, demand
fields, continuations, and model specifications as defined in
[gameplay-state.md](gameplay-state.md).

## 5. Rows, long notes, and committed histories

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
\quad
1=\mathrm{TAP},
\quad
2=\mathrm{LN\_START},
\quad
3=\mathrm{LN\_CLOSE}.
\]

Their exact lane-occupancy transitions are:

| Code | Action | Valid transition |
| ---: | --- | --- |
| \(0\) | `EMPTY` | closed \(\to\) closed or open \(\to\) open |
| \(1\) | `TAP` | closed \(\to\) closed |
| \(2\) | `LN_START` | closed \(\to\) open |
| \(3\) | `LN_CLOSE` | open \(\to\) closed |

Every other action-occupancy pair is invalid. In particular, `EMPTY` is a
row-local no-op and never closes a long note.

A materialized row is

\[
y_k=(t_k,m_k),
\qquad
m_k\in\mathcal M,
\]

where

\[
\mathcal M
=
\mathcal S^4
\setminus
\{(0,0,0,0)\}.
\]

A chart contains at most one complete row at a time:

\[
t_1<t_2<\cdots<t_k.
\]

The committed history is

\[
H_k=(y_1,\ldots,y_k),
\qquad
H_0=().
\]

History stores materialized positive events only. The fact that an interval has
been fixed to contain no additional rows is represented by the fixed-through
time \(g\), not by inserting null rows into \(H_k\).

For lane \(\ell\), define

\[
s_{i,\ell}=(m_i)_\ell
\]

and its row-aligned projection

\[
H_k^{(\ell)}
=
\bigl((t_i,s_{i,\ell})\bigr)_{i=1}^k.
\]

If lane \(\ell\) has a prior nonempty action, its recency at \(t\ge t_k\) is

\[
\delta_\ell(t;H_k)
=
t-
\max
\left\{
t_i
\;\middle|\;
1\le i\le k,
s_{i,\ell}\neq 0
\right\}.
\]

For hand-role pair \((h,\rho)\), define analogously

\[
\delta_{h,\rho}(t;H_k)
=
t-
\max
\left\{
t_i
\;\middle|\;
1\le i\le k,
a_{i,h,\rho}\neq 0
\right\}.
\]

These quantities are undefined when no qualifying prior action exists. A
concrete exact-state schema may introduce explicit initial sentinels, but the
mathematical recency functions remain partial.

## 6. Exact boundaries and derived load caches

Let

\[
x_H(t)
=
\operatorname{Replay}(x_0,H;t)
\]

be the exact state obtained by replaying every committed row at or before
\(t\), followed by exact silent-time evolution to \(t\). Its internal
chart-format and canonical-control decomposition is defined in
[gameplay-state.md](gameplay-state.md).

At a committed fixed-through time \(g_{r,s}\),

\[
x_{r,s}
=
x_{H_{k_{r,s}}}(g_{r,s})
\]

and

\[
\beta^{\mathrm{exact}}_{r,s}
=
(g_{r,s},x_{r,s}).
\]

The exact boundary is independent of \(\kappa\), \(\varrho\), and learned
quality parameters after the exact-state schema and canonical role mapping have
been fixed. It is sufficient to continue exact replay and to decide chart
legality.

For declared load dynamics \(\kappa\), let

\[
q_H^\kappa(t)
=
\operatorname{LoadRoll}_\kappa
\left(
q_0^\kappa,
x_0,H;t
\right)
\]

be the operational load state. Then

\[
q_{r,s}^\kappa
=
q_{H_{k_{r,s}}}^\kappa(g_{r,s}),
\qquad
\sigma_{r,s}^\kappa
=
(g_{r,s},q_{r,s}^\kappa).
\]

The combined runtime boundary is

\[
b_{r,s}^\kappa
=
(g_{r,s},x_{r,s},q_{r,s}^\kappa).
\]

The distinction is semantic:

- \((H_{k_{r,s}},g_{r,s},x_{r,s})\) is committed chart and exact-control truth;
- \(q_{r,s}^\kappa\) is a model-derived cache tied to the identity and version of
  \(\kappa\);
- changing \(\kappa\) may require recomputing \(q^\kappa\), but cannot change
  committed rows, fixed no-row decisions, or chart legality;
- demand readout parameters do not belong in the continuation state unless they
  themselves have dynamics.

An implementation may atomically persist \(q^\kappa\) with the exact boundary
for efficiency. It must still record that the cache is derived and invalidate
or replay it when the load specification changes.

For continuation notation,

\[
\operatorname{Replay}
\left(
\beta^{\mathrm{exact}},Y;t
\right)
\]

means exact replay initialized from an exact boundary, and

\[
\operatorname{LoadRoll}_\kappa
\left(
b^\kappa,Y;t
\right)
\]

means operational rollout initialized from the matching runtime boundary.

The complete history \(H_k\) remains a separate conditioning object. Two
histories can have the same finite runtime boundary while differing in motif,
sectional, or mapper-intent information relevant to generation quality.

## 7. Rolling candidate lifecycle

Refreshes use an index

\[
r=0,1,2,\ldots
\]

independent of row index \(k\).

At refresh \(r\):

- \(k_r\) is the latest committed row in the candidate-generation snapshot;
- \(f_r\) is the time through which chart decisions are fixed at refresh;
- \(L_r>0\) is the candidate-horizon length;
- \(e_r=\min(T,f_r+L_r)\) is the horizon endpoint;
- \(W_r=(f_r,e_r]\) is the candidate horizon.

When \(f_r=0^-\), horizon arithmetic uses numeric value \(0\), while the open
left boundary still permits a row at time \(0\).

Every row in \(H_{k_r}\) lies at or before \(f_r\). Once fixed, no row at or
before \(f_r\) may be added, removed, or changed, and

\[
f_{r+1}\ge f_r.
\]

The candidate predictor receives

\[
\Gamma_r
=
G_\theta(X,H_{k_r};W_r).
\]

\(\Gamma_r\) may contain:

- a finite opportunity bank;
- a time lattice;
- point-process or hazard parameters;
- candidate probabilities or unnormalized scores;
- dense rhythmic evidence;
- a set of temporal path proposals;
- another high-recall support representation.

The formulation does not assume that \(\Gamma_r\) is calibrated or normalized.
If the representation exposes explicit proposed times, write

\[
C_r
=
\operatorname{support}_t(\Gamma_r)
\subseteq W_r.
\]

The meaning of support belongs to the declared candidate representation.

One refresh may support several stable-prefix revisions. Let

\[
s=0,1,2,\ldots
\]

index those revisions. Initially,

\[
k_{r,0}=k_r,
\qquad
g_{r,0}=f_r.
\]

At revision \(s\), the remaining horizon is

\[
W_{r,s}
=
(g_{r,s},e_r].
\]

A revision may advance \(g_{r,s}\) without adding a row, so \(s\) cannot be
replaced by \(k_{r,s}\).

Generation at revision \(s\) uses the current exact and runtime boundaries:

\[
\Xi^{\mathrm{exact}}_{r,s}
=
\left(
X,
H_{k_{r,s}},
\beta^{\mathrm{exact}}_{r,s},
\Gamma_r,
W_{r,s}
\right),
\]

\[
\Xi^\kappa_{r,s}
=
\left(
X,
H_{k_{r,s}},
b^\kappa_{r,s},
\Gamma_r,
W_{r,s}
\right).
\]

During the lifetime of \(\Gamma_r\), a partial commit changes

\[
H_{k_{r,s}},
\quad
\beta^{\mathrm{exact}}_{r,s},
\quad
\sigma^\kappa_{r,s},
\]

but not the provenance of \(\Gamma_r\). Recomputing candidate information from
the advanced history is a new refresh, not an implicit update inside the old
one.

This convention permits expensive parallel candidate computation while
retaining causal row materialization. It also creates measurable staleness:
later revisions may use candidate evidence produced from an older chart
snapshot. The refresh policy must control this risk.

A policy-specific trigger may satisfy

\[
T_{\mathrm{refresh}}
\left(
H_{k_{r,s}},
\beta^{\mathrm{exact}}_{r,s},
\Gamma_r
\right)
=1,
\]

after which

\[
(k_{r+1},f_{r+1})
=
(k_{r,s},g_{r,s})
\]

and a new \(\Gamma_{r+1}\) is computed.

The refresh trigger and structure of \(\Gamma_r\) are open. The distinction
between the frozen candidate snapshot and current committed boundary is fixed.

## 8. Future completions, chart legality, and model support

A completion over the remaining horizon \(W_{r,s}\) is

\[
Y
=
\bigl(
(\tau_1,m_1),
\ldots,
(\tau_n,m_n)
\bigr),
\qquad
n\ge0,
\]

with

\[
\tau_1<\cdots<\tau_n,
\qquad
m_i\in\mathcal M.
\]

The row sequence together with the declared horizon represents both its
positive rows and the absence of all other rows in that horizon.

Define the chart-legal completion set

\[
\mathcal V^{\mathrm{chart}}_{r,s}
=
\mathcal V^{\mathrm{chart}}
\left(
H_{k_{r,s}},
\beta^{\mathrm{exact}}_{r,s},
W_{r,s}
\right).
\]

A future row sequence belongs to \(\mathcal V^{\mathrm{chart}}_{r,s}\) if and
only if:

1. every \(\tau_i\in W_{r,s}\);
2. row times are strictly increasing;
3. every \(m_i\in\mathcal M\);
4. the committed prefix \(H_{k_{r,s}}\) is unchanged;
5. exact replay from \(\beta^{\mathrm{exact}}_{r,s}\) accepts every row
   transition;
6. the resulting exact boundary records every open long note and pending
   obligation;
7. if the horizon reaches song end \(T\), every long note is closed by \(T\).

Intermediate horizons may end with open long notes. Committing an `LN_START`
commits open occupancy and a future close obligation. It does not, by chart
legality alone, lock one provisional `LN_CLOSE`.

The chart-legal set does not depend on:

- candidate-bank membership;
- operational load magnitude;
- demand readout magnitude;
- preferred alternation or repetition;
- hand balance;
- musical quality;
- whether a model considers the pattern likely.

Additional format-specific hard rules may be declared explicitly. They must not
be hidden inside a soft score and then described as legality.

Define the model-reachable set

\[
\mathcal V^\Gamma_{r,s}
\subseteq
\mathcal V^{\mathrm{chart}}_{r,s}.
\]

It contains legal completions representable under the active candidate,
proposal, and decoding contract.

For a hard opportunity bank, every row time in
\(Y\in\mathcal V^\Gamma_{r,s}\) must be a member of the active bank suffix. For a
soft candidate field, \(\Gamma_r\) may influence score without excluding times,
allowing

\[
\mathcal V^\Gamma_{r,s}
=
\mathcal V^{\mathrm{chart}}_{r,s}.
\]

If the best legal choreography is absent from \(\mathcal V^\Gamma_{r,s}\), that
is a support or proposal failure, not a chart-legality failure.

For a support-independent oracle quality score \(\overline S_\psi\) defined
on the complete chart-legal set, a conceptual support gap is

\[
\Delta_{\mathrm{support}}
=
\max_{Y\in\mathcal V^{\mathrm{chart}}_{r,s}}\overline S_\psi(Y)
-
\max_{Y\in\mathcal V^\Gamma_{r,s}}\overline S_\psi(Y).
\]

The first maximum is generally intractable. Oracle and controlled
representation experiments can nevertheless estimate whether candidate support
excludes high-quality completions. Pointwise timing recall alone does not fully
measure this downstream gap.

The three layers are:

| Layer | Examples | Consequence of failure |
| --- | --- | --- |
| Chart legality | Valid `LN_START`/`LN_CLOSE`, one complete nonempty row per time, strict ordering, immutable prefix, song-end long-note closure | Reject as an invalid chart completion. |
| Model support | Candidate membership, proposal vocabulary, bounded lattice, beam, or decoder reachability | Completion is legal but unavailable to the current generator. |
| Quality | Musical alignment, motif continuity, role reuse, hand balance, demand progression, style, target match | Completion remains legal and reachable but receives lower preference. |

This separation prevents soft stylistic choices from becoming accidental hard
rules. Same-role repetition, dense chords, unusual but legal long-note overlap,
high local demand, and asymmetric style remain scoreable unless an explicit
chart-format rule forbids them.

## 9. Opportunity-bank drafts and temporal paths

When \(\Gamma_r\) contains a concrete opportunity bank, write

\[
U_r
=
\bigl((u_j,e_j)\bigr)_{j=1}^{M_r},
\]

with

\[
f_r<u_1<\cdots<u_{M_r}\le e_r.
\]

Here \(u_j\) is an absolute candidate time and \(e_j\) is candidate-associated
information.

At revision \(s\), the active candidate indices are

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
\qquad
b_j\in\mathcal B,
\]

where

\[
\mathcal B
=
\{\bot\}
\uplus
\mathcal M.
\]

The distinguished symbol \(\bot\notin\mathcal M\) is candidate-level null:

- \(b_j=\bot\) means no row exists at candidate time \(u_j\);
- \(b_j\in\mathcal M\) means one complete row exists at \(u_j\);
- an `EMPTY` coordinate inside \(b_j\in\mathcal M\) means only that the
  corresponding lane has no action in that row;
- `EMPTY` on an open long-note lane preserves open occupancy;
- \((0,0,0,0)\) is never a materialized row.

The associated temporal selection mask is

\[
P_{r,s}
=
(p_j)_{j\in J_{r,s}},
\qquad
p_j
=
\mathbf 1[b_j\neq\bot].
\]

\(P_{r,s}\) specifies selected times but not lane actions.

Candidate-row extraction is

\[
\operatorname{Rows}_{U_r}(B_{r,s})
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

The operator preserves order and removes candidate nulls. It does not establish
chart legality, control-load continuity, or quality.

A temporal path \(P_{r,s}\), a complete draft \(B_{r,s}\), and its extracted row
sequence \(Y(B_{r,s})\) are different formal objects:

\[
Y(B)
=
\operatorname{Rows}_{U_r}(B).
\]

The final committed object is always a row sequence. A path is an internal
planning object unless and until its row actions have been materialized and a
stable row prefix has been committed.

## 10. Induced rollout, whole-completion score, and structured distribution

Fix a canonical control-load specification

\[
\Theta_0=(\kappa_0,\varrho_0),
\]

where \(\kappa_0\) defines operational load dynamics and \(\varrho_0\) defines
demand readout and calibration. The meanings of these objects are specified in
[gameplay-state.md](gameplay-state.md).

Every legal completion \(Y\) induces a branch-local rollout

\[
\mathcal R_{\Theta_0}
\left(
Y\mid b^{\kappa_0}_{r,s}
\right)
=
\left(
\widetilde x_Y,
\widetilde q_Y,
\widetilde{\mathcal D}_Y
\right)
\]

over \(W_{r,s}\). The exact component must agree with exact replay; the load and
demand components are derived under \(\Theta_0\). Below,
\(\mathcal R_{\Theta_0}(Y)\) abbreviates this boundary-conditioned rollout.

Let

\[
S_\psi
\left(
Y;
\Xi^{\kappa_0}_{r,s},
\mathcal R_{\Theta_0}(Y),
\mathcal D^\star_{r,s}
\right)
\]

be an unnormalized whole-completion quality score. The optional
\(\mathcal D^\star_{r,s}\) is a desired demand plan; when no explicit plan is
used, that argument is omitted.

The MAP decision is

\[
Y^\star_{r,s}
\in
\arg\max_{Y\in\mathcal V^\Gamma_{r,s}}
S_\psi
\left(
Y;
\Xi^{\kappa_0}_{r,s},
\mathcal R_{\Theta_0}(Y),
\mathcal D^\star_{r,s}
\right).
\]

For a finite reachable set, the score defines a structured distribution

\[
P_\psi
\left(
Y
\mid
\Xi^{\kappa_0}_{r,s},
\mathcal D^\star_{r,s}
\right)
=
\frac{
\mathbf 1[Y\in\mathcal V^\Gamma_{r,s}]
\exp S_\psi(Y)
}{
Z_\psi
\left(
\Xi^{\kappa_0}_{r,s},
\mathcal D^\star_{r,s}
\right)
},
\]

where

\[
Z_\psi
\left(
\Xi^{\kappa_0}_{r,s},
\mathcal D^\star_{r,s}
\right)
=
\sum_{Y'\in\mathcal V^\Gamma_{r,s}}
\exp S_\psi(Y').
\]

For continuous-time support, an explicitly declared base measure or
point-process formulation is required. A bare normalized sum must not be used
as though the space were finite.

For an opportunity bank, define the legal reachable draft set

\[
\mathfrak B^\Gamma_{r,s}
=
\left\{
B_{r,s}
\;\middle|\;
Y(B_{r,s})
\in
\mathcal V^\Gamma_{r,s}
\right\}.
\]

A structured distribution over complete assignments is

\[
P_\psi
\left(
B
\mid
\Xi^{\kappa_0}_{r,s},
\mathcal D^\star_{r,s}
\right)
=
\frac{
\mathbf 1[B\in\mathfrak B^\Gamma_{r,s}]
\exp S_\psi(Y(B))
}{
Z^B_\psi
\left(
\Xi^{\kappa_0}_{r,s},
\mathcal D^\star_{r,s}
\right)
}.
\]

This representation distinguishes:

- local candidate evidence;
- uncertainty over temporal selections;
- uncertainty over row materializations;
- exact global legality;
- induced control-load trajectories;
- whole-completion quality.

Hard-invalid assignments receive zero probability rather than merely a large
negative preference. Depending on architecture, inference may compute or
approximate:

- MAP complete drafts;
- candidate-selection marginals;
- row-action marginals;
- suffix mass;
- samples;
- a stable-prefix posterior;
- a top-\(K\) branch set.

The formulation does not require exact partition-function computation. It does
require every approximation to be identified as an approximation rather than
quietly replacing the target structured distribution with independent local
probabilities.

The candidate predictor's score, a diffusion denoising residual, or another
proposal model's internal score is not automatically a calibrated
whole-completion quality score. The quantities coincide only if the objective
and normalization explicitly make them coincide.

For analysis, the joint score may be decomposed into diagnostic
responsibilities:

\[
S_\psi
=
\mathcal A_\psi
\left(
Q_{\mathrm{music}},
Q_{\mathrm{history}},
Q_{\mathrm{game}},
Q_{\mathrm{target}}
\right).
\]

The responsibilities are:

- \(Q_{\mathrm{music}}\) evaluates correspondence with complete audio and local
  or sectional musical structure;
- \(Q_{\mathrm{history}}\) evaluates continuity with committed chart history,
  including motifs and long-range structure not summarized by runtime state;
- \(Q_{\mathrm{game}}\) evaluates exact control continuity, operational load,
  hand interaction, realized demand progression, and style geometry;
- \(Q_{\mathrm{target}}\), when present, compares realized demand with an
  externally planned target.

A weighted additive instance is possible, but not fixed:

\[
S_\psi
=
\lambda_{\mathrm{music}}Q_{\mathrm{music}}
+
\lambda_{\mathrm{history}}Q_{\mathrm{history}}
+
\lambda_{\mathrm{game}}Q_{\mathrm{game}}
+
\lambda_{\mathrm{target}}Q_{\mathrm{target}}.
\]

This form does not assert statistical independence, equal units, or non-overlap
among terms.

The score must be calibrated across variable-length completions:

- \(Y=()\) must not win merely because every added row accumulates negative
  score;
- dense drafts must not win merely because every row contributes positive
  score;
- per-event, per-time, and whole-section terms need declared normalization;
- candidate evidence must not be double-counted accidentally in proposal and
  quality terms.

No particular aggregation or normalization is fixed by the formulation.

## 11. Completion-aware temporal-path factorization

An architecture may propose temporal paths before materializing row actions.
For an opportunity-bank draft \(B\), define

\[
\operatorname{sel}(B)=P.
\]

For a proposed path \(P\), its legal completion set is

\[
\mathcal C(P)
=
\left\{
B\in\mathfrak B^\Gamma_{r,s}
\;\middle|\;
\operatorname{sel}(B)=P
\right\}.
\]

If

\[
\mathcal C(P)=\varnothing,
\]

then the path has no legal row materialization under the current boundary and
support.

Two completion-aware path values are

\[
V_{\mathrm{MAP}}(P)
=
\max_{B\in\mathcal C(P)}
S_\psi(Y(B))
\]

and

\[
V_{\mathrm{marg}}(P)
=
\log
\sum_{B\in\mathcal C(P)}
\exp S_\psi(Y(B)).
\]

Set either value to \(-\infty\) when \(\mathcal C(P)\) is empty.

Under the structured draft distribution,

\[
P_\psi
\left(
P
\mid
\Xi^{\kappa_0}_{r,s},
\mathcal D^\star_{r,s}
\right)
=
\sum_{
B:\operatorname{sel}(B)=P
}
P_\psi
\left(
B
\mid
\Xi^{\kappa_0}_{r,s},
\mathcal D^\star_{r,s}
\right).
\]

This is a whole-path marginal, not a product of independently plausible
candidate positions.

A staged implementation may use

\[
\text{path proposer}
\longrightarrow
\text{completion-aware path evaluation}
\longrightarrow
\text{row materializer}
\longrightarrow
\text{exact and control-load rollout}
\longrightarrow
\text{final score}.
\]

The proposer may be autoregressive, diffusion-based, set-based, lattice-based,
or another model. Its proposal likelihood is not, by definition,
\(V_{\mathrm{MAP}}\) or \(V_{\mathrm{marg}}\).

Freezing a path before row materialization is formulation-consistent only when:

1. the path has at least one legal row completion;
2. its evaluation accounts for the quality or probability mass of those
   completions;
3. the approximation error introduced by early freezing is measured.

Otherwise the system has reverted to position-only optimization and may select
a musically plausible path with no coherent choreography. The final committed
object remains the complete row sequence \(Y\).

## 12. Desired demand plans and realized demand

An optional planner may provide a desired demand trajectory

\[
\mathcal D^\star_{r,s}(t),
\qquad
t\in W_{r,s}.
\]

A more general target may be a time-varying admissible set or distribution:

\[
\mathfrak T_{r,s}(t)
\subseteq
\mathbb R_{\ge0}^{C}.
\]

For candidate completion \(Y\), the gameplay rollout produces realized demand

\[
\widetilde{\mathcal D}_Y(t).
\]

The two objects have different semantics:

- \(\mathcal D^\star\) or \(\mathfrak T\) expresses a design target;
- \(\widetilde{\mathcal D}_Y\) is a consequence of the candidate chart under the
  fixed control-load specification.

A target term may take the form

\[
Q_{\mathrm{target}}(Y)
=
-
L_{\mathrm{traj}}
\left(
\widetilde{\mathcal D}_Y,
\mathcal D^\star_{r,s}
\right).
\]

The target is not automatically feasible. It may conflict with:

- exact long-note obligations;
- candidate support;
- the remaining horizon;
- declared load dynamics;
- intended musical alignment;
- other quality requirements.

Generation must therefore search for legal realizations rather than treating a
demand plan as a command that overrides chart semantics.

An alternative is an implicit policy

\[
P_\psi
\left(
Y_{\mathrm{future}}
\mid
X,
H_{k_{r,s}},
b^{\kappa_0}_{r,s},
\Gamma_r
\right)
\]

without a separately supervised demand planner. Whether explicit demand
planning offers enough control to justify its supervision and feasibility
failures remains an open research question.

## 13. Branch isolation and atomic stable-prefix commit

Every proposal branch starts from the same committed objects

\[
\left(
H_{k_{r,s}},
\beta^{\mathrm{exact}}_{r,s},
\sigma^{\kappa_0}_{r,s}
\right)
\]

and owns provisional copies

\[
\left(
\widetilde H,
\widetilde x,
\widetilde q
\right).
\]

After structured inference or reranking, the system selects a stable time prefix
of one winning legal branch. Suppose it fixes every row and no-row decision
through

\[
g'>g_{r,s}.
\]

Let

\[
I=(g_{r,s},g']
\]

and let \(Y_I\) be the ordered rows selected in that interval. If \(Y_I\)
contains \(n\ge0\) rows, then

\[
k_{r,s+1}
=
k_{r,s}+n,
\]

\[
H_{k_{r,s+1}}
=
H_{k_{r,s}}
\mathbin{\|}Y_I,
\]

where \(\mathbin{\|}\) denotes ordered history concatenation.

The exact boundary advances by replay:

\[
x_{r,s+1}
=
\operatorname{Replay}
\left(
\beta^{\mathrm{exact}}_{r,s},
Y_I;
g'
\right),
\]

\[
\beta^{\mathrm{exact}}_{r,s+1}
=
(g',x_{r,s+1}).
\]

The derived load cache advances under \(\kappa_0\):

\[
q^{\kappa_0}_{r,s+1}
=
\operatorname{LoadRoll}_{\kappa_0}
\left(
b^{\kappa_0}_{r,s},
Y_I;
g'
\right),
\]

\[
\sigma^{\kappa_0}_{r,s+1}
=
(g',q^{\kappa_0}_{r,s+1}).
\]

The promotion of

\[
\left(
H_{k_{r,s}},
g_{r,s},
x_{r,s}
\right)
\]

to

\[
\left(
H_{k_{r,s+1}},
g',
x_{r,s+1}
\right)
\]

is atomic. A runtime may promote the versioned load cache in the same
transaction, but the cache remains derived rather than chart-semantic truth.

If \(n=0\), materialized history is unchanged while \(g\), exact clocks, and
operational load still advance through silent flow.

The branch and commit invariants are:

- losing branches cannot mutate committed rows, fixed no-row decisions, exact
  occupancy, exact clocks, or operational load caches;
- committing `LN_START` commits open occupancy and a future close obligation;
- committing `LN_START` does not commit one particular future `LN_CLOSE` unless
  that close row is also inside the stable prefix;
- intermediate boundaries may carry open long notes;
- a provisional close cannot change persistent occupancy before its row is
  committed;
- generation continuing under the same refresh uses the advanced history and
  boundary but the original \(\Gamma_r\).

The lifecycle is

\[
\text{committed scene}
\rightarrow
\text{branch-local paths and rows}
\rightarrow
\text{exact legality and control-load rollout}
\rightarrow
\text{structured whole-completion score}
\rightarrow
\text{stable time prefix}
\rightarrow
\text{atomic exact promotion}.
\]

## 14. Uncertainty and decision semantics

Several objects carry different kinds of uncertainty:

| Object | Meaning |
| --- | --- |
| \(\Gamma_r\) | Candidate or musical-opportunity information; it need not be normalized. |
| \(P_\psi(Y\mid\cdot)\) | Model uncertainty over complete reachable legal choreographies. |
| \(P_\psi(B\mid\cdot)\) | Model uncertainty over complete opportunity-bank assignments. |
| Path marginal \(P_\psi(P\mid\cdot)\) | Probability mass of all legal row materializations sharing one temporal path. |
| Posterior or ensemble over \(\theta,\psi,\Theta_0\) | Parameter and model uncertainty, separate from future-chart uncertainty. |

The structured distribution is a declared model, not an empirical guarantee of
calibration. MAP, marginalization, sampling, risk-sensitive decoding, and
stable-prefix posterior rules are different decision procedures over the same
target space.

A local candidate probability is not a row-sequence marginal unless the model
and inference procedure establish that identity. Independent local
probabilities generally do not preserve whole-path legality, completion mass,
or control-load coherence.

## 15. What is fixed, conventional, and open

### Fixed by the target problem

- The committed object is a sequence of complete rows.
- Absolute row identity is independent of internal tokenization.
- Chart legality is decided by exact replay.
- Candidate support is distinct from chart legality.
- Candidate-level null is distinct from lane-level `EMPTY`.
- Every scored completion is paired with its induced exact and control-load
  rollout under the declared evaluator.
- Committed history remains available alongside finite runtime state.
- Branch-local state is isolated.
- Only a selected stable time prefix mutates committed state.
- History, fixed-through time, and exact boundary advance atomically.
- Planning operations never enter materialized history.

### Current structural conventions

- Complete audio is available before generation.
- Candidate information is computed in rolling windows.
- One candidate snapshot may support several commit revisions.
- The base 4K lane-to-role mapping is fixed and deterministic.
- Temporal paths may be proposed separately, but path evaluation must be
  completion-aware before an irreversible freeze.
- Generation and analysis share the same declared control-load rollout.

### Open implementation choices

- the structure and training objective of \(\Gamma_r\);
- opportunity bank, lattice, hazard, set-query, or continuous-time support;
- autoregressive, diffusion, dynamic-programming, beam, sampling, or refinement
  inference;
- exact or approximate partition-function computation;
- score aggregation and variable-length normalization;
- stable-prefix selection policy;
- candidate refresh trigger;
- explicit target-demand planning;
- whether and when a provisional long-note close is additionally locked;
- concrete model architectures and tokenizations.

## 16. Falsifiable hypotheses and open questions

### Falsifiable hypotheses

1. **Completion-aware path scoring improves choreography quality.**  
   Paths ranked using legal row completions should outperform paths ranked only
   by local timestamp plausibility. No controlled gain rejects the added
   completion-aware machinery for the tested setting.

2. **Structured uncertainty improves branch decisions.**  
   Retaining path and materialization mass should improve calibrated
   uncertainty, oracle top-\(K\) quality, or stable-prefix decisions relative to
   an equally sized single-path decoder. No gain supports a simpler decoder.

3. **Exact branch-local replay reduces lifecycle failures.**  
   Carrying exact occupancy, obligations, and fixed-through state should reduce
   illegal long-note transitions and cross-window discontinuities relative to
   history-only or unversioned hidden-state baselines.

4. **A demand trajectory can provide a useful control interface.**  
   Demand-conditioned generation should achieve better local style and
   intensity control than global tags or one scalar difficulty condition alone.
   Failure under controlled evaluation rejects the added interface for the
   tested setting.

5. **Rolling candidate refresh is computationally useful without unacceptable
   staleness.**  
   Reusing one candidate snapshot across several commits should reduce proposal
   cost while preserving downstream quality within a measurable refresh policy.
   If staleness dominates, the lifecycle convention or refresh policy must be
   revised.

### Open questions

- How should candidate coverage expose both local timing recall and downstream
  support loss?
- Which score normalization prevents event count or silence from dominating?
- Should path evaluation use MAP completion value, marginalized completion
  mass, or a risk-sensitive alternative?
- How far may a stable prefix advance without destroying useful revision?
- When does candidate staleness require a new refresh?
- Should a future `LN_CLOSE` ever be locked when its `LN_START` is committed?
- Does an explicit demand planner justify its added supervision and feasibility
  failures?
- Which approximations preserve enough structured mass for stable-prefix
  decisions?

## 17. Scope boundary

This page fixes:

- absolute-time row identity;
- lane serialization, action syntax, and deterministic base role mapping;
- committed-history and fixed-through semantics;
- exact, derived-load, and runtime boundary identities;
- rolling candidate snapshot semantics;
- future completion, chart-legality, and model-support spaces;
- candidate-indexed drafts and temporal paths;
- whole-completion scoring and structured-distribution targets;
- branch isolation and stable-prefix commit ownership.

This page does not fix:

- the internal decomposition of exact control state;
- operational load dynamics;
- named demand channels or their calibration;
- player response or error models;
- candidate, proposal, materialization, or scoring architectures;
- concrete training objectives or inference approximations.

The first four omitted topics belong to
[gameplay-state.md](gameplay-state.md). Concrete architectures, experiments,
and evidence belong in `docs/research/` rather than in the formulation layer.