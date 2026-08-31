# Generation notation

This page owns the canonical vocabulary shared by Pulsefield's target
generation formulations. It defines chart syntax, coordinate conventions,
rolling candidate snapshots, branch boundaries, and commit semantics. It does
not define gameplay-state dynamics or chart quality.

All row times are absolute audio times in seconds.

The page distinguishes three kinds of statements:

- **Fixed formal invariant**: every conforming implementation must preserve it.
- **Canonical system convention**: Pulsefield currently adopts it, but replacing
  it would require changing the formulation rather than merely changing model
  parameters.
- **Open implementation choice**: alternative architectures may instantiate it
  differently without changing the formal problem.

## 1. Formal scene and invariants

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

At candidate refresh \(r\), candidate information is computed from a snapshot

\[
\Gamma_r
=
G_\theta(X,H_{k_r};W_r).
\]

One \(\Gamma_r\) may remain active across several stable-prefix commit
revisions. At revision \(s\), the current generation scene is

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
\left(g_{r,s},z_{r,s}\right)
\]

is the timestamped committed boundary. The scalar \(g_{r,s}\) is the time
through which both row and no-row decisions are fixed, while \(z_{r,s}\) is the
canonical gameplay state at that time.

The distinction between

\[
H_{k_r}
\quad\text{and}\quad
H_{k_{r,s}}
\]

is semantic:

- \(H_{k_r}\) is the history snapshot that produced \(\Gamma_r\);
- \(H_{k_{r,s}}\) is the current committed history;
- \(\Gamma_r\) does not silently recondition after a partial commit;
- generation under the active \(\Gamma_r\) nevertheless starts from the current
  history and current boundary state.

The fixed lifecycle invariants are:

1. A committed chart time always carries one complete nonempty row.
2. Materialized row times are strictly increasing.
3. Candidate nulls, planning tokens, edits, and losing branch decisions never
   enter \(H_k\).
4. A fixed-through time may advance without adding a row; therefore row index
   and stable-prefix index are different objects.
5. Committed history, fixed-through time, and canonical boundary state advance
   atomically.
6. A provisional branch never mutates committed history or committed state.
7. Candidate-level absence and lane-level `EMPTY` are different symbols.
8. A complete chart closes every long note, while an intermediate boundary may
   carry open long notes forward.

The canonical system conventions are:

- all of \(X\) is available before chart generation begins;
- lanes are represented in serialized order \((1,2,3,4)\);
- hands use the canonical within-hand order `(outer, inner)`;
- candidate information is refreshed in rolling windows and remains frozen
  during the lifetime of one refresh.

The structure of \(\Gamma_r\), the refresh policy, the proposal model, the
search procedure, and the internal tokenization remain open.

## 2. Worked example

Consider the committed chart prefix

| \(i\) | \(t_i\) | \(m_i\) | Meaning |
| ---: | ---: | :---: | --- |
| 1 | 29.40 | `1000` | Tap lane 1. |
| 2 | 29.70 | `0020` | Start a long note in lane 3. |
| 3 | 29.90 | `0100` | Tap lane 2; lane 3 remains open. |
| 4 | 30.00 | `0030` | Close the long note in lane 3. |

Thus

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

Because the right hand uses canonical order `(outer, inner)`, the right-hand
action for row 2 is

\[
a_2^R
=
\bigl((m_2)_4,(m_2)_3\bigr)
=
(0,2).
\]

The serialized row remains `0020`; the reversed pair exists only in the
canonical right-hand coordinate system.

Suppose refresh \(r\) begins from

\[
k_r=4,
\qquad
f_r=30.00,
\qquad
L_r=5.00,
\qquad
W_r=(30.00,35.00].
\]

The predictor computes

\[
\Gamma_r
=
G_\theta(X,H_4;W_r).
\]

Assume one realization of \(\Gamma_r\) is an opportunity bank whose first
candidate times are

\[
30.12,\quad 30.25,\quad 30.49.
\]

A candidate-indexed draft may begin

\[
B_{r,0}
=
(0001,1000,\bot,\ldots),
\]

where \(\bot\) is candidate-level null. It proposes

\[
y_5=(30.12,0001),
\qquad
y_6=(30.25,1000),
\]

and no row at \(30.49\).

If only the time prefix through \(30.25\) is committed, then

\[
H_4\longrightarrow H_6,
\qquad
k_{r,1}=6,
\qquad
g_{r,1}=30.25.
\]

The candidate-null decision at \(30.49\) remains provisional. Generation
continues from current \(H_6\) and \(\beta_{r,1}\), but the active candidate
information is still

\[
\Gamma_r
=
G_\theta(X,H_4;W_r).
\]

If the stable prefix were instead committed through \(30.49\), then the
no-row decision at \(30.49\) would become fixed and

\[
g_{r,1}=30.49
\]

while the row index would remain

\[
k_{r,1}=6.
\]

This is why stable-prefix index \(s\), row index \(k\), and refresh index \(r\)
cannot be merged.

## 3. Audio and absolute time

Let the song duration be \(T\). The audio feature field is defined over

\[
t\in[0,T].
\]

\(X(t)\) denotes features associated with the neighborhood of absolute audio
time \(t\). Future audio is known; future chart rows are not.

An implementation may internally use time-shift tokens, relative positions, or
segment-local coordinates. These are encoding choices only. A materialized row
always has one absolute timestamp, and planning tokens never become chart
history.

Rows at time \(0\) are allowed. An implementation that requires a boundary
strictly before the first possible row may use the formal sentinel \(0^-\).
The sentinel is used only for boundary ordering; its numeric audio position is
still \(0\).

## 4. Lanes, hands, and physical roles

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
and physical-role interpretation are canonical gameplay conventions; they are
not claims that every human player uses the same fingering.

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

The row is reconstructed by

\[
(m_k)_{\lambda(h,\rho)}
=
a_{k,h,\rho}.
\]

The pairs must not be concatenated directly: \((a_k^L,a_k^R)\) enumerates
lanes as \((1,2,4,3)\), not serialized order \((1,2,3,4)\).

Define the left-right mirror of a row as

\[
\mu(m_1,m_2,m_3,m_4)
=
(m_4,m_3,m_2,m_1).
\]

This exchanges hands while preserving the semantic roles `outer` and `inner`.
For a chart history,

\[
\mu H_k
=
\bigl((t_i,\mu m_i)\bigr)_{i=1}^k.
\]

The mirror operator is used by the gameplay-state formulation to state
hand-symmetry priors without confusing them with serialized lane order.

## 5. Rows and histories

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
boundary \(g\), not by inserting null rows into \(H_k\).

For lane \(\ell\), define

\[
s_{i,\ell}=(m_i)_\ell.
\]

Its row-aligned lane projection is

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
\mid
1\le i\le k,\;
s_{i,\ell}\neq 0
\right\}.
\]

The quantity is undefined if no such action exists. A canonical gameplay
profile may introduce an explicit initial sentinel, but the mathematical
recency function itself remains partial.

## 6. Rolling candidate lifecycle

Refreshes use an index

\[
r=0,1,2,\ldots
\]

independent of row index \(k\).

At refresh \(r\):

- \(k_r\) is the latest row in the candidate-generation history snapshot;
- \(f_r\) is the time through which chart decisions are fixed;
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

\(\Gamma_r\) is candidate information. It may be an opportunity bank, lattice,
set of scored proposals, point-process parameterization, dense field, or
another representation. It is not assumed to be a normalized probability
distribution.

If the representation exposes explicit proposed times, write

\[
C_r
=
\operatorname{support}_t(\Gamma_r)
\subseteq W_r.
\]

The meaning of support belongs to the chosen candidate representation.

One refresh may support several stable-prefix commits. Let

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

The canonical boundary is

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

It contains every committed row at or before \(g_{r,s}\) and the silent
between-event evolution from the last row to the boundary.

Generation at revision \(s\) receives

\[
\Xi_{r,s}
=
\left(
X,
H_{k_{r,s}},
\beta_{r,s},
\Gamma_r,
W_{r,s}
\right).
\]

A policy-specific refresh trigger may satisfy

\[
T_{\mathrm{refresh}}
\left(
H_{k_{r,s}},
\beta_{r,s},
\Gamma_r
\right)
=
1,
\]

after which

\[
(k_{r+1},f_{r+1})
=
(k_{r,s},g_{r,s})
\]

and a new \(\Gamma_{r+1}\) is computed.

The trigger and the structure of \(\Gamma_r\) are open. The distinction between
the frozen candidate snapshot and the current committed boundary is not open.

## 7. Candidate-indexed drafts and temporal paths

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

A candidate-indexed draft is

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
- \(b_j\in\mathcal M\) means a complete row exists at \(u_j\);
- an `EMPTY` coordinate inside \(b_j\in\mathcal M\) means only that the
  corresponding lane has no action in that row;
- `EMPTY` on an open long-note lane preserves the open occupancy;
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
chart legality, gameplay continuity, or quality.

A temporal path \(P_{r,s}\) and a complete draft \(B_{r,s}\) are therefore
different formal objects. The choreography-generation formulation defines when
a path may be evaluated or frozen before its row actions are materialized.

## 8. Branch, boundary, and commit ownership

Every proposal branch starts from the same committed pair

\[
\left(
H_{k_{r,s}},
\beta_{r,s}
\right)
\]

and owns provisional copies

\[
(\widetilde H,\widetilde z).
\]

Suppose a winning branch fixes every row and no-row decision through

\[
g'
>
g_{r,s}.
\]

Let

\[
I=(g_{r,s},g']
\]

and let \(Y_I\) be the ordered rows selected in that interval. If \(Y_I\)
contains \(n\ge 0\) rows, then

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

The new boundary is

\[
\beta_{r,s+1}
=
(g',z_{r,s+1}),
\]

with

\[
z_{r,s+1}
=
\operatorname{Roll}_{\vartheta_0}
\left(
\beta_{r,s},
Y_I;
g'
\right).
\]

The promotion

\[
\left(
H_{k_{r,s}},
\beta_{r,s}
\right)
\longrightarrow
\left(
H_{k_{r,s+1}},
\beta_{r,s+1}
\right)
\]

is atomic.

If \(n=0\), the materialized history is unchanged, but \(g\) and the boundary
state still advance through silent flow.

The branch and commit invariants are:

- losing branches cannot mutate committed occupancy, execution belief, demand,
  history, or fixed no-row decisions;
- committing `LN_START` commits open occupancy and a future closure obligation;
- committing `LN_START` does not commit one particular future `LN_CLOSE` unless
  that close row is also inside the stable prefix;
- intermediate boundaries may carry open long notes;
- a provisional close cannot change persistent occupancy before the close row
  is committed;
- generation that continues under the same refresh uses the advanced history
  and boundary but the original \(\Gamma_r\).

## 9. Scope boundary

This page fixes:

- absolute-time row identity;
- lane serialization and action syntax;
- canonical hand-role coordinates;
- materialized-history semantics;
- the distinction between candidate null and `EMPTY`;
- refresh, revision, row, and fixed-through indices;
- branch isolation and atomic commit.

This page does not define:

- the internal decomposition of \(z(t)\);
- parity or fingering inference;
- continuous demand dynamics;
- chart-legal completion sets;
- candidate-support restrictions;
- structured quality scores;
- proposal, refinement, or search algorithms.

Those belong respectively to the canonical gameplay-state formulation and the
choreography-generation formulation.