# Chart and generation contract

Ensomi generates a 4-key chart from complete source audio. This page defines
its row language, legal continuations, and committed-prefix semantics.
[Gameplay state](gameplay-state.md) defines the target gameplay response,
frontier, style observations, and their use in generation.

## Chart object and time

Let $A$ be the complete source audio, $T$ its duration in seconds, and
$X=\Phi(A)$ an audio representation available before generation begins.
The representation and model architecture are open choices.

A materialized chart is a finite ordered sequence

$$
H=((t_1,m_1),\ldots,(t_N,m_N)),
\qquad N\geq 0,
$$

where $t_i\in[0,T]$ is an absolute audio timestamp and $m_i$ is one complete
nonempty four-lane row. Row times are strictly increasing:

$$
t_1<t_2<\cdots<t_N.
$$

All simultaneous lane actions belong to the same row. Internal beat coordinates,
time-shift tokens, lattices, and relative offsets must resolve to this chart
object. The formulation imposes no additional minimum spacing between rows.

## Row language and long notes

Serialized lane order is always $(1,2,3,4)$. Each lane has one action per row:

| Code | Action | Legal occupancy transition |
| ---: | --- | --- |
| $0$ | `EMPTY` | closed $\to$ closed or open $\to$ open |
| $1$ | `TAP` | closed $\to$ closed |
| $2$ | `LN_START` | closed $\to$ open |
| $3$ | `LN_CLOSE` | open $\to$ closed |

Every other action-occupancy combination is illegal. In particular, `EMPTY`
preserves an open long note; it is not a release.

The row alphabet is

$$
\mathcal M=\{0,1,2,3\}^4\setminus\{(0,0,0,0)\}.
$$

The four actions in a row are simultaneous. A row can contain only one action
on each lane, so the language does not represent a same-lane close and restart
at the same timestamp. Strictly ordered rows also exclude zero-duration long
notes.

A complete chart starts with every lane closed and closes every long note by
$T$. Closure requires a materialized `LN_CLOSE`; song end does not implicitly
release open lanes. An intermediate chart prefix may retain open long notes.

A planning representation may encode the absence of an entire row. That
absence must remain distinct from `EMPTY` on one lane, and it must not insert a
$(0,0,0,0)$ row into the materialized chart. Its internal encoding is an
implementation choice.

## Canonical hand-role coordinates

The canonical mapping assigns two roles to each hand:

| Lane | Hand | Role |
| ---: | --- | --- |
| $1$ | left | outer |
| $2$ | left | inner |
| $3$ | right | inner |
| $4$ | right | outer |

For $m=(m_1,m_2,m_3,m_4)$, its hand-role view is

$$
\operatorname{Act}_{\lambda}(m)
=
\bigl((m_1,m_2),(m_4,m_3)\bigr).
$$

Each hand uses $(\mathrm{outer},\mathrm{inner})$ order. Concatenating these
pairs directly would produce $(1,2,4,3)$ rather than serialized lane order.
This is a deterministic coordinate projection, not inferred fingering or a
separate physical execution. A two-role chord activates both roles; it does
not create uncertainty over which role was used.

The left-right mirror is

$$
\mu(m_1,m_2,m_3,m_4)=(m_4,m_3,m_2,m_1).
$$

Apply it to each chart row without changing timestamps. It exchanges hands
while preserving outer and inner roles. Gameplay response and style symmetry
are specified in [gameplay-state.md](gameplay-state.md).

## Committed history and exact replay

During generation, $H$ denotes the committed chart prefix. Its semantic state
is the pair

$$
(H,g),
$$

where $g$ is the fixed-through time. Every row in $H$ has time at or before
$g$, and no further row may be inserted at or before $g$. Thus $g$ fixes both
row and no-row decisions. Advancing it need not add a row to $H$.

Initially $H=()$ and all lanes are closed. Use the formal boundary $g=0^-$
when the decision at time $0$ is still open. The sentinel precedes $0$ in the
ordering; it is not a negative audio timestamp. The interval $(0^-,e]$ denotes
$[0,e]$, so rows at time $0$ remain representable.

A committed prefix must replay legally from the initial closed occupancy. At
$g=T$, it must also satisfy the complete-chart closure requirement.

Exact state is derived by replay:

$$
x_H(t)=\operatorname{Replay}(H;t).
$$

Replay processes all rows with timestamp at or before $t$. Its required
format information is lane-wise long-note occupancy:

$$
\omega_H(t)\in\{0,1\}^4.
$$

An implementation may also derive exact chart summaries such as action clocks.
These remain functions of chart history and query time. They do not require a
separate committed control-memory object. Occupancy changes only at rows;
time-dependent summaries must also advance through intervals without rows.

At a row time $t$, $x_H(t^-)$ denotes the state before that row and $x_H(t)$
the state after the complete row. For a committed boundary, $x_H(g)$ therefore
includes any row at $g$.

The pair $(H,g)$ is the source of truth. Retained replay caches must agree
with it, but they are not additional independent semantic state. Exact replay
summaries need not contain every arrangement distinction in the full history;
the history remains available to generation.

## Legal continuations

For a committed boundary $(H,g)$, choose a generation window

$$
W=(g,e],
\qquad g<e\leq T.
$$

A continuation is a finite sequence

$$
Y_W=((\tau_1,n_1),\ldots,(\tau_M,n_M)),
\qquad M\geq 0.
$$

Write $\mathcal V_{\mathrm{legal}}(H,g,e)$ for the set of continuations that
satisfy all of the following:

1. Every row time lies in $W\cap[0,T]$, and row times are strictly increasing.
2. Every $n_i\in\mathcal M$.
3. Replay from the boundary occupancy accepts every row under the transition
   table.
4. If $e=T$, every lane is closed after the last row and remains closed at $T$.

Appending a continuation leaves $H$ and its fixed no-row decisions unchanged.
The empty continuation $Y_W=()$ is legal at an intermediate horizon, including
when long notes remain open. At song end it is legal only if the boundary has
no open long notes. Once $g=T$, generation is complete and there is no further
window to generate.

Local row legality and complete-chart legality differ at song end. An
`LN_START` on a closed lane at $T$ satisfies the local occupancy transition,
but cannot belong to a legal completed chart because there is no later time
for its close. A required `LN_CLOSE` at $T$ can be legal.

These rules define chart legality. Audio correspondence, gameplay demand,
style, model probability, and decoding support affect generation choices;
they do not change the legal continuation set.

## Generation and optional controls

The generated object is the complete row continuation. A probabilistic
formulation is

$$
Y_W\sim p_\theta\left(
\cdot\mid X,H,g,W,
[c_W^{\mathrm{style}}],
[c_W^{\mathrm{demand}}]
\right),
$$

with probability one on $\mathcal V_{\mathrm{legal}}(H,g,e)$. Brackets mark
optional arguments. With neither request present, the generator still produces
a chart with realized gameplay organization and demand.

The two requests have different meanings:

- $c_W^{\mathrm{style}}$ expresses a requested gameplay-style tendency.
- $c_W^{\mathrm{demand}}$ expresses a requested gameplay-demand response under
  a declared response specification.

Neither request fixes a row or overrides committed decisions and long-note
obligations. Requests may conflict with each other or be unattainable under the
current boundary and implementation support. Generation may seek a legal
compromise or report that the request cannot be met. Request interpretation and
matching belong to [Controls](gameplay-state.md#controls); this notation does
not assume numerical demand coordinates or a style score scale.

The [target response and frontier](gameplay-state.md#target-response-and-frontier)
come before a proposed demand representation $d$ in the semantic definition.
Once a response specification exists, such a representation may provide derived
features for generation or evaluation. This page requires neither a particular
representation nor a runtime demand-state input. Concrete response definitions
and their relation to [style observations](gameplay-state.md#style-observations)
belong to the gameplay formulation.

Complete audio is available during generation. Models may jointly reason about
provisional future rows, including later rows in the same continuation. Those
rows cannot be treated as already committed history. Planning tokens, edits,
response representations, and search branches become chart history only through
their selected materialized rows.

The probability model must account for variable-length timed sequences. A
continuous-time formulation needs a declared probability measure or equivalent
construction; normalization by a finite sum applies only to finite support.
The formulation does not prescribe an energy score, partition function,
factorization, or decoding algorithm.

## Provisional branches and prefix commit

A proposal branch extends the same committed pair $(H,g)$ and may compute its
own provisional replay and gameplay responses. Neither the branch nor its
caches may mutate committed rows, the fixed-through time, or committed caches.
This is a semantic isolation requirement, not a requirement to copy the full
history into each branch.

Select a legal continuation $Y_W$ and a boundary

$$
g<g'\leq e.
$$

Committing through $g'$ fixes every row and no-row decision in $(g,g']$.
Writing $Y_W|_{(g,g']}$ for the selected rows in that interval, the update is

$$
(H,g)
\longrightarrow
\left(H\mathbin{\|}Y_W|_{(g,g']},g'\right),
$$

where $\mathbin{\|}$ denotes ordered concatenation. Rows exactly at $g'$ are
included; rows at the old boundary $g$ cannot be added or changed. A branch
defined only through $e$ cannot commit decisions beyond $e$.

The committed history and boundary advance together. Any published derived
cache must describe the new pair, or be invalidated and recomputed. Losing
branches and the uncommitted suffix remain provisional. The choice of prefix
and the confidence needed to select it are implementation questions.

A commit with no rows still advances $g$. It preserves any open long-note
occupancy and advances time-dependent derived quantities. An interval without
rows can contain sustained holds; it does not necessarily represent rest or
imply decreasing demand.

Committing `LN_START` commits an open lane and the obligation to close it by
song end. A proposed `LN_CLOSE` beyond $g'$ is still provisional and may move.
The generator may not erase the committed start, release it implicitly, or
reinterpret an absent closing opportunity as chart illegality.

## Legal space and implementation support

A concrete timing representation, candidate proposal, tokenization, or decoder
may reach only a subset of $\mathcal V_{\mathrm{legal}}(H,g,e)$. A legal
continuation omitted by that subset is an implementation support limitation.
In particular, an implementation can leave itself without a reachable closing
row even though the chart language permits one.

Such limitations must be evaluated against downstream chart quality and
controllability. Pointwise timing recall alone does not characterize the quality
or diversity of reachable charts. The formulation does not require a candidate
stage or prescribe its inputs, refresh policy, or internal timing paths.

## Worked example

Let $T>30.20$ and suppose the committed history is

$$
H=\bigl((29.70,(0,0,2,0)),(29.90,(0,1,0,0))\bigr),
\qquad g=29.90.
$$

Lane 3 is open: the first row starts a long note, and the second row leaves
that lane unchanged. For $W=(29.90,30.20]$:

- A row $(30.00,(0,0,3,0))$ legally closes lane 3.
- A row $(30.00,(0,0,2,0))$ illegally starts another long note on lane 3.
- A row $(30.00,(0,0,0,1))$ legally taps lane 4 while lane 3 remains open.

The third continuation is legal because the horizon is intermediate. If its
endpoint were song end, a later row closing lane 3 would be required within
that same window. Neither a style request nor a demand request can make the
second continuation legal.

Choosing $Y_W=()$ and committing through $30.05$ leaves $H$ unchanged, fixes
no-row decisions through $30.05$, and keeps lane 3 open. Its eventual close
must occur after $30.05$ and no later than $T$.
