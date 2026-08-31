# Generation notation

This page has two parts: the canonical formal language and one worked example.
It defines the generation objects and invariants without choosing a candidate
representation, refresh policy, model architecture, or materialization
factorization. All times are absolute audio times in seconds.

## 1. Strict model and formal language

### Audio

Let $A$ be the complete source audio and let

$$
X=\Phi(A)
$$

be a feature field computed over the complete song. $X(t)$ denotes the
features associated with the neighborhood of time $t$. Both $A$ and all of
$X$ are available before generation begins: future audio is known, while
future map content is not.

### Rows and histories

The lane set and hand partition are

$$
\mathcal L=\{1,2,3,4\},
\qquad
\mathcal L_L=\{1,2\},
\qquad
\mathcal L_R=\{3,4\}.
$$

Each lane uses the row-local action alphabet

$$
\mathcal S=\{0,1,2,3\}
=
\{\mathrm{EMPTY},\mathrm{TAP},\mathrm{LN\_START},\mathrm{LN\_CLOSE}\}.
$$

The action codes and lane transitions are:

| Code | Action | Valid transition |
| --- | --- | --- |
| $0$ | `EMPTY` | closed $\to$ closed or open $\to$ open |
| $1$ | `TAP` | closed $\to$ closed |
| $2$ | `LN_START` | closed $\to$ open |
| $3$ | `LN_CLOSE` | open $\to$ closed |

All other action–lane-state pairs are invalid. In particular, `EMPTY` is a
row-local no-op; it does not mean that a long note is closed.

The $k$-th materialized row is

$$
y_k=(t_k,m_k),
\qquad
m_k\in\mathcal S^4\setminus\{(0,0,0,0)\},
$$

where the coordinates of $m_k$ follow lane order $(1,2,3,4)$. A chart has
at most one row at each time, so

$$
t_1<t_2<\cdots<t_k.
$$

The materialized map history is

$$
H_k=(y_1,\ldots,y_k),
\qquad
H_0=().
$$

For lane $\ell$, define $s_{i,\ell}=(m_i)_\ell$. Its lane history is the
time-state projection of every materialized row:

$$
H_k^{(\ell)}
=
\bigl((t_i,s_{i,\ell})\bigr)_{i=1}^k.
$$

If lane $\ell$ has a nonempty action in $H_k$, its recency at time $t$
is

$$
\delta_\ell(t;H_k)
=
t-
\max\{t_i\mid 1\le i\le k,\ s_{i,\ell}\ne0\}.
$$

The quantity is undefined if that lane has no prior nonempty action.

### Rolling candidate process

Candidate refreshes use an index $r=0,1,2,\ldots$ independent of row index
$k$. At refresh $r$:

- $k_r$ is the latest materialized row in the history snapshot;
- $f_r$ is the frontier through which the chart is fixed;
- $L_r>0$ is the candidate-horizon length; and
- $W_r=(f_r,f_r+L_r]$ is the candidate horizon.

All rows in $H_{k_r}$ lie at or before $f_r$. Once fixed, no row at or
before $f_r$ may be added, removed, or changed, and

$$
f_{r+1}\ge f_r.
$$

The candidate predictor receives the complete audio field, the refresh-time
history snapshot, and the horizon:

$$
\Gamma_r
=
G_\theta(X,H_{k_r};W_r).
$$

$\Gamma_r$ is candidate information, not necessarily a probability
distribution. Its proposed times are

$$
C_r=\operatorname{support}(\Gamma_r)
\subseteq W_r,
$$

where $\operatorname{support}$ means the time positions put forward by the
chosen candidate representation.

One refresh may support several rows. While those rows are materialized,

$$
H_{k_r}\to H_{k_r+1}\to H_{k_r+2}\to\cdots,
$$

but $\Gamma_r$ remains conditioned on the snapshot $H_{k_r}$. At row step
$k$, the materializer therefore receives

$$
(X,H_k,\Gamma_r),
$$

where $H_k$ is current even when $H_{k_r}$ is older. A policy-specific
condition triggers the next refresh:

$$
T(H_k,\Gamma_r)=1
\quad\Longrightarrow\quad
r\to r+1.
$$

The formal language does not specify the structure of $\Gamma_r$, the
definition of $T$, or whether materialization chooses time first, chooses
$(t,m)$ jointly, or uses competing lane-wise processes.

## 2. Worked example and explanation

Assume $X=\Phi(A)$ has already been computed for the complete song. The
current chart prefix is

| $i$ | $t_i$ | $m_i$ | Meaning |
| --- | ---: | --- | --- |
| 1 | 29.40 | `1000` | Tap lane 1. |
| 2 | 29.70 | `0020` | Start a long note in lane 3. |
| 3 | 29.90 | `0100` | Tap lane 2; lane 3 remains open because its action is `EMPTY`. |
| 4 | 30.00 | `0030` | Close the long note in lane 3. |

Thus

$$
H_4=
\bigl(
(29.40,1000),
(29.70,0020),
(29.90,0100),
(30.00,0030)
\bigr).
$$

Two lane projections are

$$
H_4^{(1)}=
\bigl((29.40,1),(29.70,0),(29.90,0),(30.00,0)\bigr)
$$

and

$$
H_4^{(3)}=
\bigl((29.40,0),(29.70,2),(29.90,0),(30.00,3)\bigr).
$$

At $t=30.25$, the lane recencies are

$$
\delta_1(30.25;H_4)=0.85,
\qquad
\delta_2(30.25;H_4)=0.35,
\qquad
\delta_3(30.25;H_4)=0.25.
$$

$\delta_4(30.25;H_4)$ is undefined because lane 4 has no prior nonempty
action.

Now let refresh $r$ begin with

$$
k_r=4,
\qquad
f_r=30.00,
\qquad
L_r=5.00,
\qquad
W_r=(30.00,35.00].
$$

The candidate predictor computes

$$
\Gamma_r=G_\theta(X,H_4;W_r),
\qquad
C_r=\{30.12,30.25,30.49,\ldots\}.
$$

Suppose the materializer uses this candidate information to add

$$
y_5=(30.12,0001),
\qquad
y_6=(30.25,1000).
$$

The current history is now $H_6$, but the active $\Gamma_r$ is still the
result computed from $H_4$. This is the distinction between current
materialization history $H_k$ and candidate-generation history $H_{k_r}$.

If $T(H_6,\Gamma_r)=1$, the next refresh may set

$$
k_{r+1}=6,
\qquad
f_{r+1}=30.25,
\qquad
W_{r+1}=(30.25,35.25],
$$

and recompute

$$
\Gamma_{r+1}=G_\theta(X,H_6;W_{r+1}).
$$

The complete audio field $X$ was available throughout. Only the map history
and the rolling candidate snapshot changed.
