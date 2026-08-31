# Choreography generation problem

Pulsefield's target generation problem is to produce a legal, musically
grounded 4K choreography whose complete row sequence induces a coherent
canonical gameplay-state trajectory.

The formulation separates three different filters:

\[
\text{chart legality}
\supseteq
\text{model-reachable support}
\xrightarrow{\text{quality score}}
\text{preferred choreography}.
\]

A choreography can be legal but unreachable because the current candidate bank
omitted its times. It can be reachable but receive a low quality score. These
are different failure modes and must not share one label.

This page uses [generation notation](notation.md) and the
[canonical gameplay-state formulation](gameplay-state.md).

## 1. Formal scene, decision object, and constraints

At refresh \(r\), candidate information is computed from

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

At stable-prefix revision \(s\), the current generation scene is

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

The candidate snapshot \(H_{k_r}\) may be older than the current committed
history \(H_{k_{r,s}}\). Generation uses the current history and boundary while
\(\Gamma_r\) remains the frozen result of its original refresh.

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

The row times satisfy

\[
\tau_1<\cdots<\tau_n,
\]

and every action satisfies

\[
m_i\in\mathcal M.
\]

Define the chart-legal completion set

\[
\mathcal V^{\mathrm{chart}}_{r,s}
=
\mathcal V^{\mathrm{chart}}
\left(
H_{k_{r,s}},
\beta_{r,s},
W_{r,s}
\right).
\]

This set depends only on committed chart semantics, exact boundary state, and
the remaining time window. It does not depend on whether the candidate model
happened to propose the relevant times.

Define the model-reachable set

\[
\mathcal V^\Gamma_{r,s}
\subseteq
\mathcal V^{\mathrm{chart}}_{r,s}.
\]

The subset relation records support restrictions imposed by \(\Gamma_r\), the
proposal family, or the chosen decoding representation. A soft candidate field
may permit

\[
\mathcal V^\Gamma_{r,s}
=
\mathcal V^{\mathrm{chart}}_{r,s},
\]

while a finite opportunity bank usually creates a strict subset.

Every reachable legal completion induces a canonical future trajectory

\[
\widetilde z_Y(t)
=
\operatorname{Roll}_{\vartheta_0}
\left(
\beta_{r,s},
Y;
t
\right),
\qquad
t\in W_{r,s}.
\]

Let

\[
S_\psi
\left(
Y;
\Xi_{r,s},
\widetilde z_Y
\right)
\]

be an unnormalized whole-completion quality score. The MAP decision is

\[
Y_{r,s}^\star
\in
\arg\max_{Y\in\mathcal V^\Gamma_{r,s}}
S_\psi
\left(
Y;
\Xi_{r,s},
\widetilde z_Y
\right).
\]

For a finite reachable set, the score also defines a structured distribution

\[
P_\psi
\left(
Y
\mid
\Xi_{r,s}
\right)
=
\frac{
\mathbf 1
\left[
Y\in\mathcal V^\Gamma_{r,s}
\right]
\exp
S_\psi
\left(
Y;
\Xi_{r,s},
\widetilde z_Y
\right)
}{
Z_\psi(\Xi_{r,s})
},
\]

where

\[
Z_\psi(\Xi_{r,s})
=
\sum_{Y'\in\mathcal V^\Gamma_{r,s}}
\exp
S_\psi
\left(
Y';
\Xi_{r,s},
\widetilde z_{Y'}
\right).
\]

This distribution preserves uncertainty over complete legal choreographies.
The MAP branch is only one possible decision rule. Marginalization, sampling,
risk-sensitive decoding, or stable-prefix posterior criteria remain possible.

For continuous-time support, the same idea requires an explicitly declared
base measure or point-process formulation; a bare normalized sum must not be
used as if the space were finite.

The formulation-level constraints are:

1. The object eventually checked and committed is a sequence of complete rows,
   not a bare set of timestamps.
2. Hard chart legality is evaluated by exact replay state.
3. Candidate support and chart legality are separate sets.
4. Every scored completion is paired with the gameplay-state trajectory it
   induces.
5. The complete committed history remains available alongside gameplay state.
6. Provisional branch state is isolated.
7. Only a selected stable time prefix mutates committed history and boundary.
8. Difficult, imbalanced, or unusual legal patterns remain scoreable rather
   than becoming implicitly illegal.

## 2. Worked example

Suppose the committed boundary is immediately after

\[
H_3
=
\bigl(
(29.40,1000),
(29.70,0020),
(29.90,0100)
\bigr),
\]

with

\[
g_{r,s}=29.90.
\]

Lane 3 is open at this boundary.

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

It selects rows at \(30.00\) and \(30.12\), and no row at \(30.25\). The path
alone does not determine the row actions.

Three materializations of the same temporal path are:

\[
B^{A}
=
(0030,1000,\bot),
\]

\[
B^{B}
=
(0020,1000,\bot),
\]

\[
B^{C}
=
(0001,1000,\bot).
\]

For \(B^A\):

- `0030` legally closes lane 3 at \(30.00\);
- `1000` taps lane 1 at \(30.12\);
- the resulting completion is chart-legal.

For \(B^B\):

- `0020` attempts another `LN_START` on already open lane 3;
- exact replay rejects the first row;
- the completion is chart-illegal.

For \(B^C\):

- `0001` taps lane 4 and leaves lane 3 open;
- the completion is legal if this is an intermediate horizon;
- it is illegal if the horizon reaches chart end and no later row closes lane 3.

Thus one temporal path can have:

- a legal completion that closes the long note;
- an illegal completion;
- a legal intermediate completion with a different future obligation.

The induced demand trajectories of \(B^A\) and \(B^C\) may also differ because
one ends sustained lane-3 occupancy while the other preserves it.

This example shows why a path proposer cannot establish final choreography
quality using timestamp plausibility alone. It must account for the value or
probability mass of legal row materializations.

If branch \(B^A\) wins but only the prefix through \(30.12\) is committed, then

\[
H_3
\longrightarrow
H_5,
\]

\[
g_{r,s+1}=30.12,
\]

and the candidate-null decision at \(30.25\) remains provisional. If the stable
prefix is committed through \(30.25\), then the no-row decision also becomes
fixed while the row index remains unchanged after the second row.

In either case, generation continues from the advanced history and boundary.
The active \(\Gamma_r\) remains the candidate information produced from the
older refresh snapshot until a new refresh is triggered.

## 3. Candidate prediction and active-boundary semantics

At refresh \(r\),

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

The predictor receives complete-song audio features. Its chart conditioning is
causal with respect to committed chart history, but not with respect to audio.

\(\Gamma_r\) may contain:

- a finite opportunity bank;
- a time lattice;
- point-process or hazard parameters;
- candidate probabilities or unnormalized scores;
- dense rhythmic evidence;
- a set of path proposals;
- another high-recall support representation.

The formulation does not assume that \(\Gamma_r\) is calibrated or normalized.

During the lifetime of \(\Gamma_r\), a partial commit changes

\[
H_{k_{r,s}}
\quad\text{and}\quad
\beta_{r,s},
\]

but not the provenance of \(\Gamma_r\). Recomputing candidate information from
the advanced history is a new refresh, not an implicit update inside the old
one.

This convention permits expensive parallel candidate computation while
retaining causal row materialization. It also creates a measurable staleness
risk: later commit revisions use candidate evidence produced from an older
history snapshot. The refresh policy must control that risk rather than
pretending it does not exist.

## 4. Chart legality and model support

### Chart-legal completion set

A future row sequence

\[
Y
=
\bigl(
(\tau_1,m_1),
\ldots,
(\tau_n,m_n)
\bigr)
\]

belongs to

\[
\mathcal V^{\mathrm{chart}}_{r,s}
\]

if and only if:

1. every \(\tau_i\) lies in \(W_{r,s}\);
2. row times are strictly increasing;
3. every \(m_i\in\mathcal M\);
4. the committed prefix \(H_{k_{r,s}}\) is unchanged;
5. exact replay from \(\beta_{r,s}\) accepts every row transition;
6. the resulting boundary records every open long note and pending obligation;
7. if the horizon reaches song end \(T\), every long note is closed by \(T\).

Intermediate horizons may end with open long notes.

Committing an `LN_START` commits open occupancy and its future close obligation.
It does not, by chart legality alone, lock one provisional `LN_CLOSE`.

The chart-legal set does not depend on:

- candidate-bank membership;
- demand magnitude;
- parity preference;
- hand balance;
- musical quality;
- whether a model considers the pattern likely.

Additional format-specific hard rules may be added explicitly, but must not be
smuggled into a soft quality score and then described as legality.

### Model-reachable completion set

The reachable set

\[
\mathcal V^\Gamma_{r,s}
\]

contains legal completions representable under the active candidate and
proposal contract.

For a hard opportunity bank,

\[
Y\in\mathcal V^\Gamma_{r,s}
\]

requires every row time in \(Y\) to be a member of the active bank suffix.

For a soft candidate field, \(\Gamma_r\) may affect the score without excluding
times, allowing

\[
\mathcal V^\Gamma_{r,s}
=
\mathcal V^{\mathrm{chart}}_{r,s}.
\]

If the best legal choreography is absent from
\(\mathcal V^\Gamma_{r,s}\), that is a support or proposal failure, not a chart
legality failure.

A conceptual support gap is

\[
\Delta_{\mathrm{support}}
=
\max_{Y\in\mathcal V^{\mathrm{chart}}_{r,s}}
S_\psi(Y)
-
\max_{Y\in\mathcal V^\Gamma_{r,s}}
S_\psi(Y).
\]

The first maximum is generally intractable, but oracle and controlled
representation experiments can estimate whether candidate support excludes
high-quality completions. Pointwise timing recall alone does not fully measure
this downstream gap.

## 5. Structured uncertainty over complete drafts

For an opportunity bank, let

\[
\mathfrak B^\Gamma_{r,s}
=
\left\{
B_{r,s}
\;\middle|\;
\operatorname{Rows}_{U_r}(B_{r,s})
\in
\mathcal V^\Gamma_{r,s}
\right\}.
\]

Because a complete bank assignment contains both selected rows and candidate
nulls, it defines a finite structured object.

Let

\[
Y(B)
=
\operatorname{Rows}_{U_r}(B).
\]

A structured distribution over complete assignments is

\[
P_\psi
\left(
B
\mid
\Xi_{r,s}
\right)
=
\frac{
\mathbf 1[B\in\mathfrak B^\Gamma_{r,s}]
\exp
S_\psi
\left(
Y(B);
\Xi_{r,s},
\widetilde z_{Y(B)}
\right)
}{
Z_\psi(\Xi_{r,s})
}.
\]

This representation distinguishes:

- local candidate evidence;
- uncertainty over temporal selections;
- uncertainty over row materializations;
- exact global legality;
- whole-path quality.

Hard-invalid assignments receive zero probability rather than merely a large
negative preference.

Depending on the architecture, inference may compute or approximate:

- MAP complete drafts;
- candidate-selection marginals;
- row-action marginals;
- suffix mass;
- samples;
- a stable-prefix posterior;
- a top-\(K\) branch set.

The formulation does not require exact partition-function computation. It does
require any approximation to be described as an approximation rather than
quietly replacing the target structured distribution with independent local
probabilities.

The candidate predictor's score and a proposal model's internal score are not
automatically calibrated whole-chart quality scores. They coincide only if the
training objective and normalization explicitly make them coincide.

## 6. Whole-completion quality

The formulation uses one joint score

\[
S_\psi
\left(
Y;
\Xi_{r,s},
\widetilde z_Y
\right).
\]

For analysis, it may be decomposed into diagnostic responsibilities:

\[
S_\psi
=
\mathcal A_\psi
\left(
Q_{\mathrm{music}},
Q_{\mathrm{history}},
Q_{\mathrm{game}},
Q_{\mathrm{target}}
\right),
\]

where \(\mathcal A_\psi\) is a calibrated aggregation.

A weighted additive instance is

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

This additive form is optional. It does not assert statistical independence,
equal units, or non-overlap among the terms.

The responsibilities are:

- \(Q_{\mathrm{music}}\) evaluates correspondence with complete audio and local
  or sectional musical structure;
- \(Q_{\mathrm{history}}\) evaluates continuity with committed chart history,
  including motifs and long-range structure not summarized by gameplay state;
- \(Q_{\mathrm{game}}\) evaluates the induced exact execution state, parity
  belief, demand progression, hand balance, and style geometry;
- \(Q_{\mathrm{target}}\), when present, compares induced demand with an
  externally planned target.

Gameplay state does not replace chart history. Two candidate futures may begin
from the same gameplay boundary but differ in motif continuity or structural
meaning.

The score must be calibrated across variable-length completions. In
particular:

- \(Y=()\) must not win merely because every added row accumulates negative
  score;
- dense drafts must not win merely because each added row contributes positive
  score;
- per-event, per-time, and whole-section terms must have declared units or
  normalization;
- candidate evidence must not be double-counted accidentally in both proposal
  and quality terms.

No particular normalization is fixed by the formulation.

## 7. Completion-aware temporal-path factorization

An architecture may propose temporal paths before materializing row actions.
For an opportunity-bank draft \(B\), define its temporal mask

\[
\operatorname{sel}(B)
=
P.
\]

For a proposed path \(P\), define its legal completion set

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

the path has no legal row materialization under the current boundary and
support.

Two valid completion-aware path values are:

\[
V_{\mathrm{MAP}}(P)
=
\max_{B\in\mathcal C(P)}
S_\psi
\left(
Y(B);
\Xi_{r,s},
\widetilde z_{Y(B)}
\right),
\]

and

\[
V_{\mathrm{marg}}(P)
=
\log
\sum_{B\in\mathcal C(P)}
\exp
S_\psi
\left(
Y(B);
\Xi_{r,s},
\widetilde z_{Y(B)}
\right).
\]

Set either value to \(-\infty\) when \(\mathcal C(P)\) is empty.

Under the structured distribution,

\[
P_\psi
\left(
P
\mid
\Xi_{r,s}
\right)
=
\sum_{
B:
\operatorname{sel}(B)=P
}
P_\psi
\left(
B
\mid
\Xi_{r,s}
\right).
\]

This is a whole-path marginal, not a product of independently plausible
candidate positions.

A staged architecture may therefore use:

\[
\text{parallel path proposer}
\longrightarrow
\text{completion-aware path evaluation}
\longrightarrow
\text{row materializer}
\longrightarrow
\text{exact rollout and final score}.
\]

The proposer may be autoregressive, diffusion-based, set-based, lattice-based,
or another model. Its proposal likelihood or denoising residual is not, by
definition, \(V_{\mathrm{MAP}}\) or \(V_{\mathrm{marg}}\).

Freezing a path before row materialization is formulation-consistent only when:

1. the path has at least one legal completion;
2. its evaluation accounts for the quality or probability mass of those
   completions;
3. the approximation error introduced by early freezing is measured.

Otherwise the system has reverted to position-only optimization and may select
a musically plausible path with no coherent hand choreography.

The final committed object remains the complete row sequence \(Y\), even when
\(P\) is an internal proposal or planning object.

## 8. Optional target-demand trajectory

An explicit planner may provide a desired demand trajectory

\[
\overline d_{r,s}(t),
\qquad
t\in W_{r,s}.
\]

The target term may be

\[
Q_{\mathrm{target}}
\left(
\Pi_d\widetilde z_Y,
\overline d_{r,s}
\right).
\]

The target trajectory is not automatically feasible. A planner may request a
shape that cannot be realized under:

- exact long-note obligations;
- candidate support;
- the available horizon;
- the declared demand dynamics;
- the intended musical alignment.

The generation problem must therefore search over legal realizations rather
than treating \(\overline d\) as a command that overrides chart semantics.

An alternative is an implicit policy

\[
P_\psi
\left(
Y_{\mathrm{future}}
\mid
X,
H_{k_{r,s}},
\beta_{r,s},
\Gamma_r
\right)
\]

without a separately supervised demand planner.

Whether explicit demand planning offers enough control to justify its
supervision, feasibility failures, and additional model mismatch remains an
open research question.

## 9. Legality, support, and quality are different layers

| Layer | Examples | Consequence of failure |
| --- | --- | --- |
| Chart legality | Valid `LN_START`/`LN_CLOSE`, one complete nonempty row per time, strict order, immutable prefix, complete-chart long-note closure | Reject as an invalid chart completion. |
| Model support | Candidate-bank membership, proposal vocabulary, bounded beam or lattice reachability | Completion is legal but unavailable to the current generator. |
| Quality | Musical alignment, motif continuity, natural parity, hand balance, jack intensity, chord burden, demand progression, style match | Completion remains legal and reachable but receives lower preference. |

This separation prevents soft stylistic choices from becoming accidental hard
rules.

The following are scored unless an explicit chart-format rule says otherwise:

- same-side repetition and jack intensity;
- non-alternating fingering;
- hand imbalance;
- dense chords;
- chordstream burden;
- unusual but legal long-note overlap;
- high local demand;
- slow recovery;
- asymmetric style.

A difficult or imbalanced pattern may be the intended style.

## 10. Branch selection and atomic commit

Each provisional branch begins from

\[
\left(
H_{k_{r,s}},
\beta_{r,s}
\right)
\]

and owns independent provisional history and state

\[
(\widetilde H,\widetilde z).
\]

After structured inference or reranking, the system selects a stable time
prefix of one winning legal branch.

Suppose the prefix fixes every row and no-row decision through

\[
g_{r,s+1}
>
g_{r,s}
\]

and contains \(n\ge 0\) rows in

\[
I
=
(g_{r,s},g_{r,s+1}].
\]

Then

\[
k_{r,s+1}
=
k_{r,s}+n,
\]

and

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

atomically.

The new boundary is

\[
\beta_{r,s+1}
=
\left(
g_{r,s+1},
z_{r,s+1}
\right),
\]

where

\[
z_{r,s+1}
=
\operatorname{Roll}_{\vartheta_0}
\left(
\beta_{r,s},
Y_I;
g_{r,s+1}
\right).
\]

Losing branches are discarded without changing:

- committed rows;
- fixed no-row decisions;
- exact occupancy;
- parity belief;
- clocks;
- continuous demand.

If an `LN_START` enters the stable prefix while its provisional close remains
outside, the committed boundary carries an open long note and a close
obligation. A policy may additionally lock a proposed close, but close locking
is not part of baseline chart legality.

If generation continues under the active refresh, it uses

\[
H_{k_{r,s+1}}
\quad\text{and}\quad
\beta_{r,s+1}
\]

with the original \(\Gamma_r\). A new candidate computation creates refresh
\(r+1\).

The lifecycle is:

> committed history and timestamped state  
> \(\rightarrow\) branch-local temporal and row proposals  
> \(\rightarrow\) exact legality and canonical rollout  
> \(\rightarrow\) structured whole-completion scoring  
> \(\rightarrow\) selected stable time prefix  
> \(\rightarrow\) atomic history, fixed-through time, and state promotion

## 11. What is fixed, chosen, and open

### Fixed by the target problem

- The committed object is a sequence of complete rows.
- Chart legality is decided by exact replay.
- Candidate support is distinct from chart legality.
- Candidate-level null is distinct from lane-level `EMPTY`.
- Every candidate completion is evaluated together with its induced gameplay
  trajectory.
- Committed history remains separate from gameplay state.
- Branch state is isolated.
- History, fixed-through time, and state advance atomically.
- Planning operations never enter materialized history.

### Current structural conventions

- Complete audio is available before generation.
- Candidate information is computed in rolling windows.
- One candidate snapshot may support several commit revisions.
- Temporal paths may be proposed separately, but path value must be
  completion-aware.
- Generation and analysis share the same canonical gameplay rollout.

### Open implementation choices

- the structure and training objective of \(\Gamma_r\);
- opportunity bank, lattice, hazard, set-query, or continuous-time support;
- autoregressive, diffusion, dynamic-programming, beam, sampling, or refinement
  inference;
- exact or approximate partition-function computation;
- the score aggregation \(\mathcal A_\psi\);
- stable-prefix selection policy;
- candidate refresh trigger;
- explicit target-demand planning;
- whether and when a provisional long-note close is locked.

## 12. Falsifiable hypotheses and open questions

### Falsifiable hypotheses

1. **Completion-aware path scoring improves choreography quality.**  
   Paths ranked using legal row completions should outperform paths ranked only
   by local timestamp plausibility. No controlled gain rejects the added
   completion-aware machinery for the tested setting.

2. **State-aware scoring improves physical continuity.**  
   Whole-trajectory scoring should reduce cross-window discontinuities and
   improve controlled parity, occupancy, or demand judgments relative to
   position-only and history-only baselines.

3. **Structured uncertainty is useful.**  
   Retaining path and materialization distributions should improve calibrated
   uncertainty, oracle top-\(K\) quality, or stable-prefix decisions relative to
   an equally sized single-path decoder.

4. **A demand trajectory can provide a useful control interface.**  
   Explicit or implicit demand-conditioned generation should achieve better
   local style and intensity control than global tags or one scalar difficulty
   condition alone.

### Open questions

- How should candidate coverage be measured so that local timing recall and
  downstream support loss are both visible?
- Which score normalization prevents event count or silence from dominating?
- Should path evaluation use MAP completion value, marginalized completion
  mass, or a risk-sensitive alternative?
- How much parity and demand uncertainty must be retained inside a branch?
- How far may a stable prefix advance without destroying useful future
  revision?
- When does candidate staleness require a new refresh?
- Should a future `LN_CLOSE` be locked when its `LN_START` is committed?
- Does an explicit target-demand planner justify its additional supervision and
  feasibility failures?