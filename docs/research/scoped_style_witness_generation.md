# Evidence-supervised scoped style modeling

This document specifies an exploratory study of section-level style assessment
on existing 4K charts, using agent-selected evidence notes as weak auxiliary
supervision. It defines the observations, source-action representation, shared
encoder and task branches, discriminating comparisons, and implementation handoff.
The model has no reported training results. Its architecture and numerical
defaults are research choices, not Ensomi V3 requirements.

The [generation contract](../formulation/notation.md) owns legal timed rows and
committed history. The [gameplay formulation](../formulation/gameplay-state.md)
owns style observations, continuation responses, and demand semantics. This study
predicts scoped assessments and can select source objects to accompany them;
it does not generate a new chart or validate a demand state.

## 1. Question and scope

The primary question is:

> Given a complete declared chart context, a section, and a queried style,
> does weak supervision from agent-selected evidence improve prediction of
> absent, supporting, and prominent assessments over assessment supervision alone?

The primary object is how source actions and their relationships organize the
complete section. Evidence selections offer locations at which an annotator
explained a judgment; reproducing that selection process is an auxiliary task.
The first comparison is `style-only` versus `style+evidence`, with identical
assessment inputs, encoder/readout architecture, and evaluation.

Keep the outcomes distinct:

| Role | Outcome and evidence required |
| --- | --- |
| Primary | Better held-out section assessments, with presence and positive-strength distinctions reported separately and concrete relationship contrasts inspected |
| Auxiliary diagnostic | Fit the agent's recorded evidence distribution, measured by conditional selection likelihood and reference agreement |
| Optional output use | Generate useful highlights for a supplied or predicted assessment, inspected in the complete source context |

Lower selection loss or more convincing highlights do not establish better
assessment. Conversely, evidence supervision may improve assessment without
accurately reproducing the agent's choice of examples. Test its value directly
through the primary task; witness generation is not a prerequisite stage.

Selection-history and source-representation comparisons are follow-ups. A more
capable auxiliary decoder may solve selection mostly within its own state,
without improving the shared chart representation. Evaluate that possibility
through assessment performance before retaining added decoder complexity.

Inputs are chart-only. The annotation vocabulary concerns organization supported
by source actions and declared chart context, rather than concepts requiring
aligned audio or mapper intent. Audio-conditioned choreography generation is a
separate research question.

## 2. Definitions, judgments, and explanatory scope

Keep four kinds of objects separate:

| Object | Meaning | Example |
| --- | --- | --- |
| Source fact | Determined by the chart and exact replay | A lane is occupied before an attack; two groups have disjoint members |
| Scoped assessment | A semantic observation under a versioned concept definition | Trill is present/supporting on a specified interval |
| Evidence selection (witness) | Objects the agent emphasized while explaining an assessment in context | Selected members locating an alternating episode or its disruption |
| Learned representation | A computational hypothesis about useful information | A hand embedding or selection-memory state |

A witness is an object selection attached to the complete arrangement. It need
not be exhaustive, minimal, contiguous, or sufficient when shown in isolation.
Unselected notes remain part of the explanation's context and can supply
counterevidence. Replacing a highlight mask changes what is emphasized; deleting
notes changes the chart and is a different intervention.

Evidence hints at locations and relationships relevant to a judgment. It is not
an exhaustive importance annotation or a measurement of each object's causal
contribution.
An agent can select one of several equivalent examples and omit a decisive
interruption that remains visible in context. Unselected objects need not be
irrelevant; selected objects need not support presence. In an absent record,
they may locate why the queried organization does not hold. The same selected
objects can support different assessments in different complete arrangements.

The auxiliary mask target records the agent's selection behavior. Its zero bits
mean unselected in that record, not semantically unimportant. Do not define
section strength by summing note importance scores, force attention weights to
match evidence masks, or treat selection, information routing, and causal
influence as interchangeable quantities.

The five concepts are independently assessed, versioned experimental categories.
They can overlap, including multiple prominent concepts in one section. The
following distinctions come from the dataset's frozen Foundation:

| Concept | Organization to examine | Distinction the model must be able to retain |
| --- | --- | --- |
| Jack | Repeated-column organization in complete attack groups | Fixed disjoint A/B Trill can return to each lane every two rows while remaining Jack absent |
| Stream | Flow, direction, chord placement, continuity, and resets | Continuous activity or fixed A/B alternation alone is insufficient |
| Trill | Repeated alternation between fixed, disjoint column groups | Groups can be chords, and alternation can be within or across hands; speed and episode duration matter without a supplied universal threshold |
| Tech | Concrete sequence, rhythm, or articulation relationships that make an arrangement hard to anticipate or follow through familiar patterns | Variation, irregularity, density, unfamiliarity, and absence of other labels are insufficient substitutes |
| LN coordination | Presses, releases, and taps interacting across occupied columns | At least two simultaneously LN-occupied columns is necessary under this Foundation, but overlap alone is insufficient |

These distinctions guide representation and evaluation. They do not turn
computed features into semantic labels. A recurrence edge is not a Jack detector;
a gap is not automatically a reset; simultaneous holds are not automatically LN
coordination. The [Foundation](https://github.com/ensomi-labs/beatmap-lens/blob/647009ab60ed69d98190712a6ab025807cca07b8/annotation/foundations/15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97.json)
and its scoped calibration judgments retain authority over these meanings.

Supporting and prominent are ordered positive judgments of expression strength,
not confidence levels or equally spaced numerical amounts. An absent assessment
is an explicit negative for the concept. Unresolved and unreviewed describe
observation status and do not become negative examples.

## 3. Dataset and exploratory use

Use [mania-pattern-annotations v3](https://huggingface.co/datasets/sed-i/mania-pattern-annotations/tree/b22a7a443783e05fee4db4b1d22b8e573ad448ae),
HF commit `b22a7a443783e05fee4db4b1d22b8e573ad448ae`, publication schema v4,
Foundation `f-15fa68913bdb2bf3`. Its selected machine table is method
`method-5ebd91cd0db19242f14bf5d4fc96b328c792b185ec4ab2bc785ca2e13a4d056c`.

The snapshot contains 592 human judgment records and 4,403 machine judgment
records across 545 annotated 4K sources. The human records contain 300 positives
and 292 negatives; the machine records contain 1,646 positives and 2,757 negatives.

| Concept | Human positive records | Machine positive records |
| --- | ---: | ---: |
| Jack | 67 | 547 |
| Stream | 84 | 697 |
| Trill | 44 | 167 |
| Tech | 64 | 95 |
| LN coordination | 41 | 140 |

These are record counts before exact-cell deduplication. The human table has
590 exact cells, including one agreeing duplicate positive cell and one agreeing
duplicate negative cell. The machine table has 4,403 distinct cells. Treat rows
from one source and overlapping scopes as correlated observations.

All machine negative judgments also carry nonempty witnesses. The frozen
[labeler role](https://github.com/ensomi-labs/beatmap-lens/blob/647009ab60ed69d98190712a6ab025807cca07b8/annotation/methods/astra-1000-20260912/labeler.md)
requests simultaneous judgment and selection, including for absent assessments.
Those objects can supervise evidence selection conditioned on absent; they do
not become evidence for the concept's presence.

Source-level inspection of the machine positives found:

- 1,229 of 1,646 selections occupy multiple runs in the original attack-row
  sequence, with a median of three runs. A run ends at an unselected attack row;
  this count does not define a semantic episode.
- 41 positive selections contain a partially selected attack row.
- 311 positive records have eligible entering LNs; 47 select an entering LN.
- The median selection has 28 objects, and the median scope is 10 seconds.

These measurements motivate distributed-witness, partial-chord, and boundary-LN
inspection slices. None is an automatic quality filter. In particular, leaving
an entering LN unselected does not remove its occupation from context. Retain
joint masks and entering-LN decisions to express the task correctly, but do not
use aggregate gains to claim that these scarce cases have been learned. Report
their held-out record and source-group counts alongside concrete outcomes.

### 3.1 Cohorts and proportionate provenance

The initial training comparison uses all resolved records from the selected
machine method: absent, supporting, and prominent. Both arms receive the same
assessment cohort and annotation method. The positive-only counts above describe
the evidence audit, not the full training population. Report each concept and
assessment class separately, including source-group counts; Tech and LN positive
judgments have smaller support. Unresolved and unreviewed supply no class target.

Held-out human assessments directly evaluate the primary task. A human-confirmed
assessment can retain machine notes and rationale without invalidating its use
as an assessment target. Preserve judgment origin and confirmation mode; do not
call retained selections independent human evidence gold. Collapse agreeing
exact-cell duplicates within each judgment origin for evaluation, and report
unresolved conflicts rather than converting them to absent. Explicit confidence
concerns the assessment. No new note selection, confidence filter, or annotation
repair campaign is a prerequisite for this comparison.

Use the following minimal reproducibility and separation rules:

1. Pin the dataset revision and preserve source hash, cell ID, assessment,
   vocabulary, rate, and judgment origin. Verify source bytes before resolving
   source-line identities; a changed chart is a different input.
2. Assign an exact source, all its scopes, and all its labels to one split.
   Group known duplicate chart versions and known shared-song/audio identities
   when that information is already available. Do not require new audio
   acquisition or a comprehensive similarity investigation to start.
3. Keep reference membership and the true assessment out of the chart encoder
   and assessment branch. Only the auxiliary selector receives the true
   assessment and preceding reference masks during training. Exclude rationale,
   human comments, audit text, source-line numbers, chart titles, and provenance
   from learned features. Source-line numbers are output identities only.
   Selector state must not feed back into the encoder or assessment branch.
4. Record aggregate exposure to calibration examples or consulted human examples
   when available. Such exposure is a limitation of an exploratory result, not
   a default reason to discard every dependent source or construct a transitive
   provenance exclusion graph. Missing historical exposure metadata is not a
   training blocker.

Use one deterministic 80/10/10 train/validation/test grouping, with split seed
`0`, and share it across arms and training seeds. A concrete implementation can
hash the split seed and stable group ID into these three intervals. Freeze the
assignment before training; report per-tag support rather than repeatedly
redrawing splits for favorable results. If a tag has no validation or test
support, report that limitation and revise the split explicitly before a
comparison intended to evaluate that tag. Preserve per-class support; a missing
positive-strength class limits that distinction even if the concept has records.

Apply the same grouping to supplementary human records. Human records on
training-group charts can be used for development inspection but do not become
held-out evidence. Reserve test-group human assessments for final evaluation;
report them separately from machine assessments without pooling duplicate cells
across origins as independent observations. A subsequent mixed-source training
experiment must use human precedence at exact cells and state its sampling
weights; it is a distinct cohort comparison. The main run need not wait for
that extension.

Describe the initial result as source-grouped exploratory evaluation. Broader
claims about unseen songs, independently collected human witnesses, or annotation
method generalization require the corresponding additional evidence. The dataset
split named `full` supplies no benchmark split by itself.

## 4. Formal observation and visible context

Write one observation as

$$
r=(C,S,Q,\ell,y^*,E^*,\nu),
\qquad S=[a,b)\subseteq Q=[u,v),
$$

where $C$ is the exact source chart, $S$ the judged section, $Q$ its recorded
review context, $\ell$ the queried concept, $y^*$ the recorded assessment, $E^*$
the recorded evidence objects when available, and $\nu$ the identity and origin
metadata. The assessment model receives only the chart representation of $Q$,
the scope markers for $S$, and $\ell$.

The first model uses each record's published `review_context`; it does not read
unbounded chart context. Recover all attacks and releases in $Q$, complete source
objects overlapping $Q$, and exact occupation entering $Q$. Full endpoints of
those visible LN objects remain available even outside $Q$. Obtaining boundary
occupation by replay does not expose earlier chart content as learned features.

Previous/next attack features and recurrence edges use attacks visible in $Q$.
Use an explicit unavailable flag when no such neighbor is visible; do not encode
that as proof that the whole chart has no predecessor or successor. Hold age and
remaining duration may use the preserved endpoints of visible holds. Do not
quietly expand $Q$ for one model arm.

Write $C_Q$ for this visible representation and $X=(C_Q,S)$. The shared chart
encoder produces $H=F_\phi(X)$. The primary prediction is

$$
p_\eta(y\mid R_\omega(H,S,e_\ell^{\mathrm A})),
\qquad y\in\{\mathrm{absent},\mathrm{supporting},\mathrm{prominent}\}.
$$

The query asks for an assessment of $\ell$ without supplying presence or strength.
Each concept has its own three-outcome distribution; the five concepts are not
mutually exclusive softmax classes. Supporting/prominent remain distinct targets.

The auxiliary selector explains a supplied assessment:

$$
p_\theta(Z\mid H,S,\ell,y).
$$

Training supplies $y^*$ to this branch, including absent, and uses $E^*$ only for
its auxiliary supervision. The encoder is the sole shared trainable component;
branch embeddings, readouts, and dynamic states are separate. Evidence gradients
may improve $F_\phi$ during training, but target values cannot enter the assessment
forward pass through a shared state or cache.

Assessment inference needs no selector, evidence, or true assessment. Optional
highlight generation can condition on a predicted $\widehat y$; diagnostics using
$y^*$ must be labeled as supplied-assessment explanation rather than end-to-end
assessment. A selection distribution's maximum probability is not a presence
score or a substitute for the assessment branch.

## 5. Source objects, event rows, and selection decisions

A source object $n_i$ has a stable source identity, lane $j_i$, attack time $s_i$,
type, and original endpoint $e_i$. For a tap, $e_i=s_i$; for an LN, $e_i>s_i$.
Preserve source milliseconds for identity and membership. Convert elapsed times
to seconds for model features without rounding or changing interval conventions.

The eligible object set is

$$
\mathcal U_r=
\{n_i:a\le s_i<b\}
\;\cup\;
\{n_i:\mathrm{LN}(n_i),\ s_i<a<e_i\}.
$$

An LN ending exactly at $a$ supplies a boundary release fact, not an overlapping
candidate. A head at $b$ is outside $S$. A selected LN extending beyond $b$
retains its full endpoint. Invalid source-to-row conversions must be reported;
do not repair timing, clip holds, or split simultaneous same-lane actions into
invented timestamps to force compatibility with the V3 row language.

Build an encoder timeline from the union of visible attack and LN-close times,
plus boundary markers needed for $Q$ and $S$, including $a^-$ and $b^-$ for the
section readout. At each source timestamp, encode all four lanes together. A tap
has no additional LN-close action. Synthetic boundary markers have a phase flag
and do not create source actions.

The decoder timeline starts with a boundary decision at $a^-$, followed by all
actual event rows with timestamps in $[a,b)$. At $a^-$, only LNs satisfying
$s_i<a<e_i$ can be selected. At each actual row, only objects whose heads occur
there can be selected. Process the boundary before a source row at $a$, even
though their numerical timestamp feature is the same. Exact state before that
source row still includes any LN that closes at $a$.

For decoder step $d$, let $a_{d,j}\in\{0,1\}$ indicate the eligible source
object in lane $j$, and define

$$
\mathcal B_d=\{b\in\{0,1\}^4:b_j\le a_{d,j}\},
\qquad z_d\in\mathcal B_d.
$$

At most one object is eligible per lane per decision, so $|\mathcal B_d|\le16$.
Release-only rows have the single selection mask `0000`. They can update decoder
state but contribute zero log loss. A boundary with no eligible LNs behaves the
same way. Padding has neither a loss nor a state update.

Every candidate object has exactly one auxiliary selection decision. Its recorded
target bit is one exactly when it appears in $E^*$. This target fits agent
selection behavior, not object importance. Verify $E^*\subseteq\mathcal U_r$ rather
than silently dropping unmatched references. Object selection round-trips to
source IDs, including entering LNs.

$$
Z=(z_0,\ldots,z_D),\qquad
E(Z)=\{n_i:\text{its decision bit is one}\},\qquad
K(Z)=|E(Z)|=\sum_{d,j}z_{d,j}.
$$

The timeline is known, but $K$ is free. No oracle count or EOS is supplied.
`0000` is a skip selection, not an empty row materialized in a generated chart.
The distribution permits an empty output; report its frequency by supplied
assessment rather than imposing a nonempty or fixed-size constraint. An empty
selection does not predict absent.

## 6. Exact lane facts and hand coordinates

At each encoder position retain, per lane:

- Tap, LN-start, and LN-close action flags; before/after occupation.
- Time since the previous visible attack and until the next visible attack,
  with availability flags. At an attack, these refer to strict predecessor and
  successor attacks, not the current event.
- Age and remaining duration of an active or starting LN, with explicit
  applicability flags. Preserve the closing object's duration at its close.
- Boundary phase, section membership, elapsed time since the previous event,
  and relative position within the declared scope/context.

Specify before/after phase for every hold feature: a newly starting hold has
zero age after the row; a closing hold has zero remaining duration before its
close and is absent after the row. Feature extraction must agree with replay.
Elapsed time without an action does not release a hold.

Retain exact values in the prepared representation. For the network, use seconds
and a signed transform such as $\operatorname{sign}(x)\log(1+|x|/1\mathrm{s})$
for potentially large time differences. If additional scaling is fitted, fit it
on training data only. Preserve availability separately from a numerical zero.

Use canonical coordinates

$$
L=(1_{\mathrm{outer}},2_{\mathrm{inner}}),
\qquad R=(4_{\mathrm{outer}},3_{\mathrm{inner}}).
$$

The dataset's columns are zero-based; the formulation's lane names are one-based.
Centralize this conversion and the mirror permutation. Lane embeddings can share
their feature transformation, but concatenate outer and inner outputs in order.
Do not average the two roles or concatenate hand pairs as though they were the
serialized lane order $(1,2,3,4)$.

## 7. Explicit channels for source-action relationships

Construct the graph from the complete visible chart before applying any target
or generated selection. Nodes are hand-row representations $(t,h)$, each retaining
its ordered outer/inner features. The graph is label-independent; concept-specific
interpretation is learned from its use by the model.

| Channel | Direct connections | Required attributes and purpose |
| --- | --- | --- |
| Event succession | Each row to the immediately preceding/following event row in each hand | Signed elapsed time and source/synthetic phase; includes release-only transitions |
| Attack-group succession | Consecutive global attack rows, and attack rows two positions apart, across hand nodes | Complete group membership in query-relative hand coordinates, role intersections, elapsed time; exposes changing chords and A/B returns despite intervening release rows |
| Simultaneous interaction | The two hands at the same row | Original lane actions and occupied roles; preserves full chords and simultaneous press/release organization |
| Same-lane recurrence | Consecutive visible attacks in each original lane, in both directions | Outer/inner role, elapsed time, count of intervening global attack rows, and endpoint attack groups; does not equate recurrence with Jack |
| LN object identity | A visible head and its visible close, in both directions | Same source object, original duration, endpoint roles; connects the action that opens a hold with its actual release |
| Occupied-role interaction | An action row to the visible head/close of each hold active immediately before or after that row, including other lanes | Acting role, occupied role, before/after occupancy, same/cross hand, and endpoint timing; exposes actions occurring within another hold's lifetime |

Outside-$Q$ endpoints do not become fabricated event rows. Their object identity,
age, and remaining duration are retained as exact lane/boundary features; emit
an endpoint edge only when its endpoint is an encoded node. Same-row interaction
still exposes all active lanes at the action row. Record missing endpoint-edge
availability so absence of an edge is not interpreted as absence of a hold.

Use a self edge at every node. Multiple relationships can connect the same node
pair; combine their type and role-pair flags into one edge descriptor so the
neighbor appears once in the attention normalization. Preserve multiple role
pairs instead of arbitrarily retaining the first one.

All lane/group attributes are expressed relative to the querying hand and its
other hand, in outer/inner order. Absolute left/right IDs must not break the
canonical mirror transformation. The graph does not contain inferred labels,
semantic episode boundaries, predicted difficulty, or selection-derived edges.

This design exposes facts needed to compare interpretations. It does not require
the neural model to call a boundary a reset, recognize a fixed group as Trill, or
treat independent releases as strong coordination. Those remain hypotheses to
evaluate against the scoped judgments.

## 8. Shared encoder, assessment readout, and auxiliary selector

### 8.1 Shared hand encoder and relation attention

For each hand, concatenate ordered lane embeddings and row-time features, then
apply the same BiGRU parameters to the two separate sequences:

$$
B^L=F_{\mathrm{hand}}(X^L),\qquad B^R=F_{\mathrm{hand}}(X^R).
$$

Apply one relation-aware attention block over the hand-row nodes. For a query
node $i$ and a visible neighbor $j$ with relation descriptor $R_{ij}$, one head is

$$
\begin{aligned}
e_{ij}&=\frac{(W_QB_i)^\top(W_KB_j)}{\sqrt{d_k}}+b_\rho(R_{ij}),\\
\alpha_{ij}&=\operatorname{softmax}_{j\in\mathcal N(i)}e_{ij},\\
\widetilde B_i&=\sum_{j\in\mathcal N(i)}\alpha_{ij}
\left(W_VB_j+r_\rho(R_{ij})\right).
\end{aligned}
$$

Use residual connections, layer normalization, and a small feed-forward block.
The resulting $H=\{c_t^L,c_t^R\}_t=F_\phi(X)$ retains two contextual hand vectors
at every encoded position. Both branches read these immutable vectors. They
depend on chart facts and scope/context markers, not the queried concept,
assessment, or evidence mask. Relation values are included as well as relation
biases so relationship type can affect what is transmitted.

### 8.2 Primary section assessment

For the queried concept, compute

$$
r_\ell=R_\omega(H,S,e_\ell^{\mathrm A}),
\qquad
p_\eta(y\mid r_\ell)=\operatorname{softmax}_{y}
\operatorname{MLP}_\eta(r_\ell).
$$

The three logits describe absent, supporting, and prominent for this concept.
Categorical cross-entropy does not assign equal numerical distances to the
ordered judgments. Presence and the supporting/prominent distinction are
evaluated separately in Section 11.1.

Use a small temporal readout with the following starting parameterization:

1. At each original event row inside $S$, combine the two contextual hand vectors
   by averaging a shared nonlinear MLP projection applied to $(c_t^L,c_t^R)$ and
   to $(c_t^R,c_t^L)$. Each projection sees both ordered hand vectors; this retains
   capacity for within-row interactions while making the row representation
   mirror-invariant. Apply the same projection to the boundary markers.
2. Append the branch's concept embedding, elapsed time since the preceding
   readout position (zero at $a^-$), section-relative time, and source/boundary
   phase. Process the full chronological sequence with one small BiGRU. Include
   synthetic $a^-$ and $b^-$ markers with exact boundary
   facts to represent entering occupation and time after the last in-scope event.
   A source row at $b$ is excluded; a source row at $a$ follows $a^-$. Padding
   neither updates state nor enters aggregation.
3. Concatenate the terminal forward/backward states, the masked mean of outputs
   at actual in-scope event rows, and the transformed section duration. Use an
   explicit empty-row indicator and zero mean when there are no such rows; the
   boundary states still represent silence or sustained entering holds. A small
   MLP maps this summary to the assessment logits.

This readout can use the temporal distribution of relationships, repetitions,
duration, interruptions, and coexistence of organizations. It does not reduce
the section to its strongest local response or pool only near evidence notes.
Outer review context influences the contextual vectors, but its rows never
enter the section aggregation as in-scope activity. Synthetic markers are not
attacks or evidence candidates. No fixed coverage or duration threshold defines
prominent; the assessment boundary is learned from the scoped supervision.

Neither $y^*$, $Z^*$, generated selections, nor selection-GRU state enters
$R_\omega$ or the assessment head. With fixed weights in evaluation mode,
removing the auxiliary branch must leave all assessment probabilities unchanged.

### 8.3 Auxiliary selection history

Maintain exact selection facts separately from learned memory:

- The most recent selected in-scope attack in each lane, if any, and its time.
- Whether the previous original attack in that lane was selected. Use an
  unavailable state when that attack had no selection decision in this scope;
  a context-only predecessor is not an observed rejection.
- Whether a currently active LN was selected, including the boundary-selected
  entering objects. An unselected active LN remains active in chart state.

Selection time at $a^-$ and an entering LN's original head time are different
quantities. Do not invent an attack at the boundary or use the boundary timestamp
as that object's original recurrence position. Object IDs maintain these links;
their arbitrary numeric source-line values are not embedded.

Every beam hypothesis or sampled trajectory owns its learned and exact selection
histories. Chart context, relation indices, and replay facts are shared immutable
inputs.

### 8.4 Auxiliary joint row distribution

Let $q_{d-1}^L,q_{d-1}^R$ be selection-memory states, initialized to zero before
the boundary decision. Build $v_d^h$ from chart context, both prior memory states,
exact selection history, and selector-owned embeddings $e_\ell^{\mathrm E}$ and
$e_y^{\mathrm E}$ of the queried concept and supplied assessment. All hand-indexed
fields use self/other order, retaining outer/inner order within each hand.
For a candidate mask $b=(b^L,b^R)$ in canonical hand-role coordinates, use

$$
\begin{aligned}
s_d(b)&=U_\theta(v_d^L,b^L)+U_\theta(v_d^R,b^R)
+V_\theta(v_d^L,v_d^R,b^L,b^R),\\
p_\theta(z_d=b\mid z_{<d},H,S,\ell,y)
&=\frac{\exp s_d(b)}{\sum_{b'\in\mathcal B_d}\exp s_d(b')}.
\end{aligned}
$$

Each hand has four two-bit mask choices. $U$ is shared across hands. A concrete
symmetric interaction is

$$
V(v^L,v^R,b^L,b^R)=\tfrac12\left[
G(v^L,v^R,b^L,b^R)+G(v^R,v^L,b^R,b^L)\right].
$$

This decomposition organizes parameter sharing; $U$ and $V$ are not identified
physical contributions or interpretable hand-demand measurements.

After deciding the complete row, update both hands from the pre-update states:

$$
q_d^h=\operatorname{GRU}_\psi\left(
[c_d^h,q_{d-1}^{\bar h},\operatorname{Emb}(z_d^h),
\operatorname{Emb}(z_d^{\bar h}),e_\ell^{\mathrm E},e_y^{\mathrm E},\Delta t_d],q_{d-1}^h\right).
$$

The GRU parameters are shared. Do not update one hand from the other hand's
already-updated state. Update exact selection history after reading its prior
values for the decision. Forced-zero release rows still advance the learned
state and update the selected-active-LN facts through the source release.

The sequence distribution is

$$
p_\theta(Z\mid H,S,\ell,y)
=\prod_{d=0}^{D}p_\theta(z_d\mid z_{<d},H,S,\ell,y).
$$

This gives generation and scoring of complete selections over the same support.
It does not assign each source note a history-independent semantic importance.
Autoregression constrains selection history only: the encoder sees the complete
declared review context, including later source rows. This is retrospective
selection, not a causal restriction on chart observation.

### 8.5 Mirror behavior

For the mirror $\mu$ that exchanges hands while preserving outer/inner roles,
require, in deterministic evaluation mode,

$$
\begin{aligned}
p_{\mathrm A}(y\mid\mu X,\ell)&=p_{\mathrm A}(y\mid X,\ell),\\
p_\theta(\mu Z\mid F_\phi(\mu X),S,\ell,y)
&=p_\theta(Z\mid F_\phi(X),S,\ell,y).
\end{aligned}
$$

Here $p_{\mathrm A}$ abbreviates the complete assessment predictor. Check its
readout symmetry as well as the graph transformation, per-step selection
probabilities with mirrored histories, and full-sequence scores. Shared encoders
alone are insufficient. Dropout is disabled for this check. Tied maxima can
produce different single decoded outputs under an asymmetric tie-break rule;
compare probabilities and mapped tied
candidates before calling that a distributional symmetry failure. Arbitrary lane
permutations and playback-rate changes are not label-preserving augmentations.

## 9. Primary and auxiliary training objectives

For a resolved record, the primary loss is

$$
L_r^{\mathrm A}
=-\log p_\eta\left(y^*\mid
R_\omega(F_\phi(X),S,e_\ell^{\mathrm A})\right).
$$

Let $J_r=\{d:|\mathcal B_d|>1\}$ be the nontrivial selection decisions. For a
record with usable evidence, the auxiliary loss is

$$
L_r^{\mathrm E}
=-\frac{1}{|J_r|}\sum_{d\in J_r}
\log p_\theta(z_d^*\mid z_{<d}^*,F_\phi(X),S,\ell,y^*).
$$

Use teacher forcing with the complete preceding target row masks, confined to
the auxiliary branch. This is a length-normalized conditional likelihood of
agent selections: many decision rows do not automatically give a record greater
weight. Also retain unnormalized sequence NLL for diagnostics.

Validate source and evidence identities before applying an availability mask.
Write $m_r=1$ when evidence is recorded and has nontrivial decisions; otherwise
omit the auxiliary term and set its contribution to zero. Missing evidence and
an explicitly empty recorded selection are distinct: an empty selection with
eligible objects still has a
mask target. A section with no eligible objects can retain its assessment loss;
a nonempty reference with no eligible decision or an unresolved source identity
is a data-contract error, not missing evidence to silently mask away. Report
auxiliary eligibility and failures by assessment. Both arms retain the same
valid assessment cohort regardless of evidence availability.

The combined record loss is

$$
L_r=L_r^{\mathrm A}+\beta m_r L_r^{\mathrm E},\qquad\beta\ge0.
$$

Here $L_r^{\mathrm E}$ is defined as zero when $m_r=0$. The two losses update
the shared encoder; each also updates only its own branch parameters. Assessment
logits always come from the chart-only forward path. The true assessment enters
only the auxiliary conditioning, and selection histories never feed back into
$H$ or the primary readout.

The evidence term is deliberately biased toward this annotation method's
selection policy. It penalizes alternative masks even when they could locate
equally useful relationships. Keeping this ordinary NLL does not assert that
zero-bit objects lack semantic value. Its justification is improvement on the
primary validation task, not maximal reference-mask fit. No hard evidence
bottleneck, attention-mask matching loss, sampling-gradient estimator, or RL
objective is required.

Sample a concept uniformly, then a training group uniformly among groups with
that concept, then a resolved record uniformly within that group/concept.
Do not resample by evidence membership or selection count. Both arms use the
same assessment records, sampled order, batches, and number of update
opportunities. This balances concepts and groups without claiming the sampled
assessment proportions match whole-chart prevalence. Any class reweighting is
a separately declared change applied equally to both arms.

Use ordinary supervised gradients and decoupled AdamW weight decay. Start the
paired pilot with $\beta=0$ for `style-only` and $\beta=0.1$ for `style+evidence`.
Select checkpoints and decide whether to retain the auxiliary weight using
validation assessment NLL, with the class distinctions in Section 11.1 as guards.
If a different positive weight is needed, specify a bounded validation-only
comparison and its added budget before running it, freeze the weight before
test evaluation, and report the search. Selection NLL must not select $\beta$
or the checkpoint. The initial pair does not require a weight sweep.

### 9.1 Optional evidence decoding

Assessment inference runs $F_\phi$, $R_\omega$, and $p_\eta$ only. When evidence
outputs are also wanted, supply a declared assessment to the auxiliary selector
and decode

$$
\widehat Z_\lambda\approx\arg\max_Z
\left[\log p_\theta(Z\mid H,S,\ell,y)-\lambda K(Z)\right].
$$

Beam expansion adds original model log probabilities and subtracts
$\lambda\operatorname{popcount}(b)$. Do not subtract the cost from logits and
renormalize: history-dependent normalizers generally change the sequence
objective. Forced decisions retain zero log probability and zero count while
advancing state. A penalty on reference count would be constant in the trainable
parameters; a generated-count training objective is outside this initial study.

Use $\lambda=0$ for the default diagnostic, with greedy and beam width 8. An
optional quantity sensitivity grid is $\{0,0.05,0.1,0.2\}$ nats per object.
Report the grid without choosing a favorable test-set point, and compare quality
at overlapping achieved counts. Selections at different costs need not be nested.
Count cost is a post-training output preference, not learned explanation value,
style strength, or a reason to change the assessment prediction.

Ancestral diagnostics draw each joint mask from the original normalized valid-mask
probabilities at $\lambda=0$ and temperature 1, updating history with the sampled
mask. Use no reference history, top-k/top-p truncation, empty-output rejection,
or best-sample reranking. Section 11.3 specifies the diagnostic budget. These
procedures apply to trained auxiliary selectors; `style-only` has no trained
evidence output to compare.

## 10. Discriminating comparisons

### 10.1 Evidence supervision: the first comparison

| Arm | Training supervision | Assessment inference |
| --- | --- | --- |
| `style-only` | $L^{\mathrm A}$ only | Complete chart context, section, and queried concept to three assessment probabilities |
| `style+evidence` | The same $L^{\mathrm A}$ plus $\beta m L^{\mathrm E}$ | Exactly the same inputs and assessment architecture; selector removed |

Use identical shared-encoder and assessment-readout architecture, initial weights
for matching components, paired seeds, assessment minibatches, optimizer settings,
and validation checkpoint rules. Isolate random streams so initializing or
sampling the auxiliary branch does not change the baseline's chart/assessment
initialization or data order. Report additional training parameters and runtime;
inference uses the same components and input information in both arms.

This intervention tests whether agent evidence is useful training supervision
for section assessment. Evidence is privileged training information in the
ordinary sense that it is unavailable and unnecessary at assessment inference;
no teacher-student system is required. A gain supports the utility of this
auxiliary supervision under the tested cohort and capacity. It does not alone
identify a particular relation channel as the mechanism or prove semantic
understanding. If assessment gains occur without clear relationship-specific
changes, generic auxiliary regularization remains an alternative explanation.

### 10.2 Auxiliary selection history: a follow-up

When evidence supervision helps, or evidence generation has a separate declared
use, compare two `style+evidence` variants while keeping the primary task,
assessment readout, cohort, $\beta$, and model-selection rule fixed:

| Auxiliary variant | Selector input history |
| --- | --- |
| `relations-context` | Chart/concept/assessment recurrence only |
| `relations-history` | The same recurrence plus learned and exact selection history |

The initial auxiliary branch uses `relations-history` as specified in Section 8.
For `relations-context`, keep GRU state sizes, update schedule, and readout
capacity but replace mask embeddings and exact selection fields with dedicated
constant null inputs. Verify that changing preceding selections cannot change
its probabilities. It still sees the complete chart context and supplied
assessment; its output decisions are conditionally independent given those inputs.

This compares the learned and exact history bundle. Exact-only or learned-only
history variants can attribute a later benefit, but are not initial arms. In the
context variant the additive log-probability/count objective can be maximized
exactly row by row; greedy attains the sequence optimum at every cost. The
history variant generally uses approximate search, which must be distinguished
from a difference in the learned distributions.

History can learn to maintain one of several equally valid example choices, or
to extend a highlight run, without improving gameplay distinctions. Conversely,
a deterministic evidence mapping from full context can be represented without
output dependence. Lower selection NLL neither proves better chart representation
nor determines which variant to retain: assessment improvement remains primary,
with any separate evidence-output utility reported on its own terms.

### 10.3 Source-action representation and other follow-ups

Hold the task supervision and decoder choice fixed when comparing the structured
encoder with a generic four-lane row BiGRU and ordinary context interaction.
Supply the same exact lane facts, context extent, cohort, and comparable capacity.
The generic encoder retains ordered lanes without hand parameter sharing or typed
relation edges. State its attention support and parameter count.

That comparison tests a representation bundle. To isolate relation descriptors,
retain hand sharing and sparse neighbor topology while removing descriptors;
to isolate hand sharing, hold relation processing fixed. Apply any mirror
augmentation equally and report measured symmetry behavior. Exact occupation
is computable from chart input, so supplying it explicitly adds a representation
of known facts, not an independent physiological measurement.

Other diagnostics have limited purposes:

- On a scope with assessments for multiple concepts, query each independently,
  including assessed absences, and examine its three-outcome probabilities.
  Correlated assessments or shared evidence are not failures. A label-null
  training comparison is a follow-up for attributing conditioning benefits.
- Inspect partial-chord and entering-LN cases. Scarce targets limit claims about
  those selection capabilities; an optional independent-lane head comparison
  is not a prerequisite for the primary experiment.
- A previous-mask-only decoder can test whether short auxiliary history suffices.
  Arbitrary selection-GRU states cannot be merged by previous mask for ordinary
  CRF forward-backward or Viterbi inference. A CRF is not an initial requirement.
- Training-set assessment priors provide a class-imbalance reference. Empty/full
  selections and position/activity heuristics are auxiliary agreement references,
  not semantic labels or substitutes for the primary paired baseline.

## 11. Evaluation and interpretation

### 11.1 Primary assessment performance

Use $L_r^{\mathrm A}$ as the primary metric. For each concept, average record
losses within each source/group, average groups, then macro-average concepts
with test support. Apply the same aggregation for validation checkpoint selection.
Report per-concept and per-assessment results with record and group counts;
aggregate machine and held-out human assessments separately.

The primary paired quantity is

$$
\Delta_{\mathrm{evidence}}
=\overline L^{\mathrm A}(\mathrm{style\text{-}only})
-\overline L^{\mathrm A}(\mathrm{style\text{+}evidence}).
$$

Positive values favor evidence supervision. Report each training seed, the mean
paired difference, and a 95% paired group-bootstrap interval using 1,000
resamples. Preserve all labels, records, and paired outputs of a sampled group;
compute each seed's difference and average across seeds within each resample.
Redraw resamples missing a concept required by that reported aggregate. The
interval conditions on trained models and does not include every source of
training variation. Rows, overlapping scopes, and repeated outputs are not
independent samples. State sparse-support limitations for human comparisons.

Do not summarize assessment by total accuracy alone. For each concept, report
three-way NLL and the absent/supporting/prominent confusion matrix using the
three-class argmax, with raw counts and within-reference-class rates. Also report
the following separate distinctions, using the same group aggregation:

| Distinction | Prediction and evaluation population |
| --- | --- |
| Presence | On all resolved records, use $p_P=p_{\mathrm A}(\mathrm{supporting})+p_{\mathrm A}(\mathrm{prominent})$; report binary NLL and balanced accuracy with threshold 0.5 |
| Positive strength | On all reference-positive records, use $p_{\mathrm{strong}}=p_{\mathrm A}(\mathrm{prominent})/p_P$; report supporting/prominent conditional NLL and balanced accuracy with threshold 0.5 |

The positive-strength population includes records the model predicted absent;
do not evaluate only correctly detected positives. Compute these probabilities
from the same three-way head, using stable log-space operations. At record
level, three-way NLL equals presence NLL plus positive-strength NLL when the
reference is positive. The conditional metric diagnoses strength discrimination;
its population and weighting differ from the all-record aggregate. For each
binary task, compute recall separately within each reference class by averaging
records within represented groups, then groups; balanced accuracy is the mean
of the two class recalls. It is unavailable when either class has no support.
NLL remains reportable on a nonempty evaluation population; positive-strength
NLL is unavailable if there are no reference positives. Do not encode ordinal
strength as a regression target with assumed equal spacing.

The initial decision is exploratory. A positive mean $\Delta_{\mathrm{evidence}}$
with the same sign across three paired seeds supports auxiliary supervision for
assessment fitting; an interval entirely above zero strengthens that evidence.
There is no universal practical threshold in nats. Report effect size and
uncertainty, and examine whether any gain comes only from absent/present while
supporting/prominent stagnates or regresses. A repeated class-specific regression
qualifies the result and motivates revising or removing the auxiliary weight,
not an unqualified success claim. Mixed seed signs or a broad interval leave
the benefit unsettled.

Report held-out human assessment results even when their evidence was inherited
from a machine. They test assessment agreement under their recorded confirmation
mode, not independent human selection fit. Machine-only gains remain evidence
about that annotation method; conflicting human results or limited human support
must remain visible rather than being averaged into a more favorable result.

### 11.2 Assessment of concrete relationship distinctions

Prepare a fixed diagnostic sample before inspecting model differences: target
30 held-out resolved cells, up to six per concept, aiming for two absent, two
supporting, and two prominent where available. Include existing contrasts in
complete attack groups, interruptions, recurrence, entering holds, and release
order. Preserve cell/group identity, original assessments, and judgment origin;
report missing strata. Additional failure-driven cases are exploratory additions,
not replacements. These cases support specific interpretations, not a stable
semantic success-rate estimate for every concept.

Use complete sections and their declared review contexts. First identify the
source relationships relevant to the recorded assessment without showing model
identity, predictions, or evidence highlights. Then compare each arm's three
probabilities in randomized order. Record:

1. The concrete relationship and section-level organization at issue, including
   its temporal distribution, repetition, interruptions, or coexistence with
   other organizations. Distinguish source facts from the scoped judgment.
2. Whether the predicted assessment agrees with the recorded judgment and what
   contrast it succeeds or fails to distinguish. For example, compare real judged
   Jack cases with fixed disjoint A/B Trill cases assessed Jack absent despite
   lane returns; examine overlap-only versus judged LN coordination cases.
3. Whether the difference concerns presence, positive strength, or both, and
   whether count/activity, scope/context confusion, or annotation disagreement
   remains a plausible explanation. Retain unresolved interpretations.

For strength, use already judged cases where a local relationship appears in
different section organizations; do not assume similar local notes imply equal
strength. Prefer existing human-assessed contrasts where available. Editing a
chart does not automatically supply the modified chart's label. Prediction
contrasts can support a behavioral distinction, but do not reveal a hidden
state's meaning or establish the model's causal reasoning.

Evidence highlights may be inspected separately after the assessment comparison.
A convincing auxiliary explanation cannot excuse an incorrect assessment or
prove that the primary branch used that explanation. Report human inspection
as human judgment only when a human performed it; machine inspection remains a
machine assessment.

### 11.3 Auxiliary evidence diagnostics

These metrics describe trained selectors and never replace primary assessment
evaluation. `style-only` has no trained selector; do not compare its untrained
head with `style+evidence` or invent a selection-loss gap for the initial pair.

For supplied-reference assessment diagnostics, score $L_r^{\mathrm E}$ using
$y^*$ and reference histories, with concept/group aggregation on eligible records.
Report by absent/supporting/prominent and judgment origin, with eligibility
counts. Add reference object precision/recall/F1, generated and reference count,
selected-row fraction, empty/full outputs, and greedy/beam differences. If exactly
one set is empty, precision/recall/F1 are zero; for two empty sets define them
as one and report their count separately. Alternative useful evidence can
disagree with the sole recorded selection.

For optional output inspection, show the complete source context, highlights,
queried concept, and supplied assessment, while hiding model identity, decoding
method, scores, and original rationale. Record the specific relationship located
by the highlighted objects, whether full context supports, weakens, or defeats
that explanation of the supplied assessment, and a verdict of useful, misleading,
no relevant relationship located, or unresolved. An absent explanation must
locate evidence relevant to absence, not merely fail to show a positive pattern.
Recognizing a style elsewhere in the chart does not assess the highlights.

Do not require every selected object to be indispensable, both relationship
endpoints to be selected, or the selection to be sufficient without context.
Useful failure categories include manufactured recurrence/alternation after
ignoring intervening attacks, missing hold/release relationships, wrong-scope
inference, quantity-only agreement, selection-policy imitation, and decorative
highlights unconnected to the explanation. Multiple selected runs are allowed;
attention weights and the $U/V$ score decomposition do not identify reasoning.

If generation quality is examined, freeze a subset of Section 11.2's cells and
output seeds before viewing differences. Alongside greedy and beam 8, generate
four ancestral witnesses per cell and trained-selector seed with Section 9.1's
unmodified temperature-1 procedure. Use sampling seed `0` and reproducible streams
keyed by cell, supplied assessment, training seed, and sample index; record the
derivation and share it across any compared selector variants. Keep all draws,
including duplicates and empty/full outputs, without choosing a best sample.

Report per-cell count distributions, distinct selections, varying source
locations, and relationship/failure judgments on a predeclared inspection subset.
Preserve all draws in quantitative summaries and state the inspection denominator;
samples of one cell are not independent cases. A high-probability point need
not be typical: 100 independent binary choices with selection probability 0.1
have expected count 10 but an all-zero MAP vector. Poor greedy/beam highlights
with useful samples can reflect output choice or approximate search. Good
teacher-forced fit with poor samples instead exposes difficulty starting or
maintaining selections under generated histories.

Keep $y^*$-conditioned explanations separate from outputs conditioned on the
predicted $\widehat y$. In the latter, report assessment correctness separately:
a selector can produce plausible highlights for an incorrect assessment.
Neither inspection uses generated evidence as an input to the primary classifier.

#### 11.3.1 Decision-level gains for a history follow-up

Only in the trained-selector comparison of Section 10.2, write
$\Xi=(X,\ell,y^*)$ and save both arms' log probabilities under the same preceding
reference masks. Here each probability includes its arm's own chart encoder.
Define

$$
\begin{aligned}
\delta_{r,d}
&=\log p_{\mathrm{history}}(z_d^*\mid z_{<d}^*,\Xi)
-\log p_{\mathrm{context}}(z_d^*\mid\Xi),\\
L_r^{\mathrm E}(\mathrm{context})-L_r^{\mathrm E}(\mathrm{history})
&=\frac{1}{|J_r|}\sum_{d\in J_r}\delta_{r,d}.
\end{aligned}
$$

Keep target masks, record/group IDs, source row time/phase, and these strata as
evaluation fields rather than model inputs or semantic labels:

| Stratum | Operational definition |
| --- | --- |
| Skip / nonempty | Whether the target mask is `0000`; exclude forced-zero decisions from $J_r$ |
| Run start / continuation | A nonempty attack-row target starts a run when the preceding original in-scope attack row is unselected or absent; otherwise it continues. Release-only rows neither start nor end runs; entering-LN boundary choices are separate |
| Same-lane return | A selected head has an earlier selected in-scope attack in its lane; retain the latest such attack, time gap, and intervening complete attack groups |
| Same-hand role switch | Compare current selected roles with that hand's latest earlier selected attack-row mask; flag a newly selected role and retain both masks |
| Cross-hand change | Compare selected-hand sets at this and the latest earlier nonempty attack-row target, retaining left/right/both membership |
| Selected-active LN | Previously selected holds active before the current row, including entering LNs, with endpoints and current releases/actions on other lanes |

Use explicit no-predecessor states. Return/role/hand comparisons apply to nonempty
attack targets; selected-active LN facts also apply at skips. Preserve unselected
interruptions and complete source groups. A run is an attack-row statistic, not
a semantic episode, and a return flag does not imply Jack.

Report stratum decision/record/group counts, descriptive mean gain, and its
contribution to the auxiliary loss gap. For contributions, zero other decisions
in the sum above, retain the original $|J_r|$, and use the auxiliary aggregation
on all eligible records. Empty-stratum means are unavailable. Only disjoint
exhaustive partitions such as skip/nonempty sum to the full auxiliary gap;
overlapping relation flags must not be added. Retain concept, assessment, and
seed breakdowns. These gains do not decompose $\Delta_{\mathrm{evidence}}$.

Skip/run-continuation gains motivate count and persistence inspection; return,
hand-change, and hold-interaction gains motivate inspection of those relationships.
Neither proves benefit to the shared representation. Compare primary assessment
results before attributing value to selection-history capacity.

### 11.4 What results permit

| Observation | Supported interpretation | Next decision |
| --- | --- | --- |
| Better assessment NLL, preserved class distinctions, and better judged relationship contrasts | Evidence supervision helps section assessment in the tested setting | Retain the auxiliary weight and identify the supported concepts/contrasts |
| Better assessment NLL only for presence | Evidence helps absent/present discrimination; positive strength remains unresolved or regresses | Inspect strength errors and qualify the benefit |
| Better evidence fitting or highlights without assessment gain | The auxiliary branch learned selection behavior without demonstrated primary-task value | Prefer `style-only` for assessment; retain generation only for a separately useful purpose |
| Better assessment with imperfect reference-mask agreement | Evidence can help training without reproducing the agent's exact choice | Judge the auxiliary term by assessment outcomes, not mask fidelity alone |
| Machine-assessment gains with conflicting or inconclusive human results | Method-specific fitting remains an explanation | Report both origins and inspect disagreements before broader claims |
| A stronger selector improves its NLL but not assessment | Extra capacity may be solving the auxiliary task within the decoder | Do not retain complexity solely for lower selection loss |
| No stable assessment gain or poor results in both arms | This supervision/representation/optimization combination is not established | Inspect class support and concrete failures; revise the weight or supervision before adding model families |
| Benefit limited to well-supported concepts | Evidence is concept-specific under uneven support | Do not infer Tech, LN, partial-chord, or entering-LN success from aggregate scores |

No detected auxiliary benefit does not show that evidence lacks useful
relationships. Better classification does not by itself establish understanding
of style or sufficiency for continuation responses. State the narrowest claim
supported by the held-out assessments and inspected real cases.

## 12. Implementation handoff and bounded execution

Implement this study in an isolated research package, for example
`src/ensomi_model/research/scoped_style_modeling/`, with nearby focused tests.
These paths name proposed ownership; no implementation is implied. Do not build
the model on legacy mapper/tokenizer/training behavior. Follow repository config
guidance where applicable without treating retained mapper presets as V3 designs.

Keep the interfaces small:

| Component | Input and output contract |
| --- | --- |
| Dataset adapter | Pinned resolved assessments and optional evidence to immutable record IDs, scopes, review contexts, separate targets/availability, and split groups |
| Replay and relation preparation | Complete visible source facts to lane states, encoder/decoder timelines, candidate masks, relation descriptors, and object-decision mapping |
| Shared encoder | Chart facts and relation graph to two contextual hand vectors per encoded position; no assessment or selection input |
| Assessment readout/head | Complete section sequence and queried concept to absent/supporting/prominent probabilities; no auxiliary states or targets |
| Auxiliary row decoder | Chart context, concept, supplied assessment, and its own history to valid-mask probabilities and simultaneous next histories |
| Trainer | Primary assessment loss plus weighted, availability-masked evidence loss; teacher forcing confined to selector; checkpoint selection by assessment validation |
| Assessment evaluator | Selector-free inference, grouped assessment metrics, per-class distinctions, human comparison, and complete-context relationship cases |
| Optional evidence evaluator | Declared assessment conditioning, sequence NLL/agreement, greedy/beam and ancestral outputs, count costs, and inspection records; decision gains only for trained-selector comparisons |

Store replay facts separately from selection targets. In particular,
`context_note_refs` is a delivery complement of witnesses, not a separately
selected negative input channel. Recover the complete source context and remove
the witness/context distinction before encoder feature construction.

### 12.1 Small-model starting configuration

These values define a starting preset, not tuned performance claims:

| Setting | Initial value |
| --- | --- |
| Lane embedding | 16 dimensions |
| Shared hand BiGRU | One layer, 32 hidden dimensions per direction; 64 output dimensions per hand |
| Relation block | One block, four attention heads, 64 model dimensions, 128 feed-forward dimensions |
| Assessment row projection | Symmetrized shared pair MLP, one 64-unit hidden layer and 64 output dimensions |
| Section readout BiGRU | One layer, 32 hidden dimensions per direction; terminal states plus in-scope mean and duration/empty-row features |
| Assessment head | One 64-unit hidden layer and three logits per query |
| Branch-owned concept embeddings | 16 dimensions each, separate for assessment and evidence |
| Auxiliary assessment / hand-mask embedding | 8 / 8 dimensions |
| Shared selection GRU | 64 hidden dimensions per hand |
| $U$ and symmetric $G$ readouts | One 64-unit hidden layer each |
| Dropout | 0.1 in encoder/readout; disabled in evaluation |
| Optimizer | AdamW, learning rate $3\times10^{-4}$, weight decay $10^{-4}$, gradient norm cap 1 |
| Batch and epoch | 16 sampled records; one epoch has the training cohort's record count of draws |
| Auxiliary weight | $\beta=0$ versus $0.1$ for the initial pair |
| Training bound | At most 30 epochs, early stopping after five epochs without validation assessment macro-NLL improvement |
| Training seeds | Pilot seed 17; paired repeats 29 and 43 |
| Checkpoint selection | Lowest validation assessment macro-NLL, using the same rule for both arms |
| Assessment inference | Shared encoder plus primary readout/head; auxiliary branch removed |
| Optional evidence inference | Greedy/beam 8 at $\lambda=0$; four temperature-1 ancestral samples per fixed diagnostic cell/selector seed as in Section 11.3 |

First verify one batch and a small overfit slice, then time the paired pilot.
Use the explicit `mps` extra on Apple Silicon or `cuda` on NVIDIA Linux, and
record the actual dependency lock and device. An initial run allocation is
20 minutes per arm/seed and two hours total training for six runs, excluding
dataset recovery and human inspection. If that allocation truncates training,
report the truncation and reconsider the common run budget before interpreting
unequally converged models. Do not introduce a large hyperparameter sweep.

Mirror transformations are useful consistency checks and an optional shared
augmentation for generic encoders. Do not time-stretch these 1× annotations as
though speed changes preserve all five assessments.

### 12.2 Required correctness evidence

Test the contracts that can change the experiment's meaning:

1. Object-to-mask-to-object identity, including entering/ending boundary LNs,
   an actual row at $a$, a head at $b$, and LN tails outside context.
2. Exact occupation and complete groups remain identical for different selection
   masks. A source A/B alternation cannot become a different source graph after
   selecting only A.
3. Relations and strict-neighbor features obey context limits, preserve original
   row adjacency and endpoint identity, and combine multiple edge roles correctly.
4. Valid-mask probabilities normalize; impossible masks have no support;
   release-only decisions add zero auxiliary NLL; padding does not update states
   or enter section aggregation. Explicit empty evidence and missing evidence
   follow their distinct loss rules without deleting assessment examples.
5. At fixed weights, changing true assessments, reference masks, teacher-forced
   histories, or generated selections cannot change chart vectors or assessment
   logits. Removing the selector preserves predictions. Gradient paths connect
   each loss to its own branch and the shared encoder, not the other branch;
   the zero-weight model reproduces the assessment-only update under paired RNG.
6. The primary readout includes all in-scope event rows, boundary occupation,
   trailing time, and empty-section handling. External context rows and a head
   at $b$ do not enter its in-scope pooling or event counts. Changing evidence
   cannot change this membership.
7. The three probabilities normalize within each queried concept; absent evidence
   is conditioned on absent, and unresolved/unreviewed are excluded from both
   supervised class and conditional evidence targets. Presence/strength NLL
   decomposition and grouped paired assessment aggregation match their definitions.
8. Dataset grouping has no exact source/cell leakage and the encoder has no
   target-bearing input channel. Training-only transformations fit training data;
   both arms use the same assessment cohort, split, and minibatch stream.
9. Mirror transformations preserve assessment probabilities, mapped auxiliary
   probabilities, and sequence scores, including the section readout.
10. Auxiliary hands update simultaneously from their prior states, with fresh
    state for each record or generated trajectory. Source replay remains shared
    and immutable.

When implementing optional decoding or a selector comparison, also require:

- Tiny-chart enumeration agrees with sequence scoring and sufficiently wide
  beam search under the original log-probability/count objective. The context-only
  variant is invariant to selection history and its rowwise maximum is exact.
- Ancestral sampling uses original joint probabilities and sampled history,
  reproduces fixed-stream draws, retains empty/duplicate outputs, and advances
  forced rows. Tiny enumerated distributions agree with sampled frequencies
  within a declared Monte Carlo tolerance. Beam/sample histories are isolated.
- Decision-gain sums recover the auxiliary loss difference, not the primary
  assessment gap. Skip/nonempty partitions reconcile; run boundaries use original
  attack rows, and overlapping relation flags are not double-counted.

These tests validate implementation semantics, not pattern accuracy. One-batch
gradient checks and tiny-slice overfitting diagnose training viability; neither
is a held-out result.

### 12.3 Run record and deliverables

Each actual comparison records its recoverable source revision, document/config
revision, dataset and split manifest, arm, auxiliary weight, seed, actual command,
device, training/inference parameter counts, runtime bound, and output location
before training. Record the command once the research entrypoint has been
implemented and its interface verified.
Use a fresh artifact directory per run. Resume only a matching configuration and
dataset/split identity, and keep checkpoint-selection and budget usage visible.
Record decoding and sampling runtime separately from the training allocation.

The initial handoff consists of the adapter, replay/relations, shared encoder,
section readout/head, auxiliary selector, paired training, correctness evidence,
and a compact assessment report. Include class/group support, evidence
availability, weight and checkpoint choices, three-way/presence/strength metrics,
held-out human comparison, and concrete relationship cases. Report auxiliary
likelihood/agreement separately. If optional evidence generation or history
comparisons run, include their conditioning mode, quantity curves, all draws and
sampling manifest, inspections, and decision-gain strata. State the distinct
cells/groups behind rare-case and semantic claims, failures, and deviations.
Raw artifacts remain outside tracked product sources.

Numerical instability, incorrect source identity, or representation-contract
failures stop the affected run. Missing exposure metadata, absent independent
human selection gold, or an unavailable optional highlight inspection does not
block the assessment comparison. Missing human assessments or source-relationship
inspection limits the resulting claim and must be reported.

A formal accepted Experiment Card and executed Result Logs, when created, belong
to an owning Agent Note under the repository research workflow. This document
provides the self-contained model and evaluation specification; it does not
assert card acceptance, completed implementation, or authorization for a run.

## 13. Connection to Ensomi responses

The useful transfer is from verified arrangement distinctions to mapper-facing
questions about legal continuations. Examples include whether a continuation
preserves repeated-column organization or redistributes it, extends fixed-group
alternation or changes group membership, and introduces independent actions or
releases against existing LN occupation.

Such questions still need scope, horizon, positive/negative/equivalent examples,
and a declared response comparison. This encoder sees later chart context in
its declared review window. It is not a causal prefix state; later source rows
would need to be provisional continuations when used during generation.

The section representation $r_\ell$, selection states $q$, relation counts,
assessment probabilities, and witness size are not gameplay demand coordinates.
Better recognition of a scoped concept also does not establish that a state
preserves the full continuation-response function. The target response
specification precedes that adequacy claim.

## 14. Closest research analogues

- [Carton, Kanoria, and Tan (2022), What to Learn, and How](https://aclanthology.org/2022.findings-acl.86/)
  studies learning label predictions from rationale supervision and finds that
  maximizing rationale fit need not maximize prediction accuracy. The relevant
  comparison is the utility of auxiliary evidence for the primary task. Here,
  agent-selected source objects locate chart relationships and need not provide
  an independently sufficient input.
- [Lei, Barzilay, and Jaakkola (2016), Rationalizing Neural Predictions](https://aclanthology.org/D16-1011/)
  compares independent and recurrent history-dependent binary selection from
  bidirectional context. The direct analogue is the selector's factorization and
  memory. Its latent-rationale training and isolated-rationale sufficiency objective
  differ from this weak auxiliary selection task with a full-chart assessment path.
- [Shaw, Uszkoreit, and Vaswani (2018), Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/)
  motivates transmitting declared relationships through attention. The proposed
  chart graph instantiates domain-specific action relations rather than relying
  solely on sequence distance.
- [DeYoung et al. (2020), ERASER](https://aclanthology.org/2020.acl-main.408/)
  separates reference-rationale agreement from faithfulness. Its deletion-based
  measures cannot be imported as unchanged style judgments on an edited chart.

The provisional contribution is an adaptation to source-linked 4K action
relationships and evidence about whether agent explanations improve scoped style
assessment. A shared encoder and auxiliary GRU are an experimental mechanism,
not a claim of a new general rationale-learning method.
