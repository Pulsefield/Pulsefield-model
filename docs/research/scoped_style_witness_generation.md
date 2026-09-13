# Scoped style witness generation

This document specifies an exploratory study of label-conditioned selection on
existing 4K charts. It defines the observation, source-action representation,
candidate model, discriminating comparisons, and implementation handoff. The
model has no reported training results. Its architecture and numerical defaults
are research choices, not Pulsefield V3 requirements.

The [generation contract](../formulation/notation.md) owns legal timed rows and
committed history. The [gameplay formulation](../formulation/gameplay-state.md)
owns style observations, continuation responses, and demand semantics. This study
selects source objects; it does not generate a new chart or validate a demand state.

## 1. Question and scope

The primary question is:

> Given a complete declared chart context and a concept known to be present in
> a section, does explicit selection history improve held-out witness fitting
> under the available data and model capacity, and do those gains accompany more
> reliable highlighting of specific source relationships?

A second comparison asks whether explicit hand coordinates and source-action
relation channels improve this task over an encoder receiving the same source
facts without those channels. Separate these interventions so their effects can
be interpreted.

The study has three distinct possible outcomes:

| Outcome | Evidence required |
| --- | --- |
| Learn the recorded selection distribution | Held-out conditional selection likelihood and reference agreement |
| Generate useful witnesses for a scoped concept | Inspection of generated highlights in the complete source context |
| Learn a representation useful for style recognition | A separate presence/ordinal-strength prediction comparison |

Success at one level does not establish the next. In particular, a selection
model can learn annotation location and quantity preferences without improving
the relationships it highlights or style recognition. The first two levels are
the initial study; recognition is a bounded extension after their results are
available. The comparison tests the usefulness of explicit selection dependence,
not whether witnesses or gameplay have structure in the abstract.

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
| Witness | Source objects selected to explain an assessment in context | Selected members of an alternating episode |
| Learned representation | A computational hypothesis about useful information | A hand embedding or selection-memory state |

A witness is an object selection attached to the complete arrangement. It need
not be exhaustive, minimal, contiguous, or sufficient when shown in isolation.
Unselected notes remain part of the explanation's context and can supply
counterevidence. Replacing a highlight mask changes what is emphasized; deleting
notes changes the chart and is a different intervention.

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
coordination. The [Foundation](https://github.com/Pulsefield/beatmap-lens/blob/647009ab60ed69d98190712a6ab025807cca07b8/annotation/foundations/15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97.json)
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
[labeler role](https://github.com/Pulsefield/beatmap-lens/blob/647009ab60ed69d98190712a6ab025807cca07b8/annotation/methods/astra-1000-20260912/labeler.md)
requests simultaneous judgment and selection, including for absent assessments.
Those objects explain a negative judgment; they are not positive-concept targets.

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

The initial training comparison uses the selected machine method's positive
records. This gives both arms one annotation method and the same cohort. Include
all five concepts, report each separately, and emphasize the Jack/Trill contrast
when interpreting recurrence behavior. Tech and LN results have smaller support.

Human observations provide a supplementary assessment layer, not an automatic
note-selection gold set. Human confirmation of a label can retain machine notes
and rationale. Explicit confidence concerns the assessment. Do not require a new
selection review, filter by confidence, or repair annotations as a prerequisite
for this exploratory study.

Use the following minimal reproducibility and separation rules:

1. Pin the dataset revision and preserve source hash, cell ID, assessment,
   vocabulary, rate, and judgment origin. Verify source bytes before resolving
   source-line identities; a changed chart is a different input.
2. Assign an exact source, all its scopes, and all its labels to one split.
   Group known duplicate chart versions and known shared-song/audio identities
   when that information is already available. Do not require new audio
   acquisition or a comprehensive similarity investigation to start.
3. Keep reference membership out of the chart encoder and out of each decoder
   decision until the declared selection-history update. Teacher forcing supplies
   preceding target selections only. Exclude rationale, human comments, audit
   text, assessment strength, source-line numbers, chart titles, and provenance
   fields from model features. Source-line numbers are output identities only.
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
comparison intended to evaluate that tag.

Apply the same grouping to supplementary human records. Human records on
training-group charts can be used for development inspection but do not become
held-out evidence. A subsequent mixed-source training experiment must use human
precedence at exact cells and state its sampling weights; it is a distinct cohort
comparison. The main run need not wait for that extension.

Describe the initial result as source-grouped exploratory evaluation. Broader
claims about unseen songs, independently collected human witnesses, or annotation
method generalization require the corresponding additional evidence. The dataset
split named `full` supplies no benchmark split by itself.

## 4. Formal observation and visible context

Write one observation as

$$
r=(C,S,Q,\ell,y,E^*,\nu),
\qquad S=[a,b)\subseteq Q=[u,v),
$$

where $C$ is the exact source chart, $S$ the judged section, $Q$ its recorded
review context, $\ell$ the concept, $y$ the assessment, $E^*$ the witness objects,
and $\nu$ the identity and origin metadata. The selector receives the chart
representation of $Q$, the scope markers for $S$, and $\ell$.

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

Write $C_Q$ for the visible representation just defined. The primary selector is
conditioned on a known positive:

$$
p_\theta(Z\mid C_Q,S,\ell,\mathrm{present}).
$$

Supporting/prominent are pooled for training this selector and retained for
stratified evaluation. They are not input features. Negative and unresolved
records are excluded from this loss, not converted to empty positive witnesses.

Two extensions have different meanings:

$$
\begin{aligned}
p_\theta(Z\mid C_Q,S,\ell,y)
&\quad\text{explains a supplied assessment},\\
p_\eta(y\mid C_Q,S,\ell)\,
p_\theta(Z\mid C_Q,S,\ell,y)
&\quad\text{predicts an assessment and explains it}.
\end{aligned}
$$

Neither is part of the initial selector comparison. A normalized distribution
over selections does not supply concept presence: its mass sums to one for
every concept, and its maximum probability measures concentration as well as fit.

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
plus boundary markers needed for $Q$ and $S$. At each source timestamp, encode
all four lanes together. A tap has no additional LN-close action. Synthetic
boundary markers have a phase flag and do not create source actions.

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

Every candidate object has exactly one selection decision. Its target bit is
one exactly when it appears in $E^*$. Verify $E^*\subseteq\mathcal U_r$ rather
than silently dropping unmatched references. Object selection round-trips to
source IDs, including entering LNs.

$$
Z=(z_0,\ldots,z_D),\qquad
E(Z)=\{n_i:\text{its decision bit is one}\},\qquad
K(Z)=|E(Z)|=\sum_{d,j}z_{d,j}.
$$

The timeline is known, but $K$ is free. No oracle count or EOS is supplied.
`0000` is a skip selection, not an empty row materialized in a generated chart.
The distribution permits an empty output; report its frequency on known-positive
sections rather than silently imposing a nonempty or fixed-size constraint.

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

## 8. Candidate encoder and decoder

### 8.1 Shared hand encoder and relation attention

For each hand, concatenate ordered lane embeddings and row-time features, then
apply the same BiGRU parameters to the two separate sequences:

$$
H^L=F_\phi(X^L),\qquad H^R=F_\phi(X^R).
$$

Apply one relation-aware attention block over the hand-row nodes. For a query
node $i$ and a visible neighbor $j$ with relation descriptor $R_{ij}$, one head is

$$
\begin{aligned}
e_{ij}&=\frac{(W_QH_i)^\top(W_KH_j)}{\sqrt{d_k}}+b_\rho(R_{ij}),\\
\alpha_{ij}&=\operatorname{softmax}_{j\in\mathcal N(i)}e_{ij},\\
\widetilde H_i&=\sum_{j\in\mathcal N(i)}\alpha_{ij}
\left(W_VH_j+r_\rho(R_{ij})\right).
\end{aligned}
$$

Use residual connections, layer normalization, and a small feed-forward block.
The resulting $c_d^L,c_d^R$ are chart-context representations at each decoder
position. They do not depend on any witness mask. Relation values are included
as well as relation biases so relationship type can affect what is transmitted.

### 8.2 Exact selection history

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

### 8.3 Joint row distribution

Let $q_{d-1}^L,q_{d-1}^R$ be selection-memory states, initialized to zero before
the boundary decision. Build $v_d^h$ from chart context, both prior memory states,
exact selection history, and concept embedding $e_\ell$. All hand-indexed fields
use self/other order, retaining outer/inner order within each hand.
For a candidate mask $b=(b^L,b^R)$ in canonical hand-role coordinates, use

$$
\begin{aligned}
s_d(b)&=U_\theta(v_d^L,b^L)+U_\theta(v_d^R,b^R)
+V_\theta(v_d^L,v_d^R,b^L,b^R),\\
p_\theta(z_d=b\mid z_{<d},C_Q,S,\ell)
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
\operatorname{Emb}(z_d^{\bar h}),e_\ell,\Delta t_d],q_{d-1}^h\right).
$$

The GRU parameters are shared. Do not update one hand from the other hand's
already-updated state. Update exact selection history after reading its prior
values for the decision. Forced-zero release rows still advance the learned
state and update the selected-active-LN facts through the source release.

The sequence distribution is

$$
p_\theta(Z\mid C_Q,S,\ell,\mathrm{present})
=\prod_{d=0}^{D}p_\theta(z_d\mid z_{<d},C_Q,S,\ell).
$$

This gives generation and scoring of complete selections over the same support.
It does not assign each source note a history-independent semantic importance.
Autoregression constrains selection history only: the encoder sees the complete
declared review context, including later source rows. This is retrospective
selection, not a causal restriction on chart observation.

### 8.4 Mirror behavior

For the mirror $\mu$ that exchanges hands while preserving outer/inner roles,
require, in deterministic evaluation mode,

$$
p_\theta(\mu Z\mid\mu C_Q,S,\ell)
=p_\theta(Z\mid C_Q,S,\ell).
$$

Check the graph transformation, per-step probabilities with mirrored histories,
and full-sequence scores. Shared encoders alone are insufficient. Dropout is
disabled for this check. Tied maxima can produce different single decoded outputs
under an asymmetric tie-break rule; compare probabilities and mapped tied
candidates before calling that a distributional symmetry failure. Arbitrary lane
permutations and playback-rate changes are not label-preserving augmentations.

## 9. Training and decoding

Let $J_r=\{d:|\mathcal B_d|>1\}$ be the nontrivial selection decisions. For a
positive record, define

$$
L_r(\theta)=-\frac{1}{|J_r|}\sum_{d\in J_r}
\log p_\theta(z_d^*\mid z_{<d}^*,C_Q,S,\ell).
$$

Use teacher forcing with the complete target row mask. This is length-normalized
supervised conditional likelihood: records with many decision rows do not
automatically dominate. It differs from an unweighted corpus sum of sequence
NLL. Also record unnormalized sequence NLL so both quantities remain available.
Positive records with no eligible decision are a data-contract error, not a
zero-loss training example.

Sample a concept uniformly, then a split group uniformly among training groups
with that concept, then one of its eligible records uniformly. Use the same sampled
record order and batching plan for paired arms. This weighting targets a balanced
exploratory comparison, not the natural prevalence of concepts in whole charts.

Use ordinary supervised gradients and decoupled AdamW weight decay. There is no
sampling-gradient estimator and no RL objective. A penalty on $K(E^*)$ would be
constant in the parameters; a penalty on expected generated count would involve
the model's own history distribution and is outside this initial objective.

Decode with

$$
\widehat Z_\lambda\approx\arg\max_Z
\left[\log p_\theta(Z\mid C_Q,S,\ell)-\lambda K(Z)\right].
$$

Beam expansion adds the original model log probability and subtracts
$\lambda\operatorname{popcount}(b)$. Do not subtract the cost from logits and
then renormalize: history-dependent normalizers generally change the sequence
objective. Forced decisions retain their zero log probability and zero count.

Use $\lambda=0$ for the primary comparison. An initial sensitivity grid is
$\{0,0.05,0.1,0.2\}$ nats per object, with beam width 8 and a greedy result for
reference. Report the grid rather than selecting a favorable test-set point.
Equal $\lambda$ values need not yield equal counts across models; compare quality
at overlapping achieved count ranges as an additional analysis. Selections at
different costs need not be nested. Count is an output preference, not style
strength, relationship completeness, or a common information cost for taps/LNs.
It is a post-training decoding preference; this loss does not teach a value for
concise explanations.

For `relations-context`, probabilities and valid masks do not depend on selected
history, so maximizing each row's log probability minus its count cost gives
the exact sequence optimum for every $\lambda$. Greedy therefore attains it;
beam search cannot improve its score. `relations-history` generally needs
approximate search. Report this asymmetry when interpreting decoded differences.

Also generate ancestral samples from the original distribution: at $\lambda=0$
and temperature 1, draw each joint mask from its normalized valid-mask
probabilities and update history with that sampled mask. Use neither reference
history nor top-k/top-p truncation, rejection of empty outputs, or count-based
reranking. Forced decisions still advance state. Section 11.3 fixes the sample
budget and diagnostics; sampling does not change training or checkpoint selection.

## 10. Discriminating comparisons

### 10.1 Selection history: the first comparison

| Arm | Chart encoder | Decoder history |
| --- | --- | --- |
| `relations-context` | Shared hand BiGRU plus source-action relation block | Chart/label recurrence only |
| `relations-history` | Identical encoder architecture and input facts | Learned and exact selection history |

For `relations-context`, retain the same GRU state sizes, updates, and readout
dimensions, but replace selected-mask embeddings and exact selection-history
fields with dedicated constant null inputs. Its hidden states can still process
the chart and label; they cannot depend on previous selected masks. Consequently
its row probabilities are conditionally independent given chart and label even
though it has a recurrent computation. Do not remove an entire processing layer
and attribute the resulting difference solely to selection history.

Writing $X=(C_Q,S,\ell,\mathrm{present})$, the distinction is

$$
\begin{aligned}
p_{\mathrm{context}}(Z\mid X)
&=\prod_d p_{\mathrm{context}}(z_d\mid X),\\
p_{\mathrm{history}}(Z\mid X)
&=\prod_d p_{\mathrm{history}}(z_d\mid z_{<d},X).
\end{aligned}
$$

For fixed-length outputs, unlimited capacity, and population-optimal unnormalized
expected NLL, the entropy chain rule gives

$$
\mathcal L_{\mathrm{ind}}^*-\mathcal L_{\mathrm{AR}}^*
=\sum_d H(Z_d\mid X)-H(Z\mid X)
=\sum_d I(Z_d;Z_{<d}\mid X).
$$

This ideal identity describes uncertainty removed by preceding selections. It
is not an estimator for the measured loss gap with finite models, optimization
error, finite data, and the record normalization in Section 9.

Selection dependence can reflect source relationships or annotation strategy.
If a section has two equally useful episodes and the annotator chooses one,
history can keep that choice consistent while a context-only model mixes their
marginals. That is useful organization of an explanation, but it need not reveal
a new gameplay distinction. Conversely, if the witness is a deterministic
function of $X$, a sufficiently capable context-only model can concentrate on
that whole witness. An absence of history gain is therefore compatible with
structured witnesses. Decision-level gains and generated-highlight inspection
distinguish these explanations more directly than aggregate likelihood alone.

Train both arms independently with paired initialization seeds, minibatch order,
optimizer settings, and model-selection budget. Verify that changing a preceding
selection cannot change `relations-context` probabilities. History dependence
in the other arm is permitted, not guaranteed by merely having the inputs.

This intervention tests the learned and exact selection-history bundle. If it
helps, an exact-history-only or learned-history-only comparison can determine
which part contributes. Do not claim the first comparison isolates GRU memory
from those exact history features.

### 10.2 Source-action representation: a subsequent comparison

Hold the decoder/history choice fixed and compare the structured encoder with
a generic four-lane row BiGRU and ordinary context interaction. Give both the
same exact lane facts, context extent, training cohort, and comparable capacity.
The generic encoder retains four ordered lanes but has no hand parameter sharing
or typed relation edges. State its attention support and parameter count.

That comparison tests the structure bundle. To attribute a benefit specifically
to relations, keep shared hand encoding and the sparse neighbor topology while
removing relation descriptors; separately compare ordinary temporal/full-context
interaction if testing the topology. The generic baseline need not be exactly
mirror-equivariant; apply any mirror augmentation equally to compared arms and
report its measured symmetry behavior. To attribute a benefit to hand sharing,
change the sharing while holding relation processing fixed. These are follow-up
interventions, not mandatory arms of the first run.

Exact occupation is computable from the chart. Removing it tests the value of
providing that computation explicitly, not the addition of a new independent
measurement or proof of a physiological mechanism.

### 10.3 Diagnostics with distinct purposes

- On held-out scopes with the same review context and multiple positive concepts,
  query each of those concepts, compare outputs, and score each recorded witness
  under all those concepts. Different labels may legitimately share witnesses.
  An absent concept is outside this known-positive selector's task; failure to
  reject it is not a presence-detection error. A trained label-null version is a
  follow-up for attributing a benefit to conditioning, not a third initial arm.
- Inspect partial-chord cases for the joint mask head. A four-sigmoid comparison
  is optional; scarce partial-row targets limit a global performance claim.
- A previous-mask-only finite-state decoder can test whether short output history
  suffices. A CRF is not required for the initial comparison. An arbitrary
  selection-dependent GRU hidden state cannot be merged by previous mask for
  ordinary forward-backward or Viterbi inference.
- Full-selection, empty-selection, and chart-only position/activity baselines
  provide count and agreement reference points. They are not semantic ground
  truth. Fixed-size random candidate ranking is not the primary task.

## 11. Evaluation and interpretation

### 11.1 Likelihood and agreement

Use the record loss $L_r$ for the primary likelihood measure. For each concept,
average records within a source/group, then average groups, then macro-average
concepts with test support. Report the same aggregation separately by concept,
judgment origin, and supporting/prominent assessment. Preserve sample counts.

The primary paired quantity is

$$
\Delta_{\mathrm{history}}=
\overline L(\mathrm{relations\text{-}context})
-\overline L(\mathrm{relations\text{-}history}).
$$

Positive values favor selection history. Report each training seed, the mean
paired difference, and a 95% paired group-bootstrap interval using 1,000
resamples. For each resample, compute each seed's paired difference and then
average the differences across seeds. Preserve all labels, records, and model
outputs of a sampled group; if a resample lacks a required label, redraw it.
State that this interval
conditions on the trained models and does not include every source of training
variation. Do not treat individual rows or overlapping scopes as independent.

Also report reference object precision/recall/F1, predicted and reference count,
selected-row fraction, empty/full outputs, and greedy/beam differences. With
nonempty positive references, an empty prediction has F1 zero. Name these as
reference-agreement measures: an alternative valid witness can disagree with
the sole recorded selection.

The initial decision is exploratory. A positive mean difference with the same
sign across three paired seeds supports retaining history for selection fitting;
a group interval entirely above zero strengthens that conclusion. There is no
predefined universal practical effect threshold in nats. Report effect size and
uncertainty rather than converting this criterion into a semantic quality claim.
Mixed seed signs or a broad interval leave the benefit unsettled.

#### 11.1.1 Decision-level history gains

For each held-out reference and paired training seed, save the target mask and
both arms' log probability at each nontrivial decision under the same preceding
reference masks. Define

$$
\begin{aligned}
\delta_{r,d}
&=\log p_{\mathrm{history}}(z_d^*\mid z_{<d}^*,X)
-\log p_{\mathrm{context}}(z_d^*\mid X),\\
L_r(\mathrm{context})-L_r(\mathrm{history})
&=\frac{1}{|J_r|}\sum_{d\in J_r}\delta_{r,d}.
\end{aligned}
$$

Retain record/group identity, original row time and phase, and the following
diagnostic strata. These are evaluation fields, not new model inputs or labels.

| Stratum | Operational definition |
| --- | --- |
| Skip / nonempty | Whether the complete target mask is `0000`; forced-zero decisions are excluded from $J_r$ |
| Run start / continuation | A nonempty attack-row target starts a run when the preceding original in-scope attack row is unselected or absent; otherwise it continues the run. Release-only rows neither start nor end runs; entering-LN boundary choices are separate |
| Same-lane return | A selected head has an earlier selected in-scope attack in its lane; retain the latest such attack, elapsed time, and all intervening original attack groups |
| Same-hand role switch | For a hand selected now, compare selected outer/inner role masks with that hand's latest earlier selected attack row; flag use of a previously unselected role and retain both masks |
| Cross-hand change | Compare the selected-hand sets of this and the latest earlier nonempty attack-row target, retaining left/right/both membership |
| Selected-active LN | Record which previously selected holds are active immediately before the current row, their original endpoints, and any current releases or actions on other lanes; include boundary-selected entering LNs |

Use explicit no-predecessor states. Return/role/hand comparisons apply to
nonempty attack targets; selected-active LN facts also apply at skips. They can
overlap and do not supply semantic Jack, Trill, or coordination labels. Preserve
complete source groups and unselected interruptions when relating selected
objects. A run is only an attack-row statistic, not a semantic episode.

For each stratum, report decision, record, and group counts, the descriptive
arithmetic mean gain over its decisions, and its contribution to the primary gap.
Compute the contribution by
zeroing other decisions in the sum above, retaining the original $|J_r|$, then
using the primary record/group/concept aggregation. Mark an empty stratum's mean
unavailable. Contributions sum to $\Delta_{\mathrm{history}}$ only for an
exhaustive disjoint partition, such as skip/nonempty; overlapping relation flags
must not be added together. Preserve seed and concept breakdowns without treating
decisions as independent statistical samples.

Gains concentrated in skips or run continuation motivate checking count and
highlight persistence. Gains on returns, hand changes, or hold interactions
motivate inspecting those source relationships. Neither location alone establishes
semantic improvement; these diagnostics localize fitting gains without changing
the loss or requiring new annotation.

### 11.2 Generated-witness inspection

Prepare a compact view with the complete section and its review context,
highlights, original attack groups, full LN endpoints, and the queried concept.
Hide model identity, decoding method, model scores, and original rationale;
randomize display order. Source facts may be displayed; do not show a clipped
or selection-only replacement chart.

Use a fixed small sample selected before inspecting model differences: a target
of 30 held-out positive cells, up to six per concept where available. Include
examples with distributed selections, partial chords, and entering holds as
separately identified diagnostic cases. Report human judgments if a human performs
the inspection; machine-only inspection remains a machine assessment. Freeze
cell IDs and the output/seed subset for inspection before viewing differences.
Additional failure-driven cases are exploratory additions, not replacements for
the fixed sample. These 30 cells supply diagnostic case evidence, not a stable
estimate of semantic success rates across all five concepts. Multiple outputs
of one cell do not increase the number of independent cases.

For each output, record:

1. The specific relationship located by the highlights, identifying selected
   source objects and any contextual objects needed to describe it. If the
   highlights locate none, say so; recognizing the concept elsewhere is not
   evidence for this witness.
2. Whether the original context supports, weakens, or defeats that interpretation,
   citing relevant complete attack groups, interruptions, or hold/release order.
3. A verdict of useful, misleading, no relevant relationship located, or unresolved,
   with a short reason tied to the highlights. For paired count-cost outputs,
   record whether the quantity change loses an essential relationship.

For example, a Trill judgment should identify how the highlights locate fixed
disjoint A/B groups and whether intervening complete groups sustain that reading.
Saying only that the section contains Trill does not assess the highlights.
The highlights must participate in locating the relationship; each selected
object need not be indispensable, both endpoints need not be selected, and
the selection need not be sufficient without context. Allow both compared
outputs to be useful, both to fail, or the comparison to remain unresolved.
Do not require a unique minimal witness or explanations for every unselected note.

Useful failure categories are:

| Failure | Concrete diagnostic |
| --- | --- |
| Manufactured recurrence | A selected subsequence looks Jack-like only after intervening original attacks are ignored |
| Manufactured alternation | Selected A/B notes conceal changing groups or decisive extra attacks in the source |
| Missing hold relationship | Selected heads are interpreted without entering occupation or relevant release order |
| Wrong scope inference | One local gesture is used to claim an organization characterizes the entire section |
| Quantity-only improvement | Reference agreement improves mainly because count matches, without better relationships |
| Selection-policy imitation | Likelihood improves while generated relationships remain equally useful or equally misleading |
| Decorative highlighting | The explanation identifies the concept in the chart but cannot connect the highlighted objects to the claimed relationship |

Multiple selected runs are not a failure category. A relationship can span
unselected contextual objects, and multiple locations can jointly explain a
section. Likewise, attention weights and the $U/V$ score decomposition do not
establish the annotator's reasoning process.

### 11.3 Ancestral-sampling diagnostics

A high-scoring single output and a representative draw answer different
questions. For 100 independent binary choices with selection probability 0.1,
the expected count is 10 while the unique MAP vector is all zero. Thus greedy
or beam outputs alone do not characterize the learned selection distribution.

On the fixed inspection cells, generate four complete ancestral witnesses per
arm and training seed using Section 9's unmodified sampling procedure. Use
sampling seed `0` with reproducible streams keyed by cell ID, training seed, and
sample index; record the stream derivation and share it across arms. Keep every
draw, including duplicates and empty/full outputs. Do not select the best sample
by model score, reference agreement, count, or inspection verdict.

Report per-cell count distributions, empty/full frequencies, distinct object
selections, and which source locations vary across draws, alongside greedy and
beam 8 at $\lambda=0$. Apply Section 11.2's relationship and failure judgments
to a predeclared subset of draws if inspecting all is impractical, using the
same sample indices and training seeds for both arms. Preserve all draws in
quantitative summaries and report the inspection denominator. Do not pool draws
as independent cells or equate diversity with semantic quality.

Compare teacher-forced gains with behavior under generated histories: the model
may score a reference continuation well yet fail to start or maintain a useful
highlight sequence itself. More useful samples than mode-seeking outputs suggest
a distribution/search distinction; failures across samples reveal limits hidden
by a favorable single output. These are case-level diagnostics under a small
sample budget, not a new aggregate semantic benchmark.

### 11.4 What results permit

| Observation | Supported interpretation | Next decision |
| --- | --- | --- |
| Better likelihood and better inspected relationships | History helps fitting; inspected cases support improved relationship highlighting | Retain it and identify which relationships improved |
| Better likelihood only | History improves reference fitting; annotation strategy remains a plausible explanation | Localize decision gains; semantic benefit remains open |
| Improvement concentrated in count | A quantity or selection-policy explanation remains plausible | Compare overlapping achieved count ranges |
| Gains concentrated in skips or run continuation | Highlight persistence or quantity may explain the fitting benefit | Inspect run boundaries and count before attributing a gameplay distinction |
| Better teacher-forced likelihood with poor sampled highlights | Reference-history fitting does not yield reliable generation under the model's own histories | Inspect how selections start and how failures propagate |
| Useful samples with empty or misleading greedy/beam outputs | A high-scoring point may poorly represent useful mass; history-arm search is also approximate | Separate distribution quality from the choice and search of a representative output |
| Similar likelihood and useful outputs | This context/model/data combination may not need selection history | Prefer the simpler successful arm unless another measured need appears |
| Poor results in both arms | Representation, supervision, optimization, and concept ambiguity remain alternatives | Inspect concrete failures before adding model complexity |
| Benefit limited to Jack/Stream | Evidence is concept-specific under uneven support | Do not infer success for LN coordination or Tech |

A lack of detected gain does not prove that selection dependence or structure
is absent. Conversely, architectural capacity to represent a dependency does not
show that training used it meaningfully.

## 12. Implementation handoff and bounded execution

Implement this study in an isolated research package, for example
`src/pulsefield_model/research/witness_selection/`, with nearby focused tests.
These paths name proposed ownership; no implementation is implied. Do not build
the model on legacy mapper/tokenizer/training behavior. Follow repository config
guidance where applicable without treating retained mapper presets as V3 designs.

Keep the interfaces small:

| Component | Input and output contract |
| --- | --- |
| Dataset adapter | Pinned judgments plus exact source objects to immutable record IDs, scopes, review contexts, targets, and split groups |
| Replay and relation preparation | Complete visible source facts to lane states, encoder/decoder timelines, candidate masks, relation descriptors, and object-decision mapping |
| Encoder | Chart facts and relation graph to two contextual hand vectors per decision; no selection input |
| Row decoder | Context, concept, and per-hypothesis history to valid-mask log probabilities and simultaneous next histories |
| Trainer | Targets update history under teacher forcing; padding and forced decisions follow the declared loss rules |
| Decoder/evaluator | Greedy/beam outputs and ancestral samples, original sequence log probability, separate count cost, decision-level gains, agreement metrics, and complete-context inspection records |

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
| Concept / hand-mask embedding | 16 / 8 dimensions |
| Shared selection GRU | 64 hidden dimensions per hand |
| $U$ and symmetric $G$ readouts | One 64-unit hidden layer each |
| Dropout | 0.1 in encoder/readout; disabled in evaluation |
| Optimizer | AdamW, learning rate $3\times10^{-4}$, weight decay $10^{-4}$, gradient norm cap 1 |
| Batch and epoch | 16 sampled records; one epoch has the training cohort's record count of draws |
| Training bound | At most 30 epochs, early stopping after five epochs without validation macro-NLL improvement |
| Training seeds | Pilot seed 17; paired repeats 29 and 43 |
| Checkpoint selection | Lowest validation macro-NLL, using the same rule for both arms |
| Inference | Greedy and beam 8; primary $\lambda=0$; four temperature-1 ancestral samples per fixed diagnostic cell/arm/training seed, sampling seed 0 |

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
   release-only decisions add zero NLL; padding does not update histories.
5. `relations-context` is invariant to supplied selection histories; the history
   arm has independent state per beam hypothesis or sampled trajectory and
   simultaneous hand updates.
6. Mirror transformation preserves mapped probabilities and sequence scores.
7. Exhaustive enumeration on tiny charts agrees with sequence scoring and with
   sufficiently wide beam search under the original log-probability/count objective.
   Context-only rowwise maximization attains that exact optimum at every cost.
8. Dataset grouping has no exact source/cell leakage and the encoder has no
   target-bearing input channel. Training-only transformations fit training data.
9. Ancestral sampling uses the original normalized joint probabilities and sampled
   history, reproduces draws under fixed streams, and retains empty/duplicate
   draws. On a tiny enumerated distribution, sampled frequencies agree within a
   declared Monte Carlo tolerance; forced rows still advance state.
10. Decision-gain sums recover each record's paired loss difference and the
    primary aggregate. Skip/nonempty partitions reconcile; run boundaries use
    original attack rows, and overlapping relation flags are not double-counted.

These tests validate implementation semantics, not pattern accuracy. One-batch
gradient checks and tiny-slice overfitting diagnose training viability; neither
is a held-out result.

### 12.3 Run record and deliverables

Each actual comparison records its recoverable source revision, document/config
revision, dataset and split manifest, arm, seed, actual command, device, parameter
count, runtime bound, and output location before training. Record the command
once the research entrypoint has been implemented and its interface verified.
Use a fresh artifact directory per run. Resume only a matching configuration and
dataset/split identity, and keep checkpoint-selection and budget usage visible.
Record decoding and sampling runtime separately from the training allocation.

The implementation handoff consists of the adapter, replay/relations, paired
models, objective-preserving decoder, focused correctness evidence, and a compact
paired result report. Include cohort flow/counts, validation choice, per-tag and
aggregate metrics, decision-gain strata, quantity curves, all ancestral draws and
their sampling manifest, complete-context inspection records, failures, and
deviations. State the number of distinct cells/groups behind rare-case and
semantic claims. Raw artifacts remain outside tracked product sources.

Numerical instability, incorrect source identity, or representation-contract
failures stop the affected run. Missing exposure metadata, absent independent
human selection gold, or an unavailable optional semantic inspection does not
block the likelihood experiment; report the corresponding interpretation limit.

A formal accepted Experiment Card and executed Result Logs, when created, belong
to an owning Agent Note under the repository research workflow. This document
provides the self-contained model and evaluation specification; it does not
assert card acceptance, completed implementation, or authorization for a run.

## 13. Connection to Pulsefield responses

The useful transfer is from verified arrangement distinctions to mapper-facing
questions about legal continuations. Examples include whether a continuation
preserves repeated-column organization or redistributes it, extends fixed-group
alternation or changes group membership, and introduces independent actions or
releases against existing LN occupation.

Such questions still need scope, horizon, positive/negative/equivalent examples,
and a declared response comparison. This selector sees later chart context in
its declared review window. It is not a causal prefix state; later source rows
would need to be provisional continuations when used during generation.

The selection states $q$, relation counts, concept probabilities, and witness
size are not gameplay demand coordinates. Better recognition of a scoped concept
also does not establish that a state preserves the full continuation-response
function. The target response specification precedes that adequacy claim.

## 14. Closest research analogues

- [Lei, Barzilay, and Jaakkola (2016), Rationalizing Neural Predictions](https://aclanthology.org/D16-1011/)
  compares independent and recurrent history-dependent binary selection from
  bidirectional context. The direct analogue is the selector's factorization and
  memory. Its latent-rationale training and isolated-rationale sufficiency objective
  differ from this supervised, full-chart-context witness task.
- [Shaw, Uszkoreit, and Vaswani (2018), Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/)
  motivates transmitting declared relationships through attention. The proposed
  chart graph instantiates domain-specific action relations rather than relying
  solely on sequence distance.
- [DeYoung et al. (2020), ERASER](https://aclanthology.org/2020.acl-main.408/)
  separates reference-rationale agreement from faithfulness. Its deletion-based
  measures cannot be imported as unchanged style judgments on an edited chart.
- [Carton, Kanoria, and Tan (2022), What to Learn, and How](https://aclanthology.org/2022.findings-acl.86/)
  finds that better rationale supervision fit need not improve label prediction.
  It motivates measuring any recognition transfer separately.

The provisional contribution is an adaptation to source-linked 4K action
relationships and evidence about which dependencies help. An encoder/attention/
GRU combination by itself is not a claim of a new general selection method.
