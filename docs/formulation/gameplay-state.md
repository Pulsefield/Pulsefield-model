# Gameplay demand, continuation responses, and style

Ensomi describes gameplay demand through the responses that a chart history
creates for possible future actions. The gameplay frontier collects those
responses. A demand state $d$ is a proposed representation of that object.

The target responses take semantic priority over their representation. Their
concrete definitions will be established from a mapper's perspective after the
initial chart dataset is complete: which gameplay distinctions matter, which
do not, and what evidence establishes each distinction. The initial dataset's
style judgments help develop these definitions; they do not already specify
numerical demand or a response evaluator.

[notation.md](notation.md) owns chart syntax, exact replay, legal continuations,
and committed history. This page owns the meaning and evidence requirements of
demand, frontier representations, style observations, and semantic controls.

## Canonical gameplay profile

The canonical profile $\pi_0$ fixes the 4K lane-to-hand-role mapping, successful
execution of legal actions at their chart times, and left-right symmetry.
It supplies one reproducible convention for discussing gameplay organization.
The target-response specification will supply the associated demand semantics.

This scope excludes individual capacity, misses, timing noise, adaptive
fingering, handedness-specific profiles, physiological fatigue, and subjective
pain. Mapper judgments concern a chart under the declared convention. They
are not observations of a particular player's internal state.

Exact facts and gameplay interpretations have different sources:

| Quantity | Source |
| --- | --- |
| Row times, actions, and long-note occupancy | The materialized chart and exact replay |
| Recurrence, chord sizes, timing relationships, and hold/release relationships | Computation from declared chart context |
| Style presence and strength | Scoped semantic judgments under a versioned vocabulary |
| Target continuation responses | The mapper-facing response specification to be defined from the initial dataset |
| Predicted responses and runtime demand state | A chosen representation and its dynamics |

Demand and style can influence preferences among legal charts. Neither changes
chart legality.

## Target response and frontier

Let $H_{\le t}$ contain the materialized rows through time $t$, with all row and
no-row decisions through $t$ fixed. For a legal continuation $Y$ over $(t,e]$,
write the target response as

$$
\mathcal C_0(H_{\le t},t;Y,e).
$$

This notation names the response that the canonical specification should assign
to that history and continuation. Its output quantities, units or ordinal
scales, and comparison rules remain to be defined. It does not designate an
existing simulator or a measured response dataset.

The horizon $e$ is explicit because a continuation includes time after its last
row. An empty continuation can still change demand through elapsed time and
sustained long-note occupancy.

The gameplay frontier at $t$ is the function

$$
\mathcal F_H(t):
(Y,e)\longmapsto\mathcal C_0(H_{\le t},t;Y,e),
$$

over legal future continuations and their horizons. It describes how different
possible futures would respond to the same past. Only $H_{\le t}$ supplies
historical chart information; the proposed $Y$ supplies the hypothetical future.

A current difficulty or intensity scalar need not distinguish all such
responses. Whether two histories should have different frontiers is decided by
the target-response semantics, rather than by whether a candidate state encoder
happens to distinguish them.

### Defining the response from mapper evidence

The initial chart dataset provides concrete arrangements, local style judgments,
and their surrounding context. After its completion, response definitions must
state:

- the mapper-facing question each response answers, including its exclusions;
- the historical context and legal continuation on which it depends;
- positive examples, contrasting cases, and cases intended to be equivalent;
- how judgments are expressed and compared, including unresolved cases;
- the relevant horizons and the evidence supporting each claimed distinction.

Repetition, alternation, recovery intervals, and occupied-lane interactions can
identify useful comparisons. They do not determine response channels or
numerical update rules by themselves.

The same mapper understanding informs style and demand. Style annotations can
identify histories and continuations for which a response distinction matters.
Response definitions can, in turn, clarify the gameplay relationships behind
a style concept. This relationship must be developed through the examples and
definitions; neither vocabulary automatically supplies the other's labels.

## Representing the frontier with demand state

A representation with specification $\psi$ may compute

$$
d_\psi(H;t)=D_\psi(H_{\le t},t).
$$

Its purpose is to preserve the information needed for the declared target
responses in a useful, computable state. A fixed-size state, its dimension,
and its internal coordinates are representation choices.

Given exact replay state $x_H(t)$, the representation predicts responses through

$$
\widehat{\mathcal C}_\psi
\bigl(t,x_H(t),d_\psi(H;t);Y,e\bigr)
\approx
\mathcal C_0(H_{\le t},t;Y,e).
$$

The approximation must be assessed against a declared continuation family,
horizon range, response comparison, and acceptable discrepancy. These criteria
depend on the target-response specification. A representation need not preserve
all information useful for music, motif identity, or style recognition.

State sufficiency concerns the entire retained summary, including any exact
history features supplied alongside $d$. Storing omitted history in another
cache does not establish that a smaller total state is sufficient. The generator
continues to have access to committed chart history regardless of the chosen
demand representation.

### Dynamics and reproducibility

A state model may combine an event update and an advance through an interval
without new rows:

$$
d^+=U_\psi(d^-,x^-,y),
\qquad
d(t+\Delta)=V_\psi(t,d(t),x_H(t),\Delta).
$$

The first operation processes one legal row from its pre-event state. The
second applies only when no row occurs in $(t,t+\Delta]$. Exact occupancy and
other supplied exact facts follow chart replay. These interfaces do not choose
a particular accumulation law, decay kernel, or neural architecture.

Elapsed time is not automatically rest: an open long note remains occupied.
Whether occupancy sustains demand, a release changes it, or earlier actions
have lasting effects depends on the target semantics and the dynamics selected
to represent them.

For a fixed deterministic specification, replaying the same history must
produce the same state and predictions. A cached state must agree with replay
under the same specification and boundary, including initialization before
the first row and advance through silent time. Splitting a time advance into
smaller intervals must preserve that result within the declared numerical
tolerance. This establishes reproducibility; it does not establish that the
predictions preserve the intended responses.

In particular, at the same boundary time, matching $(x,d)$ and applying the
same predictor to the same continuation and horizon must give matching
predictions. That equality cannot reveal information already lost by the
representation. Sufficiency requires comparison with target
responses whose basis is independent of the candidate compression. A
full-history reference model can provide such a comparison, but the resulting
claim is relative to that reference model.

### Interpretable quantities and symmetry

An exposed demand quantity needs a declared interpretation, scale, and
supporting response comparisons. A coordinate does not acquire a gameplay
meaning merely from its name or correlation with a style label. Internal
memory features may remain unnamed.

Local intensity and section or map difficulty require their own definitions.
Neither is automatically a norm or average of the state. Desired intensity,
realized style strength, and confidence in a judgment are different quantities.

Under the symmetric canonical profile, mirrored histories and continuations
should have correspondingly mirrored target responses. A representation must
declare the matching transformation of its outputs. This permits interactions
between hands; it does not prescribe separate independent hand models.

## Style observations

Gameplay style describes the recognizable organization of a materialized chart
over a declared scope. Concepts can overlap and correlate. A section may express
several strongly, express one weakly, or have no salient concept in the selected
vocabulary. The vocabulary need not exhaust all chart organization.

The initial section annotation vocabulary in @ensomi-labs/beatmap-lens comprises
Jack organization, Stream organization, Trill organization, Tech, and LN
coordination. These are experiment-specific, versioned concepts. Their local
definitions and calibration examples accompany the dataset; the five names are
not a universal taxonomy or the final set of generation controls.
Historical records retain their pinned vocabulary and definitions. Adding a
concept or revising a definition does not retroactively label that concept or
reinterpret an earlier judgment.

The annotation scope is chart organization supported by source actions and
declared chart context. It excludes concepts whose distinguishing evidence
requires aligned audio, external presentation assets, provenance, or unobserved
mapper intent. A concept needing a broader chart context must declare that
context rather than assume a finite entry state preserves it.

For a mirror-invariant concept, mirroring the chart and its context preserves
the assessment. Any directional concept must declare how its meaning
transforms under reflection.

### Presence and ordinal strength

Each concept receives its own assessment:

| Assessment | Meaning |
| --- | --- |
| Present, supporting | The concept is present and contributes to the section, with weaker expression |
| Present, prominent | The concept is present and strongly characterizes the section |
| Absent | An explicit negative for this concept after inspection of the declared scope |
| Unresolved | Inspected, but the evidence or semantic boundary does not settle the assessment |
| Unreviewed | No assessment has been made; an omitted claim has this status |

Supporting and prominent are ordered positive judgments of how strongly the
section expresses the concept. Supporting is not low confidence. Typicality
can inform strength, while note coverage, duration, speed, and repetition count
alone do not define it.

These observations establish an ordinal distinction, not equal spacing,
probabilities, or calibrated continuous values. A recognizer may predict
probabilities of absent, supporting, or prominent outcomes, or use continuous
internal scores, with separately declared meaning and calibration. Unresolved
and unreviewed describe the observation status; they are not additional
realized styles.

Multiple concepts can be prominent in the same scope. Assessments do not sum
to one, and judging one concept does not resolve the others. Positive-first,
partially exhaustive collection leaves missing assessments unreviewed.
Unresolved and unreviewed remain distinguishable even when both are excluded
from a supervised presence loss.

### Scope, context, and evidence

A claim binds a concept and assessment to an exact chart source and a source
interval. Its judgment concerns the complete arrangement in that interval.
Selected witness notes explain the judgment; they are neither exhaustive
note-level labels nor a replacement for the surrounding chart.

@ensomi-labs/beatmap-lens uses source milliseconds and half-open scopes
$[a,b)$. Its review context can extend beyond the scope. Claims may overlap,
and different concepts may have different boundaries. Successive local
patterns do not automatically establish that both characterize an entire
larger interval.

Converting milliseconds to seconds preserves half-open membership; it does
not turn $[a,b)$ into the generation interval $(a,b]$. Source note identities
and original hold endpoints must survive conversion. A hold overlapping a
scope is a source object; its start and close are distinct timed row actions,
which may lie outside that scope. Entering occupancy and boundary actions must
be obtained from exact replay, not reconstructed by clipping the hold.

Boundary uncertainty and an observed transition carry their own meanings.
Neither resolves uncertainty about the concept itself. A recognizer must
declare the chart context it receives, including whether it has access to
context after the judged interval. During generation, chart context beyond
the committed prefix can come only from the provisional continuation.

### Provenance and coverage

Retain the source identity, vocabulary version, scope, assessment, and judgment
origin needed to interpret an observation. Human judgments, machine proposals,
deterministic query results, and independent machine audits are different
evidence. Machine agreement does not create a human judgment.

A rejected proposal is not an absent label. A deferred decision supplies no
settled assessment. Read the assessment itself even when a historical record
carries human provenance; provenance alone does not establish presence.
Older annotation formats retain their original meanings and require explicit
reinterpretation before being combined under another scale.

Overlapping claims, related difficulties, and examples exposed during
calibration must remain grouped appropriately for evaluation. A collection of
selected positive episodes does not establish exhaustive chart coverage or
whole-map style prevalence.

## Relationship between style and demand

Style and target responses concern the same arrangements from related mapper
perspectives. Their connection is part of defining the gameplay problem, not
merely an auxiliary classifier experiment.

Exact recurrence, timing, chord, and long-note relationships provide evidence
for style judgments and can motivate demand-response comparisons. A computed
structural match alone need not settle a style concept or its strength. Context
and the complete local organization remain relevant.

A style recognizer may use exact chart context and demand features together.
It does not require a separate section-geometry object or a mandatory demand
bottleneck. A style distinction can remain useful even when the response
specification intentionally treats its alternatives as equivalent.

Conversely, if mapper-defined target responses distinguish two histories,
a demand representation must retain that distinction within its claimed
scope, whether or not the histories receive different style labels. Better
style recognition alone cannot establish response sufficiency.

Chart-only, demand-only, and joint recognition comparisons can help examine
this relationship. Their available history, model capacity, and evaluation
examples must be declared: a gain from longer context is not, by itself,
evidence of a distinct demand mechanism. Since demand is derived from chart
history, it adds a representation of that history rather than a new external
observation.

## Community observations

Community difficulty-level tags and local section concepts have different
vocabularies and evidence scopes. A shared name does not establish equivalence.
Some community predicates require audio, whole-map comparisons, or external
context outside the section annotation scope.

An explicit alignment may make a community observation useful as weak evidence
for a local concept. It must declare the direction of the implication, source
and target meanings, and relevant scope. Broad Jack organization, for example,
does not identify which particular jack subtype received community votes.
No community alignments or pooling rules are fixed by this formulation.

Retain catalogue identity and available vote information. Votes are neither
independent training examples nor calibrated probabilities of truth. Missing
tags remain unobserved, and a map tag must not be copied to every section.
Any attempt to predict map tags from section observations needs a separately
evaluated aggregation rule that accounts for selection and missing coverage.

## Controls

The generation problem accepts optional style and demand requests,
$c_W^{\mathrm{style}}$ and $c_W^{\mathrm{demand}}$, as defined in
[notation.md](notation.md#generation-and-optional-controls). Either, both, or
neither may be supplied.

A style request asks the generator to favor a declared concept or combination.
An absent request permits the learned natural style distribution; it does not
request an absence of style. Realized presence and strength are assessed on
the resulting chart. The request is not itself an observation of that result.

A demand request refers to quantities or relationships in the target-response
specification. Its concrete interface remains open until those semantics are
defined. A target in the coordinates of one learned state is meaningful only
with that representation and its mapping to the declared response quantities.
Renaming arbitrary latent coordinates as demand controls does not establish
mapper-facing semantics.

Requested style tendency, style adherence, desired demand, and sampling
temperature have separate meanings. A style amount control must specify
whether it concerns local expression, coverage, repetition, or another
calibrated property. The ordinal annotation scale does not by itself define
a numerical control interface.

Style and demand requests can be correlated or incompatible. Separate
interfaces do not guarantee independently achievable effects. Generation must
preserve legality and committed decisions, and use a declared compromise or
infeasibility policy when requests cannot be jointly realized.

Neither control requires a separate planner or an explicit demand trajectory
before row generation. When a representation is available, its state and
predictions can be derived from each proposed branch without becoming committed
facts. Generation and evaluation must preserve the branch isolation defined in
[notation.md](notation.md#provisional-branches-and-prefix-commit).

## Evaluation questions

The initial dataset supports assessment of style recognition and provides
examples for defining target responses. Demand representation evaluation
requires that response specification first. The following comparisons have
distinct purposes:

| Question | Required comparison |
| --- | --- |
| Does style recognition recover the declared concepts and strengths? | Held-out scoped judgments, retaining explicit negatives, unresolved assessments, related-source grouping, and ordinal strength |
| Does a demand representation preserve the target responses? | Predictions against independently specified responses on held-out legal continuations and declared horizons |
| What does demand contribute to style recognition? | Chart-only, demand-only, and joint predictors with declared context and density, event-count, or difficulty baselines |
| Does a style request produce its intended semantic effect? | Fixed audio and committed history, feasible comparison conditions, and independent judgments of the generated organization |
| Does demand control change the intended response? | Defined target-response comparisons, checking style changes and other tradeoffs rather than assuming independence |
| Does a representation respect the canonical symmetry? | Mirrored histories and legal continuations with the declared output transformation |

Continuation probes require legal starting states. A start, hold, or release
probe cannot be applied indiscriminately to every occupancy state. A paired
comparison needs a continuation legal from both histories, or an explicitly
declared correspondence between their actions.

Changes in note count, global difficulty, or decoding entropy alone do not
establish successful semantic control. Human judgments or independently
validated evaluators must recognize the intended change. Consistent replay
and mirror behavior are useful checks, but do not prove sample efficiency,
frontier sufficiency, or agreement with mapper-defined response semantics.
