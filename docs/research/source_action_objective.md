# Source-action prediction and evidence contract

The research objective is to learn conditional dependencies among complete
four-lane action rows across time, relationships and interval scales, and make
those dependencies reusable by different queries. Conditional reconstruction,
use of specific context, and semantic reuse are separate evidence requirements.
No single loss establishes all three.

This document owns the probability, weighting and diagnostic meanings used by
[`source_action_modeling`](../../src/ensomi_model/research/source_action_modeling/).
The [foundation](source_action_stage1.md) owns source observation and legality;
the [composition guide](source_action_stage2.md) owns the representation bank.
The [V3 formulation](../formulation/README.md) retains authority over generation
and gameplay demand. Corpus action prediction does not calibrate player ability,
effort, success probability or a demand state.

## Prediction object and conditions

Let $A=(a_1,\ldots,a_N)$ be complete source-event action rows, including
release-only events. Each lane uses the V3 action codes `EMPTY=0`, `TAP=1`,
`LN_START=2`, `LN_CLOSE=3`. Same-lane close/tap and close/head coincidences are
rejected. The supplied skeleton $\Gamma$ contains the union of real
attack and release times, with declared scope/context and synthetic markers.
It reveals event existence and timing, but not hidden attack/release types,
lane membership, attack ranks or LN identities. This task predicts arrangement
at supplied times; it does not predict the next event time or event existence.

For a contiguous event block $I$, the condition $C_I$ comprises:

- the fixed skeleton, scope and context;
- visible action facts and their availability, plus any supplied summaries;
- declared context-entry occupation and the decoder's stated block-entry
  occupation, each with unknown values retained;
- the observation, target-selection and query policies, including target
  positions and any metadata those policies expose.

Features, relation edges and local state estimates are deterministic descendants
of these permitted facts. Construct availability first and derive them again
from that observation. A visible LN head does not disclose its full endpoint;
hidden actions invalidate occupation and recurrence knowledge. Never use target
facts to choose graph topology or infer a post-block legality constraint.

The implemented distribution is

$$
p_\theta(A_I\mid C_I)
=\prod_{j\in I}p_\theta(a_j\mid A_{I,<j},C_I,j).
$$

Each factor normalizes over complete four-lane rows. Legality reads the stated
block-entry occupation and preceding decoded rows only; an unknown lane permits
the union of legal possibilities until an observed action resolves it. A real
event excludes the all-silent row. Teacher forcing scores a row before consuming
its true action. Padding neither contributes loss nor advances state.

This defines a normalized conditional block distribution under prefix-only
legality. It does not enforce compatibility with every visible suffix fact or
guarantee that conditionals for different masks derive from one global chart
distribution. Its lane alphabet and local occupation transitions match V3; supplied
event times and incomplete-context conditions still differ from the complete
V3 generation contract.

## Joint hand output family

[`JointDecoder`](../../src/ensomi_model/research/source_action_modeling/model.py)
scores 16 candidates per hand with shared unary terms and a bilinear potential.
Let $e_a$ be a shared projected action embedding, and $q_L,q_R$ the hand queries
formed from both encoded contexts, both recurrent states and permitted
occupation. The pair term is

$$
J_x(a,b)=\frac{e_a^\top B(x)e_b}{\sqrt D},\qquad
B(x)=\frac{M(q_L)+M(q_R)^\top}{2}.
$$

$M$ is an unconstrained learned linear map into $D\times D$ matrices. Exchanging
hands exchanges the queries, transposes $B$, and transposes the joint score
table. At symmetric inputs $B$ is symmetric but need not be positive
semidefinite. For four legal combinations, the log odds contrast is

$$
\log\frac{p(a,a\mid x)p(b,b\mid x)}{p(a,b\mid x)p(b,a\mid x)}
=\frac{(e_a-e_b)^\top B(x)(e_a-e_b)}{\sqrt D}.
$$

This contrast can change sign with context. Unary terms and the softmax
normalizer cancel. The default interaction dimension is eight; the interaction
table remains rank-bounded by the projected embeddings. A 256-index softmax
therefore provides joint normalization without claiming an arbitrary categorical
family. Mirror, gradient, padding and legality checks verify the decoder
contract. The construction below verifies an expressible distinction; it does
not establish a learned Trill mechanism or improved encoder representation.

### Why context-dependent signed coupling is required

The [affine-concatenation decoder at `790add7`](https://github.com/ensomi-labs/ensomi-model/blob/790add7b07ecf08e1b277797a39ce28e9683a5b8/src/ensomi_model/research/source_action_modeling/model.py)
formed each interaction vector with one linear map over context and candidate
embedding. Consequently, its vectors could be written as

$$
v_L(a,x)=c_L(x)+e_a,\qquad v_R(b,x)=c_R(x)+e_b.
$$

Their dot product contains a constant, two context-dependent unary terms, and
the static pair term $e_a^\top e_b$. The constant and unary terms cancel from
the four-combination log odds. Whenever all four combinations are legal,

$$
\log\frac{p(a,a\mid x)p(b,b\mid x)}{p(a,b\mid x)p(b,a\mid x)}
=\frac{\|e_a-e_b\|^2}{\sqrt D}\ge 0.
$$

This restriction holds for every encoder output and every setting of the
decoder parameters. Even an encoder that identifies a relationship exactly
cannot make this decoder express a negative contrast.

For example, consider two hand-local candidates $a,b$ and normalize over their
four legal joint combinations:

| Context | $aa$ | $ab$ | $ba$ | $bb$ | Log odds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefer matching candidates | 45% | 5% | 5% | 45% | $+\log 81\approx+4.394$ |
| Prefer different candidates | 5% | 45% | 45% | 5% | $-\log 81\approx-4.394$ |

The second row is impossible under the affine-concatenation family. These
probabilities condition on membership in the four displayed combinations;
other legal classes may still receive probability in the full distribution.

The bilinear family constructs both rows with one active coordinate. For a
unit vector $u$, set $e_a=u$, $e_b=-u$, equal unary scores on these candidates,
and

$$
B(x_+)=\sqrt D\log 3\;uu^\top,\qquad
B(x_-)=-\sqrt D\log 3\;uu^\top.
$$

The four logits are respectively $(\log3,-\log3,-\log3,\log3)$ and their
negatives. Their softmax distributions give the table above. The
[`test_context_can_reverse_joint_hand_log_odds_with_all_four_choices_legal`](../../tests/research/source_action_modeling/test_model.py)
regression constructs these distributions through the actual decoder with its
default interaction dimension. This establishes removal of the sign
restriction. It does not prove that rank eight suffices for corpus structure
or that training will discover the required coupling.

## Sequence cost, row cost and training risk

Retain both code length and rate, in natural-log units:

$$
S_I=-\log p_\theta(A_I\mid C_I),\qquad
\bar S_I=S_I/|I|.
$$

`Prediction.row_nll` contains the chain-rule terms, `sequence_nll` their block
sum, and `mean_row_nll` their block mean. `loss` averages `mean_row_nll` equally
over batch entries. The former ambiguous `block_nll` field is removed. No
length correction turns either measure into a length-independent measure of
structural understanding.

For a fixed declared condition $C$, let $P(\cdot\mid C)$ be the population
conditional induced by the source and selection policies. Expected sequence
NLL satisfies

$$
\mathbb E_{A\sim P(\cdot\mid C)}[-\log p_\theta(A\mid C)]
=H(P(\cdot\mid C))+
D_{\mathrm{KL}}\!\left(P(\cdot\mid C)\,\|\,p_\theta(\cdot\mid C)\right).
$$

The entropy term is independent of model parameters. This makes NLL a justified
objective for approximating the specified conditional distribution. The
identity imposes no requirement on where the predictive information resides:
an encoder that passes through source facts, a capable decoder, a broad chart
prior, or the true prefix can improve likelihood without improving a reusable
encoder representation. Context-use and frozen-readout evidence therefore
remain separate requirements.

For example, if every row repeats one unknown fair bit, the first row costs
$\log 2$ and subsequent teacher-forced rows cost zero. Then $S_I=\log 2$ for
every block length while $\bar S_I=\log 2/|I|$. This is the cost of one shared
decision. Averaging dilutes it in long blocks; teacher forcing gives later
positions progressively more true prefix. These are objective choices, not
numerical errors to remove through normalization.

[`BlockSampler`](../../src/ensomi_model/research/source_action_modeling/sampling.py)
defines the baseline distribution $q_0$: uniform represented group $g$, uniform
context $c$ within that group, uniform feasible row count $s\in\{4,16,64\}$,
then uniform event start $u$. With $N_c$ source events and feasible set $F_c$,

$$
q_0(g,c,s,u)=\frac{1}{|G|\,|C_g|\,|F_c|\,(N_c-s+1)}.
$$

Each draw records this marginal block `sampling_probability`. Counts and
source identities select positions; hidden action values and labels do not.
Conditioned on the supplied skeleton, this selection adds no hidden action
attribute. It still inherits the chosen corpus and context population, and it
is not uniform over all charts, events, milliseconds or globally pooled scales.
Short contexts redistribute mass over their feasible scales. Realized attack
count is analysis metadata only.

The single-view risk is $\mathbb E_{q_0}[\bar S_I]$. Paired training evaluates
all three views per block with equal weight:

$$
R(\theta)=\mathbb E_{(g,c,s,u)\sim q_0}
\left[\frac{1}{3}\sum_{v\in\{n,d,c\}}\bar S_{I,v}\right].
$$

There is no target-action, capability or duration reweighting in this risk.
The loss policy is `equal-block/mean-row-nll-v1`; paired update reports also
record the sampler policy and view weight. Contexts with equal group weight
need not have equal individual probability. The probability in each draw does
not include a view factor because every paired draw exposes every view.

Retaining this baseline weighting keeps the architecture comparison tied to
one stated risk. It is not a claim that equal-block mean-row weighting is
optimal for structure learning. Dividing by the known block length preserves
the preferred distribution at a fixed condition, but changes the relative
importance of conditions for a model with shared, finite capacity. A later
weighting change requires its own declared target risk and comparison.

Changing the exposure distribution to
$q_{\mathrm{mix}}=(1-\alpha)q_0+\alpha q_h$ has two distinct interpretations.
Unweighted updates optimize the new risk. To retain $q_0$ risk, use the exact
weight $q_0/q_{\mathrm{mix}}$ where support is covered and both probabilities
are known; clipping or self-normalizing weights changes the estimator. No
heuristic mixture or importance weighting is implemented here. A rule based
on hidden actions can also make mask shape informative about the answer;
reporting exposure counts alone does not account for that selection effect.

## Evaluation and claims

Structural reports retain sequence and mean-row costs, their paired gains, raw
row costs, first-row cost and later-row mean. They stratify by event count and
actual duration, and report each zero-based decoder position. A later position
contains only blocks long enough to reach it; it is not the same population as
the first position. Inspect the raw scale/position records together before
interpreting a length trend. Position summaries retain group means without
bootstrap intervals; the aggregate and scale/duration comparisons carry the
paired group uncertainty.

For near or coarse view $v$ versus detailed view $d$, retain
$G_I=S_{I,v}-S_{I,d}$ and $\bar G_I=G_I/|I|$. The comparison holds target,
teacher-forced prefix, skeleton, model and decoder legality fixed. Detailed and
coarse also receive identical unordered summaries. Positive gain means the
model assigns the target higher likelihood with the detailed condition. It
does not estimate true conditional mutual information or isolate one encoder
module's contribution.

Evaluation first averages available block metrics within each represented
source group, then averages groups. Group bootstrap intervals resample means
of already paired differences. A fixed validation manifest defines this
finite evaluation population; it need not reproduce the training distribution
over contexts and feasible scales. Freeze that manifest across comparisons.
Do not silently interpret group-macro validation as a Monte Carlo estimate of
$q_0$ risk or pool all target rows into the training objective.

| Research claim | Required evidence and limit |
| --- | --- |
| Predicts action dependencies | Held-out conditional likelihood under a fixed population and information policy, with scale, duration and decoder-position costs. A lower loss alone may reflect corpus priors or true-prefix use. |
| Uses specific contextual organization | Matched near/detailed/coarse gains beyond shared timing and summaries. These identify a benefit of the supplied content, not a uniquely necessary local layer. |
| Makes dependencies reusable | Fixed-capacity semantic probes on frozen representations, compared with matched untrained encoders and the same raw-fact access. Human distinctions remain independent evidence; complete-input probes use a different observation policy from masking. |
| Routes the same facts compatibly | Small fixed-information path diagnostics, reported with prediction quality. Agreement alone permits collapse to an uninformative prior. |

These criteria permit a negative research conclusion even after reconstruction
improves. If lower NLL does not also improve the specified context-use contrasts
and matched frozen-readout results, the claim of learned reusable structure
remains unsupported. A readout failure alone does not establish that the
encoder contains no relevant information; readout access and capacity remain
possible explanations. Software checks establish the probability and diagnostic
contracts. Corpus benefits require a held-out comparison under the declared
population and information policies.

Gradient reachability, attention weights and the presence of retained states
do not establish reuse. Existing actual-update diagnostics measure the change
in negative mean-row risk or a fixed positive-minus-negative semantic response,
with disjoint module contributions and a checked linearization residual. They
do not attribute an update to individual examples or prove that a module with
a small contribution stores no useful information.

## Fixed-information prefix routing

[`prefix_path_pair`](../../src/ensomi_model/research/source_action_modeling/consistency.py)
splits an already selected hidden block into nonempty prefix $K$ and suffix $J$.
Choose the split index independently of hidden action values. Compare

$$
p_\theta^{K\cup J}(a_j\mid A_K,A_{J,<j},V,\Gamma)
\quad\text{and}\quad
p_\theta^J(a_j\mid A_{J,<j},V\cup A_K,\Gamma).
$$

The first path receives $K$ through decoder history; the second receives it as
visible encoder actions. The suffix remains hidden and both decoders receive
the same preceding suffix rows. The pair constructor enforces:

1. Preserve scope, context, all times/phases/markers, base context-entry state
   and all other action facts/unknowns. The base must have no far summaries.
2. Reveal only $K$, then rebuild derived features and relations from the new
   observation. Their values may change; their information cannot exceed
   $V\cup A_K$. Current/future suffix values never enter either encoder.
3. Advance the original decoder entry condition through $K$ to obtain the
   second decoder's entry at $J$. Do not substitute a stronger occupation
   estimate available only through a different view's legality policy. Known
   query-entry values must be derivable from the base observation; a weaker
   common entry is permitted. Independently supplied entry conditions are
   outside this diagnostic's supported policy.
4. Keep target identity, prefix values and legal class support identical at
   each compared suffix position. Fail on support mismatch instead of reporting
   a divergence between different prediction tasks.

The base view is constructed once. Far summaries currently encode only a side
and count vector, without the source interval supporting those counts. Retaining
the same vector while moving the target boundary does not provide an explicit
guarantee that both routes can identify the same summarized facts. The first
diagnostic therefore rejects summarized inputs. Use an unsummarized observation,
the near view, or remove summaries once from the base before constructing both
routes. Supporting summarized paths requires explicit summary provenance in the
input contract.

Calling `paired_views` again on $J$ would move the near neighborhood and alter
summaries and entry conditions. Such a pair is rejected. Source actions, times
and availability determine all changed
features; the constructor never uses action content to choose the split. The
caller remains responsible for the original block-selection provenance.
Selection of which derived entry values remain known must also use only the
permitted facts. The pair constructor cannot establish this provenance from
the resulting values alone. Supporting independent external state requires
specifying how its values and availability are supplied or recoverable in both
routes.

Equality of resulting occupation checks decoder legality, not the provenance
of every condition. For example, a known-held lane and an unknown lane can both
become known-unheld after a prefix containing a close. This many-to-one state
transition alone does not prove information loss: a complete legal prefix
whose first action on that lane is a close also implies that it entered held.
The base-derived-entry requirement is a conservative scope restriction, not a
theorem that every independently supplied entry would make the paths
incomparable. The
[`test_path_pair_restricts_query_entry_to_base_observation`](../../tests/research/source_action_modeling/test_consistency.py)
regression verifies that policy. If the entry fact follows from the base
observation, both routes retain access to it through that observation; a weaker
common decoder entry remains permissible.

`evaluate_path_consistency(model, pairs, batch_size=...)` evaluates without
parameter or gradient updates and restores module modes. For every common
suffix row it compares the full 256-index distributions using Jensen–Shannon
divergence, total variation and both KL directions. Illegal classes contribute
zero. It also records both routes' target NLLs, suffix sequence/mean-row costs,
legal class counts, position alignment, duration and batch hashes. Target NLL
alone can miss redistribution among equally legal alternatives.

Row divergences are evaluated along the common true prefix. Their average is
not a divergence between complete suffix sequence distributions. These are
per-pair diagnostics without an added training penalty or an automatic success
threshold. They test compatibility of two encodings of the same facts, not
equal loss for tasks with different information or full global joint coherence.
The encoder-observation route may be outside the trained mask distribution;
a discrepancy motivates examining routing, exposure and approximation together.

```python
from ensomi_model.research.source_action_modeling.consistency import (
    evaluate_path_consistency, prefix_path_pair,
)

# base_example is one fixed BlockExample; model is the predictor to evaluate.
pairs = [prefix_path_pair(base_example, prefix_rows=k) for k in (1, 2)]
report = evaluate_path_consistency(model, pairs, batch_size=2)
```

The example requires at least three target rows. A research comparison must
fix its base examples and split indices before examining the predictions.
Software tests exercise matched views, known/unknown occupation, LN releases,
hidden-suffix invariance and full-distribution differences. They establish the
diagnostic contract; corpus improvements require separate measured evidence.
