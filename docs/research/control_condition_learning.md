# Learning difficulty and style from real conditional variation

Difficulty/style balancing and explicit condition learning address different
problems. Balancing changes which genuine arrangements the model sees. A
condition-sensitive objective changes whether it must use the supplied request
to explain them. The current corpus supports testing both without adding an
inference network or assigning source labels to generated histories.

The [paired-outcome pilot](paired_scope_control_learning.md) improved difficulty
control too little. It trained only existing difficulty-input columns on twelve
charts. Its failure does not establish that a larger network, more policy-gradient
updates or a new skeleton representation is needed. A subsequent corpus
inventory and factual-condition audit instead found usable conditional variation
and a specific learned difficulty bias.

## Observed coverage

The frozen ranked TRAIN manifest has 6,923 charts:

| Whole-chart difficulty | Charts |
| --- | ---: |
| 2–3 | 2,558 |
| 3–4 | 2,300 |
| 4–5 | 1,611 |
| 5–6 | 454 |

Whole-chart bins describe coverage, not local supervision. A sixteen-second
passage may differ substantially from its chart's rating. The inventory retained
61,446 full sixteen-second windows with at least 32 H events, tracking H rate
and actual heads per H independently. Neither statistic is a style label or a
target to be emitted by the skeleton.

Exact matching of audio identity, clock range and every H timestamp found 2,382
groups with distinct materializations. A preliminary filter required at least
.5 whole-chart rating difference and .25 heads-per-H difference. After recomputing
the actual scoped difficulty of both arrangements, 1,339 pairs across 424 audio
identities had both local readouts in 2–6 and a difference of at least .5.

Of those pairs, 896 across 253 audios had local LN-head fraction at most .1 in
both arrangements. Another useful view is the 1,143 pairs across 357 audios whose
local LN fractions differ by at most .1. These are candidate pools after declared
filters, not counts of every useful pair in the corpus. Native recovery-profile
compatibility still needs checking when selecting a fitting panel.

This provides concrete examples of R1 making different arrangements under the
same audio and H times. Their histories and entering LN states can differ.
Matching timing therefore never authorizes transplanting one chart's next-row
labels onto another chart's prefix, nor proves both arrangements are reachable
from an arbitrary generated prefix.

## Semantic style coverage is much smaller

The current TRAIN cohort has 289 human-assessed cells on 112 charts. Concepts are
independent multilabel ordinal assessments, not mutually exclusive genres.
Unknown labels stay unknown. The prominent subset is:

| Concept | Prominent cells/charts | Local difficulty coverage |
| --- | ---: | --- |
| Jack organization | 11 / 11 | Includes 2–6, plus one passage below 2 |
| Stream organization | 21 / 21 | Includes 2–6, plus three passages below 2 |
| Trill organization | 13 / 13 | Includes 2–6, plus two passages below 2 |
| Tech | 5 / 5 | All five local readouts are in 3–4 |
| LN coordination | 8 / 8 | Four in 2–3, two in 3–4, two in 4–5 |

These counts use the stored human layer, not a newly filtered high-confidence
subset. Labels retain their original scope, rate, confidence and provenance.
Repeating five Tech passages cannot establish Tech control throughout 2–6.
Unlabelled ranked passages still teach natural event/row structure; they must
not be silently converted into absent-style examples.

For balancing, difficulty, H rate/rhythm and factual materialization should be
separate sampling dimensions. A slow chord passage and a fast single-note
passage can have similar difficulty while requiring different R1 choices.
LN amount and observed hold/attack relationships further distinguish their
execution. These structural sampling attributes are not semantic annotations.
Balancing should be capped and retain broad corpus exposure, rather than
equalizing empty or very small semantic cells by repeated memorization.

## Does R1 use the correct difficulty on genuine histories?

A bounded audit selected twelve distinct-audio pairs with identical H times,
local difficulty difference at least .75, local difficulties in 2–6 and at most
.1 LN fraction in both arrangements. Both sides of every pair passed the
current source-support collator; none was excluded.

Each of the 24 real source scopes was scored twice on its own genuine history:
once with its actual local difficulty and once with its partner's difficulty.
Full audio, H preview, all row labels, scope clocks, the chart's whole-LN request
and unknown style fields stayed fixed. Only the sixteen-second difficulty value
changed. The audit summed the neural R1 row log probabilities; it excluded
external recovery preferences and LN feedback so those policies could not
manufacture apparent neural condition sensitivity.

| Genuine source arrangement | Correct condition preferred | Mean correct-minus-swapped row log probability |
| --- | ---: | ---: |
| Lower difficulty | 12 / 12 | +1.685 |
| Higher difficulty | 2 / 12 | −1.323 |

Both sides ranked correctly in only two of twelve pairs. Overall, 22 of 24
scopes preferred the lower condition, including ten of twelve genuinely harder
arrangements. This small TRAIN diagnostic identifies a bias on factual
histories. It is not a population accuracy estimate, generation-quality score,
or proof that all controls are ignored. It also does not isolate which internal
R1 path causes the bias.

The bias gives a direct reason to test conditional learning before attributing
the entire failure to generated-history drift or insufficient model capacity.
Merely giving the model a control vector has not made its factor probabilities
consistently prefer the observed request even on these real arrangements.

## Proposed learning comparison

Use the same balanced source exposure in two arms: genuine-source generation
learning alone, and that learning plus a scoped condition-contrast objective.
Use real same-audio, same-H pairs to concentrate the comparison on materialization
variation. Retain each chart's own history and complete LN interpretation.
Difficulty targets come from actual scoped outcomes; semantic style targets come
only from applicable human assessments.

For a factual scope, compare its R1 factor score under the observed condition
with its score under a clearly mismatched condition. Apply the comparison to
the complete scope, allowing individual rows to serve different roles. It must
not impose a rule that every row gains heads as difficulty increases. Candidate
negative conditions should respect observed timing/control coverage and a
meaningful difficulty tolerance. For styles, unknown or ambiguously adjacent
ordinal values are not automatic negatives.

The closest analogue is
[Condition Contrastive Alignment](https://arxiv.org/html/2410.09347v1), which
fits an autoregressive model using matched and mismatched conditions and a
frozen-reference likelihood ratio. The proposed adaptation concerns native-time
scopes, continuous difficulty, partial multilabel style and responsible model
factors; it is not a claim of a new general contrastive method.
[CLICK](https://aclanthology.org/2023.findings-acl.65/) instead contrasts generated
sequences with attribute labels. That alternative would require trustworthy
quality/style labels for generated beatmaps, which the current scalar difficulty
readout does not provide.

The comparison should allow R1's existing composition and condition interactions
to adapt, rather than limit learning to difficulty-input columns again. It need
not enlarge the network. H retains timing ownership; R1 retains cardinality,
TAP/LN, release-subset and placement ownership. A factor-specific R1 contrast
prevents H alone from absorbing the condition objective.

LN articulation requires joint attention to R and R1. An early sampled R event
requires an actual release; R1 can choose the released subset but cannot turn
that event into a no-op. Conversely, different LN starts/subsets change the
occupancy and age inputs driving R. A future articulation fit must address that
coupling rather than only reward LN fraction or only tune row difficulty weights.
H still receives no materialized-row history. Long musical memory remains
outside this proposed intervention.

Condition contrast remains a training proxy. Improving this audit without
improving full generated continuations would be a failure of the proposal.
Qualification must keep difficulty/style/LN scopes separate, preserve committed
rows and entering holds, test restoration and dense passages, and inspect
meaningful repeated, mixed and sustained relationships with Lens. True native-H
generation is required before a playable-system claim. No new fitting run or
model adoption is reported by this investigation.

## Provenance

Executable source: `2c2b8af0da7411804697863df2f8efb605f3c840`.
Artifact owner: `20260926-control-coverage-v1`.
Manifest SHA-256:
`4cea2672387b6293a4da0846be479d8bd9c857e55535dc8143bd11d65b06d2c4`.
Human-cohort SHA-256:
`252fe593514adc5498011b55dbf62224cba55651025233fc1ed9235078bb95b8`.
The condition audit used the unchanged core checkpoint
`0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8`
and deterministic pair-selection seed 261390.

The inventory completed in 6.08 seconds, exact scoped-pair verification in
19.18 seconds, and the neural audit in 9.22 seconds on the Mac M5. The latter
used PyTorch 2.11/MPS with one CPU thread and twelve distinct TRAIN audio
identities. All raw candidate-pair and scope identities are retained in the
artifact owner; no generated chart was assigned a human label.
