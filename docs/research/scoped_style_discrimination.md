# Difficulty-stratified style learning and scoped generation

Supplementary factual style discrimination improved a matched source-learning
baseline, but did not establish reliable control of generated organization.
The candidate does not replace the selected runtime model. Its most revealing
failure is a common-prefix trill request: two of three absent/prominent forks
produced exactly the same target rows, while the third still mainly repeated
groups in pairs instead of sustaining the requested exchange.

The model already has the small
[condition/history modulation](row_condition_interactions.md) inside R1. This
comparison changes its learning signal, with no additional inference network.
H still owns head times. R1 retains head cardinality, TAP/LN selection, release
subsets and columns; full audio remains available to both timing and placement.

## Matched supplementary learning

Both arms started from the 4.60M-parameter modulated-128 checkpoint. Audio and H
were frozen; R and R1 trained. Each arm received the same 800 genuine examples
over 400 updates: one human-assessed style cell and one ranked population scope
per update. The human pool contained 125 explicit absent/prominent cells after
excluding four Low-confidence judgments and retaining only consistent labels
with local difficulty in [2,6). The population pool contained 884 eligible scopes.

Sampling gave each of five concepts and two label levels forty draws, cycling
available integer difficulty bins within each group. Population draws were
100 per local difficulty bin. The actual draws covered 102 distinct human cells,
277 population scopes and 326 charts on 323 audios. Both arms shared every source
identity and its order. Empty difficulty/style cells remained empty; no new style
labels were inferred from unannotated charts.

The source arm optimized R timing NLL plus deployed R1 row NLL per second,
averaged over the two examples. The aligned arm added the existing
[condition alignment loss](../../src/ensomi_model/research/controlled_audio_continuation/condition_alignment.py)
to the human example. It scored the same complete factual row sequence under
its observed style and the opposite absent/prominent value, changing only that
field on its annotated scope. Difficulty, LN request, other known styles and
physical history stayed fixed. Supporting and unknown fields were not negatives.

The two logistic terms use summed neural row log probabilities relative to the
frozen initial checkpoint, scale .1 and weight one within the human example's
loss. This adapts
[Condition Contrastive Alignment](https://arxiv.org/html/2410.09347v1).
Explicit binary flips are not independent marginal samples, and row-factor scores
are not the full H/R/row trajectory likelihood. No exact density-ratio or guided
sampling equivalence is claimed. Every supervised next-row target retains its
own genuine history; generated prefixes receive no transplanted source suffix.

Both arms used AdamW, learning rates 3e-5 for inherited parameters and 3e-4 for
condition, composition, preview and modulation paths, weight decay .0001 and
gradient clipping at 1. Neither arm used a trajectory-kernel training loss.
Four-update integrations preceded fresh main fits. On the 24 GiB M5 Mac, the
main fits completed in 255 and 304 seconds with unchanged audio/H hashes.
Conditional modulation received finite nonzero alignment gradients at all eleven
scheduled checks. Peak sampled footprints were 7.13 and 7.29 GiB.

## Matching the evaluation condition to its reference

An inherited style comparison requested whole-chart difficulty and LN fraction,
but measured distance to a local source segment. For the Tech reference, those
difficulty values were 5.803 and 3.414; LN fractions were .2691 and .0118. A
generator can respond to the supplied request while moving away from that local
reference. Such a distance does not isolate style fidelity.

The corrected comparison adds factual local difficulty and LN-fraction overrides
on exactly the measured source scope, preserving other fields and the three
original seeds per case. Initial, source-only and aligned endpoints all use
that same schedule. The change was specified from code inspection during fitting,
before the aligned endpoint or aggregate comparison existed. Historical outputs
remain available; the revised evaluation is reported as exploratory rather than
an unchanged preregistered experiment.

The metric is the
[physical short-block kernel U-statistic](physical_trajectory_matching.md),
averaged over three draws per source scope. Smaller means closer to the empirical
reference distribution, not more playable or semantically correct.

| Matched source scope | Initial | Source-only | Plus style alignment |
| --- | ---: | ---: | ---: |
| Tech | .04077 | .04405 | .03563 |
| Jack organization | .02030 | .02447 | .01276 |
| Stream organization | .00676 | .01157 | .01135 |
| Trill organization | .12306 | .14830 | .12291 |
| Mean | .04772 | .05710 | .04566 |

Alignment improved the matched source-only mean by .01144, about 20%, exceeding
the .005 and 10% progress gates. Relative to the initial checkpoint, improvement
was only .00206, about 4.3%. The aligned stream result was worse than the initial
one, and trill distance was nearly unchanged. The comparison does not isolate
rebalancing itself as the cause of the source-only regression: that arm also
continues factual learning without the initial candidate's trajectory objective.

The unchanged five-case difficulty test used low/high requests and three future
seeds from common committed prefixes. Difficulty uses the repository's scoped
strain readout, rather than assigning a whole-chart rating to an arbitrary crop.
Mean absolute error was .73150 initially,
.76817 after source-only learning and .73811 with alignment. For the fast-single
low request 2.190, means were 3.689/3.734/3.642. For the slow-chord high request
3.500, they were 2.750/2.583/2.514. Supplementary learning therefore did not
recover the unused difficulty response identified in those examples.
Restored-range and LN-error regression allowances passed; passing an allowance
does not make those absolute control errors acceptable.

## Factual discrimination did not establish a controllable motif

Reserved source-condition score gaps improved strongly for jack and stream:
factual-minus-opposite neural row log probability changed from 6.68 to 12.73 and
8.21 to 18.30 relative to source-only learning. Tech became more negative, from
-.92 to -3.43, and trill stayed small, from .094 to .195. These are four factual
sequences on their own histories, not generated style accuracy.

The common-prefix diagnostic reused a frozen 3,336-row neutral prefix and the
actual H plan. It requested trill absent or prominent on [288156,290040) ms,
at fixed local difficulty 5.076 and unchanged LN fraction, using three matched
future seeds. Other style fields were unspecified. Each endpoint rebuilt its
own caches from identical committed rows. All twelve continuations preserved
their prefixes and H times and completed with actual LN endpoints.

The aligned endpoint changed zero, fifteen and zero of the twenty-four target
rows across its three absent/prominent pairs. The source-only endpoint changed
one, zero and zero. In the second source-only draw, both requests produced exact
`A,A,B,B` repetitions, with `A=[23]`, `B=[01]`, at 78–79 ms row gaps. The aligned
prominent draw interrupted that pattern but still mainly repeated each group
twice. Changing fifteen rows did not establish sustained A/B exchange. A hard
ban on repetition would also erase legitimate jack patterns; the missing behavior
is choosing the appropriate relationship under control.

Lens rendered sixteen matched-style and common-prefix contexts into fifty pages.
Focused review viewed sixteen pages and read the full target action sequence for
the decisive fork. In matched seed-zero views, the jack examples retained
repeated chord membership, and stream retained moving singles/chord accents with
occasional same-column repetitions. Tech retained irregular H with long stretches
of complementary pairs. Trill views remained mixed chord/single flow or repeated
groups. No generated human annotation, listening judgment or human playtest was
claimed. The required common-prefix semantic improvement was not established;
the six native expansion cases were therefore not run.

## Consequence for control learning

Difficulty-stratified exposure is useful coverage, but this recipe is insufficient
as the main repair. Improving factual condition scores does not directly teach
how to change an ongoing generated arrangement. Range-level labels also do not
mean that every row instantiates their named relation. Sparse coverage and the
learning objective remain competing explanations; this test does not prove an
architectural impossibility or justify simply increasing the loss weight.

The next difficulty question is more directly posed on common physical prefixes:
can full R1 learning produce distinct requested outcomes using real same-H
arrangements to supply empirically supported target ranges? Different source
prefixes do not prove that both targets are attainable from every common state.
Source imitation must retain its
own histories; the generated branches need outcome-level credit. This differs
from the earlier limited update of difficulty-input columns. Style preservation
must be checked independently, and unknown style labels cannot be invented for
the arrangement pairs.

H must participate when a control requires different timing. A fixed-H negative
style request is not automatically feasible, particularly for timing-dependent
Tech structure. The trill diagnostic has a concrete source arrangement on that
H, so its missed exchange cannot be explained by timing incompatibility alone.
These ownership distinctions remain essential when extending the learning signal
to the complete audio-to-chart system.

Executable source: `0591f905809876d7dc4f41ebf1217ba4c2c31381`.
Artifact owner: `20260926-scoped-style-discrimination-r1-v1`.
Frozen panel SHA-256: `cab2e71e39c1e365f15c431b06cb7ad321af05977cbcd3e9edc8c8422d4f430f`.
Source-only checkpoint: `95a55c34b26214c06f6b8ec9c9f6cde29144d854b1214c80296af58efb084c07`.
Aligned checkpoint: `a867547cf39d1284ca66daaca83e7f58f70851c3e8706c9971a1e7a665790214`.
All 150 generated qualification, matched-reference and style-fork exports
completed. Concurrent CPU qualification overlapped part of the aligned MPS fit;
those durations are resource records, not inference benchmarks.
