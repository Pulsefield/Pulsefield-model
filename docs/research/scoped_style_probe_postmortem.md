# Scoped style probes: results and postmortem

The seed 17 audit, frozen-backbone readout comparison, and end-to-end pilot
completed on 2026-09-14. The multiscale readout improves validation loss over a
fresh original readout on the same frozen states. Training the temporal and
multiscale extensions from scratch produces only small, unstable differences
from the original architecture. Neither comparison repairs human Trill or LN
positive detection at the declared threshold. The combined architecture does
not outperform the original architecture in this pilot.

The failures differ by concept. Trill has weak ranking as well as missed
positives, including on human training labels. LN retains useful ranking but
predicts every human validation positive absent. Tech receives substantial
training exposure but has only one positive in the primary validation view.
These observations narrow the next diagnostics; they do not establish a single
root cause or justify rejecting the temporal and multiscale directions.

This report concerns the scoped style research classifier, not a Pulsefield V3
reference architecture. The [original seed 17 report](scoped_style_seed17_results_and_next_questions.md)
describes the earlier evidence-supervision experiment; the
[probe guide](scoped_style_probes.md) owns executable commands and model details.

## Questions and controlled comparisons

All models predict five independent concepts, each as absent, supporting, or
prominent: Jack, Stream, Trill, Tech, and LN coordination. Concepts may coexist.
Supporting and prominent describe expression strength rather than confidence.

| Experiment | Question | Comparison and fixed conditions |
| --- | --- | --- |
| A: saved-score audit | Did the historical models rank positives and distinguish Jack/Stream on the same input? | Reuse both historical checkpoints' saved validation logits; no training or forward passes |
| B: frozen-backbone readouts | Can a different readout use the historical encoder's contextual states more effectively? | R0: fresh original assessor; R1: fresh multiscale composition and local/ordered readout; identical cached encoder states, targets, and sampled updates |
| C: end-to-end pilot | Do timing coordinates and multiscale composition help when the encoder can adapt? | C0: original; CT: enhanced time; CM: multiscale; CTM: both; all initialized from scratch with matched common parameters, targets, and sampled updates |

B/C use zero evidence-loss weight. They do not test whether the original
auxiliary selector becomes useful with the new architecture. The temporal
package adds performed-time, local-pace, and local-redline normalization to
event, lane, and relation inputs. Multiscale composition retains contextual
hand states and three residual dilated-convolution outputs, then combines
local maxima and attended vectors with an ordered section summary. CM/R1
receive composition-span metadata without the enhanced temporal package.
CTM also supplies temporal metadata to its multiscale readout.

Identical parameter components are copied explicitly across arms. Added modules
have deterministic initialization, but their initial predictions differ.
Parameter counts also differ: these comparisons measure the complete module
changes, including capacity, rather than an isolated inductive bias at equal
parameter count. B's cached encoder has already read the complete review context;
an R1 gain cannot establish a purely local detector.

## Data, measurement, and execution

The source-group split and original event graphs are preserved. B/C select human
labels before machine fallback for Jack, Stream, Trill, and LN. Unresolved human
cells block fallback. Tech uses only effective human judgments with explicit
High confidence. The selected training population is 3,200 cells: 382 human and
2,818 machine. Sampling is uniform over concept, then represented source group,
then cell; it is not balanced over classes or annotation layers. All inputs use
1x playback rate.

| Concept | Selected training cells, human / machine | Primary validation absent / supporting / prominent |
| --- | ---: | ---: |
| Jack | 79 / 732 | 3 / 6 / 4 |
| Stream | 79 / 699 | 4 / 2 / 6 |
| Trill | 100 / 704 | 10 / 1 / 3 |
| Tech | 26 / 0 | 5 / 0 / 1 |
| LN coordination | 98 / 683 | 10 / 4 / 2 |

Primary validation contains **61 human cells across 18 source groups**. Historical
human validation has 79 cells across 26 groups; machine validation has 487 cells
across 53 groups. The primary view removes 18 human Tech cells that do not meet
the new confidence rule. These views must not be substituted for one another
when comparing losses.

Three-class NLL averages cells within each concept/group, then groups within
each concept, then the five concepts. Lower is better. Presence uses
$p_P=p_{\mathrm{supporting}}+p_{\mathrm{prominent}}$ with threshold 0.5.
Presence balanced accuracy (BA) averages group-weighted negative and positive
recall. Conditional strength uses $p_{\mathrm{prominent}}/p_P$ on every
reference-positive cell, including missed positives. Three-class argmax and
binary presence decisions are different: absent can be the largest single
class while the other two classes together exceed 0.5. Presence AUROC/AP assign
equal total weight to each source group; a missing binary class makes ranking
and BA unavailable. Primary Tech strength BA is therefore unavailable.

B/C ran on MPS with Python 3.10.20 and PyTorch 2.11.0, one CPU thread, seed 17,
batch size 16, AdamW learning rate 0.0003, weight decay 0.0001, dropout 0.1,
and gradient norm cap 1. Each arm spent five disposable updates measuring
throughput, then restored its initial parameters and optimizer. The runner
chose a common training count within the update and estimated time bounds.
B completed 272 updates and 4,352 sampled cells per arm; C completed 423 updates
and 6,768 sampled cells per arm. Both stopped at their planned common count,
with no failed arm. Neither stopped because convergence was established.

Checkpoints minimize primary human macro NLL at update zero, each 100 updates,
and the planned final update, with the earlier checkpoint retained on a tie.
B selects update 272 for both arms. C selects update 400 for C0, CM, and CTM,
and 423 for CT. Final and selected checkpoints are reported separately. No test
partition was evaluated, no threshold was tuned, and no multi-seed confirmation
was run. There was no fixed practical-effect threshold or numerical regression
tolerance for declaring a successful architecture; this is exploratory evidence.

## The historical audit changes the comparison

| Historical checkpoint | Primary human NLL, 61 cells | All-human NLL, 79 cells | Machine NLL | Joint Jack/Stream correct |
| --- | ---: | ---: | ---: | ---: |
| Style-only | 0.8084 | 1.0393 | 0.4952 | 8/12 |
| Style+evidence | 0.8289 | 1.0722 | 0.4832 | 7/12 |

A confirms that lower machine NLL from evidence supervision does not establish
better human classification. Both historical arms miss all four human Trill
positives and all ten human Tech positives. Trill presence AUROC is 0.45/0.50;
all-human Tech AUROC is 0.5635/0.5476, in style-only/evidence order. These scores
do not support a strong detector merely hidden by the threshold. On the much
smaller High-confidence Tech slice, the historical AUROC is 1.0/0.8; those
rankings involve just one positive and are not stable estimates of Tech quality.

The 12 matched Jack/Stream inputs include one neither-positive input, two
Stream-only, three Jack-only, and six both-positive inputs. The historical
style-only arm gets eight pairs correct, including five of the six coexistence
cases. Aggregate Jack/Stream scores alone did not reveal this joint behavior.

**Every new B/C arm has worse primary NLL than historical style-only's 0.8084.**
The historical pair completed 2,701 updates per arm with machine-only supervision,
selecting epochs 9/10 by machine validation. B reinitializes the assessor; C starts
the entire model anew. Supervision, exposure, initialization, and selection all
change. The historical comparison is a useful performance reference, not a
causal estimate of either architecture or human-inclusive training.

## A better frozen readout, with unresolved detection failures

| B metric at update 272 | R0: original readout | R1: multiscale readout |
| --- | ---: | ---: |
| Primary human macro NLL | 0.9350 | 0.8831 |
| Primary presence NLL | 0.6356 | 0.6036 |
| Primary conditional-strength NLL | 0.7528 | 0.7080 |
| All-human validation NLL | 1.0011 | 0.9564 |
| Machine validation NLL | 0.6920 | 0.6698 |
| Selected human training-fit NLL | 0.9652 | 0.9077 |
| Selected machine training-fit NLL, four represented concepts | 0.6693 | 0.6348 |
| Joint Jack/Stream correct | 5/12 | 6/12 |

R1 reduces primary NLL by 0.0520, or 5.6%, and improves all five concept NLLs.
It improves both training and validation fit under the frozen-state comparison.
However, R1 has 107,931 trainable parameters versus R0's 43,539. The experiment
does not separate the new aggregation path from that added capacity. R1 is also
worse than R0 at updates 100 and 200 before overtaking it at 272, so the endpoint
gain is not a consistent advantage throughout training.

The companion metrics prevent a stronger success claim. Jack presence BA rises
from 30.0% to 51.7%, but Stream BA falls from 62.5% to 50.0%: R1 predicts Stream
present on every validation cell. LN conditional-strength NLL worsens from
0.7607 to 0.7965 despite its lower three-class NLL. R1's one extra correct
Jack/Stream pair is a Stream-only case; it still misses all three Jack-only
cases. Trill, LN, and primary Tech positive recall remain zero for both arms.
The proposed short-local recognition and selectivity improvements were not
demonstrated.

## End-to-end changes have small or unstable effects

| C arm | Parameters | Best update | Selected primary NLL | Final primary NLL | All-human NLL at selected checkpoint | Machine NLL at selected checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0: original | 106,423 | 400 | 0.8971 | 0.8974 | 0.9531 | 0.7348 |
| CT: enhanced time | 110,397 | 423 | 0.8937 | 0.8937 | 0.9577 | 0.7127 |
| CM: multiscale | 170,815 | 400 | 0.8906 | 0.9512 | 0.9500 | 0.6971 |
| CTM: both | 175,941 | 400 | 0.9096 | 0.9145 | 0.9596 | 0.7222 |

Against C0, selected CT improves primary NLL by 0.0034 (0.4%), CM by 0.0065
(0.7%), and CTM worsens it by 0.0125 (1.4%). CM's loss increases from 0.8906 at
update 400 to 0.9512 at 423. Its conditional-strength NLL rises from 0.6842 to
0.8110 over that interval. The two observations establish endpoint sensitivity,
not a sustained overfitting trajectory or a diagnosed optimization failure.

| Primary concept NLL at selected checkpoint | R0 | R1 | C0 | CT | CM | CTM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jack | 1.2635 | 1.1590 | 1.2058 | 1.1826 | 1.0645 | 1.1513 |
| Stream | 0.7120 | 0.6563 | 0.8190 | 0.8003 | 0.8940 | 0.8439 |
| Trill | 0.9122 | 0.8587 | 0.8725 | 0.8517 | 0.9261 | 0.8746 |
| Tech | 0.7745 | 0.7449 | 0.7318 | 0.7169 | 0.7176 | 0.7981 |
| LN coordination | 1.0130 | 0.9963 | 0.8564 | 0.9171 | 0.8510 | 0.8802 |

CM's selected aggregate gain is dominated by Jack, while Stream and Trill NLL
worsen. CT improves four concept losses but worsens LN. All four C arms predict
Stream present on every primary validation cell, yielding 50% Stream BA. C0/CT
get 5/12 joint Jack/Stream pairs correct; CM/CTM get 6/12. For CM/CTM, all six
correct cases are both-positive inputs; every selective or neither-positive
case is wrong. A 50% joint score therefore does not demonstrate separation of
Jack from Stream.

At the common final update, the interaction contrast
$I=(L_{CM}-L_{CTM})-(L_{C0}-L_{CT})$ is +0.0330 after averaging concepts.
Per-concept values are Jack -0.0557, Stream +0.1484, Trill +0.0322, Tech +0.0367,
and LN +0.0036. Positive $I$ means timing helps more with multiscale composition
at that comparison point. It does **not** mean CTM beats C0: final CTM is worse
than C0, and CM's late deterioration strongly affects the contrast. No
reproducible beneficial interaction is established.

### How uncertain are the loss differences?

Define gain as comparator NLL minus candidate NLL. A paired source-group
bootstrap of selected-checkpoint predictions gives:

| Comparison | Gain | Conditional 95% interval |
| --- | ---: | ---: |
| R0 minus R1 | +0.0520 | [+0.0061, +0.0981] |
| C0 minus CT | +0.0034 | [-0.0449, +0.0503] |
| C0 minus CM | +0.0065 | [-0.0331, +0.0410] |
| C0 minus CTM | -0.0125 | [-0.0659, +0.0276] |

These post-run percentile intervals use 2,000 accepted resamples with NumPy
`default_rng(170914)`. Sort the 18 group IDs lexically, draw 18 groups uniformly
with replacement, retain paired cells and each drawn group's multiplicity, and
redraw a sample missing a concept. Average within concept/group before averaging
groups and concepts. The same resamples serve all comparisons. Inputs are the
saved `best/primary-human/scores.json` predictions, with cell, input, label, and
group identities checked for equality between arms.

B's interval supports a conditional loss improvement on this validation sample.
It excludes training-seed variation and checkpoint-selection uncertainty;
validation selected the checkpoints, and there is no multiple-comparison
adjustment. All C intervals include zero. None of these intervals constitutes
independent test confirmation or establishes convergence.

## Why aggregate loss is insufficient

### Trill remains poorly separated, including in training

Every selected B/C checkpoint misses all four primary Trill positives. Their
presence AUROCs are R0 0.25, R1 0.40, C0 0.325, CT 0.45, CM 0.20, and CTM 0.35.
Lowering the threshold could recover individual positives while adding false
positives; it cannot create good ranking that is absent from these scores.

The selected models also have zero recall on all 36 human Trill training
positives. Human training AUROC ranges from 0.308 to 0.499, whereas machine
training AUROC ranges from 0.582 to 0.673. Thus the failure is not limited to
unseen human validation sections. Differences in label-layer semantics or
source distributions, class imbalance, insufficient optimization, and input
representation remain compatible explanations. These are not matched
human/machine judgments of identical cells, so their gap does not measure
annotator disagreement directly.

The two High-confidence short positive Trill sections are both **training**
examples, not held-out localization evidence:

| Section duration and reference | R0 $p_P$ | R1 $p_P$ | C0 $p_P$ | CT $p_P$ | CM $p_P$ | CTM $p_P$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 382 ms, supporting | 0.128 | 0.091 | 0.265 | 0.243 | 0.252 | 0.261 |
| 1.884 s, prominent | 0.132 | 0.156 | 0.173 | 0.182 | 0.136 | 0.189 |

Every arm misses both at its selected checkpoint. The multiscale exports
include responses at four scales for these sections and the four positive
validation sections. No independently localized internal intervals were supplied.
Composition spans sit on already contextualized BiGRU states: neither those
spans nor attention weights establish a detected Trill episode, localization
accuracy, or causal attribution.

### LN retains ranking while losing the operating point

All selected B/C checkpoints miss the six human LN validation positives, but
their presence AUROC remains 0.85–0.90. In C, human training LN recall is also
zero despite AUROC 0.888–0.900. This is materially different from Trill's weak
ranking: there is a useful association in the scores even though the learned
probabilities do not cross the declared threshold.

The historical style-only model recalls all six validation positives, with
90% negative recall and AUROC 0.90. Reinitialization and the shorter, differently
supervised runs have lost that operating behavior. The evidence is consistent
with inadequate fitting of presence probabilities under the training mixture;
it does not show that LN relations are absent from the representation, nor prove
that threshold calibration alone would preserve strength and false-positive
performance. R1's worse LN strength loss is a concrete warning against such an
assumption.

### Tech is exposed repeatedly but has too little validation support

The completed B and C streams contain 831 and 1,332 Tech draws respectively,
covering all 26 eligible training cells. Their positive draws number 309 and
488. Uniform concept sampling therefore gives Tech substantial repeated
exposure despite its small dataset. A simple claim that the sampler never
showed positive Tech examples is inconsistent with the recorded stream.

At selected checkpoints, human Tech training-positive recall is 40%/50% for
R0/R1 and 60%/50%/20%/50% for C0/CT/CM/CTM. Training AUROC is 0.681–0.794.
There is partial training fit, but every arm misses the one High-confidence
validation positive, with AUROC between 0 and 0.4. CT has lower Tech NLL than C0
while ranking that positive below all five negatives. With five absent cells,
one prominent cell, and no supporting cell, a small aggregate Tech NLL gain
cannot establish improved positive recognition or strength discrimination.

### Class priors explain part, but not all, of the fit

A constant baseline estimated from the selected training labels uses class
frequencies averaged within source group, then across groups for each concept.
This matches the class distribution induced by the concept/group/cell sampler;
it uses no validation labels for fitting and no smoothing. Its primary NLL is
0.9510, all-human NLL 1.0033, and machine NLL 0.8005. All selected B/C checkpoints
beat its primary aggregate, but every new arm has worse Tech NLL than the prior's
0.6794.
For Trill, only R1 and CT slightly beat its 0.8627.

The selected training prior is 77.7% absent for Trill and 81.0% absent for LN.
Actual C draws contain 1,009 absent out of 1,345 Trill draws and 1,134 absent out
of 1,368 LN draws. Positive targets were encountered: 336 Trill draws and 234 LN
draws. This establishes both exposure and imbalance, not that either alone
caused failure. Human labels supply only 197 of the Trill draws and 204 of the
LN draws. Human precedence at matching cells does not make the overall training
objective predominantly human-supervised for those concepts.

## Runtime and provenance lessons

| Arm | Charged seconds | Final diagnostics seconds | Budget seconds | Overshoot seconds |
| --- | ---: | ---: | ---: | ---: |
| R0 | 951.61 | 343.50 | 900 | 51.61 |
| R1 | 967.17 | 342.05 | 900 | 67.17 |
| C0 | 1,325.60 | 278.13 | 1,800 | 0 |
| CT | 1,209.63 | 278.71 | 1,800 | 0 |
| CM | 1,313.71 | 347.64 | 1,800 | 0 |
| CTM | 1,299.10 | 370.70 | 1,800 | 0 |

Charged time includes preparation, disposable updates, restoration, training,
validation, and final diagnostics. Shared work is charged to every arm, so
summing arm charges is not elapsed wall time. The backbone cache is separate:
1,040 inputs, 170.44 seconds, and 54,337,056 recorded payload bytes (51.82 MiB).

B's diagnostic reserves were 242.14/215.62 seconds, versus actual costs of
343.50/342.05. The cell-count-scaled forecast underestimated export and
evaluation costs; B exceeded its caps even though both arms respected the same
272-update stop. C's estimates also varied substantially from actual diagnostic
cost, but all four arms finished within budget. C retained 474–590 charged
seconds of headroom because its common update count was fixed from conservative
initial timing estimates. The 1,000-update ceiling was never a promise that
the models would receive 1,000 updates, and the unused time was not reassigned.

The runner completed its paired updates, checkpoints, validation reports,
training-fit reports, and multiscale inspection exports. The remaining failures
concern recognition and B's budget estimate. Sampled
process/device memory counters are overlapping maxima and can miss transient
peaks; they do not identify a memory bottleneck in these experiments.

A/B record HEAD `f802ad4084608bc0e88a23a329942c11cc25fc7a`, before the probe
implementation was committed. C records
`6a2cdc11f7921d0cc968bad1ddc47c1f47fcae4b`. All three saved source manifests have
the same fingerprint, and every Git-tracked file in them matches `6a2cdc1`.
The dependency lock is ignored by Git; each run archives a local copy.
Thus the A/B recorded HEAD alone does not reproduce those runs; the archived
source and dependency snapshot is necessary. C's executable sources are
recoverable from its commit, but its exact dependency resolution also requires
the archived lock. This provenance qualifies the runs as exploratory rather
than retroactively establishing a clean, predeclared confirmation experiment.

## Interpretation and focused follow-up

The strongest positive result is **better finite-budget readout loss on fixed
historical encoder states**. It does not validate short-local Trill recognition,
improved LN operating behavior, or reliable Jack/Stream selectivity. The C pilot
does not yet establish a useful temporal/multiscale interaction, and the old
model remains the stronger primary-loss reference under unmatched training
conditions. Neither broad parameter expansion nor discarding the proposed
features follows from this evidence.

The next diagnostics should address the observed failure modes:

1. **Establish training fit and exposure before a larger architecture grid.**
   Use a bounded set of clear human Trill/LN/Tech positives and negatives to
   verify loss contribution, positive-probability movement, and ability to fit
   those targets. The recorded draws rule out zero exposure, but not competing
   gradients, weak per-cell exposure, masking defects on real inputs, or
   insufficient optimization. Compare the achieved fit with the existing
   training predictions before interpreting validation generalization.
2. **Separate Trill representation from the supervision mixture.** A controlled
   change in human/machine weighting, with architecture held fixed, can test
   whether the human ranking gap responds to the training objective. A failed
   bounded fit test would instead prioritize real-input tensor, relation, and
   optimization inspection. Neither branch is resolved by aggregate NLL.
3. **Treat LN probability fitting separately from ranking.** Inspect held-out
   score calibration and presence/strength behavior while preserving a fixed
   operating point for the architecture comparison. Thresholds chosen on these
   same six positives would be an operating-point exploration, not validation
   evidence of an architecture gain.
4. **Strengthen the comparison only after fitting is credible.** A capacity-
   matched original readout can test whether B's gain needs the multiscale
   path. A longer common-update comparison with a separately fixed budget,
   practical-effect criterion, and regression bounds can test C beyond its
   unstable endpoint. Seeds 29/43 and more independent High-confidence Tech
   validation positives are needed before a strong generalization claim.
5. **Measure the complete diagnostic workload when setting budgets.** Include
   best/final evaluation, training-fit exports, and position plots in a
   representative timing measurement. Preserve common update counts and log
   source cleanliness and the dependency snapshot explicitly.

These are proposed diagnostics. No further training, threshold search, new
annotation, or model adoption is part of the completed experiments.

[Source-action representation directions](source_action_representation_directions.md)
develops the broader architectural question: learning action-block structure
with direct access to source, local, and contextual representations, while
using scoped style as a semantic readout. It preserves the findings and limits
of these results as the empirical baseline.

## Evidence locations and reproducibility

All run paths below are relative to `artifacts/scoped-style-modeling/`. They
identify local derivative evidence and may be absent in a fresh clone; the
measurements and limitations needed to read this report are reproduced above.

| Evidence | Run path | SHA-256 of `summary.json` |
| --- | --- | --- |
| Saved-score audit | `probe-a-v2/` | `e6ffbb17a10529a4a2453f29ec062bc0b1dc383acf9461c7afab3d5efa0cd75a` |
| Frozen readout comparison | `probe-b-v1/` | `4959e59bcb2b672e53645036cbf21792a7be0e45aed536d573cccc717a83cf80` |
| End-to-end pilot | `pilot-c-v1/` | `5abced97ccaa2f557f1e34402f02dabc8bd50620c47afaa747579d2bc6cebdc5` |

The common source fingerprint is
`cec611d975e9a7ac34feecdb54a38964bf292815fd3552e04187b9c4dfbf6153`.
The prepared cohort hash is
`252fe593514adc5498011b55dbf62224cba55651025233fc1ed9235078bb95b8`,
split hash
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`,
and target-policy hash
`8823765a934e90323f6b5d247cd327880b122e3ef770b2daa6bb1665ba6ac081`.
The frozen historical backbone checkpoint hash is
`d16fa2aee5b7e13c3c034eddbe15c80d31693227161f95240005ee1cadae59fa`.

For B/C, `config.json`, `source.json`, `cohort.json`, `throughput.json`,
`sampled-cells.json`, `updates.jsonl`, and `validation-history.json` identify
the inputs and execution. Per-arm `best/` and `final/` reports retain checkpoint
hashes, logits, labels, and group/input identities. Training-exposure counts
above use only the first `common_updates * batch_size` entries of the recorded
sample stream, excluding disposable updates and the unused pre-generated tail.
The prior and bootstrap are post-run calculations over saved labels and scores;
they add no model forward passes or selection opportunities. Tiny differences
between checkpoint-selection NLL and exported NLL are below $10^{-7}$ and do
not affect the reported four-decimal comparisons.

The owning implementations are the
[target policy and sampler](../../src/pulsefield_model/research/scoped_style_modeling/probe_data.py),
[model comparisons](../../src/pulsefield_model/research/scoped_style_modeling/probe_model.py),
[runner and time accounting](../../src/pulsefield_model/research/scoped_style_modeling/probes.py),
[assessment metrics](../../src/pulsefield_model/research/scoped_style_modeling/metrics.py),
and [ranking and matched-input reports](../../src/pulsefield_model/research/scoped_style_modeling/probe_metrics.py).
The [probe tests](../../tests/research/scoped_style_modeling/test_probes.py)
cover common initialization, target isolation, sampled-loss multiplicity,
padding/mirror behavior, cache identities, metric ties, and the bounded runner.
Those software checks do not establish semantic correctness of predictions.
