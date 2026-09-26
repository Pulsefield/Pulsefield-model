# Balanced condition learning and generated-history stability

Balanced source exposure plus difficulty-condition alignment improved factual
condition discrimination, but did not adequately improve generated control or
LN organization. Neither fitted endpoint replaces the reference model. A
focused trill diagnosis located a large preference reversal in R1's learned
history path, even when its three most recent rows matched the source.

This study follows the corpus evidence in
[control condition learning](control_condition_learning.md). It tests whether
better exposure and explicit condition learning solve the weak scoped difficulty
response of the 4.58M-parameter full-audio model. Difficulty readouts, conditional
likelihood and legal export are intermediate measurements; generated organization
remains the quality criterion.

## Matched learning comparison

Both arms started independently from core2500. Full-song audio and H were frozen;
R and R1 trained, including existing condition, composition and history paths.
No network was enlarged. H continued to own timestamps; R1 retained head
cardinality, TAP/LN choice, release subsets and columns. Training used each
source's genuine history and complete LN replay.

| Fitting pool | Available examples | Source examples drawn per arm |
| --- | ---: | ---: |
| Same-audio, identical-H arrangement pairs | 128 pairs on distinct audios | 1,200 |
| General ranked population | 1,024 scopes | 600 |
| Genuine human-assessed style cells | 251 cells | 600 |

The population contained 256 distinct-audio charts in each whole-star bin 2–3,
3–4, 4–5 and 5–6. Whole-chart bins selected coverage; actual scoped difficulty
supplied the target. Paired arrangements had local difficulty in 2–6, difference
at least .75 and LN-head fraction at most .1. Their exact H times matched, but
their own row histories remained separate. All designated qualification, native,
condition-audit and style-guard audio identities were excluded from fitting.

Sampling balanced difficulty together with H rate and factual materialization
attributes. Inverse-square-root cell weights had a final probability cap of four
times uniform. Human labels kept their original native scopes, independent
concepts and unknown fields. After exclusions, prominent fitting cells numbered
10 jack, 20 stream, 9 trill, 3 Tech and 5 LN coordination. Balancing cannot create
the missing semantic coverage.

Both arms received exactly the same 2,400 positive examples over 1,200 updates
and 703 unique charts. Source loss combined R timing NLL and deployed R1 row NLL,
normalized by scope duration with a one-second minimum denominator. The aligned
arm additionally contrasted each paired source scope under its correct difficulty
and its partner's difficulty, relative to frozen-reference row scores. The
[alignment helper](../../src/ensomi_model/research/controlled_audio_continuation/condition_alignment.py)
uses two logistic terms with scale .1. It does not prescribe monotonically
larger chords at every row or assign invented style negatives.

The objective adapts
[Condition Contrastive Alignment](https://arxiv.org/html/2410.09347v1).
Empirical paired negatives and a row-factor score do not establish an exact
independent-marginal density-ratio or full-trajectory likelihood interpretation.
Training used source-global LN requests and local difficulty overrides. Human
style fields retained independent scopes. The native test also used local LN
overrides and global difficulty; those condition-clock distributions were not
explicitly matched by this fit.

## Condition discrimination improved more than generation

A reserved diagnostic scored 24 genuine arrangements from twelve identical-H
pairs under correct and swapped difficulty. It used neural row probabilities on
each chart's own history, excluding external recovery preferences and LN feedback.

| Correct condition preferred | Core | Balanced | Aligned |
| --- | ---: | ---: | ---: |
| Lower-difficulty arrangements | 12/12 | 12/12 | 10/12 |
| Higher-difficulty arrangements | 2/12 | 5/12 | 10/12 |
| Total | 14/24 | 17/24 | 20/24 |

Generation qualification used five fixed-H cases, two scoped difficulty requests
and three future seeds per case: 30 continuations per endpoint. Every endpoint
rebuilt its temporal caches from the same committed core-generated prefix;
trained history weights never reused core hidden caches. Published rows and
entering LNs stayed fixed. Difficulty was evaluated separately in the override
and restored ranges. LN error used its own unchanged global scope.

| Generated measurement | Balanced | Aligned | Required aligned result |
| --- | ---: | ---: | --- |
| Mean absolute scoped difficulty error | .784 | .716 | Improvement ≥ .20 |
| Mean high-minus-low difficulty | .194 | .313 | ≥ .40 |
| Low-request mean absolute error | 1.092 | .873 | Regression ≤ .15 |
| High-request mean absolute error | .476 | .560 | Regression ≤ .15 |

The .068 error improvement and .313 response gap missed both primary criteria.
The high-request stratum regressed by .083, within its guard. Whole-LN and runtime
regression guards passed. Starry Jet's low-request restored-range error rose from
.121 to .383, exceeding the .25 regression allowance by .013. These are results
on five cases with three seeds, not a population estimate or an impossibility
bound for either requested arrangement.

The slow-chord case illustrates remaining action freedom: mean aligned output
difficulty was 2.253 for request 2.0 and 2.576 for request 3.5. Lens showed more
three-key chords in the higher request, but the response remained too small.
The fast-single case still overshot its low request: 3.575 against 2.190.
Conditional ranking can improve without making sampled continuations satisfy
the requested scope.

## Generated organization

All 60 qualification continuations, 16 reserved style-guard charts and 12 native
charts completed and reparsed after export. Beatmap Lens produced action and LN
articulation tables for 75 contexts and 200 rendered pages. Focused inspection
viewed 37 pages and read the relevant complete action tables; this was not an
exhaustive visual review of every page or a human playtest.

Both arms retained recurring chord groups in the source-H jack guard. The aligned
stream guard retained moving flow but often followed a chord with one of its
members 75 ms later. The Tech guard retained its irregular H sequence by
construction while changing many slower pulses into two-key layers. These
observations do not assign source human labels to generated outputs. Guard
requests used true global difficulty, which can exceed the local source passage's
readout; exact local density matching was not required.

LN organization remained inadequate. The following counts use LNs whose heads
fall in the review range and their full endpoints. Interior H means a head time
strictly between an LN's head and tail. These describe the tested passages, not
a universal requirement that all LNs end on H.

| Review range | Version | LNs | Median duration, ms | LNs spanning interior H | Tails on H |
| --- | --- | ---: | ---: | ---: | ---: |
| Miraie, 162686–166686 ms | Source | 28 | 183 | 7 | 28 |
| Same | Aligned low | 41 | 93 | 3 | 4 |
| Same | Aligned high | 54 | 91.5 | 8 | 1 |
| Starry Jet, 196574–206575 ms | Source | 38 | 209 | 15 | 35 |
| Same | Aligned low | 41 | 114 | 6 | 12 |
| Same | Aligned high | 40 | 122.5 | 6 | 13 |
| Shippaisaku, 84687–95276 ms | Source | 111 | 111 | 30 | 100 |
| Same | Aligned low | 125 | 111 | 46 | 50 |
| Same | Aligned high | 137 | 111 | 48 | 39 |

In Starry Jet, the source's sustained lane under moving shorter actions was
largely replaced by isolated short LN/tap combinations. In Shippaisaku, matching
the 111 ms median concealed much less coordinated tail placement and more
overlapping holds. LN amount and duration medians alone are insufficient quality
objectives. R can require an early release, while R1's starts and release subsets
determine the occupancy and ages seen by subsequent R queries.

## A history-dependent transition failure inside R1

The reserved trill source alternated `[01]` and `[23]` at 78–79 ms intervals in
288156–290040 ms. Each finger returned after approximately 157 ms. The aligned
generated chart formed these groups, but repeated the previous group eight times,
including four successive `[23]` rows. Its only known human style field was
prominent trill; other styles were unknown. Repetition is not intrinsically
forbidden, but this output did not sustain the source's alternating organization
and changed the repeated-finger burden substantially.

On the genuine source history, the model preferred the complementary group over
the previous group at all 23 tested transitions. On its generated history, it
preferred repetition at six of the eight observed repeats. Lower temperature
would reinforce several of these choices rather than repair them.

Exact two-TAP alternatives isolate placement because both have identical head,
LN-start and release counts; count-family normalization cancels. At 288705 ms,
both histories ended in the same three rows, `[01], [23], [01]`, with no occupied
lanes. Positive log-odds favor repeating `[01]` over alternating to `[23]`.

| Repeat-minus-alternate log-odds contribution | Source history | Generated history |
| --- | ---: | ---: |
| Main joint-row head | −4.257 | +2.510 |
| Head-routing residual | +.303 | +.062 |
| Release residual | 0 | 0 |
| Learned frontier2 energy | +.049 | +.024 |
| Neural row law | −3.905 | +2.596 |
| External recovery preference | −.938 | −.938 |

Across the six repeats preferred by the deployed policy, the main joint-row
contribution ranged from +1.771 to +3.875; frontier2 contributed only +.019 to
+.090. This locates the dominant preference in the row policy for this example,
not in skeleton cardinality, LN amount feedback or the frontier residual.

A diagnostic substitution isolated the history input further. Audio, controls,
H preview, occupancy and legal masks were identical between the two queries.
Replacing only the generated temporal-history activation with the source
activation changed neural log-odds from +2.596 to −4.048. Replacing only exact
replay features left it at +2.592. The reverse history substitution changed the
source query from −3.905 to +2.616.

These are intermediate-input diagnostics, not valid alternative physical
trajectories or training labels. They establish sensitivity of this decision to
the learned history representation; they do not estimate the share of all
long-form failures caused by R1 or prove one memory architecture superior. Recent
correct rows need not erase an earlier generated history's influence.

Frontier2 remains a learned candidate-row energy. Its passively advanced next-H
clocks and possible earliest release do not unroll future actions. NLL training
has not made it a calibrated implementation of the canonical player-response
comparison in the [V3 gameplay formulation](../formulation/gameplay-state.md).

## Native generation and scoped control changes

Native testing used three audios, two endpoints and two schedules: static
difficulty 3/LN fraction .2, and the same defaults with difficulty 4.5/LN .6 on
64000–96000 ms. The runtime update followed publication through 63999 ms.
Corresponding arms shared seeds; static and switch modes used different seeds,
so their within-arm difference does not isolate the control change.

H timestamps were equal between endpoints in all six matched settings.
Differences therefore arose downstream of H. Static whole-chart difficulty
remained too high:

| Audio | Balanced | Aligned | Request |
| --- | ---: | ---: | ---: |
| Zenithfall | 3.985 | 3.895 | 3 |
| Hysteric | 3.717 | 3.696 | 3 |
| Take | 4.584 | 4.205 | 3 |

Aligned switch runs had these field-specific scope readouts:

| Audio | Difficulty before / override / restored | LN fraction before / override / restored |
| --- | --- | --- |
| Zenithfall | 3.024 / 3.650 / 3.690 | .199 / .631 / .185 |
| Hysteric | 3.206 / 4.406 / 3.543 | .198 / .713 / .166 |
| Take | 4.009 / 4.178 / 4.184 | .185 / .774 / .222 |

The local LN request overshot substantially in Hysteric and Take. Reconstruction
from committed rows found the amount controller's lower log-odds cap active
before 26.5% and 44.3% of H rows in those override scopes. Feedback updates by
`(requested_fraction * heads - LN_heads) / 8`, bounded to ±2 and reset per
effective LN episode. Saturation limits corrective authority; this trace does
not identify the sole cause or justify raising the bound without checking local
organization.

With the model loaded and Mel cached, first 30 rows took .197–.388 s. The slowest
measured eight-second publication window took .346 s; full songs took
3.16–7.46 s. This excludes waveform decoding, Mel extraction and checkpoint
loading. It shows generation headroom on this panel, not end-to-end cold-start
latency or a worst-case scheduler guarantee.

## Learning implication

Balanced coverage remains useful, but the tested alignment objective is
insufficient for adoption. The next question is how to improve complete
continuations under the model's own history while preserving multiple legitimate
arrangements. Further factual condition-ranking improvement alone is inadequate.

[Generative Adversarial Imitation Learning](https://arxiv.org/html/1606.03476v1)
offers a relevant analogue: learn from states and actions visited by expert and
generated trajectories without needing an expert next action at each generated
state. A conditional continuation comparison could use ranked chart/audio
examples, identical supplied H and scoped controls, replaying each trajectory
under its own actions. This would adapt imitation learning to finite native-time
choreography; it would not define canonical player cost or label every generated
alternative bad.

[Professor Forcing](https://arxiv.org/html/1610.09038v1) instead aligns training
and sampling dynamics. The history probe motivates considering that mismatch,
but hidden-state distribution matching alone would not guarantee physical
organization. A continuation objective should observe actual attacks, occupancy
and release relationships at multiple short scales. It must not attach source
next-row labels to changed generated prefixes. A learned critic would remain a
proxy, vulnerable to density shortcuts and sparse style coverage, requiring
independent Lens inspection and scoped-control qualification. No such fit is
reported here.

## Reproduction boundary

Executable source: `b2a53511b876484e14e5b657635e0d0cb9487d6f`.
Artifact owner: `20260926-balanced-condition-r1-v1`.
Reference checkpoint SHA-256:
`0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8`.
Balanced checkpoint:
`d3b4820842f65f581b54b1e49bdab957821bc1877f00155beda7b900c85221ea`.
Aligned checkpoint:
`67ef81fbb9fb2ecfc5ca19d8fc6767040b846e1ba8c41d4b8d61067abc08ee26`.

Ranked manifest SHA-256:
`4cea2672387b6293a4da0846be479d8bd9c857e55535dc8143bd11d65b06d2c4`.
Human cohort:
`252fe593514adc5498011b55dbf62224cba55651025233fc1ed9235078bb95b8`.
Panel selection and fit draws used seed 261410; guard selection used 261411.
Exact inputs and draw records remain local artifacts, not distributed datasets.

Both fits used PyTorch 2.11/MPS on a 24 GiB Mac M5, one CPU thread per process,
AdamW with 3e-5 inherited and 3e-4 composition/control/preview learning rates,
weight decay .0001 and gradient clipping at 1. They trained 3,187,773 parameters.
The fits overlapped in wall time, completing in 819 and 906 seconds. Sampled
physical footprints peaked at 6.79 and 7.90 GiB; MPS driver allocations at 2.21
and 3.32 GiB. These ledgers overlap and must not be added. Frozen audio/H hashes
stayed unchanged; all 1,200 source-draw records matched between arms.
Eight-update integration runs were discarded before the main fits.
