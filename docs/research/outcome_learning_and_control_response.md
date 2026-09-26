# Outcome learning and R1's difficulty response

Adding a whole-chart control objective to R1 did not improve the primary
fixed-timing difficulty diagnostic. It partly repaired an underfilled chord
chart, while the mostly single-note chart remained too dense. A subsequent
fixed-prefix audit found weak neural responses to difficulty requests alongside
large differences between source and generated histories. The fitted models
are research endpoints, not a selected playable release.

The [controlled model](controlled_audio_continuation.md) keeps head cardinality,
TAP/LN choices, release subsets and placement inside R1. This experiment changed
the learning objective without adding parameters or changing that ownership.

## Learning from the deployed distribution

Training and sampling previously used different row distributions. Deployment
applies empirical recovery preferences and a bounded, projected LN-amount
correction after the neural row scores. The correction depends on the actual
committed prefix and the effective LN-control episode.

The new `sampling.replay_row_scores` reconstructs that deployed distribution
$q_\theta$ on a completed trajectory. It replays feedback from the full prefix,
then scores only the requested interval. It neither resets open holds at a crop
boundary nor attaches source next-row labels to a generated history. CPU float64
preference arithmetic remains differentiable back to MPS model weights.

The existing LN projection had valid inference probabilities but undefined
derivatives for empty count groups. Finite unused normalizers remove those
NaNs without changing supported probabilities. Eighteen focused tests covered
sampling/rescoring agreement, finite gradients, scoring partitions, scoped
controls and CPU/MPS agreement. A complete 1,123-row Miraie generation remained
identical after the implementation change.

Audio encoding, H timing and release-timing parameters were frozen. Active-LN
audio cues were disabled. Release times still react to generated LN state;
freezing their weights does not fix their sampled times. On a fixed sampled
trajectory, however, those factors have no direct parameter derivative. The
parameter-dependent trajectory score is therefore the sum of R1 row log
probabilities. Training shared audio or timing parameters would require their
additional score terms.

## Objective and measurement scope

The study adapts sequence-outcome score gradients, as used in
[minimum risk training](https://aclanthology.org/P16-1159/), to completed generated
beatmaps. It uses an on-policy score-function estimator rather than a
renormalized finite candidate-set objective.

For generated whole-chart difficulty $D$, LN-head fraction $\rho$ and requested
values $D^*,\rho^*$, the cost is

$$
c(Y)=\max(|D-D^*|-.35,0)^2+
100\max(|\rho-\rho^*|-.03,0)^2.
$$

Difficulty uses the repository's native-rate mania star calculation. It is a
control proxy, not a measure of musical fit or a complete playability response.
The deadbands allow approximate control. Source imitation remains an independent
term, and generated arrangements require separate Lens inspection.

Three independent complete continuations are sampled at the same requested
controls. Each cost advantage subtracts the other two candidates' mean cost:

$$
a_i=c(Y_i)-\frac{1}{2}\sum_{j\ne i}c(Y_j).
$$

Every candidate starts at BOS and runs to the actual audio end, with all holds
resolved. This avoids defining a duration-dependent difficulty target by
artificially closing a hold at a training-window edge.

If the song has $M$ eight-second clock intervals, sampling interval $J$ uniformly
gives the surrogate

$$
L=L_{\rm source}/T+
\frac{2}{3}\sum_{i=1}^3 a_i M
\sum_{t\in J}\log q_\theta(y_{it}\mid Y_{i,<t},X,C,K).
$$

Here $T=(\mathrm{duration\_ms}+1)/1000$ covers the scored native clock including
its terminal millisecond, $X$ is complete audio, $C$ is the control
schedule and $K$ is the fixed H preview. Source NLL is also estimated with a
uniform interval and its corresponding importance weight. Costs, sampled
actions and the leave-one-out baselines are detached; log probabilities are
differentiable on each candidate's own history. The interval retains its full
prefix and actual LN/feedback state.

The outcome term estimates the gradient of whole-range expected cost. An initial
eight-update integration run additionally divided that term by $T$, which would
optimize cost per second and underweight longer songs. The duration factor was
removed before the main outcome fit. The source-imitation arm was unaffected.
No intermediate outcome checkpoint was selected by quality or likelihood.

## Paired training and generation

Both arms started from the same 4,583,985-parameter model; 2,737,331 R1 parameters
were trainable. A frozen panel contained twelve ranked TRAIN charts from distinct
audio groups, six with LN fraction at most .2 and six at least .5. Source ratings
were 2.726–4.327 and audio lengths 89.1–176.9 seconds. Prior diagnostic audio
identities were excluded from this fine-tuning panel. These remain development
experiments, not an unseen population benchmark.

Each arm made 128 updates on identical source chart/interval draws. Both optimized
deployed source row NLL; only the outcome arm sampled the additional three
continuations and received their outcome gradient. Source whole-chart difficulty
and LN proportion were requested throughout each song, with style unspecified.
Thus this fit did not train paired control interventions on the same prefix or
mid-song difficulty changes.

AdamW used `3e-5` for inherited R1 parameters and `3e-4` for composition, row
controls and preview projection, with weight decay `.0001` and gradient clipping
at one. The outcome arm generated 384 complete candidates; 82 updates had a
nonzero outcome gradient. Frozen audio/H/R tensors remained bitwise unchanged.
Native CPU sampling and MPS rescoring differed by at most $2.34\times10^{-5}$
in supported row log probability.

Qualification supplied only source H timestamps, generated every materialized
row from BOS and used matching seeds. All fourteen full-song generations
completed with every H timestamp preserved.

| Diagnostic | Source stars | Source-imitation continuation | Added outcome learning |
| --- | ---: | ---: | ---: |
| Mostly single notes, seed 261101 | 2.999 | 4.125 | 4.500 |
| Same audio, seed 261104 | 2.999 | 4.234 | 4.281 |
| Same audio, seed 261105 | 2.999 | 4.452 | 4.227 |
| Slower chords | 3.103 | 2.264 | 2.649 |
| High-LN Miraie | 3.005 | 3.192 | 3.244 |
| Starry Jet | 3.635 | 3.752 | 3.889 |
| Shippaisaku | 3.626 | 4.278 | 4.487 |

Mean absolute error across the three singles seeds worsened from 1.272 to 1.337.
This failed both the .30 improvement criterion and the initial 1.25 absolute-error
bound. The relative chord/Miraie star and LN-proportion guards passed, but those
guards do not make the primary result successful. Native-H expansion was not run.

Head counts increased on all five main cases: 1,212→1,232; 516→605; 1,211→1,319;
2,112→2,274; and 1,874→1,969. The direction helped the sparse chord case but did
not supply accurate context-dependent difficulty correction.

## Generated relationships

Lens images and complete action/articulation tables covered both models' selected
four-second primary peaks, source comparison passages and the two human
LN-coordination contexts. Source annotations remain attached to their original
charts and scopes; no generated chart received an inherited human judgment.

The singles output retains repeated added chords in fast flow. The slower chord
output gains a sustained passage of three-note chords, consistent with its
higher rating but insufficient to claim broadly correct control. Miraie's LN
median changes from 145.5 to 149.5 ms, with 15.2% versus 13.75% at most 80 ms;
both remain much more fragmented than the inspected source arrangement.

Expression did not uniformly collapse. In Starry Jet's inspected context, the
outcome model creates a hold from 201157 to 203241 ms spanning nine intervening
H times while other columns continue. It also overlaps a 1042-ms hold in another
column. The matched continuation's longest hold starting in the annotated range
is 557 ms. This is a concrete retained sustained relationship, not evidence that
every longer hold is better or that the source pattern was reproduced.

Shippaisaku retains mixed short/long overlaps in both arms, together with many
additional independent release times. The outcome arm's whole-chart LN fraction
is .824 for requested .826, yet its rating is 4.487 for requested 3.626.
Matching amount does not establish appropriate burden or articulation.

## Fixed-prefix control audit

A separate audit held audio, H preview and all prefix rows fixed. It changed only
the difficulty request over singles $[56000,64000)$, chords $[8000,16000)$ and
holds $[160000,168000)$ ms. It used both genuine source histories and the same
source-imitation arm's generated histories for both models. The table reports
the outcome model's neural expectation at H queries, before external recovery
and LN feedback.

| Context | Source history, request 3 | Generated history, request 3 | Generated history, requests 2→6 |
| --- | ---: | ---: | ---: |
| Fast singles | 1.099 | 1.463 | 1.452→1.500 |
| Slower chords | 2.020 | 1.447 | 1.432→1.496 |
| LN mixture | 1.684 | 1.941 | 1.928→1.979 |

The observed neural difficulty response is small relative to these history
differences. Mean total variation between the entire neural row distributions
at requests 2 and 6 on generated histories is .043/.059/.055 for the three
contexts. The deployed distributions change more in some cases because the
empirical recovery preference also uses difficulty.

These are one-step conditional queries, not counterfactual eight-second generated
outcomes. History changes include placement, LN state and resulting support;
their effect is not solely a count-history effect. The audit also freezes H,
which would normally respond to native difficulty changes. It therefore does
not assign a percentage of system error to R1 or establish that every control
channel is unused.

The combined evidence supports testing controlled continuations from a shared
generated prefix under different scoped requests. That comparison can supply
direct evidence about following a new request while retaining committed rows and
open holds. The current whole-song fit pairs each source chart with its own
single request, so it does not isolate that intervention. More training, richer
inputs and alternative estimators remain possible contributors; the current
result does not establish which one alone would solve playability.

The subsequent [paired scoped-control study](paired_scope_control_learning.md)
tests that shared-prefix intervention and reports its generated results.

## Provenance and resources

Executable source: `c3befb2fa397a6149e0f3090419e1210764d33b4`.
The original model checkpoint is SHA-256
`0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8`.
The experiment owner is `20260926-range-outcome-r1-v1`; `comparison.json` records
the panel, drivers, draw logs and terminal checkpoint identities.

- Source-imitation checkpoint: `2b63b1d9ff8d4f33417937a768e0f2769f1903645af10fe013fc4b2c7f6e53fb`.
- Outcome checkpoint: `c9ecb88c8667b75893a47a7fe8c874fbbd530517e08b271e6fef13ee77eb9d0a`.

On the 24-GiB Mac M5, PyTorch 2.11/MPS with one CPU thread completed source
continuation in 27.64 seconds and outcome fitting in 1065.57 seconds. Update
counts and source exposure were matched; sampling compute was deliberately not
matched. Peak sampled process physical footprint was 2.70/3.43 GiB and MPS driver
allocation 1.36/1.61 GiB. Those ledgers overlap. Qualification excludes waveform
decoding and Mel preparation; this study does not certify native timing,
first-thirty-row latency or runtime control-switch quality.
