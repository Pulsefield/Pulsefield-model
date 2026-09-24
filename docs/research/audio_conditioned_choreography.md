# Audio-conditioned choreography: target distribution and modeling primitives

This is a research proposal, not a V3 architecture specification. The objective
is a distribution of musically meaningful, playable arrangements. The existing
R1 checkpoint is a reproducible baseline and possible initialization; its timing
interface, memory, factorization and audio-free action decoder are revisable.
The [generation formulation](../formulation/notation.md) continues to own chart
legality and committed-prefix semantics.

The immediate research question is which dependencies a model must preserve to
turn complete music into varied, coherent timed actions under a playback deadline.
Selecting a larger network or reducing timing error does not answer that question.

The [full-audio and history-path comparison](audio_context_history.md) adds an
optional coarse full-song branch, bounded timing modulation and shared-history
interval likelihood. Its four matched cells separate the audio-context and
history-path hypotheses; native playability remains an independent evaluation.

The [skeleton/row information contract](audio_skeleton_information_contract.md)
sets the next prototype's dependency boundary: skeleton history is separate from
row-content history, explicit LN-state feedback remains available, and rows
receive audio as well as skeleton information. It records a measured violation
of that boundary in the current timing residual and a proposed head-plan/release-
clock factorization. The existing joint models are diagnostic baselines for it.

## Current bounded direction

Fix the owner-confirmed repository music Mel representation and jointly learn
audio conditioning, event timing and complete-row generation from paired audio
and beatmaps. Do not make an encoder-sufficiency study a prerequisite: useful
information depends on the chosen decoder and training objective, and a freely
authored chart does not identify a unique sufficient audio representation.

Use `MUSIC_MEL_CACHE_CONFIG` and `compute_log_mel_10ms` from
[`mel_base.py`](../../src/ensomi_model/features/mel_base.py), together with the
repository waveform-loading convention. The frontend is mono 24 kHz, 128 Mel
bins, 10 ms hop, a 40 ms Hann window, `n_fft=win_length=960`, 20–12,000 Hz,
`center=False`, power 2, `norm=1`, natural log and a `1e-5` floor. Frame $i$ has
support $[10i,10i+40)$ ms and center $10i+20$ ms; repository right padding and
frame-count behavior remain authoritative. Feature spacing is not output-time
precision.

The first learned encoder should be small and explicit: a projection retaining
frequency information, a few residual temporal-convolution blocks with a declared
receptive field, and a shared audio-conditioning readout. Preserve the input time
resolution initially. Complete audio permits bidirectional context. Both timing
and action prediction receive the encoded audio; timing can inspect a bounded
upcoming sequence, and row prediction also receives context at its selected time.

Develop the generation structure and this encoder together. The available direct
supervision is paired beatmap/audio: event timestamps, complete actions, exact
replayed state and observed continuation/no-event intervals. Separate charts of
one recording remain separate training examples. No MERT-style pretraining
corpus, acoustic teacher labels, instrument annotations or complete style/demand
labels are assumed. BeatThis and broader encoders remain optional later
comparisons, not prerequisites for this baseline.

Defer new long-term musical-relation memory. Retain exact state and sufficient
recent generated history. The first implementation retains R1's 511-row finite
history encoder, historical query projection and compatible action heads. It
omits the seed residual, landmark memory and future-candidate consequence module.
This initializes a different model; it does not preserve the released R1 policy.

## What the first evidence changes

A bounded TRAIN audit compared different arrangements of the same hash-verified
audio. Its seed-eligible, at-least-60-second subset contains 416 charts and 584
same-audio arrangement pairs. Pair medians are descriptive: groups with more
arrangements contribute more pairs, and these are not population estimates.

| Observation | Consequence for the research question |
| --- | --- |
| Median head-time F1 at 20 ms is 0.785 between alternative arrangements; all-release F1 is 0.281 and release-only F1 is 0.092 | Matching one reference cannot define the full set of acceptable timings. Release choices are especially arrangement-dependent. |
| Among alternatives whose head densities differ by at most a factor of 1.2, head F1 is 0.941 but release-only F1 is 0.169 | A global rate condition leaves substantial hold/release ambiguity. |
| 78.7% of observed release rows coincide with a head row | Release-only events are not a separate exhaustive release target. Their membership depends on the chosen heads. |
| Long constant-spacing head runs occur, including hundreds of heads | Long rhythmic continuation must remain representable. This fact alone identifies neither Jack organization nor acoustic-onset correspondence. |

Source charts are examples of organization, not automatically positive playability
labels. The audit did not listen to audio or identify acoustic events. In
particular, a single musical cue elaborated into a long Jack is an explicit target
capability, not a conclusion inferred from constant-spacing counts.

The 48-chart audio pilot has one chart per audio recording. It therefore provides
no direct paired supervision of alternative arrangements of the same recording.
Its two-channel frame objective and threshold decoder are useful audio-transfer
baselines, but have not established a distribution over coherent alternatives.
With supplied coarse density conditions, frozen BeatThis features improved head
F1 from 0.677 to 0.732 on six assessment songs; release-only F1 remained low
(0.035 to 0.051). These are small, single-seed development results, not a quality
benchmark. The scores use the original fine-offset predictions before the later
native-millisecond materialization correction.

That pilot also used a separate experimental frontend (`n_fft=1024`, centered
windows, log10, a `1e-10` floor and different Mel/normalization settings), not the
owner-confirmed repository music frontend. Its audio-transfer results therefore
do not benchmark the newly fixed input contract. Preserve its original feature
identity; a canonical-frontend run needs a separate cache and evidence owner.

The R1 candidate experiment supplies a different observation. Adding unused
non-head opportunities changes the action distribution even though required head
times and the playable seed are fixed. Each intervention has 24 chart/seed pairs
from 12 songs and two seeds. For each pair, divide the changed chart's median
suffix-born LN duration by the baseline chart's median, then aggregate those
ratios. In the integer-millisecond rerun, adding one opportunity in every fourth
eligible gap produces a median ratio of 0.686 and a median LN-head-share increase
of 25.6 percentage points. Inserting into every eligible gap produces 0.375 and
68.6 percentage points respectively. This is total sensitivity to
the schedule, including changed timing features and RNG consumption; it does not
isolate a release hazard.

Beatmap Lens inspection separates this sensitivity from candidate quality failures.
In one generated SCREW passage, a lane closes at 90369 ms and is pressed again
at 90381 ms, while another lane is available. A second such relationship occurs
at 90709–90721 ms. These motivate comparison with equal-composition alternatives.
Conversely, inspected human examples contain legitimate short LNs, repeated
chords, mixed rhythmic gaps and asynchronous releases. Neither repetition,
irregularity, LN quantity nor disagreement with source density is a general BAD
label. No player trial or audio-listening verdict has been obtained.

## The modeled object

Let $A$ be complete source audio, $(H,g)$ the committed chart and fixed-through
time, and $c$ the optional style/demand requests from the formulation. For a future
window $W=(g,e]$, the target is

$$
P_\theta(Y_W\mid A,H,g,W,c),
\qquad
Y_W\in\mathcal V_{\mathrm{legal}}(H,g,e).
$$

This conditions on already chosen choreography as well as music. Two valid
arrangements of one recording need not have identical heads, durations, releases,
lane paths or local difficulty. A source imitation likelihood can be useful
training evidence without making a particular source the only correct output.

One possible latent factorization is

$$
P(Y_W\mid A,H,g,W,c)
=\int P_\psi(Z_W\mid A,H,g,W,c)
       P_\phi(Y_W\mid A,H,g,W,c,Z_W)\,dZ_W.
$$

$Z_W$ would describe a persistent arrangement choice, such as a continuing
rhythmic figure and its development. It is not assumed to be an annotated chorus,
a fixed section class, or a five-style label. A direct autoregressive joint model
is an alternative that can represent multimodality without an explicit latent
variable. The need for a separate $Z$ is a hypothesis, not a prerequisite.

The output remains complete simultaneous rows $(t_i,m_i)$. The first joint model
factors the next event into a waiting time and a conditional nonempty row:

$$
P_\theta(\Delta t_i\mid F_\eta(A),H_i,x_i,c)\,
P_\phi(m_i\mid F_\eta(A),H_i,x_i,\Delta t_i,c).
$$

$F_\eta$ is the small learned Mel encoder and $x_i$ is exact replay state.
The history contains the actions actually chosen, so a newly opened LN can affect
the next timing decision. Derive head-only, release-only and combined events from
the sampled complete row. An extra event-kind predictor is an optional future
factorization, not necessary supervision for the first model. This must also
account for no event before the horizon. A continuous-time construction needs a
declared density/measure and finite-sequence behavior. A discrete-time
implementation must state its clock and support restrictions. Quantization for
native `.osu` export is a materialization choice, not a new minimum-spacing rule
for the V3 chart language.

### Audio is not a list of permitted attacks

Music can support omission, accent selection, rhythmic elaboration, sustained
control, repetition and variation. There is no required one-to-one alignment
between acoustic onsets and chart heads. A transient may initiate a long rhythmic
figure; a sustained sound can support either holding or repeated articulation.
Additional acoustic onsets need not each become a required head.

An onset-expanded Jack is also not a timing-only property: repeating timestamps
can be realized as a Jack, a stream or alternating groups. The arrangement choice
must reach the action decoder, or timing and action must be modeled jointly.
This is a reason to reconsider the factorization, not to hard-code a Jack mode.

### A minimal source skeleton differs from serving opportunities

For a complete chart $Y$, let $R_Y$ be its actual row times, $B_Y$ its head-bearing
times, and $E_Y=R_Y\setminus B_Y$ its release-only times. These are a projection of
the complete arrangement. They do not identify a unique superset of unused but
acceptable release opportunities.

The released R1 can abstain at an extra non-head candidate, but its source timing
was the actual event union. Treating an arbitrary denser candidate schedule as
equivalent changes the input distribution. Conversely, an unobserved candidate
is not automatically a musically incorrect timestamp.

If a skeleton remains useful, distinguish actual event intent, uncertain proposals
and available execution support. Their meanings and supervision differ. A joint
time/action model can instead predict actual release rows directly, making a
separate opportunity process optional.

## Expressivity and boundary requirements

| Case | Required capability | A restriction that would lose it |
| --- | --- | --- |
| Regular subdivisions | Sustain a chosen pulse or subdivision with stable relative organization | Independent local mode switching or admission only at detected acoustic onsets |
| Tech, mixed fractions, tuplets and expressive offsets | Preserve fine, changing and non-grid timings; relate them to surrounding organization | A finite beat grid without a residual/general-time path; treating irregularity as an error |
| One cue expanded into a long Jack/dump | Continue a rhythmic and action relationship over a chosen span, with entry, development and exit | One head per audio peak, universal repetition penalties, or timing-only intent with unrelated lane generation |
| Repeated music with variation | Reuse relevant past organization while responding to changed sound content and current gameplay state | A hard chorus identity that copies a fixed chart template |
| Independent LN releases | Close one or several held lanes between heads or together with heads on other lanes | Release-only audio detection as the entire release model; independent lane clocks that cannot coordinate |
| Short LN articulation and long overlapping holds | Represent both, retaining exact endpoints and interacting taps | A minimum-duration cleanup rule, a universal LN percentage, or forcing every hold to fit one generation block |
| Rests and sustained intervals | Fix no-row decisions while preserving occupancy and advancing clocks | Treating absence of heads as absence of gameplay state or resetting memory after silence |
| Dense transitions | Anticipate upcoming demands and compare feasible action organizations | Filling the same density with arbitrary lane choices or suppressing difficult material to improve a score |
| Cold start | Generate from true BOS and the closed initial occupancy | Treating 30 supplied heads as a task law or inserting unplayed dummy history |
| Song end | Materialize every required close by the true end | Treating a crop boundary as terminal or implicitly releasing at playback end |

Finite compute or export support can bound an implementation, but those bounds
must be measured and declared rather than renamed as style or playability rules.
The current client's incremental LN protocol permits an emitted start whose close
is still unresolved. It therefore does not require short holds or waiting for all
future endpoints before playback.

## Shared audio conditioning

Complete-song audio is available during both training and inference. Playback
deadlines constrain when chart actions must be ready; they do not impose an
audio-observation causality mask. Final note placement, including LN releases,
is the target. Redline metadata and estimated beats do not define timing truth
or the allowed event support.

The finite Mel encoder currently has equivalent cropped-training and full-song
inference outputs when its full halo is supplied. That equivalence only covers
the implemented local receptive field. It does not give each position access
to whole-song musical relationships. A broader audio representation must use
the same complete-song information in training and inference, independently of
target-selected query bounds. This remains an open design question, distinct
from the mismatch between teacher-forced and generated chart histories.

A timing-only interface implicitly proposes that the skeleton and chart history
are sufficient for downstream action decisions. In probabilistic terms, dropping
audio asserts an approximation such as

$$
P(\text{actions}\mid A,S,H,c)
\approx P(\text{actions}\mid S,H,c).
$$

The first joint design avoids requiring this approximation: the shared Mel encoder
conditions both predictions. Timbre, pitch movement, accents, texture and sustained
energy can affect arrangement through task training. Whether a particular network
uses those features well is an architectural and optimization question, not a
standalone information-sufficiency pass/fail metric. A scalar density or style
token is not assumed to preserve all relevant musical information.

Three complementary views are worth distinguishing:

- Local acoustic detail for transient shape, sustained activity and precise
  alignment. Temporal resolution and receptive support must be recorded.
- Metrical evidence, including uncertain beat/downbeat activity and relative
  phase. BeatThis contributes a useful learned view; its outputs are observations,
  not a compulsory clock or the entire representation of music.
- Broader musical content and relationships across time. Frame/sequence features
  from general music encoders can retain information beyond a beat objective.
  Full-song average embeddings alone cannot describe where the music changes.

[MERT](https://arxiv.org/abs/2306.00107) combines acoustic and CQT-based musical
pretraining targets. [MusicFM](https://arxiv.org/abs/2311.03318) studies pretrained
music representations across tasks. These provide candidate feature primitives,
not evidence that their embeddings already encode the arrangement semantics needed
here. They are deferred analogues. The current experiment learns its small encoder
from the chart-generation objective and does not require their pretraining setup.

## Primitives and the mechanisms they contribute

| Primitive and closest analogue | Mechanism to consider | What does not transfer automatically |
| --- | --- | --- |
| [Marked temporal point processes](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html) | History-dependent event timing, marks and explicit probability of no event | Separate lane processes do not supply simultaneous complete rows; a generic intensity does not ensure musical structure or fast sampling |
| [Mixture inter-event-time models](https://arxiv.org/abs/1909.12127) | Multiple plausible gaps with tractable density, survival and direct sampling | Repeated local draws can drift between subdivisions; positive gaps alone do not prove coherent phrases or non-explosive conditional dynamics |
| [Hierarchical latent music generation](https://proceedings.mlr.press/v80/roberts18a.html) | A persistent choice can guide several lower-level decisions and their variations | Independent fixed subsequences would break occupancy and cross-boundary holds; latent codes are not automatically semantic controls |
| [Inverse sequence transformations](https://proceedings.mlr.press/v97/gillick19a.html) | Learn elaboration from deliberately simplified musical representations, preserving multiple possible realizations | A simplified chart is not an observed acoustic onset sequence; a drum grid is not the legal timing support for mania |
| [Music Transformer](https://research.google/pubs/music-transformer-generating-music-with-long-term-structure/) and [Museformer](https://arxiv.org/abs/2210.10349) | Relative relationships, detailed access to relevant older material and cheaper summaries elsewhere | Fixed bars, prescribed bar lags and source MIDI tokenization are not our section or timing contract |
| [Beat-aligned chart sequence generation](https://arxiv.org/html/2311.13687) | Direct audio-conditioned event sequences and useful metrical inductive bias | Its fixed beat-relative support and corpus exclusions do not cover all required dense/irregular cases; timing F1 is not playability |
| [Receding-horizon generative policies](https://diffusion-policy.cs.columbia.edu/) | Plan a longer joint continuation, commit a shorter prefix, reuse conditioning computation | Robot-action smoothness is not rhythmic quality; incomplete refinement is not a validated fallback policy |

These are competing or composable mechanisms, not a proposal to assemble every
named model. The source-bounded novelty assessment is adaptation of established
conditional sequence, latent-state and planning ideas. Any claim of a new
representation or useful combination requires evidence on the actual task.

### Musical coordinates as evidence

A soft beat coordinate can make regular subdivisions easy to learn while absolute
time preserves generality. One candidate uses a metrical position plus a residual;
another predicts a multimodal physical-time gap conditioned on beat evidence.
Both must represent fine fractions and non-grid events. Neither should turn a
half/double-tempo error into an irreversible exclusion of valid charts.

The research question is whether the musical coordinate improves learning and
coherence while retaining support. It is not whether a sufficiently fine lattice
can approximate the existing timestamps.

### LN release as conditional continuation

Two serious alternatives remain:

1. **Object duration planning:** propose a head/type and a duration or endpoint
   distribution jointly. This makes long hold intent available early, but future
   decisions must remain consistent with committed starts and other lanes.
2. **State-conditioned release events:** condition future complete rows on held
   lanes, hold ages, action history, future audio and the chosen arrangement.
   This naturally supports incremental closes, but needs learned duration and
   cross-lane dependencies rather than an unconditioned close probability.

A hybrid can carry a provisional duration plan while realizing releases through
the row process. An endpoint in a provisional plan is not automatically committed
history. Windowed training must handle holds whose endpoints lie outside the
window without fabricating closes at its boundary.

Release without a head is representable independently of head occurrence; release
is not statistically independent of heads, held state or other releases. The
entire simultaneous row remains the unit of exact validation.

For a waiting-time model, advancing through a quiet interval also requires
consistent residual survival. If $S$ is the current waiting-time survival and
$u$ has already elapsed without an event, remaining survival over $v$ is
$S(u+v)/S(u)$. Restarting the clock every time the scheduler replans would change
the modeled rest distribution.

## Deferred: memory without fixed musical sections

This section preserves future design considerations. It is not a prerequisite
for the initial shared-Mel, joint timing/row experiment.

Complete audio and generated-history memory answer different questions. Future
audio is available; future chosen actions are not committed facts.

Keep four kinds of information distinct:

1. **Exact replay state:** open lanes, actual starts and action clocks. This must
   survive arbitrary gaps and cannot be replaced by a compressed neural vector.
2. **Recent detailed history:** timing, complete rows and local relationships,
   with a measured receptive span in both events and seconds.
3. **Older paired music/arrangement memory:** summaries and selected detailed
   exemplars of what was generated in response to earlier musical material.
4. **Provisional plan state:** an uncommitted branch's intended continuation and
   possible durations. It is isolated from the committed memory.

A candidate older-memory entry binds an audio span to its actually committed
choreography. Current audio queries similar earlier spans through soft content
relations. The retrieved value describes their arrangement and boundary state;
the decoder can reuse, transform or depart from it according to current audio,
controls and exact occupancy. Similarity does not command literal copying.

This permits a repeated musical passage to recall a related pattern while a
changed melody, texture or accent changes its realization. No chorus label or
fixed section boundary is required. Overlapping time scales, learned span
summaries and selected event exemplars are candidate implementations. Their
adequacy is judged by retained relationships, not memory size alone.

The existing R1 landmark interval of 64 head rows is a bookkeeping choice. Its
duration varies with density and does not define a musical phrase. Keeping all
older information at equal detail is also unnecessary; the unresolved question is
which relationships need exact recall and which tolerate summarization.

## Information dependencies and real-time commitments

```mermaid
flowchart LR
    A[Complete audio] --> X[Canonical Mel and small learned encoder]
    H[Committed chart and fixed-through time] --> E[Exact replay state]
    H --> M[Recent history and consistently retained existing state]
    X --> P[Arrangement and event generation]
    E --> P
    M --> P
    C[Optional controls] --> P
    P --> Y[Provisional timed action sequences]
    Y --> V[Exact validation and quality comparison]
    V --> G[Prefix commitment under playback deadlines]
    G --> H
```

This is a dependency proposal, not a requirement for separate executables. Timing
and action heads can share a backbone; a slower interpretation module and a faster
event decoder are another factorization. R1 can initialize part of the action
decoder, acquire audio/shared-plan conditioning, or be replaced if another
factorization better preserves the required responses.

Real-time generation concerns publication deadlines relative to playback, not
causal observation of unknown audio. The scheduler owns clocks, budgets,
condition versions and the immutable published prefix. The fixed-through boundary
may be ahead of the player. No-row commits and open LNs remain meaningful state.

Plan duration, published buffer and player reading lead are distinct quantities.
Longer joint plans can support coherent dense passages while only a short prefix
is published. Audio representations can be cached; branch-dependent action state
cannot be shared as if all branches chose the same history.

The released R1's whole-schedule timing summaries are a compatibility issue for
that checkpoint, not an intrinsic task requirement. A new model can use bounded
future musical context and explicit uncertainty. Any change to planning horizons
must be reflected in training and cache dependencies. Compute degradation needs
its own quality evidence; dropping holds or reducing density is not a neutral
way to meet a deadline.

## Joint training with the available data

For teacher-forced source histories, optimize time/event-sequence likelihood and
complete-row likelihood jointly, including the probability of no event before a
censored window boundary. Both losses update the shared audio encoder and any
shared history/readout parameters. The row loss evaluated at source timestamps
does not backpropagate through a sampled time or train the time-head parameters
via that timestamp. This is joint statistical training, not a claim of direct
playability gradients through discrete generation.

The released R1 needs explicit adaptation to this task. Its future 16-candidate
roles/offsets, whole-schedule counts and normalized schedule position cannot
remain source-provided inputs during training if they are unavailable in native
generation. Replace that condition path with available audio/time/history
features. Reuse compatible R1 weights as initialization without claiming that
zeroing missing fields preserves its learned policy.

Likewise, use exact local row legality instead of feasibility tied to an unknown
future candidate list. An actual event row is nonempty; four held lanes still
permit a release-only row. Termination cannot leave an open LN, and a crop end is
not the song end. Train true BOS and short prefixes; do not expose future hold
endpoints unless the inference condition actually supplies them.

### First implementation: conditional hazards on the native clock

The bounded implementation uses a discrete event hazard at every integer
millisecond. Given the unchanged physical history, each candidate clock receives
its local audio context and exact elapsed-time features. A 10 ms query bin returns
ten hazard logits; this is vectorized computation, not a ten-millisecond event
lattice. After any sampled event, the next query can select the following
millisecond. Noninteger V3 times remain outside this native-export experiment.

For logits $l_j$, let $h_j=\operatorname{sigmoid}(l_j)$. The probability of the
next event at $j$ is $h_j\prod_{k<j}(1-h_k)$; no event through a window has the
corresponding survival product. An exponential draw is compared with accumulated
$\operatorname{softplus}(l_j)$ mass. Carry its remaining mass budget through
empty scheduler windows. Absolute bins, audio and physical history stay the same
across partitions, so splitting a query changes neither the distribution nor
the random draw. Advancing the fixed-through clock does not create a history row.

This branch lets timing inspect candidate-local music directly and gives long
rests an exact survival interpretation. A gap mixture would need an additional
mechanism to align its modes with future audio. Its potentially cheaper inference
remains a comparison if the hazard scan becomes the measured bottleneck.

The Mel encoder projects 128 frequency bins to 96 channels and uses six residual
depthwise temporal convolutions with kernel five and dilations 1 through 32.
Its halo is 126 Mel frames on each side. Interpolation uses actual frame centers
at $20+10i$ ms. Training crops include this halo and mask padding beyond the song;
native generation can cache the same full-song encoding. Timing and row heads
share it, without downsampling or pretrained acoustic supervision.

True BOS uses a cursor of -1 ms, allowing an initial event at zero. A normal
query end censors waiting and never forces a release. At the actual decoded
audio end, an open hold forces an event whose legal row closes every held lane
and opens none. With no held lanes, the model may finish without another event.
This terminal safety condition does not teach musically appropriate LN duration;
earlier releases must be learned from paired charts.

The experimental entrypoint is
`python -m ensomi_model.research.joint_audio_continuation.hydra`, with the packaged
`joint_audio` configuration. Preparation creates a fresh canonical Mel cache
and retains alternative TRAIN arrangements as separate samples. Training samples
groups, then charts; its BOS/event-prefix/absolute-time/outro query mixture is
explicitly recorded. Fixed-query memorization is a diagnostic, and validation
joint likelihood selects a development checkpoint only. Neither score is a
playability verdict. Each run requires a clean source revision and fresh output
directory; generation requires an explicit checkpoint path and SHA-256.

Preparation normally resolves `AudioFilename` beside the catalog's source file.
If that file is an imported copy without its audio, `source_aliases_file` and
`source_aliases_sha256` can supply an explicit, pinned pairing manifest:

```json
{
  "format": "joint-audio/source-aliases-v1",
  "catalog_sha256": "<catalog SHA-256>",
  "sources": {"<source SHA-256>": ["dataset/mapset/chart.osu"]}
}
```

Paths resolve under `catalog_root`. Each candidate must have exactly the same
source bytes; song titles and mapset metadata are insufficient. Differing audio
bytes across identical-source candidates are ambiguous and rejected. The corpus
retains the catalog identity and records `paired_source_file` separately, then
verifies that source and its audio reference when loading. This restores a file
location relationship; it does not establish audio uniqueness across splits or
authorize changing an existing corpus. The options are preparation-only.

Long waits need positive transition supervision as well as censored prefixes.
With `full_wait_supervision=true`, a selected BOS/event-prefix example covers
the complete observed wait in disjoint bounded queries. Their timing losses and
the final row loss are summed and normalized as one logical example. The trainer
microbatches expanded queries and takes one optimizer step per logical batch;
resource stops occur between those updates. Uniform-time and outro queries keep
their conditional bounded-window interpretation. `coverage_pass=true` inserts
each TRAIN transition longer than the horizon once at the beginning of a run,
without changing random draws for the remaining examples. Separate full-gap VAL
probes complement the unchanged common-query panel. Window selection uses labels
for supervision; neither wait length nor a target-selection flag becomes a
predictor input. Full-audio/crop parity prevents query extent from changing scored
audio features.

Inference query length can differ from the training horizon. Shorter queries
reduce discarded future computation after each sampled event; longer queries
may be cheaper during rests. Neither choice may change the hazard's absolute-bin
features, actual history or carried survival draw. A trained-checkpoint probe
produced identical native rows with 500 ms and 4000 ms query blocks. Its measured
speedup is one-case evidence, not a worst-case latency guarantee.

For a new audio file, use `mode=infer_audio` with `audio_file`, an explicit
`checkpoint_file` and `checkpoint_sha256`. This path computes canonical Mel and
uses checkpoint normalization without loading a chart corpus, source beatmap or
seed. It writes fresh output under `root/inference/run_name`, including the
input audio, verified `.osu`, feature identity and per-stage latency. A constant
120 BPM export placeholder serves editor/scroll presentation only; it is not an
estimated musical beat grid. Model timestamps remain authoritative for playback.

### Bounded joint-model evidence

The first corpus contains 121 separate TRAIN arrangements from 48 song groups
and 12 validation songs. The model has 2,950,458 parameters, including 463,392 in
the learned Mel encoder. The input contract and model architecture stay fixed
across the following learning checks:

| Check | Observation | Supported conclusion |
| --- | --- | --- |
| Memorize 32 queries | TRAIN joint NLL 9.619 to 0.00124; validation 10.494 to 45.349 | Both heads can fit; the fit is not a useful native generator. |
| Random queries across TRAIN | Best fixed-panel validation joint NLL 6.071; native samples lose many earlier millisecond repetition storms | Learning transfers beyond the memorized queries, but some duplicate-like TAP choices remain. |
| Complete waiting intervals and rare-transition coverage | Common validation NLL 6.091; two full-gap examples improve from 17.127 to 14.628 | The targeted coverage correction helps these development examples without a material common-panel regression. |

The validation panel has only 48 queries and is unblinded development evidence.
The two long-gap cases are diagnostics, not a population estimate. Time and row
losses can move differently: the late-start case improves its timing likelihood
while its first-row likelihood worsens. No TEST result or listening/playtest
quality estimate follows from these scores.

Lens review found both negative and positive native examples. Duplicate-like
same-column TAPs remain in some dense sampled passages. Conversely, one complete
YOASOBI output has 896 action events, 967 heads and 301 LNs, with coherent tap
motion, LN chains and independent held/released roles. Its complete event sequence
and systematic/targeted renders supported handing that exact chart off for
prototype playtesting. Musical correspondence, enjoyment and calibrated player
demand remain unverified. This is one positive candidate, not reliable generation
across seeds and songs.

Short nominal release-to-head gaps require a different interpretation from rapid
repeated TAPs. Two initially flagged 10/13 ms tail gaps occur between same-lane
heads 189 ms apart, compatible with LN-jack articulation. They are unresolved
preference cases, not established BAD labels. No global release-gap filter was
introduced. Density differences from the reference are likewise not sufficient
to reject an alternative arrangement.

Diagnostic interventions also rejected a simple timing-jitter explanation for
two duplicate-TAP witnesses. Moving the consumed source row 1/3/5 ms earlier
left the next-5-ms event probability negligible, while native histories gave
large short-gap mass. The native prefixes had much higher event rates and
different hold composition. Nonphysical swaps of encoded history and exact
features identified network sensitivity to those histories; they do not prove a
causal mechanism in valid charts. This separates a history/state question from
an unsupported decision to enlarge the audio encoder or forbid close events.

On one 4630-row sample, 500 ms and 4000 ms inference queries produced identical
row and `.osu` bytes; CPU generation fell from 38.22 to 9.39 seconds. For the
reviewed YOASOBI sample, a fresh Python process with warm OS disk caches covered
the first 8 audio seconds in 1.44 seconds and the first 31 heads in 1.51 seconds,
including imports, model loading, decoding, Mel and generation. It generated the
full 242.7-second song in 3.38 seconds. Interpreter startup before the script,
network and client costs are excluded. These are single-machine probes, not
deadline guarantees.

The implementation exposes audio/history/exact-state conditioning but does not
yet implement requested style or difficulty controls. A control adapter can
enter the shared condition used by both heads; generated-window readouts can
receive scoped annotation supervision separately. Missing annotations must not
be treated as negative labels, and style presence must not become a BAD-pattern
or numerical-difficulty label. New long-term musical-relation memory remains
deferred.

### Optional marked head-spacing prior

An experimental decoder setting, `head_spacing_ms`, defaults to zero and leaves
the trained distribution unchanged. A positive scale adds a continuous factor
to proposed heads using time since the last head on the same lane. For a proposed
row $m$ at $t$, its acceptance factor is

$$
a(m,t)=\prod_{j:\,m_j\in\{\mathrm{TAP},\mathrm{LN\_START}\}}
\min\left(1,\left(\frac{t-t_j^{\mathrm{last\ head}}}{\tau}\right)^4\right).
$$

A first head contributes one; releases do not contribute. In particular, a short
tail-to-next-head gap is not treated as a short head-to-head interval. If the base
hazard is $h_t$ and the conditional row distribution is $q_t(m)$, accepted event
mass is $h_t q_t(m)a(m,t)$ and no-event mass is
$1-h_t\sum_m q_t(m)a(m,t)$. The generator implements this by rejecting proposals
before they enter exact replay or learned history. At a true occupied terminal,
it instead normalizes weighted legal rows and forces closure. An independent
acceptance RNG preserves the proposal stream until a rejection occurs.

The initial probe uses $\tau=27$ ms, the smallest consecutive same-key head
interval in the pinned TRAIN catalog: 11,564 charts and 16,078,013 heads, with no
interval at or below 20 ms. This is a property of the admitted corpus, not a
universal physical limit. All its observed event rows have factor one at their
actual prefixes; novel positive intervals retain positive mathematical support.
The fixed fourth-power ramp is a hypothesis, not a fitted motor model. No global
event spacing, beat subdivision or release-gap constraint is introduced.

History-dependent point-process work provides an analogue for separating input
drive from post-event recovery. It also warns that a refractory constraint alone
can produce unrealistic rate saturation; its neuronal stability results do not
transfer directly to this TCN. See
[Gerhard, Deger and Truccolo (2017)](https://doi.org/10.1371/journal.pcbi.1005390).
The probe therefore requires native quality review and a check for new interval
pileup, not just fewer short gaps. It changes the generated distribution and has
not established reliable playability. When enabled, the row resource cap also
bounds rejected proposals; a capped run never invents LN endpoints.

In the initial seed-matched six-TRAIN/twelve-VAL probe, same-lane TAP intervals
at or below 10 ms fell from 42 to zero; those at or below 20 ms fell from 157 to
10. Median per-chart head-count ratio was 0.9866. The already reviewed YOASOBI
sample remained byte-identical. No four-interval same-lane run in the 24–30 ms
band appeared, and all outputs passed exact replay, export/reparse and strict
Lens admission. Scoped Lens review retained ordinary repeated-key figures,
chord repetition and independent LN releases, but also found remaining 14–20 ms
TAP pairs. The option remains disabled by default: these results support a local
correction, not reliable playability or equal preservation of every sampled mode.

### Learning the local correction on generated histories

The [native-window objective](../../src/ensomi_model/research/joint_audio_continuation/distillation.py)
can distill a corrected next-event law without adding inference parameters.
For each fixed generated prefix, its teacher assigns probability to every
native-time/complete-row outcome and to surviving the entire window. The loss
is KL over that complete distribution, including censor mass. Original chart
futures are discarded before collation; they cannot label an altered prefix.
The head factor changes both event hazard and conditional row probabilities.

A paired continuation tested this objective with the same 2.95M model and source
examples: 600 updates of source likelihood alone versus source likelihood plus
80 ms native-window KL. The correction used five TRAIN songs' generated histories;
a sixth song was withheld from correction supervision. Its real chart remained
in ordinary TRAIN. The 640 correction examples represented 390 unique prefixes;
the 128 held-out examples represented 70. Both final checkpoints were sampled
without the decoder prior on the same six TRAIN and twelve validation songs.

| Observation | Starting model | Source-only continuation | With native correction |
| --- | ---: | ---: | ---: |
| Fixed 48-query validation joint NLL | 6.091 | 6.449 | 6.436 |
| Held-out native-window KL | 0.0530 | 0.2012 | 0.1029 |
| Same-lane TAP intervals at or below 10 ms | 42 | 11 | 2 |
| Median per-chart head-count ratio to start | 1.000 | 0.642 | 0.688 |
| Total LN heads across 18 outputs | 1839 | 3202 | 4091 |

The correction reduced its TRAIN pressure-context KL from 0.1163 to 0.0492,
but did not transfer to the held-out histories or preserve the sampled
composition. All 36 new outputs passed mechanics, export/reparse and Lens
admission. Lens inspection still found 3 ms and 9 ms same-key TAP pairs, while
also finding organized chord motion and independent LN handoffs. One validation
output contained only ten heads at 7.046–7.820 seconds of a 121.033-second song.
These checkpoints failed the predeclared preservation and transfer guards and
were not adopted. A different density or LN fraction alone is not a BAD label;
the paired result does not establish an improvement in playable generation.

The silent tail exposes a separate recovery question. On that fixed eight-row
prefix, the corrected model's integrated future hazard was 5.0165, below its
sampled waiting threshold of 6.3416. Independent integration and generation
agreed, and 500 ms versus 4000 ms queries produced identical rows. Conditional
on already waiting four seconds, the predicted probability of no further event
before audio end was 0.754, versus 0.512 for the starting model on the same
prefix. This is one development example, not a population estimate. It identifies
a learned continuation risk that a short-window repetition objective does not
resolve. Further work must assess activation after rests and arrangement
composition alongside local press/release quality. It supplies no evidence that
a larger audio encoder or long-term memory is necessary.

## Research trajectory and the next decision

Next-time likelihood and conditional complete-row likelihood form a valid chain
factorization. Evaluating the row loss at the reference time, without a gradient
through a sampled time, is not itself an incorrect joint likelihood. A marked
intensity can represent the same distribution: set
$\Lambda_t=-\log(1-h_t)$ and $\lambda_{t,a}=\Lambda_tq_t(a)$.
Changes to that parameterization need an optimization or computational rationale,
not a claim that the current factorization assumes independent time and action.

The next experiments separate three explanations:

1. **Paired-data coverage.** Hold the model, likelihood, query policy and logical
   exposure budget fixed while expanding the training songs. A prepared corpus
   has 240 TRAIN groups with 585 separate arrangements and 36 validation songs,
   retaining all 133 charts of the first corpus. Encoded-audio and canonical PCM
   identities are disjoint across its splits. A catalog-wide audit found one
   exact-audio cross-split collision outside both selected cohorts; this does not
   guarantee perceptual, crop or speed-variant independence. Compare the original
   12 validation songs and additional 24 separately, and check R1's pretraining
   exposure. A matched short budget cannot reject larger data merely because it
   has not converged.
2. **Full-song audio and history coupling.** On fixed data and a fixed training
   objective, compare local versus local-plus-coarse-full-song audio, crossed with
   the original timing fusion versus an audio/hold-state timing base plus bounded
   history modulation. A small bidirectional coarse branch is a candidate, not an
   adopted architecture. Coarse audio token spacing does not quantize event times.
   Decaying the timing modulation during free-lane rests must preserve content
   history and exact holds, and must be tested on legitimate rests and long Jack
   figures. It must not force events or hide a missing audio drive.
3. **Persistent arrangement intent, only if needed.** A small variable sampled once
   per song could condition both heads if finite history cannot preserve early
   arrangement choices. It is not automatically a style or difficulty label.
   Reference-conditioned posterior reconstruction cannot substitute for generation
   from an audio-only prior. A larger codebook is not the default response to
   latent collapse or fixed-loop outputs.

Continuous source-time interval likelihood and explicitly missing content-history
views are candidates for the second stage, not simultaneous changes in the data
comparison. Interval sampling must state whether it estimates total chart, per-
song or per-time likelihood and apply the corresponding weights. Hiding history
observations with an unknown/truncated marker is different from changing the
actual prefix and reusing its original future. Unknown history is not BOS.

Recovery evaluation must include BOS, prefixes with 1–29 heads, mature prefixes,
and held/free states. The 144 mature free-lane probes excluded the ten-head SCREW
failure by construction. Missing early content, unavailable strata and incomplete
outputs remain visible outcomes. Report unconditional gap probability separately
from conditional recovery, and count short TAP relations both absolutely and per
eligible consecutive TAP-to-TAP transition. Head activity, LN occupancy time and
duration distributions guard against replacing taps with holds or silence.

For controlled data comparisons, training can use a pinned `normalization_file`
and `normalization_sha256`. The statistics must come from a nonempty subset of the
current corpus's TRAIN audio under the canonical frontend. The run records the
actual file and digest; inference uses checkpoint buffers. By default, training
continues to use its corpus normalization. Freezing this transform avoids changing
the initial raw-Mel function when the supervised corpus is expanded.

Short-horizon source likelihood does not establish rollout quality. Training on
generated prefixes or adding preferences may become necessary, but their targets
must be justified: paired audio/charts do not automatically label arbitrary
sampled continuations as good or bad. Do not convert a numeric gap threshold into
a universal playability rule or improve metrics by deleting difficult modes.

Evaluation keeps separate distribution coverage, musical correspondence, exact
mechanics, inspected action quality and deadline behavior. Style annotations
constrain declared scoped concepts; they are not numerical playability labels.
Human playtesting is still needed before final quality acceptance. Metrics should
also be tested against deliberate failure constructions: a repeated common loop
must not win simply because it raises self-similarity or likelihood. Recent
[chart-generation evaluation work](https://arxiv.org/html/2607.12857) motivates
that diagnostic practice, without replacing the Lens Foundation or human review.

## Evidence identity

- Joint implementation: `09b919cdeab90e3856fee03a9198d59b4dc527af`; complete-wait
  supervision: `608f092e6cd534638e8e47432bcb98973b79a5a4`.
- Canonical joint corpus manifest SHA-256:
  `4b995029a5344569d4506ff6b11249f61585d2bf7649285754340909bb06c21b`.
- Selected complete-wait training checkpoint SHA-256:
  `85f643077d127f9fe3e5256dc7b88512912d9ce8be39d6dbe164ef3ef4c9327e`.
- Equivalent inference-only checkpoint SHA-256:
  `29237d5bf25ed40fe1db4a8e022280ee834eae29521d62e3666462c110a71f49`.
- Joint learning, native outputs, Lens traces, targeted diagnostics and latency
  receipts: `artifacts/joint-audio/20260923-v1/`.
- Product baseline for the audit and native-millisecond generation:
  `068988e670e174621f96627827dc28386b1e6775`.
- Released R1 checkpoint SHA-256:
  `4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70`.
- Study manifest SHA-256:
  `d416a1953bb4f0126c084457cd8d6c597c96533da9f40d8e3245949006af934c`.
- TRAIN distribution audit:
  `artifacts/audio-skeleton/20260923-v1/distribution-audit/`.
- Integer-millisecond candidate comparison:
  `artifacts/audio-skeleton/20260923-ms-sensitivity/sensitivity/summary.json`.
- Audio pilots and native continuation:
  `artifacts/audio-skeleton/20260923-v1/training/` and `integration-ms/`.
- Lens source/render review and actual tool traces:
  `artifacts/audio-skeleton/20260923-v1/lens-review/`.
- Lens implementation: `22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`; frozen
  Foundation: `15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97`.

Generated artifacts may be absent in a fresh clone. The observations and their
limits are stated here independently of those local files.
