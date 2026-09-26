# Active LN audio origins and generated release organization

Retrieving an active LN's original audio improved release organization in one
inspected regular LN passage. It did not resolve excessive chord mass or restore
longer sustained roles in another passage. Ordinary continued training accounted
for much of the improvement in the primary short-LN diagnostic. The optional
branch remains disabled by default.

This study concerns the [controlled audio continuation model](controlled_audio_continuation.md).
H supplies head-bearing timestamps. R1 owns head cardinality, TAP/LN choices,
release identities and column placement; release timing supplies release-only
event times. Both timing and R1 receive complete audio. Difficulty and duration
statistics below are diagnostics, not definitions of playability or style.

## The information change

An open LN has an exact committed start even when its head has left the local
training crop. The added module retrieves audio at that start from the complete
song encoding and combines it with current audio and elapsed hold age. A shared
64-wide MLP constructs one value per occupied column. A pooled projection enters
the release clock before its hidden activation; ordered relative-hand values
enter R1 before composition and layout scoring. This adds 73,920 parameters to
the 4,583,985-parameter baseline.

The retrieval changes an observation, not an output decision. It supplies no
future source endpoint, release identity, chord count or R1 hidden state to H.
The release clock still cannot choose a release subset. Both factors' likelihoods
train the shared cue encoder. The existing frontier comparison and sampling
preferences remain present.

## Matched comparison

Executable source was `c1260d3737198e670f6b40254acbaf14fe0cf602`.
Both arms started from checkpoint SHA-256
`0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8`.
One added the zero-initialized cue projections; the other continued the existing
architecture. Audio encoding and H weights stayed frozen and bitwise unchanged.
Release timing and R1 were trainable in both arms. Thus the comparison isolates
the added representation under this fitting budget, not the value of training
release timing itself or of learning a new audio encoder.

Each arm completed 1,500 updates with identical 3,000 accepted interval draws
from 1,889 TRAIN charts: 2,216 population proposals and 784 proposals around human
annotations. Training used eight-second loss intervals, complete-song audio,
independently scoped controls, 15% control-family dropout and fresh AdamW state.
Inherited parameters used learning rate `3e-5`; composition, controls, preview
projection and new cues used `3e-4`. Each batch contained two intervals. Terminal
weights were used without selection by validation likelihood.

Generation began from BOS on five complete songs, with source H timestamps as
the only supplied chart events. Counts, columns, LN starts and all releases were
generated. Controls requested the source whole-chart star rating and LN fraction.
The two human LN-coordination examples additionally requested prominent strength
only over their annotated ranges. These are development mechanism checks; the
human examples are TRAIN charts. Source judgments do not label generated output.

The five paired seeds were 261101, 261102, 261103, 261231 and 261232. All ten
generations completed and preserved every supplied H time. The ordinary LN
amount feedback and recovery preference were identical; count priors, rate
readouts and attack-response projection were disabled.

## What improved, and what remained

The primary LN chart was *The Place You Promised To Show Me [Absent Promise]*.
Its source has median LN duration 222 ms, no holds at most 80 ms, star rating
3.005 and LN fraction .844. The original checkpoint generated median 154 ms and
12.9% of holds at most 80 ms.

| Generated quantity | Ordinary continuation | Added origin cues |
| --- | ---: | ---: |
| LN duration median | 183 ms | 183 ms |
| LNs at most 80 ms | 4.53% | 3.38% |
| Star rating | 2.576 | 2.558 |
| LN fraction | .846 | .855 |
| Head objects at the same 875 H times | 1,175 | 1,176 |

The cue arm missed the prespecified 185-ms median criterion and improved the
short-LN share by 1.14 percentage points against continuation, below the required
three points. Both met the absolute star-error and LN-fraction bounds for this
case. The matched comparison therefore gives little evidence that origin access
caused the large improvement from the original checkpoint. Native-H expansion
was not run.

The fast, mostly single-note source remained a more substantial unresolved
failure. Its 783 H times contain 861 source heads at 2.999 stars. Ordinary
continuation generated 1,259 heads at 4.375 stars; origin cues generated 1,226
at 4.162 stars. Lens showed repeated extra chords in the fast flow. Since H
times were supplied identically, this remaining excess cannot be explained by
overproduction of H events. The slower chord source generated identical rows in
both arms: 671 heads at 2.654 stars versus 786 source heads at 3.103 stars.

### Short holds can have coherent release structure

*Shippaisaku Shoujo [inabakumori Remix] [Forlorn]* provides a different question
from the primary duration diagnostic. Its source median is 111 ms. The human
LN-coordination range $[86452,93511)$ ms uses short and longer overlapping holds,
including simultaneous starts with different ends. Inspection included entry
and exit context $[84687,95276)$ and full endpoints beyond the crop.

| Holds starting inside the annotated range | Source | Ordinary continuation | Origin cues |
| --- | ---: | ---: | ---: |
| LN heads | 83 | 88 | 83 |
| Median duration | 111 ms | 110 ms | 110 ms |
| Release coincident with an H time | 73 | 21 | 71 |
| Noncoincident release within 20 ms of any H | 0 | 20 | 2 |

The origin-cue output retains 110/220/331-ms relationships and occasions where
one column continues while other columns change. The ordinary arm inserts many
additional independent release times into that regular flow, including near
coincidences with attacks. Its physical rows within the inspected context number
159 versus 99 with cues, although both have the same 85 H times. Whole-chart
short-LN share is 8.29% versus 3.40%, with median 110 ms in both arms.

This is evidence of a local improvement in temporal organization, missed by
the median-duration comparison. It is not a general preference for longer holds
or for releases coincident with heads. Irregular releases can be intentional;
the reference arrangement and full episode make the distinction here. Musical
alignment to the waveform was not independently listening-tested.

### Remembering an origin did not establish a sustained role

In *Starry Jet [Star]*, the human range $[198241,204908)$ contains two 1,250-ms
holds, each spanning five intervening H times while other columns continue their
own actions. Another hold spans three intervening H times. Neither generated
arm produces a hold starting in that range with more than one intervening H.
Their maximum durations there are 445 ms without cues and 417 ms with cues.

Complete context $[196574,206575)$ shows mixed TAP/LN and changing short holds,
but not the source's persistent roles. The cue arm produces long holds elsewhere
in the song, so this is not an absolute duration-support limit. Nor does the
comparison require reproducing source columns or every source endpoint. It
identifies a missing relationship under a prominent LN-coordination request.

## Interpretation and limits

NLL does not rank these outcomes reliably. On the same 22 scored validation
midpoint intervals, row NLL per second changed from 12.199 to 11.880 with ordinary
continuation and 11.890 with cues. Release NLL changed from 1.322 to 1.343 and
1.349, respectively. The small likelihood difference coexists with a visible
difference in the Shippaisaku episode and a shared failure in Starry Jet.

Supervision for that style is sparse. The TRAIN cohort contains eight human
prominent LN-coordination cells on eight charts. Only 27 sampled intervals
intersected those cells in this fit, covering ten distinct chart/time intervals.
These are potential exposures before control dropout, not a count of enabled
labelled losses. The study cannot distinguish every architectural limitation
from limited style supervision.

The added information is therefore a useful optional candidate for release
organization, with insufficient evidence for general adoption. Both factors
received cues and both were trained; this comparison does not assign the local
benefit separately to the release clock or R1. Frozen audio, one fit seed and
one generation seed per chart further limit the conclusion. Neither arm resolves
scoped difficulty calibration, persistent roles or the full audio-to-playable-map
task. A richer input does not itself supply an outcome objective on generated
histories, as discussed in the [response-control analysis](causal_response_control.md#design-implications).

## Resources and provenance

The Mac M5 has 24 GiB unified memory. PyTorch 2.11 with MPS, Python 3.10.20 and
one CPU thread completed the cue/ordinary fits in 841.7/789.7 seconds, including
validation. Peak sampled process physical footprints were 14.07/13.56 GiB;
MPS driver allocations were 7.40/7.16 GiB. These overlapping ledgers are not
additive. Full-audio encodings used an eight-identity cache in each arm.

The ten source-H generations took .80–4.03 seconds each on one CPU thread,
including model audio encoding from cached Mel. These timings exclude waveform
decoding and Mel extraction and do not establish native-H, first-thirty-row or
control-switch latency. No training process ran concurrently with qualification.

Artifacts belong to `20260926-active-ln-audio-cues-v1`. The paired draw log,
checkpoint and report identities are recorded in its `comparison.json`.
Terminal checkpoint SHA-256 values are:

- Origin cues: `f17b97dd4205539c68ffb20e59befe20f42e378f55bdb3b2bbc79fa514d429cd`.
- Ordinary continuation: `b9688cf1664de34e3efc7952786242fa139f5d29d6ec1e81e94b2182d6529a2d`.

Lens review covered both human examples with complete context, each generated
chart's selected four-second peak, and matched primary source passages. The
identical chord output needed no duplicate inspection. Images and paginated
action/articulation tables were read together; no new human labels were created.
