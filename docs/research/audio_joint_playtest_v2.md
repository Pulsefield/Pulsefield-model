# Joint audio/R1 playtest candidate v2

This research candidate generates complete timed 4-key rows directly from
audio and BOS, including LN starts and later releases. It has 3,461,828
parameters and a 13,912,295-byte inference checkpoint. The model combines the
canonical Mel frontend, a local TCN, coarse full-song audio context, R1-derived
finite action history and bounded historical timing modulation. The
[architecture and interval objective](audio_context_history.md) describe its
information paths. This is a playtest candidate, not an adopted V3 architecture
or a universal playability result.

## What the experiments establish

The native-ms hazard followed by a conditional complete-row distribution is
a valid joint factorization. These experiments retained that object and the
existing four-action lane alphabet. A raw TRAIN audit parsed 11,604 sources
containing 16,078,030 objects without finding same-lane, same-millisecond LN
close-and-restart. Five other raw sources failed parsing. That scope supplies
no evidence for expanding the alphabet before this study.

Increasing paired training from 48 song groups/121 arrangements to 240/585 at
the same 38,400 logical-example budget reduced additional-24 validation query
NLL from 6.4852 to 5.4883. It did not resolve native stability: the expanded
model produced three below-30-head outputs in 84 sampled runs. Wider data
coverage helped conditional prediction, while more work remained on generation.

The next experiment crossed local/global audio with original/bounded timing,
holding interval supervision, input data, initial shared parameters and exposure
fixed. Each cell used 4800 intervals, 37,300,370 ms, 248,836 complete event rows
and 339,728 heads. Source code was
`b5c5ee33cacb3950b3c9b6a4a05d9d627c46aa64`; data manifest SHA-256 was
`cc60dd39920c626f3be5498ab8ba7aa342dd0956c0a548cc393d85a1b03f5173`.

| Cell | VAL interval NLL/s | Native heads | Native LN heads | Same-key head pairs <=20 ms |
| --- | ---: | ---: | ---: | ---: |
| Local/original fusion | 41.9187 | 225579 | 92291 | 17 |
| Global/original fusion | 42.1303 | 225817 | 110318 | 35 |
| Local/bounded history | 41.8309 | 199812 | 92161 | 11 |
| Global/bounded history | 41.9304 | 184509 | 102118 | 4 |

Each native column covers the same 42 songs at two seed bases, not 84 distinct
songs. All 336 outputs completed and exceeded 30 heads; every final head
occurred after 85% of its audio duration. No cell met the planned 3% additional-
panel likelihood improvement threshold. The new recipe's improvement over
the earlier query-trained model cannot be attributed to interval sampling
alone: objective, seed and event exposure also changed.

The full-song branch affects predictions, but its temporal use remains unclear.
For global/bounded, additional-24 interval NLL/s is 37.7294 with correct context,
41.5211 when zeroed and 37.7460 when shifted by half a song. This is compatible
with substantial song-level conditioning and weak position-specific use under
teacher forcing. It does not establish that musical relationships are unnecessary.

Global/bounded is the next candidate because short-head rates improve within
transition types and inspected organization survives. TAP-to-TAP <=20 ms falls
from .07237 to .04857 per 1000 eligible TAP transitions; TAP-to-LN falls from
.39885 to .04868. Its median paired head ratio is .814, and LN total ratio is
1.106. It is more LN-heavy and less active than the interval control, so these
changes require playing-demand judgment rather than a short-repeat score alone.

## Fixed decoder refinement

The selected training checkpoint is
`b7d56ea062b3ab0d4bdf27fcfb10835a94e376e8568761e37811391a3ebc31eb`.
The candidate recipe adds the existing optional 27 ms head-age prior:
each proposed head contributes `min(1, (same_lane_head_age/27)^4)`.
It uses time since a head, not time since a release. It neither bans repeated
figures nor quantizes timing. The scale is an empirical research prior, not
a universal physical minimum. Rejected rows do not enter history; occupied
true terminals reweight legal rows and still resolve every hold.

The fixed comparison retained all 84 outcomes. Twenty-three trajectories could
change; two additional unaffected cases were rerun to verify exact coupling.
After both checks matched row bytes, 59 invariant trajectories were reused.
Forced terminals were rerun because their reweighting path can alter draws.
Thus there were 25 new sampling runs, not 84 new runs.

The refined set has 184563 heads, 102003 LN heads, all outputs complete and
above 30 heads, and zero observed same-key head intervals <=20 ms. Eighty
outputs are byte-identical in their timed rows. Median paired head ratio is
1.0; LN total ratio is .99887. All four changed cases stay within 5% of their
baseline head/LN counts. The result removes the located extreme repeated
presses in this panel; it is not a guarantee for every future sample.

## Structural review and its limits

Beatmap Lens admitted every complete output through its strict canonical
parser. Targeted reviews read full paginated actions, articulation and
time-proportional pages, using an unchanged frozen 204-example human reference
bundle. They include early/middle/late regions, dense chords, coordinated LN
releases, repeated-chord figures and each corrected short-head location.

Inspected examples preserve materially different organizations: YOASOBI has
staggered LN starts and shared releases; Tsuikou has dense tap/chord motion;
Dotabata mixes LN coordination and taps. Repeated two-/three-key figures remain
available, including five left-hand chords in FORViDDEN ENERZY separated by
156–170 ms. The affected short-head fixes retain their surrounding motion.
Short LN durations alone were not classified as BAD.

These are scoped structural observations. They do not amount to blind listening,
player validation, a whole-corpus Tech/dump verdict or calibrated difficulty.
The candidate's LN preference, music alignment, consistent authored intent
and future controls remain open questions. No TEST split selected the model.

### Annotation-anchored follow-up

A fixed-weight diagnostic at source
`de2ff6207158edf1e1cd71ae1697b4195ba73854` generated Who?, death piano,
Prom Queen and Good Luck, Babe! from BOS at seeds 17 and 19. These are catalog
TRAIN songs used for diagnosis, not new held-out-song evidence. All eight runs
completed, exceeded 30 heads and had no same-key head interval <=20 ms.
Complete Lens action rows and time-proportional pages nevertheless show an
expressive-coverage gap: both death piano outputs are predominantly LN-based
in the reference's tap-only Tech episode; both Prom Queen outputs replace
much of the reference's changing-chord repetition with staggered holds.
Different arrangements are legitimate, but these outcomes do not demonstrate
the reference relations merely by being active or complex.

A conditional-row probe scored every event in each episode using that chart's
own actual prefix and event time. The statistic below is the sum of expected
LN starts divided by the sum of expected heads from the legal full-row
distribution. It excludes the optional decoder prior.

| Episode, source ms | Real-prefix expected LN fraction | Native-prefix fraction, seed 17 / 19 |
| --- | ---: | ---: |
| Who?, [75000, 85000) | 65.03% | 71.97% / 75.00% |
| death piano, [103453, 106630) | 0.73% | 88.83% / 82.26% |
| Prom Queen, [75838, 78338) | 0.87% | 85.67% / 75.09% |
| Good Luck, Babe!, [128423, 131252) | 0.71% | 3.58% / 11.45% |

The reference's observed LN fraction is 66.67% for Who? and zero for the other
three episodes. Across 795 source/native events, direct full-audio row scoring
agreed with canonical crop scoring within 3.82e-6 log-probability on twelve
selected parity checks. The predictor received no future chart or LN endpoint.
Native self-likelihood is not a quality measure. These are different trajectories
with different times and occupancy, so the contrast does not isolate history
causally. It does show that tap-heavy conditional predictions remain available;
native arrangement selection and persistence need separate investigation before
adding a latent or declaring the action decoder incapable.

This review also found a pairing omission: 490 catalog sources pointed to
imported copies without adjacent audio, although byte-identical originals with
audio existed locally. Explicit verified aliases recovered all 490: 427 TRAIN
and 63 VAL, including 113 human-reference sources. Thirty TRAIN alternatives
share the existing selected audio exactly. A fresh corpus restores those targets
while retaining every old chart, audio asset, normalization value and VAL chart;
it has 615 TRAIN arrangements and has not been fitted for this candidate.
The [pairing contract](audio_conditioned_choreography.md) preserves catalog
identity and rejects differing-audio ambiguity. The original experiment corpus
and checkpoint remain unchanged.

Local evidence is under
`artifacts/joint-audio/20260924-expanded-v1/lens-review/context-gb-prior27-v1/style-audit-v1`.
The conditional-probe result SHA-256 is
`9305057f4284d73dfa9a3ca756a178818f025b43f30f76ba1fed47cdfc6106b6`.
No human labels were edited, and no listening or player test is implied.

The Python research API also accepts
`rollout(..., prefix=ObservedPrefix(rows, coverage_ms))` for conditional
continuation diagnostics. It replays the entire observed prefix for exact
occupancy and clocks, and rebuilds only the finite neural suffix with its true
predecessor gap. Unknown hold endpoints are not supplied. Sampling uses a fresh
survival draw after known coverage; it does not restore an earlier RNG stream.
Returned rows retain the prefix, exported rows identify observed versus sampled
origin, and callbacks publish only newly generated rows. Caps and first-30
metrics count new material. Diagnostic startup means eight seconds beyond the
observed clock and cannot be reported as audio-only startup latency. The
source-free command continues to use BOS.

## Actual prefix publication and runtime

`rollout` now exposes immutable incremental rows and fixed-through coverage.
Source-free inference flushes an `events.jsonl` stream before whole-song export;
LN heads and their later CLOSE rows are separate updates. A consumer can begin
using the fixed prefix without waiting for the completed `.osu` file.

The following observations use fresh Python/uv processes on this Apple M5 Mac,
CPU with one thread and warm OS disk cache. They include model verification,
audio decode, canonical Mel, full-song encoding and reading the flushed local
stream. No network or client renderer latency is included.

| Input | Duration | Process to readable fixed 8 s | Process to 30 heads | Decode step p99 |
| --- | ---: | ---: | ---: | ---: |
| Dense I sample, peak 34 heads/s | 153.86 s | 1.590 s | 1.355 s | 2.50 ms |
| Longest native-panel input | 441.10 s | 2.060 s | 2.073 s | 2.51 ms |
| YOASOBI | 242.67 s | 1.467 s | 1.530 s | 2.49 ms |

All three source-free runs reproduce the frozen candidate rows exactly. A
producer-trace simulation starting playback at fixed-8 availability retains
at least eight seconds of coverage lead. This supports a small ahead-of-playback
buffer on this hardware, not a deployed-client deadline guarantee.

A separate cached-Mel check changes query horizon from 500 to 100 ms and
reproduces all three row sequences exactly. Whole-generation time decreases
by about 24–29%, while one dense-case p99 increases. Query sizing is a concrete
scheduler optimization to investigate before adding speculative models. The
delivered recipe remains the fully evaluated 500 ms configuration.

## Local artifact and reproduction

The local artifact owner is
`artifacts/joint-audio/20260924-expanded-v1/delivery/context-r1-playtest-v2`.
It contains the standalone inference model, model card, four `.osz` playtests,
stream/latency receipts and inference instructions. These assets are not Git
sources and may be absent in a fresh clone. Inference model SHA-256:
`1e86b79dd1144bca282a01fb798dc304094357bff5b80deeb099982749f5d48c`.

Use the packaged `joint_audio_continuation.hydra` entrypoint with
`mode=infer_audio`, a pinned checkpoint, new `audio_file`, `device=cpu`,
`cpu_threads=1`, `timing_horizon_ms=500` and `head_spacing_ms=27`. No source
beatmap, redline timing or external seed is read by that path. Each output
directory must be fresh. The playtests use constant scroll presentation and
OD 5; their original sampled note placements are independently reparsed and
unchanged by packaging.
