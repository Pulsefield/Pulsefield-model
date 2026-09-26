# Scoped difficulty learning from a shared generated prefix

Paired low/high-request outcome learning improved R1's scoped difficulty response
slightly but failed the pilot's control criteria. It did not repair excessive
chords in a fast single-note passage or fragmented releases in the inspected LN
passage. The endpoints remain research candidates; the selected generation
checkpoint is unchanged.

This follows the [whole-chart outcome study](outcome_learning_and_control_response.md).
The [controlled architecture](controlled_audio_continuation.md) keeps H timing
separate from R1's cardinality, TAP/LN, release-subset and column decisions.
Release timing reads actual LN state. Both timing and rows receive full audio
and their applicable controls.

## What the intervention isolates

Both arms started from the same 4,583,985-parameter checkpoint. They trained only
the difficulty value, known bit and two scope clocks entering `row_control` and
the first `composition` affine: 48 columns in each 128-output matrix, or 12,288
effective weights. Parametrization protected every other weight, including
untouched columns under AdamW decay. Export folded the result into ordinary
inference weights; no inference module was added.

This can change how difficulty affects layout, composition and the learned
consequence comparison, but cannot relearn their shared representation or output
basis. It is a constrained adaptation test, not a test of all possible R1
architectures or training procedures. H/R weights and the audio/history encoders
were frozen. R times still respond to the request and actual generated LN state.

The panel had twelve TRAIN charts and two sixteen-second scopes per chart.
All five qualification audio identities were excluded from this fitting panel.
Each scope had at least 32 H events. A fixed core-generated prefix bank preserved
every row through scope start minus one millisecond, including entering holds,
recovery history and LN feedback. It contained no source materialized prefix.
Only source H timestamps were supplied as an explicit diagnostic substitution.

For each update, the outcome arm branched that same prefix into lower and higher
requests, with two independent draws per request. Corresponding low/high draws
shared random streams. All four candidates continued to true audio EOF.
The control override lasted sixteen seconds and then restored the original
whole-chart difficulty request. The LN request retained its whole-song scope.

## Outcome and probability accounting

Let $D_0$ be the source scope's offline strain readout. Requests were
$\operatorname{clip}(D_0,2.75,5.25)\pm .75$. The scoped readout keeps every object
whose head is before the scope end, its real tail and the entire incoming prefix.
It is a duration-normalized mania-strain proxy, not official local stars or a
complete playability measurement.

Difficulty cost was squared absolute error beyond a .35 deadband. Whole-chart
LN cost was 100 times squared proportion error beyond .03. Each candidate used
the other independent draw under the same request as its detached baseline.
Difficulty's row-score horizon ended once all relevant pre-boundary LNs had
closed; LN amount retained the whole-song horizon. A uniform eight-second score
interval and its interval-count importance weight estimated each suffix score.
The final partial interval was masked at its own dependency boundary.

No probability of the fixed prefix entered the conditional suffix objective.
Generated candidates were scored on their actual histories using the deployed
row distribution, including recovery and LN preferences. Source imitation used
genuine source histories and labels on the same sixteen-second scope, normalized
per second. It never attached a source action label to a generated prefix.

The outcome gradient had weight two relative to imitation. Both arms used
AdamW at `1e-3`, weight decay `.0001`, gradient clipping at one and the same
96 source-context draws. A separate four-update integration run was discarded.
The outcome arm generated 384 complete candidates. All 96 updates had nonzero
difficulty gradients; every whole-LN error stayed inside its deadband, so that
cost supplied no gradient in this particular fit.

## Matched continuation results

Qualification used five new core-generated common prefixes, two requests and
three matched future seeds for each model: 60 completed continuations. Every
committed prefix and supplied H timestamp was preserved. Export/reparse checks
preserved all rows. Reported values below are means over three seeds.

| Context | Scope, seconds | Request low/high | Imitation low/high | Paired outcomes low/high |
| --- | --- | --- | --- | --- |
| Fast singles | 56–72 | 2.190 / 3.690 | 3.811 / 4.384 | 3.738 / 4.354 |
| Slower chords | 8–24 | 2.000 / 3.500 | 2.201 / 2.264 | 2.171 / 2.278 |
| Miraie LN mixture | 160–176 | 2.019 / 3.519 | 2.775 / 2.697 | 2.649 / 2.824 |
| Starry Jet | 198.240–214.240 | 2.360 / 3.860 | 2.918 / 3.050 | 2.910 / 3.005 |
| Shippaisaku | 80–96 | 2.563 / 4.063 | 3.785 / 3.898 | 3.626 / 3.901 |

Mean absolute error decreased from .80865 to .75583, an improvement of .05282
against the required .20. Mean high-minus-low response increased from .16037 to
.25390, below the required .40. Low-request error decreased .87176→.79222;
high-request error decreased .74554→.71943. Both primary criteria failed.

Whole-LN error and restored-range error stayed within their relative regression
bounds of .03 and .25. Restoration was measured over
`[override_end, audio_end+1)`, with its first sixteen seconds also reported.
These guards compare endpoints; they do not establish absolute playable quality.
Existing human style requests retained their own scopes, independently of
difficulty and LN amount.

## What the generated patterns show

Beatmap Lens produced 76 time-proportional pages for 29 selected contexts.
Inspection covered their complete action tables and derived articulation facts,
plus eighteen selected image pages across the five comparisons and restored
suffix peaks. It was not a full visual review of every exported song, listening
session or player approval. Human source labels were not copied to outputs.

In the fast-singles range 56271–60271 ms, source and both candidates share 50 H
events. The source has 53 heads; the low/high candidates have 71/78, including
six three-key rows each. A higher-request seed develops repeated `[01]`/`[23]`
chords through roughly 61479–62644 ms. Lowering the request has not restored the
predominantly single-note organization of this passage.

In the slower-chord comparison at 8091–12091 ms, the source has 45 heads on 23 H
events; candidates have 35/38. Their early rows are nearly identical across the
two requests. This is weak composition response, not a missing ability to emit
chords at all.

Miraie's 162307–166307 ms source combines 366-ms interleaved holds with 183-ms
mixed TAP/LN rows. All 28 LNs starting in that source range end at H times.
Low/high outputs have 30/33 LNs with median durations 106/92 ms; only 4/5 end at
H times. Their at-most-80-ms shares are 23.3%/27.3%, versus zero in the source.
These statistics describe the inspected fragmentation; they do not define a
universal minimum musical LN length. Shippaisaku's source, for example, uses
many legitimate 110/111-ms holds in recurring mixed-length relationships.

Expression did not disappear uniformly. Starry Jet's high-request sample holds
column 3 from 199647 to 200324 ms while six intervening H events act elsewhere.
Another hold from 203449 to 204171 spans three H events. Shippaisaku's low sample
retains a 662-ms hold across four H events amid shorter holds. Both also add
independent short releases. Neither scalar difficulty nor LN fraction captures
these relationships adequately.

## Fixed timing versus unused action support

Constructive diagnostics kept the same generated prefixes and every H timestamp.
For entering holds, they retained an already-generated continuation until the
first state with no open LN, rather than inventing endpoints. Subsequent TAP
rows preferred one, two, three or four heads, with least-recently-used available
columns, under the same exact row support and 60/50/50-ms recovery profile.
These are support witnesses, not learned policies, optimal bounds or quality
targets.

The singles one-head witness attains 2.834, below the sampled 3.738 but above
the requested 2.190. It shows remaining R1 latitude without proving whether
2.190 is attainable under fixed H. The chord witnesses attain
1.539/2.663/4.077/4.994. Therefore fixed H does not force that case to remain near
2.28, although no claim is made that uniform chord cardinality is desirable.

The outcome pilot is insufficient to choose between limited adaptation capacity,
small-panel coverage, estimator variance and insufficiently learned conditional
structure. Increasing its gain or training duration alone is not justified by
these results. The [control-coverage investigation](control_condition_learning.md)
examines an alternative using real conditional variation in the corpus.

## Provenance

Executable source: `2c2b8af0da7411804697863df2f8efb605f3c840`.
Artifact owner: `20260926-paired-scope-controls-v1`.
Initial checkpoint SHA-256:
`0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8`.

- Source-imitation endpoint: `e6deaf34613a1a1851289d8dd3e2aa1901f11a3cad9219acd50bbdfa3d8a1ba4`.
- Paired-outcome endpoint: `9f0f5a862793d7871179e21714bd09b41a30eaeacdab3a90eca3e5c7c081b73c`.

On the 24-GiB Mac M5 with PyTorch 2.11/MPS and one CPU thread, fitting took
36.26/635.69 seconds. Peak sampled process physical footprint was 2.21/2.78 GiB;
MPS driver allocation was 1.25/1.28 GiB. These ledgers overlap. Maximum supported
CPU-generation/MPS-rescoring log-probability difference was $2.29\times10^{-5}$.
No native-H expansion, startup-latency qualification or new model adoption was
performed after the failed fixed-H criteria.
