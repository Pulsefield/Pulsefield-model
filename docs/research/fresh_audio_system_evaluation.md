# Fresh-audio generation with unpublished-window screening

The audio-file generator completes 24 screened charts on eight additional audio
contexts, preserving every paired head time and removing the two observed
same-column attack pairs below 20 ms. Sampling costs 3.2% more than direct
generation. This establishes a bounded local system result, while inspection
and realized profile statistics expose substantial arrangement-control errors.
It does not establish a final playable model.

## Comparison and information contract

The comparison uses A Fool Moon Night, Operation: Zenithfall, Hysteric Night
Girl, Goodbye, Revenge remix, Take, As It Was and Yomi yori. Their decoded
durations range from 144.236 to 498.989 seconds. The eight catalog validation
groups, encoded audio hashes and peak-normalized decoded waveform hashes are
absent from the selected joint training/validation corpus. Exposure in the
earlier R1 lineage and perceptually identical alternate recordings are unknown.
The panel is curated for inspectable musical contexts, not a random sample of
all songs.

Each audio receives three arrangement requests: the learned automatic prior
and fixed TRAIN representatives 1 and 2. Each request is generated with direct
decoding and with [unpublished-window screening](unpublished_continuation_screen.md),
giving 48 API outputs. Paired policies share audio, weights, seed and arrangement
choice. Head, release and row factors receive the same profile; there are no
synthetic condition crosses. No weights are fitted in this comparison.

Every call loads the checkpoint, decodes the original full audio and computes
canonical Mel anew. Generation starts at BOS without a chart seed, target H
times, redlines, style labels or future LN endpoints. The complete audio
conditions both skeleton and row materialization. Release timing retains its
minimal committed LN-state input. Rows retain direct audio, future H preview,
causal row history and frontier2. The stronger pilot restriction that H does
not read LN state remains unchanged.

The screened policy uses eight-second publication windows, a 20-ms halo and at
most four proposals per window. It checks strictly sub-20-ms successive attacks
in the same column, including TAP and LN heads. Exactly 20 ms is excluded.
It also checks release-to-next-head gaps at most 20 ms, an experimental
corpus-calibrated preference rather than a universal human bad-pattern label.
Cross-column timing and LN duration are separate quantities. No generic onset
spacing or minimum LN duration is imposed.

An independent incremental consumer replays accepted rows and settled coverage,
tracks open holds and compares its rows with the final artifact. Only complete
outputs are exported and independently reparsed. Unresolved LN endpoints are
allowed during publication. The packaged CLI repeats the longest automatic
screened case, with stdout consumed online; it is an integration control rather
than an additional independent quality sample.

## Completion, latency and workload

Readiness means at least 30 physical rows and eight seconds of settled chart
coverage, or actual completion for a shorter chart. Producer readiness includes
checkpoint loading, audio decoding, Mel computation and generation, but excludes
Python process startup. Consumer arrival includes callback delivery. The CLI
measurement starts at subprocess launch. OS file caches are not evicted.

Measurements use Apple M5, 24 GiB RAM, Python 3.10.20 and Torch 2.11.0, with CPU
inference and one Torch CPU thread. Sampler time includes audio encoding and
generation; it excludes waveform/Mel preparation and final export.

| Measurement | Direct, 24 charts | Screened, 24 charts |
| --- | ---: | ---: |
| Complete and independently reparsed | 24 | 24 |
| Strict same-column attack pairs below 20 ms | 2 | 0 |
| Experimental release-to-head pairs at most 20 ms | 8 | 0 |
| Total sampler time | 83.982 s | 86.680 s |
| Producer readiness from fresh audio input | — | 0.577–1.576 s |
| Consumer readiness from API call | — | 0.623–1.625 s |

Every paired H sequence is identical. Eleven proposals are rejected; 18 pairs
need no retry and remain entirely row-identical. The largest unpublished
proposal contains 105 rows, and maximum accepted-window service is 0.356 s.
No rejected row, speculative coverage or fabricated endpoint reaches the
consumer. The additional CLI run becomes ready in 2.921 s from process start;
its rows match the API and its stdout matches the saved event stream exactly.

Playback-clock replay starts at observed readiness. All screened arrival traces
have zero coverage starvation at 1x speed. Separate checks with fourfold service
cost, an injected one-second stall and two seconds of visual lead also have zero
starvation. These are offline trace checks, not measurements of client rendering,
network delivery or OS contention. The complete API/CLI driver takes 217.125 s.

## What the inspected charts retain

The predefined inspection covers all 16 human review records through 12 merged
source contexts, corresponding generated contexts for every request/policy,
the densest automatic eight-second windows, every resampled publication window
and both direct short-attack witnesses. All 209 new time pages across 75 scopes
and their complete action/articulation tables were read. Another 33 scopes reuse
exact action, entry-hold and articulation identities. This is selected-scope
inspection, not whole-song visual review or player testing.

The As It Was witness makes the narrow intervention concrete. Direct profile 1
attacks columns 1/2/3 at 92196 ms and column 2 again at 92215 ms. The screened
output keeps both H times and assigns the latter attack to column 0. It also
retains four consecutive 0/1/2 triples at 93237, 93410, 93584 and 93755 ms.
Ordinary repeated grips therefore survive alongside removal of the 19-ms
same-column conflict. Cross-column splits of 3, 7 and 17 ms remain in inspected
As It Was scopes; such gaps are not themselves the bad-pattern definition.

Expressive LN mechanisms remain available. Zenithfall's automatic early scope
contains repeated broad LN chords with independently staggered tails: a triple
at 14578 ms lasts 112/85/60 ms, and a quad at 15755 ms lasts 78/78/222/222 ms.
This gives a concrete relation to the human repeated-LN-chord reference without
requiring the same arrangement. Other inspected outputs sustain a held column
through changing taps, overlap LN starts and release subsets independently.

The source contrasts prevent LN counts from standing in for coordination.
Fool Moon's prominent LN-coordination reference combines repeated 108/109-ms
heads with overlapping 217/435-ms holds. Generated profile 2 in the matching
first context instead uses a single-LN relay: every new LN ends at the next H,
with no intervening head. Take's source Trill is a regular six-row exchange
between column 0 and the complete 2/3 group; the generated matching scopes do
not establish that figure. These differences describe missing demonstrated
organization, not a requirement to copy the reference or force its tags.

Only one human record has explicit High confidence; missing confidence remains
unrecorded. Negative tags apply to their annotated scopes. No dump-positive
label is inferred. Generated files' editor-only 120-BPM timing headers do not
provide musical beat truth. No listening or player test has established audio
alignment, comfort, difficulty or dump's acoustic elaboration.

## Control error remains after mechanical screening

The [profile model](shared_arrangement_profiles.md) supplies chart-level soft
descriptors, not commands for each local window. Profile 1 requests 7.032 H/s,
2.141 heads/H and zero LN-head fraction. Profile 2 requests 6.060 H/s,
1.429 heads/H and .734 LN-head fraction. The following are realized whole-chart
statistics; local absence of LN or chords alone would not establish a failure
of a chart-level request.

| Audio | Profile 1 heads/H, direct → screened | Profile 2 LN fraction, direct → screened |
| --- | ---: | ---: |
| A Fool Moon Night | 2.087 → 2.087 | .159 → .159 |
| Operation: Zenithfall | 1.645 → 1.360 | .242 → .242 |
| Hysteric Night Girl | 2.052 → 2.038 | .334 → .523 |
| Goodbye | 2.065 → 2.065 | .247 → .247 |
| Revenge remix | 1.032 → 1.032 | .078 → .078 |
| Take | 2.411 → 2.411 | .177 → .144 |
| As It Was | 2.526 → 2.590 | .084 → .084 |
| Yomi yori | 2.109 → 2.109 | .074 → .074 |

Profile 1's realized H rate also ranges from 2.938 to 7.209 H/s across these
audios. Revenge's inspected profile-1 context contains only single taps,
consistent with its whole-chart width of 1.032. Broad repeated chords occur in
As It Was under the same request. Thus poor control calibration is present
before screening and persists in trajectories that never need a retry.
The comparison does not isolate whether learned condition use, training
exposure, audio generalization or their interaction is the dominant cause.

One predefined descriptive composition flag fires: Hysteric profile 2 changes
LN fraction by +.1886, with 1.045 times as many note heads. All paired head-count
ratios remain within .8–1.25. Inspection finds more overlapping, independently
released LN material in the changed Hysteric scopes, alongside isolated 12–18-ms
LN objects. Higher LN fraction is neither an automatic improvement nor a bad
chart verdict; those short objects expose an articulation question that the
HH/RH screen does not answer. This does not establish a universal duration floor.

Zenithfall also changes outside its resampled window: a later direct 3480-ms
held-lane role becomes short isolated relays. Changed autoregressive history
and RNG continuation are both involved. As the
[state/stream diagnostic](continuation_state_dependence.md) explains, one paired
trajectory cannot establish a persistent collapse mechanism or population bias.

The system result supports keeping bounded screening as a research option.
It gives no reason to equate lower NLL, successful export or zero close pairs
with playable musical organization. The next model comparison needs to explain
and improve condition response, with a matched continuation-training control
before attributing gains to new condition routing. Local musical relationships
and full-chart realized descriptors must both remain in evaluation. No model
weights or default decoding policy are promoted by this result.

## Evidence identity

Implementation: `0b35eced5a8118da745a84dccee75e8b0e503bed`.
Unchanged checkpoint SHA-256:
`abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef`.
Local evidence owner: `artifacts/joint-audio/20260925-fresh-audio-system-v1`.
The assets are not required repository files.

Panel SHA-256:
`484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9`.
Native result SHA-256:
`9f3c4580adaa6239c44e04bb5d95a0ed921b8fd958b1eee26ccd8cedfa91e862`.
Completed scope review SHA-256:
`9302f23daed2ce010465437c1a3d90793754403035ac447121619fe91da5e737`.
The immutable native result records qualitative review as pending at generation
completion; the separate review records the subsequently completed inspection.
