# Current row conditioning and short continuation support

Sequential row conditioning completes all 16 count/layout panel charts with
zero screened close pairs. Adding a 20-ms feasible-continuation check reduces
rejected proposals from six to three, within the same four-attempt publication
budget. Runtime and descriptor regression checks pass. This is a bounded
reliability improvement without fitting new weights.

It is not a complete solution to expressive generation. Requested LN response
remains weak, one inspected Hysteric context loses LN layering relative to
current-only conditioning, and both policies fail the flat model's Zenithfall
profile-2 chart. The remaining failures identify release timing as a separate
dependency: row selection cannot rescue a release event that arrives too late.

## What the two policies change

The comparison uses the frozen flat and count/layout endpoints from the
[materializer study](count_layout_materializer_evaluation.md), with their own
full-audio Mel encoding, profile conditioning, online H planner, R process and
complete-row scores. Each endpoint is run with two policies:

- **Current:** admit rows with no new same-column HH interval strictly below
  20 ms and no experimental RH interval at most 20 ms.
- **Preview:** apply the same current condition and require a feasible
  one-TAP-per-H continuation through the following 20 ms, inclusive.

HH counts TAP and LN-head attacks; exactly 20 ms is excluded. RH counts only
the first attack after a same-column release. RH remains an experimental
preference, not a universal human BAD annotation. Neither policy imposes an
LN-duration floor, a minimum H gap or an anti-Jack rule.

For the already normalized complete-row model $p$, each policy samples

$$
p_M(y\mid h)=\frac{p(y\mid h)\,\mathbf{1}[y\in M(h)]}
{\sum_{v\in M(h)}p(v\mid h)}.
$$

The mask is applied after count/layout composition. Passing it into that model's
group normalizers would define a different law. Sampling consumes one draw from
the original row RNG. Log normalization keeps a nonempty conditional sampleable
even when its retained base probability would underflow in ordinary arithmetic.
No learned parameter, H timestamp or R hazard is changed.

The preview check has a small exact implementation. All H events in
$(t,t+20]$ must use distinct columns. A column still held after the candidate
cannot be released later and become RH-eligible within that interval. The
other columns' eligible sets grow monotonically as their actual clocks advance.
If $A_j$ is the eligible set at the jth future H, a one-TAP-per-H continuation
exists exactly when $|A_j|\geq j$ for every j. Current TAP columns can become
eligible again at exactly $t+20$; a column released at $t$ cannot.

This checks finite-horizon existence, not the probability of taking that future
or liveness beyond it. The [probability audit](count_continuation_mass.md)
motivated comparing current-only conditioning rather than assuming a complex
planner was necessary. Actual R timing is intentionally unchanged in this test.

An empty row conditional is an explicit unpublished rejection. The sampler
restores its previously observed coverage; a failing H never becomes a row or
settled coverage. Buffered generation retries from its saved publication
boundary, with the original eight-second window, 20-ms halo and four proposals.
On exhaustion it retains the published prefix, including unknown LN tails.

## Corpus compatibility and complete comparison

The source audit checks every target row in the pinned 615 TRAIN and 36 VAL
charts: 683,341 rows, with zero exclusions under either policy. It uses actual
source H previews and verifies source/cache identities. This demonstrates
compatibility with the admitted corpus, not with every valid mania chart or
every human preference about LN re-presses.

The native comparison attempts every combination of eight audios, profiles 1/2,
two checkpoints and two policies: 64 calls. One seed per audio is shared across
the comparisons. Empty support and attempt exhaustion are measured outcomes;
they do not stop the next independent case or receive extra retries.

| Model and policy | Complete charts | Rejected proposals | Published HH/RH |
| --- | ---: | ---: | --- |
| Flat, current | 15/16 | 19 | 0/0 |
| Flat, preview | 15/16 | 19 | 0/0 |
| Count/layout, current | 16/16 | 6 | 0/0 |
| Count/layout, preview | 16/16 | 3 | 0/0 |

All 62 complete exports independently reparse. Every published row replays,
coverage is monotone, and every H sequence or incomplete prefix matches the
original frozen H plan exactly. All rejections are empty row support, detected
before publishing a close-pair violation. Model fingerprints remain unchanged.
Current/preview row bytes are identical in 12/16 count/layout cases and 14/16
flat cases, including the identical incomplete flat prefix.

The factor comparison meets its predefined 50% retry-reduction criterion.
Three fewer proposals in one small panel is not a population estimate of
reliability. Both flat cohorts remain incomplete; descriptor means over their
successful subsets are not substituted for whole-cohort results.

## Control and runtime

Descriptors are log H/s, log heads/H and arcsin square root LN-head fraction,
standardized with the fixed TRAIN profile bank. Error is mean squared deviation
from the requested descriptor over 16 equally weighted audio/profile cases.
It measures request realization, not musical quality.

| Count/layout configuration | H error | Width error | LN error | Width + LN |
| --- | ---: | ---: | ---: | ---: |
| Unconditioned direct comparator | .873858 | 1.301076 | 3.608154 | 4.909230 |
| Current row condition + publication | .873858 | 1.319652 | 3.822162 | 5.141813 |
| Preview row condition + publication | .873858 | 1.300254 | 3.820844 | 5.121098 |

Preview's body error is 4.3% above the raw comparator, within the predefined
10% allowance; its individual width/LN bounds also pass. This guard should not
hide weak control. Profile 2 requests .734 LN-head fraction, while preview
realizes .164 in Hysteric, .158 in Take and .094 in Yomi yori. Several profile-1
outputs still contain unrequested LN material. The policy has not fixed those
learning/calibration problems.

Cached-Mel readiness includes own online H planning and body generation until
both eight seconds of coverage and 30 complete rows are available. Its maximum
is .510 seconds for count/current and .514 for count/preview. Maximum window
service is .345 and .337 seconds respectively. Both are below the two-second
bounds. Across the flat calls, readiness is at most .490 seconds and window
service at most .560 seconds, including failed windows. These measurements
exclude fresh waveform decoding, Mel construction, client rendering and input
latency; they are not fresh-audio end-to-end startup guarantees.

## Retained organization and limits of the quality gain

Lens inspection covers all declared generated/reference scopes: 24 newly
rendered scopes and all 48 time-proportional pages, eight exact generated-scope
reuses within the comparison, and six identity-verified human-reference reuses.
Complete native-ms action and LN articulation tables accompany the inspection.
No missing generated case is replaced with a direct or parent output.

The preview As It Was profile-2 peak retains a 1356-ms hold spanning six H while
a 913-ms hold joins and closes, followed by short handoffs above the persistent
hold. A later full LN chord releases three columns before the fourth. Its
profile-1 peak has a 167/257/356-ms three-column LN group whose tails separate
while other columns tap. Fool Moon adds 82- and 166-ms holds above a 368-ms
hold and releases a subset as another LN starts. These relationships rule out
a claim that the policy only succeeds by producing single taps or isolated LNs.

Variable and repeated grips also survive. The inspected Zenithfall peak mixes
doubles/triples, repeated shared columns and a quad. Take keeps 19- and 6-ms
cross-column splits; Yomi yori keeps a 10-ms triple-to-single split. Hysteric
profile 1 has a 585-ms hold supporting eight H. Short nominal H gaps therefore
remain available without requiring a repeated attack on the same column.

Quality is nevertheless mixed. At 243–247 seconds, factor/current Hysteric
profile 2 has overlapping 677/824/585-ms roles and later 179/458-ms subset tails.
Factor/preview retains some held support and a nested pair, then becomes all TAP
from 245532 ms through the end of that crop. Flat's same context retains many
independent joins and releases under both policies. Yomi yori's constrained
profile-2 peak has no LN where the raw factor peak had a 406-ms supporting hold;
that raw-to-constrained difference is not specifically a preview-versus-current
effect. Neither these bounded contrasts nor positive examples establish a
consistent musical-quality gain across seeds and songs.

A 27-ms same-column repetition in a Fool Moon peak also remains outside the
confirmed HH criterion, without being certified comfortable. No listening,
player test or new human style approval was performed. Source labels retain
their original scopes and are not required labels for generated alternatives.

## The release dependency that row conditioning leaves open

Both flat policies stop Zenithfall profile 2 at 242310 of 357796 ms, with
997 H delivered. Their failed eight-second windows take .560/.555 seconds;
the interruption is proposal exhaustion, not observed compute saturation.

Every captured preview rejection has all four columns held. Its actual proposed
R event leaves too little time before the next H:

| Failed-window attempt | Proposed R | Next H | Gap |
| --- | ---: | ---: | ---: |
| 0 | 248976 | 248983 | 7 ms |
| 1 | 247570 | 247590 | 20 ms |
| 2 | 247580 | 247590 | 10 ms |
| 3 | 247808 | 247810 | 2 ms |

At each R query there are 15 physically legal nonempty release subsets, but none
can make a column RH-eligible by that H. Current-only conditioning instead admits
the release and detects empty support at the later H. Rejecting earlier within
the same proposal does not repair its preceding commitment.

In attempt 0, holds start at 248770 in columns 0/1, 248840 in column 3 and 248900
in column 2. All columns are held after 248900; the next H is 83 ms away, outside
the row predicate's 20-ms horizon. A suitable column would need release by
248962 under the experimental RH screen, but the actual R sampler chooses
248976. The existing physical full-hold waiting law only requires an event
before H, using a deadline of H minus one millisecond. Physical feasibility
and the joint HH/RH preference therefore impose different waiting-time support.

All three remaining factor/preview rejections have the same full-occupation
mechanism: R→H gaps of 2 ms in Hysteric profile 1, 14 ms in Hysteric profile 2
and 12 ms in Take profile 2. Resampling recovers these cases within budget; it
does not establish an invariant that the R process will release early enough.

The next coupling question concerns waiting as well as row choice. A scheduler
must preserve a feasible continuation while no event occurs, then communicate
any necessary release deadline to the actual R/row process. Such a deadline is
different from forecasting that the earliest possible R will occur. It should
preserve supported long/short LN organization and be checked on source targets
and complete generated charts before use. No new release policy is implemented
by this comparison.

## Reproduction identities

Source `ac7fa3a59696a8cf23d3825a32a7d300bdba0fbb` adds only the optional Python
row constraints, diagnostics, tests and guide. No weights are fitted. The
79 unique planned-model owner tests cover the new predicate, native failure
coverage, cache/RNG ownership, retries and existing behavior; the final focused
nine also verify conditioning when retained base mass underflows.

Execution uses Apple M5, 24 GiB RAM, Python 3.10.20, Torch 2.11.0 and one CPU
thread. The source audit takes 36.593 seconds; native generation takes 288.957
seconds. Both run inside their recorded time/resource bounds. The local owner
is `artifacts/joint-audio/20260925-current-preview-constraint-v1`; it uses about
168 MB including rendered evidence, below its 2-GiB limit. No failed case is
resumed or granted additional proposals.

Checkpoint, corpus, normalization, bank and panel identities are pinned in the
[materializer report](count_layout_materializer_evaluation.md#reproduction-identities).

| Record | SHA-256 |
| --- | --- |
| Corpus audit | `43c7c01a2b43ba472073247b0be04f9520e3c793a67b0e25570483ad36e37790` |
| Native execution freeze | `17fe384034f4bb52f2fb7bd7f12e8c4e838bb8f7c91a0ace04595dafec0c6873` |
| Native comparison | `2dcbb708c1861d3d6d58bd2ec52c8bb744fe719fd01a6d964ce7b3a27f4034c6` |
| Completed Lens review | `a2424ce15df942f121e2daff1de51a74bd5850107ce3c067afb67b3b7abca0a7` |
