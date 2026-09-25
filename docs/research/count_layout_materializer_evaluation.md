# Separating row counts from column realization

An explicit row-count conditional improves mean width/LN descriptor error by
22.5% relative to a matched flat-row continuation, but produces more same-column
attacks below 20 ms and exhausts the publication retry budget in Hysteric.
Some independent LN relationships survive; others weaken in the inspected
paired contexts. The tested factorization is not a playable-system improvement.

The failure exposes a specific coupling problem. After the count model assigns
probability to a head count, normalizing R1 within that count prevents R1 and
frontier2 from reducing its total probability. Column realization cannot rescue
a count that exceeds the available rested columns. Preserving every legal row
does not preserve the former model's joint policy over counts and layouts.

## Conditional structure and controlled comparison

The [planned model](planned_audio_continuation.md#optional-count-and-layout-factorization)
generates attack times H, release-only times R, and complete four-column rows Y.
An H row can also release holds. The flat materializer normalizes its scores
over all physically legal complete rows. The alternative defines the mark
$m(y)=(\#\text{heads},\#\text{LN starts},\#\text{releases})$, with 35 possible
triples across the 256-row vocabulary, and uses

$$
p(y\mid c,h)=q(m(y)\mid c)\,
\frac{\exp s_{R1}(y\mid h)}
{\sum_{v\in G_{m(y)}}\exp s_{R1}(v\mid h)}.
$$

$G_m$ contains physically legal rows with mark $m$; empty groups receive zero
mass. The count condition $c$ contains full profile-conditioned audio, the
16-H preview, a 31-row causal history of elapsed time and count triples, the
last-row age, and sorted active LN ages. It contains neither tap-column
assignments nor generic R1 row embeddings. R1's condition $h$ retains direct
audio, the preview, its 511-row history, exact state and frontier2. The complete
256-way law is sampled once with the original row RNG.

Both arms start from the same shared-profile checkpoint. Full-audio encoders,
profile projection/prior and all H modules are frozen in both arms. R and row
materialization are trained; the alternative adds 166,339 parameters, for
4,416,513 total versus 4,250,174. This common freeze isolates downstream learning
from H drift; it is not a proposed final requirement to freeze audio. R retains
its audio, skeleton-history and committed-LN input contract. The pilot H model
still ignores LN state; that is a stronger approximation than a general
skeleton model that may read committed LN occupation.

Each arm receives the same 4,800 eight-second-or-shorter intervals over 1,200
updates: 37,392,112 ms, 239,124 H rows and 14,139 R rows. The corpus contains
615 TRAIN arrangements from 240 audio groups and 36 VAL charts. Profiles 1 and 2
occur in only 116 and 133 training intervals. Both endpoints are fixed by the
update budget, not chosen by NLL. Training uses source histories and source H
previews; inference feeds back each arm's sampled counts, rows and LN state.
Full audio is available in both phases. No source chart or redline enters
native generation.

Frozen parameter bytes remain identical after training and checkpoint reload.
Every attempted native call reproduces the original H plan or its published
prefix exactly. Shared seeds and exposure do not keep downstream histories
identical after sampled actions diverge. Global gradient clipping also couples
the magnitude of R updates to the changed row gradients, despite separate
neural inputs; R changes are not isolated evidence of a semantic count effect.

## Descriptor response and system results

The native panel contains eight pinned audios and two explicit profile requests,
with one seed per audio. Error is the equally weighted mean squared difference
between requested and realized TRAIN-standardized descriptors: log H/s, log
heads/H and arcsin square root LN-head fraction. These measure control response,
not musical quality or playability.

| Direct generation, 16 charts per arm | H error | Width error | LN error | Width + LN |
| --- | ---: | ---: | ---: | ---: |
| Original checkpoint | .873858 | 2.370422 | 4.096142 | 6.466564 |
| Flat continuation | .873858 | 2.563333 | 3.770444 | 6.333777 |
| Count/layout continuation | .873858 | 1.301076 | 3.608154 | 4.909230 |

The alternative passes the predefined numerical criterion: width + LN error
must be at most .85 times the flat result and no worse than the original; each
component must be at most 1.10 times flat, with H unchanged. Width improves
49.2%, while LN improves only 4.3%. Five audios improve their mean body error;
Hysteric, Take and Yomi yori worsen. There is no confidence interval establishing
generalization beyond this small, single-seed panel.

The same profile can improve in aggregate while losing a desired behavior.
Profile 2 requests an LN-head fraction of .734. Whole-chart direct fractions
change from .712 to .226 in Hysteric, .539 to .219 in Take and .463 to .084 in
Yomi yori. Reducing unrequested LN under profile 1 can compensate numerically
for weaker profile-2 LN realization. An aggregate descriptor improvement does
not establish successful independent control or expressive organization.

HH counts successive same-column TAP/LN-head attacks strictly less than 20 ms
apart; exactly 20 ms is excluded. The separate experimental RH screen counts
the first attack after a same-column release at a gap of at most 20 ms. Neither
criterion bans close cross-column events, ordinary Jack/chordjack, or short LNs.

| Direct generation | HH pairs | RH pairs | Rows containing HH |
| --- | ---: | ---: | ---: |
| Flat | 2 | 48 | 2 |
| Count/layout | 24 | 16 | 18 |

All 32 direct charts complete. Screening then uses the unchanged eight-second
window, 20-ms halo and four-proposal limit. Nine screened charts complete; the
next call, count/layout Hysteric profile 1, exhausts the attempt limit. The
remaining 22 screened calls are unattempted. Both screened cohort aggregates
are incomplete; successful subsets cannot replace the planned 32 calls.

That failure leaves coverage at 243554 of 301008 ms, with 1,716 H delivered.
The failed window takes .744 seconds, below the two-second bound. Across all
attempted calls, cached-Mel audio/H/body readiness is at most .542 seconds and
window service at most .876 seconds. These are not fresh-audio startup times.
The observed interruption is proposal exhaustion, not compute saturation.
Zero HH/RH in the published prefix does not guarantee continued playback.

## Why count and layout cannot repair each other

For each direct HH row, the audit counts columns that are free immediately
before the row and whose previous attack is at least 20 ms old. A count failure
means some such columns exist, but fewer than the selected number of heads.
A layout failure means enough exist but a recent column is selected. A past
commitment failure means none exist.

Seventeen of the count/layout arm's 18 bad rows exceed current count capacity;
the other is a layout failure. Flat has one layout failure and one past
commitment failure. These classify realized decisions, not total probability
mass or a population percentage attributable to restored R1.

The mechanism follows directly from the conditional. Adding any common score
bias to all rows in $G_m$ cancels in its normalizer. Frontier2 still ranks
layouts within a mark, but its cross-mark preference cannot reduce $q(m\mid c)$.
Once that marginal selects too many heads, every layout in the group violates
the HH condition, even if the rows remain physically legal. Physical legality
tracks key occupation and valid releases; it does not include the HH screen.

Actual rejected trajectories make the future dependency visible. All four
proposals in the failed Hysteric window repeat a column across
248930 → 248935 ms. Full trajectories were retained for its first three
attempts; the fourth has conflict metadata only because an earlier rejection
used one of the four capture slots.

| Failed-window attempt | Actions at 248930 ms | Actions at 248935 ms | Constraint |
| --- | --- | --- | --- |
| 0 | TAP in all four columns | TAP in 0/2/3 | The earlier quad leaves no rested column. |
| 1 | TAP in 0/3 | TAP in 1/2/3 | Only two columns are rested for three heads. |
| 2 | TAP in 0/1/2; LN3 remains held | TAP in 0/1 | Three recent attacks plus the old hold occupy all options. |

In attempt 2, LN3 starts at 248824 and releases at 249047 ms. A possible R
opportunity at `now + 1`, used by the existing frontier approximation, does not
mean an actual release will occur before the next H. For HH alone, releasing
LN3 between the two H rows could make column 3 available. The separate RH rule
would still require an earlier tail or a different earlier count. Changing
only the final layout cannot solve all these cases.

A useful exact capacity relation does not require generic tap-layout feedback.
For a physically legal prefix with no repeated column attacks within
$(t-20,t)$, let $C(t)$ count all TAP/LN heads in that open interval and $O(t)$
count currently held LNs at least 20 ms old. Then the number of columns free
and rested immediately before the row is

$$
K(t)=4-C(t)-O(t).
$$

Recent LN heads already appear in $C$; older held columns are disjoint from
those recent attacks. Exactly-20-ms-old heads are outside $C$; still-held LNs
of that age remain in $O$. A column released by the current complete row cannot
also attack in that row. The native audit verifies this identity wherever the
prefix condition holds. The count network does not explicitly compute or
enforce it, and learned compression of count history need not retain it exactly.

This identity addresses current capacity, not future viability, RH, hand
comfort or musical appropriateness. A current-only mask can still permit a
quad immediately before another H or rely on a release that never occurs.
Any feasibility mechanism must account for earlier counts and actual LN
release choices before publication.

## What remains expressive in inspected charts

The review covers 32 new scopes, all 60 time-proportional pages and complete
native-ms action/LN tables, plus six identity-verified human-reference reuses.
Scopes include the predefined paired contexts, selected dense windows and the
four shortest direct HH witnesses per arm, with overlaps merged. Flat has only
two direct HH pairs. Direct fallbacks are explicitly descriptive and do not
replace missing screened charts.

The Hysteric profile-2 context at 243–247 seconds illustrates a regression in
organization. Flat contains a 906-ms entering hold with later holds joining and
closing, and a 797-ms hold spanning three H while other columns add and release
subsets. Count/layout contains useful 458- and 469-ms holds with taps beneath,
but sequential single-held roles replace that independent overlap. As It Was
at 123–127 seconds replaces flat's 352/338/348-ms LN handoff with TAP doubles.
These bounded differences concern relationships, not only LN fraction.

The representation has not lost all coordination. Count/layout As It Was's
profile-2 peak contains a 1356-ms hold spanning six H, a 913-ms hold joining and
releasing while the first remains held, and shorter handoffs above it. A later
quad LN releases three columns before the fourth. Fool Moon's screened peak
adds 82- and 166-ms holds over a 368-ms hold and releases a subset while another
hold starts. These are counterexamples to a claim of universal LN collapse.

Variable tap widths, changing grips and ordinary repetition also remain.
As It Was profile 1 combines a 517-ms hold spanning four H with changing
single/double/triple/quad accents. Hysteric profile 1 contains a 585-ms hold
supporting eight H. Take has 19- and 6-ms cross-column splits without HH;
Yomi yori has a complementary two-group split at 10 ms. A universal minimum H
spacing would destroy valid representational choices.

Conversely, passing HH is a narrow test. Fool Moon's screened peak includes a
27-ms repeated column between changing chords; that remains outside the
confirmed threshold, without being certified comfortable. No listening,
player test or new human style approval was performed. Source judgments retain
their original scope and do not require a generated alternative to copy labels.

## Limits and next causal question

The result rejects this fitted count-marginal replacement as an overall
improvement. It does not reject count/layout factorization in general. The new
conditional starts with untrained weights, receives sparse profile-specific
exposure, and reads a frozen audio representation plus compressed count history.
Sorted LN ages omit hand identity, which may matter for desirable composition
even when anonymous capacity is recoverable. Source-history likelihood does not
test decisions under generated H, occupation and count histories.

On 144 fixed VAL intervals, row NLL/s is 9.880 for flat and 11.135 for the
alternative; R NLL/s is 2.028 and 2.034. These diagnostics are compatible with
incomplete learning, but neither a lower NLL nor better descriptor error would
establish playable continuation. No checkpoint was selected using these values.

The next causal question is whether count selection can preserve learned
musical organization while assigning mass only to continuations whose counts,
column occupation and releases can be realized through the upcoming H plan.
That requires separating exact mechanical state from learned preferences and
testing actual future release decisions. More independent retries, a generic
LN floor or a current-row HH mask alone would not resolve the demonstrated
dependency. A proposed solution still needs whole-chart coverage and LN/tap
organization checks under the same runtime budget.

The [fixed-prefix continuation audit](count_continuation_mass.md) measures
substantial future failure mass even when the current row is clean. It separates
mechanically impossible futures from possible futures the sampler rarely
chooses; sequential current-only conditioning remains an untested alternative.

## Reproduction identities

Implementation source is `4567d87732320f27c01b40b15f74846865219934`.
Training uses Apple M5, 24 GiB RAM, Python 3.10.20, Torch 2.11.0 and MPS;
native generation uses one CPU thread. Flat and count/layout fits take
799.512 and 910.983 seconds. The guarded native driver ends after 198.096
seconds. No capped call is resumed or granted more proposals. All 41 complete
outputs independently reparse; incremental replay and H identity checks pass.

The local artifact owner is
`artifacts/joint-audio/20260925-count-layout-materializer-v1`.
Canonical implementation and [focused tests](../../tests/research/planned_audio_continuation/test_counts.py)
are tracked; checkpoints, manifests and generated charts are local assets.
The owner tests establish exact group marginals/ratios, finite gradients,
input boundaries, CPU/MPS agreement, cached history, forks, training freeze
and checkpoint/Hydra round trips. Seventy unique owner tests passed before
the frozen experiment; this report makes no additional model change.

| Asset | SHA-256 |
| --- | --- |
| Initial checkpoint | `abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef` |
| Flat endpoint | `da08044806e99eae6a6221c06ef8c91c02711361fae68c896c9af01bde61dea2` |
| Count/layout endpoint | `41ab71640ce9571ac7f40d9f851a51e2ba54c10ba97fc6f9e54dbc54fac76185` |
| Corpus manifest | `4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1` |
| Normalization | `9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287` |
| Profile bank | `a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03` |
| Audio panel | `484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9` |
| Main training result | `debcbdf58f99e5359070465a8e77384874665008d7fb5a4f02542e33fc524908` |
| Native result | `43d130ed6291abdebc3c994e1ad27b833954ef84deed5f9dfcc8422e4a60b5ff` |
| Completed Lens review | `f804febd5204ee69b7299666dc030c50586243092eee221428388f40ace5d4f0` |
