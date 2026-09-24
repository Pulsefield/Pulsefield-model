# Requested density routing and generated arrangement stability

Removing requested head density from the release and row conditions worsens
arrangement control in a matched continuation experiment. Both models complete
all generated charts, and their source-conditioned likelihoods are similar,
but the changed model has larger descriptor errors, more short same-column
attack conflicts and weaker LN organization in several inspected contexts.
Neither continuation endpoint is adopted as a playable model.

## Intervention and information flow

The [shared-profile model](shared_arrangement_profiles.md) conditions head
timing, release timing and complete-row materialization on three chart-level
descriptors: head rows per second, heads per head row and LN-head fraction.
Let $z=(z_H,z_W,z_L)$ be their standardized representation, $a_t$ the encoded
full-audio condition at time $t$, and $W$ the shared linear profile projection.
The comparison changes only which descriptor coordinates enter that projection:

| Factor | Shared continuation | Routed continuation |
| --- | --- | --- |
| Head timing H | $a_t+Wz$ | $a_t+Wz$ |
| Release timing R and rows | $a_t+Wz$ | $a_t+W(0,z_W,z_L)$ |

Both arms retain full audio, generated H preview, R's committed LN-state
projection, causal row history and frontier2. The pilot H generator depends on
its own history and audio, independently of materialized row content. Row
generation directly reads audio as well as the skeleton. Both conditioned views
come from the same unconditioned encoding; complete audio is encoded once.
No timing grid, future target tail or source chart enters native generation.

At fixed weights and fixed audio/H/state, the routed R/row scores are invariant
to the removed coordinate. That implementation property does not make trained
H plans invariant: R/row losses still update the shared audio encoder and
profile projection. The intervention also removes information that finite H
preview may not recover about future global density. These are two distinct
possible explanations of a failed routing change.

Both arms start from the same profiled checkpoint, copying every tensor,
including the learned audio prior, and use fresh optimizers. Each receives
1200 updates, two songs per update and two eight-second intervals per song.
The exposure plan is identical: 4800 intervals, 37,396,141 sampled milliseconds,
240,283 H rows and 14,600 release rows. TRAIN contains 615 arrangements in 240
audio groups; 36 charts are held out for likelihood evaluation. Alternative
arrangements remain separate targets. The model has 4,250,174 parameters in
both arms. Learning rates are 3e-4 for new modules and 3e-5 for inherited modules,
with weight decay .01 and gradient clipping at 1.

Fitting takes 906.859 seconds for shared conditioning and 1041.464 seconds for
routed conditioning on Apple M5 MPS with 24 GiB RAM, Python 3.10.20 and
Torch 2.11.0. Native inference uses one CPU thread. A separate 32-update
preflight checks execution integrity; the main runs restart from the original
checkpoint. No endpoint is selected by NLL.

## Native control differs despite similar likelihood

Evaluation uses the eight audios from the
[fresh-audio study](fresh_audio_system_evaluation.md): A Fool Moon Night,
Operation: Zenithfall, Hysteric Night Girl, Goodbye, Revenge remix, Take,
As It Was and Yomi yori. Each model generates automatic selection and explicit
TRAIN representatives 0–3, giving 80 direct full-chart outputs. Audio, requests
and per-audio seeds are paired across arms. Every output completes and
independently reparses.

The primary comparison uses the 32 explicit requests per arm, weighting audio
and request uniformly. Error is the mean sum of squared differences between
requested and realized descriptors, using the TRAIN transformations and scales.
The transformations are log H rate, log heads/H and arcsin square root LN
fraction. This measures soft-control calibration, not difficulty or playability.

| Measurement | Shared | Routed |
| --- | ---: | ---: |
| H-rate component error | 1.994592 | 3.610711 |
| Chord-width component error | 2.257701 | 3.930808 |
| LN-fraction component error | 1.944378 | 3.000517 |
| Total descriptor error | 6.196672 | 10.542035 |
| Centered H control gain | .3551 | .0117 |
| Centered width control gain | .3091 | .2294 |
| Centered LN control gain | .4930 | .2729 |
| Finite requested-density → realized-width coefficient | −.06638 | −.11051 |
| Finite requested-width → realized-width coefficient | .20757 | .10697 |
| Reference-profile conditional VAL NLL per second | 40.1827 | 40.3493 |

Centered gain compares within-audio changes with a constant-output predictor;
zero gain would not establish use of a control. The response coefficients fit
the four requested standardized profiles and average over audios. They are
finite responses, not infinitesimal derivatives or causal path attributions.

Routed descriptor error is higher on every audio. It fails the predefined 15%
improvement in total and width errors, the 10% H/LN non-regression bounds and
the criterion of reducing density-to-width response while retaining own-width
response. Similar teacher-forced NLL does not resolve this native difference.

Ordinary additional training is also not an established improvement. On the
exactly matched eight-audio, requests-1/2 subset, total error is 7.340423 for the
initial checkpoint, 9.191414 for shared continuation and 15.914283 for routed
continuation. Shared fitting improves the LN component but worsens H and width.
These 16-output values use a different request set from the primary comparison
and must not be compared directly with its 32-output averages.

## Short attacks depend on the composed continuation

The human bad-pattern criterion is successive attacks in the same column
strictly below 20 ms, including TAP and LN heads. Exactly 20 ms is excluded.
Call these HH pairs. Release-to-next-head gaps at most 20 ms are a separate
experimental preference, RH. Cross-column timing and LN duration are separate
quantities; no generic onset-spacing or minimum-hold-duration rule is imposed.

Across the 40 direct calls per arm, shared produces 8 HH and 10 RH pairs;
routed produces 38 HH and 12 RH pairs. These totals include exact automatic/
explicit duplicates, so they are not counts of independent samples. The 46 HH
pairs occur in 43 rows. Native replay classifies the actual prefix at each row:

| Available columns under the strict HH criterion | Shared bad rows | Routed bad rows |
| --- | ---: | ---: |
| Enough rested, free columns for the chosen number of heads | 4 | 25 |
| Some rested, free columns, fewer than the chosen heads | 1 | 8 |
| No rested, free column | 2 | 3 |

This separates immediate layout errors from consequences of earlier decisions.
For example, routed Hysteric attacks columns 0/1 at 152363 ms and 2/3 at
152371 ms, then repeats column 0 at 152381 ms. The last row has no rested
column. Earlier chord sizes must change if all three H times are retained.
Shared Zenithfall holds column 1 while a triple occupies 0/2/3 at 108489 ms;
a new LN head on column 0 at 108501 ms then repeats after 12 ms. The sampled
release of column 1 does not occur in time to create another choice.

An HH-only substitution can also exchange one problem for another. Routed
Goodbye attacks column 1 at 244189 ms and starts an LN there 14 ms later.
Columns 2/3 are HH-rested, but were just released at 244189 ms. Reusing them
would violate the experimental RH preference. Earlier chord/release decisions
are relevant to satisfying both criteria.

None of the 80 generated H plans contains five H within a strictly sub-20-ms
span. For one TAP per H in four columns, this condition is necessary by the
pigeonhole principle and sufficient by cyclic column assignment. This only
proves weak HH feasibility. It does not supply a good chart or establish
feasibility with requested chords, held lanes, release articulation or music.
Changing earlier rows can change the prefix classifications above.

The [unpublished-window screen](unpublished_continuation_screen.md) is evaluated
on Fool Moon, Hysteric, Revenge and As It Was, for automatic selection and
requests 1/2: 12 additional outputs per arm. It tests actual R/row continuations
before publication, preserving the H plan.

| Measurement on the screened subset | Shared | Routed |
| --- | ---: | ---: |
| Corresponding direct HH / RH pairs | 1 / 2 | 13 / 3 |
| Screened HH / RH pairs | 0 / 0 | 0 / 0 |
| Completed and independently reparsed | 12 | 12 |
| Rejected proposals | 4 | 21 |
| Maximum cached-Mel 30-row/eight-second readiness | .493 s | 1.120 s |
| Maximum publication-window service | .813 s | 1.375 s |

All 24 paired H sequences are identical. Screening is not tested on all 80
direct charts here; it therefore cannot be credited with removing all 46 HH
pairs. Readiness includes encoding and generation from cached canonical Mel,
not fresh audio decoding or process startup. Total direct sampler time is
212.360 seconds shared and 258.117 seconds routed, within the predefined 2x
bound. These are local producer measurements, not client or network latency.

## What the chart inspection establishes

Lens inspection covers every direct HH witness with one-second context,
corresponding screened human-reference contexts and the densest automatic
eight-second windows. All 180 new time pages across 84 scopes and their complete
native-ms action/LN articulation tables were read. Six reference scopes reuse
exact identities from the completed parent review; one duplicate generated
scope reuses another inspected scope. All 16 human records were retrieved.
This covers the declared scopes, not every page of every full song.

The LN request illustrates a structural loss. Profile 2 requests 6.060 H/s,
1.429 heads/H and .734 LN-head fraction. Its screened whole-chart results are:

| Audio | H/s, shared → routed | LN fraction, shared → routed |
| --- | ---: | ---: |
| Fool Moon | 3.661 → 13.639 | .167 → .042 |
| Hysteric | 14.674 → 16.209 | .101 → .039 |
| Revenge | 2.920 → 1.710 | .672 → .184 |
| As It Was | 6.001 → 7.340 | .465 → .032 |

In the shared Fool Moon context, a 6458-ms hold supports other-column taps,
new overlapping holds and subset releases. Shared Hysteric at 243545 ms keeps
column 1 held while other columns enter and exchange shorter holds; columns
0/3 later release together while 1 remains. Shared Revenge retains staggered
holds, including a 564-ms hold spanning three other H. Routed versions of
these contexts are predominantly or entirely TAP. Shared Hysteric's high H
rate still misses the request, so these positive local relations do not make
that endpoint calibrated.

The routed model nevertheless retains some expressive capacity. Its automatic
Hysteric peak has many overlapping holds, independently staggered releases,
and a 374-ms hold spanning five H. Its As It Was wide-profile context contains
repeated outer doubles and changing chord groups. Shared As It Was retains
recurring triples under the wide-profile request and independently overlapping
LN roles under profile 2. The models cannot accurately be described as capable
only of single taps.

Fine cross-column timing survives screening. Routed Hysteric profile 2 attacks
0/3 at 250802 ms, 1 at 250809 ms and 2 at 250819 ms: four heads across 17 ms
without repeating a column. Shared outputs also retain splits below 20 ms.
Such examples explain why the HH criterion must be column-specific; they do
not establish comfort or musical fit. Isolated 12–43-ms holds in inspected
contexts are separate articulation questions, not evidence for a universal
hold-duration floor.

Source tags describe their human-reviewed scopes and do not prescribe the
generated alternative arrangement. No listening or player test establishes
audio alignment, difficulty, comfort, Tech quality or dump's acoustic
elaboration. Editor-only 120-BPM headers are not musical timing evidence.

## Training approximations and the remaining causal question

Source-history likelihood trains local conditional factors, including the
source H preview supplied to rows. Native generation instead feeds back sampled
H history, row history and committed LN state. Physical replay invariants do
not enforce HH, requested descriptors or sustained musical organization.
The [bounded H residual](head_wait_recovery.md) limits historical logit magnitude
and recovers audio influence after a long wait; it does not constrain the
accumulated number or distribution of generated heads. Frontier2 remains active,
but its earliest-possible-release feature is not a forecast of actual R output.
The short-attack witnesses expose consequences of that distinction.

The data does contain the LN relations that some outputs fail to produce.
All twelve TRAIN charts assigned to profile 2 have overlapping holds, starts
under existing holds and subset releases. Their time with at least two held
columns ranges from .233 to .736 of audio duration. This establishes physical
target support, not sufficient coverage or annotated musical quality.

Exposure is limited: profiles 1 and 2 each have twelve TRAIN charts, receiving
93 and 134 of the 4800 sampled intervals. The 32-update preflight sees no
profile-1 interval and only two profile-2 intervals from one chart. A continuous
shared projection also learns from other profiles, so these are not its entire
supervision. Nevertheless, a passing short preflight cannot validate learning
of these requests. Representative quantization, correlated attributes, finite
preview, one training seed and one native seed per audio remain confounders.

The routing comparison rejects this input-removal change as an improvement
under the tested conditions. It does not identify a percentage of failure
attributable to restored R1, prove that density must always enter every factor,
or establish that the encoder or corpus is sufficient. Additional training
also changes the comparator adversely on the matched original subset.

The next causal question is where that training-induced drift enters the
generated process: the full-audio H base, its sampled-history correction, or
the R/row response to the resulting H spacing. Measuring those contributions
on the saved trajectories and separating them with fixed-input probes can
distinguish loss of conditional information from altered shared representations
and autoregressive amplification. Any crossed probe is diagnostic and may be
outside training support. A further architecture or training change requires
that distinction, alongside native and Lens evidence; NLL alone cannot select it.

## Reproduction identities

Implementation source is `a2bce683642995627c17b154d5f83e8d774787f0`, compared
with `e28435f10cfb154acfe8eb9f0507d2a6c242ff79`. The entrypoint is the packaged
`ensomi_model.research.planned_audio_continuation.hydra` runner; the intervention
is `profile_head_rate_downstream=false`. Both arms use `bounded_head=true`,
`head_bound=4`, `head_decay_ms=1000` and `condition_full_holds=true`.
Training, exposure and validation seeds are 252801, 252802 and 230943;
native per-audio seeds are 251701–251708 in the panel's stored order.

Local owner: `artifacts/joint-audio/20260925-profile-routing-v1`.
The frozen settings, commands, exposure, native outputs and per-scope inspection
records are derivative local research assets, absent from a fresh clone.

| Identity | SHA-256 |
| --- | --- |
| Initial checkpoint | `abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef` |
| Shared endpoint | `7849e07a77febf12968b8197e844d9c26f8327ce789d0776117e02dc83de788f` |
| Routed endpoint | `d2d943a8bb565559480121645128c70824d792b9830c013981780bbfb74a3f22` |
| Corpus manifest | `4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1` |
| Normalization | `9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287` |
| Profile bank | `a8973b7f94fde90d9f3cc639eb63e781dd295435622ce79fd54fe244789e6f03` |
| Native panel | `484d173628856a431e034b1dbdacab9173c40f3edf696d1b9a19721aeec70dc9` |
| Native result | `0097d705919e6203a000906cef464337f039d52b861c63ae561a9ece5f57d9d6` |
| Completed scope review | `4e4d074e4f02b63b05d8d2d710700adf440e243aa5e036e5a1e244684967d659` |
