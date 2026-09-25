# Scoped demand and the decision boundary between skeleton and R1

The typed prototype's remaining failures belong to more than one module. A
fixed-program comparison shows that R1 can remove many rapid recovery choices
without changing the times or LN objects. It cannot repair short LN lifetimes
that the skeleton has already fixed. Better whole-chart star calibration also
does not establish better local action structure.

This investigation uses native 4K at 1.0x, the existing full-song Mel encoding,
and the declared HH/RH/HR envelope of 37/25/21 ms. The primary ranked reference is
[documented separately](ranked_2to6_action_reference.md). These empirical chart
facts do not define a physiological player model or the complete V3 frontier.

## Reference and action conservation

Intersecting the ranked reference with the existing TRAIN catalog yields 6,924
charts and 252,167 nonempty 8-second windows with 4-second hop. Weighting gives
equal mass to song groups, then charts and windows. Difficulty bands refer to
whole-chart ratings; they are not local star labels.

In 2.5–3.5-star charts, windows with at least 50% LN heads have median head and
press-plus-release rates of 7.25 and 11.875 per second. Their median within-window
release duration is 196 ms. The preceding high-LN generated examples had median
action rates of 20.75, 19.062 and 15.625 per second, with release durations
120, 128.25 and 175 ms. A high LN proportion cannot be treated as a cosmetic
change to a fixed head density.

For a scope, let A count TAP/LN-head attacks, L count new LN heads, R count
releases, and B_start/B_end count held keys at the boundaries. Exact replay gives
`R = L + B_start - B_end`. Therefore total press/release actions are
`A + L + B_start - B_end`. These actions need not have equal gameplay cost, but
the count relation is an invariant that any demand controller must respect.

## Head-history comparison

Both arms start from the same 1,600-update candidate and receive 800 further
updates using the same sampled windows and seed. The 3.896M-parameter baseline
keeps shared typed history. The 4.301M-parameter intervention adds independent
63-head memory and a last-head clock, reports only remaining resource wait, and
factors head probability before conditional release-only probability. It changes
these together; the comparison does not isolate their individual effects.

Both arms complete 15 native cases on three audio files, including scoped changes.
Static control metrics use the 12 cases without a mid-song change.

| Measure | Shared history | Head-owned clock |
| --- | ---: | ---: |
| Requested-star mean absolute error | 1.235 | 0.979 |
| LN-fraction mean absolute error | 0.0516 | 0.0675 |
| LN durations at most 40 ms / all LNs | 374 / 8,738 | 752 / 10,264 |
| Short-LN fraction | 4.28% | 7.33% |
| Maximum cached-Mel window service | 0.374 s | 0.439 s |

Training took 281 and 343 seconds on MPS. Native generation used one CPU thread.
The new structure reduces the selected star error while increasing short-LN
prevalence. It is not selected as a playability improvement. The shared model
also produces under-dense cases; lower density alone is not accepted as progress.

## R1's conditional frontier

A post-hoc experiment fixes every typed event, head/LN count and LN lifetime,
then regenerates only column assignment. Before intervention, fixed-program
rendering reproduces each of the 24 original static row sequences exactly.
The intervention samples among legal rows minimizing the number of attacks
within 80 ms of a previous same-column attack or release.

The 80-ms value is a diagnostic preference, not a new universal BAD label or
hard generation rule. The comparison is greedy at each actual prefix, not a
proof of globally optimal geometry or preserved style.

| Cohort | HH below 80 ms | RH below 80 ms | LN durations at most 40 ms |
| --- | ---: | ---: | ---: |
| Shared, original → reassigned | 91 → 66 | 989 → 523 | 374 → 374 |
| Head-owned, original → reassigned | 191 → 137 | 1,187 → 684 | 752 → 752 |

The largest absolute star change is 0.070 and 0.049 respectively. Thus substantial
local recovery changes can be nearly invisible to the star metric. The unchanged
lifetimes identify a different decision boundary: after release identities and
times are fixed, R1's row distribution cannot move those tails. Merely retaining
the `frontier2` scorer does not restore the release choices that have moved into
the skeleton. Release-related responses must influence that earlier decision.

A separate feasibility check retains all head times and columns, caps tails at
the actual audio end, and asks whether short LNs could last at least 80 ms.
Of 374 short LNs in the shared cohort, 336 permit that extension while retaining
25-ms release-to-next-head separation, and 231 still permit it with 80-ms
separation. The head-owned numbers are 679 and 456 of 752. This identifies unused
room in the sampled schedules; it does not mean stretching every hold would be
musically or physically preferable.

## Guidance and the limits of conditional signals

The models were trained with 15% optional-control dropout. A fixed-strength-two
probe guides clock, mark and row distributions with
`log q = 2 log p_requested - log p_stars_unspecified - log Z`, on their shared
physical support. Only the star value and its known bit are omitted in the second
pass. This adapts categorical autoregressive
[classifier-free guidance](https://arxiv.org/abs/2306.17806); it is not a
playability guarantee or a new training result.

All 24 guided cases complete. Star error becomes 0.936 for shared history and
0.836 for head-owned history. Short-LN fractions are 3.90% and 7.82%, and some
low requests remain below two stars. The conditional signal can be strengthened,
but its absolute calibration and lifetime behavior remain wrong. The maximum
guided cached-Mel window is 0.599 seconds; waveform preprocessing and the client
are excluded from these timings.

Lens inspection covers five aligned attribution contexts and all ten pages,
plus two guided peak contexts and all four pages, with complete action and LN
articulation tables. Reassignment removes a 25-ms same-column release/head pair
at 212456 ms while retaining the same LN object times. A three-LN group at
31521–31561 ms demonstrates a fixed 40-ms lifetime that columns cannot change.
The Hysteric comparison shows both long, sparse holds and denser mixed movement;
the guided Zenithfall peak still contains many short additions. Inspection does
not establish their musical alignment. No listening or player test was performed.

## Representation issues exposed by the comparison

The compared checkpoints expose two concrete representation constraints. The
[typed-model repair](typed_audio_continuation.md#timing-base-and-local-preference-repair)
implements the changes below and reports their subsequent native results.

First, `TypedAudioModel` reuses an inherited configuration containing
`bounded_head=True`, but its typed clock computes an audio affine plus an
unbounded MLP residual. It does not implement the earlier bound or waiting-time
decay. A bounded history path must leave direct audio, current controls and real
LN obligations available in the base. This omission is observable in code;
its share of the native generation error has not yet been isolated.

Second, the bounded binomial LN base imposes a local distribution restriction.
For an ordinary three-head event where all LN counts 0–3 are feasible, with
rho=0.7 and residual bound 1, the all-TAP probability cannot exceed
`0.3^3 * exp(2) / (0.3^3 * exp(2) + 1 - 0.3^3)`, approximately 17%.
An audio-conditioned residual cannot override that bound. A scoped 70% request
should still allow a locally almost-certain pure-TAP chord passage. Positive
support for the pattern is not enough to model its intended conditional
probability. Scope-level allocation and local preference must therefore remain
distinct; a count target must not become an unintended per-head independence
assumption.

## Range control and ownership of consequences

A scoped request describes a family of acceptable arrangements over its range:
approximate difficulty, independently specified style strengths and an LN-head
proportion. It allows local variation and does not impose a quota on every
row or subwindow. Changing the request preserves the physical history and any
crossing LN obligations. Evaluation partitions the actual effective schedule;
separate ranges cannot cancel one another's control errors.

The [scoped target implementation](typed_audio_continuation.md#scoped-difficulty-supervision)
addresses one supervision mismatch: every interval previously inherited its
chart's whole-song rating. Its full-prefix strain proxy provides a consistent
range description for a bounded joint adaptation. It is still incomplete for
release execution and does not define the full gameplay frontier.

The factors own different opportunities to change demand:

| Decision owner | Decisions it can change | Information required |
| --- | --- | --- |
| Skeleton | Event time, head count/type, release subset and LN lifetime | Direct audio, past skeleton, exact resource/hold state, scoped conditions and responses relevant to those choices |
| R1-derived arranger | Column organization among complete rows realizing the plan | Direct audio, typed preview, own row history, exact column state, scoped conditions and row consequences |
| Publication scheduler | How much uncommitted work to prepare or revise before publication | Playback horizon, measured service cost, control ranges and persistent physical obligations |

This division explains why the next quality change must reach the decision
that creates the observed problem. A 26-ms LN followed by a 30-ms same-column
recovery involves both a fixed lifetime and a column choice. R1 can sometimes
improve the latter; the skeleton owns the former. A useful upstream response
must distinguish the consequences of holding, releasing and introducing a new
head. Release counts also obey the scope conservation law above, so a high-LN
request changes workload even if attack count stays constant.

One small structural candidate is an explicit audio query at each active LN's
head, combined with audio at the proposed release and its exact age. The current
planner retains at most 63 event tokens and LN start clocks; a long hold can
outlive the token containing its initiation. Four keyed birth-audio references
would give the release decision direct access to the musical relation between
the two endpoints. This is per-object context, bounded by four open holds; it
does not require a song-level motif memory or named musical sections. Joint
subset-release scoring remains available for coordinated tails.

That candidate needs paired endpoint queries in training's local Mel crops and
inference's full-song audio cache. Its inputs are already generated head times,
current candidate times and the available full audio. Source future tail times
may supervise the output but cannot enter its conditioning state. The same
restriction applies to using the offline strain trace as runtime feedback:
that trace reads completed LN endpoints. A causal response model must instead
use committed history plus explicitly proposed futures, as required by the
[gameplay formulation](../formulation/gameplay-state.md#target-response-and-frontier).

These are architectural reasons for investigating endpoint context and
upstream responses. They do not establish that either mechanism alone will
fix native release quality. Scoped adherence, real articulation and coherent
audio-responsive variation remain the joint target; a lower loss or a better
single difficulty proxy is insufficient for selecting a playable system.
