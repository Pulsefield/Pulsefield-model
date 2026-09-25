# Typed resource planning with scoped controls

The `research.typed_audio_continuation` prototype makes timing, note counts and
LN identities one causal score. Its second factor uses R1-derived complete-row
scoring to assign columns. Both factors read the same trainable local Mel and
bidirectional full-song audio encodings. This is an exploratory model, separate
from the packaged planned-head inference path.

An event is a native-millisecond time, TAP count, new-LN count, and release mask
for four reusable LN identities. A mark may contain both heads and releases.
The planner reads past events and four abstract resource states, never a row
embedding or past TAP columns. Its finite temporal encoder has 64 channels.
The arranger reads its own row history, exact physical state, direct audio,
current/future typed events, current LN-to-column bindings and semantic controls.
Thus row choices cannot silently change the planner's learned state.

An active LN occupies one resource. An inactive resource has an earliest next
attack time. A TAP consumes a free resource through its attack recovery; an LN
release frees its resource after both attack and release recovery have elapsed.
Releases do not provide a head in the same event. The initial envelope is
HH >=37 ms, RH >=25 ms and HR >=21 ms, motivated by the
[ranked reference](ranked_2to6_action_reference.md). It excludes the reference's
rare 19/20-ms LNs and is not a physiological comfort model. It preserves short
cross-column events. Difficulty requires additional learned responses to
sustained counts, LN burden and geometry.

All already-eligible free resources are interchangeable for future feasibility.
The abstract state canonicalizes their deadlines at each event; any supported
column assignment preserves the same future eligible counts. R1's distribution
is normalized over rows matching the exact mark, LN bindings and recovery law.
No retry or forced simplification is needed to repair a count/time disagreement.
The learned model can still produce musically poor or excessively tiring plans.

LN IDs use the lowest free identifiers. Simultaneous starts bind in ascending
column order, and IDs can be reused after their previous releases. This is a
lossless source convention under the resource envelope, not an assertion that
identity labels are invariant under column permutations.

## Scoped conditions

`ControlSchedule` contains half-open intervals with independently optional
`stars`, `ln_fraction` and vocabulary-named ordinal style requests. Later spans
override supplied fields; other fields retain their earlier values. At the end
of a span the earlier condition resumes. Each attribute has an explicit known
bit, so unknown and explicitly absent style judgments remain different.

LN fraction means `LN heads / (TAP heads + LN heads)` inside the requested
interval. A hold entering the interval affects occupancy and release workload,
but its earlier head does not count toward that interval's requested fraction.
The request allows local variation: a high-LN interval may contain a pure-TAP
chordjack passage. A style request may name several simultaneous organizations;
it does not select one mutually exclusive chart category.

The initial style vocabulary is the existing five scoped concepts: jack,
stream and trill organization, tech, and LN coordination. Their training values
are absent, supporting and prominent. These are organization judgments, not
GOOD/BAD labels. Unreviewed or conflicting cells provide no target. The model
constructor declares its vocabulary; merely adding a slot does not establish
semantic control.

Both the plan and row query consume controls. Training uses whole-chart native
1.0x star labels, scoped LN-head fractions and available human style judgments.
Optional-condition dropout lets the factors learn with missing attributes.
Whole-chart stars are weak local supervision: a scoped 3-star request is not a
claim that a fragment has a well-defined independent star rating. Joint requests
also need evaluation where their requested combination is rare in the corpus.

`TypedSession.update_controls` retains published rows, occupancy and confirmed
empty time, retains queued events before the changed scope, and regenerates the
remaining unpublished plan. A hold crossing the scope remains an obligation;
a lower LN-fraction request cannot erase its head. This session is an in-memory
research implementation, not a durable client scheduler or crash-recovery API.

For example, this starts with difficulty 4 and 20% LN heads, then requests
prominent tech for 32 seconds while retaining those other conditions:

```python
controls = ControlSchedule(
    (ControlSpan(0, duration_ms + 1, stars=4., ln_fraction=.2),),
    model.style_names,
)
session = TypedSession(model, mel, duration_ms, controls)
session.publish_to(8000, minimum_rows=30)
start = session.coverage + 1000
session.update_controls(
    ControlSpan(start, start + 32000, style={"tech": 1.}),
)
```

The same update can supply `stars` and `ln_fraction`. Its start must be later
than published coverage. After its end, the earlier requested values resume;
the generated history and crossing holds remain, so later sampled rows need
not match a run that never received the update. The model currently receives
one pair of scope clocks from the last active span. Attribute values resolve
independently, but separate deadlines for overlapping partial requests are not
encoded. Exact per-attribute quota tracking would need that additional state.

The intended control behavior is approximate, scoped and jointly playable.
Difficulty should stay reasonably close to the request; exact star or LN-count
fulfillment is not an acceptance requirement. Local variety and coherent
transitions remain desirable. A control value is a requested tendency within
its range, subject to existing physical obligations.

`evaluation.describe_control_ranges` partitions observations at every declared
control boundary using `ControlSchedule.resolved_ranges`. Each range retains its
effective difficulty, LN request and independently optional style values.
Observation windows never cross a control boundary. Recovery clocks still read
the real prefix, and crossing holds retain their original heads and endpoints.
Reports distinguish new heads, entering/leaving holds and releases of entering
holds. LN proportion counts only heads inside the range; release burden includes
all releases experienced there.

Assess each range against its own request, with boundary context and appropriate
ranked examples. A whole-chart star value is useful when the complete chart has
one static request. It does not measure the adherence of a shorter high-difficulty
override inside a longer easy chart. Range reports therefore expose action load,
recovery, hold articulation and organization descriptors without inventing a
fragment star rating. A single pooled score across differently controlled ranges
cannot determine whether those controls succeeded. The aggregate pilot metrics
below describe experiments, not a replacement for this scoped evaluation.

## Learning and evidence limits

The probability law factors native-clock event type, feasible mark, and matching
row assignment. Clock queries emit ten native positions per 10-ms audio query;
there is no beat-grid quantization. Teacher forcing exposes source plans to the
arranger during training; generated plans must therefore be evaluated separately.
The source clock includes silence and the true audio endpoint. Crop boundaries
do not close LNs, and training's coarse audio encoder always sees the full song.

The implementation retains the R1-derived `frontier2` scorer and adds the typed
preview. Its inherited release-opportunity feature is still an optimistic
approximation; the new resource support is exact only for the declared recovery
law. Neither component is yet a calibrated player-specific response model.
Scoped control effects, generated structure and runtime require native rollouts
and Lens review. NLL is a learning diagnostic, not the acceptance criterion.

## Explicit LN proportion experiment

The optional `ln_base` factor separates a mark's head-count/release-mask group
from its LN count. Group probability remains the learned distribution. Within a
group with `h` heads, LN count `l` has probability proportional to
`binomial(h,l) * rho**l * (1-rho)**(h-l) * exp(residual[l])` on feasible counts.
The learned residual is centered within the group and bounded by one logit in
either direction. The LN request is masked from its residual-network inputs;
audio, event history, other controls and scope remain available.

At fixed audio/history/support and scope, increasing `rho` increases expected LN
count without changing the head-count/release-mask marginal. This is a local
conditional property, not a guarantee that a sampled scope attains its requested
percentage: changed holds affect future timing and eligibility. Requests are
proportions, with numerical endpoint clipping, not hard all-TAP/all-LN commands.
Missing LN conditions retain the unconstrained learned law. Training and native
sampling use the same factor.

## Small joint-fit findings

A local Apple M5/MPS pilot uses 614 TRAIN charts and 36 held-out VAL charts,
retaining separate arrangements and full-song audio. A single TRAIN chart is
outside the initial HH envelope; its three HH intervals below 37 ms are excluded.
The row/audio initialization is the planned flat checkpoint `da080448`; the
prototype has 3,895,879 parameters. The first joint fit took 1,200 updates and
413 seconds. An additional 400-update fit of the explicit LN base took 147
seconds, also mixing whole-song and 8/16/32-second control scopes. Because this
second fit changes both factorization and sampling, its entire improvement is
not attributable to factorization alone.

Both candidates completed all 15 native cases: three previously unseen audios,
each with low/high difficulty, low/high LN proportion and a 32-second mid-song
change. Published rows remained fixed and all same-column HH intervals were at
least 37 ms. The second candidate's maximum cached-Mel startup/window service was
0.381/0.344 seconds on one CPU thread. This excludes waveform/Mel preprocessing
and client rendering.

At requested difficulty 3, measured LN-head fractions were:

| Audio | Concatenated control: request 20% / 70% | Explicit LN base: request 20% / 70% |
| --- | ---: | ---: |
| Zenithfall | 14.3% / 17.3% | 8.5% / 74.8% |
| Hysteric | 19.5% / 21.7% | 10.2% / 79.1% |
| As It Was | 39.7% / 45.8% | 17.1% / 73.9% |

The explicit-base 32-second high-control scopes yielded 70.3%, 83.5% and 75.2%
LN heads. The following 32 seconds, after restoring the 20% request, yielded
10.0%, 12.3% and 20.4%. These are three sampled trajectories, not confidence
bounds or evidence of exact percentage control. Surviving LNs and history carry
across the request boundary.

Difficulty remains inadequately calibrated. With LN=20%, the first candidate's
3/5-star requests yielded Hysteric 3.010/4.729, Zenithfall 4.625/5.080 and As It
Was 2.574/2.972 stars. The second candidate's 3-star requests yielded
4.097, 5.202 and 3.168 respectively. Thus better proportion control does not
establish better playability or preserve difficulty automatically.

A fixed-history VAL probe also found the first model's mark expectation changed
only 1.3–1.9 percentage points for an LN request change from 20% to 70%.
Changing the request scope from the whole song to 16 seconds moved this result
by less than 0.5 points. Weak direct conditioning is therefore observable before
free-running state feedback, although feedback can add further error.

Lens inspection covers seven first-candidate contexts and all 14 pages, plus
three high-LN peak contexts and all six pages for the second candidate, with
complete action/articulation tables. Mixed chords, layered holds, subset releases
and handoffs survive. Short 21/29/32-ms LN additions and dense low-request passages
remain. In the second candidate's high-LN Zenithfall output, 173 of 3,005 LNs
last at most 40 ms; its median LN duration is 120 ms. The explicit proportion
factor has not solved release-demand quality. No listening or player test was
performed.

Only 11 TRAIN charts overlap the existing human style judgments. Five semantic
input slots and their training path exist, but reliable style control is not
established. These candidates are useful for investigating demand-conditioned
planning and release behavior; neither is an accepted playable-system release.

## Owning the head clock

The optional `head_stream` experiment keeps a separate 63-head temporal history
of head gaps, TAP counts and new-LN counts. Release-only events do not advance
that history. Its exact phase clock is time since the last head; active LNs and
resource recovery still condition the prediction. Releases and marks retain the
complete typed-event history. Both paths consume direct full-song audio and
scoped controls. The model has 4,301,393 parameters with this option.

At each native position, the clock law is `P(H)=h`, `P(R-only)=(1-h)*r`, and
`P(none)=(1-h)*(1-r)`, after applying support. This keeps a changed release logit
from directly renormalizing head probability. It does not make future heads
independent of the actual LN occupation created by earlier choices.

The resource abstraction deliberately forgets which already-eligible free
resource had which old recovery deadline. In this experiment its neural feature
therefore reports remaining wait, clipped to zero when eligible. Signed past
deadlines would expose an arbitrary canonicalization age as apparent recovery
information. Head-stream caches are preserved when a scoped-control revision
rolls the unpublished planner back.

A broader TRAIN-only reference contains 6,924 ranked charts and 252,167 nonempty
8-second windows with 4-second hop. Weighting gives equal mass to song groups,
then charts and windows. In charts rated 2.5–3.5 stars, windows with at least
50% LN heads have median head/action rates 7.25/11.875 per second and median
within-window LN-release duration 196 ms. The high-LN generated examples above
have median action rates 20.75/19.062/15.625 and release durations
120/128.25/175 ms. These are distribution comparisons, not local star ratings
or physiological limits.

There is an exact constraint behind the coupling: over any scope, releases equal
new LN heads plus entering held keys minus exiting held keys. Hence increasing
LN-head proportion while keeping attack count fixed generally increases total
press/release actions. Those actions need not have equal gameplay cost. Difficulty
control must account for their joint organization instead of equating a desired
star value with a universal note-rate target.

The [scoped demand investigation](scoped_demand_frontier.md) reports the matched
head-clock comparison, fixed-program R1 counterfactuals, guided decoding, and
two remaining representation constraints. Neither experimental clock variant
is accepted as the final playable-system architecture.

## Timing-base and local-preference repair

`bounded_clock` is an explicit typed-model option. Its base reads full audio,
current controls and active-LN ages/occupation. It does not read learned event
history or chart counters. A separate history contribution is bounded by
`head_bound * exp(-age/head_decay_ms)`: head modulation uses last-head age,
release modulation uses last-event age. BOS contributes no historical
modulation. Support remains exact, and LN obligations stay in the base while
history fades. This option is separate from the head-owned experiment; inherited
`bounded_head` settings alone do not select it.

`ln_prior` selects a different amount mechanism from the bounded binomial base.
Within each head-count/release-mask group, it shifts learned LN-count logits by
`LN_count * (logit(request) - logit(reference_fraction))`. Group masses remain
unchanged and local learned preferences are unbounded. A locally pure-TAP chord
can therefore remain nearly certain inside a globally high-LN request. Known
reference-amount and unspecified-amount inputs retain different presence bits.
The reference fraction comes from TRAIN; this conditional prior does not enforce
an exact realized scope quota.

Checkpoints record `probability_options`, the constructor settings actually
consumed by the typed model, separately from its inherited backbone settings.
Neither repair is a playability certificate; native structure, control response
and full-song behavior still decide its usefulness.

## Broader paired coverage and control response

The repaired model has 3,947,227 parameters. A 1,200-update MPS fit on the
614-TRAIN/36-VAL corpus took 451 seconds. A broader admitted corpus contains
6,923 ranked 2–6-star TRAIN charts from 2,573 groups and retains the same 36 VAL
charts. It supplies canonical full-song Mel and overlaps 112 human-annotated
TRAIN charts, compared with 11 in the smaller paired corpus. Missing style
assessments remain unspecified; the 289 human cells are not quality labels.

The broader fit samples song group, chart and clock interval for 75% of draws,
with a separately declared human-annotated interval objective for 25%. It also
corrects the population interval weight: uniform interval sampling uses
`1000 * interval_count / audio_duration_ms`, so a short final interval does not
receive an oversized weight. Audio normalization and the LN reference prior
remain fixed. The comparison therefore changes coverage, weighting and training
duration together; it does not isolate the effect of more data alone.

On the same three-audio panel, each checkpoint completes 15 native cases:
12 static combinations of difficulty 3/5 and LN fraction 0.2/0.7, plus three
32-second mid-song changes. Static results are:

| Measurement | Repaired small corpus | After 800 broader updates |
| --- | ---: | ---: |
| Mean absolute whole-chart star error | 0.739 | 0.588 |
| Mean absolute LN-head fraction error | 0.0745 | 0.0911 |
| LNs lasting at most 40 ms | 649 / 15,120 | 345 / 17,309 |
| Fraction of LNs lasting at most 40 ms | 4.29% | 1.99% |

The shorter-tail improvement is not uniform. Hysteric at requested difficulty 3
and 70% LN increases from 24 to 55 LNs of at most 40 ms; its median hold duration
falls from 170 to 127 ms. The higher request produces 78.5–82.5% LN heads across
the three 32-second switched scopes after broader fitting; the following
32 seconds produce 14–17% after restoring the 20% request. Published prefixes
remain unchanged. These are sampled responses, not exact quota fulfillment.

Lens review covers four small-corpus contexts and eight pages, then two broader
peak contexts and four pages, including every action and articulation table.
Layering, subset releases, mixed TAP/LN groups and short cross-column events
remain. In the small-corpus As It Was example, an LN from 88937 to 89106 ms
survives the control expiry at 89000 ms. The broader Zenithfall peak contains
fewer isolated microholds but retains a 34-ms LN; Hysteric still has short
release-to-head recovery in an outer-column repeated figure. Neither candidate
is an accepted playable-system release. No listening or player test was done.

A separate small-corpus style probe holds difficulty 4 and LN fraction 0.2
fixed on Hysteric. Four conditions share the same 32-second update procedure,
including an unspecified-style baseline. Over that scope, adjacent-head
same-column reuse is 17.8% for baseline, 27.6% for prominent jack, 14.9% for
prominent stream and 13.3% for prominent tech. This primitive is not the style
readout. Eight inspected contexts and 16 pages show weak immediate differences
and do not establish the requested prominent organizations. Semantic style
control remains a learning and native-evaluation question.

To examine the LN/difficulty coupling, a fixed-prefix probe uses one middle
8-second interval from each of the 36 existing VAL charts. Audio, teacher
history, physical state, scope and requested difficulty 3 stay fixed; only the
LN request changes from 0.2 to 0.7. Mean conditional head-event probability
increases by 2.18% in the small-corpus model and 1.34% after broader fitting.
In contrast, the latter's autonomous head counts increase from 2,102 to 3,327
on Zenithfall and 1,900 to 2,998 on Hysteric. These are different operating
distributions: the comparison identifies generated-state feedback as an
investigation target, but does not assign a causal percentage to history,
occupation, release decisions or R1.

The broader run finishes 6,000 updates in 2,813 seconds, having sampled 4,641
distinct TRAIN charts. Both the 2,400- and 6,000-update checkpoints complete
40 full-song cases on eight audios. The 32 static cases share whole-song
requests; the other eight have a temporary override and are evaluated by range.

| Static-case measurement | 2,400 updates | 6,000 updates |
| --- | ---: | ---: |
| Mean absolute whole-chart star error | 0.553 | 0.959 |
| Mean absolute LN-head fraction error | 0.0770 | 0.0838 |
| LNs lasting at most 40 ms | 544 / 31,671 | 1,430 / 44,899 |
| Fraction of LNs lasting at most 40 ms | 1.72% | 3.18% |

The later fit is not selected despite its lower teacher-forced losses. In its
Zenithfall 3-star/70%-LN case, the chart measures 6.110 stars. Lens inspection
of 74050–78051 ms finds 78 head rows, repeated 24–39-ms holds and rapid release
activity. A subsequent pure-TAP passage survives, so expressive support alone
does not explain or solve the excessive burden. The earlier checkpoint is a
retained research baseline, not an accepted playable release. Its inspected
Fool Moon passage has coherent layered holds, while its Revenge high-control
peak still contains a 26-ms LN followed 30 ms later by another same-column head.

Effective-range reports cover 87 completed cases and 133 ranges across native
and style cohorts. For example, the earlier Hysteric override at
105000–137000 ms requests difficulty 5 and 70% LN, producing 85.4% LN heads
and 15 press/release actions per second. After restoring difficulty 3 and
20% LN, the remaining range produces 13.0% LN heads and 5.40 actions per second,
including two releases of inherited holds. These ranges remain distinct.
The older switch procedure chooses its start relative to publication coverage,
which may differ between checkpoints; those switched trajectories are not
identical-time paired interventions.

A seven-case style probe at 6,000 updates compares explicit absent versus
prominent jack, stream or tech, retaining the same known bit, difficulty 4,
20% LN request and 105000–137000-ms scope. Other concepts remain unspecified.
Jack's identical adjacent masks increase from 3 to 12; a fully inspected
four-second context also shows a repeated-column figure under sustained holds.
Stream changes toward more moving handoffs. Tech remains inconclusive. These
facts show conditional structural response, not reliable strength calibration
or mutually exclusive styles. The six paired contexts, all 12 pages and their
complete tables were inspected; no whole-scope semantic labels or player-test
claims are inferred from them.

The retained model has 3.947M parameters and a 15.9-MB standalone weights file.
For 499 seconds of actual audio on one M5 CPU thread, a fresh Python process
takes 2.12 seconds from process entry through decoding, Mel extraction, full-song
encoding and publication of 61 rows covering the first eight seconds. One LN
remains open. OS caches were not flushed, and client rendering/reading time is
excluded. This is an initial service measurement, not a cold-machine guarantee.

## Scoped difficulty supervision

`difficulty_targets.ScopeStrainTrace` provides an optional offline supervision
proxy. A 5-star chart can contain a quiet passage; assigning 5 to every sampled
scope confounds whole-chart capacity with the demand inside a requested range.
This is a semantic reason to revise targets without increasing model capacity.

The trace reuses the repository's 20241007 mania strain calculation on complete
native-1.0x objects, retaining earlier state and real LN endpoints. Inside each
requested half-open range, it takes the maximum of inherited decaying strain
and new-object strain in cells of at most 400 ms. Cells begin at the scope start;
the last partial cell is retained. No future head or earlier peak is assigned
to the range. For descending peaks `p[0], ..., p[n-1]`, the readout is
`0.018 * sum(0.9**i * p[i]) / (1 - 0.9**n)`.
The denominator removes finite-cell truncation of the weight sum: constant
peaks have the same level for different scope lengths. The scale approaches
the existing whole-chart scale for long ranges, with differences from boundary
alignment and the declared observation endpoint.

`source_schedule(..., difficulty_trace=trace)` uses this value and the observed
LN fraction over the same scope. Both model factors consume the resulting
condition. The default remains whole-chart supervision; checkpoints trained
with the new target must record `full-prefix-mania-strain-scope-v1`. This is a
proposed learning target in approximate star units, not an official local star
rating, a physiology model or an independent quality metric. It does not measure
release execution adequately and cannot replace the articulation observations.

This adapts the achieved-goal relabeling idea in
[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495): pair an observed
trajectory with the result it actually achieves. Here the trajectories are
ranked beatmaps, the targets are scoped descriptions, and learning remains
supervised joint likelihood; there is no reinforcement-learning reward dataset.
The analogy motivates consistent conditioning, not a claim that this scalar
establishes playability.

The trace is deliberately offline. The underlying strain algorithm reads LN
endpoints, including tails not yet decided at an incremental head. Source-derived
control targets may summarize the complete target scope; generated-state
features may not use undecided future objects. This readout is therefore not
inserted into the causal skeleton or substituted for `frontier2`. Native
generation must still be assessed for approximate range adherence, musical
variation and release quality together.

### Bounded scoped-target result

A TRAIN-only sample of 256 song groups, one random chart per group, supplies
2,940 consecutive 16-second scopes. With equal group mass and equal scope mass
within each chart, the local-minus-whole-chart proxy difference has
5th/50th/95th percentiles -2.647/-0.538/-0.131. About 24.3% of scopes are more
than one unit below their chart label. Full-song proxy values agree closely with
the existing rating: their median difference is effectively zero. This supports
distinguishing the two target meanings; it does not validate the proxy as a
complete difficulty measure.

A 1,000-update joint adaptation starts from the retained 2,400-update weights,
keeps the 6,923-TRAIN/36-VAL corpus and 3.947M architecture, and replaces only
the scoped difficulty labels. The optimizer restarts with seed 251928 and the
same 3e-5/3e-4 body/planner learning rates. It takes 512 seconds on MPS and sees
1,395 distinct TRAIN charts. This comparison includes additional optimization
and a new sample sequence; it does not isolate the causal effect of relabeling.

All 15 native cases complete, but the adapted weights are not selected. Across
the same 12 static cases, mean star error rises from 0.488 to 1.644, LN-fraction
error from 0.0777 to 0.1018, and LNs lasting at most 40 ms from 256/12,515
(2.05%) to 2,239/23,866 (9.38%).

Three additional matched comparisons use the same absolute 105000–137000-ms
override, requested at the first publication reaching 96000 ms. All begin with
difficulty 3/LN 20%, override with difficulty 5/LN 70%, then restore the original
conditions. The approximate strain readout is reported separately for each
range:

| Audio | Baseline before / override / after | Adapted before / override / after |
| --- | --- | --- |
| Zenithfall | 1.638 / 4.432 / 3.973 | 4.013 / 5.145 / 5.250 |
| Hysteric | 2.961 / 2.788 / 2.775 | 4.279 / 5.056 / 5.083 |
| As It Was | 3.457 / 3.054 / 3.228 | 4.045 / 4.987 / 4.291 |

The overrides approach their requested scalar while the surrounding 3-star
ranges become harder. Elevated difficulty is already present before the
override, so the result cannot be attributed only to carryover from the changed
control. Pooling those ranges, or selecting on the override proxy alone, would
misrepresent the outcome.

Lens review covers two contexts, all four pages and every action/articulation
table. The adapted Zenithfall 3-star/high-LN peak contains 80 head rows in
4.001 seconds and repeated 25–39-ms LN additions. Its Hysteric override begins
with coherent 717/477/493-ms layered holds and a real incoming release at
105057 ms. That short positive witness does not establish the remaining scope's
quality. No listening or player test was performed. The scoped readout remains
an optional research tool; these weights do not replace the retained baseline.

## Bounded amount feedback

`TypedSession(..., ln_feedback=LnFeedback())` and `rollout` optionally apply a
small inference correction to LN allocation. The fitted probability model and
its default sampling remain unchanged. This policy addresses observed requests
for 70% LN that drift toward 90%, increasing the release workload identified by
the scope conservation law. It is not a difficulty controller or a claim that
LN percentage alone determines playability.

For the active LN request episode, let H and L count generated skeleton heads
and LN heads. With requested proportion rho and pseudocount k=32, the smoothed
realized proportion is `(L + k*rho) / (H + k)`. The policy applies the difference
between requested and realized log odds, multiplied by strength 1 and clipped
to [-1, 1], as an additional within-group LN-count tilt. The head-count/release-
mask marginal of that mark query is preserved. Future occupation and timing
can still change as a consequence of choosing different head types.

The finite correction leaves local learned preferences unbounded, so an
intentional pure-TAP passage remains possible. There is no end-of-scope quota
or forced repayment. The episode follows the effective LN field's owner:
difficulty/style-only changes do not restart it, fully shadowed boundaries
do not interrupt it, and returning to an earlier amount request begins a new
episode. Future revisions do not discard counts before their actual start.
Physical holds continue across every boundary.

Allocation is stored with each planner snapshot. Invalidating an unpublished
suffix restores its matching counters along with resource state, temporal
cache and RNG. It never reads materialized TAP columns or R1 hidden state.
The runtime records the policy parameters in generation metrics. Native range
calibration, release articulation and style organization determine whether this
optional sampling policy is useful; no benefit follows from the interface alone.

On the retained 2,400-update model, one fixed feedback setting completes the
three-audio, 15-case panel without retraining. The 12 static cases improve
LN-fraction error from 0.0777 to 0.0380. Star error changes from 0.488 to 0.580,
and short LNs change from 256/12,515 (2.05%) to 236/12,217 (1.93%). This is a
useful amount-calibration result, not a general playability improvement.

The fixed 105000–137000-ms 70%-LN overrides remain heterogeneous. Zenithfall
changes from 89.8% to 74.8%; Hysteric from 74.5% to 81.4%; As It Was from
77.4% to 79.9%. After restoring the 20% request, the remaining ranges yield
15.1%, 15.5% and 13.7%. These ranges retain their own action and difficulty
observations rather than being pooled into a successful average.

Two native Lens contexts and all four pages retain LN layering, simultaneous
and subset releases, mixed rows and a pure-TAP passage. The Zenithfall high-
control peak still has 27–38-ms holds and an avoidable 25-ms same-column
release/head transition. Counter feedback controls how often LNs are introduced;
it cannot by itself establish good lifetimes or recovery geometry.

An explicit absent/prominent Jack comparison at difficulty 4/LN 20% also remains
inconclusive. In the 32-second style range, LN fractions change from
16.4%/6.4% without feedback to 17.8%/12.9% with it. Adjacent-head column reuse
changes from 18.2%/30.3% to 21.8%/25.5%, but that primitive is not a semantic
readout. Four inspected contexts and eight pages show chordal TAP movement
without feedback and longer LN anchors with feedback; neither establishes
reliable requested Jack strength. Occupation couples amount and organization,
so a better LN percentage does not prove preservation of every other control.
The policy remains optional research functionality.

## Empirical recovery preferences

`RecoveryPreference` is a separate optional inference policy for decisions whose
consequences the amount counter cannot repair. It uses the first percentile of
same-column head/head (HH), release/next-head (RH), and LN head/release (HR)
intervals in 6,923 native ranked TRAIN charts from 2,573 song groups. Each
transition receives weight `1 / (charts_in_group * audio_seconds)`, matching the
population objective's exposure measure up to a constant. Stars below identify
whole-chart reference bands, rounded to the nearest integer within 2–6.

| Reference star band | HH first percentile | RH first percentile | HR first percentile |
| --- | ---: | ---: | ---: |
| 2 | 169 ms | 107 ms | 89 ms |
| 3 | 143 ms | 83 ms | 75 ms |
| 4 | 107 ms | 65 ms | 53 ms |
| 5 | 89 ms | 55 ms | 44 ms |
| 6 | 83 ms | 49 ms | 43 ms |

These are soft reference points, not new support limits or universal BAD labels.
The 3-star band's weighted fraction of LN durations at most 40 ms is 0.0217%;
the 5-star band's is 0.498%. Their very different tail distributions motivate a
difficulty-dependent preference rather than a single hard duration floor.
The reference calculation took 14.3 seconds; no VAL or TEST charts entered it.

For a proposed interval `delta` and interpolated reference `tau`, the cost is
`min(4, 4 * max(0, log(tau / delta)))`. Intervals at or above the reference have
zero cost. Missing requests or historical intervals have zero preference;
out-of-range difficulty requests use the nearest reference endpoint. Subtracting
a finite cost from categorical log scores and renormalizing preserves legal
support and permits strong learned local evidence to override the preference.

The skeleton's release-only clock uses the minimum cost among currently eligible
held identities. This is an optimistic approximation to the still-unselected
subset's burden, not an expected player response. The mark scorer then charges
each identity it proposes to release. An old LN can keep the release clock open
while the mark preference discourages also closing a very young LN. True audio-
end closure receives no delaying cost. Counts, subset releases and native-time
support remain available, including short cross-column events.

R1 applies the larger of HH and RH costs to each proposed attack column, then
sums across the complete row. RH applies only to the first head after a release;
the maximum avoids charging the same attack twice. This can prefer an available
column over one released 25 ms earlier. The joint row model and typed feasibility
still govern the coupled choice. LN lifetime preferences belong upstream because
row assignment cannot move a tail that the skeleton has already fixed.

The implementation composes with `LnFeedback` and records both policies in runtime
metrics. It reads only generated resource/column state, candidate times and current
controls. It has no reference-tail leakage or new neural parameters, and does not
change training likelihood. Native evaluation must establish whether fewer extreme
intervals are accompanied by preserved musical organization, range control and
style; rarity alone is insufficient evidence of quality.

With the retained weights and amount feedback, the fixed recovery preference
completes the 15-case three-audio panel. On the 12 static cases, star error
changes from 0.580 to 0.435, LN-fraction error from 0.0380 to 0.0377, and short
LNs from 236/12,217 (1.93%) to 73/12,371 (0.59%). The initial sub-0.5% target
is not met. Head-count ratios relative to amount feedback have
minimum/median/maximum 0.910/1.009/1.050, so the improvement is not explained by
overall sparsification. Maximum cached startup/window service is 0.427/0.432 s.

The 32-second overrides still do not reliably realize difficulty 5. Their
approximate levels are 3.863, 3.256 and 3.496 for Zenithfall, Hysteric and
As It Was; the later restored difficulty-3 ranges are 3.122, 3.270 and 3.417.
Each range retains its own requested LN proportion and physical boundary facts.
Improved static averages do not establish short-range difficulty control.

Two Lens contexts and all four pages were read with complete tables. Mixed
TAP/LN groups, sustained layers, subset releases and short cross-column timing
remain. The Zenithfall high-control peak still contains 25/30/31/39-ms holds;
two heads at 297227 ms end 1 ms apart. The Hysteric high-LN peak has varied
107–279-ms holds with 362/472-ms layers and no LN at most 40 ms inside that
four-second witness. No listening or player test was performed.

A residual 45-ms same-column attack at Hysteric 138817 ms has exactly one legal
row realization: the other three columns are held. The previous head on its only
free column was at 138772 ms. A 57-ms Zenithfall attack at 287306 ms likewise
has only one realization for its requested two heads and fixed release subset.
These failures cannot be repaired by selecting a different current row.
The skeleton's head time/count decision must account for recent attacks and
ongoing holds together. Hard recovery feasibility alone does not preserve the
information needed for every softer gameplay response.

The optional `RecoveryPreference(head_pressure=4)` adds a small skeleton-owned
workload state for this case. At candidate time t, let tau be the requested
difficulty's HH reference. Count recent heads in `(t-tau, t)` and subtract from
four the LNs held throughout that interval. If n is the recent count and k the
remaining key count, use the overload potential
`Phi(n,k) = 4 * max(0, n-k)**2 / max(k,1)`.
Adding h heads costs `min(4, Phi(n+h,k)-Phi(n,k))`. The head clock evaluates
one new head; the mark scorer evaluates its actual count. Release-only events
have zero added head cost. Setting `head_pressure=0` retains the earlier policy.

This distinguishes the observed three-held-key repetition from a rapid
four-column expansion. Two heads followed 3 ms later by two more need not
overload four free keys. In the Hysteric failure, two keys are held throughout
the lookback and two recent heads consume its remaining two-key capacity;
adding another head 45 ms later has positive cost.

The potential is a coarse workload preference. It cannot identify a column,
separate all previous repetitions from new ones, or guarantee comfortable
future realization. Released keys and detailed geometry still need their own
responses. The implementation stores only recent skeleton head times/counts
over the maximum reference horizon, 169 ms for this profile. Planner snapshots
carry this history across rollback, and control changes preserve it while its
contents expire by actual elapsed time. No R1 TAP-column history, hidden state,
reference future or new neural parameter enters the planner.
