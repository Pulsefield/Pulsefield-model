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
