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
