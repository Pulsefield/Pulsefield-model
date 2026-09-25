# Scoped timing with complete-row R1 decisions

This research implementation restores the decision boundary between a timing
skeleton and an R1-derived arranger. The typed resource prototype fixed TAP/LN
counts and release identities before R1 ran. That restriction prevented R1 from
reducing an excessive chord or choosing a different release. It is not part of
the [V3 generation contract](../formulation/notation.md).

## Information and decisions

| Component | Inputs | Decision |
| --- | --- | --- |
| H timing | Complete audio, previous H times, scoped controls | Next head-bearing row time |
| Release timing | Complete audio, H/R timing history, H preview, committed LN ages/occupancy, controls | Release-only event time |
| R1 | Direct audio, full row history, exact replay, timing-only preview, scoped controls, candidate consequences | Complete simultaneous row: head count, TAP/LN kinds, release identities and columns |
| Scheduler | Published prefix, head lookahead and R1 feasibility responses | Query order, constrained release timing, publication and future-control revision |

H requires at least one head; it never specifies a chord size. Release-only
events require a nonempty release row. H rows may also close existing holds.
Only actual LN projection feeds the release preference network. Changing TAP
count or layout while preserving H/R times and LN state leaves both skeleton
network inputs unchanged. R1's execution-feasibility response additionally
constrains the release sampler, as described below. The effective release law
therefore need not remain unchanged. There is no typed future count or
source-tail input to R1.

The shared encoder retains the canonical fine Mel branch and full-song coarse
context. Complete audio is available in training and inference. Generated head
lookahead is provisional timing, not committed future rows. Future release times
depend on actual LN decisions and are generated online. BOS uses the learned
empty-history state without a supplied thirty-row seed.

Difficulty, LN fraction and each style attribute use independently owned control
scopes. Their values and known bits reach R1's composition and geometry paths
directly, as well as both timing factors. Missing style remains unspecified.

## Complete-row probability and the frontier

R1 internally groups candidates by `(head count, new LN count, release count)`.
A small readout sees audio, R1 context, head preview and controls and assigns
count-family mass. The inherited complete-row scorer supplies relative geometry
inside each family. These are factors of R1's own policy, not skeleton tokens.

For a supported complete row `a` in count family `k(a)`, the final law is

$$
q(a)\propto p_{\rm count}(k(a)\mid X,H,C,K)\,
p_{\rm layout}(a\mid k(a),X,H,C,K)\,
\exp g_{\rm frontier}(H,C,K,a).
$$

The frontier score enters after within-family normalization and before one
normalization over all rows. A four-key row therefore retains its candidate cost
even when it is the only layout in its family. The inherited `frontier2` features
remain an approximate consequence representation; this does not establish a
complete or calibrated V3 gameplay frontier.

The LN proportion uses unbounded learned local preferences and a scoped log-odds
shift within a fixed head-count/release-count group. The group's mass reads the
actual controls. Local all-TAP passages remain possible under a high-LN request.
Empirical head/release recovery preferences compare R1's complete candidates.
Optional LN-amount feedback then tilts the resulting distribution inside each
fixed `(head count, release count)` family. It preserves that family's probability
mass and conditional layout odds at fixed new-LN count. The old scalar
object-count correction is not applied to the timing skeleton.

The amount controller remembers a bounded log-odds correction. After committing
`h` heads with `l` new LNs, it updates that correction by `(rho*h-l)/8` and clips
it to `[-2, 2]`. A finite integral state can compensate a persistent preference
bias without requiring persistent proportion error. Projection discards further
accumulated debt at the bound. Its effect is not guaranteed when contextual
preferences exceed the finite correction or a scope contains few heads.
Each effective LN request episode starts with zero correction; difficulty/style
boundaries alone do not reset it. There is no per-row quota, scope-expiry catch-up
or remaining-time input. Learned local all-TAP and LN passages remain possible.
Training uses the learned law; this controller is an explicit sampling policy
whose quality must be assessed on generated, separately reported ranges.

R1 can additionally receive an audio/control prediction of mean head objects per
second. Its optional finite demand feedback compares that mean with its own
recent committed head count and softly changes complete-row probabilities by
candidate head count. It neither supplies counts to the skeleton nor changes
H timing. At fixed head count, all LN/release/layout odds remain unchanged by
this particular tilt; the subsequent LN feedback preserves the resulting
head/release-count mass. Both mechanisms act inside R1's joint decision.

The nominal mean is not a plan or a difficulty measure. Every H still requires
at least one head, so an overly active skeleton can make a lower head-object
reference unattainable. A timing-rate model would instead need onset-row targets.
The retained demand readout requires the full-audio encoder and per-field control
encoding on which it was fitted. Control revisions rebuild its future reference
while retaining actual count history; there is no expiry quota.

An independent optional H activity readout estimates head-bearing chart rows per
second from the same audio/control representation. It must be fitted to H-row
counts: simultaneous chord members count once and release-only rows count zero.
The object-demand weights are not interchangeable with this readout. Neither
readout identifies acoustic transients; one sound may still support many chart
events, including a sustained jack construction.

The H sampler can compare its own exponentially discounted onset count with the
integrated mean activity and apply a finite native-logit correction. The current
research policy uses a four-second decay, a four-onset pseudocount, gain two and
a correction bounded to `[-2, 2]`. Millisecond timing, the learned local history
modulation and all supported timing patterns remain available. This is a mean
calibration hypothesis, not an exact density target or a stability theorem.
Its sampling ledger counts every provisional H once; scoped lookahead revision
restores that ledger with its timing cache and RNG. R1 materialization never
updates it. The low-rate readout's 500-ms pooling sets its context resolution,
not the output timestamp grid.

The training distinction matters. A layout loss conditioned on a supplied count
group is invariant to an additive group score and cannot calibrate group mass.
The restored joint likelihood supervises R1's count and layout choices together.
Widening the old inference mask alone would leave those scores uncalibrated.

## Feasibility and scoped publication

The research response profile retains HH/RH/HR intervals of 60/50/50 ms. It is an
implementation support choice, not a change to legal V3 rows or a physiological
law. R1 checks candidate transitions against this profile and a finite future
H horizon. The optimistic continuation may close existing LNs and use one TAP per
future H; it verifies existence, not the probability or comfort of that future.

R1/scheduler derives a release window from exact replay and proposed H times.
Simulating one TAP per H on currently closed keys identifies the first H that
requires another key. Its time minus RH is a necessary release deadline; actual
LN ages determine the earliest release. These two bounds constrain the release
sampler without selecting a release identity or a chord. The preference network
still reads only audio, timing history, LN projection and controls.

This coupling is necessary under distinct recovery intervals. Closed keys can
still be recovering from TAPs. An old held key may become usable after an earlier
release, before those TAP keys recover. Counting only unoccupied keys loses that
distinction and can let the sampler wait beyond the last viable release time.
The response is an execution-feasibility projection for this continuation family,
not the full V3 gameplay frontier.

For example, after a row taps two columns, suppose one other column is ready and
the fourth has an older LN. Upcoming H times are 19 and 53 ms later. The ready
column can serve the first; the tapped columns recover only after 60 ms. Releasing
the LN 1–3 ms after the current row makes it usable by the second H under RH=50.
The release window is therefore real despite three columns being unoccupied.
This case occurred in native generation and cannot be repaired by a lower NLL.

A necessary release wait is normalized conditionally on an event by its deadline
in both training and inference. A publication window does not truncate that
normalizer, force a tail or pretend the song ended. All holds close at true audio
termination.

A future control update keeps published rows, fixed empty time and open holds.
Queued H times before its start remain. Timing is also retained through the
published boundary's fixed recovery horizon, 100 ms under this profile: R1 has
already selected published actions against that preview. Remaining H timing is
regenerated. R1 and release timing receive the requested control values at their
actual scoped times; a very near-term request may retain H timing for those
first 100 ms. This preserves feasibility without feeding TAP clocks or chord
counts back into skeleton sampling. The retained timing is a scheduler policy,
not a claim that unpublished rows are committed or that a request is exact.

`ControlledSession` provides `publish_to`, `coverage` and `update_controls` with
the existing `ControlSchedule`/`ControlSpan` interface. The scheduler is an
in-memory research implementation; durable service recovery is not supplied.

## Learning and qualification

Initialization retains compatible full-audio and R1 tensors from the ranked
typed study. The typed mark predictor and typed preview are not copied. Timing,
scope projections and the R1 count readout have explicit initialization records.
The first bounded fit freezes the audio encoder while updating the probability
factors and R1 decision modules. This is an experimental choice, not a permanent
boundary around R1 or audio adaptation.

Each source chart supplies its own continuous eight-second training intervals.
Source-scoped strain/LN controls and original human style scopes remain separate.
The strain label is an offline proxy with complete source endpoints; those
endpoints never enter generated state. Profile-incompatible intervals are
rejected and counted, not edited into new labels. Proposal sampling conditioned
on that rejection need not retain uniform accepted group mass.

Likelihood is a diagnostic for fitting the restored law. Selection still requires
native audio generation, separately evaluated control ranges, actual timing and
action organization, LN articulation, expressive coverage and service latency.
A fixed audio-generated H trace can isolate R1's restored choices, but cannot
replace native qualification of the complete system.

The 2,500-update research checkpoint was evaluated on Zenithfall, Hysteric and
Take with separate whole-song 3-star requests at LN fractions .2 and .7. The
following errors are means over three fixed audio/seed pairs per control cell;
they are not population uncertainty estimates.

| R1 sampling policy | Requested LN fraction | Absolute LN-fraction error | Absolute star error |
| --- | ---: | ---: | ---: |
| Proportional amount feedback | .2 | .0548 | 1.1451 |
| Projected integral amount feedback | .2 | .0017 | 1.0559 |
| Integral feedback plus R1 mean-head reference | .2 | .0017 | .8527 |
| Proportional amount feedback | .7 | .1185 | 1.5868 |
| Projected integral amount feedback | .7 | .0089 | 1.4433 |

The weights, audio and H times were fixed across these sampling comparisons.
Later row/release states differed after changed choices. The checkpoint identity
is SHA-256 `0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8`;
the optional mean-head readout is the separately fitted scoped 2,000-update
demand predictor. These checkpoints and generated maps are local research assets.

Three additional 32-second difficulty/LN overrides produced LN fractions
.7410/.7225/.7126 for requested .7 with integral feedback. They were evaluated
separately from the preceding and restored ranges. A short restored Take range
contained only 34 heads and produced .2941 for requested .2. Longer-range amount
calibration therefore does not establish precise short-range control.

Lens inspection retained overlapping holds, differing release times, TAP passages
and changing chords. Low-difficulty calibration still fails: a dense Zenithfall
crop already contains mostly single-head rows, whereas a Take peak remains
predominantly repeated double groups. An aggregate head count cannot distinguish
these demands. The R1 mean-head reference improved the tested star error by only
.2032, below its .25 expansion criterion, and remains optional. Neither policy
establishes musical alignment, broad style control or final playable quality.
