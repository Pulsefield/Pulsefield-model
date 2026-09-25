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
Optional finite LN-amount feedback is applied inside R1 and is not a per-row or
end-of-scope quota. Empirical head/release recovery preferences also compare R1's
complete candidates. The old scalar object-count correction is not applied to
the timing skeleton.

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
