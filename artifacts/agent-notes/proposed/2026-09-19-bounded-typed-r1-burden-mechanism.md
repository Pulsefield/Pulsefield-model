# Agent Note: Are the remaining R1 bursts forced by timing or learned occupancy?

Note ID: 2026-09-19-bounded-typed-r1-burden-mechanism
Status: proposed
Kind: investigation
Created: 2026-09-19
Updated: 2026-09-19
Product revision: 15d27db0b0723d7f606b1429c901d26c1f61e5ac
Scope: Retrospective diagnosis of frozen R1 outputs on the 24 additional VAL groups
Related: 2026-09-19-bounded-typed-r1-unused-group-confirmation

## Finding and limits

The inspected rapid bursts are not unavoidable consequences of the supplied
timing. Initialization 172 has 25 below-40-ms same-lane pairs with only one lane
free immediately before the later attack. For every one, the existing R1 support
permits releasing a generated hold at the preceding candidate and attacking
that other lane next, without a below-40-ms head pair. These are verified legal
two-row alternatives, not generated repairs or proof of their naturalness.

Some harmful predecessor choices have substantial model probability: about 25%
and 26% before the two masked 172 pairs, with the latter choice locally most
probable. Initialization 171 also has a different failure path: an exceptionally
low-probability mixed row introduces a hold before a later optional repeat.
Thus neither unavoidable legality pressure nor rare sampling tails alone explain
all the observed failures. Why the model assigns these probabilities remains
open: optimization, representation, training coverage and generated-history
feedback have not been isolated from one another.

Preserve the original confirmation's REFINE result and all eight masked judgment
bytes. This investigation does not establish overall playability, select a new
checkpoint or alter the task. Its scope is the fixed 24-source/144-generation
collection and the explicitly selected local histories below.

## Conditions and measurements

Both 2M-exposure R1 checkpoints, source revision, external R/H candidates,
complete seed objects, native temperature-one sampling and generation seeds
17/19/23 remain unchanged. The owning confirmation Note identifies both complete
checkpoint digests, group selection and original numerical/semantic guards.
No training, resampling, source edits or output filtering occurs here. All new
artifacts belong under `artifacts/bounded-typed-continuation/confirmation-20260919-v1/`
in the implementation worktree.

A rapid pair means consecutive heads on one lane less than 40 ms apart, using
the existing diagnostic locator rather than a new universal quality cutoff.
Occupancy is measured immediately before the current row. Final LN endpoints
are reconstructed from completed output for diagnosis only; R1 suffix-born LNs
do not expose those future endpoints to the model. They choose releases rowwise.
Only supplied seed objects have fixed future endpoints in R1.

The diagnostic recounts every generated output and each source once. All 24
source summaries match the six repeated source summaries in the original raw
screen. Both model aggregates match the prior independent readout. The admitted
sources already exclude same-lane close/head coincidences, so their absence here
cannot validate that admission policy over the original full corpus.

| Collection | Outputs | Suffix heads | LN heads | Rapid pairs | Pairs with one free lane | Maximum rapid run |
| --- | --- | --- | --- | --- | --- | --- |
| Source | 24 | 48553 | 8747 | 0 | 0 | 1 |
| R1 171 | 72 | 109174 | 27667 | 10 | 0 | 2 |
| R1 172 | 72 | 118827 | 63997 | 43 | 25 | 3 |

All ten 171 pairs and eighteen 172 pairs have multiple lanes free. Of the other
25 pairs in 172, 24 have a different lane actually releasing at the later attack;
one has no such release. Counting those releases as newly available would change
the vocabulary and support, not merely fix a mask bug. Simply reclassifying those
existing releases does not remove the multiple-free cases or the first pair of
the longest run; a retrained, changed-vocabulary model has not been tested.

## Why same-time reopening is not the selected remedy

The local schema deliberately permits one of EMPTY/TAP/LN_START/LN_CLOSE per
lane per row. `Schedule.row_possible` allows only CONTINUE or CLOSE on an open
lane; `_future_room` requires a lane already free before a required onset.
The corresponding native contract tests exercise this restriction explicitly.
It was not introduced accidentally by the diagnostic.

The upstream osu! implementation also treats same-column end/start coincidence
as a conflict. At revision `ebaf7e9910ef3755308dec2c7950d916aabc545b`, the
[mania checker](https://github.com/ppy/osu/blob/ebaf7e9910ef3755308dec2c7950d916aabc545b/osu.Game.Rulesets.Mania/Edit/Checks/CheckManiaConcurrentObjects.cs)
restricts the shared concurrency test to a column. Its
[base predicate](https://github.com/ppy/osu/blob/ebaf7e9910ef3755308dec2c7950d916aabc545b/osu.Game/Rulesets/Edit/Checks/CheckConcurrentObjects.cs)
includes equality at the preceding object's end, with 2 ms leniency from
`CheckUnsnappedObjects`. The
[mania tests](https://github.com/ppy/osu/blob/ebaf7e9910ef3755308dec2c7950d916aabc545b/osu.Game.Rulesets.Mania.Tests/Editor/Checks/CheckManiaConcurrentObjectsTest.cs)
also flag a separate 7.25 ms release-to-next-head gap as almost concurrent.
This is source inspection of editor behavior, not a new physical playtest or a
reason to import every ranking guideline into the model contract.

The native predecessor check replays each relevant generated history. At the
candidate immediately before every one-free rapid pair, it changes one unknown
hold's CONTINUE to CLOSE while retaining the other actions, then tests a TAP on
that lane at the following candidate. All 25 cases have at least one legal
alternative whose time since the lane's last head is at least 40 ms. Known seed
obligations are never shortened. This confirms a local avoidance route within
the current task, while leaving release-to-head burden and subsequent musical
quality unproven. None of these alternatives replaces an original chart.

## Probability audit distinguishes failure paths

The six inspected histories are the three original masked pair witnesses,
the deterministic longest 172 run, and each model's shortest LN. At 47 generated
query positions, rebuilding exact state and finite raw context reproduces the
logged chosen-row log probability exactly: maximum absolute difference zero.
The same positions are also scored with source history as a descriptive control.
Those controls change the whole history and cannot isolate one hidden cause.

The next-candidate hazard sums the probability of current legal rows whose
resulting state leaves every lane either held or attacked less than 40 ms before
the next required candidate. It is measured only when that immediate next
candidate is an onset. This quantity is diagnostic; no hazard penalty or mask
is applied to training or decoding.

| Generated-history decision | Chosen row, columns 0-3 | Probability of forcing a rapid attack next | Consequence |
| --- | --- | --- | --- |
| 172, dense source 485ff4418e5e, seed 17, 43422 ms | TAP, continue, continue, continue | .249440 | Forced column-0 attack 26 ms later |
| 172, ordinary source 231ed043719f, seed 17, 116982 ms | TAP, continue, continue, continue | .263214 | Forced column-0 attack 35 ms later; chosen row is local argmax |
| 172, dense source 5e7685136684, seed 19, 65234 ms | continue, continue, continue, TAP | .158111 | First 29 ms repetition |
| Same output, 65263 ms | continue, continue, continue, TAP | .085853 | A further 29 ms repetition |

At each forced later attack, the rapid-attack probability is one under the
observed state. The choice that makes it unavoidable occurs earlier. The source
history at these locations has distributed TAP motion, and the audited source
label does not create the same next-row hazard. This rejects the explanation
that the external onset times alone require the burst.

Initialization 171's masked wider pair is different. At 128935 ms, it samples
the joint row `(TAP, LN_START, EMPTY, TAP)` with probability about .000067
(0.0067%). The resulting column-1 hold persists until 129508. At 129274, the
repeat on column 0 is optional: its chosen row has probability .024324 and
the total rapid-attack mass is .028593; columns 2/3 remain free. The low
probability refers to the complete mixed row, not the marginal LN-start event.
This supports a tail-sensitive path for this particular failure, without making
tail truncation a demonstrated remedy for the high-mass 172 cases. Local greedy
selection would still choose the harmful 172 predecessor at 116982.

## Short holds and complete diagnostic views

The duration recount uses strict less-than comparisons. The source has no LNs
below 20 ms and thirteen below 40 ms; its minimum is 26 ms. Across the 72 outputs
per model, 171 has five below 20 ms and 71 below 40 ms, minimum 18 ms. Model 172
has three below 10 ms, 26 below 20 ms and 411 below 40 ms, minimum 8 ms. These
are descriptive counts, not new rejection thresholds.

All 27 diagnostic images were inspected: three selected extremes, each with the
source and both generated initializations at the same generation seed, each
shown over three pages of a complete 6s context. The judgment core is 1s.
Selections are deterministic from the fixed report: longest primary run, then
smallest mean gap/source SHA/seed/time/lane; shortest LN per model by
duration/source SHA/seed/time/lane. These are output-selected diagnostics, not
another blinded confirmation or a collection of human gold judgments.

The longest 172 run is column 3 at 65234/65263/65292, under holds on columns
0/1/2. Its 64763-65763 core has 24 heads, compared with the source's 34: lower
head count does not prevent concentrated same-lane burden. Source heads at those
three times use columns 0/3/2; initialization 171 uses 1/3/0. Both keep the burst
distributed. The surrounding 172 output has independent LN relationships, but
those positive relationships do not settle this localized quality concern.

The shortest 171 hold is column 0 at 235943-235961, inside an articulated LN
chain: column 2 at 235831-235850, column 1 at 235850-235943, then column 0 and
column 1 at 235961-236183. The source uses TAPs on the close onset pairs and has
no new LN heads in the 235443-236443 core; both generated versions retain a
LN-led phrase. The 18 ms hold is about .081 beats at the local 222.222229 ms
beat length. Its duration alone neither establishes impossibility nor invalidates
the coherent surrounding relationships.

The shortest 172 hold is column 0 at 182217-182225. Columns 2/3 already hold
182164-182225, and column 1 releases at 182217 then re-enters at 182225. Thus
the supplied close onset pair becomes an 8 ms held action inside a persistent
LN passage. It is about .0192 beats at the 416.666667 ms local beat length.
The 181717-182717 core has 28 LN heads and one TAP for 172; source and 171 each
have 25 TAPs and no holds, using different lanes for the 8 ms-spaced heads.
Both arrangements prove that the timing candidates do not require this short LN.
The probability of 172's actual LN-start row is .400286 and of its subsequent
release row .133436 under their respective generated histories. This is not
explained by an exceptionally unlikely final draw alone.

These contrasts identify local articulation concerns while preserving the
Foundation-calibrated recognition of real moving and independent held structure.
The original masked quality uncertainties remain unresolved. No human difficulty
judgment or musical intent is inferred from these unannotated source counterparts.

## Evidence and execution

All paths below are relative to the artifact owner named above. Input, source,
row-log and output hashes are checked by the drivers; original masked judgments
were also rechecked after all diagnostics.

| Evidence | SHA-256 |
| --- | --- |
| `diagnose_burden.py` | `b47e26d9159a64f285d862a7cb0ba2ca320068b9050df768140c086d1deb7c6d` |
| `burden-diagnostic-v2/report.json` | `9d226c790a78b9a302ae08f6c0d0ca4b294e3ccf9df9f494051f537e2851c6a8` |
| `verify_predecessor_support.py` | `538e2cf878c3265f665dcb5f684eb145638355f2bb5f8f98c8c2620ce03fc95d` |
| `burden-diagnostic-v2/predecessor-support.json` | `8515d54b2eed77028f57eb13107ca40205095ae89a73b4a0528858f50f5fa918` |
| `audit_local_probabilities.py` | `690ec1a738b163b3512fef237093ce86589fa869cab1cc32b5f3069018f1c111` |
| `burden-probabilities-v1/report.json` | `7568afcf13f2f34674565fe7f332229b80bb575b3b7db23561d8f5397a94646d` |
| `render_burden.py` | `37c0f2d296ab8fbcba5843d1a4d024aaf9d95516038991100db943da27a9569b` |
| `burden-inspection-v1/manifest.json` | `5fd2826c83b212405fd02bfd770001f23f05a57a32d0e35e17e2ab5bbf3fc4dc` |
| `burden-inspection-v1/review.json` | `8de196b0fd390621fef49f03a5bf521de6e5fe340c3351af1296726a8ef8ac2c` |

The first recount driver failed during selection serialization because a locator
and note type both used `kind`. Its bytes and failure are preserved under
`burden-diagnostic-v1/`; it wrote no report. The corrected driver uses a separate
`locator` field and a fresh v2 output, completing in .589s. This preparation
failure changed no measurements or original outputs. The probability audit
completed six cases in 4.947s on CPU with one thread, largest sampled physical
footprint 213435328 bytes and zero swap growth. The predecessor support check and
rendering also exited successfully. Commands used Python 3.10 with the explicit `mps`
extra; inference itself ran on CPU. All processes are terminal.

## Decision and next discriminator

Keep the current same-time close/restart restriction. No demonstrated witness
requires loosening it or adding oracle LN information. The result does not prove
that a different timing representation or richer condition could never help.
Avoid an uncalibrated hard minimum attack gap or LN duration: short cross-lane
onsets, brief held articulation and repeated same-lane demand are different.

The next research comparison should separate probability calibration from
learning consequences of occupancy on generated histories. A decoding-only
change could test sensitivity to improbable mixed rows, but must retain real
LN coordination and cannot be credited with solving a high-probability harmful
predecessor without direct evidence. A learned action-consequence representation,
training on recoverable perturbed histories, or a more stable optimization policy
are other hypotheses, not established fixes. Freeze one bounded comparison and
its quality/coverage guards before new sampling or training; do not continue an
unbounded inspection search for a passing example.

The reusable native inference entrypoint remains a separate practical gap and
can now be implemented: every source-pinned diagnostic is terminal. It should
preserve the frozen generator, external timing/seed boundary, exact restoration
and verified export rather than embed an untested quality policy. No product
source, checkpoint, Note lifecycle status or remote branch changed in this
investigation. The overall goal remains active and unfulfilled.
