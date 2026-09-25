# Ranked 2–6-star action reference and frontier calibration

The first playable target is native 4K osu!mania at 2–6 stars. A census of
8,774 byte-verified ranked charts contains no same-column attack-to-attack or
LN-release-to-next-attack interval at most 20 ms. This supports retaining a
strict response constraint for that population. Relaxing release screening to
complete more generated charts would admit relationships absent from this
reference population.

The same charts contain 8,960 adjacent head-row gaps at most 20 ms, across
566 charts. Their columns do not repeat across those gaps. A global minimum
skeleton interval would therefore remove real arrangements while failing to
express the relevant distinction: which actions the player must perform after
which previous actions and held states.

Whole-chart star rating is insufficient as a generation-quality check. Most
outputs from the small planned-model panel already lie within 2–6 stars, yet
contain close attacks or implausibly short holds. The existing `frontier2`
features, training preference and row-count factorization have separate
limitations described below. No new model was trained in this census.

## Population and measurement

The local index contains 14,689 charts. Official per-beatmap metadata fetched
on 2026-08-05 identifies 9,536 as ranked, unconverted mania, four keys, with
`difficulty_rating` in inclusive [2, 6]. Source MD5 matches the metadata's
checksum for 8,774 of these, covering 3,387 beatmapsets. The other 762 versions
are excluded from the primary ranked reference because their bytes differ;
they are not counted as clean or bad charts.

The primary star value is the unrounded official metadata value at that
snapshot, at 1.0x playback. Index values are rounded and do not decide inclusion.
Recomputing the 17 visually inspected source charts with the repository's
`compute_mania_star_rating_20241007` differs by at most 0.00000437 stars.
This identifies the measurement version, not an assertion about later server
recalculations.

All 8,774 original files parse successfully. They have 12,953,523 objects and
2,272,555 LNs; no same-column overlapping objects, duplicate heads or coincident
tail/head pairs are found. The census does not use a timing-anomaly-filtered
index or remove tested short-gap events. Native source milliseconds and full
LN endpoints are preserved, without quantization, tempo normalization, snapping
or synthetic TAP releases.

The measured relations are:

- **HH:** consecutive TAP/LN-head attacks in the same column, including an
  intervening LN tail when present.
- **HR:** an LN's own head to its release, hence its full duration.
- **RH:** an LN release to the next object's attack in the same column.
- **Adjacent H rows:** distinct consecutive times with at least one attack,
  irrespective of column. Simultaneous chords are one H row.

Counts at `<20`, `==20` and `<=20` are retained separately. They describe exact
chart actions under the canonical execution convention. Neither hit-window
tolerance nor an observed player's input timing is substituted for chart time.
This is a census of a fixed local corpus, not an independent random sample of
all ranked charts; millions of note pairs do not supply millions of independent
player observations.

| Relation | Eligible instances | At most 20 ms | Affected charts | Minimum |
| --- | ---: | ---: | ---: | ---: |
| HH | 12,918,427 | 0 | 0 | 37 ms |
| HR | 2,272,555 | 8 | 1 | 19 ms |
| RH | 2,259,510 | 0 | 0 | 25 ms |
| Adjacent H rows | 8,765,631 | 8,960 | 566 | 1 ms |

The eight short HR instances are four 19-ms and four 20-ms LNs in one chart.
They account for about 0.00035% of LNs, or one of 8,774 charts. Their actual
organization is examined below rather than silently deleting them or treating
them as permission for arbitrary short holds.

## Difficulty changes the relevant reference distribution

The star strata are [2,3), [3,4), [4,5), and [5,6]. Local density counts every
attack note in a sliding half-open one-second window, including simultaneous
chord members. The table reports the median and 95th percentile of each chart's
maximum, not pooled note-weighted averages.

| Stars | Charts | Minimum HH | Minimum HR | Minimum RH | Peak attacks/s: median / P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2–3 | 3,238 | 83 ms | 30 ms | 42 ms | 12 / 16 |
| 3–4 | 2,906 | 55 ms | 25 ms | 32 ms | 18 / 22 |
| 4–5 | 2,047 | 37 ms | 19 ms | 25 ms | 24 / 28 |
| 5–6 | 583 | 50 ms | 24 ms | 26 ms | 28 / 33 |

These extrema are observations, not proposed per-star hard limits. Sparse
exceptions, song length, LN organization and the star calculator affect them.
In particular, the minimum HH is not monotonic in star level. A useful response
model must preserve local context and interactions instead of assigning every
history one scalar capacity derived from these minima.

## Source structures retained by Lens inspection

Independent Beatmap Lens parsing agrees on all 59,011 notes in the 17 selected
charts, including source lines, columns and endpoints. Inspection covers 18
contexts and all 35 time-proportional pages, with complete action and LN
articulation tables. The selection includes three minimum-gap charts per
relation and two source-hash-selected charts per star bin, drawn from the lower
and upper halves of whole-chart LN proportion. The second occurrence of the
sole short-HR exception is an additional context. Whole-chart LN strata are
sampling strata, not labels for their selected local sections.

Three examples constrain the architecture particularly clearly:

1. [Astral Empire, Witness the Fall / Extreme](https://osu.ppy.sh/beatmaps/4258022)
   is 5.5208 stars at the snapshot. At 228938–229290 ms, TAP movement uses
   repeated 17-ms cross-column pairs separated by 50 ms. The surrounding
   passage contains 66/67-ms LNs whose releases coincide with attacks in other
   columns, followed by double/triple accents. Rejecting every 17-ms skeleton
   gap would remove this actual organization.
2. [I'm kidding, Infinite Jest](https://osu.ppy.sh/beatmaps/4432298), 4.7045 stars,
   opens four LNs at 37202 ms. Column 3 releases at 37535, before its next TAP
   at 37602; the other three LNs release at that TAP. This is a concrete
   continuation relationship for frontier modeling: a needed column is freed
   early while other tails retain their shared articulation. Later, a 200-ms
   LN in column 0 releases at 39002 and is followed by a TAP at 39027, within
   a three-column grace figure.
3. [lost memory, a light received from you.](https://osu.ppy.sh/beatmaps/2012530),
   4.0845 stars, contains all eight HR exceptions. Starting at 333567 ms, four
   19/20-ms LNs accompany TAPs in two-note chords on a 156/157-ms attack pulse.
   Following holds lengthen to 39, 78 and 104 ms. The progression repeats ten
   seconds later with mirrored columns and different preceding material.
   This is a specific repeated duration progression, not a common short-LN
   population or evidence that arbitrary 3-ms releases are suitable.

Other inspected contexts include alternating chord grips, irregular TAP
movement, LN subset releases, staggered tails, temporarily all-held lanes,
and sustained holds supporting several interior attack rows.
[SCREW, Dub's Black Another](https://osu.ppy.sh/beatmaps/3177794) includes
21/22-ms LNs at roughly 88-ms head steps and longer overlapping holds nearby.
A substantially larger universal LN-duration floor would discard real source
structure. Conversely, preserving expressive support does not require giving
frequent mass to every physically legal duration.

These are source-backed agent readings. Ranked status supplies a strong
reference for normal arrangements; these inspections do not create new human
section annotations or measured player-response targets. The official
[mania ranking criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!mania)
also distinguish note rhythm, sustained holds and release coordination, with
difficulty-dependent guidance. Their named difficulty categories are not
numerical star bins.

## Existing generated charts expose additional failures

The comparison reuses all 32 complete raw flat/count-layout exports and all
62 complete current/preview exports from the
[row-conditioning study](row_constraint_evaluation.md). The two incomplete
flat outputs retain that status. File hashes and exact export-to-row replay
are checked; there is no new sampling or selection of a better seed.

Generated SR uses the same named 20241007 calculator at 1.0x. The following
counts include only complete outputs whose computed SR lies in [2,6]. Requests
remain the original arrangement profiles, which were not calibrated star
requests. These are descriptive subsets, not matched difficulty experiments.

| Model / policy | Complete outputs in 2–6 stars | HH ≤20 | HR ≤20 | RH ≤20 |
| --- | ---: | ---: | ---: | ---: |
| Flat, raw | 14/16 | 2 | 85 | 48 |
| Count/layout, raw | 15/16 | 24 | 16 | 16 |
| Flat, current | 13/15 | 1 | 64 | 0 |
| Flat, preview | 13/15 | 1 | 58 | 0 |
| Count/layout, current | 15/16 | 3 | 13 | 0 |
| Count/layout, preview | 15/16 | 2 | 15 | 0 |

The constrained HH cases are all exactly 20 ms: the prior predicate checks
strictly `<20`. Its earlier zero-HH reports remain correct under that predicate;
they do not establish zero at the inclusive boundary. HR was not screened.
Six of the 15 in-range count/preview charts contain an HR at most 20 ms.

Five additional Lens contexts, all five pages and complete numerical tables,
verify actual generated witnesses. Take profile 1 repeats column 3 across
`[03]` at 96744 and `[23]` at 96764. Flat Yomi yori opens 3-ms and 111-ms LNs
together at 456757 while two columns are free and the next H is 304 ms later.
Count/layout Hysteric opens an isolated 14-ms LN with three free columns.
Count/layout Zenithfall releases one of two LNs after 9 ms alongside attacks
in already-free columns; the other LN lasts 84 ms.

These short holds are not all forced by imminent lane exhaustion. They are
distinct from the previously traced all-held/late-R failure. A release deadline
alone therefore cannot correct both failure families.

## What frontier modeling must account for

The [formulation's frontier](../formulation/gameplay-state.md#target-response-and-frontier)
describes responses to hypothetical continuations under a declared gameplay
profile. The implemented arrangement profile—head rate, chord width and LN
fraction—is not a player-response specification. The ranked population can
calibrate structural expectations for the initial difficulty range without
being misrepresented as individual physiological observations.

Three implementation facts explain why the current prototype can pass its
existing checks and still violate that target:

- [Response preference](../../src/ensomi_model/research/bounded_typed_continuation/response.py)
  penalizes attacks too close to previous attacks/releases. It optimistically
  uses one TAP at each of two future H events and earliest possible unknown-LN
  releases. An LN's own very short head-to-release interval is not a cost term.
  Possible release time is also not the time the learned R process will choose.
- [Consequence features](../../src/ensomi_model/research/bounded_typed_continuation/consequence.py)
  include release age, but a feature being available does not establish a
  trained response semantics. Teacher-row likelihood can learn duration
  correlations without enforcing them on self-generated histories.
- [Count/layout composition](../../src/ensomi_model/research/planned_audio_continuation/counts.py)
  separately normalizes layouts within each count group. A consequence score
  constant across a group cancels from that group, so it cannot alter the
  probability of that chord/LN/release count. Frontier influence on layout
  alone does not ensure appropriate composition.

For the initial 2–6-star target, near-20-ms HH and RH should remain explicit
constraints, with exactly 20 ms included. HR needs its own declared treatment
and regression measurement. This recommendation does not change the prototype's
runtime defaults or retrospectively alter its completed comparisons.

The next model comparison should connect these responsibilities: candidate row
effects on LN occupancy, release timing and subset choice, and the demand of
the actual upcoming head plan. Candidate consequences must affect count choices
as well as layouts. Evaluate them under the deployed R process rather than
assuming an ideal earliest release. Any scheduler condition supplied back to
the skeleton must obey the existing
[LN-only information contract](audio_skeleton_information_contract.md).

Training can use complete ranked arrangements as positive structural examples,
grouped by song for held-out comparison, alongside declared contrastive response
targets for clear bad actions. A synthetic perturbation is not automatically a
bad chart label. Evaluation needs native rollouts and inspected local structure,
not only teacher-forced NLL, whole-chart SR or descriptor error. Removing all
LNs would trivially remove HR/RH failures while destroying the target support;
LN response, overlapping organization, chord variation and rapid cross-column
placement must remain regression checks. Full-audio conditioning of both
skeleton and rows remains required.

## Provenance and limits

The census uses Ensomi source `b7769afd9d7f0106f41b45d03103b719ddf1ca76`
and Lens `22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`, on an Apple M5 with
24 GiB memory and Python 3.10.20. The source census took 133.48 seconds on CPU;
the 94-export measurement took 3.77 seconds. Neither stage fits weights.

Local evidence owner: `artifacts/joint-audio/20260925-ranked-2to6-reference-v1`.
The census result SHA-256 is
`393e813bc02ea0bf7e9b7cd287193846c21d0f688092ec3749c9a71900cf8770`;
the generated comparison is
`00a120a758dbb837300ea967eca454db4300dca16559369a53f8d5d288c444ef`.
The source review is
`cda5a049dfc4e541841d13a56835f497e8738c0c67cd9a2a15b77a27120a25b5`;
the five generated-witness review is
`8614cc1d28ff12e56b12d9796cf6762aa32b501fc303d019ac4f2ad3304aa1f3`.
Their freezes bind exact source/metadata files, scripts, scope selections and
rendered evidence. Generated assets may be absent in another checkout.

The population is bounded by acquisition coverage, historic metadata and exact
byte matches. Whole-chart SR can hide local spikes; repeated song families are
not independent. No listening, input trace or player test was performed.
The census establishes strong reference-distribution evidence and concrete
modeling omissions, not an exact fraction of all generation failures attributable
to R1 or a validated complete player simulator.
