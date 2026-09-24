# Composing head plans with complete R/row models

Exchanging generated onset plans between the original and continued shared
models does not produce a consistently better controllable generator. The
original plan with the continued materializer retains useful LN relationships
in some audios, but fails the predefined control criteria. The reverse pairing
also exposes a system failure: four rejected proposals exhaust the publication
budget before a song is complete, despite modest window service time.

These results distinguish expressive capacity, requested control and continued
playback coverage. Neither combination is adopted as a playable endpoint.

## What is exchanged

The original shared-profile checkpoint is denoted I, and its shared continuation
from the [density-routing study](profile_density_routing_evaluation.md) is S.
The [head-factor diagnostic](head_factor_drift.md) found mixed audio-base,
residual and sampled-history contributions to H-count drift. This comparison
asks whether the two checkpoints nevertheless contain complementary conditionals:

$$
p_{a,b}(H,R,Y\mid A,z)
=p_a(H\mid A,z)\,p_b(R,Y\mid H,A,z).
$$

Here H is the sequence of times containing at least one attack, R denotes
release-only event times, and Y denotes complete row actions. An H row can also
release an existing hold. B abbreviates the complete R/row materializer.
The prospective pairing is $H_I/B_S$; $H_S/B_I$ is the reverse diagnostic.
The saved $H_I/B_I$ and $H_S/B_S$ charts are the diagonal comparators.

Each conditional retains its own full-audio Mel encoding, profile projection
and learned representation. Only the generated H sequence crosses models.
Rows read audio directly, the H preview, exact committed state, row history
and frontier2. R reads audio, skeleton history and the committed LN projection.
The pilot H factor reads its own history and audio, without runtime R/row or
LN-state feedback. That last restriction is a pilot approximation, not a
requirement that future skeleton models ignore LN occupation.

No source chart, redline timing, seed chart, hidden source tail or human style
label enters generation. Full audio is available to both factors. The comparison
uses eight pinned audios, explicit profiles 1 and 2, and seeds 251701–251708.
The two profiles are joint descriptor requests, not isolated style interventions.
No weights are fitted during this comparison.

Four preliminary own-plan replays—Fool Moon and Hysteric, profile 2, each
checkpoint—exactly reproduce their saved row hashes and independently reparse.
This verifies fixed-plan/online-preview parity for those controls. Crossed
histories can still lie outside training support; shared seeds cannot hold
R/row history fixed after the trajectories diverge.

## Control response on the complete direct cohort

The measured descriptor is $(\text{H/s},\text{heads/H},\text{LN-head fraction})$.
For each chart, log H rate, log heads/H and arcsin square root LN fraction are
standardized with the same TRAIN-derived profile bank. Squared differences are
averaged over the 16 audio/request cases, with equal case weights; total error sums the three
components. This measures requested descriptor realization, not playability.

| H source / materializer | H error | Width error | LN error | Total |
| --- | ---: | ---: | ---: | ---: |
| I / I, original diagonal | .873858 | 2.370422 | 4.096142 | 7.340423 |
| S / S, continued diagonal | 1.763494 | 3.983159 | 3.444761 | 9.191414 |
| I / S, prospective composition | .873858 | 2.698137 | 3.812930 | 7.384925 |
| S / I, reverse composition | 1.763494 | 2.980134 | 5.462095 | 10.205724 |

The prospective pairing was required to reduce total and LN error by at least
15% relative to I/I, while increasing width error by no more than 10% and
preserving the H component. Instead, total error increases .6%, LN error falls
6.9%, and width error increases 13.8%. Only H preservation passes.
The reverse pairing improves width relative to S/S but worsens LN and total
error. Simple composition does not supply the missing overall improvement.

The mean conceals different materialization responses. Profile 2 requests
6.060 H/s, 1.429 heads/H and .734 LN-head fraction. Its direct whole-chart LN
fractions are:

| Audio | I / I | I / S | S / I | S / S |
| --- | ---: | ---: | ---: | ---: |
| Fool Moon | .159 | .134 | .108 | .167 |
| Zenithfall | .242 | .627 | .040 | .100 |
| Hysteric | .334 | .663 | .031 | .042 |
| Goodbye | .247 | .062 | .318 | .126 |
| Revenge | .078 | .303 | .140 | .672 |
| Take | .177 | .258 | .051 | .208 |
| As It Was | .084 | .024 | .043 | .510 |
| Yomi yori | .074 | .165 | .084 | .201 |

With a fixed materializer, changing H changes subsequent LN occupation, releases
and row history. With fixed H, changing the materializer changes all those
decisions too. The contrasting Hysteric and Revenge responses show why a single
statement such as “denser skeletons destroy LN” does not explain this cohort.
The intervention isolates the supplied plan from the materializer's weights;
it does not separate every downstream source of trajectory divergence.

## Short attacks and a publication failure

HH counts successive attacks in the same column with a gap strictly below
20 ms; TAP and LN heads both count. Exactly 20 ms is excluded. The separate
experimental RH screen counts the first attack after a same-column release
when its gap is at most 20 ms. Neither predicate bans close cross-column
timing, ordinary Jack/chordjack repetition, or short LN duration.

All 32 direct crossed charts finish: I/S contains 0 HH and 16 RH pairs; S/I
contains 28 HH and 20 RH pairs. Those totals cover the whole direct cohort,
not only inspected excerpts.

The [publication policy](../../src/ensomi_model/research/planned_audio_continuation/buffering.py)
proposes eight seconds plus a 20-ms sampled halo,
keeps H fixed, and permits four R/row proposals per window. Only a validated
prefix is published. Of the planned 32 screened calls, 21 finish, the next call
stops on the attempt limit, and ten are unattempted. Both screened aggregate
control comparisons are therefore incomplete. Successful subsets are not
substituted for the declared cohort.

The capped call is S/I, Take, profile 1. Publication stops at 113296 ms of
144236 ms, with 1476 H delivered. Its requested next window ends at 121296 ms;
all four proposals reach a 121315-ms cut and check through 121335 ms.

| Attempt | Rejected same-column attacks, ms | Gap |
| --- | --- | ---: |
| 0 | column 3: 115114 → 115119 | 5 ms |
| 1 | columns 2 and 3: 115105 → 115119 | 14 ms each |
| 2 | column 3: 114285 → 114294 | 9 ms |
| 3 | columns 2 and 3: 115105 → 115119 | 14 ms each |

The failed window takes .604 seconds; the largest window service in this call
is .795 seconds, below the two-second bound. This is observed proposal-budget
exhaustion, not observed compute saturation. Its published prefix has zero
HH/RH, yet cannot provide continued coverage. A publication safety check alone
does not establish liveness.

Full rejected trajectories were not captured; only exact conflict pairs and
proposal metadata remain. The separately generated direct Take counterpart
contains a useful, distinct witness: columns 0/2/3 attack at 115105, column 1
at 115114, then column 0 at 115119. All columns have attacked in the preceding
14 ms, so changing only the last column cannot remove HH. With those H times
preserved, an earlier chord must contain fewer heads: reassigning five attacks
across four columns within 14 ms cannot satisfy HH. No hold participates here.
Nearby, a 9-ms 0/3 → 1/2 exchange avoids HH without changing its timing.

Other witnesses have enough rested columns for the chosen head count but select
a recently attacked column. The Take profile-2 direct chart also combines a
9-ms HH with a 9-ms release-to-attack: its only HH-rested column has just
released. Current row choice, earlier chord size and LN release state therefore
produce different constraints, even when the reported bad-pattern count is one.

## Inspected organization

The available-output review covers 80 new scopes, all 258 rendered time pages
and their complete native-ms action/LN articulation tables. Six human-reference
scopes reuse exact identities from completed parent reviews. It includes every
direct HH witness, all available accepted windows that required resampling,
and the selected fixed/dense contexts. As It Was and other explicitly labelled
direct fallbacks do not replace missing screened results. Completing this review
does not complete the original 64-call system comparison.

The prospective Hysteric profile-2 long context retains independent held roles:
802-, 1363- and 1408-ms holds span three, four and six H respectively, while
other columns tap, start holds and release in subsets. Its resampled contexts
also contain common releases, staggered joins and short groups that close
before the next H. Zenithfall includes a 2083-ms hold spanning eight H with
other-column taps and new holds beneath. These are concrete expressive
relationships, beyond a high LN fraction.

The same pairing's selected Fool Moon profile-2 contexts and peak are all TAP.
Its As It Was profile-2 direct context is also all TAP, although alternating
0/2 and 1/3 doubles and a quad preserve chord organization. The reverse pairing
has mostly TAP in Hysteric's profile-2 contexts and its 148-H eight-second peak,
with some brief overlapping LN episodes elsewhere. These local contrasts are
consistent with unreliable control, not absence of all expressive capacity.

Repeated arrangements must remain representable. Reverse As It Was profile 1
repeats the 1/2/3 triple six times at roughly 160–190-ms spacing, with a 1027-ms
hold in column 0 spanning the first five H. Conversely, many LN-rich scopes
contain short simultaneous groups whose tails all finish before the next H;
their contribution to LN fraction differs from a sustained independent role.
Neither observation warrants a generic repetition ban or minimum-hold rule.

Screened charts preserve 1–19-ms cross-column splits and complementary double
exchanges. Some same-column pairs are exactly 20 ms or slightly longer. Passing
the narrow HH predicate does not certify comfortable difficulty, musical
correspondence or player experience. No listening or player test was performed;
source tags describe their own human-reviewed sections, and editor-only
120-BPM headers provide no musical timing evidence.

## Implications for model coupling and training

Frontier2 remains active in the migrated row model. Its
[candidate consequences](../../src/ensomi_model/research/planned_audio_continuation/features.py)
include actual attack/release clocks and future H times, but use `now + 1` as
the earliest possible R opportunity. That is not a prediction or commitment of
the release model. Exact replay and physical row support prohibit inconsistent
LN actions; they do not reserve rested columns for upcoming H or guarantee a
good continuation under HH/RH constraints.

Training on source histories and source H previews optimizes local conditional
likelihood. Generation instead feeds back sampled H, complete rows and LN
occupation. Descriptor requests are also only conditions, with sparse exposure
to the two selected profile regions in the
[training comparison](profile_density_routing_evaluation.md#training-approximations-and-the-remaining-causal-question).
NLL, legal replay and count calibration each check a different approximation;
none guarantees requested organization or bounded-time continuation.

The evidence motivates evaluating short-horizon continuation feasibility
before committing chord size and LN occupation, while preserving audio and
H-preview conditioning. A feasible continuation must concern actual available
columns and release choices, rather than a blanket minimum H interval. It must
also retain repeated grips and independent LN roles. Whether an explicit
typed/cardinality decision, joint finite-horizon search or a different learned
conditional achieves that remains open. This comparison does not assign a
percentage of system failure to restored R1 or justify scaling NLL training alone.

## Verification and reproduction

Execution uses source `fd2ddfdbc66c1177167487085bccfbac818ddb39`, Apple M5,
24 GiB RAM, Python 3.10.20, Torch 2.11.0 and one CPU thread. The driver stops
after 211.678 seconds under its declared guard; no run is resumed and no
proposal budget is increased. The 53 complete main outputs independently
reparse, preserve supplied H exactly, and satisfy incremental replay/coverage
checks. The capped prefix has no complete-chart export. Checkpoint files remain
unchanged; the final in-memory parameter-fingerprint assertion was not reached.

Times here concern cached canonical Mel and an already generated H plan, plus
body encoding/generation. They do not measure fresh-audio end-to-end startup.
One native seed per audio and a small curated panel limit generalization.

Checkpoint, profile-bank and audio-panel identities are pinned in the
[routing report](profile_density_routing_evaluation.md#reproduction-identities).
The local owner is
`artifacts/joint-audio/20260925-head-materializer-composition-v1`;
generated charts, checkpoints, scripts and detailed review records are derivative
research assets that may be absent from a fresh clone.

| Evidence | SHA-256 |
| --- | --- |
| Frozen run inputs | `1f2366a214aea3fcb11a4c3311febff9a9af0be26ec8f89a5e88da9cf3f2f60e` |
| Terminal partial result | `125a0c3f51570a73fb11e361db323ff3087c1c0b1051144c55b8ac75f5702fa1` |
| Failure record | `818032b1a0170f93c9a0e3a3dc5a2173ed3610eb6ebc6b57c11197e3eac0bad7` |
| Available-output Lens plan | `1e8c08cba0a16c6bcc585dc1a3fe4913bc68d4ea329a8b9581a42c819ee8011f` |
| Completed available-output review | `c7a1f1a764856b9038854e6f95d5c74f56c3e6e31183768b72387779f9591eaf` |
