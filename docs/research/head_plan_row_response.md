# Head-plan sensitivity and candidate action responses

Conditioning full-occupancy releases before the next H removes a reproduced
deadline artifact. A matched joint fit also has zero screened close pairs, but
changes chart composition. Crossing generated head plans between checkpoints locates
strong plan-to-row effects in two selected cases. Candidate-score inspection
then distinguishes an upstream chord that forces a short repeat from a later
row that chooses an avoidable re-press.

This is an exploratory analysis of the
[planned audio model](planned_audio_continuation.md). It does not select a final
playable endpoint or quantify a population failure rate attributable to R1.

## Matched fitting and native readout

Both fits use 615 training arrangements across 240 audio groups, the same frozen
1200-update exposure plan, R1 initialization, optimizer settings and model seed.
Each has 4247438 parameters. The intervention enables the
[feasible first-release law](release_wait_conditioning.md) in both training and
generation. Audio, head, release, row and frontier2 modules continue to train.

The conditional fit completes in 907.01 s on MPS. Its checkpoint is
67b8fc8fbac5f7ca2debe99524e29ac002546f12cb8a3c4c9acf68d68d7f3839.
The original bounded checkpoint is
47d41844afc673788cb640c67a1d5e41b2b9ca75ae679298c10072200925bce6.
Population joint NLL/s changes from 40.556 to 40.463; the waiting law also
changes, so this is a fitting diagnostic, not a playability verdict.

Nine fixed audio/seed cases complete and independently reparse. The conditional
fit has zero same-lane head/head or release/next-head gaps at most 20 ms, and
every fixed review window contains heads. However, six cases fail the declared
five-percentage-point LN-fraction stability guard. A composition change does
not automatically make an arrangement bad, but this endpoint does not pass
the comparison as a clean improvement.

Good Luck seed 17 illustrates why counts must be separated. H rows decrease
from 671 to 626, while note heads increase from 825 to 1344. Mean chord size
rises from 1.230 to 2.147. LN heads increase from 204 to 227 even though their
fraction falls from 24.73% to 16.89%. The output has more chordal material,
including repeated grips and independently sustained holds. It is not simply
losing LNs or becoming uniformly sparse.

Forty new Lens pages cover the nine fixed contexts and all nine full-occupancy
episodes in seven merged contexts; fourteen reference pages match previously
inspected evidence byte-for-byte. Piano remains sparse: one reviewed output
has a 3274 ms LN without an interior attack, followed by a 6900 ms H gap.
Fine Tech and dump coverage, musical appropriateness and player experience
remain unresolved. No listening or player-verdict claim follows from these
visual and action traces.

## Crossing generated head plans

Let A denote the original bounded weights with conditional-release decoding,
and B the weights trained under that law. For Prom Queen seed 19 and Good Luck
seed 17, take each model's complete generated H plan and render both plans with
both conditional models. The weight factor includes audio encoding, release
timing and row generation; it is not an R1-only substitution.

Every arm uses the same source audio and original row/release seed. The plans
contain generated head timestamps only, with no reference rows, LN tails or
future occupancy. All four diagonal controls reproduce their complete saved
outputs exactly. All crossed arms preserve their supplied H times and
complete/reparse. No parameters are fitted.

| Conditional weights | Generated H plan | Prom Queen LN fraction | Good Luck mean chord size |
| --- | --- | ---: | ---: |
| A | A | 18.04% | 1.230 |
| A | B | 2.85% | 1.933 |
| B | A | 17.11% | 1.215 |
| B | B | 2.03% | 2.147 |

For both declared readouts, the crossed outputs lie closer to the diagonal
sharing their plan. This demonstrates a substantial plan effect in these
paired trajectories. It does not establish that every composition feature has
the same cause: Good Luck's B/A LN fraction is 8.83%, versus 24.73% for A/A,
despite their similar chord sizes. Small probability changes can also redirect
an autoregressive sample, so a shared integer seed does not fix an arrangement
choice across parameter changes.

The crossed outputs contain one 19 ms same-column attack pair and one 11 ms
release-to-head pair absent from the diagonal samples. Eighteen new Lens pages
cover all crossed fixed contexts, five full-occupancy episodes and both close
pairs. This locates concrete interactions without estimating their prevalence.

## An earlier chord can make a later bad attack unavoidable

Successive same-column attacks below 20 ms are treated as a high-confidence bad
pattern. The threshold is strict, applies to TAP and LN head attacks, and does
not by itself classify cross-column timing, LN duration or release-to-head gaps.

In Good Luck A/B, all lanes are free and their previous attacks are older than
20 ms at 47163 ms. The sampled row taps columns 0, 1 and 3. The next required
heads occur at 47166 and 47182 ms. Thus five attacks must occupy four columns
within 19 ms. Once that first three-head row is committed, at least one short
repeat is unavoidable; it occurs in column 1. At 47182 every column was last
attacked either 16 or 19 ms earlier.

The same three H timestamps admit a continuation without that local repeat
if the first row uses at most two heads. For a candidate with $k$ current
heads in this particular rested state, the minimum number of short repeated
future heads is

$$
c(a)=\max(0,k+2-4).
$$

This is a consequence of the candidate and the required future attacks. It
does not require deleting closely spaced H events or imposing a minimum LN
duration. The appropriate decision point is the earlier chord.

## An available alternative can still receive insufficient probability

In Good Luck B/A, a column-2 LN that began at 185086 ms ends in the H row at
187136 ms, alongside taps in columns 0 and 1. At 187147 ms the model taps
column 2. Column 3 was last attacked at 186095 ms and remains available.
The later re-press is therefore avoidable at that row.

This release occurs in a head row; it is not a release-only deadline scheduled
by the full-occupancy waiting law. Its release-to-head relation also differs
from the confirmed same-column attack criterion above. The existing combined
head/release close-gap diagnostic is retained here as a separate research
measurement.

## What frontier2 contributes at these states

A behavior-neutral observer reproduces both complete crossed charts and
captures the row context, candidate features and scores. On each fixed context,
compare the current consequence residual, its removal, and the released R1
consequence tensors evaluated on the same current inputs. The last comparison
isolates those tensors; it does not reproduce the original R1 policy.

| State and positive-cost family | Without frontier2 | Current frontier2 | Released frontier2, same inputs |
| --- | ---: | ---: | ---: |
| 47163: candidates forcing a short repeat in the next two H | 49.16% | 45.86% | 46.98% |
| 47182: any legal head repeats a recently used lane | 100% | 100% | 100% |
| 187147: current head has a head/release gap at most 20 ms | 6.02% | 5.19% | 5.10% |

At the first decision, the sampled three-tap row has probability 23.38%; a
two-tap alternative on columns 0/1 has probability 16.75%. The frontier path is
active but only weakly suppresses choices that commit an unavoidable repeat.
Greedy decoding on this fixed prefix would also select a positive-cost row:
the sampled unsafe row already exceeds every zero-cost candidate's probability.
Swapping back the released tensors does not solve that response under the
current inputs, so parameter drift in this module is not a sufficient account.

At the avoidable later re-press, tapping column 3 already has probability
94.08%, while the sampled column-2 tap has probability 4.65%. The model's
preferred action is sensible at that state, but its remaining tail is material.
These are different failures: insufficient foresight before a commitment and
residual probability on an avoidable later action. An optimistic minimum over
future actions does not guarantee that the future sampler chooses that minimum.

The next response representation should preserve this distinction. A finite
candidate evaluator can expose unavoidable short-attack counts over the planned
heads inside the relevant horizon. Its assumptions about intervening LN
releases must remain explicit. Any decoding policy that conditions on such
responses is distinct from the trained proposal likelihood and needs its own
full-generation checks. Sparse-piano and arrangement-control questions remain
separate; a universal anti-repeat objective would discard useful structure.

## Optional candidate-response correction

The planned rollout accepts `correct_short_attacks=True`, defaulting to false
for checkpoint reproduction. It draws the original row proposal first, then
evaluates that row against the other physically legal candidates. The evaluator
counts strict same-column attack pairs in the current row and minimizes the
additional count over all previewed H events before the current time plus
20 ms. A 16-state dynamic program tracks which columns have been used inside
that horizon. Each future H needs one hypothetical TAP; committed LN occupancy
can clear at its earliest possible native release clock. No actual future
actions or endpoints are inputs.

When the proposal already has minimum response cost, selection returns it
unchanged and consumes no correction randomness. Otherwise it restricts to
minimum-cost rows, then minimizes changes in head count, LN-start count, release
count and lane-action Hamming distance, in that order. Original proposal
probabilities select within the remaining family using a separate random
generator. The policy changes the generated distribution; the original model's
NLL does not evaluate this correction kernel.

The computation is optimistic about unknown releases and limited to the
supplied preview. A zero minimum therefore does not guarantee that the actual
future sampler will avoid short pairs. Positive minima are recorded as
unresolved decisions, and complete generated charts receive a separate strict
attack-pair diagnostic. This policy preserves H timestamps and imposes neither
a general onset gap nor a minimum LN duration. It is an optional research
intervention, not an established solution to musical arrangement or playability.

The bounded comparison at source
7ea2e82eebd9afbc97ec8db23cf361885fc8e0a1 checks 651 real charts and eleven fixed
generated cases. All 683341 source rows have zero optimistic response cost;
none of those source charts contains a strict short-attack pair. The policy
leaves ten already healthy generated outputs row-identical. In the remaining
Good Luck A/B case, one correction changes the three-tap row at 47163 to
TAP0/3. TAP2 at 47166 and TAP1 at 47182 preserve the original three H clocks
without the 19-ms same-column pair. All eleven outputs have zero strict pairs.

The full-chart comparison nevertheless fails its release-to-head regression
guard. The changed autoregressive suffix closes four holds at 193538, then
taps column 3 at 193542. Aggregate RH<=20 rises from one to two, although
head-count, LN-fraction, physical validity and startup checks pass. Eleven new
Lens pages cover the correction, fixed musical scope, late occupancy episodes
and this RH4 witness. Chords and independent LN releases remain present, but
neither the local correction nor their presence establishes a playable chart.
The policy remains optional and off by default. Its result is a local response
improvement with a failed whole-chart guard, not an overall model promotion.

The [playback-budget study](audio_playback_system.md) separates this unresolved
musical/physical interaction from measured end-to-end compute capacity.

## Evidence identities

All fitting, native comparison and score-audit code runs at clean product
7c316e6ea3d22aff1fd4761798f1b4aa41d47e89. Local owner:
artifacts/joint-audio/20260924-feasible-release-v1.

- Fitted native result: accf33033b475afdb808d7128ea6c2b614815092aa01c293dd2232e74a893235.
- Fitted Lens review: 8916408cbe5d430642961ab1943271e49e53c2701f8350c9c2dea3a2a994340d.
- Crossover result: 2b3b9315d384e5458f9a631384b9b830f3de0f328debd8ebec6d112144950284.
- Crossover Lens review: bbaa35677281605ef264da1ff616bdd0e0c2a5670cbf8a869ea2f1d0c74ed06a.
- Frontier score-audit freeze: 39ceff1aef2e040c142ef4bfc9a39f48f876cb0e462aa7cfc883a61ffa23524d.
- Frontier score-audit result: da94eadf75b24f04a294a9056972866c0f0a164a4f95118dec4a9297aa4fb104.

The crossover took 9.78 s and the score audit 3.43 s on one CPU thread. The
trained endpoint fails its composition guard and remains a research model.

The correction comparison's local owner is
artifacts/joint-audio/20260924-short-attack-response-v1. Corpus result:
5f9d45de29756e188b6b6568e0887060b1c1b9dd1e3ecb2d7e2409a0b73372c6;
native result: 1c5fcedf09f8c26d4e51015867d0eec7371e5c86718ef8c53c9d48c1a2cdb7ce;
Lens review: b708a90c48f08d58b9fbb299a9dc3c3eeeb525d93e99155da83b7cfe3c265c8a.
