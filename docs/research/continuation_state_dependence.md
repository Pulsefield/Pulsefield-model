# State dependence in audio-conditioned continuation

Changing an unpublished continuation can change later LN organization, even
when audio, generated head times and the arrangement request remain fixed.
A diagnostic of two such changes finds both state dependence and substantial
sampling variation. It does not identify a single persistent tap-collapse
mechanism or assign responsibility to restored R1 weights.

The question follows the [buffered continuation study](unpublished_continuation_screen.md),
whose physical screens removed three strict short-attack pairs while two outputs
failed an LN-composition guard. In this diagnostic, all model weights and
decoding distributions remain unchanged. Only the starting continuation state
and subsequent release/row RNG states are crossed.

## Comparison and measurement

Write an ordinary continuation as $Y=F(S,U;A,H,z)$, where $S$ is a coherent
post-row state, $U$ contains the initial release and row RNG states, $A$ is
complete encoded audio, $H$ is the fixed generated head plan, and $z$ is the
downstream arrangement profile. The implementation is deterministic with these
inputs and unchanged parameters. $S$ includes exact replay and both row and
skeleton caches, not only a learned history vector.

For each case, $S_O$ and $S_N$ are the original and screened states immediately
after the first H following the last resampled publication window:

| Case, generated H0/downstream profile 2 | Anchor | Audio duration | Subsequent H |
| --- | ---: | ---: | ---: |
| Prom Queen, original seed 19 | 56907 ms | 136569 ms | 278 |
| Airborne, original seed 33 | 32420 ms | 271490 ms | 1934 |

Captures preserve the complete accepted prefix. Their future H queues and
encoded audio agree exactly, and both release-survival residuals are empty
at the post-H anchor. No future reference row or LN endpoint is supplied.
The two original trajectories are reproduced exactly. Continuing each captured
state with its own original RNG states also reproduces its saved full row hash.

Four continuations cross the two states with their two saved RNG streams.
A further 16 fresh stream seeds, 239251 through 239266, are paired across both
states. All suffixes use ordinary unscreened generation to isolate the existing
conditional law. The 72 resulting complete charts independently reparse and
preserve every fixed H time and their respective prefixes. They are diagnostic
outputs, not newly screened candidates for publication.

The primary measurement is the LN-head fraction among heads strictly after the
anchor. TAP and LN starts both count in the denominator; releases do not.
For each fresh seed, subtract the original-state fraction from the
screened-state fraction, then average the 16 differences. The reported
percentile interval uses 10000 paired bootstrap resamples, analysis seed 239267.
It quantifies RNG variation conditional on these two selected state pairs,
not uncertainty over songs or the choice of prefixes.

Pairing initial RNG states is related to
[common random numbers in simulation](https://doi.org/10.1287/mnsc.38.6.884).
State-dependent events change subsequent draw consumption, so this comparison
does not assume event-aligned randomness or guaranteed variance reduction.

## Fixed streams exaggerate one case's state effect

The four saved-stream cells measure the following suffix LN-head fractions:

| Case | $F(S_O,U_O)$ | $F(S_O,U_N)$ | $F(S_N,U_O)$ | $F(S_N,U_N)$ |
| --- | ---: | ---: | ---: | ---: |
| Prom Queen | .6344 | .6039 | .2171 | .2842 |
| Airborne | .4701 | .4369 | .4672 | .3374 |

Averaging the two possible intervention orders assigns Prom Queen's change
of -.3501 to a state term of -.3684 and a stream term of +.0183. For Airborne,
the total -.1327 divides into -.0512 for state and -.0815 for streams. These
terms share the state/stream interaction equally and sum algebraically to the
diagonal difference. They describe four fixed trajectories; they are not
unique module-responsibility percentages or population effects.

Fresh streams substantially change that assessment:

| Case | Original-state mean | Screened-state mean | Mean paired difference | Conditional bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Prom Queen | .5627 | .5221 | -.0406 | [-.1267, +.0510] |
| Airborne | .4977 | .4159 | -.0818 | [-.1300, -.0300] |

Prom Queen does not establish a directional mean state effect in this sample.
Airborne supplies evidence of a negative conditional state effect, but its
mean does not reach the diagnostic's declared -.10 material-effect threshold.
Neither case establishes practical equivalence either: that criterion required
the interval to lie inside [-.10, +.10] and the absolute mean below .05.
The two cases therefore remain unresolved under the predeclared decision rule.

Individual trajectories vary widely under either fixed state. Prom Queen's
original-state fractions range from .268 to .854, and its screened-state
fractions from .183 to .761. Airborne's corresponding ranges are .268 to .677
and .212 to .581. A large change between one original/screened pair is therefore
insufficient to diagnose persistent collapse. Conversely, random variation
does not explain away Airborne's measured conditional state difference.

## Actual organization differs across the paired continuations

The Lens inspection reads nine scopes, all 33 time pages and complete action
and LN-articulation tables. For each case, the selected fresh seed has the
largest absolute suffix-fraction difference. Inspection covers eight seconds
at the anchor and eight seconds beginning at the most different 64-H block.
These are deliberately contrasting examples, not representative quality samples.

Prom Queen's selected seed 239252 reverses the original apparent direction:
the screened-state suffix has .699 LN-head fraction, versus .310 from the
original state. At 124814–132814 ms, its overlapping short holds and independent
release subsets contrast with the other arm's initial broad tap chords. Both
later retain long holds over sparse H, and both preserve the 1-ms cross-column
H pair at 126455/126456 ms without a same-column attack violation.

Airborne's selected seed 239258 has lower LN use from the screened state over
the whole suffix, but higher use immediately after the anchor. In the first
eight seconds, the screened-state output mainly makes sequential LN handoffs;
the original-state output contains a long tap sequence followed by grouped
holds. Much later, at 223054–231054 ms, the relationship reverses sharply:
both arms have exactly 91 H and 115 heads, but the original-state arm makes
100 LN starts and the screened-state arm 10. The latter still retains a
610-ms hold spanning six intervening H. The former includes independent
release subsets and longer holds alongside its many short LNs.

These are changes in organization, not merely different counts of identical
objects. They establish neither musical superiority nor loss of all LN
capability. Short objects, including a 4-ms LN and a 20-ms paired LN group in
the original-state Airborne scope, remain separate articulation questions;
they are not automatically violations of the strict same-column attack rule.

An inspected unscreened exact-stream cross contains a confirmed bad pair:
Airborne taps columns 0/1 and releases column 2 at 252770 ms, while column 3
remains held. At 252778 ms it starts an LN in column 0, giving an 8-ms
TAP-to-LN-head repeat. Column 2 was physically free at the second H, although
using it would create a separate short release-to-head gap. The 72 diagnostic
suffixes contain two strict HH violations and 71 experimental RH<=20 pairs;
the second HH is an 11-ms pair in another fresh-stream output. These suffixes
were not screened, so these counts do not contradict the buffered policy's
earlier zero-violation result.

## What the state comparison does and does not isolate

The state intervention changes occupation, exact lane clocks, cumulative row
and note counts, row history and skeleton history together. Those are all
legitimate consequences of the already-generated prefix. It cannot identify
which individual input carries the conditional effect.

The current architecture has several distinct propagation paths:

- Row generation reads direct audio, H preview, exact replay and learned row
  history. A changed action changes later row inputs and candidate consequences.
- LN starts and closes change the minimal LN projection used by release
  timing. A changed R event also changes physical row history and the separate
  H/R skeleton history. This respects the LN-only feedback contract; no generic
  row-layout cache is passed into skeleton timing.
- Exact row queries include transformed cumulative physical-row and note
  counts in `joint_audio_continuation.state.exact_features`. These can retain
  differences after old rows leave a finite TCN cache. Their causal contribution
  has not been isolated here.

Even without cumulative features, a finite neural receptive field would not
bound the duration of an autoregressive perturbation: changed generated rows
become inputs to later rows. Cache length alone is therefore not a stability
guarantee. The observed effects also cannot be assigned specifically to
restored R1, the new audio encoder or release timing without a further
intervention on those paths.

Teacher-forced NLL measures predictions on supplied histories. It does not
measure how a resampled prefix changes the distribution of future organization,
nor enforce the strict HH invariant. A useful next architectural comparison
must separate the permitted state paths while retaining direct audio, H preview
and exact LN legality. Merely enlarging memory, penalizing the observed LN
fraction difference or adding another global profile does not follow from
this diagnostic.

## Provenance and limits

The comparison runs at clean source
`65ef95c092e7d3024d99ed60e37c2c22ee041df9`, with unchanged checkpoint
`abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef`.
It takes 199.22 seconds including reconstruction, exports and analysis on
Apple M5, one Torch CPU thread, Python 3.10.20 and Torch 2.11.0. Suffix generation
alone takes 182.57 seconds. These are offline diagnostic costs, not startup
latencies or the cost of serving all branches to a player.

The two state pairs were selected because they failed an earlier composition
guard. They share an artificial H0/downstream-profile-2 cross. No conclusion
extends automatically to other songs, automatic profiles or a retrained model.
No listening or player testing has been performed, and no new decoder or
checkpoint is adopted.

Local evidence owner: `artifacts/joint-audio/20260925-continuation-state-rng-v1`.
Native result SHA-256:
`b0302e1769243c2212993ebe898f484a664c9500e7d455b29dc0c0657d900074`.
Completed Lens review SHA-256:
`bf0ee0cfcf0db05fb5d756e82f3fb69f252a8c59556daf9f7314bcd8c479d9b5`.
