# Audio-conditioned head demand and generation feedback

Source-conditional likelihood and generated-trajectory quality diverge in the
[paired continuation study](typed_audio_continuation.md#paired-continuation-result).
A small demand model separates a nominal activity prediction from the local
event model's response to its own generated history. A bounded fit and native
comparison improve some control behavior while exposing remaining calibration
limits; this is a research candidate, not a validated playable release.

## Information and supervision

`AudioDemand` reads the retained encoder's fine and full-song coarse audio
features, pooled into 500-ms cells, and the actual optional controls. Two
96-wide hidden layers predict log head objects per second. A learned nonnegative
star slope makes nominal demand monotone in known difficulty for fixed audio,
LN fraction and style. Other conditions retain nonlinear effects. This is an
inductive bias on the mean, not a restriction on sampled counts or pattern form.

The training target is each observed chart's head count in an elapsed-time cell;
alternative charts of the same recording remain separate examples. A Poisson
count loss estimates mean demand and is not a playability score. Runtime input
contains neither those counts nor source future LN endpoints. Full audio is
available in training and inference. The core encoder and event/row model are
frozen for the initial bounded demand fit because unconstrained continued
training recently degraded native behavior; freezing is not a final system
requirement.

The optional `per-field-scope-v1` demand input assigns a separate start/end clock
pair to difficulty, LN fraction and each known style attribute. Partial overrides
replace only the supplied attributes and their owning spans. On expiry, the
earlier values and original extents resume. This distinguishes, for example, a
whole-song difficulty request with a short style override from a short difficulty
request under the same style value. The shared-clock encoding collapses those
inputs. Neither encoding creates an exact count budget or resets generated state.
The checkpoint declares its encoding; existing demand weights use
`shared-scope-v1`. The frozen core's control representation is unchanged.

This input can be paired with the existing full-prefix source strain proxy over
the difficulty control's own range. Such labels describe observed arrangements;
they are not online strain inputs or official fragment ratings. Source LN
fractions and human style judgments retain their respective scopes. Matching
the range of a label to its condition is a supervision contract; it does not
establish that the resulting demand will calibrate the complete generator.

## Feedback and decision ownership

Let the nominal head rate be `nu(t)`, and let generated skeleton event i introduce
`h_i` heads at `t_i`. The desired and actual discounted counts use a four-second
time constant:

$$
E(t)=\int_0^t e^{-(t-s)/\tau}\nu(s)\,ds,
\qquad H(t)=\sum_{t_i<t}h_i e^{-(t-t_i)/\tau}.
$$

`DemandFeedback` adds `clip(2*log((E+4)/(H+4)), -2, 2)` to head-event log odds.
Conditional head marks receive that shift for each additional head beyond the
first. Release-only/no-event odds remain unchanged. All conditionals retain
their exact physical support, existing empirical recovery preferences and
optional LN-amount feedback. This local reweighting is not an exact global
count-conditioned law, nor a proof that realized head rate equals its prediction.

The actual ledger contains only skeleton head times/counts. It neither reads
R1's generic column history nor delays mandatory releases to reduce a counter.
R1 continues to read direct audio, typed preview, its own history and row
consequences. Star rating also depends on geometry and LN organization, so
accurate head demand cannot alone establish correct difficulty.

The nominal curve is constant within 500-ms audio cells, split at every declared
control boundary. Audio features use the fixed cell; control queries use each
piece's left endpoint. Consequently, adding a future partial-cell override
preserves the rate and integral before its start. Actual counts survive control
changes and roll back with unpublished planner state. Their exponential decay
depends on elapsed time, never on the number of release rows. There is no
deadline repayment term, exact per-cell quota or reset of physical history at a
control boundary. Generated head and release times retain native-ms support.

## Analogy and limits

The closest primitive is the count feedback in
[Isham and Westcott's self-correcting point process](https://www.sciencedirect.com/science/article/pii/0304414979900085).
Their fixed-rate analysis does not provide guarantees for this audio-dependent,
marked, leaky and capped policy. The transferable idea is negative feedback on
accumulated activity. The comparison with
[Neural Hawkes](https://arxiv.org/abs/1612.09328) concerns history-dependent
excitation and inhibition; it does not establish that a learned history alone
will regulate desired gameplay demand.

Native evaluation must retain separate control cells and effective ranges,
inspect both under-generation and excessive density, and preserve short bursts,
moving patterns, chordjack, dump and LN layers. A demand curve that predicts
conditional averages poorly can flatten musical variation or amplify a rest.
Even a well-fitted curve can leave bad release choices and geometry. Actual
generation, Lens inspection and later player assessment decide usefulness.

## Bounded learning and native results

The retained 3,947,227-parameter core supplies frozen full-audio features for
2,665 recordings. Caching takes 187 seconds on MPS and produces 0.777 GiB of
500-ms features. The demand network has 36,866 parameters, including its actual
60-wide control input. A 2,000-update CPU fit takes 21 seconds, sampling 16,000
intervals from 5,294 distinct TRAIN charts with the existing population/human-
annotation mixture. On 24 validation midpoint intervals inside 2–6 stars, mean
absolute head-count error falls from 31.26 to 13.77 heads. The core is unchanged.
This count prediction improvement is not a chart-quality measurement.

On the initial three-song panel, adding demand improves difficulty-3/low-LN
star MAE from 0.463 to 0.160, but difficulty-3/high-LN MAE rises from 0.291 to
0.533. Difficulty-five/high-LN short tails increase from 2.01% to 2.34%. Four
Zenithfall seeds narrow the low-LN star span from 1.511 to 0.281 and the high-LN
span from 0.982 to 0.482, but their centers are too high: the new outputs span
3.392–3.673 and 3.814–4.296 for the respective difficulty-three requests. These
seed results precede the stronger recovery profile below. More consistent
activity does not establish accurate difficulty or good releases.

The combined candidate adds the fixed
[60/50/50-ms recovery profile](typed_audio_continuation.md#session-recovery-profile).
It retains amount feedback, empirical recovery preferences, head pressure four
and selective star guidance two. Eight development audio files each receive
four static requests and one fixed 105000–137000 ms override; all 40 complete.
Each table cell averages only eight full-song outputs at the same request.
The comparison changes both demand and recovery relative to the retained recipe;
it does not isolate their effects on the five added songs.

| Request: stars, LN fraction | Retained recipe star MAE | Combined candidate star MAE | Candidate LN-fraction MAE |
| --- | ---: | ---: | ---: |
| 3, 0.2 | 0.555 | 0.200 | 0.0348 |
| 3, 0.7 | 0.304 | 0.357 | 0.0301 |
| 5, 0.2 | 0.311 | 0.238 | 0.0364 |
| 5, 0.7 | 0.398 | 0.501 | 0.0651 |

All combined static outputs stay within 2–6 stars. Individual deviations remain:
Zenithfall requested at 3/high-LN reaches 3.814, while FoolMoon requested at
5/high-LN reaches 3.961. Scoped calibration is weaker than these full-song
averages. The eight difficulty-five override proxies span 3.327–4.502; restored
difficulty-three proxies span 2.747–3.484 apart from Take's short 7.237-second
tail, which reads 1.628. These are the full-prefix scoped strain proxy, not
official local star ratings. Each before/override/after range remains separate.

On the initial three-song difficulty-five/high-LN cell, the stronger recovery
profile reduces LN lifetimes at most 60 ms from 13.21% to 4.01%; only 0.181% land
exactly at 50 ms. The median of the three chart-level LN medians rises from 124
to 139 ms. Release-to-next-head intervals at most 60 ms fall from 9.09% to 2.32%.
Absence of at-most-40-ms LNs follows from support and is not credited as learning.

Three combined-candidate Lens contexts, all six pages and complete tables,
retain varied articulation: Zenithfall 72347–76348 ms has grouped starts,
staggered tails and 149–259-ms layers; its shortest new LN is 54 ms. Take
127738–131739 ms is locally pure TAP with varied pairs/triples and separated
quads, retaining an 8-ms cross-column stagger. Hysteric's 69394–73395 ms peak
has paired handoffs, TAP interleaving, 184–356-ms layers and a 69-ms minimum LN.
No listening, player test or formal semantic-style assessment was performed.

The candidate therefore provides a more usable execution envelope and some
control gains, with scoped difficulty and style adherence still unresolved.
Nominal head demand is not an inverse model of the complete generator's
difficulty response. Whole-chart difficulty labels and the shared scope-clock
input also remain imperfect descriptions of independently scoped requests.

## Independent scope targets and native qualification

A 63,362-parameter demand fit uses per-attribute scope clocks and the existing
full-prefix strain proxy over each source difficulty range. The retained core
and all sampling coefficients stay fixed. The fit takes 89 seconds on one CPU
thread and uses exactly the same 16,000 TRAIN draw identities as the earlier
2,000-update demand fit. This comparison changes encoding and supervision
together; it does not attribute their effects separately.

All 40 cases on the same eight development audios complete. Static results are
stratified by the complete request, with eight charts per cell:

| Request: stars, LN fraction | Shared-scope demand star MAE | Independent-scope demand star MAE | New LN-fraction MAE |
| --- | ---: | ---: | ---: |
| 3, 0.2 | 0.200 | 0.141 | 0.0363 |
| 3, 0.7 | 0.357 | 0.316 | 0.0290 |
| 5, 0.2 | 0.238 | 0.210 | 0.0334 |
| 5, 0.7 | 0.501 | 0.533 | 0.0661 |

For the eight 105000–137000-ms difficulty-five/high-LN overrides, median absolute
proxy error falls from 1.159 to 0.639. Individual new proxies remain disparate:
Zenithfall 4.131, Hysteric 4.519, Take 4.684, FoolMoon 3.513, Goodbye 4.534,
Revenge 4.267, As It Was 3.590 and YomiYori 4.455. These are scoped strain proxies,
not official local stars. Before and restored ranges remain separate; Take's
restored range is only 7.237 seconds. Whole-chart high-LN requests also remain
weak on FoolMoon and As It Was, yielding 3.930 and 4.118 for five requested stars.

Lens inspection of Hysteric's entry, dense override and restoration covers six
pages and complete action/articulation tables. Entry includes paired releases
over an 829-ms layer and a later 1,444-ms hold. The peak retains independent
tails and 420/666/463-ms layers, alongside a repeated 65/54/51/75-ms LN sequence
that remains a playability concern. Three holds crossing the override's end
release separately at 137037, 137097 and 137265 ms; subsequent TAP motion and a
1,210-ms layer retain continuity without a boundary reset.

Eight additional requests cover difficulty two and six at both LN fractions on
Hysteric and Take. Their absolute whole-star deviations are at most 0.528, but
the Take six-star/low-LN peak exposes a grouping failure: 124 heads in 39 rows
over four seconds, including five consecutive all-TAP quads 101–123 ms apart.
A pattern-specific retrieval over 6,923 admitted ranked TRAIN charts in 2–6
stars finds only one run of at least five consecutive TAP quads with every gap
at most 125 ms. That source, Hold On Tight [Tetris], has five chords 125 ms apart
at 107715–108215 ms and a whole-chart rating of 5.1007. This establishes
rarity in that corpus, not a universal BAD cutoff or a ban on chordjack.

### Count feedback does not specify grouping

In the generated Take peak, predicted demand is 25.3–34.5 heads/second and the
feedback shift remains positive, 0.088–0.472. Applying it to every extra head
favors a four-head mark over a one-head mark by `exp(3*shift)`. The clock and
cardinality decisions therefore both respond to the same accumulated deficit.
R1 cannot reduce a four-key count that the skeleton has already committed.

A three-case routing probe removes the mark contribution while retaining clock
feedback and every learned weight. Take moves from 6.417 to 6.008 stars, its
original peak drops to 84 heads in 36 rows, and Hysteric's override proxy remains
4.571 with LN fraction 0.744. However, inspecting the new Take peak reveals eight
consecutive quads 126–153 ms apart. Four pages and complete tables cover both
the original and new peaks. The numeric improvements do not establish that the
grouping problem is fixed, so clock-only routing is not the packaged recipe.

A nominal head count cannot distinguish many small rows from fewer large chords.
Their decision ownership must remain distinct: the skeleton proposes timing;
R1 chooses chord size, TAP/LN allocation, release identity and column layout,
conditioned directly on audio, scoped controls, timing and gameplay state. The
typed prototype's upstream count contract removes those choices from R1. That
restriction is an interface problem, not a reason to move more composition
control into the skeleton. An onset-rate estimate can describe head-bearing
rows, while composition predictors and responses belong inside R1.

Restoring the complete-row choice also changes the training task. In a layout
loss conditioned on an upstream count/release group, adding the same score to
every member of that group cancels on normalization. Such a loss cannot
calibrate relative mass across groups. A restored R1 must train those choices,
and candidate frontier effects must survive a final normalization across full
rows. Merely widening the inference mask does not supply the missing calibration.
This repair must preserve deliberate chordjack and dump; a fixed ban on repeated
chords would discard valid target arrangements.

### Bounded style response

One Hysteric comparison holds difficulty four and LN fraction 0.2 while requesting
unspecified, absent or prominent Jack organization in 105000–137000 ms. The
prominent output has more persistent shared-column chord figures near the end
than the absent output. The latter still contains a local repeated-chord passage.
Their respective scope LN fractions are 0.157/0.259/0.162, exposing coupling to
an independently requested quantity.

Complete head timelines and LN endpoints were read for both specified requests,
with eight time-proportional pages covering matched 126000–130000 and
133000–137000-ms contexts. Human High-confidence examples supply a contrast:
Prom Queen [Lin's Insane], 75838–78338 ms, is prominent Jack with shared columns
through changing chords; Catalinesie [Catalyst of Amnesia], 113311–117485 ms, is
Jack absent despite an isolated repeated pair within flowing motion. Both source
review contexts and all five pages were inspected. The generated comparison
supports a directional response, not reliable whole-scope salience control or
generalization to the other four concepts. No listening or player test was done.

## Loading one complete system recipe

`system.load_system` loads a `typed-audio-system-v1` bundle containing the core,
demand model and all five sampling-policy records. It uses data-only Torch
loading, checks the audio/control dimensions and rejects missing or unconsumed
sampling fields. A bare core checkpoint cannot silently become this candidate.
Training audio caches are not required at inference.

Given canonical full-song Mel and its duration, a caller can use the stateful
interface below. The bundle owns control-vocabulary ordering and the fixed
recovery profile. Publication extent belongs to the playback buffer; future
control updates must start after already-published coverage.

```python
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.system import load_system

system = load_system("candidate.pt", device="cpu")
controls = ControlSchedule((ControlSpan(0, duration_ms + 1, stars=3, ln_fraction=.2),))
session = system.session(mel, duration_ms, controls, seed=251702)
session.publish_to(8000, minimum_rows=30)
start = session.coverage + 8000
session.update_controls(ControlSpan(start, start + 32000, stars=5, ln_fraction=.7))
```

The first rows are generated from BOS without a source-chart seed. The minimum
row count above is a publication-buffer request, not privileged context for the
model. `TypedAudioSystem.generate` provides the same factors for offline export.
A serialization check compares the loaded and direct components through native
generation, a future control change, restoration and terminal LN closure.

A shared-scope demand bundle contains 3,984,093 parameters in 16.05 MB, with
both models and the tested 60/50/50-ms recipe. Loading the trained bundle
reproduces the full 736-row Take comparison sequence exactly on the same cached
Mel, CPU thread setting and seed. A fresh-process test recomputes YomiYori's
498989-ms waveform/Mel and publishes 8000 ms, 67 rows and two open LNs in 2.380
seconds from child entry, or 2.519 seconds including process roundtrip. OS caches
are not flushed; client rendering and player reading are excluded. Across the
40-case panel, the largest observed cached-Mel publication-window service is
0.603 seconds for an 8-second step. These are measured service observations,
not worst-case guarantees or player validation. A local three-chart playtest
archive preserves the generated note objects and includes their audio.

An independent-scope demand bundle contains 4,010,589 parameters in 16.16 MB.
It retains clock-and-mark feedback; the exploratory clock-only routing is not
included. Loading this trained bundle reproduces its 735-row Take reference
exactly. A fresh-process YomiYori observation publishes 8000 ms, 73 rows and one
open LN in 2.707 seconds from child entry, or 2.862 seconds including process
roundtrip, with the same cache/client exclusions above. Its five-chart Hysteric
archive adds the paired Jack requests to the static and difficulty-switching
examples. The documented dense-chord failure, high-LN calibration error and
limited style evidence remain properties of this candidate.
