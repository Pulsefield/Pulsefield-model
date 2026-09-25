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
