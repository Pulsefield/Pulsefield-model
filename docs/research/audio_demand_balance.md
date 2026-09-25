# Audio-conditioned head demand and generation feedback

Source-conditional likelihood and generated-trajectory quality diverge in the
[paired continuation study](typed_audio_continuation.md#paired-continuation-result).
A small demand model separates a nominal activity prediction from the local
event model's response to its own generated history. This research option is
implemented, with learned calibration and native usefulness still to evaluate.

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
