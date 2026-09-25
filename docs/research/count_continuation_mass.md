# Current row validity and future continuation probability

The [count/layout experiment](count_layout_materializer_evaluation.md) increases
short same-column attacks despite improving mean descriptor response. Fixed-prefix
probability probes distinguish two causes: choosing a row with no clean nearby
continuation, and choosing a row whose clean continuations exist but receive
little probability from the subsequent sampler. Both occur in the tested model.

This diagnostic conditions only the first row on current validity. It does not
compare a decoder that applies a current-validity mask at every subsequent row
with a lookahead decoder. It therefore motivates that comparison without proving
that a more complex planner is required.

## What is conditioned and measured

The flat and count/layout checkpoints are the two fixed endpoints of the matched
materializer experiment. Each model receives the same literal generated prefix,
the same full canonical Mel, profile 1 and fixed H plan. Each retains its own
audio encoding and history caches. The prefixes come from factor-arm trajectories;
the flat queries are counterfactual, not its own native-history distribution.

At query time $t$, let $C_0$ mean that the first complete row has neither an HH
pair nor an RH pair against the prefix. HH means successive same-column attacks
strictly below 20 ms; TAP and LN heads count, exactly 20 ms does not. RH is the
separate experimental screen for the first attack within at most 20 ms after a
same-column release. Define

$$
Z_0=P(C_0\mid\text{prefix}),\qquad
w_m=P(m(Y_t)=m\mid C_0,\text{prefix}).
$$

Both quantities are obtained from the complete 256-way first-row distribution.
For each count triple $m$ with nonzero clean mass, 64 independent first layouts
are drawn from that group's clean conditional. The original R/row sampler then
runs through the specified horizon, without correction, masking or retries.
Actual release times and release subsets are sampled; hypothetical early tails
are not substituted. The estimated clean continuation mass is

$$
Z_{\mathrm{future}}=Z_0\sum_m w_m\,
P(C_{\mathrm{suffix}}\mid C_0,m,\text{prefix}).
$$

HH-only and HH+RH suffix outcomes are both reported. Even the HH-only suffix
metric uses the same first-row condition $C_0$. Clopper–Pearson intervals use
Bonferroni correction across count groups within each query; weighting their
bounds yields a conservative 95% interval for that query's mixture. There is no
across-query population claim. Zero successes in 64 trials do not prove zero
probability.

## Conditional results

All eight queries have $Z_0$ within $2\times10^{-7}$ of one. Current-only
conditioning at the first row therefore changes almost no probability mass.
The following values concern HH+RH through the horizon:

| Prefix and query time | Horizon | Flat mass, 95% interval | Count/layout mass, 95% interval |
| --- | ---: | ---: | ---: |
| Hysteric, no entering hold, 248930 ms | 20 ms | .993 [.894, .999] | .149 [.078, .290] |
| Hysteric, earlier decision, 248824 ms | 120 ms | .949 [.816, .993] | .041 [.006, .167] |
| Hysteric, LN3 entering, 248930 ms | 20 ms | .771 [.648, .839] | .004 [.002, .102] |
| As It Was, cross-column split, 148234 ms | 20 ms | 1.000 [.906, 1.000] | .997 [.902, .998] |

The two 248930-ms queries include H at 248935 ms. The earlier Hysteric query
includes H at 248930 and 248935 ms. As It Was includes H at 148239 ms, also a
5-ms separation. Its healthy result shows that close H spacing alone does not
explain failure.

The predefined diagnostic signal requires $Z_0\geq.95$ and an upper mixture
bound at most $.90Z_0$ in a failure context. All three factor-arm failure queries
meet it; the flat model also meets it in the entering-LN context. This is bounded
evidence of future failure mass, not a test of whole-song generation quality.

## Possible futures and likely futures differ

Without an entering hold, the factor model assigns .634 first-row mass to three
TAPs and .295 to two TAPs. Three TAPs leave a clean one-head continuation at the
next H, but only 2 of 64 sampled continuations in that group are clean. The
two-TAP group succeeds 25 of 64 times. A saved witness selects TAP1/2/3 at
248930, then TAP0/2/3 at 248935; a successful sample instead uses TAP0 alone at
the latter time. The first row is identical in those witnesses.

Only .0387 first-row mass in this no-hold query is mechanically incapable of a
clean next H. The much larger estimated failure mass therefore cannot be
explained solely by already selecting an unavoidable dead end. Most mass is on
prefixes with possible clean continuations that the unmodified sampler often
does not choose. Flat assigns .656 to two TAPs and .316 to three TAPs, with
64/64 and 63/64 clean continuations respectively. The difference includes
subsequent count and layout response, not merely the current width distribution.

The 120-ms query also fails frequently after an initial pure TAP. Factor assigns
.565 mass to one TAP and .399 to two TAPs, with only 1/64 and 5/64 clean
continuations. Thus choosing an LN at the earlier decision is not necessary for
this failure. Altering only early LN starts or releases would leave the strong
later count-response problem unresolved.

## LN occupation and release timing change feasibility

In the entering-LN query, column 3 has been held since 248824 ms. Under the RH
screen, neither releasing it in the current row nor at a later R makes it usable
at 248935: the release-to-attack gap would be at most 5 ms. Three new attacks in
the other columns therefore leave no clean next head. Summing exact first-row
probability over such candidates gives .6188 for factor and .1283 for flat.
This is a proof of impossibility for these one-future-H windows, independent
of the Monte Carlo sample count. It is not a general claim about horizons beyond
20 ms, where an actual early release can matter.

HH alone is different. Flat's group with three TAPs plus the current LN3 release
has 58/64 HH-clean continuations but 0/64 HH+RH-clean continuations. The first
saved continuation attacks column 3 five milliseconds after its tail. A release
can solve the head-to-head condition while violating the experimental RH
preference. Combined survival must not be described as confirmed HH safety.

For the entire entering-LN query, HH-only mass is .942 for flat and .048 for
factor, versus joint mass .771 and .004. The other two factor Hysteric queries
have HH-only masses .149 and .043; their main deterioration is not an RH-only
effect. These results retain the distinction between exact occupation, recent
attack clocks, release clocks and the model's probabilities over future actions.

## Implication and limits

A useful next comparison is sequential current-row HH/RH conditioning versus
conditioning that also preserves a short feasible continuation. The former may
repair many possible-but-unlikely futures; the latter must avoid committing
rows whose next clean support is empty. Neither policy automatically resolves
an older LN that needed to release before the current horizon. Actual release
choices and publication coverage remain part of the system question.

This distinction follows the conditional-inference idea of weighting a prefix
by its probability of a valid future, rather than only whether a future exists.
[Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) studies that distinction
for constrained language generation. Its results do not establish beatmap
quality, and preserving the base model's constrained distribution is not itself
a playability metric.

The probe covers four selected factor histories, two checkpoints and finite
horizons. It cannot attribute a population percentage of failure to restored R1,
establish a style distribution, choose a trained replacement or certify a player
experience. The previous whole-chart Lens review remains the quality evidence.
No new full chart, listening test or player evaluation is produced here.

## Reproduction

Source is `151f9c265a9c5eaaa43bc6736f41cdc626da5554`; model behavior remains at
`4567d87732320f27c01b40b15f74846865219934`. Checkpoint, bank, Mel and panel
identities are in the [materializer report](count_layout_materializer_evaluation.md#reproduction-identities).
The four prefix files are pinned by SHA in the execution freeze. Exact replay,
finite cached/dense history agreement, native first-query probability parity
and forced-row cache parity are checked before sampling. Every suffix replays
and preserves all H through its horizon. Parameter fingerprints remain unchanged.

There are 120 nonempty clean count groups across eight queries, yielding 7,680
sampled suffixes. The successful driver takes 51.382 seconds on one CPU thread
of an Apple M5 with 24 GiB RAM, Python 3.10.20 and Torch 2.11.0. Seeds derive
from base 253101 and stable query/arm/mark/repetition/stream identifiers. Separate
streams sample the first layout and future R/rows. No model fitting occurs.

The local owner is `artifacts/joint-audio/20260925-count-continuation-mass-v3`.
Two earlier attempts terminate during setup with no sampled queries: model
construction under inference mode conflicts with cache parameter-version checks,
and unconverted float row clocks violate the fixed-plan integer interface.
Both are retained separately; neither is included in probability estimates.

| Record | SHA-256 |
| --- | --- |
| Probe source | `0b9c4dbf100599171036451548db703edfaea47ace916326ebf77d63e033c175` |
| Execution freeze | `5059e03991736516301ac0f5de579a23d03965518c14f7bdb1f4331de21a19bf` |
| Complete result | `77fee8403e7425b8c53e6081ebcbac81b6a8d8e77027a0c58667dd015e4b9075` |
