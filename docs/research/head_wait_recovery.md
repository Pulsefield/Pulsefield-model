# Head waiting-time suppression and recovery

The first [planned head/release model](planned_audio_continuation.md) learns
some repeated rhythmic spacing but can leave long regions of music without
heads. Mechanical completion and improved source likelihood do not establish
usable generation. This investigation locates a failure in the separated head
process and proposes a bounded historical contribution to its event hazard.

## Scope and observed generation

Source e04b3490e60b4f7a7550f9abde20e6f551382dcc implements the 4,245,188-parameter
model. A 1200-update MPS fit uses 615 TRAIN arrangements from 240 groups,
36 VAL arrangements and the frozen 4800-interval exposure plan. Checkpoint
SHA-256 is
20cea8e5d00e554c93141af7403a47e7108a3ee8fbd2c2dd57b8c9c415aa6205.
The fit takes 991.64 seconds. Source-conditioned population NLL/s changes
61.65 to 40.28, and BOS NLL/s changes 41.33 to 27.23.

Nine source-free native outputs all finish, close their holds and reparse
successfully. None triggers the all-held release deadline or the descriptor's
same-lane head/head and release/next-head counters at 20 ms or less. Yet four
outputs end their head stream well before later positive human-reference episodes:

| Audio and seed | Generated H rows | Last head | Audio end |
| --- | ---: | ---: | ---: |
| Who?, 19 | 213 | 56.015 s | 116.368 s |
| Death Piano, 19 | 412 | 91.236 s | 177.372 s |
| Good Luck, Babe!, 17 | 441 | 125.935 s | 218.448 s |
| Good Luck, Babe!, 19 | 213 | 58.555 s | 218.448 s |

Lens inspection covers all 26 generated time-view pages in the fixed nine
contexts, all action/articulation pages, and 14 byte-identical human reference
pages already inspected with the preflight. Five generated contexts are wholly
empty. Airborne has no new heads and only the tail of an entering 5536 ms LN.
This includes an internal gap in an output that later resumes, so last-head
position alone does not capture the failure.

There is also partial learning: Who seed 17 exhibits repeated roughly 234 ms
and 117 ms spacing, and Prom Queen exhibits approximately 208/416 ms placements
with more chords than its 32-update output. These observations do not establish
Tech/Jack coverage, audio alignment or player approval. Human arrangements are
comparisons, not unique valid targets.

Empty windows were retained by extending only generated visualization ranges
to known full-audio coverage, recording the original parsed note ranges.
No note or human judgment was changed. Dropping windows outside the last
generated note would hide the failure.

## The sampler follows the learned waiting law

Until the next H, the committed head history stays fixed while the query clock
and audio advance. With native-ms head logits $z_u$, survival is

$$
P(T_{\mathrm{next}}>t\mid A,H)
=\exp\left(-\sum_{u=t_{\mathrm{last}}+1}^{t}
\operatorname{softplus}(z_u)\right).
$$

The sampler draws one unit-exponential threshold and waits for the cumulative
hazard mass to cross it. Its residual threshold is preserved across query
chunks. Replaying the head model alone reproduces every saved head time in
the four selected failures. R1 row actions, LN occupancy and consequence scores
are absent from this head path.

| Case | Remaining audio | Total future hazard mass | Drawn threshold | Probability of no further H |
| --- | ---: | ---: | ---: | ---: |
| Who?, 19 | 60.353 s | 3.36936 | 4.56417 | 3.44% |
| Death Piano, 19 | 86.136 s | 2.89453 | 3.75976 | 5.53% |
| Good Luck, Babe!, 17 | 92.513 s | 1.85614 | 4.07153 | 15.63% |
| Good Luck, Babe!, 19 | 159.893 s | 2.09168 | 4.56417 | 12.35% |

The independently summed mass agrees with the consumed threshold to less than
$4\times10^{-14}$. Thus these silent tails are predictions of the head model,
not a lost RNG draw, row legality blockage or resource deadline. The conditional
probabilities in the table are not population failure-rate estimates; cases
were selected after observing the failure.

The original stronger hypothesis of almost zero mass after a two-second wait
is not supported: the remaining mass is 0.736–1.862, above the proposed 0.01
criterion in every case. This is strong finite-horizon suppression, not proof
of an infinite-horizon absorbing state.

## Recency is an active suppressing input

A controlled query keeps current full audio, current time and the entire
head-interval sequence unchanged. It translates the legal head history so its
final head is 100 ms before the query. The head encoder reads only intervals,
so its state is identical; the elapsed-since-head input changes. No altered
prefix is trained against the original suffix.

At 30 seconds after the original last head, this intervention increases the
next-ms hazard by 632, 133, 500 and 617 times respectively for the four table
rows. Absolute probabilities remain conditional query outputs, not successful
generated charts. The experiment locates a recency-dependent suppression path
with fixed audio; feature-vector norms alone would not establish that.

The actual supervision makes long head age a strong end-of-chart cue. In the
frozen 1200-update plan, 88.21% of clocks with a prior head more than two seconds
ago occur after the reference's final head. The fractions are 94.53% after five
seconds and 94.48% after ten seconds. Under uniform group, arrangement and chart
time, the corresponding population fractions are 87.17%, 90.38% and 85.17%.
These counts exclude BOS before the first H.

None of the 615 TRAIN charts ends its head stream before 60% of its paired
audio. The reference arrangements for the four failures contain 60, 22, 49 and
65 H rows in the five seconds after the generated final head. Therefore, the
readout is not explained simply by a target corpus dominated by half-song
charts. Post-final target time also does not imply silence in the waveform.

The combined evidence supports a narrower mechanism: source histories associate
large head age mainly with no further heads; a sampled missed passage creates
that same clock during ongoing music; the unconstrained historical path can
strongly suppress recovery from later audio. This does not quantify the cause
of failures in the earlier flat joint model or establish that R1 row generation
has no independent quality problems.

## Candidate recovery invariant

Retain the same skeleton and row dependency contract, with an audio-only head
base and a bounded historical residual:

$$
z_H(A,H,t)=b_\theta(A,t)
+B\,g(\Delta)\tanh r_\theta(A,H,\Delta,t),
\qquad
g(\Delta)=\exp(-\Delta/\tau).
$$

At BOS the history gate is zero. Then
$|z_H-b_\theta|\le B\exp(-\Delta/\tau)$, so an old head history cannot indefinitely
veto subsequent audio evidence. Both likelihood and sampling must use this
same law. It does not force a head, impose a minimum interval or restrict
events to detected beats. The audio base can still predict silence.

The prototype exposes this law through its bounded_head option, using a linear
audio base with 2250 parameters, a default bound of 4 and a 1000 ms decay time.
The old timing MLP becomes the historical residual with a zero initial output
bias. Every other shared initial parameter draw is unchanged. This is an
architecture hypothesis, not a completed remedy. Risks include
forgetting useful density/phase choices across rests and an audio base that
itself fails to distinguish later activity. Evaluation must retain genuine
rests, one-cue repeated figures, irregular placement and LN articulation,
alongside recovery at the located failures. NLL and larger generated note
counts cannot establish improvement by themselves.

## Evidence identities

Local owner:
artifacts/joint-audio/20260924-planned-head-release-v1.
Main native result SHA-256:
7f9ee93823ead44e1ce5a9d171d67a12fb236d35e6f074547229a7cdbcc4fab8.
Lens manifest:
ef71d998ec207cc91ef195fed15fa97fa94cb8ab688ec55d7aea80d2cfa4b6a3.
Head-survival audit result:
dbc2e3143a5e3f48da09b031338e8e3a83313314cc0b87a15708691e0fb6266b.
Target-clock audit:
d2fc84af6bdf9b605b468517049059358812264af4e78bdf1b1b47e92d3f5c55.
These are exploratory, selected-scope results; no listening or player trial is
included and this endpoint has not been promoted as playable.
