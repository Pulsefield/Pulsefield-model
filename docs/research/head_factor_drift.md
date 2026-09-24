# Head-count drift across continued models

Further fitting changes generated head counts through a mixture of audio-base,
learned residual and sampled-history effects. A fixed-history comparison of
three saved models reproduces all 48 original H sequences, but finds no single
component meeting the predefined dominance criterion across the eight audios.
History feedback is substantial in several density increases; other cases have
large base changes or opposing contributions. This locates useful distinctions
without identifying a universal cause or a better playable endpoint.

## Compared processes

The models are the original shared-profile checkpoint and the shared/routed
continuations from the [density-routing comparison](profile_density_routing_evaluation.md).
The eight-audio panel receives explicit profiles 1 and 2 with the original
paired seeds. These profiles request different joint density, chord-width and
LN-fraction combinations. They are not single-coordinate interventions.

Each model's head logit has the implemented form

$$
\ell_\theta(t\mid h)=b_\theta(A,z,t)
+r_\theta(A,z,h_{<t},t),
\qquad
|r_\theta|\le 4\exp(-\Delta_H/1000).
$$

At BOS, the residual is zero. The base reads full encoded audio and the profile.
The residual additionally reads encoded H intervals, head age and the absolute
clock. It also has its own audio path; attributing a change to this residual is
not the same as attributing it to the head-history encoder alone. R/row content
and LN state do not enter the pilot H process at runtime.

The diagnostic exchanges the additive base output between the original and
each continuation, holding the residual output and H history fixed. Each source
computes its own audio and history encodings; no hidden representation or weight
is transplanted. Queries use both endpoints' saved H histories. Only heads
strictly before the native query time enter history, including when multiple
heads share a 10-ms scoring bin. Canonical bin anchors are retained.

This distinction between external conditions, history modulation and the
counting process has analogues in
[Neural Hawkes](https://arxiv.org/abs/1612.09328) and
[time-rescaling analysis](https://www.stat.cmu.edu/~kass/papers/rescaling.pdf).
The computation here uses the exact discrete Bernoulli probabilities.
[Discrete-time rescaling](https://pubmed.ncbi.nlm.nih.gov/20608868/) explains why
continuous-time goodness-of-fit assumptions require care with finite bins.
No time-rescaling significance test or Hawkes stability theorem is used here.

## An accounting identity, with a limited causal meaning

For initial model 0 and continued model 1, define

$$
C_{ab}(h)=\sum_{t=0}^{T}
\sigma\bigl(b_a(A,z,t)+r_b(A,z,h_{<t},t)\bigr).
$$

This is a sum of conditional H probabilities along a fixed history. For an
own-model history it is the discrete counting compensator. Under a crossed
history it is a conditional propensity measurement, not the expected count of
a freely sampled crossed model. The survival law instead sums softplus of
logits; substituting that quantity would change the accounting.

For each history $h$, the symmetric base and residual contributions are

$$
B(h)=\tfrac12[(C_{10}-C_{00})+(C_{11}-C_{01})],
\qquad
R(h)=\tfrac12[(C_{01}-C_{00})+(C_{11}-C_{10})].
$$

Average each over the original and continued histories $h_0,h_1$. Define

$$
D=\tfrac12[(C_{00}(h_1)-C_{00}(h_0))
+(C_{11}(h_1)-C_{11}(h_0))],
$$

$$
E=[N_1-C_{11}(h_1)]-[N_0-C_{00}(h_0)].
$$

Then the observed count difference is exactly $N_1-N_0=B+R+D+E$.
$D$ measures the effect of substituting the realized history with weights fixed;
$E$ is the difference of realization remainders. The symmetric averages avoid
choosing one arbitrary order of exchanges. They do not make the contributions
independent causes: the generated history is itself downstream of the model,
and the two learned logit paths can compensate for each other. Crossed terms
may be outside the training distribution.

## Results and distinct mechanisms

The table shows selected profile-2 cases, rounded to one decimal head. All 32
original-to-continuation comparisons, including improvements and small changes,
are retained in the result artifact.

| Audio and continued model | Original → continued H | Base B | Residual R | History D | Remainder E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hysteric, shared | 1297 → 4417 | 371.7 | 69.8 | 2674.6 | 3.9 |
| Fool Moon, routed | 1078 → 3778 | 987.1 | −27.6 | 1830.5 | −90.0 |
| Take, shared | 928 → 1258 | 644.9 | −149.5 | −161.1 | −4.4 |
| Yomi yori, shared | 2939 → 3590 | 1683.9 | −10.2 | −988.8 | −34.0 |

Hysteric's shared continuation strongly increases the H count, while the base
alone has expected counts of 1001.05 initially and 1089.60 after fitting. The
median residual on each model's own history changes from −1.747 to +1.657.
The fixed-history exchanges identify substantial history-dependent probability
change; an increased audio-base count alone is not a sufficient description.

Yomi yori differs. Its shared base-only expected count increases from 1952.93
to 3323.68, while the substituted generated history offsets part of the rise.
Take also has a positive base contribution opposed by residual/history effects.
Thus a small net count change can conceal sizeable opposing changes, and the
same architecture can exhibit different mechanisms across audio contexts.

For the descriptive dominance test, a case is eligible when its count changes
by at least 10% of the initial count. A component must have median share at
least .60 of $|B|+|R|+|D|$, agree with the count-change sign in at least 75% of
eligible cases, and have median $|E|/|N_1-N_0|$ at most .25. At least four cases
are required.

| Measurement | Shared continuation | Routed continuation |
| --- | ---: | ---: |
| Eligible cases of 16 | 13 | 15 |
| Median absolute base share | .218 | .376 |
| Median absolute residual share | .156 | .111 |
| Median absolute history share | .495 | .465 |
| History contribution agrees with count-change sign | 10/13 | 14/15 |
| Median relative realization remainder | .017 | .019 |
| Component meeting every dominance condition | None | None |

The answer under this rule is mixed. Summing all changes would emphasize the
largest increases and conceal the opposite-signed examples, so an aggregate
percentage is not reported as a share of total model failure. The small paired
remainder also does not quantify all seed sensitivity; only one seed per audio
is examined.

## Implications for the joint generator

Bounding history logits protects against an indefinitely strong historical
veto after a long wait. It does not normalize generated density or prevent
history-dependent amplification during active passages. Joint teacher-forced
likelihood trains the sum of the two paths; it does not separately ground the
base as an independently calibrated musical activity model.

This diagnostic does not explain all LN or row-control loss. For example, the
routing evaluation's Revenge profile-2 output has *lower* H density but weaker
LN realization; its As It Was comparison also shows a large LN change with a
much smaller density change. R/row response must be evaluated with H held fixed
before assigning those failures to a dense plan or to restored R1 alone.

A useful next comparison exchanges complete generated H plans between the
original and shared continuation while each R/row model retains its own full
audio encoding, profile and committed state. This tests whether the earlier H
process and later materialization retain complementary improvements. It also
tests the reverse pairing, which is needed to distinguish a plan effect from a
materialization effect. These are coherent conditional-model queries, but their
generated histories may still be outside training support. Native completion,
short attacks, requested controls and Lens-inspected LN/chord organization must
decide whether any composition is useful.

Changes of representation remain open. A
[conditional renewal model](https://proceedings.neurips.cc/paper/3740-time-rescaling-methods-for-the-estimation-and-assessment-of-non-poisson-neural-encoding-models.pdf)
is an analogue for separating an audio-dependent time scale from interval
organization. It is not an implemented solution, and a temporal-fit statistic
would still not establish musical correspondence. Exact count conditioning,
history calibration and separate audio branches also impose different
approximations; this accounting alone does not select one.

## Verification and provenance

The run uses clean product source `9e6dbc61511ec9b1a45bc55148a344886e81d682`,
Apple M5 with 24 GiB RAM, Python 3.10.20, Torch 2.11.0 and one CPU thread.
It takes 74.945 seconds without fitting weights or producing new row charts.
All 48 pure HeadPlanner replays exactly reproduce their saved H sequences.
Dense and online head histories agree at 768 selected positions within 2e-5
absolute/relative tolerance. Synthetic partitions cover BOS, H at zero,
multiple heads in one bin and the final clock. Every native clock is counted
once; factor and eight-second/whole-song accounting errors are below 1e-6 H.
Parameter fingerprints and input hashes remain unchanged.

Checkpoint, corpus, bank, panel and seed identities are given in the
[routing report](profile_density_routing_evaluation.md#reproduction-identities).
Local owner is
`artifacts/joint-audio/20260925-head-factor-drift-v1/attempt2`.
The corrected diagnostic reads the original packaged export from
`chart/rows.jsonl` and the continuation exports from `rows.jsonl`, each verified
against its existing hash. The preceding path-resolution failure produced no
measurements and is preserved separately.

Result SHA-256:
`4aaa7036b335c90ec75af4128f9e1ad3e9dc5031420b4a3fe0d73a819b74b1e9`.
Freeze SHA-256:
`301530989f1468e9f31b7667eafc1b27275d2079e1e23b6fe03d4d2dbabe5c9d`.
Diagnostic script SHA-256:
`c99084123613c161d8f5f0cc038bcf9aed6efd4bdd6c7bc1e9e44979f2ff9b97`.
