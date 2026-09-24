# Persistent arrangement condition experiment

The [observed-prefix study](audio_joint_playtest_v2.md#conditional-continuation-result)
found that real prefixes can preserve tap/chord organization while native
prefixes on the same audio sustain very different LN composition. Short early
prefixes and longer continuations remain variable. This motivates a controlled
comparison with a persistent arrangement condition; it does not prove that a
latent variable is necessary or that an LN alternative is bad.

The control continues the existing global-audio/bounded-history model from its
fixed candidate checkpoint. The intervention adds four categorical states,
without changing integer-ms hazards, legal complete rows, finite action history,
physical occupancy or the canonical Mel frontend. Both arms use the same paired
targets, source-time intervals and initial common weights. Further fitting on
restored targets is shared, so any common gain cannot be attributed to the latent.

## Model and information ownership

`intent_model.py` adds 5,192 parameters to the 3,461,828-parameter candidate.
One centered 128-dimensional code offset enters the existing coarse audio
context coordinates. Those coordinates already reach the row decoder, timing
base and bounded historical timing path. The offset is fixed throughout a song,
including silence; it is not a beat grid or a section plan. Zero initial offsets
make every conditional decoder identical to the starting control.

The full-audio prior is a 128→32→4 MLP over the mean of real coarse tokens.
The training recognition network is an 8→32→4 MLP over whole-target descriptors:
log head rate, LN/head fraction, four attack-row chord-size fractions, occupied
lane-time fraction and log median LN duration. Only the categorical state reaches
the decoder. These reference descriptors are unavailable and unused at inference.
The code carries at most two bits, has no predefined style/difficulty name, and
does not disclose future onsets or LN endpoints. Multiple charts for one audio
remain separate targets. Existing partial human labels are evaluation evidence.

For complete chart $Y$, audio $X$, code $z$ and actual integer-clock duration
$D=(T+1)/1000$ seconds, the conditional mixture is

$$
p(Y\mid X)=\sum_{z=1}^{4}p_\psi(z\mid X)p_\theta(Y\mid X,z).
$$

The restricted recognition family is $q_\phi(z\mid s(Y))$. For an interval
$W_j$ drawn with probability $p(j\mid Y)$, training minimizes

$$
\widehat{\mathcal L}
=\sum_z q_\phi(z\mid s(Y))\frac{\ell_j(z)}{p(j\mid Y)D}
+\frac{\mathrm{KL}(q_\phi(z\mid s(Y))\|p_\psi(z\mid X))}{D}.
$$

The likelihood includes every event, no-event millisecond and complete row.
Four-state expectations are enumerated exactly after shared audio/history
encoding. There is no straight-through estimator or sampled replacement prefix.
The KL belongs to the whole chart and is normalized consistently with its
likelihood. No fresh latent is chosen at an interval boundary.

This adapts the input-prior/target-recognition distinction of
[conditional generative models](https://proceedings.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf).
It does not import MusicVAE's bar resets or quantized output. A strong decoder
can ignore a latent; [posterior-collapse research](https://arxiv.org/html/1901.05534)
also cautions that KL magnitude alone does not demonstrate useful information.
State usage and prior-sampled chart behavior therefore require separate checks.

## Execution and evaluation

The packaged entrypoint is
`python -m ensomi_model.research.joint_audio_continuation.intent_hydra` with
`states=1` or `states=4`. It requires pinned initial weights and corpus bytes,
a clean checkout and a fresh run directory. Both arms consume the same frozen
group/arrangement/interval plan. Checkpoint normalization remains unchanged;
AdamW starts fresh. Outputs distinguish fixed endpoint selection, resource stops,
actual exposure, negative-ELBO estimates and posterior reconstruction.

Population interval evaluation estimates the whole-chart negative ELBO per
second. BOS reconstruction and chart-normalized KL are reported separately.
Exact marginal evaluation must first sum each code's likelihood over the entire
chart, then apply one prior logsumexp. Independent per-interval mixtures would
describe a different generator and are not valid whole-song evaluation.

Native generation uses the existing source-free `joint_audio` entrypoint.
An intent checkpoint draws one prior code with RNG seed
`generation_seed XOR 0x17C0`, independently of event sampling, and records its
code and prior probabilities. `intent_code=0..3` selects a fixed state for
diagnostics; other model types reject it. Target-conditioned reconstruction and
fixed-code inspection must remain distinct from audio-prior sampling results.
The regular incremental row/LN publication and physical validity rules apply.

The experiment remains open. A lower bound, distinct codes, low short-head
counts or valid export alone cannot establish playability. Evaluate actual
head/LN composition and scoped Tech, repetition, tap flow and LN coordination
through Beatmap Lens, preserving valid stylistic alternatives.
