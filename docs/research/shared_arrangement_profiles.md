# Shared arrangement profiles

This research model adds one chart-level arrangement choice to the planned
audio generator. The selected profile conditions head timing, release timing
and row materialization together. A small prior selects it from complete audio;
an explicit request can select the same profile instead. This is an implemented
model hypothesis, not an established improvement in native chart quality.

## Observable condition and its approximation

Each complete source chart has three descriptors: H rows per decoded-audio
second, note heads per H row, and LN-head fraction. They distinguish timing
density, average chord width and use of LN starts. They do not define difficulty,
Tech, Jack or LN coordination, nor prescribe local density or LN duration.
Alternative arrangements of the same audio remain separate targets.

The preparation function `profiles.build_profile_bank` uses TRAIN charts only.
It transforms the dimensions with log, log and arcsin of the square root,
respectively, then standardizes them with weights uniform over song groups and
over each group's charts. Deterministic weighted medoids select actual joint
profiles. The first is the weighted squared-distance optimum; later initial
representatives maximize weighted distance from the nearest selected profile.
Nearest-profile assignment and replacement by a cluster member nearest its
weighted centroid repeat until stable or 50 iterations. Source-hash order
breaks ties. Empty clusters and insufficient distinct profiles fail preparation.

The first experiment uses 16 representatives. This approximates a continuous
arrangement distribution with finite support; it does not restrict the
native-ms event alphabet, chord layouts or LN endpoint support. Prepared data
records the scales, representative source identities, assignments, population
masses and quantization error. The runner reproduces the bank from its verified
TRAIN corpus before fitting and rejects changed bytes or preparation results.

The computable-attribute principle has a nearby music-generation analogue in
[MuseMorphose](https://slseanwu.github.io/site-musemorphose/), which conditions
piano generation on rhythmic and polyphonic attributes. The persistent-code
principle also appears in [CTRL](https://arxiv.org/abs/1909.05858). This prototype
adapts those ideas to separate native-time hazards and legal four-lane rows.
Full-audio context remains available, and no bar grid or fixed section labels
are introduced.

## Shared information and probability factors

Let $k$ identify a representative, $A$ be complete audio, and $Y$ contain the
generated head stream, releases and complete rows. The model factors

$$
p(k,Y\mid A)=p(k\mid A)\,p(Y\mid A,k).
$$

The prior is a 128-to-16 linear softmax over the mean of valid full-song coarse
audio tokens. Padding cannot enter that mean. A bias-free 3-to-224 linear
projection of the standardized representative is added to the audio condition
used by all three generation factors. It is constant across a chart, while
the actual audio continues to determine local musical changes. The two new
modules add 2736 trainable parameters with the default backbone.

The projection starts at zero, preserving common-model generation for every
profile. The prior's initial weights are zero and its bias is the log TRAIN
class mass. Generation encodes full audio once and chooses one profile with an
independent CPU RNG, using the run seed xor 0x61F9. Head, release and row RNG
streams retain their existing identities. The bank and its normalization are
checkpoint buffers; native generation does not open a source chart or bank file.

The condition does not reconnect row-content history to skeleton timing.
The release factor retains its LN-only projection, and the pilot H stream
retains its own history. Rows retain direct audio, future H preview, causal row
history, exact replay and frontier2. No actual future LN endpoint is exposed.

## Supervision and interpretation of likelihood

A chart's nearest representative supplies its fixed training assignment. This
is an observed auxiliary assignment derived from the complete target, not a
latent reselected independently at each interval. The conditional head/release/
row likelihood uses the existing chart-time importance weights. The prior
contributes one cross-entropy factor per chart, divided by
`(duration_ms + 1) / 1000`, the same inclusive integer-clock seconds used by
the event objective. Repeated interval samples average that factor rather than
multiply it. The descriptive H rate still uses decoded audio duration. Both
paths can update the shared audio encoder.

The log reports conditional NLL, prior NLL and their joint sum separately.
Validation with the reference-derived profile is labeled `reference_profile`.
It is not the audio-only marginal likelihood or a native evaluation. Inference
uses the audio prior or an explicit profile request. Its realized chart can
deviate from the requested descriptors because the condition is soft. Lower
conditional NLL cannot establish control or playable organization.

## Fitting and native use

`PlannedTrainConfig` accepts paired `initial_checkpoint_file` and
`initial_checkpoint_sha256` fields for a warm start from an unprofiled planned
checkpoint. Every common tensor is copied; architecture, corpus identity and
audio normalization must agree. Optimizer state starts fresh. This path replaces
R1 weight transfer for that run; it is not optimizer resumption.

Paired `profile_bank_file` and `profile_bank_sha256` enable the shared-condition
model and require planned initialization. An otherwise matched warm-start run
without these fields is the unconditioned control. The packaged training command
remains `ensomi_model.research.planned_audio_continuation.hydra`; it saves the
resolved settings, reproduced bank and initialization receipt.

Profiled checkpoints use `joint-audio/planned-profile-v1`. The
[audio-file entrypoint](planned_audio_continuation.md#generate-and-stream-from-an-audio-file)
loads that family directly. `arrangement_profile=null` samples the prior once;
an integer requests a bank index. An override absent from the checkpoint is
rejected. Results record the selected index, raw descriptor values, prior
probabilities, profile RNG seed and whether selection was requested or sampled.

Native evaluation must test realized descriptor control and complete-chart
failures alongside Lens-inspected musical organization. The prior's own
calibration is a separate question. Neither the profile approximation nor
successful streaming establishes the final playable model.
