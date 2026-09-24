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
The native metric `arrangement_values` contains the selected representative,
not statistics measured from the generated chart. Evaluation must recompute
realized descriptors from the output rows.

Native evaluation must test realized descriptor control and complete-chart
failures alongside Lens-inspected musical organization. The prior's own
calibration is a separate question. Neither the profile approximation nor
successful streaming establishes the final playable model.

## Measured control and remaining coupling failures

A matched comparison at source
`10ddaa8daf7db1b5851d1f8c10733744da368b7e` continues the same planned checkpoint
for 1200 updates, with and without the profile modules. Both arms use the same
615 TRAIN arrangements, 240 audio groups, exposure plan and fresh optimizer.
Fitting takes 964.4 and 920.7 seconds, respectively, on Apple M5 MPS. No endpoint
is selected by validation NLL. The conditioned model has 4250174 parameters.

Native evaluation uses five audio assets and nine fixed audio/seed cases:
two automatic arms plus four explicit requests per case give 54 complete,
independently reparsed outputs. The requests are the first four TRAIN medoids;
they are not derived from the evaluation arrangements. Aggregate measurements
describe this small cohort, without a population confidence claim.

| Measurement | Unconditioned continuation | Shared profiles |
| --- | ---: | ---: |
| Mean squared distance to the requested standardized descriptors | 13.2592 | 7.5809 |
| Strict same-column successive attacks below 20 ms, all outputs | 0 | 0 |
| Release-to-next-head gaps at most 20 ms, automatic outputs | 0 | 3 |

The descriptor error decreases 42.8%, with improvement in every component.
Centering requests and outputs within each audio/seed case excludes a constant
output that merely approaches the request mean: error relative to that constant
case decreases 32.0%, 34.4% and 26.3% for H rate, chord width and LN-head fraction.
This establishes partial use of all three controls in these trajectories.
It does not establish calibrated realization or playable organization.

The finite response matrix makes the coupling visible. Rows are requested
standardized descriptor changes; columns are realized standardized changes.
The matrix fits the four fixed requests and averages across the nine cases;
it is not an infinitesimal derivative.

| Requested change | H rate | Chord width | LN-head fraction |
| --- | ---: | ---: | ---: |
| H rate | .363 | .491 | .050 |
| Chord width | .200 | .468 | .040 |
| LN-head fraction | .073 | -.069 | .177 |

Increasing requested H rate also increases chord width substantially. The
LN-heavy representative requests .734 LN-head fraction, but its nine outputs
realize only .036 to .282. The audio prior's mean reference-class cross-entropy
on 36 VAL charts is 2.534, versus 2.588 for the constant TRAIN-mass prior. That
modest change is separate from native control and quality.

Lens inspection covers all 43 declared scopes: 118 time-proportional pages,
complete action tables and LN articulation. Broad chord flow appears in Good
Luck and Prom Queen under the broad-chord request, while Who remains mainly
single-note flow under the same request. Independent LN release survives,
including long holds spanning other-column attacks. Few LN starts can still
produce substantial held-lane time: one Death Piano scope contains only one
new H, but entering/new holds last 4.389, 6.673 and 6.968 seconds. LN-head fraction
therefore cannot substitute for sustained occupancy or coordination. Fine Tech
and dump coverage remain unresolved; no listening or player verdict was taken.

All individual completion, startup and strict-attack guards pass, but the
automatic release-to-head guard fails. All automatic fixed scopes remain
nonempty; the controlled profile-0 Death Piano seed-19 scope is empty, outside
the declared automatic nonempty-scope guard. The three automatic and two explicitly
controlled witnesses have distinct action dependencies:

| Generated case | Release to next head | Observed dependency |
| --- | --- | --- |
| Automatic Good Luck, seed 17, column 2 | 197518 to 197536 ms | Four columns were held at 197436. The first release leaves only 18 ms before the required H. |
| Automatic Prom Queen, seed 17, column 0 | 84345 to 84351 ms | Column 1 is available and rested; the row instead chooses the just-released column. |
| Automatic Airborne, seed 33, column 2 | 133516 to 133519 ms | The preceding row attacks columns 0/3 and releases 1/2, leaving no column avoiding both short HH and RH at the next H. |
| Profile 1 Good Luck, seed 17, column 1 | 184037 to 184055 ms | Three holds release, leaving only column 0 rested. A single head is available, but the chosen double requires a just-released column. |
| Profile 3 Prom Queen, seed 17, column 2 | 39201 to 39204 ms | The preceding row attacks columns 1/3 and releases 0/2, again consuming every rested choice for the next H. |

The confirmed bad-pattern criterion remains strictly below 20 ms between
successive same-column attacks, including TAP and LN heads. RH gaps are a
separate diagnostic. Across the 651 admitted paired source charts, all 151003
release-to-next-head pairs exceed 20 ms, with a minimum of 30 ms. The source
parser and admission path preserve literal starts and endpoints rather than
clamping these gaps; corpus selection can still bias their distribution.
This audit does not establish a universal RH threshold.

These failures distinguish release timing, immediate row choice, and an
earlier row's consumption of future playable choices. Reducing one close-gap
count or improving descriptor error alone cannot promote the model. Crossing
generated H plans while holding downstream profile conditions fixed can
separate the plan-mediated and downstream paths of the measured control
response before changing their architecture.

The unconditioned endpoint is
`5f26b7b15d97fa2d6964cf77a4cadea016f5ace9dfd578ce2fae5dc7c8a0e121`;
the conditioned endpoint is
`abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef`.
Local evidence owner: `artifacts/joint-audio/20260924-shared-profile-v1`.
Native result SHA-256:
`de5544efff6ffdaeca0c0388636e60dcfe67dc2691d20466c10704be4f8abea4`;
completed Lens review:
`0a7fdd8562fc8885a034a998eecc1ba75ccd16802bd612a956c7747bbe703e1e`.
