# Condition–history interactions in R1 placement

Without layout modulation, the main R1 row head cannot make its response to a shared additive condition
depend on the current history. In particular, that condition cannot change its
relative score for a row and its four-column reflection. Other nonlinear paths
in the complete policy remain capable of such interactions. This is a limitation
of the dominant readout path, not an impossibility result for the whole model.

The [history diagnosis](balanced_condition_alignment.md) makes this distinction
relevant: at a repeated-group decision, the main row head contributed +2.510
repeat-versus-alternate log-odds, compared with +.062 from head routing and +.024
from frontier2. The two alternatives were mirror-related two-TAP rows.

## Where the interaction disappears

In [ControlledAudioModel](../../src/ensomi_model/research/controlled_audio_continuation/model.py),
nonlinear fusion first combines temporal history with exact replay. Direct local
audio, global audio, H preview and control projections are then added to the
result. With active-LN cues disabled, the two hand inputs have the form

$$
z_L=h_L+d(X,H,C),\qquad z_R=h_R+d(X,H,C),
$$

where $h$ depends on history and replay, and $d$ is shared across hands.
[JointHead](../../src/ensomi_model/research/bounded_typed_continuation/model.py)
has linear hand unaries and a linear state-to-coupling-matrix map. Its action
embeddings are learned parameters, not state-dependent activations. Consequently,
for fixed network parameters the complete score vector is affine in the hands:

$$
J(z_L,z_R)=L_Lz_L+L_Rz_R+b.
$$

The condition increment is therefore $(L_L+L_R)d$, independent of $h$. Because
the head is mirror equivariant and the increment is shared by both hands, that
increment is identical for every reflected candidate pair. For such a pair
$a,Ma$, the contribution
$J_a(h_L+d,h_R+d)-J_{Ma}(h_L+d,h_R+d)$ is independent of $d$.

This matters for transitions such as repeating the previous two-key group versus
switching to its complementary reflected group. A style request should be able to
change that preference relative to whichever group is currently active. A fixed
candidate bias cannot express this change through the main path. Direct audio
and future H preview have the same additive-path limitation when history and
exact replay are held fixed. Audio can still affect chosen H times and thereby
change later replay inputs indirectly.

## What does not cancel

Head routing and frontier2 contain nonlinear functions of the conditioned hand
states. R1's composition readout also has a nonlinear interaction between
context and controls. Those paths mean the full policy is not condition-blind.
However, composition cannot change relative placement odds between two candidates
with identical head/LN/release counts: its family mass and within-family
normalizer cancel in that comparison. LN-amount feedback cancels there as well.
Difficulty-dependent external recovery preferences may still change those odds.

The restriction concerns a current query with its support fixed. It does not
claim that complete rollouts under different conditions remain identical; changed
actions, timing and occupancy can propagate into future history.

A numerical audit of the aligned-1200 checkpoint used 32 synthetic hand-context
pairs. A shared additive perturbation changed the main-head increment across
contexts by at most $5.25\times10^{-6}$ and violated reflected-candidate symmetry
by at most $3.81\times10^{-6}$, consistent with float32 arithmetic. Changing
trill from absent to prominent changed the main `[01]`/`[23]` difference by at
most $2.98\times10^{-6}$. The nonlinear routing contribution changed by as much
as .00902 on those synthetic inputs. These measurements verify the algebra;
they do not estimate generated quality or typical control strength.

## Shared conditional feature modulation

A small multiplicative interaction can remove this limitation without adding
skeleton decisions or a pattern vocabulary. For example, apply a shared,
condition-generated feature scale to each hand's historical state before the
main row head, retaining the existing additive projections. Initialize the scale
to one so the old model is recovered exactly. Sharing the modulation network
preserves mirror equivariance; scaling different historical hand states allows
the condition to change their relative action preference.

The closest established primitive is
[FiLM](https://arxiv.org/abs/1709.07871), which uses condition-dependent feature-wise
affine transformations. Applying it here would be an adaptation, not a new
general conditioning method. An alternative is a small nonlinear residual fusion
before the readout. Either belongs inside R1. Count ownership, H/R interfaces,
physical replay, scoped controls and the candidate frontier remain intact.

The experimental `layout_modulation=True` option implements a bias-free
hidden-by-hidden matrix $W$, shared across hands and initialized to zero. Using
the existing projected audio, preview and control sum $d$, only the main row
head receives

$$
\widetilde z_h=z_h\odot\left(1+\tanh(Wd)\right).
$$

The scale is bounded between zero and two and starts at one. The 128-wide model
adds 16,384 parameters. Routing, release routing, count composition, the
actual/reference LN condition branches and frontier continue to read their
existing contexts. In particular, count-family normalization still precedes the
frontier; this is not a new skeleton count plan or player-response cost.

The option is disabled by default. Checkpoints record it in probability options;
older checkpoints without the field load the unmodulated architecture. A new
modulated architecture can import the existing tensors with only the zero
modulation matrix absent. Once saved, loading remains strict. Focused CPU/MPS
tests verify initial probability identity, learnable condition/history coupling,
reflection symmetry and cached-generation/scoring agreement. These properties
do not establish a generation-quality benefit.

The controlled comparison below tests this capacity with actual continuation
learning. More expressive conditional logits do not themselves establish stable
trill, jack, Tech or LN organization.

Executable source audited: `3cf169cda06c09b3487f12f6dcdacf81b4fc74da`.
Checkpoint SHA-256:
`67ef81fbb9fb2ecfc5ca19d8fc6767040b846e1ba8c41d4b8d61067abc08ee26`.
Numerical probe: `conditioning-probe.json` in artifact owner
`20260926-trajectory-kernel-r1-v1`, seed 261580. The source argument applies to
the code's affine main head, independently of these checkpoint values.

## Continuation learning and scoped control

Adding modulation improved the measured continuation distributions and scoped
difficulty response, but did not establish reliable style or LN-amount control.
The candidate remains experimental; it does not replace the selected core model.

The comparison repeated the 128-update
[physical trajectory study](physical_trajectory_matching.md#paired-rr1-learning-result)
with only the zero-initialized modulation matrix added. Both models started from
the same aligned-1200 weights. All 128 target scopes and 384 genuine source
examples, rollout seeds, three independent trajectories per update, loss weights
and optimizer settings matched. Audio and H stayed frozen; R/R1 trained, with
the new matrix using the existing condition-path learning rate. A four-update
unmodulated reproduction matched all twelve sampled trajectory hashes and
kernel objectives. Reordering the shared projections caused at most
$1.12\times10^{-6}$ source-NLL and $3.85\times10^{-6}$ final-weight differences;
the reused full comparator is not a bitwise reproduction.

Qualification generated thirty scoped difficulty continuations, twelve style
charts, nine LN charts and six native charts for the new endpoint. The source-H
comparisons use three seeds per request; the native panel uses three audios with
one static and one switched request each. Style and LN distances are empirical
short-block kernel U-statistics, not semantic accuracy or playability scores.

| Measurement | Unmodulated trajectory learning | With modulation |
| --- | ---: | ---: |
| Mean distance across four reserved style scopes | .06534 | .04953 |
| Mean scoped difficulty absolute error | .85922 | .73150 |
| Mean distance across three LN scopes | .03087 | .03248 |

The style improvement was .01581, exceeding the predeclared .005 absolute and
15% relative gates. Scoped control and restored-range regression checks also
passed. Those checks allowed bounded regressions against continued factual
learning; they were not absolute acceptance criteria for the playable system.
In particular, the native gate checked difficulty regression, not LN accuracy.

### What the generated charts retained and missed

Focused Lens inspection covered complete-row action relationships, true LN
endpoints and time-proportional context. In the Tech example, irregular source
timing survived, but much of the viewed placement became regular complementary
pairs. The stream example had moving singles with chord accents. The jack view
retained overlapping chord attacks with changing groups and occasional LNs.
Starry Jet gained sustained hold roles; Shippaisaku still contained many
independently staggered releases. These are local visual observations, not new
human annotations or a full-song playtest.

For Miraie, median LN durations were 157/107/106 ms in the three draws, compared
with 183/183/183 ms in the unmodulated trajectory candidate. The pooled style
gain therefore does not imply improvement in every physical relationship.
The source-H trill guard had no exact repeated full group in its target passage,
but also failed to sustain its source's fixed complementary-group exchange.
Removing repetition alone did not recover trill organization.

A separate control intervention froze one generated 3,336-row prefix and its
H plan, then forked trill absent/prominent requests on [288156,290040) ms under
three matched future seeds. Difficulty and LN requests were identical across
each pair; other styles were unspecified. Endpoint-specific caches were rebuilt
from the same committed rows, with no entering hold at the override.

In one modulated draw, the prominent request increased exact repeated-group
transitions below 100 ms from 7 to 13. The full action sequence showed groups of
three to five repeated `[01]` or `[23]` rows followed by group changes, rather
than sustained A/B exchange. Both requests shared the first eight target rows.
This supports condition sensitivity, but not correct semantic control. It is
not evidence that every repetition is unplayable: other style fields were
unspecified. R receives controls too, so the full continuation contrast is an
R/R1 experiment even though H is fixed.

### Native controls and service time

For static difficulty 3 and LN fraction .2, native results were:

| Audio | Difficulty readout | Realized LN head fraction |
| --- | ---: | ---: |
| Zenithfall | 3.796 | .112 |
| Hysteric | 3.818 | .148 |
| Take | 4.308 | .163 |

Switched requests were difficulty 4.5 and LN fraction .6 on [64000,96000) ms,
with difficulty 3 and fraction .2 before and after. Zenithfall realized LN
fractions .095/.444/.085 across those three separately measured ranges;
Hysteric .190/.623/.167 and Take .161/.605/.180. The override does not erase
committed rows or close pre-existing holds. Static and switched modes used
different seeds, so their difference is not a pure control effect.

All six native H sequences exactly matched their unmodulated counterparts.
The observed differences therefore arise downstream of the frozen H policy.
Replaying the deployed bounded LN feedback from committed rows found it at
its upper correction limit before 46.2% of Zenithfall's static H choices and
60.4% of its restored-range choices. A binding cap can limit correction, but
does not identify why the learned distribution underproduces LNs or establish
that raising the cap would preserve appropriate articulation.

On the 24 GiB M5 Mac, the new fit took 1,209 seconds. Peak sampled footprint was
9.59 GiB and MPS driver allocation 3.50 GiB. Native first-thirty-row publication
took .240–.542 seconds; the slowest measured eight-second service window took
.327 seconds. These timings start from cached canonical Mel with a loaded model;
they exclude waveform decoding, Mel preparation and model loading. This panel
provides substantial generation-ahead margin, not a production deadline guarantee.

## Implication for supplementary style learning

Difficulty-stratified style exposure can reduce sampling imbalance where real
examples exist. It cannot supply missing combinations. In the frozen eligible
human pool, prominent Tech had three local 3–4 examples and none in the other
2–6 bins; prominent trill had 2/3/2/0 examples in the four bins. Prominent stream
had 3/7/4/3 and jack 1/5/2/1. These counts describe this filtered fitting pool,
not the complete ranked corpus.

A style label applies to its annotated range, which may contain several different
figures. Inspection of the five prominent trill actor targets found clear
exchange episodes embedded in broader motion, holds or repeated groups. Training
every row as a positive trill instance would change that supervision. Missing
labels also remain unknown, not absent. Human confirmation of a label does not
turn inherited machine evidence masks into trusted episode localization.

The modulation result supports keeping a condition/history interaction inside
R1. The remaining question is whether explicit factual style-condition
discrimination can use that capacity more directly than short-block matching,
and whether the distinction survives a shared generated prefix. An appropriate
comparison holds difficulty-balanced exposure fixed, changes only the learning
signal, and evaluates the actual scoped continuations. Source likelihood alone
cannot decide that question.

Learning source: `24fc4f1f724137787758ff50a5b5750d09d96a48`. Artifact owner:
`20260926-layout-modulation-r1-v1`. Candidate checkpoint SHA-256:
`b8aecd3e1f43339d2c1e3aff6d245a41e32a12008ddc20fd47e9eccb99544546`.
Qualification and native results, common-prefix style forks and the read-only
`review-audit` retain the complete measured scopes and generated exports.
