# Condition–history interactions in R1 placement

The main R1 row head cannot make its response to a shared additive condition
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

## Consequence for the next architecture comparison

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

Such a change still needs actual continuation learning and generated-chart
evaluation. More expressive conditional logits do not themselves establish stable
trill, jack, Tech or LN organization. The ongoing physical-trajectory objective
comparison retains its original architecture and is not modified by this audit.

Executable source audited: `3cf169cda06c09b3487f12f6dcdacf81b4fc74da`.
Checkpoint SHA-256:
`67ef81fbb9fb2ecfc5ca19d8fc6767040b846e1ba8c41d4b8d61067abc08ee26`.
Numerical probe: `conditioning-probe.json` in artifact owner
`20260926-trajectory-kernel-r1-v1`, seed 261580. The source argument applies to
the code's affine main head, independently of these checkpoint values.
