# Shared action spacing for H, R and complete rows

The optional `minimum_action_gap_ms` setting applies one common constraint to
same-column attack spacing (HH), LN duration (HR), and release to next attack
(RH). A positive value $g$ requires all three intervals to be at least $g$
native milliseconds. Zero retains the existing distribution. The
[ranked reference census](ranked_2to6_action_reference.md) motivates $g=21$ for
the initial 2–6-star population. This is one explicit response condition, not
a complete player model or a claim that every ranked object satisfies it.

The setting belongs to the model checkpoint and packaged training schema.
Positive values require `condition_full_holds=true` and at least nine future
heads of lookahead. They cannot combine with the separate `row_constraint` or
short-attack correction policies. Old checkpoints load with gap zero. Enabling
the law adds no parameter tensors; planned warm starts record its config change.

## Finite continuation support

Four columns whose attacks are separated by $g$ require distinct H times to
satisfy $h_i-h_{i-4}\geq g$. The H process therefore excludes clocks before
the fourth preceding H plus $g$, using only its own skeleton history. Close
cross-column heads remain possible. Incompatible fixed H plans are rejected.

For a candidate complete row at $t$, check immediate HH, HR and RH against
exact committed state. In its post-state, a free column's earliest next attack
is the later of its last attack and release, plus $g$. A held column starting
at $s$ can release no earlier than $\max(t+1,s+g)$ and attack again no earlier
than $\max(t+1,s+g)+g$. The $t+1$ prevents releasing a retained hold in the
already committed row. A column released by that row instead uses $t+g$.
Every retained hold must still be closable by the true audio end.

Test future Hs through $t+2g$ by assigning one TAP to an available column and
moving its availability to H plus $g$. Existing holds may release at their
earliest eligible clocks. Choosing any already-available column is equivalent
for later feasibility; all other eligible columns remain available. Extra
attacks and new holds cannot improve this existence test.

All restrictions inherited from the candidate state expire by $t+2g$. A later
failure would require all four columns to have been used by Hs less than $g$
earlier. The failing H would be a fifth event in that interval, contradicting
the H-capacity law. Thus the finite horizon suffices under that global law.
The preview must reach the horizon or declare completion; nine future Hs also
suffice, because at most eight can fit in the two-gap horizon.

This is an existence argument with freely chosen future TAPs and releases. It
does not estimate the learned policy's probability of taking a desirable future
or establish musical quality of the chosen chord/LN organization.

## Actual release timing preserves the continuation

After commitment, let $F$ be the number of free columns. Find the first future
H, $\tau$, whose cluster of events less than $g$ apart contains $F+1$ heads.
Those heads cannot all use the existing free columns: a held column must
release by $D=\tau-g$. For $F=0$, $\tau$ is the next H. This deadline reads
only the LN projection and H preview, without TAP layout or generic replay
feedback. Combine it with the earliest clock at which a current LN reaches
age $g$, and the true audio-end closure bound.

If $D$ precedes the next H, condition the first-R law on an eligible release
by $D$. Its last eligible hazard is certain; earlier hazards use the normalized
first-event distribution from the existing R model. If H occurs first, that row
changes the state and the deadline is recomputed. A coincident H/end deadline
is satisfied in the complete row. Every release subset still passes post-row
viability before commitment. A crop or publication boundary never closes holds.

The inherited frontier2 earliest-native-opportunity feature remains its stated
approximation. The exact spacing condition is separate and does not turn that
feature into a forecast of actual R. The new deadline law, in contrast, changes
the actual release process.

## Shared probability law and verification scope

Compute the existing row distribution $p$, including count/layout composition,
then condition it on the admitted complete-row set $M$:

$$
p_g(y)=\frac{p(y)\mathbf 1[y\in M]}{\sum_{v\in M}p(v)}.
$$

Applying this after count-group normalization allows a group's total mass to
change when its layouts are inadmissible. It does not restore a learned
frontier2 group bias canceled inside the original normalizer.

Interval fitting uses the same H support, eligible R clocks, normalized waits,
forced hazards and complete-row law. Hypothetical waits can extend beyond an
interval: their normalizers use complete available audio and the declared H
plan, never actual future LN endpoints. Splitting an interval only censors its
loss. Source targets outside the configured condition raise explicit errors;
data admission must record these cases without retiming them into other charts.

Native generation reads the setting from the checkpoint. The buffered verifier
checks all HH/HR/RH against it and uses a halo of at least $g-1$ ms; $g=21$
retains the existing 20-ms halo. Resource caps and incomplete publication retain
their existing open-hold semantics.

[`spacing.py`](../../src/ensomi_model/research/planned_audio_continuation/spacing.py)
owns the constraint calculations. Focused checks compare them with an independent
native-clock action search and cover conditional likelihood arithmetic, interval
and gradient consistency, CPU/MPS agreement, skeleton TAP-layout independence,
count-marginal conditioning, cached scores, chunking, forks, EOS and actual
training/checkpoint round trips. Whole-song structure, control response and
runtime require the separate native comparison; these checks alone do not
establish a playable endpoint.
