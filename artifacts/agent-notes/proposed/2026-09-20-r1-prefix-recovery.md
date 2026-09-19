# Agent Note: Does source-prefix replacement restore R1 type balance?

Note ID: 2026-09-20-r1-prefix-recovery
Status: proposed
Kind: research
Created: 2026-09-20
Updated: 2026-09-20
Product revision: e525bda7883d1c165f63707a529780191f30f880
Scope: Frozen R1 4.5M history-replacement diagnostic on eight reused VAL groups
Related: 2026-09-20-r1-action-consequence; 2026-09-19-r1-additional-exposure

## Existing evidence and competing explanations

The matched consequence experiment fails its primary improvement gate despite
retaining independent LN organization in its eight prominent-control LN cores.
The much larger shared change from4M to4.5M deserves investigation before adding
more features. Existing full-suffix diagnostics already measure conditional type
probability on source-head lanes where both TAP and LN are legal.

Across28 equally weighted groups, that teacher-history probability increases
from.262043009 to.281137625 for continued none; the corresponding source-positive
fraction is.266065522 in both. Brier score improves.087357853 to.086732965 and
binary NLL improves.274348974 to.272431157. There are85114 eligible source-head
lanes and16697 positives. Native mean LN fraction increases.180447819 to.489053111.
Source suffix LN fraction is.265980798. The exact heads/cohort and all source
identities agree across checkpoints. Paired whole-group descriptive90% intervals
for the changes are[.013370707,.025232895] teacher probability and
[.227055974,.393430191] native LN fraction. These have different conditioning and
denominators; their contrast is not a causal exposure-bias estimate.

This read-only aggregation reuses existing logs and runs no model. Its owner is
`action-consequence-20260920-v1/calibration-drift-v1.json`, SHA
`7ed55f0a1d60e1293c4a5d1ad6a950e0871c0b8ed01fc01edd755c09c0a8074f`.
The source/query implementation has511 learned physical tokens plus exact global
row/note counts and clocks. It has no explicit desired star rating, LN fraction,
or persistent summary of the original30-head seed.

Live explanations differ. Additional learning or optimizer noise may shift a
local type prior; generated histories may amplify that shift; or the original
condition may leave global style intent underdetermined. A large native change
alone cannot choose among them. More data, averaging/annealing and explicit
external intent are separate possible interventions, not combined in this probe.

The closest diagnostic analogue is replacing generated prefixes with real ones
in [He et al.,2021](https://aclanthology.org/2021.emnlp-main.415/). That text study
finds limited incremental distortion and self-recovery; its result is a reason
to test rather than assume a feedback mechanism in mania. The distribution-
dependence framing also follows [Ross et al.,2011](https://proceedings.mlr.press/v15/ross11a.html).
DAgger's expert-action oracle is not available for arbitrary generated mania
states, so this experiment does not adopt its training algorithm. No novelty
claim follows from this adaptation.

## Experiment Card: r1-prefix-recovery-v1

Card ID: r1-prefix-recovery-v1. Revision:1. Accepted revision:none.
The user separately authorizes autonomous bounded implementation, execution and
local commits toward the active goal. This proposed diagnostic changes no weights,
training data, native sampler or legal support, and does not adopt a new task.

Hypothesis: replacing a generated prefix with the corresponding source prefix
reduces subsequent type-proportion divergence on the same supplied timing. A
positive result would prioritize conditioning/history-feedback investigation;
a negative result would weaken that explanation and prioritize local learning
or calibration. Neither result alone identifies a unique cause or a quality fix.

### Fixed identities and intervention

Use clean source e525bda7883d1c165f63707a529780191f30f880, a documentation-only
descendant of the evaluated9894761 runtime. Freeze the none4.5M checkpoint SHA
`2c452a954b57f0b412a0806dead0adffe11b227a22955f4fb9920d0fdf03d288`, trained at
4a98c0230549baa19ad860c3a13caf301e0f1f71. Reuse condition manifest SHA
`7c757dbe90fcdaecb667af38e6b17b98a192595542d6aa62fcb742ea9e543bc7` and native
readout SHA `b7e825369e6fe8065184406868bcf5f07c68593a24a841b192577d4b9cef1ac1`.
TEST stays unread. Source indices are6/9/11/19/23/27, selected from observed
native LN excess, and17/20 as LN-rich contrasts. These eight groups are reused
development examples, not prospective population confirmation.

For each group, anchor immediately before suffix onset floor(N/2), where N is
its count of post-seed H. Every chosen group has at least1072 suffix H, leaving
at least512 after the anchor. Use each existing none4.5M generated prefix from
seeds17/19/23. Pair it with the exact source prefix through the same R candidate.
The entire prefix state changes: learned raw history, occupancy, clocks and
counts. It is not an intervention on learned memory alone.

Only the original30-head seed retains supplied future LN ends. Every source-
prefix LN born afterward still has an unknown future end until its actual release
has occurred. Never treat the replacement prefix as a new complete-object seed.
Both paths retain the identical full R/H and true terminal, with native T=1 and
full support. Build learned buffers from the last511 raw physical events; this
construction cannot inspect any future action. For old prefix seed s, initialize
both future samplers with s+1700. The48 continuations cover512 required H each,
including their intervening R candidates, without imposing an artificial terminal
or closing unfinished LNs at the diagnostic boundary.

### Measurement and decision

The main measurement is absolute difference between generated and source LN-head
fractions over the first128 required H after the anchor. Source reference uses
those same H times and all heads there. Average the three paired seeds within
one group, then the eight groups equally. Record every result. The concurrent
free-prefix arm is the causal control; its fresh-RNG baseline value is measured
with the intervention, not borrowed from the original continuation.

A useful prefix-recovery signal requires at least25% lower mean absolute error
and a negative upper paired90% bootstrap interval for source-minus-generated-
prefix error. Resample the eight whole groups10000 times with seed1215. This is
a diagnostic decision threshold, not a playable-generation criterion. Report
both early and final128-H bins, all four consecutive128-H bins, raw head/LN
counts and the two LN-rich groups separately. Report how much initial learned
history remains at each H; the last128-H bin need not be entirely beyond the
511-physical-row field. Global counts and surviving held commitments can remain
in exact state even after learned history expires.

Require all48 windows to cover exactly the declared H, every mechanical transition
to validate, no finite-probability/support discrepancy and all below40ms head/head
and release/head events to remain visible as separate diagnostics. The LN-rich
contrasts must be reported even if they worsen. Matching source type proportion
is not semantic correctness, desired difficulty or proof of playability.

Before the main run, verify each reconstructed native-prefix first query against
its frozen original chosen log probability and each source-prefix first query
against dense teacher recomputation (maximum finite-log-probability error3e-6,
identical support). Inspect exact anchor occupancy and initial transitions; no
new human/gold annotation or generated-quality claim is part of this diagnostic.

### Execution bounds

Run one artifact-owned driver from the clean product worktree using
`uv run --offline --python 3.10 --extra mps python` on the M5, CPU1, Torch2.11.
Freeze driver, inputs and resolved plan digests before execution. Fresh owner:
`artifacts/bounded-typed-continuation/prefix-recovery-20260920-v1/`; main outputs
in its fresh `run-v1/`. No overwrite or automatic resume. Bound the whole run to
1800seconds,6GiB RSS/footprint,128MiB swap growth and256MiB outputs. Stop on any
parity, nonfinite, mechanical or resource failure; preserve an explicit failure
receipt. Save initial/final raw snapshots and every generated decision for audit.
No training, external compute, network or source-label changes are involved.

Interpretation remains exploratory and REFINE. A successful recovery signal
motivates a distinct conditioning/history experiment; it does not justify using
source refresh at inference. A weak effect, a worsening LN-rich contrast or an
effect explained by occupancy reset remains visible. Any new learning or decoding
change requires its own declared design.


## Execution freeze

The artifact driver compiles and the product worktree remains clean at the
declared e525bda revision. No product code is changed. Factory checks preserve
unknown R1 ends, compare complete source exact state, match24 frozen native
chosen probabilities and compare8 dense/cached source distributions before
sampling any diagnostic window. Per-window initial/final raw snapshots and
decision journals remain bound to the original timing condition.

Driver SHA: `b074f4423e1fe106d70d2d1196ab7542e4219f185c20c3967300e1bce70f3d22`.
Execution-plan SHA: `c41185500d9ea3e705d2dff1a027ba0d02de36c105e3afd90fd98967747f85bd`.
Command: `uv run --offline --python 3.10 --extra mps python artifacts/bounded-typed-continuation/prefix-recovery-20260920-v1/run_probe.py`.
All48 windows are pending. The Note remains proposed, accepted revision none.
