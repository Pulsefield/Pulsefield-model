# R1 restoration, transfer and long-form stability

The current audio model inherits selected R1 weights. It does not preserve the
complete policy released as `r1-restored-6.75m`. The actual restored lineage
shows improvements in several fixed-condition diagnostics, while the transfer
omits modules responsible for part of that behavior. Neither observation gives
a percentage attribution for failures of the new audio-to-chart system.

## What actually transfers

The released checkpoint has SHA-256
`4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70` and 3,084,432
parameters. The [transfer function](../../src/ensomi_model/research/joint_audio_continuation/model.py)
copies 2,444,688 parameters from temporal, exact, fuse, joint, head-routing and
release-routing modules. It omits 523,776 parameters in `seed_residual`,
`long_memory` and `row_consequence`, and discards 115,968 columns' worth of
exact-projection weights. The projection input changes from 1435 to 529 features.
Parameter fractions are not fractions of causal responsibility.

The last restoration stage trains only `frontier2` row consequence. Its parent
weights and Adam states are frozen. The actual 6.5M release checkpoint, SHA
`8000beb0dee81b92f0252fcec876823afd60fe33a182fb344ae0bed02d03f991`, and released
6.75M checkpoint were each passed through the current transfer function with
identical construction seed and global/bounded configuration. All 227 initial
state tensors are bit-identical; both have SHA
`1b9ef122bd7ac55128ac3733e11cf07c31afb82780845cb636f1311e7bbbd3eb` under the
recorded name-and-tensor hash procedure.

Consequently, that final fine-tune has zero direct parameter effect on the
current joint initialization. This does **not** make its removal irrelevant:
the correction it supplied in the full R1 policy is absent. Calling the new
decoder a restored R1 backend would incorrectly imply behavioral continuity.
Head and release routing do transfer; their effects after joint fitting remain
possible contributors. Earlier seed/memory stages also changed the shared
backbone while those conditions were available.

R1 predicts actions on supplied source R/H timing with an original observed
seed. The audio model learns event time as well as actions, starts at BOS and
permits every legal nonempty row. These condition/support changes compound the
module omissions. Source-conditional R1 likelihood and completion do not measure
the new system's startup, timing errors or behavior under its own full history.

## Actual restored readouts, not the earlier historical candidates

This audit re-reads the checkpoints, ledger and complete native rows from the
actual restoration that produced the released bytes. Training source is
`cdbc6870d9e6471a5dcb75d8a77f55c30544bf78`. Earlier historical 48-output results
belong to different checkpoint identities and cannot be substituted for these
readouts.

The restoration screen has eight VAL songs at seeds 17/23: 16 complete native
outputs per stage, all using fixed source timing and genuine initial seeds.
Suffix durations are 181.727–277.537 seconds. Likelihood uses 24 fixed VAL
windows totaling 6144 required onsets, alongside an equally sized TRAIN readout.
The following fractions average each native chart equally. Reference suffix
LN fraction averages 18.60%; its difference from generated fractions is a
composition diagnostic, not a definition of the unique correct arrangement.

| Stage | VAL NLL/onset | Mean native LN fraction | Mean absolute LN-fraction difference from reference | Head→head `<30 ms` | Release→head `<30 ms` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain 4.5M | 1.7715 | 43.12% | 31.30 pp | 6 | 306 |
| Seed 5M | 1.7652 | 31.81% | 17.39 pp | 3 | 87 |
| Memory 6M | 1.7273 | 26.35% | 14.18 pp | 2 | 35 |
| Head routing 6.25M | 1.7311 | 27.42% | 13.33 pp | 2 | 68 |
| Release routing 6.5M | 1.7290 | 27.67% | 14.72 pp | 2 | 67 |
| Response 6.75M | 1.7386 | 19.25% | 7.24 pp | 5 | 7 |

Short-gap columns count individual suffix heads and use strict less-than.
A head can meet both conditions, so the columns must not be added as disjoint
events. Supplied seed heads are excluded, while a new head can be compared with
its observed predecessor. All input/row/checkpoint hashes were verified.

The plain model's mean LN fraction rises from 27.23% to 53.96% across its first
and last required-onset quarters; the references rise from 13.78% to 21.11%.
The final model's corresponding means are 19.13% and 16.67%. These aggregate
observations are consistent with improvements in the fixed-condition behavior.
They do not prove preservation of every LN, Tech or repetition relation.

Local corrections can change later state allocation despite frozen inherited
weights. The longest unchanged three-hold run in this screen rises from 2 to
31 required onsets after head routing, then returns to 2 after release routing.
This is a locator for inspection, not an automatic BAD label. It illustrates
why fixed-state conditional-preservation properties do not guarantee the same
sampled long-form organization.

## Targets and evaluation limits

Ordinary source supervision uses a legitimate conditional likelihood over the
provided R/H task. This audit has not found mislabeled source actions or an
arithmetic defect in that loss. A lower aggregate likelihood can still obscure
composition changes or failures on generated histories.

The three recovery stages add machine-defined preferences on selected native
TRAIN states. Head/release correction targets persistent allocation witnesses.
The [response objective](../../src/ensomi_model/research/bounded_typed_continuation/response.py)
counts a head when either its same-lane prior-head gap or prior-release gap is
under 30 ms. Its future cost uses an optimistic current-plus-two-required-onset
continuation. It preserves head/LN counts when possible and anchors source
predictions, but this remains a short-horizon surrogate rather than a complete
playability objective.

The actual final-stage reduction predominantly concerns release→head gaps:
67→7 below 30 ms, while head→head changes 2→5. This is not evidence that all
rapid repeated presses improved. A release gap and a repeated-press gap describe
different player actions; a universal merged threshold cannot decide whether
an LN articulation is inappropriate. Semantic relationships and execution
context still require inspection.

The [restoration worker](../../src/ensomi_model/research/r1_restore/run.py)
advances stages when training and mechanical native execution finish. Its final
ledger explicitly says `requires_longform_ln_tap_and_local_response_review`.
That automated completion state is not a playability acceptance gate. The
three-to-five-minute readout also cannot establish behavior on arbitrary longer
audio, generated startup seeds or a new timing distribution. Separate manual
reviews, if any, need their own source/checkpoint/scope evidence; they are not
implied by the ledger.

## Attribution and next comparison

The evidence supports separating three questions: remaining weaknesses of the
full R1 policy, loss of its conditions/correction modules during transfer, and
new dynamics introduced by joint audio/timing training. Their effects interact;
there is no defensible additive percentage decomposition from the existing
experiments. The last-stage transfer identity is the narrow exception above.

A useful next diagnostic keeps original R/H and physical seeds fixed while
removing only the persistent seed/memory residual outputs from a pinned parent.
This measures dependence on those neural conditions before changing the task.
A later matched joint-training comparison can use plain 4.5M, memory 6M and
release 6.5M initialization with identical audio, sampling and learning-rate
assignment. Release 6.5M and response 6.75M need not be separate transfer arms.
Such comparisons measure specific interventions, not universal blame shares.

Local evidence owner:
`artifacts/joint-audio/20260924-r1-lineage-audit-v1`; read-only audit result SHA:
`09b58360f7f08b2c75002d17e3fcab04d8921e9e31d76d4a28e20bf99ae297a5`.
The audit does not refit models, alter original artifacts or add human judgments.
