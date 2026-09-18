# Oracle-time continuation: Mac runtime and parameter evidence

The full causal path supports disk-backed source admission, sequence training,
seed-only sampling, independent output replay, osu! export and durable recovery.
On the tested 24 GiB Apple M5, a 637-second full-backbone MPS stress run completed
29 updates with stable update-boundary active memory and no resource stop.
The first generated long charts nevertheless failed structure review. Runtime
success and usable generated arrangements are separate acceptance conditions.

The tested task supplies event times and at least 30 complete seed notes. It
does not consume audio or future actions. Source playback metadata is copied
only when exporting, so the generated difficulty can reference the original
mapset's audio, BPM and scroll-velocity points. Audio is not bundled. A generated
difficulty receives a new beatmap identity.

## Model and operating choices

The original baseline has **1,281,820 parameters**, about 4.89 MiB of FP32
parameter payload. It remains useful for contracts and regression measurements.
The Mac capacity profile has **19,976,776 parameters**, with the additional
capacity concentrated in temporal composition; its measurements follow below.

| Owner | Choice | Parameters | Rationale and limit |
| --- | --- | ---: | --- |
| History facts | Shared ordered-hand encoder, width 128 | 78,848 | Keep exact occupancy, clocks and role ordering; learned time inputs use bounded seconds plus asinh |
| Local summaries | Three layers, dilations 1/2/4, support 3/7/15 events | 515,072 | Ordered rhythm/action cells at bounded cost; event support is not a fixed millisecond interval |
| Note relations | 12 attacks and 4 releases per lane, at most 4 additional active heads; 4 attention heads | 236,036 | Retain lane recurrence and exact open-LN obligations without unbounded history |
| Temporal memory | Two layers, 4 heads, recent 512, archive groups 16, coarse capacity 64 | 416,520 | About 1,536 events of accessible learned history at full occupancy; exact open heads have their own retention |
| Joint-row head | Shared hand factors, coupling rank 16, legality over 256 rows | 35,344 | Simultaneous lane choices and coupled hands; no independent lane sampling |
| Training | FP32, chunk 128, effective/microbatch 2, clip norm 1 | — | Maximum tested differentiable chunk and microbatch; long targets retain their full duration through TBPTT |
| Optimizer | AdamW LR `1e-4`, weight decay `0.01` | — | Lower LR was more stable in the paired small-data calibration; weight-decay preference remains unresolved |
| Decoding | Temperature 1, top-p 1, beta 0 | — | Preserve all positive legal LN-close probabilities; 0.95 remains an explicit comparison |
| Source cache | 16 arrays and 256 MiB, with 4,096-row staging for an oversized source | — | Metadata handles do not pin parsed arrays after LRU eviction |
| Execution | CPU, one Torch thread; batched local/relation frontiers | — | CPU was faster on the measured real window; MPS remains an explicitly tested option |

### Temporal capacity profile

`oracle_time_train_mac.yaml` keeps facts/local/relation at width 128 and projects
both hand streams into a 512-wide, six-layer temporal module with eight attention
heads. The joint head reads that wider representation. A separate 64-wide MLP
computes time-edge biases. Increasing this pairwise MLP with content width is
expensive because it runs for every query/history pair, rather than once per row.

| Module | Parameters in the capacity profile |
| --- | ---: |
| History facts | 78,848 |
| Local summaries | 515,072 |
| Note relations | 236,552 |
| Temporal module, including input projection | 19,006,512 |
| Joint-row head | 139,792 |
| Total | **19,976,776** |

The profile uses FP32, microbatch 1 and Q=64. It retains recent512/coarse64 and
the exact relation/LN rules. Compared with Q=128, the shorter TBPTT interval
reduces current-writer gradient reach; it does not truncate prefix replay or
target duration. Coupling rank remains 16, already sufficient for the rank of
the 16-by-16 pair of hand-action states.

Each effective batch has eight windows: two independent chart selections with
four windows each, sorted within the chart for same-update prefix reuse.
These draws preserve expected average risk but are correlated. A paired
reuse measurement reduced computed prefix rows from 5,800 to 1,834 and elapsed time
from 24.25 to 9.37 seconds for the same eight windows. Sequence NLL changed by
0.0000153 over 324 target rows; gradient norm matched, and maximum post-step
parameter difference was about 1.1e-6. Reused carries never cross an update.

Three expanded candidates completed two full updates each on CPU and MPS.
Each update used a 1,600-row prefix and two 128-row targets, split into Q=64,
with a full temporal archive. These synthetic capacity inputs exercise the
actual backbone, loss, backward and AdamW; they are not quality evidence.

| Temporal width/layers | Total parameters | MPS peak active / driver | CPU peak RSS | Checkpoint bytes |
| --- | ---: | --- | ---: | ---: |
| 256 / 4 | 4,110,628 | 653,233,152 / 1,656,930,304 | 1,216,544,768 | 49,512,411 |
| 384 / 6 | 11,659,464 | 1,156,955,648 / 1,880,473,600 | 2,174,664,704 | 140,125,707 |
| 512 / 6 | 19,976,776 | 1,488,711,680 / 3,115,679,744 | 2,410,332,160 | 239,927,307 |

No arm added system swap, and MPS active returned to zero after unload.
The guard in this probe allowed 6 GiB driver; observed peaks stayed below the
packaged 4 GiB stop. The largest candidate's second update took 9.72 seconds
with one CPU thread and 11.18 seconds on MPS. A four-thread CPU repeat took
7.88 seconds and peaked at 1,795,670,016 RSS bytes. Other diagnostic work
overlapped some measurements; these timings guide the four-thread CPU preset,
not a hardware benchmark claim.

An earlier 256-wide/four-layer attempt expanded the edge-bias MLP as well and
used microbatch2/Q128. Its first forward caused 4,075,356,160 bytes of additional
system swap and was stopped by the guard. The supported expanded configuration
therefore requires microbatch1/Q64 and bias width at most 64. The larger
configuration still needs sustained real-corpus and generated-quality evidence.

The capacity preset uses LR 3e-5 with 20 linear warmup updates and weight decay
0.01. The paired 100/200-update comparison below favors this rate over 1e-4;
it does not establish an optimal LR or decay. Its approximately
229 MiB training checkpoint uses a 512 MiB hard cap and is published every
25 complete updates plus the final requested update. Five hundred updates
therefore write about 4.8 GB of periodic optimizer snapshots rather than about
120 GB at one publication per update, excluding initial/final weights and
filesystem write amplification. One snapshot is retained on disk at a time.

The base training objective defaults to joint-row NLL without extra marginal
weight. The Mac profile uses the measured `objective.lambda_struct=0.3`
configuration. The three exact marginals share that total weight;
their losses and logit-gradient magnitudes are recorded separately. This is a
tested configuration, not evidence that 0.3 is optimal.

At LR `1e-4`, weight decay `0.01` multiplies parameters by `0.999999` per AdamW
update before the gradient contribution. Over 1,000 steps that direct factor is
about `0.999`. Weight decay does not determine when replay facts or neural memory
are evicted.

## Calibration and time extrapolation

Five exploratory arms used the same eight training charts, initialization seed
17, sampling seed 17, and 40 updates. Each consumed 3,943 supervised target rows
and 52,402 prefix rows. Readouts used nine fixed windows on three held-out charts,
352 target rows in total. One short chart contributes a deterministic terminal
row. These exposures are too small to establish corpus quality or an optimal
optimizer.

| Time encoding | LR | Weight decay | Validation nats/row at update 40 |
| --- | ---: | ---: | ---: |
| Linear seconds | 0.0003 | 0.01 | 4.250602 |
| Linear seconds | 0.0001 | 0.01 | 3.996281 |
| Bounded seconds plus asinh | 0.0003 | 0.01 | 4.369458 |
| Bounded seconds plus asinh | 0.0001 | 0.01 | **3.731153** |
| Bounded seconds plus asinh | 0.0001 | 0.001 | 3.731235 |

The larger LR's original-time validation trajectory was 3.902869 at update 10,
5.105086 at update 20, and 4.250602 at update 40. The lower LR was more stable.
The 0.000082-nat difference between the two bounded-time weight-decay arms is
insufficient to prefer a different decay value. No confidence interval from
independent training seeds has been established.

A controlled age-only probe of the original encoder produced hidden norms
15.22, 60.46, 151.66 and 607.93 at chart ages 100, 400, 1,000 and 4,000 seconds.
The linear seconds channel amplified age beyond the short training exposure.
The new channel is `seconds/(1+abs(seconds))`, alongside `asinh(seconds)` and
the physical time basis. Exact replay still stores unmodified timestamps.
This representation has a distinct cache schema and
`oracle-time-continuation/weights-v2-bounded-time` format; earlier weights must
use their original runtime.

## Complete generation and quality failures

The longest admitted source has 26,976 rows, 40,841 notes and a 4,348.526-second
span. Three early-model rollouts completed its full skeleton after a 20-row,
30-note seed: two raw-model samples and one top-p 0.95 sample, with seeds
17/19/18 respectively. All **80,868 generated rows** passed the independent
legality, nonempty-row, time, occupancy and LN-closure replay. Every LN endpoint
in the export came from a generated close action.

Those eight-update linear-time outputs failed quality. One raw sample repeated
an identical row 12,224 times and held an LN for 1,838 seconds. The 0.95 sample
repeated a row 22,397 times and held an LN for 3,628 seconds. The source's maximum
LN was 13.2 seconds and its longest identical-row run was 10. Top-p removed
positive LN-close probability on 45,204 lane-row occasions in that sample.
Different random seeds and a weak model confound causal attribution to top-p;
the diagnostic establishes that truncation actually removed closure options.

These are failed examples, not useful long-form generations. A forced terminal
close repairs the final legal state but cannot repair the earlier sustained
occupation. Structural evaluation must therefore inspect continuation before
the terminal row and retain both raw and policy probabilities.

Three later bounded-time, 40-update small-slice rollouts completed another
80,868 generated rows on the same longest skeleton. They took 439.56, 588.53
and 523.70 seconds for raw/seed17, top-p0.95/seed18 and raw/seed19. Longest
identical-row runs were 4/6/5, and checkpoints were 4,671,002–4,677,078 bytes.
Reset RSS was 173,473,792 / 165,969,920 / 172,474,368 bytes. The raw seed17 run
still held one LN for 89.564 seconds across a 77.643-second event gap. The other
maximum LN durations were 15.113 and 7.833 seconds. Both representation and
training exposure changed from the eight-update failures; this does not isolate
the time encoding's effect or establish playability.

### Full-corpus baseline and current gold

The 1.28M baseline completed 500 updates on the entire eligible training
population, consuming 44,580 target rows and replaying 386,185 prefix rows.
Twenty-four fixed windows on eight held-out charts contain 1,126 target rows.
Their pooled NLL was 2.820800 nats/row at update200 and 2.680243 at update500.
The windows and chart allocation were unchanged between readouts.

The Beatmap Lens canonical reader found 171 current High-confidence human
observations on the selected candidate sources: 124 in train, 32 in validation
and 15 outside the inherited catalog. The current Foundation SHA-256 is
`15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97`.
Human-confirmed claims and their scopes are used as references; a machine-written
rationale attached to a confirmed claim is not represented as a human comment.

Eight fixed validation gold charts were generated fully at update500, using
only their minimum seeds and skeletons. Full-chart LN fractions ranged from
5.0% to 8.5%, while the references ranged from 0% to 71.8%. At update148, the
three inspected outputs all had roughly 22–25% LN notes. More training changed
the common output mix, but source-dependent organization remains weak.
Different legal arrangements need not match the source's exact LN fraction;
the diagnostic does not by itself decide quality. Beatmap Lens action renders
still showed incidental short holds and rapidly changing chord/return choices
in the inspected fast-flow scope, and weak persistence of the source examples'
LN roles and fixed-group exchanges. No structure-quality pass is recorded.

Beatmap Lens comparison uses current effective human observations with explicit
High confidence, complete source notes, entering holds and exact review scopes.
Its Foundation separately describes Jack, Stream, Trill, Tech and LN coordination.
Agreement with one tag, generic irregularity, overlap counts or note density
does not establish playability. The selected held-out human examples cover
flowing tap passages, chord recurrence, fixed-group exchanges and independent
LN retention/release roles. Generated charts must be inspected in those terms,
including entry, exit and larger transitions. Actual player testing and a
matched comparison with osuT5 or Mug-Diffusion have not been established.

### Expanded model: optimizer and free-running stability

Two 19,976,776-parameter arms use the same initialization and window draws,
effective batch8, two charts per update, four windows per chart, warmup20,
weight decay0.01 and marginal weight0.3. Only their main learning rates differ.
At update100 each has consumed 36,134 targets and replayed 273,010 prefix rows;
same-update reuse computes 139,793 of those prefix rows. At update200 the
1e-4 arm has consumed 74,354 targets and replayed 600,441 prefix rows, computing
297,035. Validation remains the same 24 windows and 1,126 rows.

| Main LR | Update | Validation nats/row |
| --- | ---: | ---: |
| 1e-4 | 100 | 3.951721 |
| 3e-5 | 100 | 3.288606 |
| 1e-4 | 200 | 3.031633 |
| 3e-5 | 200 | 2.883296 |

Full generation on the fixed LN-rich, tap-flow and chord-recurrence examples
exposes a larger issue than pooled likelihood. At 1e-4/update100, LN fractions
were 2.53% / 2.00% / 2.03%; at update200 they were 89.32% / 71.33% / 76.26%.
At 3e-5/update100 they were 9.86% / 10.26% / 8.14%, and at update200 they
were 17.98% / 12.26% / 9.89%. These are actual generated
closed LNs divided by generated-file note counts, including the given seed.
They describe a changing common output mix, not a required match to reference
ratios. The 1e-4/update100 tap example's complete inspected review context was
dominated by outer-column attacks and rapid same-lane returns, with occasional
short holds; source-like flowing organization was not sustained.

A fixed true-prefix diagnostic found expanded-model query RMS around one,
rather than an exploding hidden amplitude. In the LN-rich example at row640,
the 1e-4/update100 distribution had entropy1.33 nats and mean maximum probability
0.592, compared with entropy2.41 and maximum0.306 at 3e-5/update100. Lower rate
improves this particular calibration readout; the early evidence does not isolate
whether optimization noise, insufficient exposure or weak prefix conditioning
dominates free-running drift.

A paired 100-update seed-summary experiment restarted AdamW from the same
3e-5/update200 weights, with draw seed29, backbone LR3e-5 and the new zero
matrix alone at3e-4. Each arm consumed 36,609 further targets. Fixed validation
was 2.882688 without the feature and 2.877058 with it. Full raw/seed17 outputs
had LN fractions .4053/.3141/.3358 without it and .3663/.3252/.3190 with it.
This small difference does not demonstrate useful generation improvement; the
feature was removed, while the frozen experiment remains reproducible.

The 1e-4 arm subsequently reached update500 and 183,849 targets, with fixed
validation **2.759030 nats/row**. Its three full outputs had LN fractions
.0807/.0403/.0622. Complete visual inspection of the e67f review context
(32,071–39,845 ms, four Beatmap Lens pages) found sustained moving taps and a
clear eight-row `01↔23` jump alternation at 37,415–37,840 ms. Short holds and
local density transitions remain inconsistent. This establishes an identifiable
local organization in that sample, not a whole-chart playability verdict.
Changing the reference's pattern family or LN ratio is not itself a defect.

A full 20M/3e-5/update200 generation of the longest chart completed all 26,956
post-seed events. It used 1,277.97 seconds of measured execution, peaked at
475,267,072 RSS bytes, added no swap and retained a 45,507,463-byte checkpoint.
After unload RSS was 344,981,504 bytes. At event10,349/time1,679,374 ms, however,
it opened two LNs immediately before a 77,643 ms event gap and closed them at
the next event. This is an explicit organization failure despite legal output.
Resource timestamps span about14,980 seconds, including machine inactivity;
that span must not be described as uninterrupted active generation.

The revised information contract permits a bounded, unlabeled projection of
future times already present in the supplied skeleton. A 16-position MLP encodes
both offsets and successive gaps, adding110,848 parameters at width128. It has
no future actions, LN pairing, labels or sample-window coordinates. Fixed
skeleton/future-action isolation, mirror behavior, CPU/MPS gradients and
step/chunk/recovery parity are checked separately. The paired continuation and
gap-response measurements below have not established a useful long-gap effect.

### Larger temporal resource probe

A subsequent isolated probe used four CPU threads and the same two full
prefix1600/target128 updates on each device. Every model had six temporal layers,
eight heads, a64-wide bias MLP and the16-position timing module. All six runs
finished without additional swap; MPS active returned to zero after unload.

| Width | Parameters | Second CPU / MPS update, seconds | CPU peak RSS, bytes | MPS peak active / driver, bytes | AdamW checkpoint, bytes |
| --- | ---: | --- | ---: | --- | ---: |
| 512 | 20,087,624 | 6.68 / 7.38 | 1,803,911,168 | 1,492,016,640 / 2,935,865,344 | 241,262,123 |
| 768 | 43,800,136 | 7.58 / 7.69 | 2,932,998,144 | 2,113,002,240 / 3,100,262,400 | 525,799,979 |
| 1,024 | 76,949,832 | 8.42 / 8.16 | 3,752,116,224 | 2,623,603,968 / 4,133,175,296 | 923,584,043 |

The1024-wide capacity fits this24 GiB Mac under a1 GiB checkpoint limit and
6 GiB driver/8 GiB RSS guards. A100-update publication interval keeps checkpoint
write volume per update comparable to the20M profile's25-update interval.
This is a two-update capacity measurement, not a long-duration stability or
quality result. Temporal widening of the trained512-wide/u500 model preserved
pooled NLL over the24 fixed real windows/1,126 targets to within5.3e-10 nats/row.
The307,874,913-byte converted weights use zero-sum split noise to break gradient
symmetry. This verifies its initialization, not the benefit of later training.

A separate 77M CPU continuation completed 300 updates and 104,003 supervised
target rows. It computed 452,045 prefix rows after same-update reuse, compared
with 885,324 logical prefix rows. Update timers total 4,823.36 seconds. Sampled
peak RSS was 3,658,989,568 bytes, minimum system available memory was
9,061,548,032 bytes, and maximum swap growth was zero. Other training and
diagnostic processes overlapped parts of this run, so these times do not isolate
the throughput effect of model size. This extends the CPU resource evidence;
the two-update MPS probe remains the limit of the 77M MPS training evidence.

### Supervised coverage and broader validation

The 20M and widened 77M continuations inherit the same 500-update ancestor,
then train for 300 further updates with the same 104,003 target exposures.
Across the ancestor and either continuation, 287,852 supervised exposures
cover 249,676 distinct event rows from 1,460 charts and 1,276 song groups.
That is 2.17% of the 11,522,113 training event rows. It is a coverage fraction,
not an epoch count: group/chart/stratum sampling is nonuniform over rows.
Replaying a prefix without gradients does not add supervised target exposure.

The original validation uses 24 windows on eight charts from seven groups.
At 300 additional updates it favors 77M: 2.477381 nats/row, versus 2.536786
for 20M with timing and 2.533591 for 20M without timing. A broader fixed
comparison samples 128 distinct validation groups uniformly without replacement,
one chart per group, then a feasible context stratum, start and 1/4/16-second
horizon using seed 20260918. Its manifest is fixed before loading any weights.
All three models score the same 5,709 targets after 48,807 full-prefix rows.

| Weights | Pooled nats/row | Equal-group mean nats/row |
| --- | ---: | ---: |
| 20M, original 500 updates | 2.542761 | 2.607688 |
| 20M with timing, 300 additional updates | 2.317914 | 2.342508 |
| 77M with timing, 300 additional updates | 2.334990 | 2.347054 |

The continuations improve 103 and 105 of the 128 groups over the ancestor.
Paired group-bootstrap 95% intervals for pooled NLL changes are
[-0.291704, -0.163115] and [-0.268784, -0.148582], using 5,000 replicates
with seed 20260918. Continued exposure accompanies a fresh optimizer, lower
main LR, new draws and the timing module; this comparison does not isolate
which of those changes supplied the improvement.

For 77M minus 20M timing, the pooled change is +0.017076 with interval
[-0.005530, +0.040169]. The equal-group change is +0.004546 with interval
[-0.022249, +0.030873], and exactly 64 groups favor each model. The apparent
capacity advantage on seven groups therefore does not establish a broader
gain. These are diagnostic validation samples used for model selection, not
an untouched final test or generated-quality assessment. The limited coverage
and continuing likelihood gains motivate improving supervised exposure per
unit of compute before drawing a general conclusion about required capacity.

### Target horizon and training efficiency

A paired CPU probe starts two copies of the 20M timing-u300 model with fresh
AdamW and sampling seed 20260918. Both use four Torch threads, B8/cohort4,
microbatch1, Q64, LR3e-5/timing1e-3, warmup20, weight decay.01, clip1 and
structural weight.3. Only the sampled physical target horizons differ. The
groups, charts, starts and horizon indices match at each update; execution
order alternates to reduce order effects. Every target retains its complete
true prefix and real chart-terminal condition.

| Target horizons | Targets in 12 updates | Active update seconds | Targets/second, updates 3–12 |
| --- | ---: | ---: | ---: |
| 1/4/16 seconds | 4,708 | 99.42 | 48.19 |
| 4/16/64 seconds | 16,152 | 197.06 | 83.00 |

Both arms compute exactly 18,802 prefix rows. Extending targets amortizes that
prefix cost, increasing measured supervised throughput by **1.72x** after
omitting the first two updates. All twelve updates are still within LR warmup;
the omission concerns cache/setup cost. Annotation reading overlapped part of
the probe, and desktop load was not controlled as a hardware benchmark.

The combined process, retaining both models and optimizers, peaked at
2,711,371,776 RSS bytes with zero additional swap and at least
9,804,595,200 system-available bytes. All six modules received gradients and
measured parameter updates. Median unclipped gradient norms were 2.6769 and
8.5773; both arms clipped every update. The fixed loss denominator remained
1,024, so longer horizons change training risk, target exposure and summed
gradient accumulation. Similar update magnitudes after AdamW do not imply
the same gradient direction or effective objective. This probe establishes a
bounded efficiency gain; it does not establish likelihood or generated-quality
improvement by itself.

The sustained comparison starts from the same timing-u300 weights with fresh
AdamW and sampling seed 91. It retains the other settings above and pairs all
2,400 window choices over 300 updates per arm. Both compute 476,759 prefix rows.
The fixed 128-group validation gives:

| Target horizons | Supervised targets | Pooled NLL | Equal-group NLL | Retained update seconds |
| --- | ---: | ---: | ---: | ---: |
| 1/4/16 seconds | 115,215 | 2.308612 | 2.337321 | 2,399.35 |
| 4/16/64 seconds | 408,820 | 2.259720 | 2.301557 | 4,576.65 |

The pooled difference is −.048892 nats/row, with paired group-bootstrap 95%
interval [−.071915, −.027364]. The equal-group difference is −.035764, interval
[−.058921, −.013135]; 81 of 128 groups improve. Both pass the declared .02
regression guard. Retained updates give 1.860x target-row throughput, above the
1.5x efficiency criterion. This supports the longer-horizon policy on these
prediction and cost measures, while generated quality remains a separate gate.
Longer windows change both exposure and gradient aggregation, so the result
does not isolate supervised coverage as the cause.

The original process hit its two-hour bound during update 298. Its atomic
checkpoint at 275 restored model, optimizer and RNG state, and recomputed
updates 276–297 before completing 300. The observed loss, gradient norm and row
counts at 297 match their pre-interruption values exactly. Recovery and remaining
validation took 470.54 seconds. Retained update timings exclude discarded work
and do not measure total wall throughput; concurrent generation and diagnostics
also limit timing comparisons. Logged peak RSS is 2.749 GB, minimum available
memory 8.267 GB and swap growth zero.

### Shifted temporal stream: feasibility and limits

A separate cost probe compares the two-stream model with an autoregressive
token whose input at row i contains only earlier actions and permitted skeleton
times. Self-attention can include that token without observing its target.
This follows the shifted-input principle in the
[Transformer decoder](https://arxiv.org/html/1706.03762v7), while retaining the
existing bounded temporal archive. Its cache represents pre-row history rather
than post-row content; changing the representation does not preserve the old
model function or make its caches interchangeable.

One variant retains facts/local/relation inputs. Another uses history facts
and known timing directly, bypassing learned local/relation computation while
retaining exact replay and joint legality. Small deterministic checks cover
action-causal isolation, mirror equivariance, step/chunk output and parameter
gradient agreement, archive/cache agreement, parameter-version invalidation
and rejection of illegal commits. The tested future-action perturbation changes
earlier logits by exactly zero; maximum step/chunk gradient differences are
4.77e-7 and 3.58e-7. CPU/MPS output comparisons are also checked on that fixture.
These checks establish the tested execution properties, not representation
adequacy or generated quality.

The runtime comparison uses three fixed TRAIN charts, prefixes 512/1600 and
128 target rows per case: six actual forward/backward/AdamW updates per arm
and device, with Q64 and B1. Each arm starts from the same pinned 20M weights;
shifted variants require adaptation because that transfer changes the function.
Arm order rotates between cases. Other CPU corpus training overlaps the probe.

| Execution | Two streams, seconds | Shifted with local/relation, seconds | Shifted facts/time, seconds |
| --- | ---: | ---: | ---: |
| CPU, four threads | 35.96 | 32.39 | 24.82 |
| MPS | 44.35 | 41.77 | 29.32 |

The full-input shifted variant gives only 1.11x CPU/1.06x MPS total throughput
on these prefix-heavy cases, below the probe's 1.3x threshold. Its target/backward
cost falls, but prefix replay dominates. Bypassing learned local/relation gives
1.45x CPU/1.51x MPS and qualifies for a separate learning comparison. Those
representations may carry useful organization; speed is not grounds to remove
them from the working model without prediction and generation evidence.
The B1 case costs also do not substitute for B8/cohort4 training measurements.

The probe retains all three models and optimizers in one process, so its peak
resources are not single-model envelopes. CPU RSS peaks at 2,503,786,496 bytes.
MPS active/driver peak at 2,674,753,536/4,277,829,632 bytes, with no added swap.
The sampled driver peak is close to the 4 GiB guard; this is not a sustained MPS
training result. The facts/time path gives gradients to 19,332,928 parameters,
while all 20,087,624 parameter values remain allocated in the prototype.
No candidate weights or compatibility claim are published from this cost probe.

### Known-time lookahead and rare-gap exposure

A matched 20M continuation pair starts from the same 500-update weights, with
fresh AdamW, draw seed 41, backbone LR `3e-5`, weight decay `.01`, warmup 20 and
structural weight `.3`. The enabled arm adds 16 known skeleton offsets and an
independent timing LR `1e-3`. Shared parameters and initial predictions match.
At 100 additional updates both arms have consumed 35,441 target rows. Fixed
24-window validation is **2.5614803192** without lookahead and **2.5590380577**
with it. This small difference does not establish a useful timing effect.

Actual exposure explains a limitation: only 18 target rows precede gaps of at
least two seconds, and none precede gaps of eight seconds or more. On fixed
generated histories, compressing two following gaps from 84.735/77.643 seconds
to 125 ms leaves the control unchanged. The timing model's any-hold probability
changes `.146628 → .146268` and `.708764 → .708848`; neither demonstrates useful
long-gap avoidance. Earlier learned states are reused only where their permitted
future times are identical, and the final 16 prefix rows are recomputed.

The timing-u100 model also completed all 26,956 generated rows of the longest
training chart in **1,483.618 seconds**. Resource timestamps span 1,482.459
seconds, peak RSS is **492,994,560 bytes**, and additional swap is zero. Its
continuation checkpoint is **45,524,359 bytes**. The maximum LN nevertheless
reaches **89.356 seconds**, including an occupied 77.643-second empty gap. This
is a complete runtime test and a retained quality counterexample. The chart is
in the training allocation, so it cannot establish held-out generalization.

An optional time-only sampler now mixes the base population with gap strata
`[2,8)`, `[8,32)` and `[32,infinity)` seconds. Training has respectively
7,263/212/9 eligible boundaries in 3,933/187/7 charts and 1,459/110/7 song groups.
The exploratory mixture uses 25% gap draws, at most 32 context rows before each
boundary, and the existing full-prefix and half-open-horizon rules. It changes
the training risk explicitly; source actions and annotations do not select
starts. Fixed validation remains unchanged.

A separate gap validation manifest contains 17 contexts from held-out groups:
eight in each of the first two bands and the sole group available in the last
band. Before curriculum training, timing-u100 scores 2.5338106804 pooled
nats/row, 3.5715677054 mean boundary NLL and .1342577122 mean per-lane
post-occupancy Brier score. Some reference charts intentionally hold lanes
across long gaps, so always closing before a gap is not the evaluation target.
The single longest-gap held-out case is a substantial uncertainty limit.

After 50 updates and 16,844 targets with the 25% gap mixture, pooled gap-window
NLL becomes 2.523482, but boundary NLL worsens to 3.581272 and occupancy Brier
worsens to .148233. Fixed ordinary-window NLL also rises from 2.559038 to
2.563679. At the two fixed generated-history counterfactuals, actual versus
compressed-gap any-hold probabilities are .193355/.189723 and .815346/.814062.
The small response has the wrong direction for avoiding these particular
occupied gaps. Additional exposure alone has not demonstrated the required
mechanism in this bounded comparison.

### Timing transmission and one-row state interventions

A fixed-weight trace separates current-query timing from timing already stored
in the preceding 16 rows. It covers timing-u100, gap25-u50 and timing-u300 at
two positions of the longest TRAIN chart, each with source and generated
history: 12 cases in total. Manual facts/local/relation/temporal/head traces
match normal engine predictions before any intervention. Compressing a future
gap preserves earlier inputs only outside the encoder's lookahead reach.

At timing-u300, timing activation RMS is about 5.4–5.6% of facts RMS at these
queries. The local fusion reduces a current-query timing perturbation to about
one tenth of its input RMS, but the change remains measurable through the head.
Changing only that query's timing alters any-hold probability by at most
.000672. Recomputing the affected prefix gives a larger response in one
generated-history case: .161893 with the actual gap versus .148354 compressed.
This still does not establish useful gap avoidance. Activation scale alone
does not identify the cause; the probe measures only two positions of one chart.

Saved AdamW moments also do not support a disconnected timing module or
widespread epsilon-limited updates at update300. Reconstructing its adaptive
update before decay gives weight-update RMS of roughly 7.37e-5–1.67e-4 for the
first projection and 8.28e-5–1.56e-4 for the output projection across the two
capacities. No 20M timing weights have second-moment scale below AdamW epsilon;
about .0021% of the 77M first-projection weights do. These moment-based values
are distinct from newly measured backward gradients or causal usefulness.

A separate e67f probe compares a source tap row, one tap changed to LN_START,
and a tap-column rotation after the same full true prefix at two positions.
Four paired seeds, temperatures 1/.85 and 128 subsequent sampled rows produce
48 branches and 6,144 rows. All 16 forced LNs close within 1–7 rows, or
61–425 ms. Mean extra LN heads after the single-LN intervention are +4.125 at
temperature 1 and -1 at .85; the tap-rotation control gives +8.5 and 0.
This does not support a strong persistent LN attractor triggered by one LN
in the tested cases. It cannot rule out other history-dependent effects.

Reusing tap-history hidden states with the LN-history legality mask reproduces
much of the immediate increase in new-head probability. This diagnostic hybrid
evaluates logits outside their original historical support. It neither proves
a legality bug nor justifies removing legal action constraints.

### How much following-gap information exists in source actions?

A train-only conditional-frequency control estimates per-lane post-row
occupancy from pre-row occupancy, the preceding gap, and optionally the known
following gap. The same fixed bins span 0/31.25/62.5/125/250/500/1000/2000/
8000/32000/infinity milliseconds, with .5/.5 Bernoulli pseudocounts. Seed rows
and true terminal rows are excluded. Admission verifies source and row-array
identities; one chart is processed at a time. The control fits 11,219,325 train
rows and scores 1,827,293 validation rows from 435 groups without fitting on them.

| Available input | Pooled validation Brier | Equal-group mean Brier |
| --- | ---: | ---: |
| Pre-row occupancy | .06446172 | .07902937 |
| Occupancy and preceding gap | .06382779 | .07852137 |
| Occupancy, preceding and following gap | .06344166 | .07811639 |

Adding the following gap improves pooled Brier by .00038614; the paired
group-bootstrap 95% interval for its change is [-.00047127, -.00030112], with
320 of 435 groups improving. The equal-group change is -.00040498, interval
[-.00048923, -.00032417]. This is a small predictive signal in the control.
It is not a causal explanation of the neural model, and this control sees
far more supervised rows than the trained generator. Its population and task
also differ from the 17-boundary neural comparison above.

The relationship is not a monotone preference for empty long gaps. Per
previously closed lane, train LN-start rates are .0674 before 125–250 ms gaps
and .2524 before 2–8 second gaps. In validation, 573 of 890 boundaries before
2–8 second gaps and 10 of 26 before 8–32 second gaps retain post-row occupancy.
Such intervals often belong to intentional long holds. The nine train boundaries
of at least 32 seconds and the sole validation boundary have no post-row occupancy,
but that tail is too sparse for a robust empirical rule: the following-gap
table worsens its single validation case from .029505 to .070000 Brier because
some conditioning cells are unseen. Neither this control nor the gap curriculum
justifies forcing all long-gap boundaries closed.

### Temperature and complete-context review

With timing-u100 weights, temperature `.85`, raw top-p 1 and seeds 17/19, the
complete e67f chart contains respectively 145/1,800 and 35/1,874 LN heads/notes.
At temperature 1 and seed 17 it contains 660/1,747. These ratios describe output;
matching the source's zero LN count is not a quality criterion. All four pages
of the 32,071–39,845 ms context were inspected for both lower-temperature seeds.
Both contain sustained moving-tap organization and short directional cells;
seed 19 also has brief inner-lane jacks. Seed 17 switches into short LN movement
near 37.4 seconds. Player feel and whole-chart structure remain unaccepted.

The LN-oriented 5b69 chart at `.85`/seed 17 has 1,305/2,009 LN heads/notes.
All five pages of its 112,106–122,107 ms context and the reference were inspected.
Generated heads and releases form recurring staggered short-LN cells, with
occasional overlap. The reference has stronger symmetric pair repetition and
more sustained overlap; its context exits with three held lanes while the
generated context exits with none. This is recognizable local organization,
with weaker preservation of those particular reference motifs. It does not
establish a whole-chart pass or a result against osuT5 or Mug-Diffusion.

The 77M timing-u300 model also completes three full raw/seed17 generations.
Its 5b69/e67f/ecc4 outputs contain 2,150/1,604/1,756 notes and 1,282/884/818
LNs, with maximum LN durations 1,875/1,457/1,214 ms. All 26 pages of the three
source/generated review contexts were inspected together with exact actions
and entering holds. The canonical reader independently revalidated the eight
gold sources' 32 current High observations; their IDs and hashes were unchanged.

In generated 5b69, column0 holds from 116,377 to 117,315 ms while columns1/2
start together at 116,690 and release separately at 116,794/116,898. A column3
tap occurs at 116,794 while columns0/2 remain held. This is a concrete
independent hold/attack/release relationship, with less repeated symmetric
pair organization than the reference. Generated e67f uses short-LN movement
and release-only events where its reference sustains moving taps and chord
accents. Generated ecc4 replaces much of the reference's four-key recurrence
with staggered holds: column3 spans 147,928–148,500 ms while inner and opposite
columns enter and release at distinct times.

These local relationships show organization, but all three outputs also share
a short-LN character despite different reference episodes. Alternate styles
can be valid; source-tag equality is not the objective. The observations do
not settle longer-range differentiation, transitions, the long-gap stress
case or whole-chart playability. They are machine inspections, not new human
judgments or player tests.

The 20M base300 continuation is evaluated more broadly at temperature .85,
top-p 1, beta 0 and seeds 17/19 on all eight gold sources. All 16 complete
generations finish in 1,042.16 supervisor seconds. Inspection covers every gold
context: 58 generated pages, 16 additional source pages and 13 byte-verified
source pages already inspected. The eight canonical human documents remain
unchanged, preserving all 32 current reference observations.

There are real local structures. In 5b69/seed17, column 0 holds from 118,877 to
119,606 ms while columns 3, 1, 2 and 3 successively start and release separate
short holds. Seed19 instead sustains mainly moving taps in this scope and never
has two simultaneously occupied columns. Both ecc496 scopes are tap-only moving
flow; both e67f scopes contain substantial repeated short-LN cells, with seed19
changing from dense tap chords into that texture near 34.6 seconds. The source
tags calibrate these relationships; differing tags do not establish failure.

Full-chart recurrence locators expose a stronger quality failure outside those
gold scopes. All attacks below are in one zero-based column:

| Output | Column | Attack times, ms | Observed context |
| --- | ---: | --- | --- |
| 85058a/seed17 | 2 | 74634, 74658, 74682, 74706 | Four taps in 72 ms amid repeated chords; source release-only events and separate LN heads become attacks. |
| 871955/seed19 | 0 | 137342, 137360, 137386, 137407 | Four taps in 65 ms, with further shared-column chords immediately before and after. |
| ecc496/seed17 | 1 | 137500, 137535, 137571, 137607, 137643, 137678 | Six attacks in 178 ms; the source moves single notes among columns, while generation repeatedly attacks overlapping chords. |

Complete four-second source/generated contexts around all three locators were
inspected, preserving entering holds and LN endpoints. These rapid repeated
chords reject this checkpoint/decode combination as a whole-chart playable
result despite its coherent local examples. The locators are post hoc and do
not define universal timing cutoffs for Foundation labels. Other whole-chart
passages, player assessment and the prior 89-second-LN stress case remain open.

The long300 model is then checked on the three failure-source skeletons with
the same temperature and both seeds. All six complete charts finish in 247.04
seconds. The full gold contexts show better moving-tap organization in 871955,
and distinct Jack/moving-tap organization across ecc496's seeds. However, both
85058a outputs still contain four same-column attacks within 72/96 ms, and both
871955 outputs contain four within 65 ms. The complete matched failure contexts
confirm dense repeated chords. The likelihood improvement therefore does not
pass the playable-quality gate for this checkpoint/decode combination.

### Recurrence probability and query-time sensitivity

A read-only diagnostic examines four consecutive rows at each of the three
base300 failure locations. It replays each full source and fixed generated
prefix under both base300 and long300 weights, for 48 original-row cases.
The ten generated-history rows with an inspected-lane attack age at most 40 ms
have temperature.85 repeat-attack probability ranges .141–.650 and .305–.798,
with means .435/.554. Both models use the identical base300 generated histories
in this comparison. These are selected failure cases, not a global calibration
estimate; their repeat decisions are not uniformly negligible sample tails.

For each fixed state, inserting a 250-ms previous-event gap raises those
probabilities by .021–.100 and .026–.068 respectively. Future offsets remain
unchanged, corresponding to translating the upcoming schedule after the added
delay. Thus elapsed query clocks do affect the output. Multiplying only the
current-query future offsets by four instead changes the inspected lane's
probability by at most .001923/.005291. Earlier cached history still encodes its
original future-time inputs, so this result does not measure the total effect
of changing the time skeleton. Neither counterfactual is committed or exported.

With true source history at ecc496's moving burst, the models still expect
roughly 2.3–2.8 attacks per row where the source uses single attacks. This is
evidence against explaining the failure solely through generated-history drift.
Source/generated comparisons on the LN cases also change legal support and
cannot isolate an occupancy effect. The diagnostic preserves those limitations;
it does not establish a universal repetition cutoff or an architecture fix.

### Event-union support under generated occupancy

The exact event-union task couples timing to the generated LN history. If all
four lanes are closed, a nonempty legal row must contain at least one attack:
there is no open LN to release. A source release-only time therefore becomes
an obligatory attack opportunity when generation reaches it with closed lanes.
The contract does not force a particular column, multiple attacks, or the
earlier choice to avoid opening LNs.

A descriptive audit checks complete source and generated journals on the three
failure sources and seeds 17/19, excluding the provided seed. The continued
long300 control at 100 additional updates gives:

| Source | Source release-only rows | Such rows with generated occupancy fully closed, seeds 17/19 |
| --- | ---: | ---: |
| 85058a | 247 | 242 / 242 |
| 871955 | 584 | 451 / 400 |
| ecc496 | 4 | 4 / 4 |

On 871955, 117/92 of the 301/241 generated same-column attack gaps below 40 ms
end at source release-only times. Of these, 94/63 occur with generated lanes
fully closed. These overlapping descriptive counts locate the interaction;
they do not identify a causal fraction of quality failures. The earlier
long300 ecc496/seed17 output has 73 such fast reattacks, none at release-only
times. Event-role ambiguity consequently cannot explain all observed failures.

This exposes a task-design issue as well as a learning question: source-history
teacher forcing scores release-only targets from occupied states, whereas
free-running generation can reach their times with a different legal support.
Making an optional no-event output available would change that support, but the
current event-only training set supplies no positive whole-row absence targets.
Its benefit needs a separately defined candidate-time task and training data;
unmasking an untrained score is not a valid test. The V3 formulation already
permits internal row absence while requiring materialized chart rows to remain
nonempty; exact coverage of a supplied event union belongs to this research
baseline's narrower contract.

### Direct physical-clock readout: initial and paired validation

An optional shared-hand MLP adds physical query features directly to the
hand-pair unary scores. It uses pre-row elapsed clocks, prior actions and
occupancy, and permitted future offsets/gaps; the input and cache contracts are
described in [the continuation interface](oracle_time_continuation.md). This
shorter path targets the dense-time failures above. Its quality benefit is an
open empirical question, and the packaged default remains disabled.

Starting from long300, a width 128 readout with 16 future positions adds 152,080
parameters, for 20,239,704 total. Every existing parameter is copied and the
new final layer is zero-initialized. Both models score 2.259719950719731 NLL on
the fixed 128-group/5,709-target validation, with exactly zero per-case NLL
difference. This verifies the initial function on that set; it is not a new
prediction or generation improvement.

Selected local checks exercise the enabled path's action causality, mirror
equivariance, time-translation invariance, step/chunk/gradient agreement, Hydra
projection and actual runner consumption. The runner test distinguishes the
expected zero first-layer gradient at the first update from positive gradients
in both layers at the second. Training recovery retains the new optimizer group
and warmup; CPU and MPS generation recovery reproduce the uninterrupted row
journal and exported chart. The evidence does not cover CUDA or constitute a
whole-project readiness claim.

Starting both arms from long300 with fresh AdamW, the first 100 updates use
129,843 targets and 800 identical window draws each. On the fixed 128-group
validation, control/clock pooled NLL is 2.292336844/2.275240250. The paired
difference is −.0170966, with a 95% group-bootstrap interval of
[−.0280513, −.0065015]. Equal-group NLL is 2.329364052/2.311868484, with difference
−.0174956 and interval [−.0276027, −.0072840]; 72 of 128 groups improve. This
passes the declared prediction guard, while both clock aggregations remain
above the initialization's 2.259719951/2.301557082. The result is a paired
benefit, not a new overall best checkpoint.

On the same fixed base300 generated histories used by the recurrence diagnostic,
the ten selected rows with inspected-lane attack age at most 40 ms have mean
temperature .85 repeat probability .439021 under control100 and .234077 under
clock100. Their ranges are .294–.601 and .027–.486 respectively. This is evidence
of a learned conditional change on those failure states; it does not measure
whole-chart failure frequency or isolate the readout from the shared parameters
that also changed during training. The 48-case/144-distribution probe completes
in 65.23 seconds.

The six free-running clock100 outputs complete in 390.89 supervisor seconds.
The same three sources and two seeds have 45 adjacent same-column attack gaps
below 40 ms in total, versus 583 under control100. Their fastest four-attack
spans are:

| Source | Control100, seeds 17/19 | Clock100, seeds 17/19 |
| --- | ---: | ---: |
| 85058a | 72 / 145 ms | 209 / 169 ms |
| 871955 | 66 / 66 ms | 131 / 174 ms |
| ecc496 | 178 / 143 ms | 250 / 214 ms |

Inspection covers all 20 generated gold-context pages and the complete
source/generated contexts around the fixed failures and each new fastest-four
locator. The dense repeated-chord failures are substantially reduced, with
connected moving single/chord organization retained. Some contexts also contain
staggered holds; these outputs have not all collapsed to one fixed tap pattern.
Residual fast repetitions remain, including four column0 attacks over 131 ms
near 257320 ms in 871955/seed17. The counts and spans are descriptive locators,
not universal style or playability thresholds. This supports retaining clock100
for broader inspection, not a whole-chart quality pass.

The other five gold sources subsequently complete both seeds under the same
clock100 policy. All 38 additional generated pages are inspected, bringing
coverage to all 58 gold-context pages for eight sources and two seeds. The
source pages are byte-identical to the prior inspected references; all eight
canonical human documents remain unchanged. This extension has no newly paired
control100 outputs on the five additional sources.

In 5b69/seed17, overlapping holds, staggered releases and intervening attacks
persist across the context. These support a medium-confidence machine
hypothesis of prominent LN coordination with supporting moving-head structure.
Seed19 instead produces coherent tap flow. The other four sources largely
produce moving singles with occasional chords. ece738/seed19 has one isolated
short hold, which does not establish LN coordination. Source human tags cannot
be transferred to these alternate generated organizations. In 713ef9/seed17,
the 4,924 ms ending LN at 315465–320389 ms matches the source closing phrase's
endpoints on a different column; its duration is not a runaway-hold finding.

| Additional source | Fastest four same-column attacks, seeds 17/19 |
| --- | ---: |
| 5b69 | 313 / 365 ms |
| 713ef9 | 307 / 231 ms |
| 98357f | 353 / 383 ms |
| e67f | 303 / 203 ms |
| ece738 | 435 / 543 ms |

These ten outputs have seven adjacent same-column attack pairs below 40 ms in
total. Their extrema are descriptive locators and have not all been visually
inspected. The long TRAIN stress source also completes: one seed has 549 LNs,
one attack pair below 40 ms, a fastest four-attack span of 222 ms, and no hold
crossing the identified 84.735 s or 77.643 s event gap. Its longest LN spans
4335854–4349054 ms, exactly the source final LN's 13,200 ms duration and endpoints
on a different column; a terminal tap is also generated. This is one TRAIN
seed, not a held-out quality estimate. Its earlier 89-second failure used a
different checkpoint/policy, so the comparison does not isolate a cause.
Whole-chart transitions, stress-chart visual quality and player assessment
remain open. Clock100 is a useful baseline, with uneven LN organization and
no controlled difficulty/style guarantee.

Twelve further source/generated pages cover the three additional outputs whose
four-attack locator falls below 250 ms. 713ef9/seed19 repeats column0 at
196773/196850/196927/197004 ms (77 ms gaps); the long TRAIN output repeats
column2 at 606593/606667/606741/606815 ms (74 ms gaps). e67f/seed19 repeats
column2 at 98629/98710/98791/98832 ms (81/81/41 ms). The surrounding moving
structure persists, but local finger-speed demand increases relative to the
source's distribution across columns. These remain difficulty/organization
concerns. The inspection does not define a universal speed threshold or resolve
whole-chart playability.

### Linked onset and endpoint representation

A read-only audit of all 11,564 TRAIN charts finds 11,014,010 onset rows and
508,103 release-only rows among 11,522,113 event rows. The 2,548,286 LNs include
1,852,846 releases coincident with an onset. Of 349,291 rows starting multiple
LNs, 175,872 have different endpoints. In the original event union, 99.9483%
of endpoints lie within 16 following events, 99.9958% within 32, 99.9995% within
64, and all within 128. Maximum physical duration is 35,375 ms. These are TRAIN
observations, not justification for dropping a rare held-out target.

Removing release-only rows alone reduces sequence length by only 4.41%. The
more consequential representation question is whether each LN head should
predict its endpoint as an object decision. Its chosen endpoint would create a
release obligation; unused candidate times would not create materialized rows.
Supplying onset roles changes the conditional task and provides information
beyond the old untyped event union. True future LN pairings must remain targets;
teacher-forced endpoints of earlier object decisions must be replaced with
predicted endpoints during generation. Four planned holds must also preserve
the feasibility of every later required onset. Conditional endpoint accuracy
alone cannot validate free-running choreography.

A frozen-feature endpoint probe selects 128 TRAIN and 32 validation groups,
one LN-rich chart per group and up to 64 heads per chart: 7,010/1,794 examples.
A 156,389-parameter head scores every future event-union time. The generic arm
reads candidate timing and onset roles; the context arm additionally reads the
frozen clock100 query, chosen current row and earlier object endpoint decisions.
Current/future head endpoint labels are excluded from input. This is conditional
endpoint prediction with teacher-forced earlier plans, not a complete generator.

At 400 paired updates, equal-group NLL is .955568/.894912 for generic/context;
the difference's 95% group-bootstrap interval [−.155882, +.017844] crosses zero.
A declared exploratory extension to 1,600 updates reproduces both 400-step
checkpoints exactly before continuing on the same sample stream:

| Endpoint head, 1,600 updates | Equal-group NLL | Equal-group exact accuracy | Mean absolute MAP error |
| --- | ---: | ---: | ---: |
| Generic timing prior | .949939 | .739007 | 89.169 ms |
| History/context | .868903 | .740960 | 88.810 ms |

The NLL difference is −.081036, with 95% paired group-bootstrap interval
[−.161373, −.013947]; 24/32 groups improve. This meets the revised pilot gate,
but the same validation set informed the budget extension. Accuracy changes
little, and always selecting the next candidate already gives 74.30% per-head
accuracy. The selected validation slice contains only one endpoint beyond
16 events. The result motivates generated testing, without establishing
long-tail generalization or playability. Frozen feature extraction takes about
36.5 conservatively charged minutes; the paired 1,600-update fit takes 48.4
supervisor seconds. Exact replay of the inherited backbone dominates this
particular preparation cost.

### Typed-onset rollout and type-probability diagnosis

A subsequent pilot supplies required onset times H and optional endpoint
candidates R. Each LN head selects its complete future endpoint; unused R
positions do not create a row. The row backbone remains frozen clock100 and
has not been trained on this factorization. Onset roles and prior planned
endpoints constrain its support, while only the endpoint head reads them as
explicit features. This changes the conditional task relative to the original
nonempty event-union generator. The object-seed contract also permits known
endpoints for seed LNs, but all three selected source seeds actually contain
only TAPs, so no extra seed endpoint is supplied in these six outputs.

With row temperature .85, endpoint temperature1, row seed17 and independent
endpoint seed100020, the generic/context endpoint arms generate three complete
validation charts each. Every output passes onset coverage, endpoint membership,
occupancy, closure and independent export/reparse checks. Per-source LN counts
are6/6 for85058a,6/6 for871955 and13/67 for ecc496; clock100 under the old task
had145/102/23. Those source types and ratios are background, not targets that a
different valid arrangement must copy.

All44standard generated pages are inspected across fixed gold, each output's
own fastest-four locator and matched earlier failure contexts. Moving taps and
small chords remain, with reduced dense repeated-chord bursts in the first two
sources. New generic/context counts of adjacent same-column attacks below40ms
are0/0,10/10 and1/2; old clock seed17 counts are2,16,1. These remain descriptive
locators, not universal playability thresholds. Five outputs have no simultaneous
held pair anywhere. ecc496/context has1071ms with at least two held columns;
additional complete contexts around all three overlap episodes show definite
short LN-coordination insertions near72.1 and121.6seconds, while an isolated
staggered pair remains unresolved. Its longest1429ms hold accompanies other-column
taps without a second hold and is not LN coordination. These are machine
structural hypotheses under the frozen Foundation, not human or player validation.

A separate no-gradient diagnostic replays the three source histories and six
generated histories under the identical clock100 weights. At each post-seed H
position it compares the same logits under original legal support, mandatory
head support, full planned-release support, and full support at temperature .85.
The generated selected raw log probabilities reproduce exactly; policy log
probability errors stay below1.8e-15. The nine trajectories take382.03seconds,
peak sampled RSS469696512bytes, with no additional swap.

The table reports expected LN heads divided by expected total heads, summed
over required onsets. It is not the realized output ratio or a quality score.

| Source / history | Original support, T1 | Required head, T1 | Full typed support, T1 | Typed support, T.85 |
| --- | ---: | ---: | ---: | ---: |
| 85058a / true | 43.965% | 45.497% | 44.481% | 44.559% |
| 85058a / generic rollout | 1.038% | 1.041% | 1.045% | .466% |
| 85058a / context rollout | 1.044% | 1.046% | 1.052% | .468% |
| 871955 / true | 46.016% | 48.311% | 46.489% | 47.274% |
| 871955 / generic rollout | .597% | .598% | .605% | .230% |
| 871955 / context rollout | .597% | .598% | .604% | .229% |
| ecc496 / true | 3.880% | 3.937% | 3.915% | 2.489% |
| ecc496 / generic rollout | 1.448% | 1.456% | 1.539% | .865% |
| ecc496 / context rollout | 5.364% | 5.466% | 5.526% | 3.845% |

Full typed masking does not immediately suppress LN mass relative to requiring
heads on these generated histories. On the two LN-rich sources the generated
history has1.30–2.36% of the true-history typed LN fraction, and temperature .85
reduces its fraction to37.9–44.6% of its T1 value. The contrast remains within
the subgroup with all four lanes currently closed: true-history typed fractions
are33.30%/29.79%, versus about1.0%/.59% in generated histories. Thus current
occupancy alone does not describe the difference. These history comparisons
are observational: they do not identify why trajectories diverged, and do not
prove that matching source LN ratios is a valid objective.

The result favors investigating type initiation and continuation under the new
training/conditioning task before attributing TAP-dominated output to endpoint
legality. It does not establish that more oracle labels, scheduled sampling or
a new architecture will improve quality. Teacher-forcing original actions after
arbitrary generated LN decisions can also produce illegal targets; a future
training change must define its supervision rather than silently repair them.

### Seed-conditioned TRAIN type population

A complete TRAIN-only census covers11563eligible charts. Among4336charts whose
complete30-head seed has no LN,802(18.50%) have no LN in the suffix;1399(32.26%)
have more than10% LN heads in the suffix. Their chart-mean suffix LN fraction is
8.51%, pooled-head fraction7.92%, and equal-song-group mean10.31%. Where an LN
does occur, the median first LN is46onset rows after the seed, using a zero-based
offset.2675of4336have an LN within the first128suffix onsets. A TAP-only opening
therefore changes the type prior but does not determine the rest of the chart.

Onset-role information further stratifies that population. Among TAP-only seeds,
charts with5–15% release-only suffix candidates have23.14% mean LN fraction
(361charts); those above15% have40.54%(72charts). Zero release-only candidates
have2.25%(1364charts). These are chart population descriptions, not the exact
nonuniform window-training risk. Roles are additional oracle information in
the typed task; neither this association nor the census makes a desired
generated ratio compulsory. It does identify relevant supplied information
that the frozen row actor does not yet receive as explicit features.

The census takes6.38seconds, peak sampled RSS250609664bytes, without swap growth.
No validation/test payload or annotation changes are involved. Evidence owners
are `typed-onset-quality-v1`, `typed-onset-type-audit-v1` and
`seed-type-population-v1` under the existing M3 artifact root. Their readout
SHAs respectively begin `f3cdc61f`, `33d0a079` and `bf17d1e9`. The typed pilot
remains a refinement result; it has no whole-chart playable-quality acceptance.

## Resource measurements

Environment: Apple M5, 24 GiB unified memory, macOS 27.0, Python 3.10.20,
Torch 2.11.0, FP32. Model-backed commands use the explicit `mps` dependency
extra even when their execution device is CPU.

The 637.13-second MPS stress run used the full 128-wide, two-layer backbone,
two windows per update, a 1,600-row prefix and target lengths 1/17/128 in fixed
and varying order. Both temporal capacities and the relation bank were exercised.
All 29 forward/backward/optimizer cycles completed. This was the only active
model workload; source admission and canonical annotation reading overlapped
the early part, and other desktop applications remained open.

| Measurement | Observed value |
| --- | ---: |
| Maximum sampled MPS active allocation | 1,943,450,368 bytes (1.81 GiB) |
| Maximum sampled MPS driver allocation | 2,838,953,984 bytes (2.64 GiB) |
| Maximum process RSS | 715,096,064 bytes (682 MiB) |
| Active allocation at every completed update | 16,566,272 bytes |
| Driver range at completed updates | 312,770,560–483,344,384 bytes |
| RSS range at completed updates | 463,044,608–711,655,424 bytes |
| Additional system swap | Maximum 0; final change −25,165,824 bytes |
| After model/optimizer unload | Active 0; driver 139,411,456 bytes; RSS 514,752,512 bytes |

The final six RSS boundaries were approximately 522, 493, 449, 461, 447 and
491 MiB. The active state was constant; driver and RSS varied with shapes and
allocator reuse rather than increasing monotonically. RSS retention after
unload remains a process/allocator observation, not live model tensor storage.
These sampled measurements cover this workload and duration; they do not prove
indefinite execution or rule out between-check operator peaks.

A separate real-chart MPS recovery run used 500-update baseline weights on a
1,226-row held-out chart. It generated 1,210 rows after the seed, completed an
export, then restored its row512 checkpoint while retaining the later journal
tail to simulate an interrupted publication. Resume truncated that tail and
regenerated byte-identical row and osu! files. Total reference-plus-recovery
time was 191.15 seconds. Peak sampled active/driver/RSS were 247,385,088 /
301,711,360 / 1,225,621,504 bytes with zero additional swap; unload active was
zero. The 20M CPU training run overlapped this measurement.

The expanded 20M model also completed the same full-chart MPS recovery with
the new 512-row publication interval and 128-row ownership refresh. The complete
generation plus replay from row512 took 347.58 seconds; both row and osu! hashes
matched exactly. Peak sampled active/driver/RSS were 518,425,344 / 1,217,904,640 /
827,883,520 bytes, with zero additional swap. The final checkpoint was44,727,943
bytes, and unload returned active to zero and driver to38,944,768 bytes. Concurrent
CPU training limits throughput comparison with the baseline measurement.

This run includes the generation-state ownership fix. Before that fix, a live
MPS state on the same skeleton retained about 706.65 MB active at its end,
although restoring its compact checkpoint used only 8.68 MB. Re-copying
persistent carry at graph-free checkpoints reduced final active to 8,677,376
bytes; sampled checkpoint boundaries ranged from 5,546,752 to 17,836,032 bytes.
Reported tensor storage alone therefore understated live backend allocation.
The exact MPS retention mechanism has not been established. The regression
checks allocation after checkpoint copying as well as logical tensor payload.

Two preceding attempts supply failure evidence. A concurrent original-path run
stopped after two durable updates when system swap growth exceeded 1 GiB. A
batched run without pre-target cache release reached 4,878,417,920 driver bytes
and stopped at its first maximum forward pass. The successful policy releases
idle MPS allocations after no-grad prefill and after the completed update.
It preserves learned state and never calls allocator eviction per event.

On a real prefix512/target128 window, batched local/relation computation reduced
MPS prefix plus forward/backward time from 25.75 to 2.65 seconds; CPU took
1.69 seconds. Source admission overlapped this benchmark, so these are indicative
whole-path times rather than isolated accelerator throughput estimates. Batched
frontiers are checked against the online engine for logits, learned state,
relation/temporal visibility, future-target isolation and parameter gradients.

## Storage and recovery

A real Hydra CLI recovery test uses timing-u100, temperature `.85`, raw top-p 1,
seed 19 and the complete 1,226-row e67f validation chart. After observing row
640, the child process receives **SIGKILL**. A fresh CLI process resumes from
the durable checkpoint; its complete row journal and `.osu` bytes exactly match
an uninterrupted reference. The three process runs take 149.764 seconds in
total; the final checkpoint is 44,706,567 bytes. This tests actual process death,
in addition to the checkpoint-tail and serialization-interruption checks.

Export staging also sets SQLite's
[`max_page_count`](https://www.sqlite.org/pragma.html#pragma_max_page_count), so
database growth fails during insertion at the configured output cap. Header and
note bytes are checked before each write. Failed staging preserves an existing
published chart and releases its temporary directory. A clustered `(time,lane)`
primary key streams notes in start order even when an LN closes later. The
query-plan check verifies that export needs no additional temporary sorting
tree, leaving the capped database and output as the two large staging files.

Training retains one atomic checkpoint plus one final weights file. Generation
retains one atomic state checkpoint plus streamed output. The tested default
training checkpoint is approximately 15.55 MB, weights 5.18 MB, and full-memory
generation checkpoint about 4.66–4.68 MB for the original 1.28M baseline. Its learned tensor payload is about
3.6 MB. Length increases the output journal, not the memory capacities or the
checkpoint's history length.

For the expanded model, three complete 1,226–1,877-row CPU generations used
checkpoints up to44,700,487–45,878,983 bytes and peak RSS456,704,000–529,448,960
bytes. With the original128-row publication interval those runs wrote
403,689,030–632,707,310 checkpoint bytes each, despite retaining one snapshot.
The current512-row default separates writing from128-row resource checks;
MPS ownership refresh also follows resource boundaries. Unit comparisons check
identical row/export bytes with different checkpoint intervals. Full real-run
measurements using the new interval remain separately reported.

`checkpoint_max_bytes=134217728` caps both tensor payload preflight and actual
serialized writes. Every tensor is copied to owned contiguous CPU storage so a
small view cannot serialize its larger backing allocation. The temporary file
is fsynced before atomic replacement; interruption, serialization failure or
an oversized payload preserves the preceding checkpoint. Publication requires
the full staging allowance plus a 512 MiB free-disk reserve. It never deletes
the last checkpoint to make room.

A real subprocess crash test interrupted PyTorch serialization with SIGKILL
three times after writes passed64 KiB. The previous checkpoint remained byte
identical and readable after each crash. A fixed per-destination staging directory
and POSIX lock prevented accumulation; the following successful save reclaimed
the partial payload. Source admission and export share that staging policy.
This tests publication crash behavior with a small tensor payload; full-model
checkpoint sizes and continuation replay are measured separately.

Rows, training journals, resource journals and each export have per-file 2 GiB
limits; that is not a single aggregate run quota. Writes check available disk,
and SQLite staging has its own space/size checks. The admitted 11,564-chart
training cache occupied 199 MiB on disk in this run; one chart lacks the 30-note
seed, leaving 11,563 eligible training charts. Parsed arrays are subject to both
the count and byte LRU limits, independently of this persistent disk cache.

Recovery verifies pinned configuration/runtime identities and durable output
prefix digests before discarding an uncommitted tail. Training restores the
last complete update's AdamW state, sampler/device RNGs and exposure counters;
an interrupted draw is repeated from that boundary. Generation restores RNG,
exact state, learned state and the next skeleton index together. It validates
the persisted row prefix against restored occupancy before truncation.

## Reproduction and provenance

The exploratory evidence belongs to product baseline
`f679269b92e96efb5bd7989e748bd42cf069379e` plus the runtime snapshots below.
It is not a clean-commit causal comparison or an accepted research result.
Artifact paths are optional local run products; the observations above remain
readable without them.

All runs are under `artifacts/oracle-time-continuation/m3-20260917/`:

- `calibrate.py`, per-arm `calibration.json`, `runtime-snapshot/` and
  `bounded-runtime-snapshot/` preserve the optimizer/time comparisons.
- `long_rollouts.py`, `long-rollouts.json` and `long-p*-s*/` preserve the three
  early full-chart failures and their legal exported files.
- `parallel_bench.py` and `parallel-bench.json` preserve real-window timings.
- `mps_soak_boundaries.py` and `mps-soak-boundaries/{summary.json,resources.jsonl}`
  preserve the successful full-model MPS stress measurements. The two earlier
  failed attempts remain separately identified in their own directories.
- `mac-runtime-snapshot/sha256.json` has SHA-256
  `8a7e3a6f9bfc41a004f8bcb4e3d6db9393f3febd4d0e8a80dd0dd750c4100630`.
  It owns the broader corpus training runtime and the subsequent long-chart
  bounded-time comparison.
- `bounded-long-rollouts.log` and `bounded-long-p*-s*/` preserve the later three
  full longest-chart samples and reset/resource observations.
- `corpus-lr1e4-wd1e2/readout-{200,500}.json`, `corpus-validation-manifest.json`,
  `quality-u148/` and `quality-u500/` preserve corpus likelihood and gold readouts.
  The update500 weights SHA is
  `e3527ebcbbf6810d0abbe66d43d2747bcd0813c54c7cab83810e60975f1a64b4`.
  Its quality runtime manifest SHA is
  `177e0454c7c275c7b5fa02a5fef72eb83bab3b38b636569dc0162d353c574a61`.
- `capacity-probe-v2/`, `capacity-threads-4/` and `prefix-reuse-bench.json`
  preserve expanded-module and shared-prefix measurements. `capacity-probe/`
  retains the rejected memory configuration's resource log.
- `mps_real_recovery.py`, `mps-real-recovery-u500.json` and its output directory
  preserve the real-chart deterministic MPS recovery and state-copy measurements.
- `mac20m-runtime-snapshot/sha256.json` has SHA
  `2c7cf6086cdcae932c13f3390ecc7b8358beb40693a54895023a6ef2844ab4a3`.
  It owns the 20M corpus run, including grouped draws, warmup and periodic saves.
- `quality-mac20m-u100/`, `quality-mac20m-u200/`,
  `quality-mac20m-lr3e5-u100/` and `head-scale-probe.json` preserve expanded-model
  generation and query-scale diagnostics. `corpus-mac20m-lr3e5-wd1e2/` holds
  the paired lower-rate run.

Additional local evidence uses `capacity-probe-v3/`, `skeleton-time-init/`,
`skeleton-time-{control,timing}/`, and `widened-time-init/`. The runtime manifest
SHA for `skeleton-time-runtime-v3/sha256.json` is
`981dc3ef877c1a71bbf00b6116dcdd38022b1ef839704c2125dada9036375e27`;
its separately pinned experiment files manifest is
`37a1945ac27f271053045078294abc02055af06c25f3530635062add631fc79b`.
`skeleton-time-runtime-v4/sha256.json` is
`c23912b30bcc47d3f3e08e689eadd338f02eb8297842a44b12a7279dbc58eb17`.
It adds temporal widening, larger resource profiles and AdamW checkpoint
preflight. Current long runs use an idle-sleep assertion; previously recorded
wall spans and measured execution time remain distinct.

The broader validation and horizon probe use frozen runtime v7, whose
`skeleton-time-runtime-v7/sha256.json` SHA is
`efd670e285bef9a7987f8882e5baf09277549dbf334dc5883c9bfdab4b06a3a8`.
`coverage-evaluation-v1/manifest.json` pins the 128-group windows, and
`readout.json` has SHA
`eee487f981b982ae6bb344f3afd14f682e41c9aa8bb5436c8bc79178fb18460a`.
Its coverage, optimizer-moment and canonical-gold records preserve the distinct
measurement scopes. `horizon-throughput-v1/readout.json` has SHA
`4d9564a2ed4ddda187b2e9d3598af4fc4e4543752ce15ec1494c97306321d8f1`;
the adjacent update records contain matched draw identities, module gradients
and parameter changes. `quality-skeleton-large-u300/context-review.json`
retains all 26 inspected render/action hashes and exact qualitative witnesses.
`mechanism-probe-v1/{branch,timing}-readout.json` owns the complete one-row and
timing interventions; partial per-case files are not substitutes for completion.

The inherited catalog is
`artifacts/oracle-time-review/20260915-adfb1ee/catalog.json`, file SHA-256
`e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28`.
The pinned annotation split is
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.
The corpus adapter checks the complete metadata allocation before opening
selected payloads, preserves inherited groups, and verifies source SHA,
arrangement SHA and event count on admission. It never reads test payloads.

`conditional-timing-signal-v1/readout.json` preserves the full-population
frequency-control comparison, with SHA
`b9a0934ce462976dc8064fd8c73ef187f0abf787449d5244e5eba0c5a1ebcab9`.
`shifted-stream-probe-v1/validation.json` and `readout.json` preserve the
separate execution and cost evidence. The prototype source SHA is
`a243c655bd8b5140cbdaf41206d6331c0f4b614f61dac6c3ebec78f4549bbffd`;
the cost readout SHA is
`ea7a42db12e4550218c50addbc60da24cbe28aa465d105a1adc7169fef89b9af`.
These artifacts define exploratory comparisons, not a replacement runtime.

`exposure-continuation-v1/paired300-readout.json` records the sustained horizon
comparison, checkpoint recovery and timing exclusions; its SHA is
`ef29cd6c91a4183172ed2bc2971213ca54b8ffc13489dcebc6b7e465ca9e27de`.
The pinned base300/long300 weight SHAs are respectively
`7665fc0bcb284af871cd065ff3824facbc88aa157d99ae3bae7182e4b7ab3604` and
`7160e330d64d9224362c5c923144ac37fb1e7966f527c933b64c09ffc79eb9f4`.
`quality-exposure-base-u300-t085/context-review.json` records the complete gold
context inspection, unresolved judgments and three recurrence counterexamples,
with SHA `0e0663a151fbb4aaf54c708c88f42c3307369369630423a89a155ea52b01973b`.

`quality-exposure-long-u300-risk-t085/context-review.json` preserves the matched
failure inspection and exact reviewed-page coverage, SHA
`23b5b64fa8d3f69c826d9fd089ef19647c51db3d49f0bde3dfb7e40a4b4a00f9`.
`rapid-recurrence-probe-v1/readout.json` contains all original and counterfactual
query results, SHA
`954e947a37ec259b490f9df6cdb4eb532e85b374bfd10d725bab9a08f300263f`.

The direct-readout comparison freezes runtime v8, manifest SHA
`09469489b1d5b0d3f9e92890fa0ae6f58c3a9ecc2bf78cad977a865a3d16790e`.
`clock-readout-v1/initial-readout.json` records the initial parity check, SHA
`1aa1d21093d85d51a6881c4c1247b4a558053aad4e121c733fc85ea3dabbf4e6`.
Its initialized weights SHA is
`99a5ae9f72ba86ee3d477d47d238c2c8a950541af5f197fde2125c4ffe192750`;
`selected-checks.json` records the scoped validation commands and results.

`clock-readout-v1/support-audit.json` records the source-role and generated-state
audit on the long300 and continued-control outputs, SHA
`4b7770d2727f05c8a94df77cd5c2c9708c4dd6760413dfeeed70dff59f0f91eb`.
The 100-update pair readout SHA is
`3d1ae52ecb6dee4af15108db6bda28eaa40a3ea4190e66aa46ec4eecca7787fc`;
`clock-readout-v1/recurrence-probe-u100/readout.json` has SHA
`64146253ef1f190201aeb81a2892660e4a4941c1a2a21d6a03a032442f556e1a`.
`quality-clockreadout-clock-u100-t085/context-review.json` preserves the six
complete-context reviews and remaining concerns, SHA
`38edc3861d404e67b326b862934a0e655fad593410cec11ee51f29de5b7f0ff9`.

The additional gold review is
`quality-clockreadout-clock-u100-gold-rest-t085/context-review.json`, SHA
`d9210e6e61556b5c4b5782021e19a5322478857a014c7253fd6d93ad7b6d95f9`.
It records all 38 newly inspected generated pages, source-page verification and
the long TRAIN source's raw endpoint checks. The gold-rest and long generation
readout SHAs are respectively
`7c6b936ab25f3bd1fc8e92c157bb61ca78881550c4030ab041ceaca5df779c2a` and
`2344a168b1af638439e6be4f407b875168c7dd126b466a9957fc3c0160e9d4c4`.
`timing-representation-audit-v1/results/readout.json` records the all-TRAIN
representation counts, SHA
`7ef9903f8856074642d92cf73f142a293e72654872bf436e2de655ef29fd016a`.
The three additional recurrence contexts are recorded in
`clock-readout-v1/coverage-extrema/review.json`, SHA
`d718c5a99f630d97bea790b1b581fa56f4562cbe1f18f62ce00d447e2bc13d35`.
The linked-endpoint feature manifest has SHA
`6bd32392e4d2512ee518f8eb6f5580ad103ab1401c42c6f26dd77462821dcbe3`;
`linked-endpoint-head-v1/fit/readout.json` and
`linked-endpoint-head-v1/fit-u1600/readout.json` have SHAs
`225cbe80437e222b0ce9f496251c6df0c2333d535bf2e4c50459ac57f9f44888` and
`dd33fec02d3bf479644cfcf0b343b7a3af10d4ea687c09e20e9a764d6c4af40c`.
