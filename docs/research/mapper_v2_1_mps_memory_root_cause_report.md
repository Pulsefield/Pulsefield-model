# Root-Cause Analysis: Persistent MPS Memory Growth in Mapper v2.1

- Analysis date: 2026-08-03
- Status: MPS allocator growth explained in the tested configuration;
  process-footprint growth and production mitigation remain unresolved
- Affected run: Mapper v2.1 training from step 44,000 to 44,250
- Reproduction baseline: checkpoint step 44,250
- Source revision: `ef9677b1fdcdd373b29e666417193a3576e0e952`
- Runtime: PyTorch 2.11.0 on MPS
- Host: Apple M5, 24 GiB unified memory, macOS 26.6

## Executive summary

The original memory-pressure symptom contains two distinct effects.

The first effect is growth in the Metal driver counter. In the tested
configuration, variable sequence lengths produce square attention buffers
that cross a PyTorch 2.11 allocator boundary. On the observed unified-memory,
no-memory-pressure allocator branch, requests above that boundary use the
1 GiB heap class. The order of earlier allocations then determines whether a
request can reuse an existing free block or requires another heap. At a
synchronized endpoint, retained non-active heap capacity explained `96.40%`
of the difference between active PyTorch MPS memory and the Metal driver
counter. `torch.mps.empty_cache()` released `2,528 MiB` of fully free heaps.

The second effect is additional process-footprint growth that the observed MPS
allocator and graphics counters do not explain. In two counterbalanced
comparisons, dynamic execution added only `24 MiB` of heap capacity and
`4.4-13.2 MiB` of graphics footprint relative to fixed execution, but added
`328.6-330.6 MiB` of task-internal footprint and `338.5-344.1 MiB` of physical
footprint. These macOS ledgers overlap, so their differences cannot be
subtracted to assign byte ownership. The available public counters do not
identify the owner of this remaining growth.

## Incident scope and reproduction environment

The continuation from step 44,000 to 44,250 reduced system available memory by
about `3.28 GiB` and raised swap to about `2.2 GiB`. Active MPS memory, the
driver counter, RSS, and system compression followed different trajectories.
System memory returned when the process exited.

All reproduction runs used the following configuration unless stated
otherwise:

| Item | Value |
|---|---|
| Checkpoint | step 44,250; SHA-256 `aa956847234315251a6553305d0c792615ca74a12e8690f56c7c7c32affd5ecd` |
| Python | 3.10.20 |
| PyTorch | 2.11.0 |
| Device | MPS |
| Batch size | 2 |
| Global-attention heads | 8 |
| Model width | 384 |
| `global_stride` | 16 |
| Data-loader workers | 0 |

The reproductions changed inputs, allocation order, synchronization, cleanup
placement, or instrumentation. They did not change the production model,
sampler, allocator policy, checkpoint, or Hydra configuration.

## Memory accounting model

Three counters separate active tensor storage from allocator capacity and the
broader Metal total:

- `A` is `torch.mps.current_allocated_memory()`, the active blocks known to the
  PyTorch MPS allocator.
- `H` is the sum of live MPS allocator heap capacities reconstructed from
  `PYTORCH_DEBUG_MPS_ALLOCATOR=31` events.
- `D` is `torch.mps.driver_allocated_memory()`, the Metal device counter.

The reconstruction of `H` tracks heap identity, adds the declared capacity on
heap creation, and removes that capacity on heap release. Block allocation and
return events change occupancy inside a heap but not `H`. Repeated runs checked
the reconstructed total against the emitted heap inventory and against exact
1 GiB changes in `D` when a traced heap was created or released.

At one aligned phase boundary:

```text
D - A = (H - A) + (D - H)
```

`H-A` contains allocator heap capacity not represented by active blocks. It
includes fully free heaps that can be discarded and slack inside heaps that
remain pinned by live blocks. `D-H` is the driver-counter residual outside the
reconstructed PyTorch allocator heaps. It is a counter difference, not proof
of a particular object type or physical residency.

Only phase-aligned values support this decomposition. Run-wide peaks can occur
at different moments, so differences between independent peak values do not
identify ownership.

The process measurements come from Darwin `TASK_VM_INFO`:

| Report label | `TASK_VM_INFO` field | Meaning used in this report |
|---|---|---|
| Physical footprint | `phys_footprint` | The task's macOS physical-footprint charge; it is not a direct resident-DRAM measurement |
| Task internal | `internal` | Memory charged to the task's internal ledger |
| Task compressed | `compressed` | Compressed memory charged to the task |
| RSS | `resident_size` | The task's resident-set-size view |
| Graphics footprint | `ledger_tag_graphics_footprint` | Memory charged to the task's graphics-footprint ledger |

Each field is a different view of process memory, and several views can include
the same underlying pages. They also overlap with the MPS counters. This report
therefore compares changes within one ledger at aligned phases and does not add
or subtract different ledgers to assign ownership. Unless a value is labeled
as a run-wide peak, measurements used for attribution are endpoint samples
collected after MPS synchronization. Raw byte counts are converted to MiB or
GiB for presentation.

Memory also has several release boundaries. Autograd can stop needing a saved
tensor before Python releases its last reference. The allocator can recover a
block before Metal releases its resource, and Metal can release a resource
before the GPU finishes all submitted work. Garbage collection,
`torch.mps.synchronize()`, and `torch.mps.empty_cache()` test different parts
of that lifetime.

## Investigation method

The investigation started with low-overhead isolation runs and added native
instrumentation only after the symptom had been reduced to variable-shape MPS
execution. The main comparisons were:

1. one fixed batch versus variable batches, to distinguish cross-step
   retention from shape-dependent behavior;
2. cache-enabled, cache-disabled, and loader-only runs, to separate the model
   from data loading and collation;
3. a one-frame padding change around the suspected allocator boundary, to
   test the effect of request size without changing valid content;
4. allocator-only replays with the same allocation multiset in different
   orders, to isolate fragmentation and reuse from model execution;
5. phase-aligned cleanup steps, to determine which memory layer released each
   group of bytes; and
6. cold/warm and counterbalanced fixed/dynamic comparisons, to separate
   first-seen runtime state from continuing process growth.

Each causal claim below requires a repeatable intervention on the suspected
variable and a matching change in the relevant counter. Results from different
phases or overlapping ledgers are reported as correlations, not byte ownership.

## Investigation evidence

### 1. Screening isolates variable-shape MPS execution

The first runs tested whether the symptom required model execution, variable
input shapes, data caching, optimizer updates, or asynchronous dispatch.

| Candidate cause | Isolation test | Result | Conclusion in the tested scope |
|---|---|---|---|
| Cross-step Python tensor or Autograd retention | Repeat one batch for 80 steps | Active MPS grew only `6.7 MiB`; no live tensor with `grad_fn` remained after a step | Rejected as a sufficient cause |
| Dataset LRUs | Dynamic replay with cache 8 versus cache 0 | Both reached about `4,017.5 MiB` driver peak | Rejected as a sufficient cause |
| Loader and collation | Consume the same 80 cache-free batches without MPS compute | RSS fell from `1,633.0` to `1,347.3 MiB`; MPS stayed near zero | Rejected as a sufficient cause |
| Optimizer update | Dynamic 40-step no-update replay | Multi-GiB task-footprint growth remained | Update is not required |
| Asynchronous overlap | Serialized dispatch and a replay with only phase-boundary measurements | The same 1 GiB heap event and the same 20-step topology remained | Overlap is not required |

The fixed and dynamic 80-step comparison captured the core split:

| Metric | Fixed batch | Dynamic, cache 8 | Dynamic, cache 0 |
|---|---:|---:|---:|
| Active MPS net change | +6.7 MiB | +4.5 MiB | +4.5 MiB |
| Driver peak | about 759 MiB | 4,017.5 MiB | 4,017.5 MiB |
| RSS net change | -303.6 MiB | +2,878.7 MiB | +3,020.9 MiB |

The driver counter repeatedly rose without a matching increase in active tensor
storage. Periodic cleanup released large amounts only in dynamic replay.
Dynamic MPS replay reproduced the continuing growth; the tested fixed replay
and loader-only arms did not. This rejects accumulation of one Python-visible
tensor set per step as a sufficient explanation without claiming that dynamic
shape is the only possible trigger.

### 2. Sequence geometry crosses an allocator boundary

Mapper pools physical frames as:

```text
G = ceil(F / 16)
```

The global encoder self-attends over `[B,G,D]`. With `B=2`, `H=8`, and
float32, one square attention storage uses:

```text
square_bytes = 2 * 8 * G * G * 4 = 64G^2
```

On this Apple unified-memory device, the PyTorch 2.11 private allocator pool
used a 10 MiB large-buffer threshold and selected the 1 GiB heap class in the
observed no-memory-pressure branch. For this `B=2`, `H=8`, float32
configuration, the verified boundary is exact:

| Physical frames | G | Square storage | Observed new heap class |
|---:|---:|---:|---:|
| 6,464 | 404 | 9.961914 MiB | 32 MiB |
| 6,465 | 405 | 10.011292 MiB | 1,024 MiB |

The causal chain is:

```text
variable F
  -> G = ceil(F / 16)
  -> [2,8,G,G] storage uses 64G^2 bytes
  -> G=405 crosses the 10 MiB allocator boundary
  -> the observed PyTorch 2.11 branch selects the 1 GiB heap class
  -> prior allocation order determines whether a suitable block exists
  -> new heaps remain cached after the active tensors die
  -> H-A dominates the synchronized D-A residual
```

#### One-frame boundary test

The controlled boundary test reused the same source, valid content, checkpoint,
and batch. It changed only masked padded length from 6,464 to 6,465 frames.
Each arm ran an evaluation-mode forward and backward pass in a fresh process
and was repeated once.

| Forced frames | G | Loss | Peak A | Peak H | Peak D | Heap inventory |
|---:|---:|---:|---:|---:|---:|---|
| 6,464 | 404 | 0.916081786 | 474.092 MiB | 560 MiB | 582.828 MiB | `14x8 + 14x32 MiB` |
| 6,465 | 405 | 0.916080594 | 481.562 MiB | 1,296 MiB | 1,320.859 MiB | `14x8 + 5x32 + 1x1024 MiB` |

The extra masked frame changed loss by only `1.19e-6`, while `H` rose exactly
`736 MiB` and `D` rose `738.031 MiB` in both repeats. The heap transfer was
`1,024 - 9*32 = 736 MiB`.

The mapped storage was FP32 `[2,8,G,G]` in
`model.global_encoder.encoder_layers.0.self_attn`. At `G=404` it joined a
32 MiB heap lease; at `G=405` it joined the new 1 GiB heap lease. This connects
padded input length, attention tensor geometry, and allocator policy on this
model and runtime path.

#### Native request trace and allocation-order reproducer

A native trace of the longest controlled batch, `G=1,280`, found a real
`100 MiB` `[2,8,1280,1280]` request during backward. The request created a
`1,024 MiB` private heap. The driver counter rose and later fell by exactly
`1,073,741,824` bytes. Serialized dispatch reproduced the event.

An allocator-only reproducer then removed the model, data, Autograd, optimizer,
and graph execution. Both layouts used the same `954 MiB` allocation multiset
and retained the same `576 MiB` live set before a 100 MiB probe.

| Layout | Largest aligned free block | Probe result |
|---|---:|---|
| Grouped allocations | 448 MiB | Reused the existing 1 GiB heap |
| Interleaved allocations | 70 MiB | Created a second 1 GiB heap |
| Interleaved repeat | 70 MiB | Created a second 1 GiB heap |

Request geometry determines the allocator class. Allocation history and the
largest aligned contiguous free block determine whether that request reuses
existing capacity or creates another heap.

### 3. Heap release accounts for the synchronized driver residual

A 20-step reproduction from checkpoint step 44,250 crossed the training loop's
periodic cleanup boundary at absolute step 44,260:

| Phase | A | H | D | H-A | D-H |
|---|---:|---:|---:|---:|---:|
| Before periodic cleanup | 315.925 MiB | 2,896 MiB | 2,979.156 MiB | 2,580.075 MiB | 83.156 MiB |
| After periodic cleanup | 315.925 MiB | 1,672 MiB | 1,755.156 MiB | 1,356.075 MiB | 83.156 MiB |

Cleanup discarded `1,224 MiB` of fully free heap capacity without changing
active memory or `D-H`. Later shapes rebuilt capacity: `H` returned to
`2,800 MiB` within the following ten steps. Cleanup changes allocation
topology, but it does not remove the geometry and allocation-history
mechanism.

At the end of the run, cleanup actions were applied one at a time:

| Phase | A | H | D | H-A | D-H |
|---|---:|---:|---:|---:|---:|
| After `zero_grad` | 228.568 MiB | 2,800 MiB | 2,895.984 MiB | 2,571.432 MiB | 95.984 MiB |
| After garbage collection | 228.568 MiB | 2,800 MiB | 2,895.984 MiB | 2,571.432 MiB | 95.984 MiB |
| After synchronization | 228.568 MiB | 2,800 MiB | 2,895.984 MiB | 2,571.432 MiB | 95.984 MiB |
| After `empty_cache` | 228.568 MiB | 272 MiB | 367.984 MiB | 43.432 MiB | 95.984 MiB |

At the aligned pre-release endpoint, `H-A` explained `96.40%` of `D-A`.
`empty_cache` released `2,528 MiB` from both `H` and `D`. The remaining
`43.432 MiB` of `H-A` was slack inside live heaps, and `95.984 MiB` remained
in the `D-H` driver-counter residual.

The aligned endpoint therefore contains four byte classes:

```text
D = 228.568 MiB active blocks
  + 2,528 MiB fully free retained heaps
  + 43.432 MiB slack in live heaps
  + 95.984 MiB outside reconstructed PyTorch allocator heaps
```

Garbage collection and another synchronization released nothing at the already
synchronized endpoint. The dominant releasable owner was retained whole heaps,
not active tensors.

Moving the existing post-update `zero_grad(set_to_none=True)` immediately
before cleanup exposed another `1,400 MiB` of fully free heap capacity and
drove `H` to `272 MiB` at each observed cleanup boundary. Physical footprint
did not improve persistently. The modified arm ended at step 118 before it
reached the same terminal boundary as the unchanged control, so the run cannot
support a persistent physical-footprint comparison. This improves allocator
topology but is not a validated production mitigation.

### 4. Autograd-saved storage pins heaps during backward

On a natural batch with pooled sequence length `G=998`, `704.064 MiB` of
non-model storage saved by Autograd had not yet been consumed by backward at
the end of the forward pass. Three square attention storages of about
`60.8 MiB` each remained until late backward. Matching storage lifetimes and
sizes against allocator lease events associated `91.05%` of unique saved
storage with allocator leases; every unmatched storage was smaller than
1 MiB.

This shows that saved Autograd buffers can pin allocator heaps during backward.
It does not explain every allocator or Metal byte. A saved tensor's first use
during backward is not the same event as Python reference release, allocator
return, or GPU completion, and this ledger cannot see native operator
workspace.

### 5. Process-level memory grows beyond the observed MPS counters

The 120-step continuation was cache-disabled, so it is a local allocator probe
rather than a production-equivalent pressure run. Its last four post-cleanup
`D` endpoints stayed within `174.859 MiB` and declined rather than
accumulating. Over the same process lifetime:

| Counter | Loaded model | Terminal |
|---|---:|---:|
| Task compressed | 0 | 9.293 GiB |
| Physical footprint | 1.547 GiB | 14.668 GiB |
| RSS | 1.658 GiB | 2.985 GiB |

Cleanup changed `D`, physical, graphics, compressed, and RSS ledgers on
different schedules. Those counters describe overlapping byte sets and do not
support a same-byte ownership equation.

Optimizer updates were not required:

| 40-step no-update arm | Peak H | Peak D | Final compressed | Final physical | Final RSS |
|---|---:|---:|---:|---:|---:|
| Dynamic resumed prefix | 3,808 MiB | 3,981.516 MiB | 1.074 GiB | 9.181 GiB | 4.815 GiB |
| One fixed batch repeated | 576 MiB | 599.000 MiB | 0 | 1.817 GiB | 1.707 GiB |

The production dataset policy uses cache 16. Its LRU was too small to explain
the effect. An adjacent cache-0/cache-16 comparison changed physical footprint
by about `107.6 MiB`, compressed memory by `13.0 MiB`, and RSS by `64.4 MiB`,
while peak `A/H` matched and peak `D` differed by only 16 KiB.

Identical cold/warm replay separated bounded first-seen state from continuing
growth:

| Arm and cycle | Delta H | Delta D | Delta graphics | Delta physical |
|---|---:|---:|---:|---:|
| Fixed cold | +416 MiB | +438.516 MiB | +456.859 MiB | +720.188 MiB |
| Fixed identical warm | 0 | 0 | +0.094 MiB | -290.469 MiB |
| Dynamic cold | +2,624 MiB | +2,736.594 MiB | +2,663.813 MiB | +4,306.517 MiB |
| Dynamic identical warm | 0 | -4 MiB | +7.297 MiB | +47.656 MiB |

The warm dynamic cycle created no new heaps and added only `47.656 MiB` of
physical footprint. This supports a bounded first-seen component, but no
cache-miss events were available to implicate a graph or kernel cache.

### 6. Counterbalanced fixed and dynamic runs isolate the remaining gap

A twelve-shape schedule, replayed in its original order and with the maximum
shape first, showed that order changes heap topology. The original order
created twelve more 32 MiB heaps, a `384 MiB` difference. Training-mode dropout
was assigned to different inputs when the order changed, so this comparison
could not isolate process footprint by itself.

Dropout-free eval repeats preserved a smaller exact order effect:

| Original minus maximum-first order | Delta H | Delta D | Delta graphics | Delta physical |
|---|---:|---:|---:|---:|
| Pair 1 | +256 MiB | +283.984 MiB | +269.094 MiB | +270.141 MiB |
| Pair 2 | +256 MiB | +284.000 MiB | +266.594 MiB | +267.359 MiB |

Both differences were exactly eight 32 MiB heaps. This confirms that allocation
order matters even without dropout, although eval mode can change graph keys
and saved tensors.

The final comparison used dropout-free evaluation mode. An input signature
here includes tensor shapes, dtypes, masks, valid lengths, and execution mode.
One arm repeated the maximum signature twelve times. The other ran twelve
different signatures, beginning with the same maximum, so both arms
established the same initial high-water shape before fixed or variable
execution continued:

| Dynamic minus repeated maximum | Delta H | Delta D | Delta graphics | Delta internal | Delta physical |
|---|---:|---:|---:|---:|---:|
| Pair 1 | +24 MiB | +31.609 MiB | +13.172 MiB | +328.563 MiB | +344.094 MiB |
| Pair 2 | +24 MiB | +31.609 MiB | +4.359 MiB | +330.625 MiB | +338.531 MiB |

A matched CPU-only comparison put the dynamic-minus-fixed process delta at
only `115.2-116.5 MiB`, below the 128 MiB equivalence threshold set before the
comparison.
A production data-loading arm and an arm that materialized the same host inputs
before the measurement window both created `2,624 MiB` of MPS heap capacity.
Moving host allocation earlier did not reduce the terminal footprint.

The repeated task-internal and physical deltas occurred while the observed
heap difference was `24 MiB`, the driver difference was `31.609 MiB`, the
graphics difference was `4.359-13.172 MiB`, and the CPU-only process
difference remained below the 128 MiB equivalence threshold. The dropout-free
repeated-maximum/dynamic pair is therefore the smallest reproduction suitable
for native cache and VM-region attribution. It does not identify the owner by
itself.

## Limits of attribution

The conclusions rely on controlled deltas, repeated boundary effects, and
phase-aligned counters. The following limits define what the results do not
show:

- The `A/H/D` decomposition applies only to aligned synchronized endpoints.
  Independent peak values cannot be subtracted.
- Probe synchronization can change scheduling. A replay that sampled only at
  phase boundaries preserved the key 20-step heap topology, but it still
  included production scalar metric materialization.
- Darwin task ledgers are process-wide and overlapping. They do not identify
  resident DRAM, Python objects, graph executables, or individual Metal
  resources.
- A saved tensor's first backward use and allocator lease return are not
  GPU-completion events. Operator-native workspace and command resources remain
  unobserved.
- The one-frame `G=404/405` boundary result is scoped to a masked
  evaluation-mode forward and backward pass on this model and runtime. It does
  not establish a universal sampler rule.
- The allocator-only order replays used the same allocation multiset, but
  training-mode input reordering changes which dropout mask applies to each
  descriptor. The clean order result therefore uses dropout-free eval repeats.
- The 120-step allocator plateau is local. It does not prove the rest of
  training is safe. The cleanup-placement arm ended at step 118 before reaching
  the same terminal boundary as its unchanged control.
- Dynamic natural content has not been varied while tensor shapes, dtypes,
  masks, valid lengths, execution mode, and one reusable host-buffer layout all
  remain constant. The diagnostic content fingerprint samples at most 4,096
  logical values per tensor, so it detects observed drift but does not prove
  byte equality.
- No graph/kernel cache-miss ledger, command-buffer completion ledger, or
  VM-region trace has been collected. The remaining process owner is unknown.

## Version scope and references

The source interpretation is pinned to PyTorch 2.11.0 and the XNU source
observed for macOS 26.6:

- [PyTorch MPS allocator thresholds and heap sizes](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/mps/MPSAllocator.h#L22-L30)
- [PyTorch MPS heap-selection branch](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/mps/MPSAllocator.h#L144-L156)
- [PyTorch Storage-to-MTLBuffer bridge](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/mps/OperationUtils.h#L106-L107)
- [PyTorch MPS graph and kernel caches](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/mps/OperationUtils.h#L213-L381)
- [XNU task VM layout](https://raw.githubusercontent.com/apple-oss-distributions/xnu/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/mach/task_info.h)
- [XNU task ledger accounting](https://raw.githubusercontent.com/apple-oss-distributions/xnu/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/task.c)
- [Apple Metal memory analysis](https://developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app)

## Root-cause assessment

The confirmed cause of MPS driver-counter growth is allocator amplification.
Variable attention geometry crosses a PyTorch 2.11 size-class boundary,
allocation history determines whether the next request can reuse existing
capacity, and fully free cached heaps dominate the synchronized difference
between active MPS memory and the driver counter.

This finding does not account for all process memory. The repeated
task-internal and physical-footprint deltas remain much larger than the matched
allocator and graphics-counter deltas. The controls rule out optimizer updates,
the configured dataset LRU, and the tested CPU-only materialization path as
sufficient explanations. Public ledgers cannot distinguish among graph or
kernel cache state, command resources, native anonymous memory, host dirty
pages, and other runtime structures, so ownership of this second effect remains
unresolved.

## Operational decision

| Proposal or claim | Decision | Basis |
|---|---|---|
| Rewrite the model | Not justified | The allocator-only reproducer explains the heap amplification without model execution. |
| Add a sampler rule at the 6,464/6,465-frame boundary | Not justified | The boundary is specific to this batch size, dtype, runtime version, allocator branch, and controlled padding test. |
| Call `torch.mps.empty_cache()` every step | Do not adopt | Cache release lowers `D` temporarily, but later shapes rebuild the heaps and repeated eviction can add synchronization and allocation cost. |
| Classify the incident as a persistent tensor leak | Contradicted in the tested scope | Fixed-batch replay left no live tensor with `grad_fn` after a step and active MPS memory stayed nearly flat. |
| Treat `D` as resident DRAM | Invalid accounting | The driver counter, task ledgers, RSS, and physical footprint are overlapping projections with different lifetimes. |
| Move `zero_grad` before periodic cleanup in production | Not yet validated | The change exposed more fully free heap capacity, but the available run did not show a persistent physical-footprint improvement and ended early. |

Zero-grad-before-cleanup is promising for allocator topology, but it has not
reduced physical pressure under a complete matched run. Production should stay
unchanged until the owner of the unmatched process growth is identified and a
targeted mitigation improves both memory pressure and settled throughput.

## Recommended next diagnostics

The allocator mechanism is resolved for the tested configuration. Further work
should target the unmatched process-footprint delta and determine whether the
result generalizes to natural input content. These diagnostics should use
separate instrumentation or diagnostic runners; they do not require a
production model, sampler, allocator-policy, configuration, or checkpoint
change.

### Attribute native runtime state

Use the smallest dropout-free comparison that preserves the process delta: one
evaluation arm repeats the maximum input signature, while a second arm starts
with that maximum and then runs the variable-signature sequence. Each arm should
run two identical cycles from checkpoint step 44,250 without optimizer updates.

A diagnostic PyTorch 2.11 build should first record graph and kernel cache
misses and entry counts. Cold-only entry growth that plateaus on the identical
warm cycle would show that cache state co-varies with the process delta, but
would not prove byte ownership. If footprint continues to grow after the cache
counts plateau, add allocator lease IDs and command-buffer completion IDs to
separate buffer reuse from delayed native work. Patched and unpatched fixed
arms must retain the same heap topology and settled behavior; instrumentation
that inserts waits, changes allocator policy, or materially changes latency
invalidates the comparison.

### Separate execution signature, content, and host materialization

An execution signature is the complete set of tensor shapes, dtypes, masks,
valid lengths, and execution mode that can select a graph or kernel path. The
current evidence does not vary natural content while holding that signature and
the host-buffer layout constant.

Use one reusable fixed-capacity CPU buffer to compare five cases:

1. fixed content with one fixed maximum signature, materialized before the
   measurement window;
2. the same content over the variable-signature sequence;
3. variable natural content projected into the fixed signature and the same
   preallocated buffer;
4. the same content, signature, and buffer layout as case 3, materialized inside
   the measured loop; and
5. the data path from case 4 without an MPS model.

Compare cold and warm changes in `A`, `H`, `D`, the task ledgers, buffer
identity, and settled throughput. A case-4 versus case-3 delta that also appears
in case 5 would identify a host-materialization contribution. A delta present
only with MPS execution would narrow the native-runtime analysis. Content
fingerprints can detect observed drift, but a sampled fingerprint cannot prove
that unsampled bytes are equal.

### Localize any remaining process delta

If native cache and lease counters leave a repeatable gap, trace the smallest
four-to-twelve-step fixed/dynamic comparison rather than a production run.
Start with a mutually exclusive VM-region partition and compare resident,
dirty, compressed, and swapped deltas within the same ledger. If that view is
inconclusive, collect labeled Metal resource events in a separate run.

A stable VM-region delta can localize the process charge to a memory-region
class. A stable Metal category shows only co-variation with a resource class.
VM-region, Metal, task-ledger, and MPS allocator totals overlap and must not be
combined into one ownership equation. If tracing changes scheduling, heap
topology, or the untraced process delta, the trace is not valid evidence.
