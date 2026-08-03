# MPS Memory and Performance Troubleshooting

Use this note when MPS training or inference shows rising memory, swap,
out-of-memory failures, or throughput decay. Allocator behavior can change with
the runtime, device, dtype, and operator path.

## Start by separating the symptom

Record the pattern before changing the model:

- Which grows: active MPS memory, the driver counter, or only process memory?
- Does one fixed input reproduce it, or are variable shapes required?
- Is growth limited to a cold cycle, or does it continue in identical warm runs?
- Which cleanup phase, if any, lowers each counter?

Each pattern points to a different owner.

## Keep the lifetime layers separate

| Layer | What it covers | Useful signal |
| --- | --- | --- |
| Python and Autograd | Containers, closures, hooks, batches, outputs, saved tensors, and gradients | Live references, tensor aliases, and saved-tensor inspection |
| Active allocator blocks (`A`) | Storage currently known as active by the PyTorch MPS allocator | `torch.mps.current_allocated_memory()` |
| Allocator heap capacity (`H`) | Active blocks plus reusable blocks and slack inside MPS heaps | Diagnostic allocator events |
| Driver memory (`D`) | Allocator heaps plus other Metal/runtime allocations | `torch.mps.driver_allocated_memory()` |
| Process memory | Python, native libraries, mappings, graphics, compressed pages, and other overlapping VM ledgers | Physical footprint, RSS, compression, swap, and system pressure |

These layers have different release points. `del` removes one Python reference,
`gc.collect()` handles unreachable cycles, and `torch.mps.synchronize()` waits
for submitted work. `torch.mps.empty_cache()` can release fully free cached
capacity, but not live storage or slack in a pinned heap.

`detach()` normally shares storage, while `clone()` allocates. `to()`,
`contiguous()`, and `cat()` may also allocate copies.

## Check shape and allocation geometry

Estimate large intermediates from padded shapes, not valid lengths. For square
attention, one storage scales approximately as:

```text
bytes = batch * heads * sequence_length^2 * bytes_per_element
```

A small shape change can select a much larger allocator class. Allocation order
also matters: free capacity may exist without one suitable contiguous block.
When tracing is available, separate request, block, and heap sizes.

## A useful investigation sequence

1. Repeat one representative fixed input in a fresh process to screen for
   cross-step Python, Autograd, or cache retention.
2. Compare fixed and variable shapes with the same model and similar content.
3. Run data and collation without MPS compute to isolate host-side growth.
4. Compare a cold cycle with an identical warm cycle to expose first-seen state.
5. Replay the same shape multiset in different orders to expose fragmentation.
6. Sample named phases: forward, backward, update, reference release,
   synchronization, cache release, reset, and unload.
7. For attribution, drop references, clear gradients where valid, collect
   garbage, synchronize, then empty the allocator cache. Note which counter
   changes at each step.

Compare aligned phases. Do not combine overlapping MPS and process ledgers.

Track throughput or request latency beside memory. Synchronization and cache
eviction can trade a lower counter for stalls or later reallocation.

## Reading common outcomes

| Observation | Direction to investigate |
| --- | --- |
| `A` grows steadily | Live tensor storage, Autograd state, histories, hooks, or application caches |
| `A` is stable, `D` grows, and `empty_cache()` lowers `D` | Retained allocator heaps, shape classes, or fragmentation |
| `A` is stable and cleanup does not lower `D` | Metal resources, in-flight work, or runtime caches |
| MPS counters are stable while process footprint grows | Host materialization, native allocations, VM regions, or compression |
| Variable shapes grow while fixed shapes stay stable | Shape-dependent buffers, allocator topology, or graph/kernel keys |
| Cleanup frees memory that immediately regrows | Cleanup changes the timing, but not the allocation pattern |

For training, inspect padded batches, saved activations, gradients, optimizer
state, metrics, and checkpoints. For inference, inspect session lifetime,
precomputed features, key/value caches, token growth, concurrency, reset, and
unload.

The
[Mapper MPS root-cause report](../research/mapper_v2_1_mps_memory_root_cause_report.md)
is a worked example. Its investigation method is reusable; its measurements
are case-specific.
