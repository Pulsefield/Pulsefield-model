# Planned heads, physical releases and audio-conditioned rows

This research prototype generates a head plan ahead of row materialization,
then interleaves LN releases and complete action rows. It implements the
[information contract](audio_skeleton_information_contract.md), including direct
audio-to-row conditioning and candidate-action consequences. It is not an
adopted V3 architecture or an established playable model.

## Distribution and available information

The head stream uses complete canonical Mel audio, a 127-head causal history
and the elapsed clock since its last head. It has native integer-ms support;
no beat grid, redline target or acoustic onset label is required. A 16-head
preview conditions row materialization. The scheduler keeps one additional
head for the current H decision and distinguishes an exhausted cache from
known audio termination.

The release-only stream reads the same audio, 63 past H/R skeleton events,
current occupied lanes and active LN ages, and the next two planned heads.
It cannot read tap locations, chord sizes, row embeddings or unrelated replay
clocks. At H, the row must contain a head and may release other occupied lanes.
A release-only event must release at least one hold and cannot create a head.
This head-stream independence from LN choices is a stronger pilot bias than
the general LN-feedback contract.

For a fixed head plan, the row/release process visits only committed physical
states. Its joint likelihood is the sum of the head hazard likelihood over
every native millisecond, release hazard likelihood on occupied non-H clocks,
and conditional legal-row likelihood. An interval boundary only censors these
factors; it does not restart their histories, hide full-song audio, or close
holds. Source head plans are teacher-forced intermediates during fitting,
not additional inference inputs unavailable to the generator.

If all four lanes are held, at least one release must occur before the next H.
The release hazard is exactly one at H minus one millisecond. At the true audio
end, remaining holds close with probability one. These deadline atoms assign
remaining survival mass to the boundary in both training and sampling.
They do not conditionally renormalize an unconstrained waiting distribution.
A row cannot leave all lanes occupied when the next H is one millisecond away.
These are feasibility conditions, not comfort guarantees.

## Audio, row state and consequences

The audio path reuses the local Mel TCN and bidirectional 2 Hz full-song branch.
Full-song context is computed once per song microbatch while local crops retain
the exact audio halo. Native generation caches the complete encoding once.
Row materialization retains the R1-derived 511-row causal encoder, historical
exact replay features, joint row scorer and head/release routing. Audio and
forward head-preview projections both enter its row condition.

The explicit transfer adapter also restores the released `frontier2`
`RowConsequence` parameters. For each candidate row, it computes immediate
post-action occupancy and clocks, passively advances them to the next H, and
supplies the second-H gap. On this native clock the earliest possible release
opportunity is now plus one millisecond. This is a structural possibility;
actual future release times and source LN endpoints remain unknown.
Candidate scores join the row logits before legal normalization.

The input opportunity distribution and row condition differ from supplied-time
R1. Weight transfer is not policy preservation. Seed and landmark modules remain
omitted, and the exact projection retains its historical slice. The old 30 ms
machine-preference objective is not used for new fitting. Joint source
likelihood trains all modules; it does not give the residual a calibrated
gameplay-demand interpretation.

## Execution and evidence

The implementation is in
[`planned_audio_continuation`](../../src/ensomi_model/research/planned_audio_continuation/).
Head, release and row samplers use separate RNG streams. Waiting-time draws
retain their exponential survival thresholds across query chunks. Candidate
evaluation neither changes the head plan nor mutates committed state.
Startup begins at BOS and generates its own first context.

The packaged entrypoint uses the existing frozen group/arrangement/time exposure
plan, keeping alternative charts separate and weighting full joint NLL by actual
chart time:

```bash
uv run --extra mps python -m ensomi_model.research.planned_audio_continuation.hydra \
  run_name=planned-preflight-v1 updates=32 validation_every=32 \
  validation_songs=6 max_seconds=300
```

The default full run uses 1200 fresh updates. Each run requires a clean product
revision, pinned corpus/normalization/R1 bytes, and a fresh output directory.
It writes resolved configuration, parameter ownership and transfer receipts,
per-factor likelihood, exposure counts, parameter changes and a fixed endpoint.
The typed config owns resource bounds and source identifiers. Local datasets
and checkpoints are not shipped repository assets.

Tests cover an analytically counted joint law, deadline atoms, native-ms head
feasibility, candidate post-state clocks, tap-layout noninterference, hidden
future tails, mirror symmetry, full-audio interval partitioning, CPU/MPS
probability and gradient agreement, teacher/cached-native score agreement,
query-partition RNG invariance, actual frontier2 transfer and the training
runner's checkpoint round trip. They establish implementation properties,
not native chart quality.

Native evaluation must report failed or empty charts, first-30-row and
first-30-head readiness, initial playback coverage, forced deadline releases,
dense-window throughput and inspected action organization. Head-to-head and
release-to-head intervals remain separate diagnostics. Lens evidence and
playtesting determine whether better NLL or different LN fractions correspond
to better charts; no numerical proxy alone selects the model.
