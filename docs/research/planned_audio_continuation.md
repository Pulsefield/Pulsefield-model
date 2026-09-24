# Planned heads, physical releases and audio-conditioned rows

This research prototype generates a head plan ahead of row materialization,
then interleaves LN releases and complete action rows. It implements the
[information contract](audio_skeleton_information_contract.md), including direct
audio-to-row conditioning and candidate-action consequences. It is not an
adopted V3 architecture or an established playable model.

The [first long-form readout](head_wait_recovery.md) exposes long silent
head-stream gaps despite valid completion and improved source likelihood.
Head-only replay and a recency intervention locate a suppression path; the
endpoint is not a playability candidate.

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

The optional bounded head mode adds a 2250-parameter linear audio readout.
Its historical correction has magnitude at most head_bound times
exp(-head_age_ms / head_decay_ms), with zero historical correction at BOS.
The default bound is 4 and decay time is 1000 ms. These settings are projected
from the training config and saved with the model. After a long wait, current
audio regains control without a forced head. The unbounded mode remains the
default for reproducing the first checkpoint; the recovery hypothesis and its
limitations are defined in the [waiting-time analysis](head_wait_recovery.md).

For a fixed head plan, the row/release process visits only committed physical
states. Its joint likelihood is the sum of the head hazard likelihood over
every native millisecond, release hazard likelihood on occupied non-H clocks,
and conditional legal-row likelihood. An interval boundary only censors these
factors; it does not restart their histories, hide full-song audio, or close
holds. Source head plans are teacher-forced intermediates during fitting,
not additional inference inputs unavailable to the generator.

If all four lanes are held, at least one release must occur before the next H.
The default law assigns remaining survival mass to H minus one millisecond;
this reproduces the first checkpoints. With `condition_full_holds=true`, the
raw first-release distribution is instead conditioned on an event before H.
Both modes have a certain final hazard, but their earlier hazards and final
event probabilities differ. The flag reaches training and is checkpointed.
No parameters are added. The [waiting-law analysis](release_wait_conditioning.md)
defines the distinction and the observed deadline artifact.

Conditional training queries hypothetical no-release states through H minus
one, extending the local audio crop and halo if necessary. No actual future
tails enter these queries. Native generation scores the same complete feasible
wait before sampling; `chunk_ms` does not truncate its normalizer. This additional
work scales with the wait length. Partial occupancy and true audio-end closure
retain the original law. A row cannot leave all lanes occupied when the next H
is one millisecond away. These are feasibility conditions, not comfort guarantees.

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
Both interval training and cached native generation use this path. The residual
receives row-likelihood gradients at the inherited R1 learning rate; it is not a
frozen or evaluation-only copy. The 16-head preview also enters the shared row
condition, while the consequence features explicitly describe the next H and
the second-H gap.

The input opportunity distribution and row condition differ from supplied-time
R1. Weight transfer is not policy preservation. Seed and landmark modules remain
omitted, and the exact projection retains its historical slice. The old 30 ms
machine-preference objective is not used for new fitting. Joint source
likelihood trains all modules; it does not give the residual a calibrated
gameplay-demand interpretation.
The [release waiting-law analysis](release_wait_conditioning.md#relationship-to-frontier2)
shows a case where this active residual shifts probability from starting two
holds toward tap-containing alternatives, yet its earliest-release feature
misses substantial probability of a release immediately before H. Retaining
frontier2 does not establish that its finite features capture the formulation's
full gameplay frontier.

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

The recovery comparison uses bounded_head=true with the fresh run name
planned-bounded-head-preflight-v1; its full fit uses
planned-bounded-head-main-v1 and the same frozen exposure plan. No preflight
model or optimizer state is reused for the full fit.

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

The [full-audio playback study](audio_playback_system.md) measures fresh-input
startup, dense and long source-H workloads, settled coverage and virtual
presentation deadlines. It separates demonstrated compute headroom from the
remaining musical-distribution and control questions.

The [shared arrangement-profile model](shared_arrangement_profiles.md) adds a
small full-audio prior and one persistent condition shared by head, release and
row factors. Its profile descriptors are separate from style and playability
judgments; native benefit remains under evaluation.

## Generate and stream from an audio file

The source-chart-free entrypoint accepts a planned-family checkpoint and audio
file. It uses the checkpoint's normalization and settings, including its
release waiting law, and generates its own initial context from BOS. It does
not load the checkpoint's training corpus. Run from a clean committed checkout:

```bash
uv run --extra mps python -m ensomi_model.research.planned_audio_continuation.infer_hydra \
  audio_file=/absolute/path/song.mp3 \
  checkpoint_file=artifacts/joint-audio/20260924-alias-restored-v1/planned-training/planned-feasible-release-main-v1/last.pt \
  checkpoint_sha256=67b8fc8fbac5f7ca2debe99524e29ac002546f12cb8a3c4c9acf68d68d7f3839 \
  output_dir=artifacts/planned-audio-playtest/song-v1 device=cpu seed=17
```

The checkpoint in this example is a local research asset, absent in a fresh
clone and not promoted as a final playable model. Substitute another pinned
planned-family checkpoint explicitly. The output directory must not already
exist. `--help` and `--cfg job` inspect packaged settings without importing
Torch. Unknown settings are rejected by typed projection.

CLI stdout and `events.jsonl` carry the same ordered JSONL records. Each record
has `kind`, `sequence` and wall `elapsed_seconds`:

| Kind | Consumer meaning |
| --- | --- |
| `start` | Stream format `planned-audio/stream-v1`, four columns indexed from zero, action vocabulary and startup requirements |
| `audio_ready` | Complete decoded audio duration, verified audio identity and preprocessing timings; chart generation has not completed |
| `update` | Optional complete `row`, inclusive `coverage_ms`, cumulative `row_count`, true-audio `completed` and `playback_ready` |
| `stop` | Generation status, stop reason, final coverage and result-file location |
| `error` | A failed producer or consumer operation, with the last committed coverage preserved in the file |

`stop` describes generation status. The result file and full event-file digest
are finalized when the producer returns; a callback should not read that report
before the call finishes.

An update atomically fixes its row and all row/no-row decisions through
`coverage_ms`. `row=null` advances settled empty coverage without adding a
history token. Row timestamps are integer-valued milliseconds. Actions are
`[EMPTY, TAP, LN_START, LN_CLOSE]` indexed by codes 0–3. LN starts contain no
guessed endpoint; later rows close the occupied columns. Consumers replay
those complete rows in order. `completed` denotes the true audio end with all
holds closed, never a cache or time-budget boundary.

By default, `playback_ready` becomes true after both 30 physical rows and
8000 ms of coverage are ready, or at true completion for shorter charts.
`startup_min_rows` and `startup_coverage_ms` change that presentation decision
without changing samples. Readiness is not a guarantee of future throughput or
musical quality. A consumer owns its player clock and must handle a later
`stop` with `status=capped` without inventing LN tails.

Python consumers call
`inference.infer_audio(AudioInferenceConfig(...), on_event=callback)`. Records
are flushed to disk before the synchronous callback receives an independent
JSON-compatible object. Slow consumers apply backpressure; they do not draw
random numbers for the model. Callback exceptions propagate after an error
record is saved. The file is an inspectable prefix, not a crash-resume format.

Settings `chunk_ms` and `head_chunk_ms` control query work partitioning;
`max_rows` and `max_seconds` bound generation. Time/resource checks occur
between preprocessing stages and scheduler iterations, so one operation,
callback or final export can exceed the elapsed-time bound. The existing
research resource guard also recognizes an output-directory `PAUSE` file,
requires 2 GiB free RAM and 40 GiB free disk, and records a capped result on a
guard stop. `correct_short_attacks` remains false by default because its
[whole-chart quality comparison](head_plan_row_response.md) failed a guard.

`screen_unpublished=true` selects the optional
[joint continuation screen](unpublished_continuation_screen.md). It validates
an eight-second unpublished window and a 20-ms halo, with at most four proposals,
before exposing the accepted prefix to the same event stream. The policy checks
strict same-column attack gaps below 20 ms and the separate experimental
release-to-head criterion. Exhaustion returns `planning_attempt_limit` with
only the previously published prefix; open LNs retain unknown tails. This
research option defaults to false and cannot be combined with
`correct_short_attacks`. Its measured physical-screen benefit does not establish
musical quality or preservation of the requested arrangement.

The run saves resolved YAML, typed settings, input/model/frontend identities,
per-stage timings, events and native row diagnostics. A complete run also
exports `chart/generated.osu` and its paired audio, then independently reparses
the chart. Presentation uses constant editor timing and SV; it does not infer a
musical redline grid. A capped run saves its available rows and open LN state
without exporting a completed chart. The total startup profile includes model
loading and fresh audio preprocessing but excludes Python imports, Hydra/Git
validation and output setup.

At source 1c95f6914fd3fa390d8a46f1d267d5311def7078, nine actual CLI processes
reproduce the preceding conditional-release baseline rows exactly. A separate
consumer replays each stdout update as it arrives, verifies row/no-row coverage
and LN state, and receives readiness before process exit. Stdout and the saved
event stream match byte for byte. Playback-ready delivery takes 1.399–1.843 s
from process launch, including each process's imports and model loading;
producer-side readiness takes .545–.966 s under the narrower profile above.
The nine-case verification takes 28.988 s. These unchanged charts retain their
previously reported musical limitations.

Local verification owner: artifacts/joint-audio/20260924-stream-entry-v1.
Freeze: 752bfa3908a192a075af22a9c85233b0ff5d771313a2f0dd24ecf9b6c72af21a.
Result: 2eef5f7b1d4ab6b71e9420b665512990424524aa3d397db6f7e6cc6bc33922b4.
