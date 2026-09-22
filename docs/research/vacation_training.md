# Vacation training queue

The worker runs three serial stages: TRAIN audio features, clean R1 teacher
training, and native generation from a separately pinned model. The default
teacher has 35,178,768 parameters: width 512, eight local levels, observed seed
conditioning and 256-wide landmark memory. No recovery or routing residual is
enabled. The existing small-model entrypoints retain their execution envelope.

The separate `vacation_training_r1_response` preset uses the recorded response
candidate's 3,084,432-parameter architecture: width 128, eight local levels,
observed seed, 256-wide landmark memory, 512-wide head and release routing,
and `frontier2` row consequences. It uses CPU1 and model seed 172, with source
training readouts through 6.75M onsets and a four-hour training budget. Audio
caching and independent fixed-model stress are disabled; native VAL readouts
remain enabled within the training stage. Standalone training uses
`bounded_typed_train_r1_response`; generation uses the ordinary small profile.

This preset rebuilds an architecture from random initialization. It does not
recover a trained checkpoint or reproduce the earlier staged native-preference
corrections. Restoring the earlier performance also requires rebuilding those
TRAIN trajectory pools, routing/release/response continuation and quality checks.
The 35M teacher remains the separate clean architecture above.
Use the [staged R1 reconstruction](r1_staged_restoration.md) and
`scripts/r1-restore.sh` when the objective is to follow the earlier module and
native-correction order. The all-module preset is not a substitute for that path.

Completion means producing assets and measurements. It does not establish
arrangement quality, select a working candidate or authorize distillation.
Frozen music-encoder features and audio/chart alignment training require a
separately selected encoder and objective; they are not included in this queue.

## Entry and budgets

Use a clean committed checkout, Python 3.10, the `mps` extra, and FFmpeg on PATH.
Install dependencies before leaving; the launcher uses `uv --offline`.

```sh
./scripts/vacation-training.sh --help
./scripts/vacation-training.sh --cfg job
```

The launcher attaches `caffeinate -is` to the worker lifetime. Display sleep
remains available. AC disconnection, a macOS thermal restriction, SIGINT, SIGTERM
or an output-directory `PAUSE` file requests a pause at the next safe boundary.
Closing the lid or forced system sleep is not a supported way to keep computing.
No background daemon or automatic retry is installed.

| Stage | Default wall budget | Products |
| --- | ---: | --- |
| Input preflight | 2 hours | Configuration/input freeze and regenerated TRAIN plan |
| Audio | 16 hours | Content-addressed shards, sample clocks and anomaly receipts |
| Teacher | 44 hours | 1M/5M/10M/20M/30M exposure checkpoints and fixed readouts |
| Native stress | 8 hours | Separate TRAIN/VAL trajectories and verified exports |

The total wall budget is 72 hours, including startup and reporting overhead.
Limits are checked at safe boundaries: an in-flight update, decode step or
audio chunk may finish before stopping. Sleep and pauses consume wall allowance;
training exposures advance only for completed work. A stage exhausting its
budget retains partial products and permits independent later work. A user,
power, thermal, resource or total-time pause stops the whole queue. Failed stages
remain failed on resume, with no automatic reselection or retry.

New products are capped at 100 GiB, audio at 80 GiB, with 100 GiB of disk reserve.
The worker observes RSS, physical footprint, available memory, memory pressure
and swap growth. Swap uses the queue's original baseline across stages/resumes.
Training/generation additionally enforce their own device guards. Sampled checks
cannot guarantee the absence of transient operator peaks. Original data,
inherited checkpoints and other run folders are never deleted.

## Input preparation

All inputs need exact SHA-256 identities. Relative paths resolve from the launch
directory; absolute paths can reference immutable caches in another worktree.
The execution freeze binds the source commit, configuration, Python, Torch,
platform and selected codec versions. Changes require a new output directory.

Create the audio inventory from an existing TRAIN plan and its catalog:

```sh
uv run --offline --python 3.10 --extra mps python -m \
  ensomi_model.research.vacation_training.prepare_audio_hydra \
  plan_file=/path/to/train-plan.json plan_sha256=REPLACE_WITH_SHA256 \
  catalog_file=/path/to/catalog.json catalog_sha256=REPLACE_WITH_SHA256 \
  catalog_root=/path/to/catalog-owning-checkout \
  source_cache_dir=/path/to/admitted-row-cache \
  output_file=artifacts/vacation-inputs/audio.json
```

Preparation reads selected TRAIN chart/audio payloads, deduplicates audio bytes,
and preserves difficulty associations. Missing paths remain explicit issues. A
directory appearing in other split metadata is quarantined conservatively; that
is not proof that its actual audio bytes cross splits. Other split audio is not
read for this inventory. Original files and allocation groups are unchanged.

The `vacation/audio-inputs-v1` manifest has an `assets` list containing `path`,
`sha256`, and `sources`. Associations record `source_sha256`, `group_id`, `split`,
`first_ms` and `last_ms`. Non-TRAIN or ambiguous assets are excluded explicitly.

The teacher uses the existing TRAIN plan's population and sampler definition.
Preflight regenerates draws for the declared milestones into `training-plan.json`.
This starts a fresh teacher, not an inherited recovery optimizer. The plan-file
limit is 128 MiB; the longer schedule must fit without dropping draws.

Provide a fixed `vacation/teacher-evaluation-v1` JSON file containing:

- `source_cache_dir` and `sources`, using corpus-plan metadata/row pins. TRAIN
  entries must equal their training-plan entries; VAL groups must be disjoint.
- `windows`, each with `source_sha256`, `first_onset` and `onset_count` from 1 to
  256. Include TRAIN and VAL windows for the reported likelihood gap.
- `native_cases`, at least one fixed VAL case in the format below. Prepare its
  external condition using the existing `bounded_typed_condition` entrypoint.

The stress file has `format: vacation/native-cases-v1` and a `cases` list. Each
case has a unique lowercase `id` using letters, digits, hyphens or underscores;
`source_sha256`, `group_id`, explicit `train` or `validation` `split`;
`condition_file`, `condition_sha256`, and an integer `seed`. Optional
`presentation_source`/`presentation_sha256` copy playback metadata only.
One case is one source/seed pair. Freeze source strata and seeds before launch;
a source or group cannot appear in both splits.

Stress generation uses `stress.generation.checkpoint_file` and its SHA-256,
selected before the queue begins. It is independent of teacher success. Native
readouts inside the teacher stage instead use that exposure checkpoint.

## Local configuration and commands

Create an ignored `artifacts/vacation-config/local.yaml`, replacing every path
and digest placeholder with frozen experiment inputs:

```yaml
defaults:
  - vacation_training
  - _self_
output_dir: artifacts/vacation-run
audio:
  manifest_file: /path/to/audio.json
  manifest_sha256: REPLACE_WITH_SHA256
teacher:
  base_plan_file: /path/to/train-plan.json
  base_plan_sha256: REPLACE_WITH_SHA256
  evaluation_file: /path/to/fixed-evaluation.json
  evaluation_sha256: REPLACE_WITH_SHA256
  training:
    source_cache_dir: /path/to/admitted-row-cache
stress:
  manifest_file: /path/to/native-cases.json
  manifest_sha256: REPLACE_WITH_SHA256
  generation:
    checkpoint_file: /path/to/fixed-candidate.pt
    checkpoint_sha256: REPLACE_WITH_SHA256
```

A deliberately excluded stage can use `enabled: false`. For teacher-sized stress
weights, also use `execution_profile: teacher35m` and the resource settings in
`bounded_typed_generate_teacher35m`, including its 1 GiB checkpoint cap. The
default stress profile supports the small working candidate.

To prepare fresh inputs from the surviving pinned September catalog/cache, use
the [input reconstruction driver](../../experiments/vacation_rebuild/README.md).
It creates separate local R1 response and 35M teacher configurations, a rebuilt
TRAIN plan, fixed evaluation conditions and a TRAIN audio inventory. Its unique
asset directory must be outside the product worktree. Fixed monitoring cases
are newly selected; they do not reconstruct the deleted historical cohorts.

```sh
./scripts/vacation-training.sh --config-dir "$PWD/artifacts/vacation-config" \
  --config-name local mode=preflight
./scripts/vacation-training.sh --config-dir "$PWD/artifacts/vacation-config" \
  --config-name local mode=run
```

Preflight verifies inputs, caches, readout separation, codecs, and native model
loading/task compatibility. It is not a
sustained hardware benchmark or scientific acceptance gate. The ordinary CPU
teacher runner has an exact optimizer-recovery and native-generation regression
test; long-run throughput still depends on corpus shape and host conditions.

```sh
touch artifacts/vacation-run/PAUSE
./scripts/vacation-training.sh mode=status output_dir=artifacts/vacation-run
rm artifacts/vacation-run/PAUSE
./scripts/vacation-training.sh --config-dir "$PWD/artifacts/vacation-config" \
  --config-name local mode=resume
```

An exclusive POSIX lock prevents concurrent workers. Training and generation
continuations each own a fresh segment directory and preserve their parent.
Training resumes the finalized runtime ledger, optimizer, draw cursor, RNG and
journal identity. A hard kill without a finalized runtime ledger requires an
audit; the worker does not invent compute time. Completed cases are not rerun.

## Products

`freeze.json`, `ledger.json`, `resources.jsonl` and `summary.json` expose inputs,
stage status, resource history and incomplete work. `bounded_typed_train_teacher35m`
and `bounded_typed_generate_teacher35m` are also usable outside this queue.
Finished queues with incomplete stages return exit code 2. Safe pauses are
recorded as `paused`; status queries remain read-only and return normally.

Audio is mono 24 kHz, stored as float16 128-bin log-mel shards. The hop is 240
samples (100 Hz), FFT/window size 1,024, without centering padding. Frame `i` is
centered at `(240*i + 512)/24000` seconds. Chunks retain overlap; final incomplete
windows are omitted explicitly. Indexes include decoded sample counts and the
unframed tail. Chart labels retain their millisecond times. No offset is guessed
from duration mismatch. Resume verifies shards and decodes the prefix again
before continuing, avoiding codec-seek clock shifts.

Teacher readouts report likelihood, available LN/TAP diagnostics and a fixed
TRAIN/VAL gap. They do not adapt the objective, select a best model or invent a
plateau rule. Stops follow exposure, wall time, pause requests and resource limits.

Stress products live in separate `train/` and `validation/` directories. Rows,
decisions, conditions, model identities and native state support exact replay.
`diagnostics.json` includes physical organization measures and complete context
around the longest identical-row run. These are locators, not style/playability
labels. Native star rating is explicitly unavailable until a calculator is
selected and pinned. VAL trajectories are never teacher-training inputs.

Run the focused checks with:

```sh
uv run --offline --python 3.10 --extra mps --group dev pytest -q \
  tests/research/vacation_training
```
