# Pulsefield Model

Pulsefield Model is the research backend for Pulsefield's real-time, 4-key rhythm-map generation stack. It turns audio into timing, control memory, grammar-constrained mapper tokens, `.osu` output, and a WebSocket stream for the client.

> This project is experimental. The timing and mapper pipelines run end to end, but generated maps are not yet production quality. Large datasets, caches, and trained checkpoints are local research assets and are not included in a fresh clone.

## What this repository owns

Pulsefield Model covers the model side of the interactive play loop:

1. Read an audio file and compute mel features.
2. Predict and fit a musical timing grid.
3. Build dense timing and learned control features.
4. Decode a 2-6 star osu!mania-like chart under grammar and replay constraints.
5. Export hitobjects or stream them to the client with playback lead time.

The current research target is 4K generation. The repository does not contain the Pulsefield client or a hosted inference service.

## System path

```text
audio
  -> mel features
  -> BeatThis timing prediction
  -> GridFitter timing grid
  -> dense timing features
  -> control encoder memory
  -> mapper decoder
  -> grammar and replay validation
  -> hitobject tokens
  -> .osu export or WebSocket stream
```

## Quick start

Use Python 3.10 or newer. Install [uv](https://docs.astral.sh/uv/) and FFmpeg first:

```sh
uv --version
ffmpeg -version
```

Use the platform extra that matches the machine.

### Apple Silicon

```sh
uv sync --extra mps --group dev
uv run --extra mps python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
uv run --extra mps --group dev pytest -q tests/test_package_layout.py
```

### Linux with NVIDIA CUDA

The CUDA extra uses the `pytorch-cu128` index declared in `pyproject.toml`.

```sh
nvidia-smi
uv sync --extra cuda --group dev
uv run --extra cuda python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run --extra cuda --group dev pytest -q tests/test_package_layout.py
```

### CPU-only documentation and config work

```sh
uv sync --group dev
```

Model imports and model-backed tests require either the `mps` or `cuda` extra. Native Windows is not a primary target; use WSL2 and follow the CUDA setup when GPU passthrough is available.

## Inspect configuration before running

Mapper training and inference use packaged Hydra configs. Inspect the composed job before starting a server or training run:

```sh
uv run --extra mps python -m pulsefield_model.inference.hydra_entry --cfg job
uv run --extra mps python -m pulsefield_model.training.mapper_training_hydra --cfg job
```

Replace `mps` with `cuda` on Linux. For an actual Linux training run, also override `runtime.device=cuda` because the canonical mapper presets target MPS. `--cfg job` shows raw Hydra composition; it does not run the semantic validation path.

The main entrypoints are:

| Task | Command |
| --- | --- |
| Show inference options | `uv run --extra mps python -m pulsefield_model.inference.hydra_entry --help` |
| Start the local WebSocket service | `uv run --extra mps python -m pulsefield_model.inference.hydra_entry` |
| Show mapper training options | `uv run --extra mps python -m pulsefield_model.training.mapper_training_hydra --help` |
| Validate a training preset without training | `uv run --extra mps python -m pulsefield_model.training.mapper_training_hydra --dry-run output.output_dir="$(mktemp -d)" output.resume_from=null` |

Inference needs the checkpoints selected by the mapper profile and timing config. Override their Hydra fields when the defaults are not present locally. A training dry run writes `hydra_resolved_config.yaml` and `legacy_run_config.yaml`; the latter is a flattened runner adapter, not a second configuration source.

The WebSocket service listens on `ws://localhost:8765` and exchanges binary `pulsefield-protocol` envelopes. A client sends `ready`, then an `audio` request with a local path, then `reference_time` updates to begin streaming. This repository does not ship a sample client.

A fresh clone can compose configs, run dry runs, and execute tests, but it cannot perform model-backed inference or training. No checkpoint or dataset download workflow is published. Inference requires mapper and control checkpoints through `mapper.checkpoint_path` and `mapper.control_checkpoint_path`. Training presets also expect local dataset indexes, control features, caches, and initialization or resume checkpoints under their `data.*` and `output.*` fields.

## Testing

Run the smallest relevant test first. Keep the accelerator extra explicit so model-backed tests do not turn into dependency skips:

```sh
uv run --extra mps --group dev pytest -q tests/inference/test_hydra_config.py
```

Run the complete suite in separate processes to release Torch and accelerator memory between the major surfaces:

```sh
uv run --extra mps --group dev pytest -q tests/models
for test_file in tests/training/test_*.py; do
  uv run --extra mps --group dev pytest -q "$test_file" || exit
done
uv run --extra mps --group dev pytest -q \
  tests/data tests/evals tests/events tests/features tests/inference \
  tests/osu_core tests/timing tests/test_package_layout.py
```

Use `cuda` instead of `mps` on Linux.
Add `--extra render` when changing the optional Reamber-backed rendering surface.

## Canonical code and configuration

Use these paths before adding another constant, config field, or preset:

| Concern | Canonical source |
| --- | --- |
| Dependencies, extras, package data, test discovery | [`pyproject.toml`](pyproject.toml) |
| Mapper training presets | [`src/pulsefield_model/configs/hydra/training/mapper/`](src/pulsefield_model/configs/hydra/training/mapper/) |
| Inference service and selector groups | [`src/pulsefield_model/configs/inference/`](src/pulsefield_model/configs/inference/) |
| Typed training schema and preset projection | [`src/pulsefield_model/training/hydra_config.py`](src/pulsefield_model/training/hydra_config.py) |
| Training CLI and runner adapter | [`src/pulsefield_model/training/mapper_training_hydra.py`](src/pulsefield_model/training/mapper_training_hydra.py) |
| Typed inference schema and runtime projection | [`src/pulsefield_model/inference/config.py`](src/pulsefield_model/inference/config.py) |
| Inference CLI and service startup | [`src/pulsefield_model/inference/hydra_entry.py`](src/pulsefield_model/inference/hydra_entry.py) |
| WebSocket request handling and streaming | [`src/pulsefield_model/inference/ws_server.py`](src/pulsefield_model/inference/ws_server.py) |
| Mapper profile metadata and protocol contracts | [`src/pulsefield_model/inference/mapper_protocol.py`](src/pulsefield_model/inference/mapper_protocol.py) |
| Public token manifest | [`src/pulsefield_model/inference/hitobject_token_manifest_v2.json`](src/pulsefield_model/inference/hitobject_token_manifest_v2.json) |

Mapper inference YAML selects a profile. `MAPPER_PROFILE_SPECS` owns profile-derived checkpoint defaults, bundle IDs, aliases, model-family metadata, vocab and grammar contracts, and protocol compatibility.

## Guides and workflows

Use [`AGENTS.md`](AGENTS.md) as the repository-wide entrypoint for coding-agent
task routing. The other resources are human-facing analysis guides and examples;
they are not prerequisites for understanding the repository.

| Purpose | Resource |
| --- | --- |
| Repository documentation and Markdown math conventions | [`docs/guides/documentation_authoring.md`](docs/guides/documentation_authoring.md) |
| Root-cause analyses, performance investigations, and postmortems | [`docs/guides/technical_analysis_writing.md`](docs/guides/technical_analysis_writing.md) |
| MPS memory and throughput investigations | [`docs/engineering/mps_memory_performance_troubleshooting.md`](docs/engineering/mps_memory_performance_troubleshooting.md) |
| Worked MPS memory root-cause analysis | [`docs/research/mapper_v2_1_mps_memory_root_cause_report.md`](docs/research/mapper_v2_1_mps_memory_root_cause_report.md) |

This README remains the shared reference for project orientation and runnable
commands.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/pulsefield_model/features/` | Audio, mel, and control-feature extraction |
| `src/pulsefield_model/timing/` | Timing providers, grid fitting, rendering, and diagnostics |
| `src/pulsefield_model/data/` | Dataset indexes and training windows |
| `src/pulsefield_model/models/control/` | Control encoders and losses |
| `src/pulsefield_model/models/mapper/` | Mapper models, vocabularies, tokenizers, grammar, replay, and generation |
| `src/pulsefield_model/inference/` | Runtime loading, model bundles, sessions, streaming, protocol translation, and `.osu` export |
| `src/pulsefield_model/training/` | Training runners, Hydra projection, resume logic, and overnight wrappers |
| `tests/` | Unit tests and focused integration tests |
| `artifacts/agent-notes/` on `agent-notes` | Git-tracked working decision and research history; absent from product branches and not a product source of truth |
| Other `artifacts/` | Ignored local evaluations, reports, checkpoints, caches, and run snapshots; not a repository source of truth |
| `ref-proj/` | Reference projects used for comparison, never as authority |

Populate the optional reference-project submodules only when that comparison work is needed:

```sh
git submodule update --init --recursive
```

## Research status

The mapper remains the main quality bottleneck. Current evidence shows sparse output, repetitive patterns, and rhythm placement that can be structurally valid while remaining musically off-grid. Decode-policy work has recovered event density in selected evaluations, but it has not established production-quality chart structure.

Timing research includes structural recognition of shapes such as BPM ramps, but a parsed timing-grid result is not audio-ground truth. Treat `.osu` red timing as a diagnostic comparator rather than a musical oracle, and treat runtime measurements as hardware- and checkpoint-specific.

## Current research focus

1. Structured generation from rhythmic representations.
   Instead of collapsing music directly into a fixed beat grid, we investigate a staged generation process: music → rhythmic representation → parallel, context-aware candidate proposal → autoregressive chart realization. The goal is to preserve rhythmic ambiguity while still producing coherent discrete note placements.

2. Player-centric latent representations of chart style.
   We investigate whether chart style can be represented in terms of latent gameplay states: how different dimensions of player demand are distributed, combined, and evolved throughout a chart. Such a representation could provide a more meaningful basis for style and difficulty conditioning than global labels or scalar difficulty alone.

These directions are connected but remain hypotheses rather than committed architecture.

## License

Pulsefield Model is licensed under the GNU Affero General Public License v3.0 only (`AGPL-3.0-only`). See [`LICENSE`](LICENSE).

The projects under `ref-proj/` retain their upstream licenses.
