# Pulsefield Model

Pulsefield-model is the research backend for Pulsefield: a fast osu!mania-oriented
beatmap generation stack intended to serve interactive play, not just offline map
export. The current research target is 4K osu!mania-like generation in the 2-6
star range, using audio timing, learned control features, and a grammar-constrained
mapper.

Status: experimental. The repository contains runnable model/runtime code,
training/eval scripts, local checkpoints, and artifact reports, but the current
mapper is not yet a production-quality beatmap generator.

## Ultimate Goal

The end goal is a Pulsefield + Pulsefield-app experience:

1. Listen to any music.
2. Recognize or load the audio.
3. Infer timing and synchronize to playback.
4. Generate 2-6 star high-quality osu!mania-like beatmaps fast enough for
   real-time play.
5. Stream playable hitobjects to the client with enough lead time that the user
   experiences "listen, recognize, sync, then play" instead of waiting for an
   offline batch generator.

This repo owns the model side of that loop: audio features, timing fitting,
control memory, mapper inference, `.osu` export, and the local WebSocket endpoint
used by the client.

## Future Roadmap

Current exploration is focused on making global map planning more measurable,
more musically grounded, and cheaper to decode. These are research directions,
not settled claims.

- Canonical beat-based representation: move core map representation away from
  raw seconds and toward beat-relative positions. `.osu` hitobject timestamps are
  limited to 1 ms resolution, and the current second-based quantization can add
  avoidable error before the model even sees the musical grid. A beat-based
  representation should make timing, subdivisions, phrase structure, and BPM
  changes easier to compare across maps.
- Latent planner: replace the current hand-authored control feature planner with
  a learned latent planning layer. The existing control fields are useful for
  inspection, but they may bottleneck map structure into features that are easy
  to name rather than features that best predict playable chart flow.
- Tokenization and embedding research: study existing high-quality maps as
  structured objects, not just event streams. The goal is to learn better token
  units, embeddings, and structural priors from the inner organization of real
  maps: beats, measures, anchors, repetitions, long-note phrases, hand balance,
  and difficulty progression.

The planner and tokenization work are coupled. If the representation exposes
meaningful musical units and the planner operates over learned latents instead
of hand-crafted fields, global planning can become measurable against real map
structure rather than only against local density proxies. A better global plan
may also reduce decoder burden by letting the decoder fill in constrained local
details instead of rediscovering structure token by token.


## Pipeline

The intended inference path is:

```text
audio file
  -> mel features
  -> BeatThis timing prediction
  -> GridFitter timing grid
  -> dense timing features
  -> control encoder memory
  -> mapper decoder
  -> grammar/replay validation
  -> hitobject tokens / .osu export / WebSocket stream
```

Important artifacts and checkpoints in this workspace:

- v2 mapper checkpoint:
  `artifacts/runs/stage2_mapper_v2/stage2_mapper_v2_phase_b_global_d768_l8_b1/checkpoint.pt`
- v2.1 sparse mapper checkpoint:
  `artifacts/runs/stage2_mapper_v2_1/stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2/checkpoint.pt`
- v2.1 step 44k checkpoint:
  `artifacts/runs/stage2_mapper_v2_1/stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2/checkpoints/checkpoint_step_044000.pt`
- control checkpoint:
  `artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt`
- control v3 feature artifact metadata:
  `artifacts/features/control_v3_artifact_metadata_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.json`

Fresh clones may not contain all local datasets, caches, and checkpoints.

## Known Limitations

These limitations are copied from local artifact reports.

- Mapper quality is not solved. The frozen v2.1 44k real held-out rollout
  completed 11/11 windows but produced 10 empty windows, only 6 lane actions,
  and an 86.75 second longest event gap.
  See `artifacts/reports/evals/mapper_v21_44000_decoder_postmortem_2026-05-20.md`.
- Greedy no-TS decode has two observed failure modes: synthetic zero-control can
  hit the token cap after an early repetitive pattern, while real control can
  fast-forward through almost all windows and emit only a late chord burst.
- EOS was not confidently learned in that postmortem. At the exact chart end,
  pre-grammar EOS rank was 23 with probability 0.00103; EOS won only after
  grammar masking.
- A later PR2 decode policy recovered density on the Riria eval song
  (`lane_action_count=722`, no dead end, no token cap), but it remained a policy
  candidate rather than a quality solution: timepoints and pattern repetition
  were still high.
  See `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_decode_policy_report.md`.
- The best PR2 export was structurally valid but rhythmically off-grid against
  the reference: starts aligned perfectly to the 10 ms grid, not to normal beat
  subdivisions; only 11.4% of starts were within 5 ms of a 1/16 grid.
  See `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_timing_drift_audit.md`.
- Runtime claims are hardware- and snapshot-limited. MPS profiling in the
  postmortem showed grammar-mask cost growing from 54 ms at prefix 16 to
  2648 ms at prefix 1024. Current code has incremental decode support, but
  grammar/replay work remains a first speed target.
- Timing is useful but not robustly solved. A local timing-oracle slice reported
  mixed mean phase error of 36.14 ms and multi-BPM mean phase error of 46.15 ms;
  the report explicitly does not support claiming robust multi-BPM accuracy.
  See `artifacts/evals/timing_oracle_result_log.md`.
- BeatThis checkpoint choice materially changes fitted timing grids:
  108/183 comparable maps had material oracle phase divergence and 97/183 had
  segment-count divergence across checkpoints.
  See `artifacts/evals/timing_oracle_checkpoint_divergence_result_log.md`.
- Raw downbeat-weighted fitting should not be advertised as an improvement. In
  the 160-map eval it worsened mean phase from 47.780 ms to 48.160 ms and had
  more material regressions than improvements.
  See `artifacts/evals/timing_oracle_downbeat_weighted_grid_result_log.md`.
- `.osu` red timing is a noisy comparator. It is useful for diagnostics, not a
  production oracle for musical correctness.

## Quickstart

This section is only for initializing a development environment on a new
machine. The repository does not publish trained checkpoints, datasets, caches,
or generated artifact reports. Any command that loads a model, resumes training,
or runs inference requires those local files to be supplied separately.

Use `uv run python`, not plain `python`, because some machines do not expose a
`python` command.

Common setup:

```sh
uv --version
ffmpeg -version
```

### Apple Silicon Mac

Use this path for local development with the PyTorch MPS runtime.

```sh
uv sync --extra mps --group dev
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
uv run pytest -q
```

### Linux With NVIDIA CUDA

Use this path for CUDA-capable Linux machines. The CUDA extra uses the
`pytorch-cu128` index configured in `pyproject.toml`.

```sh
nvidia-smi
uv sync --extra cuda --group dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest -q
```

### Windows

Native Windows is not the primary target. Use WSL2 Ubuntu with NVIDIA GPU
passthrough, then follow the Linux CUDA path. Keep datasets and local artifacts
inside the WSL filesystem for better IO behavior.

### CPU-Only Or Docs Work

Use this path for lightweight editing, documentation, and non-model checks.
Model runtime code still needs either the `mps` or `cuda` extra on matching
hardware.

```sh
uv sync --group dev
```

## Repository Map

- `src/pulsefield_model/features/`: mel/audio features and control feature
  extraction.
- `src/pulsefield_model/timing/`: BeatThis provider integration, grid fitting,
  dense timing rendering, and timing diagnostics.
- `src/pulsefield_model/models/control/`: control encoder models.
- `src/pulsefield_model/models/mapper/`: mapper vocabularies, tokenizers,
  grammar, replay, model code, and generation engines.
- `src/pulsefield_model/inference/`: runtime loading, session caching,
  WebSocket serving, streaming, rollout, and `.osu` export.
- `src/pulsefield_model/training/`: control and mapper training entrypoints.
- `tests/`: unit and focused integration tests.
- `artifacts/reports/` and `artifacts/evals/`: local experiment reports and
  postmortems used to bound claims in this README.
- `ref-proj/`: local reference projects used for analogy and comparison, not
  authority.
