# Repository Guidance

- Put implementation code under `src/pulsefield_model`.
- Organize source by stable responsibility: `osu_core`, `features`, `data`,
  `events`, `timing`, `models`, `training`, `inference`, and `evals`.
- Keep model architecture versions under `models/mapper/v*` when their contracts
  differ materially.
- Keep stage labels in configs, artifact paths, run names, and reports instead
  of creating `stage_*` source packages.
- Treat `ref-proj/` as reference code only; production imports should not depend
  on it.
