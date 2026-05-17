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
# Current Status

You're migrating s3d-i's work from ~/projects/Mapperatorinator/ to here.
The migration aims to preserve the core original code bahaviour but into a new and better structure,
so module names, file names, code orgnization can change.
The migration successful criteria is it can be run and pass tests locally under this repo

## Naming convention

if you encounter codes that are similar in function but of different version, put "_<version>" suffix.