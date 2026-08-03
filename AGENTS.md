# Pulsefield Coding Agent Guide

## Start here

Use `README.md` for the project purpose, supported environments, setup,
runtime commands, repository map, and canonical source locations. Inspect the
relevant canonical source and nearby tests before editing.

## Task routing

Use task-specific guidance only when its scope matches the work.

| Task | Resource |
| --- | --- |
| Packaged Hydra configs, mapper training presets, inference profiles, config adapters, CLI entrypoints, or Hydra tests | `.agents/skills/hydra-conventions/SKILL.md` |
| Open or still-diffuse ML research, novelty analysis, bounded experiment planning, or an accepted Experiment Card | `.agents/skills/research-triage/SKILL.md` |
| Root-cause analyses, performance investigations, or postmortems | `docs/guides/technical_analysis_writing.md` |
| MPS memory or throughput investigations | `docs/engineering/mps_memory_performance_troubleshooting.md` |

## Repository context

- Treat `artifacts/` as ephemeral local state. Do not scan or load it broadly
  unless the user explicitly names a specific artifact.
- Generated evaluations, caches, checkpoints, datasets, and run snapshots are
  not repository sources of truth and may be absent in a fresh clone.
- Put durable conclusions and reusable constraints in curated `docs/`
  documentation.
- Use `ref-proj/` only when comparison work is in scope, and treat it as a
  reference rather than authority.

## Verification

Run the smallest relevant check first, using the commands in `README.md`. Keep
accelerator extras explicit for model-backed commands: `mps` on Apple Silicon
and `cuda` on Linux with NVIDIA. CPU-only setup is suitable for documentation
and configuration work.
