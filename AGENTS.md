# Pulsefield Coding Agent Guide

## Start here

Use `README.md` for the project purpose, supported environments, setup,
runtime commands, repository map, and canonical source locations. Inspect the
relevant canonical source and nearby tests before editing.

## Task routing

Use task-specific guidance only when its scope matches the work.

| Task | Resource |
| --- | --- |
| Agent Note content and lifecycle under `artifacts/agent-notes/` | `.agents/skills/pulsf-archive-agent-notes/SKILL.md` |
| Repository prose writing, review, trimming, restoration, or comment and documentation coverage | `.agents/skills/pulsf-prose-standard/SKILL.md` |
| Evidence-backed simplification surveys, dead or duplicate surface audits, and scoped cleanup proposals | `.agents/skills/pulsf-find-simplifications/SKILL.md` |
| Outgoing-diff test selection, pre-push evidence, force-with-lease safety, or readiness claims | `.agents/skills/pulsf-pre-push-checks/SKILL.md` |
| Authoring-session, review, PR, change-narration, or reasoning-transcript leakage in durable prose | `.agents/skills/pulsf-trim-cot-leakage/SKILL.md` |
| Packaged Hydra configs, mapper training presets, inference profiles, config adapters, CLI entrypoints, or Hydra tests | `.agents/skills/hydra-conventions/SKILL.md` |
| ML research direction, analogue search, hypothesis branching, bounded experiment design, or result evaluation | `.agents/skills/research-triage/SKILL.md` |
| Root-cause analyses, performance investigations, or postmortems | `docs/guides/technical_analysis_writing.md` |
| MPS memory or throughput investigations | `docs/engineering/mps_memory_performance_troubleshooting.md` |

## Repository context

- Treat `artifacts/` as local state. Do not scan or load it broadly unless the
  user names a specific artifact. Manage `artifacts/agent-notes/` through its
  lifecycle skill.
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
