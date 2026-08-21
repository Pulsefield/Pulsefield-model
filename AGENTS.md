# Pulsefield Coding Agent Guide

## Start here

Use `README.md` for the project purpose, supported environments, setup,
runtime commands, repository map, and canonical source locations. Inspect the
relevant canonical source and nearby tests before editing.

## Task routing

Use task-specific guidance only when its scope matches the work.

| Task | Resource |
| --- | --- |
| Git-tracked Agent Note content and lifecycle on the separate `agent-notes` branch | `.agents/skills/pulsf-archive-agent-notes/SKILL.md` |
| Repository prose writing, review, trimming, restoration, or comment and documentation coverage | `.agents/skills/pulsf-prose-standard/SKILL.md` |
| Evidence-backed simplification surveys, dead or duplicate surface audits, and scoped cleanup proposals | `.agents/skills/pulsf-find-simplifications/SKILL.md` |
| Outgoing-diff test selection, pre-push evidence, force-with-lease safety, or readiness claims | `.agents/skills/pulsf-pre-push-checks/SKILL.md` |
| Authoring-session, review, PR, change-narration, or reasoning-transcript leakage in durable prose | `.agents/skills/pulsf-trim-cot-leakage/SKILL.md` |
| Packaged Hydra configs, mapper training presets, inference profiles, config adapters, CLI entrypoints, or Hydra tests | `.agents/skills/hydra-conventions/SKILL.md` |
| ML research direction, analogue search, hypothesis branching, bounded experiment design, or result evaluation | `.agents/skills/research-triage/SKILL.md` |
| Root-cause analyses, performance investigations, or postmortems | `docs/guides/technical_analysis_writing.md` |
| MPS memory or throughput investigations | `docs/engineering/mps_memory_performance_troubleshooting.md` |

## Repository context

- Product branches ignore `artifacts/` and never track `artifacts/agent-notes/`.
  Agent Notes are Git-tracked only on the orphan `agent-notes` branch and are
  managed through their lifecycle skill. Never use a product worktree's
  ignored directory as a note store.
- Do not scan generated artifacts broadly unless the user names one. The notes
  branch owns note history, not shipped product behavior.
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
