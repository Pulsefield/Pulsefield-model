---
name: pulsf-pre-push-checks
description: Inspect the final outgoing Pulsefield diff, select the smallest credible local evidence, and publish with safe Git history handling. Use immediately before pushing, force-pushing, marking a PR ready, or claiming local checks pass. Use ordinary implementation or debugging workflows until a concrete outgoing scope exists.
---

# Pulsefield Pre-Push Checks

Validate the change that will actually leave the worktree. Pushing, rewriting history, or changing PR state requires authorization from the active task.

## Establish the outgoing scope

Confirm the repository, branch, and worktree:

```sh
git status --short --branch
git rev-parse --show-toplevel
git branch --show-current
```

A detached `HEAD` needs an explicit destination. Verify the live PR base, stack parent, or user-named base, then record the merge base and committed scope:

The orphan `agent-notes` branch has no product base or merge base. For that destination, fetch and compare against the observed remote-tracking OID when it exists, or the empty tree on its first publication, then follow the Agent Note checks below. Do not apply the product-branch merge-base commands to it.

```sh
git merge-base <verified-base-ref> HEAD
git diff --name-status <verified-base-ref>...HEAD
git diff --stat <verified-base-ref>...HEAD
```

Include intended staged, unstaged, and untracked files. Read the substantive diff and affected callers. Recompute the scope after a rebase or base merge.

## Keep Agent Notes out of product history

For every destination except `refs/heads/agent-notes`, stop if the outgoing commit range ever touches `artifacts/agent-notes/` or the destination tree tracks it. Check history as well as the final diff so an add-then-delete cannot hide the violation:

```sh
git rev-list <verified-base-ref>..HEAD -- artifacts/agent-notes/
git ls-tree -r --name-only HEAD artifacts/agent-notes/
```

For `refs/heads/agent-notes`, verify that it has no merge base with `main` and that every path in both the outgoing commits and final tree is either `.gitignore` or Markdown under `artifacts/agent-notes/`. Inspect every outgoing version of each Note and run an available history-aware secret scan across all outgoing blobs; on first publication, cover the entire branch history. Any prohibited content in history blocks publication even when the final tree deletes it. It has no product PR or product test claim; publish the branch directly only when explicitly authorized.

## Select evidence from affected behavior

Every behavior change needs a focused check that would fail on its regression. Use `rg` to find owning tests and inspect their assertions before treating a pass as evidence.

- **Skills and agent instructions:** run `scripts/quick_validate.py` from the installed `skill-creator` skill for each changed skill, inspect linked resources and UI metadata, then run `git diff --check`. If the validator or its runtime is unavailable, check frontmatter, naming, links, and YAML syntax separately and report the missing canonical validation rather than treating the fallback as equivalent.
- **Packaging or packaged resources:** run `tests/test_package_layout.py` and inspect `pyproject.toml` package-data rules.
- **Hydra configuration or projection:** run the owning inference or training Hydra test; add `--help`, `--cfg job`, or the documented training dry-run when the operator path changes.
- **Models, data windows, vocabulary, grammar, replay, loss, or training:** run the exact owners under `tests/models/`, `tests/data/`, and `tests/training/` that consume the changed contract.
- **Inference runtime, bundles, streaming, protocol, or export:** run the owning `tests/inference/` files and add `tests/events/` or `tests/osu_core/` when those contracts change.
- **Timing, diagnostics, rendering, or evaluation:** run the owning `tests/timing/` or `tests/evals/` files.
- **README, curated docs, and visible strings:** verify claims against their code owner and run behavior tests for commands, configuration, protocol, diagnostics, or prompts.

Use the segmented full-suite commands in `README.md` only for a genuinely cross-cutting diff or an explicit full-rehearsal request. Do not repeat a passing check unless the outgoing content, base, environment, or relevant generated output changed.

Use a fresh temporary output directory for a training dry-run so stale files cannot make it pass or overwrite another run:

```sh
uv run --extra <mps-or-cuda> python -m pulsefield_model.training.mapper_training_hydra --dry-run output.output_dir="$(mktemp -d)" output.resume_from=null
```

Do not manufacture green evidence by deselecting a failing owner, relaxing an assertion, changing a test into a skip or expected failure, accepting zero collected tests, or narrowing a command below the affected behavior. Such a test change is part of the substantive diff and needs its own contract justification.

## Use the matching environment

- Apple Silicon model paths: `--extra mps`
- NVIDIA Linux model paths: `--extra cuda`
- Reamber-backed rendering: add `--extra render`

Keep `--group dev` on pytest commands. CPU-only work is suitable for skill, documentation, and configuration-shape checks that do not import model-backed modules.

Report unavailable checkpoints, datasets, audio, services, renderers, or hardware as evidence gaps. Do not convert a missing asset into a passing claim, and do not claim CUDA coverage from MPS or the reverse.

## Handle failures and mutations

Stop publication when selected evidence fails. Record the exact command and failure; distinguish a product defect from a proven platform or missing-asset limitation. A hook or required check may be bypassed only with explicit authorization.

After a formatter, hook, generator, commit, rebase, or conflict resolution changes relevant files, inspect the mutation and rerun only invalidated evidence.

## Protect rewritten history

Before an authorized force-push, fetch the remote branch and record its exact OID. Use an exact lease:

```sh
git push --force-with-lease=<branch>:<observed-remote-oid> origin HEAD:<branch>
```

Never use raw `--force`. After publication, verify the remote branch OID equals local `HEAD`. Rewritten heads invalidate earlier commit anchors, approvals, and validation; re-establish the outgoing scope, rerun invalidated evidence, and re-check review and CI state against the published head.

## Verify publication and readiness

After any push, confirm the exact remote OID and inspect the live PR when one exists. Verify its base and head, draft or ready state, required checks including pending jobs, unresolved review threads, approvals or change requests, and mergeability. Do not infer readiness from a successful push or local checks, and do not mark a PR ready or change review state without task authority. If the published OID differs from the tested OID, stop the readiness claim and validate the actual remote head.

## Report the result

State:

- branch, verified base and merge base (or verified orphan state), and outgoing paths;
- affected behavior and selected checks;
- exact commands and results;
- evidence gaps and deliberately untested surfaces;
- worktree state after validation;
- local and remote OIDs when publication occurred;
- live PR base/head, readiness, required-check, review, and mergeability state when applicable.

Say “selected local checks pass” unless a known complete local inventory ran successfully.
