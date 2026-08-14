---
name: pulsf-pre-push-checks
description: Use before pushing or force-pushing a Pulsefield Model branch, marking a PR ready, claiming local checks pass, or publishing a correction after a rebase. Inspect the complete outgoing diff against its verified base, select the smallest credible tests and repository checks for affected behavior, keep MPS/CUDA/render extras explicit, handle missing checkpoints or platform coverage honestly, protect rewritten history with an exact force-with-lease, and verify remote and PR state after publication.
---

# Pulsefield Pre-Push Checks

Run relevant local evidence once before publishing a Pulsefield change. Select checks from the outgoing behavior, not from habit or file count. Use focused tests for focused changes and the segmented full rehearsal only when the diff is genuinely cross-cutting or the user requests it.

This skill selects and evaluates evidence. It does not itself authorize commits, pushes, force-pushes, PR state changes, hook bypasses, dependency installation, or external messages. Obtain the authority required by the active task before performing those actions.

Pulsefield's committed checkout does not currently define a universal lint, typecheck, coverage, pre-commit, or pre-push command. Inspect the current repository and local hook configuration rather than assuming one exists. CI may own platform and checkpoint coverage that a local machine cannot reproduce; pending or unavailable evidence must remain pending or unavailable in the report.

## Inspect the outgoing change

Confirm the checkout, branch, and repository root:

```sh
git status --short --branch
git rev-parse --show-toplevel
git branch --show-current
```

Stop if the branch, worktree, or repository differs from the requested target. A detached `HEAD` needs an explicit publication plan; do not guess a destination branch.

Verify the live PR base, stack parent, or user-named base before comparing. Do not assume `main`, `origin/main`, or the current branch's configured upstream is the review base. Fetch the verified ref when current remote state matters, then inspect committed scope from the resolved merge base:

```sh
git merge-base <verified-base-ref> HEAD
git diff --name-status <verified-base-ref>...HEAD
git diff --stat <verified-base-ref>...HEAD
```

Include unpublished worktree state when the task intends to publish it:

```sh
git diff --name-status
git diff --cached --name-status
git ls-files --others --exclude-standard
```

Read the substantive diff and its callers, not only the path list. Record the base ref and merge-base OID used. After merging or rebasing onto a changed base, recompute the scope and rerun only the evidence invalidated by the combined change.

## Build an evidence map

Every behavior change needs the narrowest test or purpose-built check that would fail for its regression. Map changed owners to nearby tests before running commands.

- **Skills and agent instructions:** locate `scripts/quick_validate.py` through the installed `skill-creator` skill and run it for every changed skill; inspect references and UI metadata, then run `git diff --check`. If the canonical validator is unavailable, report that gap and inspect frontmatter and YAML structure without inventing a repository command. No repository-wide prose test is currently published.
- **Packaging and resource visibility:** run `tests/test_package_layout.py`; inspect `pyproject.toml` package-data rules when packaged YAML or JSON changes.
- **Inference Hydra and config:** run `tests/inference/test_hydra_config.py`; include `--help` or `--cfg job` when entrypoint or composition output changes.
- **Training Hydra and runner projection:** run `tests/training/test_mapper_training_hydra_config.py`; use the documented dry-run when projection, resume, output artifacts, or runner consumption changes.
- **Mapper models, vocabulary, grammar, replay, batching, or generation:** run the owning file under `tests/models/mapper/` and add training or inference tests only for contracts those paths consume.
- **Control model or feature extraction:** run the focused files under `tests/models/control/`, `tests/features/`, and the affected data-window tests.
- **Inference runtime, bundles, streaming, protocol, or export:** run the named tests under `tests/inference/`; include `tests/events/` or `tests/osu_core/` only when their public contract changes.
- **Timing providers, fitting, canonicalization, diagnostics, or rendering:** run the owning `tests/timing/` files. Add `--extra render` when the optional Reamber-backed path changes.
- **Training runner, checkpoints, split logic, or overnight wrappers:** run the exact `tests/training/test_*.py` owners in separate processes when Torch or accelerator memory retention matters.
- **Evaluation helpers:** run the owning `tests/evals/` file and state which dataset, checkpoint, renderer, or hardware evidence remains unavailable.
- **README and curated docs:** verify commands, links, source ownership, and nearby claims against current code; run behavior tests when prose changes CLI, protocol, config, or model-visible wording.
- **Notebooks:** validate notebook structure and the specific code path changed; do not execute a costly or data-dependent notebook merely because its JSON changed.

Use `rg` to locate tests importing the changed symbol and callers relying on the changed string, field, or file. Test discovery is evidence selection, not evidence by itself. Inspect selected tests before treating their pass as coverage of the outgoing behavior.

Do not use skipped tests, empty collections, relaxed assertions, or a narrowed selection to hide an affected path. When one focused test cannot exercise a shared module, add the other owning tests or narrow the behavioral claim only when excluded modules cannot be affected.

## Choose the environment explicitly

Use the platform extra that matches the machine for model-backed commands:

- Apple Silicon: `--extra mps`
- Linux with NVIDIA CUDA: `--extra cuda`
- Reamber-backed rendering changes: also add `--extra render`

CPU-only setup is suitable for documentation, skill validation, and configuration-shape work that does not import model-backed modules. Do not let a missing accelerator extra turn an intended test into a skip or import failure, and do not claim CUDA coverage from an MPS run or the reverse.

Keep `--group dev` explicit for pytest. Prefer the exact command forms documented in `README.md`, for example:

```sh
uv run --extra mps --group dev pytest -q tests/inference/test_hydra_config.py
uv run --extra mps --group dev pytest -q tests/training/test_mapper_training_hydra_config.py
uv run --extra mps --group dev pytest -q tests/test_package_layout.py
```

Replace `mps` with `cuda` on the supported NVIDIA Linux environment. If checkpoint, dataset, audio, credential, service, or optional renderer input is unavailable, separate the tests that passed from the evidence that could not run. A fresh clone does not include the model and data assets required for full inference or training.

## Validate configuration and operator paths

Treat successful raw Hydra composition as necessary but not sufficient when typed semantic validation or runtime projection changes. Select the relevant operator path:

```sh
uv run --extra mps python -m pulsefield_model.inference.hydra_entry --cfg job
uv run --extra mps python -m pulsefield_model.inference.hydra_entry --help
uv run --extra mps python -m pulsefield_model.training.mapper_training_hydra --help
pulsf_dry_run_dir="$(mktemp -d /tmp/pulsefield-dry-run.XXXXXX)"
uv run --extra mps --group dev python -m pulsefield_model.training.mapper_training_hydra --dry-run output.output_dir="$pulsf_dry_run_dir" output.resume_from=null
```

Use a new explicit temporary output directory for a dry-run and report what it wrote. Do not point a validation command at an existing run directory or checkpoint tree.

## Use the full rehearsal sparingly

Run the complete local approximation only when the user asks, while reproducing broad CI failure, or when the change crosses enough independent owners that no smaller set is credible. Follow the repository's segmented commands so Torch and accelerator state are released between major surfaces:

```sh
uv run --extra mps --group dev pytest -q tests/models
for test_file in tests/training/test_*.py; do
  uv run --extra mps --group dev pytest -q "$test_file" || exit
done
uv run --extra mps --group dev pytest -q \
  tests/data tests/evals tests/events tests/features tests/inference \
  tests/osu_core tests/timing tests/test_package_layout.py
```

Replace `mps` with `cuda` on Linux. Add `--extra render` only when that optional surface belongs to the outgoing change. Do not describe this as an exhaustive platform, asset, or real-checkpoint suite.

## Avoid duplicate runs

Do not manually repeat a passing check merely because a commit or push follows. Rerun when the staged content changes, a formatter or hook modifies relevant files, the base changes, the environment changes, or the previous command did not cover the final outgoing diff.

After any commit-time mutation, inspect the changed files and remap evidence before continuing. Never assume local hooks ran a check that the repository does not define.

## Handle failures before publication

If relevant evidence fails before an ordinary push, stop. Fix the scoped defect when authorized or report the blocker; do not publish and hope another platform differs.

Prove an environment-specific failure before classifying it:

- record the exact command, failing test, platform, Python and Torch versions, device, and mismatch;
- confirm the relevant platform-independent evidence;
- distinguish missing local assets from a product failure;
- prefer fixing cross-platform nondeterminism when the check is required;
- bypass a hook or test only when the user explicitly authorizes it, and report exactly what failed and why remote evidence is expected to differ.

Do not leak credentials, tokens, local dataset paths, or checkpoint contents while reporting failures.

## Protect rewritten history

Use a normal push for ordinary history. Before an authorized standalone force-push, fetch the current remote branch and record its exact OID. Publish only with an exact lease:

```sh
git push --force-with-lease=<branch>:<observed-remote-oid> origin HEAD:<branch>
```

Never use raw `--force`, and do not use an unspecified lease when a concurrent update must abort the push. If the remote branch did not exist at observation time, verify that state explicitly and use the publication method appropriate to a new branch.

After any rewritten publication, fetch the live head again and re-audit unresolved review threads, approvals, mergeability, and checks. Old commit hashes, approvals, and inline-comment anchors are not current evidence. If a stack tool rewrites and publishes several branches as one operation, validate each rewritten layer against its live parent immediately afterward and keep every affected PR unmerged until the selected evidence passes.

## Publish and verify

When the active task authorizes publication:

1. Run the selected evidence against the final outgoing content.
2. Commit intentionally and inspect any files changed during commit.
3. Push normally, or use the exact lease for an authorized history rewrite.
4. Fetch or query the remote and verify its branch OID equals local `HEAD`.
5. Inspect remote PR checks and review state when a PR exists.

Report pending remote checks as pending. Inspect failures before attributing them to the branch, environment, or CI.

## Report the evidence honestly

Summarize:

- branch, verified base, merge base, and outgoing paths;
- behavior surfaces affected;
- commands run and exact results;
- checks deliberately omitted and why they cannot cover the diff;
- unavailable assets, hardware, services, or platform coverage;
- worktree cleanliness after validation;
- commit and remote OIDs when publication occurred;
- pending, passing, or failing remote checks and review state.

Say “selected local checks pass” when that is what the evidence proves. Reserve “all checks pass” for a known complete check inventory whose local and remote results are all final.
