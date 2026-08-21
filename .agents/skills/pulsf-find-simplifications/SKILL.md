---
name: pulsf-find-simplifications
description: Use when surveying Pulsefield Model for non-obvious simplification candidates, reviewing dead or duplicated code and configuration, folding worthwhile ideas from another branch, coalescing superseded decision records, or adding a focused TODO/FIXME/XXX. Find evidence-backed opportunities in code, configs, tests, docs, and workflows, especially unused surfaces, duplicate sources of truth, speculative generality, legacy shims, over-defensive lifecycle machinery, or hand-rolled infrastructure that a standard or existing dependency can replace.
---

# Find Pulsefield Simplifications

Turn a broad “simplify this” request into a few evidence-backed candidates that remove, fold, or demote real surface area. Follow the code and keep judgment active. Prefer one well-proven deletion over a list of guesses, and do not treat fewer lines as the objective when the smaller design weakens an owned contract.

This skill guides discovery and proposal quality. It does not authorize code edits, dependency changes, decision-record creation, branch housekeeping, or PR updates unless the user's request does.

## Start with Pulsefield context

Read `AGENTS.md`, `README.md`, the canonical-source table, and the nearest tests before classifying anything. Apply specialized guidance when the scope matches:

- Use [`hydra-conventions`](../hydra-conventions/SKILL.md) for packaged configs, mapper presets, inference profiles, config adapters, and Hydra entrypoints.
- Use [`research-triage`](../research-triage/SKILL.md) when the apparent simplification changes an open ML hypothesis or research direction rather than established architecture.
- Use [`pulsf-prose-standard`](../pulsf-prose-standard/SKILL.md) when simplifying comments or documentation.
- Use [`pulsf-archive-agent-notes`](../pulsf-archive-agent-notes/SKILL.md) only when lifecycle-managed decision records are actually in scope.
- Use [`technical_analysis_writing.md`](../../../docs/guides/technical_analysis_writing.md) when the requested output is a root-cause analysis, performance investigation, or postmortem.

Respect repository boundaries throughout discovery:

- Do not scan `artifacts/` broadly. Inspect a named generated artifact only when the user puts it in scope, and treat it as local evidence rather than authority. Read Agent Notes from their separate branch only when their decision scope is named; they own note history, not product behavior.
- Exclude `ref-proj/` unless comparison work is explicit. A reference project can suggest a question but cannot establish what Pulsefield should remove.
- Treat generated evaluations, caches, checkpoints, datasets, and run snapshots as derivative. Trace a finding back to committed source, config, tests, or curated docs.
- Treat README research-focus items as hypotheses, not committed architecture.

## Recognize a strong candidate

A strong candidate removes ongoing cognitive, compatibility, maintenance, runtime, or verification cost and has evidence that the current design buys less than it costs. Look for:

- a public function, config field, CLI flag, profile attribute, event, adapter method, helper, or package surface with no production consumer;
- behavior consumed only by tests or docs when those consumers do not protect a supported contract;
- two owners for one fact across packaged YAML, typed schema, runner dictionaries, defaults, profile specs, protocol metadata, or public manifests;
- version-specific mapper code duplicated in `shared/`, `v2/`, and `v2_1/` without a vocabulary, grammar, replay, checkpoint, or tensor-shape reason;
- an inference or timing adapter that repeats projection or validation already owned at the process boundary;
- a compatibility shim whose operator path, serialized data, checkpoint, or external consumer no longer exists;
- defensive copying, validation, rollback, or state tracking that protects only impossible or same-process typed inputs;
- a cache, index, expected output, or special-case test that exists solely for an unused feature;
- speculative generality with no product or research owner and no current caller;
- a custom parser, retry loop, framing helper, matcher, serializer, or numerical utility that the supported Python standard library or an existing healthy dependency covers with less glue;
- a slight behavior difference whose simpler result is still coherent, observable, and easier to explain.

Thin candidates do not justify durable proposals: typo fixes, import sorting, a one-line alias with compatibility value, a single static-analyzer warning, or “this file is complex” without caller and ownership evidence. Record a small local cleanup only when it is actionable and does not need a design decision.

## Protect intentional architecture

Do not relabel documented constraints as accidental complexity. Treat these Pulsefield boundaries as intentional unless new evidence defeats their rationale:

- Packaged YAML under `src/pulsefield_model/configs/` is canonical for mapper training and inference configuration.
- Hydra and OmegaConf stay at process entrypoints; runtime modules consume typed dataclasses, paths, or plain dictionaries.
- Inference mapper groups select profiles; `MAPPER_PROFILE_SPECS` owns profile-derived bundle, alias, model-family, vocabulary, grammar, protocol, and checkpoint metadata.
- Accepted config fields must reach runtime behavior or fail validation; silently dropping a field is not simplification.
- Token vocabularies, grammar rules, replay semantics, checkpoint shapes, and protocol translations may require separate mapper-version implementations.
- The Torch-free inference help path and package-resource visibility are operator contracts.
- Tests and docs are evidence, not automatic authority. Determine what behavior they protect before keeping or deleting it.

Do not preserve a boundary merely because a skill names it. Read the current implementation and tests; a strong candidate may show that the documented owner itself should change.

## Survey broadly before selecting

When the user requests breadth or many candidates, divide the survey among independent domains and require evidence from each. Useful Pulsefield domains include:

- training presets, typed schema, runner projection, resume behavior, and legacy CLI handling;
- inference config, mapper profiles, model-bundle registry, protocol translation, and public token manifest;
- WebSocket framing, endpoint/session lifecycle, supervisor ownership, cancellation, reset, and unload;
- timing providers, fitting, canonicalization, rendering, ramp diagnostics, and export;
- feature extraction, data windows, cache contracts, batching, and checkpoint input assumptions;
- mapper shared code versus `v2` and `v2_1` vocabulary, tokenization, grammar, replay, generation, loss, and model code;
- evaluation helpers, notebooks, root-level legacy configs, tests, and docs that may pin obsolete behavior.

Use parallel subagents for a genuinely broad survey. Give each a non-overlapping domain and require exact paths, symbols, callers, counterevidence, and a rejection decision—not a count target. If parallelism is unavailable, cover the same domains sequentially. Do not stop the survey after the first attractive candidate.

Start with large or repeated production surfaces rather than only obvious unused symbols. Duplicated lifecycle, projection, validation, and versioned model machinery often carries more cost than a dead helper.

## Audit trust, state, and ML boundaries

For each copy, validator, normalization, or conversion, name the input's origin and next owner. Parsers, Hydra composition, YAML/JSON, protocol envelopes, datasets, checkpoints, model outputs, workers, processes, and durable files cross trust or representation boundaries and normally require validation. A same-process typed handoff may not.

For asynchronous inference, draw the ownership graph. Map each task, lock, sentinel, queue, cancellation path, reset flag, disposer, and terminal state to a distinct transition or owner. Propose one lifecycle controller when several mechanisms mirror the same liveness or settlement fact. Preserve separate machinery when it protects publication ordering, first-terminal-outcome arbitration, resource quiescence, or process ownership.

For model and data code, track tensor shape, padding, dtype, device, vocabulary, grammar, and checkpoint compatibility before deduplicating. One successful local artifact or fixed-shape test does not prove a validation or adapter is redundant.

## Compare hand-rolled code with dependencies

Treat a dependency swap as a simplification candidate only after proving net deletion:

1. Name the exact custom behavior and dedicated tests the dependency replaces.
2. Check the Python version floor, supported platforms, accelerator extras, maintenance, adoption, license compatibility, and transitive footprint.
3. Prefer the standard library or an already-declared dependency when it meets the contract.
4. Account for residual adapters, error translation, configuration, packaging, and tests. A wrapper that relocates the same complexity is not a win.
5. Identify behavior differences explicitly and decide whether callers can accept them.

Do not add a dependency merely because it offers a similar API.

## Prove or reject every candidate

Classify every reference before proposing removal:

- **Production:** `src/pulsefield_model/`, packaged YAML and JSON, import and runtime entrypoints, model-bundle dispatch, and operator commands.
- **Non-production:** unit tests, curated docs, comments, and test-only fixtures.
- **Ambiguous:** notebooks, `src/pulsefield_model/evals/`, repo-root `configs/`, local scripts, compatibility loaders, and smoke paths. Read their invocation and ownership before assigning them.

Use `rg` first. Search the exact symbol, imported alias, config key, Hydra override, profile name, event or wire string, filename, checkpoint field, and both attribute and direct-call forms. Then read every meaningful caller. Static results do not expose dynamic registry lookup, Hydra composition, serialized compatibility, reflection, subprocess entrypoints, or external protocol consumers; inspect those paths directly.

Reject or downgrade a candidate when:

- a production caller exists and removal is a feature decision outside scope;
- the proposal breaks an owned config, protocol, checkpoint, vocabulary, grammar, or export contract;
- tests encode a real supported failure mode or boundary that production searches did not reveal;
- the only supporting evidence is an ephemeral or generated artifact;
- the change merely moves complexity or requires unrelated churn;
- the simplification depends on an untested research claim;
- the idea is correct but too small for durable design treatment.

Record rejected candidates during a broad audit so repeated searches do not turn the same weak clue into a later claim.

## Coalesce superseded records only when in scope

Audit decision records when the user requests consolidation or when an implemented simplification makes a scoped owner obsolete. Do not expand every code survey into a repository-wide documentation cleanup.

Use `pulsf-archive-agent-notes` for records under `artifacts/agent-notes/` on the separate notes branch. For full supersession, read each predecessor before the change and prove that the final successor preserves every unique rationale, alternative, consequence, verification fact, negative guarantee, and reintroduction condition before repairing links or recommending archival. An add-then-delete diff or shared title is not proof. Keep partial supersessions cross-linked.

## Write the result at the right scale

For each durable candidate, report:

- current surface and owner;
- exact production, non-production, and ambiguous consumers;
- maintenance or behavioral cost;
- proposed deletion, fold, demotion, or dependency swap;
- strongest reason to keep the current design;
- behavior and capability given up;
- affected code, config, tests, docs, and compatibility data;
- smallest verification that could falsify the proposal;
- risk, confidence, and any evidence gap.

Prefer a compact user-facing audit unless the user requests files. When a tracked working decision record is requested, use the Agent Note lifecycle on `agent-notes`; do not add it to the product branch. Put a repository-wide durable conclusion in an existing curated `docs/` location only when it is self-contained and the request authorizes documentation changes.

Use inline `TODO`/`FIXME`/`XXX` only for a small local cleanup with a clear action. Include a stable tag and the reason it is safe to revisit, for example `TODO(config-owner): remove the fallback after the legacy entrypoint is deleted`. Do not leave speculative complaints or hide a design decision in a TODO.

## Fold another branch carefully

When asked to fold simplification work from another branch, compare that branch with its verified base or stack parent, not automatically with the current branch. Port non-overlapping, well-proven candidates; merge overlapping evidence into the current owner; and drop duplicates or weaker variants rather than preserving a candidate count. Change PR state or close duplicate work only when authorized.

## Validate and hand off

Start with the smallest relevant check from `README.md`. For Hydra or config work, include composition, semantic validation, or the documented dry-run as appropriate. Keep `--extra mps` on Apple Silicon or `--extra cuda` on NVIDIA Linux for model-backed commands; add `--extra render` for Reamber-backed rendering. Run `git diff --check` for written changes.

Report the domains surveyed, candidates accepted and rejected, intentional exclusions, decision records added or consolidated, TODOs added, checks actually run, and unresolved evidence. Do not claim exhaustive coverage when generated state, external consumers, unavailable checkpoints, or hardware paths were not inspectable.
