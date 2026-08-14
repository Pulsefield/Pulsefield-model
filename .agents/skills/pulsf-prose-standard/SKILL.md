---
name: pulsf-prose-standard
description: Use when writing, reviewing, trimming, restoring, or auditing prose in Pulsefield Model across README and docs, skills and agent instructions, research records, Python comments and docstrings, tests, Hydra YAML comments, diagnostics, prompts, protocol text, and CLI-visible strings. Preserve complete technical contracts, add missing coverage, remove reasoning transcripts and repetition, and route durable facts to their owning source.
---

# Pulsefield Prose Standard

Write enough to preserve the contract, then remove reasoning transcripts, repetition, decoration, and stale authoring context. Treat a contract as any obligation, invariant, precondition, postcondition, ownership rule, failure behavior, timing rule, or compatibility promise that a caller, operator, implementer, producer, or consumer relies on.

Use this skill for editorial judgment and required prose coverage. Use [`technical_analysis_writing.md`](../../../docs/guides/technical_analysis_writing.md) for the structure and evidentiary standard of root-cause analyses, performance investigations, and postmortems. Use [`hydra-conventions`](../hydra-conventions/SKILL.md) when prose describes Hydra ownership or projection. Use [`research-triage`](../research-triage/SKILL.md) for planning artifacts rather than finished technical narrative.

Prefer the exact API, field, type, validation, phase, counter, component, or failure state over vague nouns such as “shape,” “surface,” “seam,” or “boundary.” Keep an abstract term when it names the precise technical subject, such as a process configuration boundary or a protocol compatibility contract.

Write comments only for non-obvious contracts or rationale the code cannot express. Do not restate control flow, names, or adjacent configuration.

## Require scope and authority

Require an explicit `scope`. If none is provided, report that requirement and stop; do not infer a repository-wide prose sweep.

Accept `mode: automatic | interactive`, defaulting to `automatic`. Enter interactive mode only when the user explicitly asks for calibration or questions. Mode controls how to resolve genuine tradeoffs, not whether edits are authorized: review and audit tasks report findings, while explicit write, fix, restore, or trim tasks apply clear scoped changes.

Before judging prose, read:

1. `AGENTS.md` and the relevant sections of `README.md`.
2. The task-specific skill or guide named by repository routing.
3. The owning source, configuration, and nearby tests.
4. The target document's surrounding section and links.

Do not inspect unrelated branches.

## Respect Pulsefield source boundaries

- Exclude broad `artifacts/` discovery. Inspect only a specifically named artifact needed as evidence.
- Treat generated evaluations, caches, checkpoints, datasets, and run snapshots as derivative local state, not prose authority.
- Use `ref-proj/` only when comparison is explicitly in scope, and describe it as reference evidence rather than Pulsefield authority.
- Put reusable constraints and durable conclusions in curated `docs/`; keep raw measurements and run transcripts out of the final narrative unless a concise excerpt is necessary evidence.
- Edit an owning source before a generated or copied representation, then regenerate with the repository-owned command. Do not hand-edit a derivative artifact to make prose appear consistent.
- Treat stable CLI errors, protocol text, prompts, and other model- or user-visible wording as behavior. Validate the affected behavior instead of calling the edit “docs-only.”

## Preserve the complete proposition

Before editing a passage, enumerate its factual propositions. Preserve every relevant:

- actor and action;
- condition, timing, phase, and ordering;
- modality such as must, may, can, or never;
- negative guarantee and exception;
- ownership and transfer of responsibility;
- side effect, failure mode, and consequence;
- measurement provenance, units, environment, and confidence bound.

Remove adjectives, repetition, and narration only when every load-bearing clause survives and the result is clearer. A smaller word count is not an improvement by itself.

Keep the local contract at the point of use: state the behavior, failure, ownership, and consequence a caller or maintainer needs there. Link to the owning document for extended rationale, architecture, algorithms, history, or examples. One detailed explanation has one home; essential contract facts may repeat where readers must act on them.

Keep non-obvious rationale when omitting it would plausibly cause misuse, an invalid optimization, or an incorrect simplification. Otherwise state the consequence and link the rationale owner.

## Cover each prose surface appropriately

This is not a one-way shortening pass. Add or restore prose when code, types, tests, and structure do not communicate the required contract.

- **Public Python docstrings:** state caller-visible return distinctions, raised errors, mutations, ownership, timing, device or dtype constraints, and durability. Do not narrate private helpers.
- **Internal comments:** explain invariants, tensor or buffer ownership, protocol translation, state-machine ordering, concurrency, numerical assumptions, and surprising failure behavior. Delete branch-by-branch walkthroughs.
- **Module comments:** orient the module's role, dependencies, responsibilities, and non-obvious architecture choices. Link extended rationale to its owner.
- **Tests:** explain only non-obvious fixture design, platform accommodation, real entry path, indirect observation, or why an assertion pins a contract. Let the test body show the steps.
- **README and module guides:** state ownership, setup, real commands, configuration semantics, failure modes, fresh-clone limitations, canonical sources, extension points, and evidence-scoped quality claims. Do not turn a README into a cleanup backlog.
- **Skills and agent instructions:** state triggers, scope limits, sources of truth, stop conditions, behavioral guardrails, and verification. Keep guidance compact enough to apply; do not defend every rule at length.
- **Experiment Cards and result logs:** preserve hypotheses, controls, metrics, guards, stop conditions, observations, and interpretation. Keep planning labels inside planning artifacts rather than exporting them into a finished analysis.
- **Technical analyses and postmortems:** make the question, environment, measurement model, evidence, causal confidence, uncertainty, and version scope self-contained. Separate findings from operational decisions and proposed next work.
- **Hydra YAML and configuration comments:** explain non-obvious ownership, composition order, override semantics, side effects, compatibility behavior, or likely misuse. Let the configuration show its own inventory.
- **Diagnostics:** name the failing subject or path, the violated rule, and the expected correction when useful. Remove internal execution narration.
- **CLI, prompt, and protocol strings:** treat exact wording and field names as behavior. Preserve searchable mechanism names and validate snapshots or focused behavior tests when available.

## Workflow

1. Confirm scope, mode, edit authority, current branch or PR base, and applicable instructions.
2. Read the prose owner and the code, config, or test evidence needed to verify every claim. Use [the calibrated examples](references/examples.md) for unfamiliar or disputed cases.
3. Inspect the entire requested scope, not only its largest files. Use targeted searches and word counts for discovery, then judge passages semantically.
4. Classify each candidate as `keep`, `add`, `trim`, `restore`, `restructure`, or `defer`. Do not manufacture edits to satisfy a deletion target.
5. Update the owner before derivatives. Re-check analogous passages after learning a reusable rule.
6. Run the smallest relevant tests or documentation checks, then `git diff --check`. Validate behavior for visible strings and configuration semantics.
7. Report the inspected scope, changes, deliberate keeps, deferred cases, source-boundary exclusions, and checks actually run.

## Resolve borderline decisions

Treat a case as borderline only when two or more versions preserve every proposition but trade accepted principles. A passage with one clearly complete version is not borderline.

In automatic mode, apply clear authorized edits and report genuine borderline cases without weakening a proposition to make progress.

In interactive mode, group analogous passages by principle. Present two or three viable versions, recommend one, and state the factual or structural difference. Do not include an inferior option merely to fill a list.

When a decision yields a reusable rule, add a short project-grounded calibration case to `references/examples.md`. Record the principle, not the review conversation or PR history.
