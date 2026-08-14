---
name: pulsf-trim-cot-leakage
description: "Use when auditing or fixing Pulsefield prose that reads like a leaked authoring or reasoning session: dead citations such as ‘decision 7,’ audit codes, uncommitted plan sections, or chat references; PR, stack, reviewer, or commit vantage in durable prose; ‘used to’ change narration and indexical version stamps; control-flow walkthroughs and self-justification; hedged planning residue; or pointers to ephemeral local artifacts. Preserve every factual proposition while rewriting comments, docstrings, docs, skills, research records, diagnostics, and visible strings from the repository's resolvable current-state viewpoint."
---

# Trim Chain-of-Thought Leakage

Treat chain-of-thought leakage as prose whose viewpoint belongs to the authoring session rather than the repository. It points to context a future reader cannot resolve, narrates how an answer or diff was produced, argues with a departed reviewer, or leaves planning scaffolding where a durable contract should stand.

Never use deletion alone when a suspect passage carries facts. Apply the complete-proposition rule from [`pulsf-prose-standard`](../pulsf-prose-standard/SKILL.md): preserve each actor, condition, timing rule, modality, negative guarantee, ownership fact, failure, consequence, measurement, and uncertainty; restate them so they stand at `HEAD`, then remove the transcript around them. Delete a passage outright only when it contains no load-bearing proposition.

This skill is guidance, not a mechanical word ban. Searches find candidates; semantic judgment decides them.

## Apply the unaided-reader test

Ask of every suspect passage:

> Can a reader at `HEAD`, without the originating chat, hidden reasoning, review thread, uncommitted plan, local-only artifact inventory, or branch stack, resolve every reference and verify every claim?

If not, replace session references with committed owners, restate surviving facts from the repository's viewpoint, or remove clauses that carry no facts. If yes, the passage clears this skill's resolvability test, but it may still be misplaced change narration: current-state surfaces such as README files, API docstrings, configuration comments, and operator guides normally state current behavior. Route a necessary change story to a decision record, technical analysis, postmortem, issue, or versioned release surface that owns history.

## Taxonomy

1. **Dead session citations:** `(decision 7)`, `(audit C2)`, `design §4.1`, `plan §2`, phase labels from an uncommitted checklist, “the ledger,” “as discussed above,” or “the user requested.” Replace a real owner with its searchable name and committed path. Otherwise remove the citation and keep its factual clause.
2. **PR, stack, branch, and commit viewpoint:** “this PR adds,” “the next PR,” “later in the stack,” “the previous commit,” or “on this branch.” State the shipped mechanism. Put deferred work in an issue or actionable TODO; keep publication choreography out of durable product prose.
3. **Change narration and indexical stamps:** “used to,” “no longer,” “the old implementation,” “now,” “today,” “for now,” “v1 of this document,” or “this cut” when contrasting repository states. State present behavior. Convert a useful fixed-regression fact to a present counterfactual such as “without the padding mask, invalid positions affect attention.”
4. **Review choreography and reviewer-addressed justification:** “rejected in review,” “the reviewer confirmed,” “this is safe because,” draft ordinals, or round attributions. Keep the decision, invariant, and strongest rationale; remove the participants and conversational defense.
5. **Restatement and derivation transcripts:** branch walkthroughs, line-by-line proofs, test setup narration, “first we do X, then Y,” or prose that repeats adjacent YAML. Delete it unless a non-obvious invariant or observation method survives.
6. **Hedges and workflow residue:** “probably fine,” “should be enough,” unowned “later” work, research-triage role labels, candidate-selection narration, and internal completion checklists in a finished analysis. Replace a hedge with the measured bound and failure behavior, promote real work to an owned TODO or issue, and translate research workflow into evidence and interpretation.
7. **Ephemeral evidence pointers:** unexplained `artifacts/...` paths, local run numbers, notebook cell positions, temporary filenames, terminal scrollback, or “the result above.” Summarize the evidence needed for the claim and record durable provenance. Do not make a curated doc depend on local state that a fresh clone lacks.
8. **Authoring-language and formatting slips:** untranslated working-language fragments, private separators, prompt markers, response scaffolding, or chat role labels embedded in otherwise finished prose. Translate a real term, name a durable external reference, or remove the residue.

## Keep resolvable and load-bearing prose

Do not overcorrect. Keep these when accurate and appropriately placed:

- **Issue references and actionable TODO tags:** they resolve at `HEAD` and can own follow-up work.
- **Commit or PR evidence in a postmortem, technical analysis, or lifecycle-managed decision record:** those genres may need a change story. Link a resolvable owner and keep the evidence relevant to the argument.
- **External standards, papers, datasets, and design sources:** an RFC section, DOI, paper figure, public benchmark, or named Figma frame may resolve outside the repository by design.
- **Stable experiment identifiers:** keep an identifier when a committed result log or curated report defines it. A local run label with no retained owner fails the test.
- **Suppression and exception reasons:** linter, type-ignore, coverage-ignore, platform-skip, and empty-catch justifications can be required contracts. Fix a false reason; do not delete the reason category.
- **Present counterfactuals:** “without X, Y happens” records why a guard exists without narrating repository history.
- **Measured bounds and provenance:** “measured on MPS with sequence length 4096” distinguishes observation from definition. Preserve units, environment, sample, and uncertainty needed to interpret it.
- **Runtime old/new states:** “the old session drains before the replacement starts streaming” names live objects in one transition, not past and present repository versions.
- **Model and protocol identifiers:** mapper `v2`, `v2_1`, bundle IDs, schema versions, checkpoint steps, and API `/v1/` paths are identifiers, not indexical draft stamps.
- **Historical stages inside a section that owns history:** a root-cause timeline or decision record may say an earlier experiment shipped or failed. Avoid indexical phrases such as “this cut” even there.
- **Project voice and alternatives:** “we” can express project policy, and an Alternatives section can say a design was rejected. Remove conversational attribution only when it adds no durable rationale.

Resolvability is necessary, not sufficient. A valid issue link can still be irrelevant; a current-state doc can still contain too much history; a precise comment can still restate code. Apply the prose standard after this leakage check.

## Respect scope and repository boundaries

Require the explicit scope and edit authority defined by `pulsf-prose-standard`. Audit read-only first. A request to review or audit does not authorize edits.

Read `AGENTS.md`, `README.md`, the applicable task skill or guide, the target's surrounding prose, and the owning source and tests before changing a claim.

Do not broadly scan or rewrite:

- `artifacts/`, generated evaluations, caches, checkpoints, datasets, or run snapshots;
- `ref-proj/` unless comparison is explicit;
- `.git/`, `.venv/`, cache directories, installed metadata, or generated binaries;
- recorded model output, fixtures, snapshots, or notebook outputs whose original voice is evidence;
- frozen archived decision records when a local archive contract declares them immutable;
- this skill's calibration examples while using its search batteries.

Inspect an exact excluded target only when the user names it or it is necessary evidence. Do not modernize a historical or generated record to make a repository-wide search reach zero hits.

## Workflow

1. Confirm the explicit scope, mode, edit authority, branch or PR base, and applicable instructions.
2. Read the prose standard and current owner. Determine whether the surface should state present behavior, preserve history, capture planning, or record raw evidence.
3. Audit read-only. Run the [recall batteries](references/recall-batteries.md) with `--hidden`, then judge every match. Probes intentionally over-match and inevitably miss cases, so also read the densest prose in scope without a search phrase in mind.
4. Classify each suspect passage by taxonomy and enumerate its propositions before editing. Use [the calibrated examples](references/examples.md) to check overcorrection traps.
5. Fix owner-first. Update source docstrings or generator templates before derivatives; update Hydra owners before copied examples; treat visible diagnostics, CLI text, protocol strings, and prompts as behavior requiring focused tests.
6. Replace dead citations with committed owners only when those owners actually contain the cited fact. Do not turn an unresolvable name-drop into a plausible-looking but false link.
7. Re-run the batteries. Inspect every remaining hit as a sanctioned keep, a calibration self-hit, or an unresolved defect. Confirm that links, issue numbers, experiment IDs, and external references resolve.
8. Run the smallest checks for touched surfaces, then `git diff --check`. Keep accelerator extras explicit for model-backed behavior tests.
9. Report scope, fixes, deliberate keeps, unresolved references, evidence promoted out of ephemeral state, checks run, and excluded surfaces.

## Guard against overcorrection

Before accepting a trim, compare the original and replacement proposition by proposition. Reject a rewrite that:

- turns an obligation or planned migration into approval of the current state;
- promotes a hypothetical design or research direction to shipped behavior;
- removes a true coupling, failure, or limitation with the transcript around it;
- drops measurement provenance while keeping the number;
- changes `must`, `may`, `never`, or an exception;
- converts a bounded local result into a general product claim;
- replaces an unresolved citation with an owner that does not support the claim.

The target is durable, verifiable prose—not merely prose that sounds more confident.
