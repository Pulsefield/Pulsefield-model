---
name: pulsf-trim-cot-leakage
description: Audit or fix Pulsefield prose whose viewpoint depends on an authoring session, chat, review, branch, uncommitted plan, or unexplained local evidence. Use for dead session citations, publication narration, reviewer choreography, derivation transcripts, planning residue, and indexical version language. Route general clarity, documentation coverage, and prose structure to pulsf-prose-standard.
---

# Trim Chain-of-Thought Leakage

Rewrite repository prose so a reader can resolve every reference and verify every claim without the originating session. This is a semantic audit: search patterns find candidates, and the surrounding contract determines the fix.

## Apply the unaided-reader test

Ask:

> Can a reader at `HEAD`, without the originating chat, hidden reasoning, review thread, uncommitted plan, terminal history, or unowned local artifact, resolve every reference and verify every claim?

Replace session references with their actual owner. Product-branch prose may rely only on an owner available from that product commit or a stable external source; the separate `agent-notes` branch is a valid owner only inside an Agent Note. Promote or summarize a required fact before removing a product-branch citation to note or local evidence. Restate surviving facts from the repository's viewpoint. Delete a passage only when it carries no load-bearing fact.

Preserve every relevant actor, condition, ordering rule, modality, negative guarantee, ownership fact, failure, consequence, measurement, and uncertainty. Apply [`pulsf-prose-standard`](../pulsf-prose-standard/SKILL.md) for broader editorial judgment after the leakage-specific audit.

## Leakage classes

1. **Dead session citations:** audit codes, numbered decisions, uncommitted plan sections, “as discussed above,” or “the user requested.” Name the resolvable owner or remove the citation while preserving its factual clause.
2. **Publication viewpoint:** “this PR,” “later in the stack,” “on this branch,” or “the previous commit” in product or technical prose. State the current mechanism; place pending work in an owned issue or TODO.
3. **Change narration:** “used to,” “no longer,” “now,” “today,” “for now,” or draft-version stamps. State present behavior or a useful present counterfactual. Keep history in a surface that owns history.
4. **Review choreography:** reviewer attributions, round numbers, and defensive self-justification. Keep the decision, invariant, and rationale.
5. **Derivation transcripts:** branch walkthroughs, test narration, proofs of adjacent code, or prose that reads back configuration. Keep only a non-obvious contract, observation method, or consequence.
6. **Planning residue:** unowned hedges, internal completion lists, candidate-selection narration, and research workflow labels in finished analyses. Replace them with measured bounds, owned follow-up, evidence, or interpretation.
7. **Unowned evidence references:** terminal scrollback, temporary paths, notebook positions, local run labels, or “the result above.” Record the evidence and provenance in an Agent Note on its owning branch or a curated product report. Product prose must carry the needed conclusion without depending on the note branch.
8. **Authoring scaffolding:** prompt markers, chat roles, private separators, or accidental working-language fragments in otherwise finished prose.

## Preserve valid context

Keep accurate, appropriately placed:

- issue references and actionable TODO tags;
- commit or PR evidence in Agent Notes, technical analyses, and postmortems;
- stable experiment, checkpoint, model, protocol, and schema identifiers;
- external papers, standards, datasets, and named design sources;
- suppression, skip, and exception reasons;
- measured bounds with their environment, units, sample, and uncertainty;
- runtime old/new objects participating in one live transition;
- historical stages inside a section that owns the history;
- exact repository-relative evidence paths inside a scoped Agent Note on `agent-notes`.

## Scope and workflow

Require an explicit file or directory scope. Audit requests are read-only; fix, trim, or rewrite requests authorize edits within that scope.

Exclude generated evaluations, checkpoints, datasets, run snapshots, fixtures, recorded outputs, notebook outputs, `ref-proj/`, caches, and frozen archived Agent Notes from broad audits. An exact archived Note may be audited when named, but a prose-fix request does not override its lifecycle freeze; defer mutation to an explicitly authorized restore through `pulsf-archive-agent-notes`.

1. Read `AGENTS.md`, the target's surrounding prose, and the source or evidence that owns each claim.
2. Run the [recall batteries](references/recall-batteries.md), then read dense prose without a search phrase in mind.
3. Before any deletion or rewrite, classify the candidate, enumerate the original and proposed propositions, and check the overcorrection traps in [the calibrated examples](references/examples.md): modality, hypothesis versus shipped fact, surviving facts inside narration, provenance, and negative guarantees.
4. Rewrite the owner before generated or copied representations. Preserve visible strings verbatim unless the task includes their behavior change and verification.
5. Re-run the probes and classify every remaining hit as valid context or unresolved leakage.
6. Run the smallest checks for changed behavior and `git diff --check`.

Use the remaining calibrated examples when a historical statement, measurement, or planning label needs further calibration.
