---
name: pulsf-archive-agent-notes
description: Use when adding, reviewing, consolidating, rejecting, deleting, restoring, or archiving Pulsefield Agent Notes or explicitly named decision and research-planning records. Audit supersession, classify records by future decision value rather than age or size, preserve durable rationale and guardrails, repair inbound links, and refuse to invent archive paths, lifecycle states, manifests, or sealing commands when the repository has not defined them.
---

# Archive Pulsefield Agent Notes

Reduce stale active decision context without erasing history that can still guide work. Judge each record semantically. Age, length, status labels, and implementation completion are discovery signals, never sufficient archive criteria.

This skill supplies retention judgment and archive workflow. It does not create a note system. Pulsefield currently distinguishes curated documentation from planning material and ephemeral artifacts, but it may not have a lifecycle-managed Agent Note corpus in every checkout.

## Establish the contract and scope

Require an explicit scope: named records, a note directory, or the records related to one decision. Do not turn a scoped request into a repository-wide archive sweep.

Read, in order:

1. `AGENTS.md` and `README.md`.
2. The lifecycle or archive instructions nearest the scoped records, if any.
3. The applicable task skill or documentation guide.
4. Current code, packaged configuration, tests, and curated docs that may own the decision.
5. Inbound links and newer records that may supersede it.

Respect Pulsefield's source boundaries:

- Treat `artifacts/` as ephemeral local state. Do not scan it broadly unless the user names a specific artifact.
- Treat generated evaluations, caches, checkpoints, datasets, and run snapshots as evidence inputs, not repository authority.
- Use `ref-proj/` only when comparison is explicitly in scope, and never as authority for a Pulsefield decision.
- Put durable conclusions in curated `docs/`; do not preserve a result only by leaving it in an Experiment Card, notebook, chat-derived note, or local run log.

If the scoped checkout has no Agent Note lifecycle contract, say so. Classify the named material and recommend a destination when useful, but do not fabricate `.agents/notes/` paths, status values, language pairs, sidecars, hashes, manifests, seals, or verifier commands. Creating such a system is separate design work and requires explicit authorization.

## Audit supersession while adding a record

Every new decision record triggers a scoped audit of active records covering the same decision, mechanism, constraint, or rejected alternative. Perform the audit while writing the new record rather than leaving a known collision for a later cleanup.

Identify the current owner from shipped code, canonical packaged configuration, public contracts, curated docs, newer records, and inbound links. Titles and dates help discovery but do not prove ownership.

Classify overlap as:

- **Full supersession:** the new owner carries every still-relevant rationale, alternative, consequence, negative guarantee, reintroduction condition, verification fact, and named gap. No surviving behavior, format, compatibility obligation, or independent guardrail remains in the old record.
- **Partial supersession:** any distinct behavior, constraint, rationale, rejected alternative, or compatibility fact remains current. Keep both records and cross-link the division of ownership.
- **No supersession:** the records are adjacent but answer different decisions. Leave both active and clarify their boundaries only when the task authorizes edits.

Do not delete a record merely because its implementation was later removed. Consolidate it only after production code, configuration, schemas, serialized formats, migrations, docs, and supported tests all agree the feature is absent, and the current owner preserves why it existed, why it was removed, what was given up, and when reintroduction would be reasonable.

## Classify by future decision value

Apply the repository's defined lifecycle first. When it does not resolve retention, use these outcomes:

- **Implemented — keep active:** retain the record when it still owns architectural rationale, an ownership or import boundary, a data or wire contract, a negative guarantee, an operator constraint, a security rule, a non-obvious failure mode, or a condition for revisiting the decision.
- **Implemented — archive:** archive only when the shipped work is complete, current code and docs are authoritative, and the record's remaining content is unlikely to change a future decision. One-off experiment coordination, a narrow closed adapter choice, or implementation history with no durable constraint may qualify.
- **Proposed — keep or reject, never archive:** keep a live proposal active. If it is no longer worth pursuing, reject it honestly under the local lifecycle contract.
- **Rejected — keep as a guardrail:** retain the rejection when the losing idea remains plausible and tempting, and the record explains why it loses under current constraints.
- **Rejected — delete:** delete only when the idea is obsolete, superseded, impossible under current architecture, or no longer useful for preventing re-litigation. Repair or remove every inbound link.
- **Evidence-only material — promote or discard:** move durable findings into a self-contained curated document before retiring a card, log, or notebook narrative. Do not archive raw evidence as though it were the conclusion.

Never archive toward a quota. Inspect every record in scope, group analogous decisions under one principle, and report genuinely close cases.

## Calibrated Pulsefield examples

- Keep a concise record that explains why Hydra and OmegaConf stop at process entrypoints; the boundary constrains future runtime modules even if the current implementation is stable.
- Keep a rejected proposal to duplicate mapper profile metadata in YAML when that duplication remains an attractive shortcut and the record explains why `MAPPER_PROFILE_SPECS` must own it.
- Archive a completed one-off migration plan after canonical configs, compatibility behavior, and operator docs fully describe the result and the plan contains no unique rollback or reintroduction condition.
- Retire an Experiment Card only after its durable result, environment, measurement definition, uncertainty, and conclusion live in a self-contained report or the result has been explicitly rejected as non-reusable.
- Do not archive a root-cause report merely because the incident is old. Keep it active while its measurement model, causal evidence, version scope, or operational constraint remains useful.
- Delete a rejected idea whose premise depended only on an obsolete local artifact, provided no current code, config, test, or doc still relies on it and no plausible future mistake needs the warning.

These examples set the reasoning bar; they do not pre-classify similarly named files.

## Apply archive mechanics exactly

When a repository-local archive contract exists:

1. Move the complete record unit defined by that contract. Keep paired files, metadata, and indexes atomic.
2. Make only the metadata edits required for archival. Do not opportunistically rewrite, translate, reformat, or update facts in a frozen historical snapshot.
3. Recompute sidecars, hashes, indexes, or manifests mechanically with the owning command.
4. Search active prose for inbound links. Redirect readers to current authority, retarget the archived record only when history is intentionally cited, or remove the link.
5. Run the focused archive verifier before broader checks.

After a record is sealed or declared immutable, do not edit, move, reformat, or delete it outside the contract's explicit recovery procedure. Archived material may remain a historical link target, but it is not authority for current behavior.

When no archive contract exists, stop before moving files. Report the semantic classification and the missing mechanics instead of improvising an archive directory.

## Restore carefully

Restore an archived record only when the local contract allows it and the decision has become active again. Re-read current owners first: restoration may require a new record that cites the historical snapshot rather than mutating or unsealing it. Preserve the archive's integrity and make current authority explicit.

## Validate and report

Run the smallest checks owned by the touched surfaces, then `git diff --check`. For curated documentation, verify links and run any repository-defined documentation test. For skill changes, run the skill validator. Never claim an archive seal or outbound-link guarantee that the available verifier does not establish.

Report:

- scope and lifecycle contract used;
- implemented records kept and why;
- implemented records archived;
- proposals kept or rejected;
- rejected records kept as guardrails or deleted;
- evidence promoted before retirement;
- full and partial supersessions;
- inbound links repaired;
- borderline cases and their deciding tradeoff;
- checks actually run and any missing archive machinery.
