---
name: pulsf-archive-agent-notes
description: Manage Git-tracked Pulsefield Agent Notes on the dedicated agent-notes branch. Use to create, update, review, accept, implement, reject, archive, restore, consolidate, or delete research, decision, investigation, implementation, simplification, and process records that must persist across agent turns. Keep notes out of product branches and route durable product contracts to their owning code, configuration, tests, or curated documentation.
---

# Manage Pulsefield Agent Notes

Agent Notes preserve working decisions, research direction, evidence interpretation, and implementation state across agent turns. Invoke this workflow from a product worktree; the dedicated notes worktree is storage, not a standalone task entrypoint. Notes are Git-tracked on the orphan ref `refs/heads/agent-notes`, never on `main` or another product branch. That ref has no merge base with `main` and is checked out in one linked worktree containing only its branch-local `.gitignore` and `artifacts/agent-notes/**`. The note branch owns note identity, content, lifecycle, and history; it does not define shipped product behavior.

## Keep notes off product branches

- Default read-only review to the committed `agent-notes` ref. Inspect an uncommitted worktree draft only when the task names it, and label it as uncommitted. Locate the registered worktree with `git worktree list --porcelain`; never hard-code a machine-local path. Mutations require that dedicated worktree.
- Do not create or edit `artifacts/agent-notes/` in a product worktree, stage note paths with product changes, merge or cherry-pick note commits into a product branch, or open a note-branch PR against `main`.
- Keep the note branch limited to its `.gitignore` and Markdown files under `artifacts/agent-notes/**`. Do not use it as a shadow branch for code, configuration, tests, curated docs, or generated run data.
- If the local ref is absent, report that note storage is not initialized. A missing worktree blocks mutation but not read-only inspection of an existing ref. Creating the orphan branch or linked worktree is a separate repository-setup action and requires explicit authority; follow [the bootstrap procedure](references/bootstrap-agent-notes.md) and do not fall back to an ignored local note.
- Product prose must be self-contained from its own commit. Promote a reusable architecture, operator, protocol, or analysis contract to its product owner rather than making a product branch depend on the note branch.

Product branches retain the blanket `artifacts/**` ignore. The [bootstrap procedure](references/bootstrap-agent-notes.md) owns the exact branch-local allowlist.

Raw evaluations, checkpoints, datasets, caches, and run snapshots stay in their owning artifact location and are not committed as notes. A note may link a stable artifact identifier and repository-relative path, but must summarize the observation needed to understand the decision when that artifact is absent.

## Respect action authority

- **Review, audit, or classify:** read-only. Report the lifecycle or consolidation change that would be appropriate; do not edit, move, delete, commit, or publish.
- **Create, record, persist, or update:** authorizes the scoped note-body and metadata edits and the local `agent-notes` commit needed to make them tracked. It does not authorize a lifecycle change, changes to related notes, product-branch edits, or a remote push unless the request also names that action. The sole automatic status consequence is the required `accepted -> proposed` reset when an authorized material revision changes accepted content.
- **Accept:** requires explicit human approval of the named note at its current note revision. Research output, an agent recommendation, passing evidence, or note creation is not acceptance.
- **Mark implemented:** requires an accepted direction, evidence that its completion condition was met, and an explicit request to mark the Note implemented. Code implementation, an experiment run, or result evaluation never implies this transition. **Reject:** requires an explicit human decision, or a request to advance the lifecycle plus unambiguous satisfaction of an already-recorded objective kill criterion.
- **Archive, restore, consolidate, or delete:** requires an explicit request for that lifecycle action and the audits below.
- **Publish:** requires explicit remote authority. Push `agent-notes` directly only when requested; never open a PR from it to `main`. Routine note publication never force-pushes. History repair requires separately authorized security recovery and remote coordination.

## Canonical layout and metadata

Store one Markdown file per note, using a lowercase kebab-case filename that stays stable across lifecycle moves:

```text
artifacts/agent-notes/
  proposed/YYYY-MM-DD-topic.md
  accepted/YYYY-MM-DD-topic.md
  implemented/YYYY-MM-DD-topic.md
  rejected/YYYY-MM-DD-topic.md
  archived/YYYY-MM-DD-topic.md
```

Every note begins with:

```text
# Agent Note: <decision or research question>

Note ID: YYYY-MM-DD-topic
Status: proposed | accepted | implemented | rejected | archived
Kind: research | architecture | implementation | simplification | investigation | process
Created: YYYY-MM-DD
Updated: YYYY-MM-DD
Product revision: <full product commit OID; add dirty-state provenance when relevant>
Scope: <files, subsystem, experiment, or decision boundary>
```

`Note ID` equals the filename stem and is immutable after its first commit. Use IDs in `Supersedes`, `Superseded by`, and `Related`. `Accepted revision` is the full note-branch commit OID whose content the human approved. `Acceptance reference` names a durable issue, review, or instruction ID when one exists; otherwise record the accepting human, date, and approved scope self-containedly rather than citing “the chat above.” Add these fields, `Archived`, `Archived from`, and `Archive reason` only when they apply. Use repository-relative paths, immutable commit OIDs, and stable experiment or run identifiers. Do not record credentials or tokens, personal or private data, raw dataset samples, checkpoints, generated payloads, large logs, or absolute home-directory paths.

Use [the Agent Note template](templates/agent_note.md) for a new record.

Every Markdown file in a lifecycle directory is one complete Agent Note with this metadata. Embed an Experiment Card and append Result Logs as sections of their owning research Note; do not create sibling card or log files in lifecycle directories.

## Content contract

An active note states:

- the question or decision and bounded scope;
- relevant product revision, repository evidence, and exact external or run provenance;
- live alternatives or hypothesis branches;
- selected direction and rationale when decided;
- risks, constraints, falsifying evidence, and reconsideration conditions;
- verification or evaluation required for the next lifecycle transition.

Research content names the closest analogues, dataset slice, baseline, intervention, metric, guard, runtime bound, confounders, and interpretation limits when they affect the decision. Separate observations from inference. Update defeated claims while retaining earlier reasoning only when it explains a decision or prevents repetition of a failed path.

Dirty product state may support a proposed investigation, but not acceptance, `SUPPORTED` research evidence, or an implemented claim. Those states require clean, recoverable full product commit OIDs; a Note cannot make uncommitted product code durable by describing it.

## Apply lifecycle transitions exactly

Allowed forward transitions are `proposed -> accepted | rejected`, `accepted -> implemented | rejected`, and `implemented | rejected -> archived`. A material revision resets `accepted -> proposed` for renewed approval; restoration is the narrow correction described below. A proposed draft may be deleted only under the deletion contract. Status records the scoped decision's history; later product replacement does not make a historically implemented decision unimplemented.

- **Proposed:** the direction is open. Record alternatives, evidence gaps, and acceptance criteria. It carries no project decision.
- **Accepted:** the human approved the exact proposed-note commit. The acceptance commit may only move the file and update `Status`, `Updated`, `Accepted revision`, and `Acceptance reference`; it must not fill in or alter the approved decision or protected content. If required content is missing, revise and commit the proposed Note, then seek approval again. Any material change to the decision or an owning workflow's protected field returns the Note to `proposed`, increments any owned resource revision, and removes current acceptance metadata before renewed approval.
- **Implemented:** the accepted change or experiment is complete, its stated verification was checked, and the Note cites the clean product commit or stable result identity that supplies the evidence. Replace planned actions with verified results while retaining rationale, interpretation limits, and remaining gaps.
- **Rejected:** the direction was explicitly declined. An agent may infer rejection only when an already-recorded objective kill criterion is unambiguously met; otherwise recommend the transition. State the deciding evidence and reconsideration condition.
- **Archived:** an implemented or rejected note has no active follow-up but retains future decision value. Add the date, reason, and `Archived from` state. Its committed body and links are a frozen snapshot; only an explicitly authorized restore or delete operation may move or remove it.

Move the file and update its status metadata in the same change. Restoration corrects an erroneous or premature archive: return only to `Archived from`, remove archive metadata, update `Updated`, and record why the archive no longer applies. Reconsidering a settled decision creates a new proposed note linked by ID; it does not rewrite the old status.

## Archive by future decision value

Do not archive or delete by age, length, apparent completion, or directory count. Retain a note when it still carries a rationale, constraint, failed alternative, operational consequence, verification fact, negative guarantee, or reconsideration condition likely to affect future work. Archive only when active follow-up is finished and that history remains useful. Delete only when no such unique value remains.

## Prove supersession and repair links

Search active notes before creating a new one. Compare decision, mechanism, scope, alternatives, evidence, consequences, and reconsideration conditions rather than titles.

- **Full supersession:** the successor or product owner preserves every current rationale, constraint, consequence, rejected alternative, verification fact, negative guarantee, and reconsideration condition. Link the successor to the predecessor; backlink an active predecessor only when predecessor mutation is authorized. Apply an appropriate lifecycle transition only with its required authority, and delete any status only under the deletion contract.
- **Partial supersession:** each note retains an independent decision or live constraint. Cross-link the ownership split and keep both active.
- **No supersession:** adjacent notes answer different questions. Keep their scopes distinct.

Lifecycle moves do not change a note ID, so use IDs rather than status-directory paths for note-to-note references. Repair active references after consolidation or deletion; archived links remain part of their frozen snapshot. Before deletion, inspect every inbound reference. A frozen archived reference is safe only when it includes an immutable note-commit locator; otherwise deletion requires explicitly authorized restore, repair, and rearchive of the referrer. A supersession audit does not authorize predecessor mutation; update it only when the request includes supersession, consolidation, or lifecycle cleanup. A product owner may mention an immutable note commit as historical provenance, but required product behavior and rationale must remain understandable without checking out the note branch.

Delete a Note of any status only within an explicitly named note or bounded deletion scope, after proving that no unique future decision value or unresolved inbound reference remains. Commit authorized reference repairs and deletion together, report the last commit containing the note, and never reuse its ID. Recover through a new commit under the same ID. Ordinary deletion never rewrites history; if prohibited private material entered history, stop and request a separately authorized purge procedure.

## Workflow and verification

1. Confirm the named note, product revision, requested action, and whether the task is read-only or mutating.
2. Verify the `agent-notes` ref, its lack of a merge base with `main`, and its tree allowlist. For mutation, also verify its dedicated worktree and stop on an overlapping uncommitted note change; reject a product-worktree fallback.
3. In the invoking product worktree, read `AGENTS.md`, the relevant product source and tests, directly related notes, and exact named evidence. Include archived references when deletion or link integrity is in scope; do not scan unrelated artifacts.
4. For a mutating request, apply only the authorized content and lifecycle changes. Audit overlap, future value, transition prerequisites, and references before any move or deletion; keep review-only work read-only.
5. Before committing, verify that the note-branch `HEAD` has not advanced, the directory matches `Status`, IDs are unique, references resolve, prohibited content is absent, and only scoped paths are staged.
6. Re-read the final note for status, product revision, evidence, scope, links, and unresolved uncertainty. Run `git diff --check -- artifacts/agent-notes/`, commit the scoped operation, and verify the worktree. Push only when separately authorized.

Report files changed, lifecycle transitions, acceptance evidence, supersession decisions, promoted product facts, repaired links, commit or publication status, and unresolved gaps.
