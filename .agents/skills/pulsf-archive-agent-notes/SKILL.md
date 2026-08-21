---
name: pulsf-archive-agent-notes
description: Create, update, review, supersede, reject, archive, restore, or delete Pulsefield Agent Notes under artifacts/agent-notes/. Use for local research, decision, investigation, and implementation records whose content or lifecycle must persist across agent turns. Route generated run data to its owning artifact directory and repository documentation to its documentation workflow.
---

# Manage Pulsefield Agent Notes

Own the content and lifecycle of local Agent Notes under `artifacts/agent-notes/`. Notes preserve working decisions, research direction, evidence interpretation, and implementation state across agent turns. They are local records; shipped code, packaged configuration, tests, and curated documentation remain the repository sources of truth.

## Canonical layout

Store one Markdown file per note:

```text
artifacts/agent-notes/
  proposed/YYYY-MM-DD-topic.md
  accepted/YYYY-MM-DD-topic.md
  implemented/YYYY-MM-DD-topic.md
  rejected/YYYY-MM-DD-topic.md
  archived/YYYY-MM-DD-topic.md
```

Use lowercase kebab-case filenames. Keep the filename when a note changes status. Move the file between lifecycle directories and update its metadata in the same operation.

Every note begins with:

```text
# Agent Note: <decision or research question>

Status: proposed | accepted | implemented | rejected | archived
Kind: research | architecture | implementation | simplification | investigation | process
Created: YYYY-MM-DD
Updated: YYYY-MM-DD
Scope: <files, subsystem, experiment, or decision boundary>
```

Add `Supersedes`, `Superseded by`, `Archived`, and `Archive reason` only when they apply. Use repository-relative paths and stable experiment or run identifiers.

Use [the Agent Note template](templates/agent_note.md) for a new record.

## Content contract

An active note states:

- the question or decision;
- relevant repository state and evidence;
- live alternatives or hypothesis branches;
- selected direction and its rationale when decided;
- risks, constraints, and falsifying evidence;
- verification or evaluation needed for the next lifecycle transition.

Research content names closest analogues, dataset slice, baseline, metric, guard, runtime, confounders, and interpretation limits when those facts affect the decision. Exact paths to local runs, reports, checkpoints, or evaluations are valid evidence inside a scoped note.

Record observations separately from inference. Update a claim when newer evidence defeats it, and preserve the earlier interpretation only when it explains a decision or guards against repeating a failed path.

Tracked repository prose must not depend on an ignored Agent Note for a required contract. Promote reusable architecture, operator, protocol, or analysis conclusions to their curated owner when the task includes that publication work.

## Lifecycle

### Proposed

Use while the direction is open. Record alternatives, evidence gaps, and acceptance criteria. A proposed note carries no project decision.

### Accepted

Move to `accepted/` when the human owner selects the direction. State the selected option, bounded scope, expected evidence, and completion condition. Keep unresolved risks explicit.

### Implemented

Move to `implemented/` after the selected change or experiment is complete and its stated verification has been checked. Rewrite planned actions as the verified result, retain decision rationale and interpretation limits, and name remaining gaps.

### Rejected

Move to `rejected/` when the direction is declined or evidence crosses its kill criterion. State the deciding evidence and the condition that would justify reconsideration. Retain the note while the rejected path remains a plausible mistake or research branch.

### Archived

Move an implemented or rejected note to `archived/` when it no longer needs active attention but still carries useful history. Add the archival date and reason. Archived notes are read-only; restore a note before changing its body.

Restore an archived note to `proposed/`, `accepted/`, `implemented/`, or `rejected/` according to its current state. Remove archive metadata, update `Updated`, and record the evidence that reopened it.

## Supersession and consolidation

Search active notes before creating a new one. Compare the decision, mechanism, scope, alternatives, and evidence rather than titles alone.

- **Full supersession:** the new owner preserves every current rationale, constraint, consequence, rejected alternative, verification fact, and reconsideration condition. Link both notes. Archive an implemented or rejected predecessor; reject a displaced proposed or accepted direction, or delete a valueless draft.
- **Partial supersession:** each note retains an independent decision or live constraint. Cross-link the ownership split and keep both active.
- **No supersession:** adjacent notes answer different questions. Keep their scopes distinct.

Consolidate duplicate notes into the clearest current owner. Repair Agent Note inbound links after every move, rename, consolidation, archive, restore, or deletion.

## Deletion

Delete a proposed draft when it carries no unique evidence or decision value. Delete a rejected note when its branch is no longer plausible and no active note depends on its rationale. Delete an archived note only on explicit request. Repair inbound links before deletion.

## Workflow

1. Confirm the named note, decision, research question, or lifecycle scope.
2. Read `AGENTS.md`, current code, configuration, tests, curated docs, related Agent Notes, and named local evidence.
3. Create or update the content using the lifecycle contract above.
4. Audit overlap and inbound links before changing lifecycle state.
5. Move the file and metadata atomically.
6. Re-read the final note for status, evidence, scope, links, and unresolved uncertainty.

Report files created or changed, lifecycle transitions, supersession decisions, evidence promoted to another owner, links repaired, and unresolved decision gaps.
