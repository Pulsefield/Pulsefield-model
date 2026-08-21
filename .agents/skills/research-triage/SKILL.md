---
name: research-triage
description: Guide Pulsefield ML research through analogue search, hypothesis branching, bounded experiment design, and evidence-based result evaluation. Use for open or diffuse research questions, novelty assessment, experiment selection, an accepted Experiment Card, or interpretation of experimental results. Route established implementation, routine debugging, causal analysis, and Agent Note lifecycle work to their owning workflows.
---

# Pulsefield Research Triage

Turn an open research question into a grounded direction, a proposed discriminating experiment, or a defensible interpretation. Keep the human owner responsible for research direction, acceptance, execution authority, and adoption.

## Choose the research mode

- **Explore:** the question, novelty, mechanism, or useful implementation family is unclear. Find close analogues, branch plausible hypotheses, and identify evidence that separates them.
- **Design:** the direction is plausible but needs one bounded experiment. Select a branch and define the smallest test that can change the decision.
- **Evaluate:** results already exist. Check the planned comparison, guards, deviations, confounders, and scope before deciding what the evidence supports.

Implicit skill invocation selects the reasoning workflow; it does not authorize code changes, a run, card acceptance, discretionary note lifecycle transitions, or publication. Every mode may inspect the repository and author its scoped research output. When this workflow requires persistence, the active research task authorizes the scoped proposed-Note creation or evidence append and its local note-branch commit; it never implies acceptance, another status transition, or a remote push. An explicit request to revise a protected Card field authorizes only the required revision increment, `accepted -> proposed` reset, and local note commit; no other transition is implied. Change experiment code, implement instrumentation, or start a run only when the active task separately and clearly authorizes that action.

Use these transitions:

- Explore ends at `DROP`, `REFINE`, or `TEST`. `TEST` authorizes Design, not implementation or execution.
- Design produces a proposed Experiment Card inside one proposed owning Agent Note. The card becomes accepted only when a human explicitly accepts the exact note revision containing that card ID and revision through the Agent Note lifecycle.
- Evaluate ends at `DROP`, `REFINE`, `REPEAT`, or `SUPPORTED`. `REPEAT` reruns the same accepted card; changing a protected field returns to Design. `SUPPORTED` is bounded evidence and awaits human direction—it does not authorize adoption, merge, another run, or an Agent Note transition.

A request may span modes, but every new or revised card remains proposed until its exact revision is explicitly accepted.

## Ground the question

State the research question, current baseline, target improvement, and the decision the evidence should change. Separate:

- representation or objective changes;
- data, sampling, or supervision changes;
- model or decoding changes;
- evaluation or measurement changes;
- engineering changes that do not establish research novelty.

Prefer primary papers, project repositories, benchmark definitions, and reproducible local traces. Treat surveys and curated lists as discovery aids. Use [research workflow patterns](references/research_workflow_patterns.md) when analogue selection or evidence grading needs calibration.

## Find and compare analogues

Name the closest analogue for each serious branch. Compare the exact layer, assumptions, data, objective, evaluation, and operational constraints. A nearby name or shared component is not enough; explain the mechanism that transfers and the difference that remains.

After naming the closest analogues, present a source-bounded novelty assessment. Distinguish a new representation or learning objective from a new combination, adaptation, or implementation, and label the assessment provisional until the human owner adopts it.

## Branch the reasoning

Generate only branches that predict meaningfully different observations. For each branch record:

- claim and mechanism;
- closest analogue;
- expected supporting and falsifying signals;
- smallest discriminating probe;
- cost, confounders, and likely failure mode.

Compare branches under the same baseline and decision criteria. Merge equivalent branches and discard branches that no available observation can distinguish.

End exploration with one outcome:

- **DROP:** available evidence or repository fit defeats the direction;
- **REFINE:** a smaller or different question is needed;
- **TEST:** one bounded experiment can decide between live branches.

## Design one bounded experiment

Use [the Experiment Card](templates/experiment_card.md). Give it one owning Agent Note, an immutable card ID, and an incrementing revision. The owning note's lifecycle is the card's authority state. Define exactly:

- hypothesis, selected branch, and decision the result can change;
- closest analogues and remaining difference;
- clean baseline source revision, configuration or checkpoint, dataset slice, baseline run or evidence identity, and baseline value with aggregation, sample count, and uncertainty;
- one causal intervention, separate from behavior-neutral instrumentation;
- primary metric definition, direction, decision threshold, regression guard bound, and qualitative check;
- exact procedure, paired or independent comparison design, commands, seeds, environment, runtime and resource bounds, fresh output destination, overwrite or resume policy, and early-stop condition;
- confounders and positive, negative, and ambiguous interpretations.

Prefer a fail-fast slice over a full training run. The test must be capable of rejecting the selected branch. Any edit to the question, baseline identity or value, slice, causal intervention, metric, decision threshold, guard, procedure, comparison design, seeds, environment, resource bounds, output or resume policy, or stop condition increments the card revision, returns its owning note to `proposed`, and requires renewed acceptance.

## Preserve the execution handoff

The Experiment Card embedded in the accepted owning Note controls implementation and execution. Before implementation or a run, compare its current protected fields with the `Accepted revision`; any difference requires a revised proposed Card rather than silent continuation. Before code edits, require the implementation worktree at the accepted clean baseline OID or an explicitly authorized clean descendant whose existing diff is behavior-neutral to the comparison. Otherwise stop or return to Design. An implementation agent may add only necessary behavior-neutral instrumentation; a change that can affect the causal comparison belongs in the intervention and requires a revised card.

Implementation and execution are separate permissions unless the active task clearly includes both. Before a run, record a clean intervention source OID and audit its baseline-to-intervention diff against the accepted card; stop on an unrelated change or undeclared causal difference. Verify the accepted Note and card revisions, exact baseline and inputs, commands, seeds, environment, fresh output destination, overwrite or resume policy, compute, storage, network, and runtime bounds. Stop on the recorded guard, kill condition, authorization limit, or unavailable prerequisite. Return to Design and reacquire Note acceptance when a protected field must change. A run-only request appends reproduction, measurements, and plan-conformance fields, leaving Evaluation and Decision pending; it does not mark the owning Note implemented.

## Evaluate results

Use [the Result Log](templates/result_log.md). Verify the accepted Note and card revisions and record clean baseline and intervention source OIDs, separate or paired run IDs and commands, environment and hardware, seeds, inputs, configuration and checkpoint identities, output artifact paths, budget consumed, stop reason, and planned-versus-actual deviations. An Evaluate request fills or revises the Evaluation and Decision fields from recorded evidence. Append each Result Log; never overwrite earlier run evidence.

Compare baseline and intervention values under the planned metric and guard. Separate observed measurements from interpretation, and state the strongest alternative explanation still compatible with the data.

A run with no accepted Card or existing owner is exploratory: record it in a proposed owning Note and use `none` for unavailable acceptance or Card fields. When an accepted owner exists, append a dirty, unrecoverable, incomplete, or materially deviated run's exploratory Result Log to that Note without changing its status or Card. Never recommend `SUPPORTED` for these cases. Recommend `REPEAT` only for a conforming rerun of the same accepted Card; otherwise recommend `REFINE` and leave any Card revision to a subsequent Design task. Bound every conclusion to the tested source revisions, checkpoint, dataset slice, runtime, and measurement definition.

## Persist only when continuity requires it

Persist every proposed or accepted card, executed run, direction-changing result, or cross-turn research task in one owning Agent Note. Exploration-only brainstorming may remain ephemeral unless the user asks to retain it. Store raw run outputs in their artifact owner and link stable identifiers from the note. If required note storage is unavailable, return the proposed content and report the persistence blocker; never write an ignored fallback, claim card acceptance, or issue a final `SUPPORTED` recommendation before the required record is committed.

Use [`pulsf-archive-agent-notes`](../pulsf-archive-agent-notes/SKILL.md) for note storage and lifecycle on the separate `agent-notes` branch. Research recommendations never change note lifecycle by themselves. Publish reusable conclusions through the repository prose workflow.

## Output resources

- Use [the Idea Critique](templates/idea_critique.md) for Explore.
- Use [the Experiment Card](templates/experiment_card.md) for Design.
- Use [the Result Log](templates/result_log.md) for Evaluate.

These are content-section templates, not standalone files in a lifecycle directory. Embed one active Experiment Card in its owning research Note and append each Result Log beneath it. Load only the resource needed for the selected mode.
