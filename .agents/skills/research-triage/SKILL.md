---
name: research-triage
description: Guide Pulsefield ML research through analogue search, hypothesis branching, bounded experiment design, and evidence-based result evaluation. Use for research-direction choices, novelty assessment, experiment selection, or interpretation of experimental results. Route established implementation, routine debugging, causal analysis, and Agent Note lifecycle work to their owning workflows.
---

# Pulsefield Research Triage

Turn an open research question into a grounded direction, a discriminating experiment, or a defensible interpretation. Keep the human owner responsible for research direction and novelty claims.

## Choose the research mode

- **Explore:** the question, novelty, mechanism, or useful implementation family is unclear. Find close analogues, branch plausible hypotheses, and identify evidence that separates them.
- **Design:** the direction is plausible but needs one bounded experiment. Select a branch and define the smallest test that can change the decision.
- **Evaluate:** results already exist. Check the comparison, guards, confounders, and scope before deciding what the evidence supports.

Use ordinary repository implementation workflows for an accepted experiment. When the research content must persist as an Agent Note, use [`pulsf-archive-agent-notes`](../pulsf-archive-agent-notes/SKILL.md) for its file and lifecycle under `artifacts/agent-notes/`.

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

Make a novelty claim only after the closest analogues are explicit. Distinguish a new representation or learning objective from a new combination, adaptation, or implementation.

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

Use [the Experiment Card](templates/experiment_card.md). Define:

- hypothesis and selected branch;
- closest analogues and remaining difference;
- fixed baseline and dataset slice;
- one intervention;
- primary metric, regression guard, and qualitative check;
- procedure, runtime budget, and stop condition;
- confounders and positive, negative, and ambiguous interpretations.

Prefer a fail-fast slice over a full training run. The test must be capable of rejecting the selected branch.

## Evaluate results

Use [the Result Log](templates/result_log.md). Verify that the run followed the planned baseline, slice, intervention, metrics, and guards. Separate observed measurements from interpretation, and state the strongest alternative explanation still compatible with the data.

Recommend `DROP`, `REFINE`, `REPEAT`, or `SUPPORTED` from the evidence. Bound the conclusion to the tested checkpoint, dataset slice, runtime, and measurement definition; the human owner decides whether to accept the direction.

## Output resources

- Use [the Idea Critique](templates/idea_critique.md) for exploration.
- Use [the Experiment Card](templates/experiment_card.md) for experiment design.
- Use [the Result Log](templates/result_log.md) for evaluation.

These resources structure the research response. Persist them through `pulsf-archive-agent-notes`; publish curated documentation through the repository prose workflow. Load only the resource needed for the selected mode.
