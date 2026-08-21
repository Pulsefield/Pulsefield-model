# Writing Self-Contained Technical Analyses

This guide covers analytical documents whose question and evidence are already
defined: root-cause analyses, performance investigations, incident
postmortems, and similar reports. It offers writing principles rather than a
required template. Authors should adapt the structure to the problem.

## Where this guide applies

Use this guide when the document needs to explain an observed system behavior,
evaluate competing causes, or record a conclusion supported by measurements.
The analysis may leave part of the problem unresolved, but it should have a
specific question and an evidence trail.

Other document types need different structures:

- Open-ended or still-diffuse research questions belong in research triage and
  experiment planning until the question becomes concrete.
- Design and architecture documents explain a proposed or existing system so
  people can understand and change it. They are not causal investigation
  reports.
- Onboarding, reference, and operational guides should optimize for lookup and
  task completion.
- Raw logs and generated outputs are evidence inputs, not durable analysis by
  themselves.

## What self-contained means

A reader should not need the originating chat, prompt, agent instructions,
Experiment Card, or an unexplained local artifact to understand the document.
The report should contain the context needed to evaluate its claims: the
observed symptom, relevant environment, measurement definitions, controls,
evidence, and uncertainty.

Major sections should also work when reached through a direct link. A section
usually needs to re-establish:

- the local question;
- the conditions that changed and stayed fixed;
- the evidence relevant to that question;
- the conclusion and its scope; and
- any unresolved ownership or confounder.

Shared notation can be defined once in a clear measurement section. When a
later section changes the workload, phase boundary, runtime, or accounting
view, it should restate the condition that affects interpretation. Local
context is more useful than repeating the whole document.

## Write the analysis, not the research workflow

Organize the document around the technical argument. Internal planning labels
such as idea quality, implementation families, positive signals, kill
criteria, or result-interpretation steps describe how work was selected. They
usually do not help a reader understand the resulting system behavior.

Planning material can still inform the report. Translate it into ordinary
technical prose:

- explain why a control isolates a suspected cause;
- present the observed measurements next to their comparator;
- state what the result supports or rejects; and
- keep future experiment design separate from established findings.

Experiment identifiers are useful when they provide provenance across logs or
reports. They should not become the document's primary structure unless the
sequence itself is the subject of the analysis.

## Establish the measurement model early

Performance and memory reports are easy to overstate when counters overlap or
sample different moments. Introduce the measurement model before using it for
attribution. Useful context often includes:

- the counter or field name and what it measures;
- units and conversion rules;
- the phase boundary or synchronization point;
- whether values are endpoints, deltas, or run-wide peaks;
- ledgers that overlap and therefore cannot be added or subtracted; and
- version, device, dtype, workload, and configuration conditions that affect
  behavior.

Define derived quantities where they first appear. If a value is reconstructed
from logs, explain the reconstruction and how it was checked. A residual between
two counters remains a residual until another measurement identifies its
owner.

## Match causal language to the evidence

Use the narrowest claim supported by the comparison. A repeated observation
shows reproducibility. A controlled intervention can identify a cause when the
suspected variable changes and the relevant measurement changes with it. A
correlated counter or resource category narrows the search but does not prove
byte, object, or lifetime ownership.

Separate conclusions by confidence:

- Confirmed mechanisms have a reproduced causal chain and clearly stated
  scope.
- Rejected explanations were tested as sufficient causes under named
  conditions. They may remain possible contributors elsewhere.
- Unresolved effects stay explicit rather than being folded into the confirmed
  root cause.

Version-scoped findings should remain version-scoped. Avoid turning one
allocator branch, hardware path, dataset slice, or controlled input boundary
into a universal rule.

## Make evidence auditable

Place measurements close to the claims they support. Tables work well for
matched arms, phase-aligned counters, and before-and-after comparisons. The
surrounding prose should explain the comparison and its meaning rather than
read every table row back to the reader.

Include enough provenance to reproduce or challenge the result, such as the
source revision, checkpoint or data slice, runtime versions, relevant
configuration, and the changed variable. Link to a specific durable source
when it carries details that do not belong in the report. Summarize the
evidence that matters instead of requiring a broad scan of generated artifacts.

## Separate findings from decisions and next work

Technical attribution, production decisions, and follow-up diagnostics answer
different questions. Keeping them distinct prevents an unresolved process
delta from weakening a confirmed lower-level mechanism, and prevents a
promising mitigation from being presented as validated.

A follow-up section can state the remaining question, the smallest comparison
that preserves it, and how the result would change the interpretation. Full
Research branches and Experiment Cards belong in local Agent Notes; a finished
analytical report states the evidence, interpretation, and focused next
diagnostic without importing the planning structure.

## Optional document shape

The following sequence is a useful starting point, not a required heading set:

1. Summary of the confirmed result and remaining unknowns.
2. Incident or analysis scope, symptom, and environment.
3. Measurement definitions and accounting boundaries.
4. Evidence organized by the causal questions it answers.
5. Root-cause assessment separated from unmatched effects.
6. Limits, version scope, and sources.
7. Operational decision or current mitigation status.
8. Focused next diagnostics, when unresolved work remains.

Short analyses can combine sections. Long analyses benefit from narrower
evidence sections that each restate their local comparison and conclusion.

## Review questions

Before treating an analysis as durable repository context, ask:

- Can a reader understand the symptom and scope without the originating task?
- Does each major section explain its local comparison and conclusion?
- Are counters, units, timing boundaries, and overlapping ledgers defined?
- Are causal claims distinguished from correlation and residual accounting?
- Are confirmed, rejected, and unresolved explanations kept separate?
- Does every important table support a nearby claim without being repeated in
  prose?
- Are environment-specific findings labeled with their version and workload
  scope?
- Have research-planning labels and agent-role language been removed from the
  final narrative?

The
[Mapper MPS root-cause analysis](../research/mapper_v2_1_mps_memory_root_cause_report.md)
is a worked example. The
[MPS memory and performance troubleshooting guide](../engineering/mps_memory_performance_troubleshooting.md)
provides the investigation frame for that specific class of problem.
