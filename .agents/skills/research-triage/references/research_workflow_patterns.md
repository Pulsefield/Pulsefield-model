# Research Workflow Patterns

Use this reference to calibrate analogue search, hypothesis branching, and evidence quality.

## Analogue comparison

Compare candidates by mechanism rather than name:

- problem and prediction target;
- representation and model objective;
- data, supervision, and sampling;
- inference or decoding procedure;
- evaluation protocol and baseline;
- runtime and product constraints.

The closest analogue is the one that shares the causal mechanism relevant to the research question. Record the remaining difference before assessing novelty.

Useful discovery sources include:

- `worldbench/awesome-ai-auto-research` for research-workflow taxonomy;
- `Future-House/paper-qa` and `assafelovic/gpt-researcher` for source-grounded evidence gathering;
- `microsoft/RD-Agent` for iterative research and development;
- `Embodied-Minds-Lab/BES` for branching candidates against decomposed goals;
- `InternScience/ResearchClawBench` for checklist-based evaluation;
- `SakanaAI/AI-Scientist` as an example of broad automated research loops.

Treat each project as an analogy, not authority for Pulsefield architecture.

## Branch quality

A useful branch has a distinct mechanism and predicts an observation that separates it from competing branches. Reject a branch when it only renames another branch, cannot be falsified with available evidence, or requires a broad implementation before any local signal can be measured.

Use goal decomposition when the target metric is too sparse. Local subgoal checks can rank branches, but the primary task metric remains the final decision signal.

## Evidence grades

- **A:** reproducible local run, checked metric, or current repository trace.
- **B:** primary paper or repository with explicit data, commands, and evaluation.
- **C:** benchmark, leaderboard, or execution trace with inspectable criteria.
- **D:** survey, curated list, or secondary technical summary.
- **E:** README-only, marketing, popularity, or social claim.

Use the weakest grade needed to support a claim. Do not promote novelty or effectiveness beyond the available grade.

## Experiment pressure

A bounded experiment has a fixed baseline, one intervention, a named slice, a primary metric, a regression guard, a qualitative check, and a runtime stop. Positive, negative, and ambiguous outcomes must lead to different research decisions.

Prefer the smallest comparison that can reject a branch. Full training is justified only when cheaper probes cannot preserve the question.
