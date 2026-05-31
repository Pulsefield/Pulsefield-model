# Research Workflow Patterns

Use this reference only when source-derived workflow heuristics are needed. Recheck sources before relying on current repo status, dates, stars, or claims.

## Taxonomy

- Use `worldbench/awesome-ai-auto-research` as a map of lifecycle stages: idea generation, literature review, coding and experiments, tables and figures, writing, peer review, rebuttal, dissemination, end-to-end systems, and critical perspectives.
- Treat curated lists as indexing aids. They do not establish novelty, quality, or reproducibility.

Source: https://github.com/worldbench/awesome-ai-auto-research

## Patterns To Borrow

### Literature grounding before novelty

- `gpt-researcher` separates planning, information gathering, and publishing, with source tracking across web and local material.
- `paper-qa` emphasizes scientific-document RAG, evidence gathering, citation metadata, contextual summaries, and answer generation.

Apply this by requiring closest analogies, source snapshot, evidence grade, and citation-backed related work before novelty claims.

Sources:
- https://github.com/assafelovic/gpt-researcher
- https://github.com/Future-House/paper-qa

### Research and development split

- `RD-Agent` frames data-driven R&D as idea proposal plus implementation, execution, feedback, and iteration in concrete scenarios.

Apply this by preserving critic, planner, and executor boundaries. The planner writes one card; the executor implements only that card.

Source: https://github.com/microsoft/RD-Agent

### Autoresearch loop discipline

- Recent autoresearch skills emphasize goal, scope, metric, baseline, one focused change, verification, keep/discard logging, and bounded stuck handling.
- Useful fields are `Verify` for the improvement metric and `Guard` for regressions that must not be introduced.

Apply this by requiring baseline/comparator, verify command, guard check, result log, kill criteria, and expected runtime before execution.

Sources:
- https://github.com/leo-lilinxiao/codex-autoresearch
- https://github.com/uditgoenka/autoresearch
- https://github.com/krzysztofdudek/ResearcherSkill

### BES-inspired closed-loop pressure

- `Embodied-Minds-Lab/BES` couples forward candidate evolution with backward goal decomposition. The useful workflow analogy is not the specific algorithm, but the closed loop: decompose the target into checkable subgoals, generate candidate variants, locally verify candidates against dense subgoal signals, then apply selection pressure before spending more runtime.
- Its inference code represents decomposed goals as a goal tree with local `verify_code` scores in `[0,1]`, recursively combines those scores, expands leaves that have not been fully satisfied, and ranks programs with a raw metric as the dominant signal plus backward subgoal score as an intra-bucket pressure.
- Apply this by forcing every non-`KILL` research route to state the root objective, subgoals, 2-4 candidate variants, local verification matrix, selected variant, rejected variants, and next-loop action.
- Do not claim Pulsefield implements BES unless code actually introduces candidate evolution plus backward decomposition. Treat the pattern as a planning discipline and a nearest analogy for closed-loop research workflow.

Sources:
- https://github.com/Embodied-Minds-Lab/BES
- https://arxiv.org/abs/2605.28814

### Benchmark-style evaluation

- `ResearchClawBench` uses curated research tasks, reference papers, checklists, and per-item scoring to judge whether an agent rediscovered or exceeded known results.

Apply this by converting qualitative checks into explicit checklist items whenever a numeric metric is unavailable.

Source: https://github.com/InternScience/ResearchClawBench

### Fully automated systems as warning signs

- `AI-Scientist` demonstrates template-based idea generation, experiment execution, write-up, and review, but also highlights risk from executing model-written code and the need for containment.

Apply this by keeping the human owner in charge of novelty, interpretation, and direction. Do not adopt fully autonomous loops unless the user explicitly asks and the experiment has hard bounds.

Source: https://github.com/SakanaAI/AI-Scientist

## Evidence Grades

- `A`: local run, checked metric, reproducible command, or repository trace from the current workspace.
- `B`: primary repo or paper with runnable commands, explicit datasets, and clear evaluation.
- `C`: benchmark, leaderboard, checklist, or execution trace with enough detail to inspect.
- `D`: curated list, survey, project page, or secondary summary.
- `E`: README-only claim, star count, social post, or marketing copy.

Use the weakest relevant grade when evidence is mixed.

## Red Flags

- No baseline or comparator.
- No metric, verify command, or qualitative checklist.
- No guard against breaking existing behavior.
- Closest analogy is unnamed.
- Claimed novelty is only a renamed representation or engineering variation.
- First proposed experiment requires full training, broad rewrites, or long runs.
- Evidence comes only from a curated list or README claim.
- Executor would need to change the research question to make progress.

## Default Routing

- Choose `KILL` when the idea is already covered by close analogies and no new layer or useful experiment remains.
- Choose `MUTATE` when the idea may be useful but lacks source grounding, a baseline, a measurable metric, or a fail-fast slice.
- Choose `TEST` only when one bounded code change can answer the question with a defined baseline, verify gate, guard gate, runtime budget, and kill criteria.
