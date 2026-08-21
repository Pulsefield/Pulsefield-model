### Result Log: <result-log-id>

#### Experiment and Reproduction

- Owning Agent Note ID and accepted revision (`none` if exploratory):
- Experiment Card ID and revision (`none` if no Card):
- Baseline source revision and worktree state:
- Intervention source revision and worktree state:
- Baseline run or evidence IDs:
- Intervention run IDs:
- Baseline command or procedure:
- Intervention command or procedure:
- Environment, dependencies, and hardware:
- Baseline and intervention seeds:
- Dataset slice and identity:
- Baseline config and checkpoint identities:
- Intervention identity:
- Output artifact paths or stable IDs:
- Output collision, overwrite, and resume disposition:
- Budget consumed:
- Stop reason:

#### Results

- Baseline primary-metric value, aggregation, sample count, and uncertainty:
- Intervention primary-metric value, aggregation, sample count, and uncertainty:
- Decision threshold result:
- Regression guard value and bound:
- Qualitative observations:
- Failures or missing evidence:

#### Plan Conformance

- Planned-versus-actual deviations:
- Protected fields changed, if any:
- Deviation disposition: none | behavior-neutral | material

#### Evaluation

Leave this section pending for a run-only request. Fill or revise it during Evaluate.

- Observation:
- Interpretation:
- Strongest alternative explanation:
- Confounders:
- Scope of the conclusion:

#### Decision

Leave this section pending for a run-only request. Fill or revise it during Evaluate.

- Recommended outcome: DROP | REFINE | REPEAT | SUPPORTED
- Evidence supporting the outcome:
- Revised question or next discriminating test:
- Human direction still required:

Without an accepted card and identifiable baseline and guard, or after a material protected-field deviation, do not recommend `SUPPORTED`.
