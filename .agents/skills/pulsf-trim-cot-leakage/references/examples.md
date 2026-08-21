# Calibrated leakage examples

Use these examples to identify the governing principle, not as replacement templates. Each fixed version preserves the factual clauses that a reader at `HEAD` needs and removes only session viewpoint.

## Dead citations

### Session ordinal with a committed owner

**Leaked:** “Mapper YAML contains only the profile selector (decision 12).”

**Fixed:** “Inference mapper YAML selects a profile; `MAPPER_PROFILE_SPECS` owns bundle identity, aliases, model-family metadata, vocabulary and grammar contracts, protocol compatibility, and checkpoint defaults. See the Hydra conventions.”

The ordinal resolves nowhere. Name the committed owner and preserve the complete ownership rule.

### Audit code without an owner

**Leaked:** “Unknown mapper fields fail during projection (audit C3).”

**Fixed:** “Unknown mapper fields fail during composition or projection.”

The audit code carries no fact. The timing and failure behavior do.

### Uncommitted section number

**Leaked:** “The runtime receives plain dictionaries per design §4.2.”

**Fixed:** “Compose Hydra at the process entrypoint, then pass typed dataclasses, paths, or plain dictionaries into the runtime.”

An uncommitted section is invisible to future readers. An RFC or committed document section can remain when it actually owns the rule.

### Chat reference

**Leaked:** “As requested above, the service rejects `profile=auto`.”

**Fixed:** “The service requires an explicit mapper profile and rejects `profile=auto`.”

User intent belongs to the task history; the shipped behavior belongs in durable prose.

## Publication viewpoint

### PR narration in a README

**Leaked:** “This PR adds support for mapper v2.1 sparse bundles.”

**Fixed:** “The inference service supports the `v2_1_sparse` mapper profile.”

A README outlives the PR. State current behavior and its stable identifier.

### Stack position as an extension claim

**Leaked:** “A later PR in this stack wires the control checkpoint.”

**Fixed:** “The mapper profile selects both mapper and control checkpoint defaults.”

Keep only shipped behavior. If the work is genuinely pending, name an issue or TODO rather than a stack position.

## Change narration

### Fixed regression as a counterfactual

**Leaked:** “Padding used to let invalid token positions affect the loss.”

**Fixed:** “Without the padding mask, invalid token positions affect the loss.”

The counterfactual explains why the guard matters without requiring repository archaeology.

### Removal biography

**Leaked:** “The old root-level mapper YAML is gone now; packaged Hydra presets replaced it.”

**Fixed:** “Packaged mapper presets under `src/pulsefield_model/configs/` are canonical.”

Readers need the current owner. Preserve removal history only in a decision record or migration note that owns it.

### Version name that must stay

**Keep:** “Mapper v2.1 uses sparse event windows and its own vocabulary and grammar.”

`v2.1` identifies a model family. It is not “version one of this draft.”

## Review choreography

### Rejected in review

**Leaked:** “Rejected in review: copying profile metadata into YAML.”

**Fixed in an Alternatives section:** “**Duplicate profile metadata in YAML.** Rejected because two owners can disagree about vocabulary, grammar, protocol, and checkpoint compatibility.”

Keep the alternative and rationale; remove who rejected it and when.

### Reviewer-addressed safety claim

**Leaked:** “This cast is safe because the loader always returns the right config.”

**Fixed:** “Structured Hydra composition validates the config type before this projection; the runtime receives only `InferenceServiceConfig`.”

State the invariant. Delete the comment entirely if the type and validation are already obvious at the use site.

## Restatement and derivation

### Control-flow walkthrough

**Leaked:** “First validate the profile, then resolve defaults, then build the endpoint config.”

**Fixed:** Delete when the function body already shows those calls. If ordering is load-bearing, state its consequence: “Validate profile compatibility before loading checkpoints so an invalid bundle fails without allocating model state.”

### Test walkthrough

**Leaked:** “This test invokes `--help`, captures stdout, and asserts the process exits zero.”

**Fixed:** “Exercise the real help entrypoint with Torch imports blocked so the test pins the import-light operator path.”

The test body owns the steps; the comment owns the non-obvious observation strategy.

## Hedges and planning residue

### Vague capacity hedge

**Leaked:** “The sequence limit should be enough for now.”

**Fixed:** “The limit covers the largest evaluated window of 4,096 tokens; longer windows fail validation before batching.”

Use this fixed form only when the measurement and failure behavior are true. Otherwise retain the uncertainty and create an owned measurement task rather than inventing confidence.

### Research workflow in a finished analysis

**Leaked:** “The selected variant passed the guard, so the next-loop action is allocator tracing.”

**Fixed:** “The fixed-shape control preserved throughput and prevented the observed driver-memory increase. The remaining process-footprint delta needs allocator tracing.”

Translate selection mechanics into comparison, result, and unresolved question.

## Ephemeral evidence

### Local artifact as the whole citation

**Leaked:** “See `artifacts/run-27/output.log`; it proves the leak is fixed.”

**Fixed:** “Across 100 identical warm iterations on the recorded MPS environment, active and driver memory stayed within the stated measurement tolerance.” Add durable environment, metric, and tolerance details; cite a retained report or result record if one exists.

An unretained path cannot carry the claim. Do not invent measurements when the artifact is unavailable—mark the claim unverified instead.

### Notebook position

**Leaked:** “Cell 43 above shows timing accuracy improved.”

**Fixed:** “On the named evaluation slice, median absolute beat error changed from X to Y under the same timing checkpoint and tolerance.”

Replace `X` and `Y` only from verified evidence. Cell positions and “above” are not durable provenance.

## Keeps

### Measured bound

**Keep:** “At padded sequence length 4,096 on the recorded MPS runtime, one square attention tensor requires `batch × heads × length² × bytes_per_element` bytes.”

The environment and formula prevent the number from becoming an unexplained constant.

### Runtime old and new objects

**Keep:** “The old session drains before the replacement session begins streaming.”

This sentence describes one live handover, not repository history.

### Suppression reason

**Keep after verifying:** `# type: ignore[arg-type]  # The adapter validates and normalizes this payload before construction.`

Suppression reasons are required prose when the checker cannot see a runtime invariant. Fix or remove the suppression if the reason is false.

### Resolvable issue ownership

**Keep:** “`TODO(stream-reset):` issue #123 owns cancellation during mapper reload.”

The tag and issue can own deferred work. A bare “handle later” cannot.

## Overcorrection traps

### Turning an obligation into endorsement

**Original:** “The root-level control config remains for the legacy trainer and should migrate only after that entrypoint is retired.”

**Wrong trim:** “The root-level control config is an intentional exception.”

**Right:** Preserve the legacy consumer and retirement condition. The wrong version blesses a temporary obligation.

### Promoting a hypothesis to shipped architecture

**Original:** “A beat-relative representation may make timing changes easier to compare.”

**Wrong trim:** “Beat-relative representation makes timing changes easier to compare.”

**Right:** Keep the hypothesis modal until an experiment establishes the claim.

### Deleting a true fact with narration

**Original:** “The old diagnostic text listed the projection steps; the text is also asserted by the CLI regression test.”

**Wrong trim:** Delete the whole sentence as change narration.

**Right:** “The CLI regression test asserts the diagnostic text.”

Only the first clause is biography; the test coupling remains current.

### Dropping provenance

**Original:** “Measured on the named MPS runtime, driver memory peaks at 6.2 GiB for this workload.”

**Wrong trim:** “Driver memory peaks at 6.2 GiB.”

**Right:** Keep the measurement provenance, workload, and any uncertainty needed to interpret the number.

### Losing a negative guarantee

**Original:** “The dry-run validates and writes config artifacts but never starts training.”

**Wrong trim:** “The dry-run validates the training config.”

**Right:** Preserve the writes and the negative execution guarantee; both are operator-visible behavior.
