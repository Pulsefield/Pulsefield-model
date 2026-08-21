# Calibrated Pulsefield prose examples

Use these examples to identify the governing principle, not as text templates. A balanced version preserves every load-bearing proposition with the least explanation needed at that location.

## Preserve configuration projection and failure

**Over-trimmed:** “Validate the mapper config.”

**Balanced:** “Reject unknown mapper fields during composition or projection, and reject non-null training fields the selected runner cannot consume.”

Unknown-key timing, mapper selection, null handling, and failure rather than silent omission are distinct facts.

## Keep a local import boundary

**Over-trimmed:** “See the Hydra guide.”

**Balanced:** “Compose Hydra in the training or inference entrypoint; pass dataclasses, paths, or plain dictionaries into runtime modules. See the Hydra conventions for the full projection contract.”

A link carries extended rationale, not the local obligation that a maintainer must follow.

## Preserve a caller-visible failure distinction

**Over-trimmed:** “Loads a mapper bundle.”

**Balanced:** “Load the selected mapper bundle. Raise a configuration error before model initialization when its profile, vocabulary, grammar, and protocol contract disagree.”

State when failure occurs and what contract it protects; let implementation code show the helper sequence.

## Explain a test's observation, not its steps

**Over-detailed:** “The test starts the help command, blocks Torch imports, captures stdout, and then checks the exit code.”

**Balanced:** “Exercise the real help entrypoint with Torch imports blocked so the assertion pins the import-light operator path rather than an internal parser helper.”

The fixture actions are visible in the test. The reason for using the real entrypoint is not.

## Keep measurement ledgers separate

**Over-trimmed:** “Process memory exceeds MPS memory.”

**Balanced:** “Process footprint and MPS driver memory are overlapping ledgers; compare aligned phases, but do not subtract one from the other to assign the residual.”

Counter relationship and prohibited inference are part of the measurement contract.

## Match causal prose to the experiment

**Overstated:** “Variable sequence length causes every MPS memory increase.”

**Balanced:** “Under the recorded mapper workload and runtime, varying padded sequence geometry reproduced driver-memory growth that fixed-shape runs did not; the result does not assign the remaining process-footprint delta.”

Keep the intervention, comparator, environment scope, supported conclusion, and unresolved effect.

## Keep planning structure out of a finished analysis

**Workflow-shaped:** “The selected variant passed the guard, so the next-loop action is to investigate allocator heaps.”

**Balanced:** “The fixed-shape control preserved throughput while preventing the observed driver-memory increase. Heap attribution remains unresolved and is the next diagnostic.”

Translate experiment mechanics into evidence and a focused remaining question.

## Configuration comments explain consequences

**Over-detailed:** “The defaults list the schema, then the mapper group, then `_self_`.”

**Balanced:** “Apply `_self_` last so values in this preset override structured-schema defaults.”

Let YAML show the entries; explain only the non-obvious composition consequence.

## Diagnostics identify the subject and correction

**Vague:** “Invalid configuration.”

**Balanced:** “`mapper.profile` must name an explicit packaged profile; choose `v2_tuple` or `v2_1_sparse` instead of `auto`.”

Name the field, violated rule, and available correction. Omit an option list when it is dynamic or already printed by the owning parser.

## Fresh-clone limitations are contracts

**Over-trimmed:** “Inference needs checkpoints.”

**Balanced:** “A fresh clone can compose configs and run tests, but model-backed inference requires the checkpoints selected by the mapper and timing profiles; the repository does not publish a checkpoint download workflow.”

Keep the supported operation, missing prerequisite, and operational consequence.

## Delete reasoning transcripts

**Over-detailed:** “First the endpoint checks whether the session exists. If it does not, it returns. Otherwise it reads the next message, which is why the send below is safe.”

**Balanced:** Delete the comment when the code expresses those branches. If a non-obvious invariant exists, state only that invariant, such as: “Session removal prevents any later stream write for this request.”

Do not compress a reasoning transcript into shorter control-flow narration.

## Preserve source ownership

**Over-trimmed:** “The profile contains mapper metadata.”

**Balanced:** “`MAPPER_PROFILE_SPECS` owns bundle identity, aliases, model-family metadata, vocabulary and grammar compatibility, protocol version, and checkpoint defaults; inference YAML selects a profile rather than duplicating those fields.”

The list is justified because it defines one source-of-truth boundary. Do not shorten it to an abstraction that permits ownership drift.

## Keep limitations, not cleanup inventories

**Over-trimmed:** Omit that an evaluation is valid only for one checkpoint and dataset slice.

**Over-detailed:** List every unused helper and provisional filename discovered during the run.

**Balanced:** State the checkpoint, slice, runtime, and metric limitations that constrain interpretation. Put ordinary cleanup in a targeted issue or TODO.
