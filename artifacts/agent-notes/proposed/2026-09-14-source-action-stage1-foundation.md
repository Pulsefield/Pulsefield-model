# Agent Note: Establish the source-action block-prediction foundation

Note ID: 2026-09-14-source-action-stage1-foundation
Status: proposed
Kind: implementation
Created: 2026-09-14
Updated: 2026-09-14
Product revision: 83784046300373565d8e3a8409956367172af4ba
Scope: Partial source observations, joint action-block prediction, reference encoder integration, and bounded software verification.

## Question or Decision

Build the common prediction task needed for a later comparison of contextual-only and directly readable local representations. Stage 1 succeeds when the task has explicit information access, target-isolated inputs, a trainable reference predictor, and reproducible diagnostics. It does not establish an architecture gain.

This handoff proposes implementation and bounded verification. It does not accept a research direction, authorize execution, or prescribe a full pretraining run.

## Repository State and Evidence

The product worktree was clean at the product revision above. Its design owner is `docs/research/source_action_representation_directions.md`; the empirical baseline is `docs/research/scoped_style_probe_postmortem.md`.

Read `README.md`, `AGENTS.md`, and the design owner before editing. Relevant implementations are under `src/pulsefield_model/research/scoped_style_modeling/`: `replay.py`, `relations.py`, `tensors.py`, `model.py`, and `probes.py`. Nearby replay/model tests pin source identity, mirror behavior, padding, and target isolation.

Complete-chart features contain next-attack timing, LN endpoints, and derived relations. They are not a partial-observation API. The current encoder is already concept-independent. Its checkpoint writer does not retain optimizer state. No source-action reconstruction baseline or measured likelihood exists yet; style NLL is not a substitute.

The pinned annotation dataset revision is `b22a7a443783e05fee4db4b1d22b8e573ad448ae`, with split SHA-256 `15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`. The split is source-grouped and does not establish comprehensive song/audio deduplication. Local asset availability has not been checked for this handoff.

## Alternatives or Hypothesis Branches

Masking completed tensors leaves answer paths through derived fields and topology; reject it as the implementation approach. Adding new local operators immediately would combine task construction with the architectural intervention. Defer that comparison until the shared task works.

Use exact source-action prediction as the initial target. Learned latent-target prediction would introduce another target model and different interpretation requirements.

## Selected Direction

### Observation and target contract

Introduce a separate partial-observation type; keep the complete source objects immutable for target extraction and identity verification. Never interpret unknown actions as silence.

Retain the union of real event times as a supplied skeleton. Hide complete four-lane action rows in a nonempty contiguous block, including release-only rows. Synthetic boundary markers are not targets. Begin with event-indexed blocks of 4, 16, or 64 rows, chosen without inspecting their action values. Record realized attack-group spans for analysis only. This is arrangement prediction at supplied times, not timing generation.

Build all network features and relation edges from visible facts and declared conditions. Recompute or remove every disallowed descendant of hidden actions, including descriptors attached to visible rows outside the block. A next observed attack across unknown rows is not a known immediate successor.

For LNs, preserve source endpoints in the complete source record but hide undisclosed duration, close-time, identity, and occupation information from model inputs. Explicit entering occupancy may be supplied without the future close time; post-block occupancy is not automatically observable. Preserve same-lane close-plus-attack events without retiming them.

### Reference predictor

Use the existing shared-hand BiGRU and relation-attention computation as the reference family, initialized from scratch with mask-aware inputs. Do not coerce unknown values into the complete-chart tensor schema or reuse old checkpoints as though the input contract were unchanged.

Add one compact autoregressive decoder for a joint four-lane row distribution. Preserve within-row dependencies and close/head combinations. Feed preceding target rows only to the decoder during teacher forcing; the encoder and its graph never receive them. Legal-action masks depend only on declared conditions and the decoded prefix.

Use mean negative log likelihood per target row within each block, then mean over sampled blocks. Fix and record the sampling distribution. Expose encoder outputs through a small interface that a later local-path candidate can implement with the same decoder.

A proposed new owner is `src/pulsefield_model/research/source_action_modeling/`, with corresponding tests. Reuse verified parsing and coordinate helpers where appropriate. Preserve scoped-classifier behavior and the legacy-code boundary. Follow `hydra-conventions` if adding a configured CLI.

### Minimal update observation

Capture actual before/after parameter displacement for a few test updates. For fixed block log likelihood, compare the actual output change with the gradient-displacement dot product, grouped by disjoint parameter ownership. Use identical inputs and deterministic diagnostic evaluation; report the linearization residual.

Do not build a general Jacobian dashboard or interpret these values as per-sample or per-slot causal attribution.

## Verification or Evaluation

Required checks:

1. Change hidden targets while holding supplied conditions fixed: all encoder tensors and graph topology remain identical, including features outside the block.
2. Cover unknown versus absent, zero versus unavailable time, entering/exiting LNs, hidden releases, close/head coincidences, and synthetic boundaries.
3. Verify mirrored inputs and masks produce the corresponding joint action distribution; padding does not alter valid outputs or loss.
4. Check normalized finite row probabilities, correct loss denominators, and gradients reaching both the encoder and decoder.
5. Verify teacher-forcing isolation and save/load reproduction. A resumable snapshot includes optimizer, RNG, sampling position, and any scheduler state.

Use deterministic constructed fixtures first. A bounded real-input check may inspect at most eight existing training contexts from distinct groups, selected deterministically without labels, each at most 128 source-event rows. Record their source/scope identities. Use only verified local assets from the pinned split; no downloads, validation/test examples, or new corpus.

Limit the real-input check to 20 optimizer updates and 120 seconds total including diagnostics, whichever comes first; seed 17, logical batch size at most eight blocks. Report finite loss, parameter movement, and first/later target-row losses. This is wiring evidence, not a fit or generalization result. If this bound is inadequate, report the limitation rather than silently extending it.

On Apple Silicon use the explicit `mps` extra with `--group dev`; CPU fixture tests are also valid. Put any generated output in a fresh `artifacts/source-action-modeling/stage1-smoke/<run-id>/` directory. Do not overwrite or resume an existing run. Leave raw samples and generated payloads out of Agent Notes.

Run the new owning tests and affected existing replay/model tests, then `git diff --check`. Record exact commands and the verified product revision. No new research-training CLI is claimed to exist by this note.

## Risks and Falsifying Evidence

Stop the relevant check on target-dependent encoder inputs, invalid source conversion, split/hash mismatch, nonfinite loss, or unhandled boundary state. Do not weaken tests or drop hard LN cases to obtain a passing result.

Teacher forcing can let the decoder exploit its target prefix without learning useful encoder structure. Passing this stage does not resolve that risk. The later architecture comparison must use a common objective, observation distribution, decoder, exposure, and capacity control.

Persistent residual-slot routing, orthogonality, style fine-tuning, loss-routing experiments, optimizer searches, timing generation, and audio conditioning are outside this stage.

## Next Lifecycle Condition

Acceptance concerns this exact proposed note revision and scope. Completion requires a clean implementation revision, passing contract checks, and a concise verification report with any missing-asset limitations explicit. Missing real-input validation leaves that deliverable incomplete.

Return the input contract, reference model and parameter count, loss definition, test evidence, bounded-check observations, and remaining design questions. Then prepare the separate bounded architecture comparison. No semantic improvement or demand-state claim follows from Stage 1.
