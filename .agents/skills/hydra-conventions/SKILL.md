---
name: hydra-conventions
description: Apply Pulsefield's Hydra architecture when changing packaged configs, mapper training presets, inference profiles, config adapters, CLI entrypoints, or Hydra tests. Use for work that must preserve canonical package-local YAML, typed validation, complete runtime projection, runner-consumption checks, legacy CLI rejection, and import boundaries.
---

# Pulsefield Hydra Conventions

Use Hydra at the process configuration boundary. Select model bundles, training presets, and runtime parameters at startup, then pass typed dataclasses, paths, or plain dictionaries into the long-lived runtime.

## Sources of truth

- Put YAML under `src/pulsefield_model/configs/` so configs are packaged resources.
- Use `configs/hydra/` for training experiments and `configs/inference/` for online inference.
- Treat packaged training presets as canonical. Do not restore duplicate mapper YAML under repo-root `configs/training/` or add mapper-specific YAML loader APIs.
- Keep inference mapper groups profile-only. Store checkpoint version, model family, vocab and grammar contracts, bundle ID, aliases, protocol contract, and default checkpoint in `MAPPER_PROFILE_SPECS`.
- Keep inference selector groups minimal. Keep training presets complete enough to describe the runner configuration without a second source.
- Do not add repo-root or `temp/` config dependencies for normal mapper execution.

## Entrypoints and boundaries

- Compose Hydra in `training/mapper_training_hydra.py` and `inference/hydra_entry.py`.
- Keep Hydra and OmegaConf imports out of endpoint, session, protocol, model-bundle, and mapper runner modules.
- Preserve the Torch-free help path for `inference.hydra_entry` and `inference.ws_server`; import endpoint and model runtime code only when serving.
- Keep `mapper_v2.py` and `mapper_v2_1.py` fixed to their own presets. Use the generic training entrypoint to select another mapper group.
- Prefer Hydra overrides for new CLI usage; reject deprecated argparse flags with a clear replacement.
- Preserve narrow compatibility shims only when they reduce operator mistakes, such as `--dry-run`.

## Config Shape

- Start top-level YAML with structured schema defaults, then concrete groups, then `_self_`.
- Disable Hydra log/output side effects unless a new command explicitly needs them:
  `hydra.run.dir=.`, `hydra.job.chdir=false`, `hydra.output_subdir=null`.
- Use dataclasses as the accepted schema: `TrainingExperimentConfig` and `InferenceServiceConfig`.
- Optional overrideable fields should exist in schema with `None` defaults, not appear only in presets.
- Reject unknown keys at composition/projection time; do not silently accept `+unexpected` fields.

## Projection

- Convert Hydra config into typed dataclasses or runner dictionaries at the boundary.
- Runtime modules should receive `InferenceServiceConfig`, `WsEndpointConfig`, paths, or plain dicts.
- Project every accepted inference field into `WsEndpointConfig` and the selected backend. Reject fields that cannot affect runtime behavior.
- Validate training fields against the selected mapper's `RUN_CONFIG_KEYS`. Make `_call_kwargs` reject non-null fields the runner cannot consume.
- Preserve the difference between an omitted optional field and an explicit `null` in a canonical preset when projecting the runner dictionary.
- Write both resolved Hydra and flattened runner artifacts before training starts.
- Validate inference semantic contracts before creating `WsEndpointConfig`.
- Treat `--cfg job` as raw Hydra inspection unless a separate validated inspection path is added.

## Model Contracts

- Mapper bundle selection is explicit: no `profile=auto` in committed inference configs.
- Keep profile-derived metadata and the protocol contract consistent through `MapperProfileSpec`. Do not require the configurable runtime `mapper.model_id` to equal the profile's canonical bundle ID.
- Add a mapper by updating profile-name typing, `MAPPER_PROFILE_SPECS`, bundle and translator dispatch, a minimal inference group, any training preset, and the matching tests.
- Keep vocab/grammar translation rules near the model bundle/protocol code, not in endpoint orchestration.

## Tests

- Cover default composition, group selection, unknown-key rejection, semantic validation, and package resource visibility.
- Keep blocked-import tests for the inference help entrypoints and import-boundary tests for runtime modules.
- Add CLI regression tests for deprecated flag rejection and missing-value Hydra flags before changing argv normalization.
- Test that custom mapper and timing model IDs and timing checkpoints reach the runtime.
- Test that mapper-specific unsupported training fields fail instead of disappearing during projection.
