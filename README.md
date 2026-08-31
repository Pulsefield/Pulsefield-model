# Pulsefield Model

Pulsefield Model is the research and development repository for Pulsefield V3,
a 4-key rhythm-game choreography generation system.

## Pulsefield V3 status

Pulsefield V3 is under active research and development. Work in this repository
defines the target generation problem, causal gameplay state, constraints,
falsifiable hypotheses, and evaluation questions. It does not yet define or
ship an executable V3 reference architecture, training pipeline, or inference
pipeline.

Start with the [V3 formulation](docs/formulation/README.md).

## Legacy code boundary

> **Do not use mapper v2/v2.1, the pre-V3 timing stack, Control V3, or the
> training, inference, configuration, protocol, and test code built around them
> as design, correctness, or implementation references for Pulsefield V3.**

These are retained pre-V3 research systems. Where the required local assets are
available, they may still run and own their legacy checkpoint and protocol
compatibility. That limited ownership does not make their tokenization, timing
representation, control targets, model interfaces, or runtime structure part of
the V3 contract.

## Documentation authority

- [`docs/formulation/`](docs/formulation/README.md) owns the V3 problem
  definition, notation, invariants, and open questions.
- [`docs/research/`](docs/research/) contains experimental evidence and model
  proposals. It is not a V3 specification.

## Repository boundary

This repository covers Pulsefield's model research. It does not contain the
Pulsefield client or a hosted inference service. Datasets, checkpoints, caches,
and generated evaluations are local research assets rather than repository
sources of truth. `ref-proj/` is comparison material, never authority.

## License

Pulsefield Model is licensed under the GNU Affero General Public License v3.0
only (`AGPL-3.0-only`). See [`LICENSE`](LICENSE).

Projects under `ref-proj/` retain their upstream licenses.
