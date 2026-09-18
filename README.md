# Pulsefield Model

Pulsefield Model is the research and development repository for Pulsefield V3,
a 4-key rhythm-game choreography generation system.

## Pulsefield V3 status

Pulsefield V3 is under active research and development. Work in this repository
defines the target generation problem, causal gameplay state, constraints,
falsifiable hypotheses, and evaluation questions. The V3 reference architecture
and end-to-end training and inference pipelines remain open.

Start with the [V3 formulation](docs/formulation/README.md).

The [oracle-time continuation research baseline](docs/research/oracle_time_continuation.md)
implements verified source replay, a causal backbone, sequence training, and durable sampled generation on
supplied event times (M0–M3). M4 corpus training and generated-structure evaluation
are in progress, including a Mac profile with a larger temporal module; this
research baseline does not define the V3 reference architecture.

The [formulation research question](docs/research/oracle_time_expert_question.md)
collects the current task definitions, contrasting experimental results and
inspectable endpoint prototypes for an independent assessment of the next
learning setup.

The [bounded three-arm continuation baseline](docs/research/bounded_typed_continuation.md)
defines matched typed-row/object tasks alongside the original row task. Exact
execution support and a finite content encoder are implemented; the native
three-arm training and generated-quality comparison remain in development.

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
