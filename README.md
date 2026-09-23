# Ensomi Model

Ensomi Model is the research and development repository for Ensomi V3,
a 4-key rhythm-game choreography generation system.

## Ensomi V3 status

Ensomi V3 is under active research and development. Work in this repository
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
execution, a finite encoder, complete probability heads and bounded training
windows are implemented. Packaged commands support corpus training, preparation
of standalone timing/seed conditions, and native generation with durable recovery
and verified osu! export. Long-form quality, consistency across difficulty levels,
and varied LN/tap organization remain under evaluation.

The [vacation training queue](docs/research/vacation_training.md) provides serial
audio caching, a 35M R1 teacher profile and fixed native stress runs, with frozen
inputs, resumable segments and a macOS `caffeinate` launcher.

The [staged R1 reconstruction](docs/research/r1_staged_restoration.md) rebuilds the
small response candidate from plain R1, adding seed, memory and the three native
correction modules in order. New trajectories and final quality require evaluation;
it is separate from the 35M teacher queue.

The [audio-conditioned choreography study](docs/research/audio_conditioned_choreography.md)
implements a small shared Mel encoder with joint event timing and R1-derived
complete-row generation. Its research entrypoint supports new audio without a
source chart or seed. Lens inspection has identified a prototype for playtesting
and remaining failures; reliable playability remains under investigation.

## Legacy code boundary

> **Do not use mapper v2/v2.1, the pre-V3 timing stack, Control V3, or the
> training, inference, configuration, protocol, and test code built around them
> as design, correctness, or implementation references for Ensomi V3.**

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

This repository covers Ensomi's model research. It does not contain the
Ensomi client or a hosted inference service. Datasets, checkpoints, caches,
and generated evaluations are local research assets rather than repository
sources of truth. `ref-proj/` is comparison material, never authority.

## License

Ensomi Model is licensed under the GNU Affero General Public License v3.0
only (`AGPL-3.0-only`). See [`LICENSE`](LICENSE).

Projects under `ref-proj/` retain their upstream licenses.
