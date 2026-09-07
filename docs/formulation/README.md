# Pulsefield V3 formulation

Pulsefield's target is to generate musically coherent, legal 4K choreography
from complete audio and committed chart history. Style and gameplay-demand
requests are optional controls. The materialized output is a sequence of complete timed
rows; implementations may use different representations and generation methods.

This directory separates the generation contract from the gameplay semantics
that guide preference among legal continuations:

| Document | Ownership |
| --- | --- |
| [Generation notation](notation.md) | Chart language, absolute time, exact legality, committed history, legal continuations, optional control positions, and prefix commits |
| [Gameplay demand and style](gameplay-state.md) | Target continuation responses, the gameplay frontier, demand representations, section-style observations, and semantic evaluation |

## Research direction

Target responses take semantic priority over the state used to represent them.
After the initial chart dataset is complete, the response specification will
define, from a mapper's perspective, what gameplay demand should and should not
describe. The connection with style is part of developing that specification:
source-backed style judgments and concrete chart contrasts help identify the
responses worth preserving.

The gameplay frontier names the response function over possible legal futures.
A finite demand state and its dynamics are candidate representations of that
function. Their adequacy must be assessed against the independently defined
target responses. This formulation does not provide a completed response
specification, a calibrated demand scale, or an executable V3 model.

The initial style dataset uses the scoped, ordinal judgments supplied by
@Pulsefield/beatmap-lens. Presence, strength, unresolved judgments, and
unreviewed dimensions remain distinct. Those observations provide style
supervision; they do not directly label numerical demand.

## Authority

The chart language and commit rules are formal constraints. The canonical
gameplay profile and annotation meanings are declared conventions. Response
definitions, representation adequacy, and control behavior have the research
status stated in their owning sections.

Concrete candidate generators, state encoders, dynamics, training objectives,
pooling models, decoding methods, and experimental results belong in
[research documentation](../research/). A particular implementation's reachable
charts do not redefine the legal chart space.

The [repository README](../../README.md) defines the V3 status and the boundary
around retained pre-V3 systems. Legacy code and local generated assets are not
sources of V3 specification.
