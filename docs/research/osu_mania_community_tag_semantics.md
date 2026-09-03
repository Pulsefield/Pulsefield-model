# osu!mania Community Tag Semantics and Evidence Domains

## Scope and authority

This page records the observed structure of osu!mania community tags relevant
to Pulsefield's style formulation. It covers `style/*`, `skillset/*`,
`streams/*`, and `tech/*`, together with `expression/*` and `gimmick/*` where
they expose semantic or evidence boundaries not visible in the first four
namespaces. It is a research record, not a style-enum or annotation-schema
specification.

The naming authority is the official
[osu! beatmap tag catalogue](https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags).
The local frequency evidence uses the same osu!mania 4K snapshot and selection
rules as the
[4K style tag reference](osu_mania_4k_style_tag_reference.md). The comparison
target is the section-style model in
[`gameplay-state.md`](../formulation/gameplay-state.md#10-section-style-and-style-tags).

The six namespaces are a deliberately bounded slice of the catalogue. The
snapshot contains 59 tags applicable to osu!mania across 10 namespaces; the
44 tags below exclude `additions/*`, `context/*`, `meta/*`, and `sliders/*`.

## Dataset snapshot

- Catalogue endpoint: `GET https://osu.ppy.sh/api/v2/tags`
- Catalogue fetched at: `2026-08-07T07:02:13.461073+00:00`
- Local catalogue: `dataset/metadata/osu_tags_2026-08-07.json`
- Beatmap metadata: each `dataset/0/<beatmapset_id>/metadata.json`
- 4K selection: `mode_int == 3` and `cs == 4`
- Tag applicability: catalogue `ruleset_id` is `3` or `null`
- Count unit: one 4K difficulty whose `top_tag_ids` contains an entry for the
  tag ID
- Local population: 14,787 osu!mania 4K difficulties

Counts are not vote totals. Tags are non-exclusive, and a difficulty may
contribute to several rows and namespaces. The tag names and semantic
descriptions recorded here are those in the timestamped catalogue snapshot.

| Namespace | Applicable tags | Difficulties with at least one tag |
| --- | ---: | ---: |
| `style/*` | 17 | 2,688 |
| `skillset/*` | 6 | 1,870 |
| `streams/*` | 5 | 1,438 |
| `tech/*` | 2 | 258 |
| `expression/*` | 7 | 1,410 |
| `gimmick/*` | 7 | 393 |

The union of `style/*`, `skillset/*`, `streams/*`, and `tech/*` covers 3,814
difficulties, or 25.79% of the local 4K population. Adding `expression/*` and
`gimmick/*` produces a six-namespace union of 4,290 difficulties, or 29.01%.

## Observed tag inventory

All 44 applicable tags in the six namespaces occur in the snapshot.

| Namespace | Tag | 4K count |
| --- | --- | ---: |
| `style` | `chordjack` | 833 |
| `style` | `jumpstream` | 736 |
| `style` | `LN coordination` | 497 |
| `style` | `generic hybrid` | 320 |
| `style` | `chordstream` | 286 |
| `style` | `LN mixed` | 248 |
| `style` | `handstream` | 245 |
| `style` | `dump` | 229 |
| `style` | `LN release` | 203 |
| `style` | `LN density` | 191 |
| `style` | `longjack` | 190 |
| `style` | `mixed rice` | 146 |
| `style` | `avant-garde` | 30 |
| `style` | `quadstream` | 27 |
| `style` | `tiebreaker` | 16 |
| `style` | `o2jam` | 1 |
| `style` | `N+1` | 1 |
| `skillset` | `streams` | 733 |
| `skillset` | `speedjack` | 608 |
| `skillset` | `tech` | 458 |
| `skillset` | `reading` | 371 |
| `skillset` | `wristjack` | 248 |
| `skillset` | `gimmick` | 17 |
| `streams` | `stamina` | 829 |
| `streams` | `speed` | 565 |
| `streams` | `bursts` | 340 |
| `streams` | `doubles` | 69 |
| `streams` | `quads` | 12 |
| `tech` | `technical hybrid` | 197 |
| `tech` | `complex snap` | 76 |
| `expression` | `difficulty spike` | 597 |
| `expression` | `simple` | 474 |
| `expression` | `repetition` | 465 |
| `expression` | `progression` | 163 |
| `expression` | `high contrast` | 67 |
| `expression` | `conceptual` | 26 |
| `expression` | `inspo` | 11 |
| `gimmick` | `LN inverse` | 146 |
| `gimmick` | `2B` | 127 |
| `gimmick` | `memory` | 75 |
| `gimmick` | `delay` | 54 |
| `gimmick` | `video` | 49 |
| `gimmick` | `storyboard` | 41 |
| `gimmick` | `tag` | 6 |

The raw catalogue name for tag ID 117 is `style/LN mixed ` with a trailing
space. The table trims it for display; the ID remains the stable join key.

## Pattern vocabulary and tag vocabulary

The inspected official osu!mania pattern pages describe recurring chart
structures. Their stream, jack, and hold-note pages define object geometry such
as chord sizes, repeated columns, and long-note relationships. These terms can
be recognized within a sufficiently informative chart interval.

The beatmap tag catalogue describes community-voted labels attached to a
difficulty for discovery and characterization. Some tag names reuse pattern
terms, but the catalogue also contains predicates about skill, duration,
cross-section organization, audio interpretation, mapping philosophy,
presentation, and multiplayer context. A pattern definition can identify a
local occurrence without localizing or fully explaining the corresponding
difficulty-level community vote.

## Semantic structure of the catalogue

The namespace is not a semantic type. Tags within one namespace can differ in
their subject, time scale, evidence source, and relation to lower-level
patterns. The same semantic family can also cross namespace boundaries.

| Semantic structure | Representative tags | Predicate subject and evidence |
| --- | --- | --- |
| Local pattern or long-note topology | `style/jumpstream`, `style/handstream`, `style/quadstream`, `style/chordjack`, `style/longjack`, `style/chordstream`, `style/LN coordination`, `style/LN release`, `style/LN density`, `gimmick/LN inverse`, `tech/complex snap` | A chart interval containing the required row sequence, lane recurrence, chord sizes, hold relationships, releases, or snap divisors |
| Sequence extent, rate, or persistence | `skillset/streams`, all `streams/*`, `skillset/speedjack`, `skillset/tech` | Counts and timing across a sequence or sustained interval, rather than a single row |
| Player-facing skill, technique, or cognition | `skillset/reading`, `skillset/wristjack`, `streams/speed`, `streams/stamina`, `gimmick/memory` | A community judgement about what the chart tests or how it is played |
| Composition of lower-level concepts | `style/mixed rice`, `style/LN mixed`, `style/generic hybrid`, `tech/technical hybrid` | Joint presence of multiple constituent pattern families within the judged scope |
| Whole-map or cross-section organization | `style/tiebreaker`, `expression/progression`, `expression/high contrast`, `expression/difficulty spike`, `expression/repetition` | Duration, coverage, order, recurrence, or contrast across surrounding or separated intervals |
| Chart interpretation of audio | `style/dump`, `gimmick/delay`, `expression/high contrast` | A relation between object placement and the represented sound, or between section changes and changes in the music |
| Mapping philosophy, concept, or lineage | `style/avant-garde`, `style/o2jam`, `expression/conceptual`, `expression/inspo` | A map-level design characterization or a relation to mapping traditions, maps, or mappers |
| Lane or playstyle organization | `style/N+1` | Independent use of the leftmost column relative to the remaining columns |
| External presentation or play context | `gimmick/storyboard`, `gimmick/video`, `gimmick/tag` | Storyboard, background-video, or multiplayer context outside the timed chart rows |

These families overlap. `skillset/wristjack`, for example, combines a
fast or dense chordjack condition with an optimal wrist-based playing
technique. `gimmick/delay` contains a high-snap stream but is distinguished by
the delayed sound effect that the stream represents. `style/tiebreaker`
contains skillsets from different categories but describes their coverage over
a map.

Several descriptions characterize deliberate map design: `avant-garde`
employs experimental mapping philosophies, `conceptual` uses unusual choices
to express part of a song, `o2jam` mimics a mapping tradition, `inspo` records
direct inspiration from other maps or mappers, and `memory` is designed around
a memorization concept. These are public design or provenance predicates. The
catalogue does not expose a mapper's unobserved private mental state.

The catalogue also contains broad predicates rather than closed structural
classes. `skillset/gimmick` covers distinct or obscure gameplay outside common
skillsets. `expression/simple` characterizes accessible, straightforward map
design. `gimmick/2B` identifies simultaneous placement of two or more objects,
but its presence as a community map tag does not define a frequency threshold.

### Stream terminology crosses namespaces

The three stream-related layers are related but not interchangeable:

- `style/jumpstream`, `style/handstream`, and `style/quadstream` describe a
  stream mixed with two-, three-, or four-note chords.
- `skillset/streams` describes continuous note hits, typically more than nine.
- `streams/bursts` describes groups of five to nine consecutive tapping notes.
- `streams/doubles` and `streams/quads` describe consecutive two- and
  four-note groups. They do not mean simultaneous two- and four-note chords.
- `streams/speed` describes constant high-BPM tapping, while
  `streams/stamina` describes dense tapping over a long period.

The names therefore encode chord topology, sequence length, rate, and sustained
player demand in separate predicates.

### Co-occurrence across namespaces

The following counts are direct co-occurrences on the same difficulty:

| Pair | Co-occurring difficulties |
| --- | ---: |
| `style/jumpstream` and `streams/stamina` | 295 |
| `style/jumpstream` and `skillset/streams` | 244 |
| `style/jumpstream` and `streams/speed` | 138 |
| `style/chordjack` and `skillset/speedjack` | 178 |
| `style/chordjack` and `skillset/wristjack` | 153 |
| `skillset/speedjack` and `skillset/wristjack` | 85 |
| `skillset/tech` and `tech/technical hybrid` | 66 |
| `skillset/tech` and `tech/complex snap` | 47 |
| `style/LN coordination` and `gimmick/LN inverse` | 48 |
| `style/LN release` and `gimmick/LN inverse` | 38 |
| `style/LN density` and `gimmick/LN inverse` | 34 |

Of the 458 difficulties carrying `skillset/tech`, 100 also carry at least one
`tech/*` tag. The coarse `skillset/tech` judgement and the two finer `tech/*`
predicates are related but not equivalent. Co-occurrence alone does not define
a hierarchy or causal relationship.

## Boundary case: `style/dump`

The catalogue defines `dump` through the mapping of sound to objects: object
groups express a sound's extension or intensity instead of individual notes
following every sound timing accurately. This can describe how one sound event
or onset is interpreted as a group of playable objects. It does not name a
particular row topology.

The snapshot contains 229 tagged difficulties. Their mean total length is
191.41 seconds, with a range of 30 to 993 seconds. Frequent co-tags include
`skillset/tech` (93), `streams/stamina` (61), `skillset/streams` (59),
`streams/speed` (53), `style/jumpstream` (42),
`expression/difficulty spike` (38), and `skillset/speedjack` (38).

Those co-tags show that dump can coexist with recognizable chart geometry and
skill demands. They do not determine whether the objects are dump. The same
dense stream, jack, or technical arrangement can have a different label when
its correspondence to the audio is different. Chart geometry without the
corresponding audio cannot establish the catalogue predicate.

The catalogue definition is about an observable chart-audio relationship. It
does not establish a claim about an unobserved private mapper intention, and it
does not define dump as random or erroneous placement.

## Boundary case: `style/tiebreaker`

The catalogue defines `tiebreaker` as a map containing most skillsets from
different categories and states that such maps are usually longer than five
minutes. The duration phrase is descriptive rather than an exceptionless
threshold. The definition refers to skillset-category breadth, not specifically
to a mixture of `style/*` labels.

The snapshot contains 16 tagged difficulties. Their mean total length is
315.625 seconds, with a range of 88 to 422 seconds. Twelve are at least 300
seconds long and eleven are strictly longer than 300 seconds. Frequent co-tags
include `style/LN coordination` (9), `skillset/tech` (8),
`tech/technical hybrid` (8), `streams/stamina` (8), `skillset/reading` (7),
`expression/progression` (6), `skillset/speedjack` (6), `style/LN mixed` (6),
and `style/generic hybrid` (6).

The defining predicate is coverage across skillset categories; map duration is
a separate usual characteristic in the catalogue description. An isolated
section can contain one of the constituent skillsets, but it cannot establish
the map's breadth. It also cannot establish whether the map has the usual
longer-than-five-minutes characteristic. No individual constituent section
needs to possess a section-local pattern also named `tiebreaker`.

## Comparison with the V3 formulation

### Assumptions supported by the tag evidence

Several formulation boundaries remain consistent with the catalogue and
snapshot:

- Community tags are multi-label rather than one-of-$K$.
- A community map tag is sparse, weak aggregate evidence. It is not copied to
  every section as local truth.
- An unmarked tag is not a negative label.
- Style is not an instantaneous demand value or a simple aggregate of demand.
- A community judgement about skill or technique is not an observation of a
  player's hidden state.

### Difference in predicate domains

The V3 formulation defines a fixed section-level profile $z_H(W)$ and states
the central hypothesis

$$
z_H(W)=\operatorname{StyleRead}(r_H(W)),
$$

where a style coordinate is a predicate over section action-demand geometry.
The conservative `StyleRead` form conditions on the chart arrangement, demand,
and incoming frontier state for $W$; the separate recognition contract
conditions on the chart arrangement and demand for $W$.

The catalogue coordinates do not all have that domain:

| Catalogue fact | Difference from a section-local action-demand predicate |
| --- | --- |
| `dump` and `delay` are defined relative to represented sounds; `high contrast` follows changes in the music across song sections. | Their truth is not determined by $H\vert_W$, $d_H\vert_W$, and incoming chart state without the corresponding audio. |
| `tiebreaker`, `progression`, and `high contrast` compare skillset coverage, ordered development, or different song sections; `tiebreaker` also has a usual whole-map duration characteristic. | Their positive evidence spans the map or multiple sections, rather than one isolated section. |
| `difficulty spike` and `repetition` compare challenge or recognizable elements across intervals. | They can be judged within a section only when that section contains all intervals being compared; a crop lacking the comparison cannot establish them. |
| `mixed rice`, `LN mixed`, `generic hybrid`, and `technical hybrid` require multiple constituents. | Their truth is a composition fact; one constituent crop is insufficient, and the catalogue does not fix the window in which all constituents must occur. |
| `reading`, `wristjack`, `stamina`, and `memory` use player-facing skill, technique, or cognition language. | They are community judgements about a chart under a playing convention, not player-state variables in the canonical profile. |
| `storyboard`, `video`, `tag`, and `inspo` refer to presentation, multiplayer, or provenance context. | The materialized chart $H$ contains timed `TAP`, `LN_START`, and `LN_CLOSE` rows, not those external facts. |
| `avant-garde`, `o2jam`, and `conceptual` characterize mapping philosophy, convention, or design concept. | Their catalogue meaning is not reducible to a named local topology. |

The complete audio representation $X$ exists elsewhere in the formulation.
The difference is specifically that $X$ is not an input to the stated
`StyleRead` or style-recognition contract.

### Difference in map aggregation

The V3 map-level observation model assigns each tag $k$ a section salience
$z_{H,k}(W_j)$ and pools those same-tag values across sections. This fits a
prominent local pattern such as `jumpstream`: a map vote can summarize its
salience or recurrence without making it true everywhere.

Some catalogue tags instead describe relations among other facts:

- `mixed rice` and `LN mixed` require multiple style families;
- `generic hybrid` and `technical hybrid` require rice and long-note
  constituents;
- `tiebreaker` requires broad coverage across skillset categories; its
  catalogue description separately records a usual map-duration
  characteristic;
- `progression`, `high contrast`, and `repetition` require ordered or repeated
  comparison across intervals.

For these tags, the map-level predicate is not necessarily a pooling of a
same-named atomic predicate already true inside individual sections.

### Difference in section-annotation states

The V3 section annotation value has three states: present, explicitly absent,
and unknown or not judged. The catalogue contains predicates with an additional
scope distinction: some are not defined on an isolated section, or the section
does not contain the evidence needed to evaluate them.

For an applicable section predicate, “explicitly absent in the judged section”
and “the section was not judged” are different from “this predicate is not
defined at section scope.” The existing section states distinguish present,
absent, and unknown for a section predicate, but they do not carry a map-level
truth value or separately encode applicability to the annotation scope.
`Tiebreaker` has its positive and negative truth conditions at map scope, even
though its constituent skillsets occur in sections.

## Section-level annotation truth conditions

The catalogue descriptions imply different minimum evidence domains. They do
not imply one fixed section duration.

| Evidence domain | What a positive judgement depends on | What an isolated crop cannot establish |
| --- | --- | --- |
| Local chart sequence | Repeated rows, lane recurrence, chord-size changes, long-note overlap or releases, or mixed snap divisors | Most stream, jack, and long-note predicates from a single row; `gimmick/2B` is the exception whose structural condition can occur in one simultaneous row |
| Sustained chart interval | Sequence length, constant rate, density over time, frequent complex snaps, or stamina-scale persistence | `streams`, `bursts`, `speed`, `stamina`, or `skillset/tech` from a momentary event |
| Multiple constituents | At least two distinct rice or long-note styles for `mixed rice` and `LN mixed`; both named rice and long-note constituents for `generic hybrid` and `technical hybrid` | A composition tag from a crop containing only one constituent or one patterning style |
| Chart plus aligned audio | How an object group represents extension, intensity, exact timing, or a delayed sound effect | `dump` or `delay` from a silent chart image |
| Aligned audio plus multiple song sections | Mapping ideas that follow changes in the music and create contrast between song sections | `high contrast` from chart sections without the corresponding music, or from one section without a comparison |
| Relative interval context | A spike relative to surrounding play, or a recognizable recurrence | `difficulty spike` or `repetition` without the intervals being compared |
| Ordered sections or whole map | Progression, contrast among song sections, or skillset-category coverage; whole-map duration separately determines whether a `tiebreaker` instance has the catalogue's usual length characteristic | `progression`, `high contrast`, or `tiebreaker` from one isolated section |
| External asset, mode, or provenance | Storyboard behavior, background-video reference, multiplayer tag design, or inspiration from another map or mapper | `storyboard`, `video`, `tag`, or `inspo` from timed chart rows alone |
| Player-facing convention | The skill, technique, or cognitive demand attributed to the chart by the community | An individual player's capacity, chosen execution, fatigue, or observed internal state |

A local structural occurrence and a community map tag are not the same
annotation. A section can visibly contain a qualifying chord or stream pattern
while the snapshot supplies only a difficulty-level positive vote. Conversely,
a difficulty-level vote supplies no location for the sections that motivated
it.

## Evidence limits

- The catalogue and beatmap metadata used for the measurements are local
  generated dataset files and are not tracked by Git.
- `top_tag_ids` records positive difficulty-level community evidence; absence
  from that list is not a negative judgement.
- The counts record tag-ID presence, not vote magnitude or agreement ratio.
- Community tags do not provide section boundaries or localization.
- Catalogue definitions do not establish a minimum section duration or a
  threshold for how often a local pattern produces a map-level tag.
- Tags with `ruleset_id == null` are mode-independent definitions and may omit
  mania-specific detail.
- Co-occurrence establishes only that two tags were attached to the same
  difficulty. It does not establish synonymy, hierarchy, or causation.
- Two difficulties in the snapshot each carry 50 top tags, including 36 of the
  44 tags in this scope. The only local occurrences of `style/o2jam` and
  `style/N+1` both come from one of these high-tag-count difficulties.
- The local `tiebreaker` population is 16 and includes a tagged difficulty
  shorter than five minutes, consistent with the catalogue's use of “usually.”
- Short catalogue descriptions leave the section-versus-map boundary
  unspecified for broad tags such as `simple`, `conceptual`, and `memory`.

## Sources

- [osu! Wiki: beatmap tags](https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags)
- [osu! Wiki: osu!mania patterns](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania)
- [osu! Wiki: osu!mania stream patterns](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania/Stream)
- [osu! Wiki: osu!mania jack patterns](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania/Jack)
- [osu! Wiki: osu!mania hold-note patterns](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania/Hold_note)
- [Local 4K style tag reference](osu_mania_4k_style_tag_reference.md)
- [Pulsefield V3 gameplay-state formulation](../formulation/gameplay-state.md)
- [Pulsefield V3 notation](../formulation/notation.md)
