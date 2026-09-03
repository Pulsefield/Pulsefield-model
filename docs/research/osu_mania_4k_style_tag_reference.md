# osu!mania 4K Style Tag Reference

## Scope and authority

This page records a local dataset snapshot and public visual references for
osu!mania 4K `style/*` user tags. It is a research aid, not a Pulsefield V3
style-enum specification. The canonical distinction between section style and
map-level community evidence remains in
[`gameplay-state.md`](../formulation/gameplay-state.md#13-map-level-community-style-tags).

osu! user tags are community-voted map-level labels. A tag can describe a
prominent or recurring part of a map without applying to every section, and
multiple tags can apply to one difficulty. Treat the local tags as weak
aggregate evidence rather than section-level ground truth.

## Dataset snapshot

- Local tag catalogue: `dataset/metadata/osu_tags_2026-08-07.json`
- Catalogue endpoint: `GET https://osu.ppy.sh/api/v2/tags`
- Catalogue fetched at: `2026-08-07T07:02:13.461073+00:00`
- Reference links checked: 2026-09-03
- Beatmap metadata source: each `dataset/0/<beatmapset_id>/metadata.json`
- 4K selection: `mode_int == 3` and `cs == 4`
- Style selection: tag name starts with `style/` and catalogue `ruleset_id` is
  `3` (osu!mania) or `null` (mode-independent)
- Count unit: one 4K difficulty whose `top_tag_ids` contains the tag ID; counts
  are not vote totals and are not mutually exclusive

| Measurement | Count | Share of local 4K difficulties |
| --- | ---: | ---: |
| Beatmapsets | 4,201 | — |
| All difficulties | 17,728 | — |
| osu!mania 4K difficulties | 14,787 | 100.00% |
| 4K difficulties with any top tag | 4,568 | 30.89% |
| 4K difficulties with an applicable `style/*` tag | 2,688 | 18.18% |
| Beatmapsets containing one of those 4K difficulties | 1,868 | — |

The catalogue contains 134 user tags in total and 17 `style/*` tags applicable
to osu!mania. All 17 occur in this local snapshot. The raw catalogue value for
tag ID 117 is `style/LN mixed ` with a trailing space; use the stable ID for
joins and trim whitespace only for display.

## High-frequency 4K styles

The following 12 tags occur on more than 100 local 4K difficulties. “Static
fit” describes the minimum useful visual treatment, not an automatic
classification rule.

| Tag | Local 4K count | Visual cue | Static fit | Reference |
| --- | ---: | --- | --- | --- |
| `style/chordjack` | 833 | Evenly spaced chord rows repeatedly occupy the same columns, producing a wall-like stack. | One representative crop | [Official image](https://raw.githubusercontent.com/ppy/osu-wiki/master/wiki/Beatmap/Pattern/osu%21mania/Jack/img/chordjack.png), [video at 1:05](https://www.youtube.com/watch?v=YHgyTTSYex4&t=65s) |
| `style/jumpstream` | 736 | A single-note stream periodically includes 2-note chords; common row-size rhythms resemble `2-1-1-1` or `2-1-2-1`. | One representative crop | [Official image](https://raw.githubusercontent.com/ppy/osu-wiki/master/wiki/Beatmap/Pattern/osu%21mania/Stream/img/jumpstream.png), [video at 1:30](https://www.youtube.com/watch?v=YHgyTTSYex4&t=90s) |
| `style/LN coordination` | 497 | Several long notes remain held while other taps, long-note heads, or releases occur. | Multi-frame crop or 2–4 seconds of video | [Official definition](https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags), [o!mLN 3 category image](https://raw.githubusercontent.com/ppy/osu-wiki/master/wiki/Tournaments/o%21mLN/3/img/mappool-categories.jpg), [4K LN video](https://www.bilibili.com/video/BV152421N7F9/) |
| `style/generic hybrid` | 320 | Straightforward rice sections and long-note sections alternate or interleave in the same difficulty. | At least two time points or a long crop | [Chinese LN and hybrid article](https://www.bilibili.com/read/cv10461847/), [4K gameplay](https://www.youtube.com/watch?v=5aVquNAIe6M) |
| `style/chordstream` | 286 | A continuous stream mixes rows with different chord sizes, usually 1–3 notes in 4K. Unlike chordjack, repeated occupancy of the same columns is not required. | One sufficiently long crop | [Official structural image](https://raw.githubusercontent.com/ppy/osu-wiki/master/wiki/Beatmap/Pattern/osu%21mania/Stream/img/chordstream.png), [official pattern page](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania/Stream) |
| `style/LN mixed` | 248 | At least two long-note styles, such as coordination, release, inverse/wall, or density, occur in one difficulty. | Two or more time points or video | [LN overview at 6:05](https://www.youtube.com/watch?v=YHgyTTSYex4&t=365s), [Chinese LN classification](https://www.bilibili.com/read/cv10461847/) |
| `style/handstream` | 245 | A stream periodically includes 3-note chords; a common row-size rhythm resembles `3-1-1-1`. | One representative crop | [Official image](https://raw.githubusercontent.com/ppy/osu-wiki/master/wiki/Beatmap/Pattern/osu%21mania/Stream/img/handstream.png), [video at 1:39](https://www.youtube.com/watch?v=YHgyTTSYex4&t=99s) |
| `style/dump` | 229 | Object groups express a sound's extension or intensity rather than mapping every audible onset one-for-one. Geometry alone may resemble dense stream or tech. | Video with audio and surrounding context | [Video explanation at 4:53](https://www.youtube.com/watch?v=YHgyTTSYex4&t=293s), [4K auto example](https://www.bilibili.com/video/BV1kEEqz6EDz/), [mapping discussion](https://www.reddit.com/r/osumania/comments/1tvlk3q/rapid_fire_4k_modding_requests/) |
| `style/LN release` | 203 | Long-note tails end at different times, creating a staggered tail skyline and demanding ordered releases. | Long crop plus short video | [Official hold-note page](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania/Hold_note), [video at 6:39](https://www.youtube.com/watch?v=YHgyTTSYex4&t=399s), [4K density/release gameplay](https://www.youtube.com/watch?v=nsQviJ4LoRQ) |
| `style/LN density` | 191 | Long-note streams continue with little empty space and frequent overlapping heads, bodies, and tails. | Long crop; video confirms continuity | [Official inverse image](https://raw.githubusercontent.com/ppy/osu-wiki/master/wiki/Beatmap/Pattern/osu%21mania/Hold_note/img/inverse.png), [4K density/release gameplay](https://www.youtube.com/watch?v=nsQviJ4LoRQ) |
| `style/longjack` | 190 | A long chain of consecutive notes occupies the same column; the official pattern guide uses four or more as a practical visual starting point. | One representative crop | [Official image](https://raw.githubusercontent.com/ppy/osu-wiki/master/wiki/Beatmap/Pattern/osu%21mania/Jack/img/longjack.png), [video at 0:50](https://www.youtube.com/watch?v=YHgyTTSYex4&t=50s) |
| `style/mixed rice` | 146 | Multiple non-LN styles, such as jack, stream, jumpstream, handstream, stairs, or tech, appear across one difficulty. | Three or more time points or video | [Pattern overview at 0:39](https://www.youtube.com/watch?v=YHgyTTSYex4&t=39s), [Project Loved example](https://osu.ppy.sh/community/forums/topics/1980971) |

The official chordstream pattern image is a structural illustration rather
than a 4K-specific example. The current user-tag catalogue and the local 4K
votes, rather than historical keymode usage of the term, define this archive's
scope.

## Lower-frequency applicable styles

| Tag | Local 4K count | Catalogue meaning |
| --- | ---: | --- |
| `style/avant-garde` | 30 | Experimental design that deliberately pushes beyond ordinary gameplay or aesthetic conventions. |
| `style/quadstream` | 27 | A stream containing 4-note chords. |
| `style/tiebreaker` | 16 | A long map, usually over five minutes, containing skill sets from several categories. |
| `style/o2jam` | 1 | Mapping that imitates traditional O2Jam techniques. |
| `style/N+1` | 1 | The leftmost column is mapped independently while the remaining columns follow a standard playstyle. |

## Interpretation boundaries

- `chordjack` is distinguished from `chordstream` by repeated same-column
  occupancy, not merely by high chord density.
- `jumpstream` and `handstream` are distinguished primarily by inserted chord
  size: two notes versus three notes.
- `LN coordination`, `LN release`, and `LN density` respectively emphasize
  simultaneous held-note interaction, ordered tail events, and sustained
  long-note occupancy. One crop can exhibit more than one property.
- `generic hybrid`, `LN mixed`, and `mixed rice` are combination labels. A
  short crop that contains only one component cannot establish the map-level
  tag.
- `dump` depends on the relationship between charted objects and audio. A
  silent image can only suggest a dense or irregular cluster, not establish
  the tag.
- Community terminology can be broader or older than the current official
  catalogue. Use the official tag ID and description as the naming authority;
  use community material only for visual intuition and examples.
- Some Bilibili sources prohibit unauthorised reproduction. Link to their
  original pages instead of mirroring article images or extracted video
  frames.

## Primary sources

- [osu! Wiki: Beatmap tags](https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags)
- [osu! Wiki: osu!mania stream patterns](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania/Stream)
- [osu! Wiki: osu!mania jack patterns](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania/Jack)
- [osu! Wiki: osu!mania hold-note patterns](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania/Hold_note)
- [osu! Wiki: osu!mania mapping guide](https://osu.ppy.sh/wiki/en/Guides/osu%21mania_mapping_guide)
- [YouTube: All 4k Mania Patterns Explained](https://www.youtube.com/watch?v=YHgyTTSYex4)
- [Bilibili: Osu!Mania4k newcomer guide](https://www.bilibili.com/read/cv11073476/)
- [Bilibili: long-note and inverse discussion](https://www.bilibili.com/read/cv10461847/)
