# Timing v3 Task Definition

Status: accepted Phase 1 working definition, synchronized with the stable goal
contract on 2026-08-13.

This file describes the current delivery contract. Experiment 001-006 cards,
clarifications, repairs, results, and problem-log entries remain immutable
historical evidence. Later cards/results become immutable when accepted as
their own frozen checkpoint and may never rewrite or reinterpret Experiment
001-006 outcomes.

## Objective

Timing v3 Phase 1 consumes exactly one shift-0 cached BeatThis prediction per
audio identity and produces one conservative product result over a single
global beat axis. The public v3 unit is a constant-tempo section over absolute
beat coordinates, not an independently anchored osu!-style redline:

```text
beat [a, b)  tempo=constant  bpm=q
```

Phase 1 supports only:

1. one constant BPM for all or part of a track; and
2. a BPM jump on an integer beat boundary without resetting phase.

Every audio identity must end in exactly one product state:

- `v3_accepted`: a valid phase-continuous constant/jump grid;
- `v2_fallback`: current timing v2 is retained with an exact fallback reason;
- `hard_failure`: input, schema, integrity, or execution failed and no
  best-so-far grid is emitted. This count must be zero in the final delivery.

Fallback is a usable product degradation, not a successful v3 candidate.
Pure-v3 candidate quality and selected-product safety are always reported
separately. Ramp production, rubato, swing, arbitrary tempo curves, meter
inference, and online metadata as inference input are outside Phase 1. Ramp
examples remain an audit queue only and require a later ramp-specific goal.

## Asset snapshot

The local corpus snapshot used by this task is:

- 5,050 unique audio files and 14,689 4K mania beatmaps in 4,201 beatmapsets;
- 5,050 complete `final0`, shift-0, float32 BeatThis caches at 50 Hz;
- cache/audio linkage is exact for all 5,050 unique audio paths;
- duration: 30.24 s minimum, 148.52 s median, 300.95 s p90,
  483.52 s p99, and 4,358.12 s maximum; 29 tracks are at least 600 s;
- 14,617 maps have parseable red timing, 813 audio groups have at least one
  varying-redline difficulty, and a deliberately permissive monotonic heuristic
  finds 48 possible ramp maps;
- official osu! API metadata has been snapshotted locally for all 4,201 sets.

Artifacts are experiment inputs, not repository sources of truth. Every result
must record the index hash, cache fingerprint, code revision, config hash, and
audio grouping key.

The existing cache payload proves basic provider/checkpoint/cache-version,
shift, frame-rate, dtype, audio-key, frame-count, and source-path fields. It
does not by itself prove the historical BeatThis package version, internal
chunk size, border/padding, aggregation mode, builder command, or generator
environment. Until an auditable generator/build manifest or a versioned
regeneration/migration plan closes that gap, no experiment may treat an
assumed BeatThis chunk seam as a musical boundary or analysis variable.

## Inference/evaluation boundary

Family A, the cache-only baseline and fast path, may read only one shift-0
cache's `beat_prob`, `downbeat_prob`, frame rate, and cache/generator
provenance. It must not read `.osu`, hit objects, redlines, song titles,
official API BPM, or manually/web-audited labels.

Family B, a cache plus deterministic raw-audio verifier, may run only under a
new frozen Experiment Card that explicitly authorizes it. It must retain the
same BeatThis cache, candidate identities, and v2 comparator; freeze the
feature extractor, parameters, feature-cache schema, and source hash; and
report paired quality gain and added latency separately. It may not be chosen
or tuned on an already viewed fresh holdout. Family B is reported as a
separate inference family, but exposure remains shared by audio/cache identity
across all families; it may not treat identities exposed under Family A as
fresh. Family B cannot enter the default path unless it improves its
preregistered hard-stratum primary error by at least 10% relatively and adds no
more than five seconds p90 latency. No Family B execution is authorized by the
current Exp007 result.

Those other sources are weak supervision and evaluation evidence:

- redlines propose beat periods, phase anchors, and change boundaries;
- hit-object starts measure rational-grid concentration independently of the
  BeatThis activations; hold ends are a lower-weight signal;
- agreement among difficulties for the same audio raises confidence;
- local official API BPM is a source-stamped, map-derived cross-check, not
  audio truth;
- manually verified or external music-source evidence is a separate highest
  confidence tier and must identify the exact recording/version.

All train/tune/test splits and all aggregate statistics are grouped by resolved
audio path first. Difficulties of the same audio may never cross a split.

Viewing any identity, cache, prediction, grid, metric, diagnostic, runtime,
failure, `.osu`, render, raw audio, or batch aggregate exposes that audio
identity. Exposure is append-only and keyed by unique audio/cache identity. If
any row of a sealed stage is exposed early, the affected gate is invalid; all
identities from the exposure enter the exclusion manifest and a new
audio-disjoint held-out stage must be established before reuse of that gate.

### External-source policy

Network evidence is useful only when its role is explicit:

- AcoustID can fingerprint a local recording and link it to MusicBrainz;
  MusicBrainz supplies recording identity, ISRC, and duration, but not a tempo
  curve. This is an identity/version tier, not a BPM label.
- The official osu! API supplies scalar beatmap/beatmapset BPM fields. The local
  snapshots make these reproducible, but they remain map-derived and correlated
  with redlines.
- Spotify's audio-features endpoint exposes one estimated BPM and its audio
  analysis historically exposed beat/section tempo, but both endpoints are now
  deprecated. Spotify also states that Spotify content may not be ingested into
  an ML/AI model. Spotify data is therefore excluded from this experiment.
- A scalar catalog BPM cannot establish `constant` versus `jump` versus `ramp`.
  Variable-tempo truth requires exact-recording section evidence, a trusted
  score/performance note, or manual listening/annotation. Absence of such a
  source leaves the row weak or ambiguous.

External evidence is stored as an append-only source manifest containing URL or
identifier, provider, retrieval time, recording-identity evidence, value,
confidence, and reviewer notes. It may validate labels but may not alter a
frozen test prediction.

## Canonical representation

Let `B(t)` be continuous cumulative beat position and `T(x)` its inverse.
Sections partition one global beat axis:

```text
section[i].end_beat == section[i + 1].start_beat
T_i(section[i].end_beat) == T_{i + 1}(section[i + 1].start_beat)
```

A section owns at least:

- integer `start_beat` and `end_beat`, defining `[a, b)`;
- derived `start_time_ms` and `end_time_ms`;
- one finite constant `bpm`;
- evidence/confidence diagnostics and section/change provenance;
- cache-only or audio-assisted inference-family identity;
- optional meter/downbeat phase, which is not required to establish beat-phase
  continuity.

A jump is a derivative discontinuity between adjacent constant sections, not
a gap or phase reset. Section end time is derived from its start boundary,
integer beat count, and BPM; it is never a second independent anchor. Adjacent
beat intervals must be exactly contiguous, adjacent derived boundary times
must be continuous, and serialization round-trip seam error must not exceed
5 ms. Decoder block cuts and BeatThis chunk cuts may not force a section to
close, and no local block may establish a new phase or tempo-alias origin.

## Candidate inference architecture

The Phase 1 family is a bounded, boundary-conditioned dynamic program or beam
search:

1. Calibrate and peak-refine the single BeatThis beat/downbeat activation.
2. Build a multi-scale coarse tempo posterior over the whole track.
3. Establish an initial frontier containing global beat index, phase, current
   BPM, tempo alias family, and open-section state; retain alternatives until
   evidence justifies commitment.
4. Generate constant and jump-boundary section candidates only.
5. Assemble one global path. Transition cost includes BeatThis support,
   boundary phase, tempo/alias switching, change sparsity, and
   boundary-conditioned left/right tempo evidence.
6. Emit `v3_accepted` only for a fully valid, accepted grid. Otherwise preserve
   the complete current-v2 result as `v2_fallback` with an exact reason. Input,
   schema, integrity, and execution failures remain `hard_failure`; they do not
   become fallback or best-so-far v3 output.

`.osu` and external metadata affect offline evaluation, label confidence, and
error attribution only. They do not tune inference parameters, thresholds,
features, candidates, fallback routing, or metric definitions outside a newly
frozen authorized card, and they do not appear in steps 1-6.

## Priors

Use hard bounds only to reject impossible values. The current 20-1000 BPM
range remains a candidate-generation guard, not a uniform musical prior.

The scored prior is hierarchical:

- an audio-grouped empirical prior over log BPM;
- a global tempo-alias orbit, initially the current multipliers
  `{1/4, 1/3, 1/2, 1, 2, 3, 4}`;
- strong persistence within a section;
- sparse jumps supported by a local fit gain and change evidence;
- penalties for short unsupported sections and alias switching;
- a downbeat preference for boundaries, but no hard requirement.

Priors must be fit on the tuning split only, serialized with provenance, and
evaluated against an unmodified broad-prior candidate set so that a narrow
prior cannot hide rare tempi.

## Long-track decomposition

Independent fixed windows are forbidden because each can choose a different
phase or tempo alias. Every block shares the same absolute beat axis. Its
incoming frontier carries at least exact phase, absolute beat index, current
BPM, alias family, open-section start, and accumulated score. Blocks use
overlap/lookahead to re-evaluate incoming states, exact half-open ownership,
and fixed-lag commit so the previous core is not committed before the next
evidence arrives. A section may cross any block cut.

The first block may retain multiple phase/alias hypotheses. Long intros,
silence, or weak-rhythm regions may bootstrap from the first reliable evidence
island and backfill earlier beats. A final traceback/rerank may inspect only
compact retained states; it may not restore Experiment 004's dense whole-track
replay.

The initial duration sweep is:

- fixed `30/10`, `60/20`, and `90/30` second core/overlap candidates; and
- a musical candidate targeting 64 beats with 16-beat lookahead, clamped to
  the same 30-90 s core and 10-30 s lookahead range.

No schedule is a default hypothesis. The winner is committed only by the
frozen source-only selection protocol; weak `.osu` headline metrics cannot
select it. Long-track claims use the same single full-song cache sliced by the
runner. A block may not rerun BeatThis, use a shifted cache, or interpret an
unproved BeatThis chunk seam as a section boundary.

## Evaluation and error attribution

Every v2/v3 comparison uses the identical cached activation. Metrics are
reported at the audio-group level and stratified by label confidence, duration,
tempo family, redline density, and stable/jump/ramp-audit/ambiguous class. A
ramp-audit label diagnoses unsupported material; it does not authorize a ramp
prediction.

| Layer | Question | Required measurements |
| --- | --- | --- |
| Cache | Is the input valid and comparable? | coverage, schema/config fingerprint, frame count, NaN/range checks |
| BeatThis evidence | Is a usable beat signal present? | activation at trusted beats, local peak recall/precision, downbeat support |
| Beat path | Did decoding lose or invent beats? | F-measure, CMLc/CMLt, AMLc/AMLt, continuity runs, missing/extra-beat rate |
| Alias/tempo | Is the correct tempo family selected? | raw and alias-aware BPM error, alias switches, API-BPM agreement by trust tier |
| Sections | Are jumps localized and supported? | jump-boundary precision/recall/time error, section count, unsupported short-section rate |
| Phase | Does timing remain aligned? | mean/p50/p90/max phase error in ms and beats, 30/60 s local errors |
| Accumulation | Does error grow along the track? | endpoint beat-time error, drift slope ms/min, max prefix cumulative error |
| Assembly | Did decomposition introduce seams? | hard schema continuity, overlap disagreement, boundary phase before/after assembly |
| Object grid | Does the result explain mapper placement? | rational-subdivision residual, inlier rate, start/end residual separately |
| Product | Is the output usable? | pure-v3 coverage/failure, selected-product fallback rate/reasons, immutable product status, p50/p90/max runtime, peak memory, deterministic replay/resume |

The attribution order is cache -> BeatThis evidence -> beat decoder -> tempo
alias -> section model -> long-track assembly -> comparator uncertainty. A bad
redline or sparse/syncopated object pattern must be allowed to end as
`ambiguous`; it must not be forced into an algorithm failure.

Cumulative metrics are emitted twice: raw beat-count drift exposes half/double
tempo semantic errors, while BPM-80-160-canonicalized drift isolates alignment
after tempo-family normalization. Raw drift can be enormous under a perfectly
regular 2x alias and therefore must never be silently averaged with the
alias-normalized phase guard.

Headline guards compare paired v2/v3 results on the same eligible audio:

- mean phase error no more than 10% worse than v2;
- p90 phase error no more than 15% worse than v2;
- no section-boundary discontinuity above 5 ms after serialization round-trip;
- no segment/section explosion beyond a frozen cap;
- no material increase in fit failures;
- cumulative drift is reported even when the mean phase guard passes.

These are first-pass selection guards, not claims of perceptual equivalence.

Every stage also reports input/cache validity, BeatThis peak support, raw and
alias-aware local BPM error, phase mean/p50/p90/max in milliseconds and beats,
endpoint drift, max-prefix drift, drift slope, 30/60-second local drift, exact
seam/serialization continuity, confidence-coverage/risk curves, and explicit
ambiguous and comparator-unavailable denominators. Family B additionally
reports paired relative gain and added latency against Family A.

All metrics aggregate by audio identity first and then stratify by duration,
stable/jump, dense, long, confidence, comparator availability, and inference
family. Comparator-unavailable rows do not enter oracle phase/BPM aggregates,
but they remain in cache validity, execution, fallback, runtime, and product
coverage denominators.

## Promotion and safety gates

The final frozen candidate must satisfy all of the following:

- mean phase error relative to current v2 on the identical cache is at most
  `1.10`;
- p90 phase error relative to v2 is at most `1.15`;
- every serialized section seam is at most 5 ms;
- alias-normalized accumulation metrics regress by no more than 10% relative
  to v2;
- at least one preregistered core value metric—max-prefix drift, endpoint
  drift, or jump-boundary error—improves by at least 10% relatively on both the
  fresh holdout and broad500;
- fresh-holdout pure-candidate fallback is at most 5%;
- all 5,050 identities end in valid v3 or explicit v2 fallback, with zero
  full-corpus hard/integrity failures;
- final-path p90 runtime is at most 30 seconds, every row has a 180-second hard
  timeout, and peak memory stays within the frozen card's bound;
- long, jump, and dense strata independently pass their frozen minimum-
  denominator and regression guards; and
- deterministic replay, zero-compute resume, and source/config/cache identity
  checks pass.

Passing safety without a preregistered value improvement is a negative result,
not a promotion. Production defaults never change without explicit owner
approval.

## Ordered experiment and delivery gates

Each behavior-affecting hypothesis has one frozen Experiment Card. Source,
docs, schema, and synthetic review precede any new real data. Gates open only
in this order:

1. source-owned protocol/schema/unit/synthetic verification;
2. exposed schedule16;
3. source-only schedule-winner commit;
4. winner-only weak-evidence veto;
5. exposed repair80;
6. no-new-data acceptance review;
7. fresh audio-disjoint holdout100;
8. broad500;
9. final frozen full5050 replay;
10. delivery and disabled-by-default integration handoff.

A negative, ambiguous, timeout, integrity failure, or hard-gate failure at any
stage stops the loop before later data. Later data cannot rescue an earlier
failure. A behavior-affecting change to the algorithm, feature, threshold,
candidate, objective, fallback route, evaluator, or metric requires a new card
and applicable held-out provenance. Full5050 is coverage/tail-risk replay, not
a tuning set.

Final delivery includes the updated task definition; every Experiment Card
and Result; source and unit/integration tests; source/config/cache/exposure
manifests; holdout, broad, and full reports; runtime/quality/risk-coverage
reports; representative accepted, fallback, ambiguous, and tail-risk
diagnostics; the complete problem/decision log; and disabled-by-default
integration with an explicit current-v2 rollback runbook.

The current Exp007 result completes only gate 1. Writing Exp008 requires an
explicit human-owner decision. Exp008 may then authorize only gates 2-5; it
cannot authorize holdout100, broad500, full5050, ramp support, or a production
default change.

## Explicit Phase 1 exclusions

Phase 1 does not include production ramp output, rubato, swing, arbitrary
tempo curves, meter/time-signature inference, BeatThis training or fine-
tuning, multi-shift BeatThis ensembles, `.osu`/metadata/network/manual inputs
to inference, use of full5050 for tuning, or an automatic production-default
switch. Ramp material remains an audit queue only.

## Completion conditions

Phase 1 is complete only when all of these are proven together:

1. the constant/jump source, schema, runner, evaluator, and tests are complete;
2. final algorithm/config/feature/comparator policy is frozen before the fresh
   holdout;
3. holdout100 and broad500 pass their frozen gates;
4. the identical source/config completes full5050;
5. all 5,050 rows have immutable, traceable product status;
6. every promotion, safety, runtime, and long-stratum gate passes;
7. zero-compute resume/identity replay proves reproducibility;
8. every manifest, report, representative diagnostic, experiment card/result,
   source/test deliverable, task-definition update, problem/decision entry,
   disabled-by-default integration, and rollback document listed above is
   delivered; and
9. current v2 remains a complete rollback unless the owner explicitly approves
   a production-default change.

A synthetic, protocol, schedule, or repair pass alone is not Phase 1
completion.

## Pause conditions

Stop and report evidence rather than extending authority when the next step
needs an unauthorized data layer; an exposure boundary is broken; cache
generator provenance is insufficient for the intended claim; a frozen metric,
truth policy, or product tradeoff needs owner judgment; a blocker persists
after safe alternatives are exhausted; two independent plausible families
miss the value threshold; a runtime/resource hard stop fires; or continuing
would change the constant/jump, evidence, or production boundary.

A pause report names the completed checkpoint, reproducible evidence, blocker,
ruled-out alternatives, remaining choices, and the recommendation to `KILL`,
`MUTATE`, or create one new bounded `TEST` card.

## Closest analogies and evidence boundary

- BeatThis produces beat/downbeat frame probabilities at 50 Hz and processes
  long audio in 30 s excerpts; it does not itself provide the required global
  section representation: <https://arxiv.org/html/2407.21658v1>.
- DBN/DP trackers expose explicit tempo persistence and transition priors, but
  fixed tempo/meter assumptions can suppress real changes:
  <https://madmom.readthedocs.io/en/v0.16/modules/features/beats.html> and
  <https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html>.
- osu! uninherited timing points define beat length; inherited timing points do
  not define BPM: <https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29>.
- Osu2MIR is the closest dataset analogy for separating sparse and dense timing
  changes and treating dense redlines as requiring curation:
  <https://arxiv.org/html/2509.12667v1>.
- AcoustID and MusicBrainz are suitable for recording identity but do not
  provide a tempo curve: <https://acoustid.org/webservice> and
  <https://musicbrainz.org/doc/recording>.
- Spotify's scalar audio-features and sectional audio-analysis endpoints are
  deprecated and carry a restrictive ML/AI policy, so this project does not use
  them: <https://developer.spotify.com/documentation/web-api/reference/get-audio-features>
  and <https://developer.spotify.com/documentation/web-api/reference/get-audio-analysis>.

The proposed novelty, if any, is the Pulsefield-specific representation and
evaluation coupling: a phase-continuous beat-index section graph over one
BeatThis cache, audited with multi-difficulty redlines and object placement.
The DP, tempo priors, change-point search, and weak-label consensus are known
algorithmic families and are not claimed as novel.
