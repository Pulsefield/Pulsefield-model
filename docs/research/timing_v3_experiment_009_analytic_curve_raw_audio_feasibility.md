# Experiment Card: Timing v3 Experiment 009 — Analytic Curve and Raw-Audio Feasibility

## Mode

- Mode: planner
- Route: TEST
- Card state: draft; drafting is owner-authorized, but this card has not yet
  been accepted for implementation or execution.
- Source idea: represent constant tempo, integer-beat jumps, and linear BPM
  ramps on one analytic beat axis, then score a bounded candidate set with
  deterministic evidence derived from the existing 10 ms log-mel audio path,
  independently of BeatThis logits.
- Active owner objective, recorded verbatim from the 2026-08-13 goal service:

  ```text
  做出timing v3模块. 除开beatthis的logits prior, 还能利用原始audio的信息. 最终期望: 5s内对于10min内长度的audio, 单一bpm的情况做到99%识别, bpm跳变的情况做到80%, bpm ramp线性改变的情况做到70%以上识别, 全局offset容差70ms, 但相邻beat grid必须phase连贯. 在真实5050首歌曲上跑测试.
  相信自己, 当陷入困境时尝试挖掘出该问题的本质, 并think out of box.
  ```

- Owner drafting authorization: the exact earlier message
  `授权你之后的一切起草行为, 继续` authorizes this draft. It does not by
  itself accept this card, authorize code, or authorize opening real audio,
  caches, labels, `.osu` files, metrics, or identities.
- Goal supersession rule: the active objective expands future Timing-v3 work
  beyond the accepted Phase-1 constant/jump boundary. Experiments 001-007 and
  their results remain immutable historical evidence. The unaccepted Exp008
  draft remains preserved and unexecuted; it is not the next executable card
  under the expanded raw-audio/ramp objective.
- Repository snapshot: Git `be8993b7fe8325a98d4d8d3b80138b1bd8ffe1b7` plus the
  existing dirty/untracked Timing-v3 research tree. Relevant pre-card hashes:

  | File | SHA-256 |
  | --- | --- |
  | `docs/research/timing_v3_task_definition.md` | `f26842f86ce038c4afd1414e9aa235580c56e28a4ede567485f069da2e32954a` |
  | `docs/research/timing_v3_experiment_008_exposed_schedule_repair_execution.md` | `6fa6eab50edc138484c77a9d00657a2e8ccc4b4fa92c33cfae79771642e112fe` |
  | `src/pulsefield_model/timing/v3/schema.py` | `019ed7e942ef4a994d7ff0869de433e410dfc932c320dd7231bde83173a37e75` |

- Source snapshot / evidence grade: source inspection plus primary-source
  algorithm review; no Exp009 code, benchmark, real-audio, or 5,050-song
  evidence exists.
- Frozen reference host for the Exp009 runtime gate: MacBook Air `Mac17,3`,
  Apple M5 (10 CPU cores), 24 GiB RAM, arm64, macOS 26.6.1 build `25G76`.
  The benchmark preflight must prove AC power and Low Power Mode disabled.
  Avoiding concurrent repository tests/benchmarks is operator guidance rather
  than a provable acceptance condition; an exclusive Exp009 lock and the
  three system load averages are recorded for every run. Other platforms are
  diagnostic only and cannot pass or fail this card's runtime gate.

## Hypothesis

A phase-continuous analytic curve with exact constant, integer-beat jump, and
time-linear BPM-ramp sections can be evaluated without phase resets. A small
raw-audio verifier can then reconstruct deterministic multi-band positive
spectral-flux evidence from the repository's existing 10 ms log-mel features
and score frozen synthetic analytic candidates directly in beat phase. On
fixed synthetic audio, a curve inside the owner's 70 ms tolerance-equivalence
class should outrank every invalid offset, alias, boundary, and ramp decoy,
while the complete post-prior raw-audio path for a 600-second track stays
strictly below five seconds on the reference Apple-Silicon machine.

The key architectural claim is deliberately narrow: raw audio can add cheap,
phase-sensitive evidence to a small candidate set. This experiment does not
claim that the verifier can generate candidates, that synthetic accuracy
predicts real-song accuracy, or that any 99%/80%/70% full-corpus target has
already been measured.

## Root Objective

Establish the smallest mathematically sound and runtime-bounded foundation for
a Timing-v3 module that can eventually combine the existing shift-zero
BeatThis logits prior with independent raw-audio evidence, emit one globally
phase-continuous grid for constant/jump/ramp material, and be evaluated
honestly on all 5,050 audio identities.

## Goal Decomposition

- Subgoal 1: freeze and verify one exact definition of a linear BPM ramp and
  its continuous `B(t)` / inverse `T(x)` representation.
- Subgoal 2: freeze and verify one dependency-neutral, candidate-conditioned
  raw-audio evidence extractor/scorer using the existing 16 kHz, 10 ms,
  80-bin log-mel path.
- Subgoal 3: prove source/synthetic correctness, phase behavior, determinism,
  and the strict 600-second post-prior runtime bound before any real identity
  is opened.

## Candidate Variants

- Variant A — candidate-warped multi-band positive flux: compute onset
  novelty from the existing log-mel, sample it around each candidate's exact
  analytic beat times, contrast beat and half-beat phase, and aggregate both
  global and worst-window support.
- Variant B — low-rate waveform envelope/flux: downsample and score
  short-time energy changes directly from the waveform without a spectral
  representation.
- Variant C — generic local tempogram / predominant-local-pulse analysis:
  estimate local periodicity independently, then compare its tempo/pulse curve
  with each timing candidate.
- Variant D — another neural beat branch or external multifeature tracker:
  add a separately trained model or a library such as Essentia/madmom and use
  its output as the verifier.

## Local Verification Matrix

| Variant | Independent audio information | Ramp/phase fit | 600 s cost | Repository/dependency fit | Decision |
| --- | --- | --- | --- | --- | --- |
| A | Multi-band spectral change, independent of BeatThis logits | Directly evaluates any analytic `T(x)` and exposes local drift | `O(T + C·N)` for frames `T`, candidates `C`, beats `N` | Reuses existing mel source and NumPy; no new dependency | Select |
| B | Raw energy change | Direct phase scoring is possible | Lowest | Easy, but loses frequency-selective onset evidence and is fragile under compression/tonal energy | Reject as primary; retain as a future ablation only if A fails feature cost |
| C | Spectral novelty plus local periodicity | Good for variable tempo, but phase is indirect and window-dependent | Roughly `O(T·W)` before candidate comparison | Would require new implementation or librosa; window size adds a behavior hyperparameter | Reject for first test |
| D | Potentially rich | Depends on imported tracker | Highest and may need model warmup | Adds training/provenance or licensing/dependency risk; madmom model data are non-commercial and Essentia is AGPL | Reject |

Primary-source analogies support the ingredients but not the proposed product
claim:

- librosa documents onset-strength, autocorrelation tempogram, Fourier
  tempogram, dynamic tempo, beat tracking, and predominant local pulse:
  [onset strength](https://librosa.org/doc/main/api/generated/librosa.onset.onset_strength.html),
  [tempogram](https://librosa.org/doc/0.11.0/generated/librosa.feature.tempogram.html),
  [Fourier tempogram](https://librosa.org/doc/0.11.0/generated/librosa.feature.fourier_tempogram.html),
  [tempo](https://librosa.org/doc/0.11.0/generated/librosa.feature.tempo.html),
  [beat tracking](https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html), and
  [PLP](https://librosa.org/doc/0.11.0/generated/librosa.beat.plp.html).
- BeatThis establishes the existing neural-prior family, not an independent
  raw-audio truth source: [paper](https://arxiv.org/abs/2407.21658) and
  [repository](https://github.com/CPJKU/beat_this).
- The selected method does not import librosa, madmom, or Essentia. Those
  sources establish nearby methods and trade-offs only.

## Selected Variant

- Selected: Variant A, candidate-warped multi-band positive flux.
- Rejected: Variants B-D for the reasons in the matrix.
- Why this is the smallest useful test: the live mapper already computes or
  loads 10 ms log-mel before timing preparation, and a bounded synthetic
  candidate interface is sufficient to test whether raw evidence can
  discriminate analytic timing curves at all. Scoring candidates in their own
  continuous beat coordinates avoids a second general-purpose beat tracker,
  avoids a generic tempogram's time-window trade-off, supports ramps without
  staircase approximation, and exposes phase drift directly. The current
  BeatThis-derived source produces constant/jump proposals only; practical
  ramp candidate generation is an explicit next-card dependency, not evidence
  supplied by Exp009.

## Selection Pressure

- Primary pressure: improve candidate discrimination with evidence that does
  not reuse BeatThis logits while keeping one absolute beat axis.
- Guard pressure: no real data, weak labels, metadata, `.osu`, hit objects,
  tuning, production integration, or candidate mutation in this experiment.
- Runtime pressure: the raw-audio feature extraction plus scoring of the
  frozen candidate cap for a 600-second input must be strictly below 5.0
  seconds after process/model warmup; BeatThis prior generation is excluded
  because the objective provides it as an input prior.
- Kill pressure: kill before real data if exact ramps require phase resets,
  the raw path needs a new heavy dependency, the 600-second bound fails, or
  success needs thresholds chosen after inspecting experiment outcomes.

## Research Question

Can an isolated analytic constant/jump/time-linear-ramp curve plus a frozen
candidate-warped multi-band log-mel flux scorer (1) preserve exact phase
continuity, (2) select only curves inside the correct structural and 70 ms
tolerance-equivalence class on fixed synthetic stress cases, and (3) complete
uncached feature extraction and bounded candidate scoring for a 600-second
audio file in less than five seconds, without reading any real corpus identity
or changing the current fitter?

## Closest Analogies / Novelty Layer

- Closest analogies: onset-strength functions, phase-locked beat evaluation,
  tempograms/PLP, and continuous tempo-map integration.
- Relevant taxonomy bucket: deterministic audio signal processing plus
  structured timing representation and candidate reranking.
- Novelty layer, if any: the test establishes candidate-conditioned raw-audio
  phase warping over a frozen bounded analytic curve interface. Connecting
  that interface to real BeatThis-derived constant/jump/ramp proposals remains
  future work. It does not claim a new onset detector or a new scientific
  beat-tracking algorithm.
- Representation novelty vs engineering variation: exact wall-clock-linear
  BPM sections are a representation extension relative to the current
  constant-only v3 schema; the log-mel flux extractor is an engineering
  variation of established onset-strength methods.

## Minimal Change

After explicit acceptance, add an isolated prototype; do not modify the
current `TimingV3Grid`, fitter, inference route, cache format, or production
default.

### Analytic curve contract

The prototype owns a `PhaseContinuousTimingCurve(origin_beat,
origin_time_ms, sections)` over one global beat axis. `origin_beat` is an
integer equal to the first section's `start_beat`; `origin_time_ms` is finite
and is the curve's only independent time anchor. Sections are contiguous
half-open integer beat intervals `[start_beat, end_beat)`, with the final
endpoint query permitted for inversion/serialization checks. Beat/time queries
outside the closed curve domain fail explicitly rather than extrapolate.

Two section types are allowed:

1. `ConstantTempoSection(start_beat, end_beat, bpm)`; and
2. `LinearTimeRampSection(start_beat, end_beat, start_bpm, end_bpm)`.

For a ramp with `N = end_beat - start_beat > 0`, `q0 = start_bpm`, and
`q1 = end_bpm`, BPM is linear in elapsed wall-clock seconds, not in beat index:

```text
D = 120 N / (q0 + q1)                   # section duration in seconds
a = (q1 - q0) / D                       # BPM per second
q(s) = q0 + a s                         # 0 <= s <= D
B(s) = start_beat + (q0 s + 0.5 a s^2) / 60
```

For `u = x - start_beat`, the inverse uses the stable expression
`s = 120 u / (q0 + sqrt(q0^2 + 120 a u))`; this expression also reduces
exactly to `60 u / q0` when `a == 0`, so no data-dependent near-zero epsilon is
allowed. The selected solution is the unique `s` in `[0, D]`; an evaluation
that lands no farther than `1e-12` seconds outside an endpoint from floating
roundoff is clamped to that endpoint, and any larger excursion fails. The
radicand must be finite and nonnegative before the square root. Both endpoint
BPM values must be finite and within the existing hard
20-1000 BPM guard. `LinearTimeRampSection` requires unequal endpoint BPM;
equal endpoints must use `ConstantTempoSection`. All later section start times
are derived from the prior section's end time; no section owns an independent
time anchor.

The curve must provide:

- `time_at_beat(x)`;
- `beat_at_time(t_ms)`;
- `bpm_at_time(t_ms)`;
- exact canonical serialization and round-trip validation; and
- explicit type/class identity for constant, jump, and ramp evaluation.

Canonical serialization is exactly a JSON object with keys `version`,
`origin_beat`, `origin_time_ms`, and `sections`. `version` is exactly
`pulsefield_model.timing_v3_analytic_curve_v1`; `origin_time_ms` and every BPM
are `format(float(value), '.17g')` JSON strings; beat indices are JSON integers.
A constant-section object has exactly `type="constant"`, `start_beat`,
`end_beat`, and `bpm`. A ramp-section object has exactly
`type="linear_bpm_time"`, `start_beat`, `end_beat`, `start_bpm`, and `end_bpm`.
The ordered section list is serialized with the same canonical JSON byte rule
frozen for the design manifest below, and the curve fingerprint is the SHA-256
of those exact curve bytes.

At an interior seam, `beat_at_time` and `bpm_at_time` select the right-hand
section; at the terminal endpoint they select the final section. Adjacent
same-BPM constant sections are rejected as noncanonical rather than becoming a
spurious jump. Curve class is `ramp` if any ramp section exists, otherwise
`jump` if at least two canonical constant sections exist, otherwise
`constant`.

A jump is adjacent sections with a derivative discontinuity. A ramp is a
section with unequal endpoint BPM. Neither operation may reset cumulative beat
phase. Tempo continuity at a ramp seam is reported but is not a hard invariant;
phase/time continuity is hard.

### Raw-audio evidence contract

The scorer accepts only:

- one finite float32 `log_mel_10ms[T, 80]` produced by the frozen existing
  `MelCacheConfig(sample_rate=16000, hop_ms=10, mel_bins=80, n_fft=400,
  win_length=400, fmin=20, fmax=8000)`;
- a bounded ordered tuple of at most 64 analytic timing candidates;
- the audio duration and frozen scorer config; and
- no BeatThis values except opaque candidate identity/order. The raw score is
  returned separately and is not combined with a BeatThis prior in Exp009.

The feature calculation is frozen as follows:

1. `D[t, f] = max(log_mel[t, f] - log_mel[t-1, f], 0)` with a zero first
   frame. With the existing `center=False`, 16 kHz, 400-sample analysis window
   and 160-sample hop, mel frame `t` represents the closed-open sample interval
   `[160t, 160t+400)` and has evidence timestamp `(160t+200)/16000` seconds.
   Only frames whose entire analysis interval lies inside the unpadded input
   waveform are valid. Exact window membership below is evaluated against this
   center timestamp.
2. Aggregate four fixed mel-index bands `[0,10)`, `[10,25)`, `[25,45)`, and
   `[45,80)`. For each frame and band, take the mean of the largest
   `ceil(width / 4)` positive differences. Ties use stable index order.
3. Divide each band by its finite whole-track 95th percentile plus `1e-6`,
   clip to `[0, 4]`, and preserve all four bands. An all-zero band remains
   zero. No data-dependent feature or threshold is added after execution.
4. Before scoring, form one candidate-invariant index set. All candidates share
   the same integer beat domain. Retain exactly those integer beat indices `k`
   for which every candidate's `T(k)` and matched `T(k+0.5)` each have at least
   one valid evidence-frame center within inclusive `+-50 ms`; compare using
   float64 seconds and `abs(frame_center - event_time) <= 0.050`. Every
   candidate is scored on this identical ordered beat/control index set. For
   each retained event, take each band's maximum over that inclusive window.
5. Per event, average the largest two of the four band values. The candidate
   raw score is the mean beat support minus `0.5` times mean half-beat support,
   plus `0.25` times the 10th percentile of non-overlapping 16-beat-window
   beat-minus-half-beat contrasts. Windows start at the smallest retained
   integer beat, contain 16 consecutive retained integer indices, and discard
   a final incomplete window. Quantiles use NumPy's frozen linear method.
6. Require at least 16 common retained beats and at least one complete
   16-beat window; otherwise every candidate returns the same explicit
   unavailable reason, never a numeric best-so-far score. Candidate-specific
   denominator dropping or coverage weighting is forbidden.
7. Rank by descending finite raw score, then ascending canonical candidate
   fingerprint. No truth-derived tie-break is allowed.

The implementation may vectorize or chunk these exact operations but may not
change their mathematical result. Feature arrays are canonical float32;
accumulations, quantiles, curve math, and scores are float64. Vectorized and
scalar reference scores must differ by at most `1e-12`, and the selected
fingerprint must be identical.

### Explicit exclusions

Exp009 may not:

- inspect, decode, render, summarize, or identify any of the 5,050 real audio
  files or their caches/maps;
- run BeatThis inference or alter the BeatThis candidate set;
- edit current `timing/v3/schema.py`, `fitter.py`, local/global fitters,
  `SessionRuntime`, Hydra configs, or production routing;
- add librosa, madmom, Essentia, a model checkpoint, or any other dependency;
- tune a parameter after viewing a synthetic aggregate; or
- claim final accuracy, production readiness, or real-audio generalization.

## Files Likely to Change

Only after card acceptance:

- `src/pulsefield_model/timing/v3/analytic_curve.py`;
- `src/pulsefield_model/timing/v3/audio_evidence.py`;
- `src/pulsefield_model/timing/evaluation/exp009_synthetic.py`;
- `scripts/run_timing_v3_exp009.py`, a stdlib-only audited bootstrap;
- `tests/timing/test_timing_v3_analytic_curve.py`;
- `tests/timing/test_timing_v3_audio_evidence.py`;
- `tests/timing/test_timing_v3_exp009_synthetic.py`; and
- one immutable Exp009 result document after execution.

The accepted implementation may update `timing/v3/__init__.py` only to expose
the isolated prototype types. Any other source edit mutates this card before
execution.

## Read-Only Context Files

- `README.md`;
- `pyproject.toml` and `uv.lock`;
- `src/pulsefield_model/features/audio.py`;
- `src/pulsefield_model/features/mel.py`;
- `src/pulsefield_model/features/mel_base.py`;
- `src/pulsefield_model/timing/v3/schema.py`;
- `src/pulsefield_model/timing/v3/global_constant_jump.py`;
- `src/pulsefield_model/timing/v3/local_frontier.py`;
- `src/pulsefield_model/timing/ramp_detection.py`;
- existing Timing-v3 research cards/results/tests; and
- the primary sources linked above.

`artifacts/` is not a read-only context surface for this experiment and must
not be scanned.

## Dataset Slice

No real dataset is authorized. The fixed synthetic suite contains exactly 72
distinct 48.000-second, 16 kHz, mono float32 waveforms: 24 constant, 24
one-jump, and 24 one-ramp cases. Every truth curve has beat domain `[0,20)` and
origin time `1000 + o` ms, where origin index `j` selects
`o[j] = (0, 30, 70, 110)`.

The following six templates per class expand across all four origin indices.
Template index is `i=0..5`; each cell in the jump table is `(left BPM, right
BPM, boundary beat)`, and each ramp cell is `(start BPM, end BPM)`:

| `i` | constant BPM | jump | time-linear ramp |
| ---: | ---: | --- | --- |
| 0 | 60 | `(60, 90, 8)` | `(60, 120)` |
| 1 | 90 | `(90, 150, 10)` | `(90, 150)` |
| 2 | 120 | `(120, 180, 12)` | `(120, 180)` |
| 3 | 150 | `(180, 120, 8)` | `(240, 150)` |
| 4 | 180 | `(150, 90, 10)` | `(180, 90)` |
| 5 | 240 | `(240, 120, 12)` | `(150, 60)` |

Rows are ordered by class `constant`, `jump`, `ramp`, then `i`, then `j`.
Their ids are respectively `c{i}{j}`, `j{i}{j}`, and `r{i}{j}`. Their NumPy
`PCG64` seeds are `910000 + 4i + j`, `920000 + 4i + j`, and
`930000 + 4i + j`. Stress-profile index is `(i + 2j + p) mod 6`, with class
phase `p = 0, 1, 2` in that same class order. This expansion is the literal
72-row manifest; adding, removing, resampling, or reordering a row mutates the
card before execution.

The six stress profiles are literal:

0. `clean`: integer-beat clicks only;
1. `missing20`: omit integer beats whose absolute index satisfies `k mod 5 = 2`;
2. `offbeat`: retain integer beats and add a `0.65`-strength click at every
   half beat `k+0.5`;
3. `syncopated`: multiply odd integer beats by `0.35` and add a
   `0.85`-strength click at `k+0.5` after every odd beat;
4. `noise10`: retain integer beats and add zero-mean Gaussian noise at exactly
   10 dB signal-to-noise ratio relative to the pre-noise click-stem RMS; and
5. `tonal`: retain integer beats and add continuous 220/330/440 Hz sinusoids
   with amplitudes `0.035/0.025/0.020`.

Waveform synthesis is also literal. At 16 kHz create 768,000 zero float64
samples. Define an 800-sample click kernel

```text
K[n] = sum_j A[j] sin(2 pi f[j] n / 16000) exp(-n / (16000 tau[j]))
f   = (110, 880, 3520) Hz
A   = (0.50, 0.30, 0.20)
tau = (0.020, 0.012, 0.006) seconds
```

and divide `K` by its float64 maximum absolute value. An integer click has
base scale `0.60`, multiplied by `1.50` when `k mod 4 = 0`, then by any stress
multiplier above. Half-beat strength is relative to the unaccented `0.60`
scale. Place a click at sample `floor(16000*T(x)/1000 + 0.5)` and truncate only
at the waveform endpoint. For `noise10`, draw `standard_normal` from that
row's `numpy.random.Generator(PCG64(seed))` and set noise RMS to click-stem RMS
divided by `sqrt(10)`. After all components are summed, divide by the waveform
maximum absolute value and multiply by `0.95`, then cast once to float32. These
operations use NumPy 1.26.4 semantics. Every row is converted through the
existing `compute_log_mel_10ms`; the bulk test therefore covers
waveform-to-mel-to-score behavior but no audio codec.

Candidate construction is deterministic and truth-blind after the evaluator
builds the row. Start with the truth, apply each transformation independently
to the truth in the following order, reject invalid BPM/domain results,
deduplicate by canonical fingerprint, and finally sort the tuple by ascending
fingerprint before passing it to the scorer:

1. origin shifts in milliseconds `(-160,-110,-70,-30,30,70,110,160)`;
2. multiply every BPM/end-BPM by
   `(0.5,2.0,0.92,0.97,0.99,1.01,1.03,1.08)`;
3. for jump truth only: replace it by the duration-equivalent constant BPM
   `60*20/(truth_end_seconds-truth_start_seconds)`, swap left/right BPM at the
   original boundary, then shift only the boundary by `(-2,-1,1,2)` beats;
4. for ramp truth only: replace it by the duration-equivalent constant BPM
   `(q0+q1)/2`, replace it by constant sections `(q0 on [0,10), q1 on
   [10,20))`, reverse endpoints, create four candidates that multiply only
   `q0` by `(0.90,0.95,1.05,1.10)`, then four candidates that multiply only
   `q1` by the same factors, then multiply both deviations from mean
   `m=(q0+q1)/2` by slope factors `(0.90,1.10)`; and
5. preserve beat domain `[0,20)` and the unmodified origin for every
   transformation except item 1.

The scorer receives only canonical curves/fingerprints, never the truth flag,
row class, seed, transformation name, or pre-sort order. Every set has fewer
than 64 candidates by construction: exactly 17 for each constant row, 23 for
each jump row, and 30 for each ramp row.

The accepted design-manifest byte contract is UTF-8 JSON with
`sort_keys=True`, separators exactly `(',', ':')`, `ensure_ascii=True`, and
`allow_nan=False`; no terminal newline. Its top object is
`{"schema":"pulsefield_model.timing_v3_exp009_design_manifest_v1","rows":[...]}`.
Each row has keys `row_id`, `class`, `template_index`, `origin_index`, `seed`,
`stress_index`, `stress`, `truth_fingerprint_sha256`, and `candidates`;
candidate objects have keys `fingerprint_sha256` and `curve`. Canonical curve
JSON is the same byte rule and represents every finite float using
`format(float(value), '.17g')` as a JSON string. For the literal expansion
above under Python 3.10.20 and NumPy 1.26.4, the accepted design manifest is
533,210 bytes with SHA-256
`b62e4d464cac5008a9061efa5c78018c972af55a2a953746d821e33db240ebf1`.
A source-only test must materialize and match that exact byte length/hash,
assert 72 unique row ids/seeds and the exact per-class candidate counts, and
verify all candidate fingerprints before any score is computed. Any mismatch
mutates the card; recording a newly generated hash at runtime is not freezing
the design.

The runtime fixture is separately generated as a 600.000-second, 16 kHz,
signed-16-bit mono PCM WAV. Its truth is a `[0,1200)` 90-to-150 BPM
time-linear ramp at origin 0 ms, which has exact duration 600 seconds. It uses
the same click kernel with all integer clicks plus `0.65` half-beat clicks, a
10 dB `PCG64(909600)` noise component, and the frozen tonal bed, then the same
0.95 normalization before round-to-nearest/clipped int16 encoding. Its 64
candidates are truth plus 63 ramps indexed by `(g,h)` for `g=0..6`, `h=0..8`:
origin shift `(h-4)*20` ms, start BPM `90*(1+(g+1)*0.005)`, and end BPM
`150*(1-(g+1)*0.003)`. They are fingerprint-sorted before scoring.

Fixture generation and file writing occur before timing starts. The primary
gate is explicitly a warm-process, warm-OS-page-cache local-WAV benchmark:
each timed path starts at file open and includes pydub decode, mono/16 kHz
normalization through `load_audio_file`, uncached 10 ms log-mel computation,
feature extraction, and scoring all 64 candidates. No mel cache may exist or
be created for a timed run. A separately reported cold-process diagnostic has
no pass/fail authority.

## Baseline / Comparator

- Representation baseline: current `TimingV3Grid`, which must remain unchanged
  and must be demonstrated unable to represent a nonzero linear ramp exactly
  without multiple constant sections.
- Evidence ablation: the same candidate ranking after a deterministic
  within-row permutation of contiguous 20-frame/200 ms log-mel blocks. Take
  only the prefix of fully valid evidence frames whose length is divisible by
  20, reshape it to ordered blocks, permute block axis with
  `Generator(PCG64(row_seed + 9009)).permutation(block_count)`, and leave the
  remaining valid and padded-tail frames in place. This destroys long-range
  beat phase while preserving the marginal feature distribution and local
  frame content. It is a guard against fingerprints/order accidentally
  selecting truth, not a model comparator.
- Current-v3 regression comparator: the existing schema/global/local/fitter
  tests, run without changing their source.
- No synthetic BeatThis-prior accuracy number is a comparator; a fabricated
  prior would not establish the value of real raw-audio evidence.

## Primary Metric

### 1. Analytic representation gate

All analytic oracle cases pass:

- an equal-endpoint ramp constructor fails with an explicit instruction to use
  a constant section, while the stable inverse expression at `a == 0` equals
  the constant formula at every sampled beat/time;
- increasing and decreasing ramps match closed-form `B(t)` and `T(x)`;
- a decreasing-ramp fixture whose unconstrained quadratic has two nonnegative
  roots selects the unique root in `[0,D]`;
- constant-to-ramp, ramp-to-constant, ramp-to-ramp, and jump seams have derived
  time discontinuity at most `1e-6` ms;
- `beat_at_time(time_at_beat(x))` absolute error is at most `1e-9` beats for
  endpoints, seams, and 1,000 fixed interior samples;
- `time_at_beat(beat_at_time(t_ms))` absolute error is at most `1e-6` ms for
  endpoints, seams, and 1,000 fixed interior samples;
- exact-zero and tiny nonzero positive/negative slopes preserve their required
  class, endpoint, fingerprint, and round-trip semantics without an epsilon
  branch;
- serialization round-trip changes no canonical field and has at most
  `1e-6` ms beat-time error; and
- invalid, nonfinite, nonpositive, noncontiguous, or out-of-bound sections fail
  explicitly.

### 2. Synthetic curve-recognition gate

For evaluator-only truth comparison, define corresponding-beat residual
`r[k] = candidate.time_at_beat(k) - truth.time_at_beat(k)` in milliseconds for
every integer `k=0..20`, including the terminal endpoint. A candidate is in the
row's tolerance-equivalence class only if it satisfies all of:

- structural class equals truth (`constant`, `jump`, or `ramp`);
- `abs(median(r)) <= 70` ms and `max(abs(r)) <= 70` ms;
- maximum phase discontinuity at every candidate seam is at most `1e-6` ms;
- constant BPM relative error is at most 1%;
- jump rows have the correct direction, both section BPM errors at most 2%,
  and boundary error at most one beat; and
- ramp rows have the correct direction, both endpoint BPM errors at most 3%,
  and explicitly retain the wall-clock-linear ramp class.

A row succeeds only if the top-ranked raw-audio candidate is in that
tolerance-equivalence class and
`max_score(equivalent) - max_score(non_equivalent) > 1e-9`. Ties among
equivalent candidates, including origin shifts inside the allowed 70 ms, are
acceptable; truth fingerprint rank is reported but does not alter success.
This tests the requested tolerance boundary rather than pretending that two
perceptually accepted global offsets must be distinguishable.

The frozen positive thresholds are:

- constant: `24/24` rows succeed;
- jump: at least `22/24` rows succeed;
- ramp: at least `20/24` rows succeed; and
- all three class gates must pass; micro-average cannot rescue a failed class.

These thresholds are feasibility pressure above the owner's later real-data
targets. They are not estimates of 99%, 80%, or 70% real-song accuracy.

### 3. Evidence-destruction gate

Repeat the exact reducer after the frozen 200 ms block permutation. The
permuted result must fail each of the constant, jump, and ramp class thresholds
above, and its micro-average success rate must be at least 30 percentage points
below the unpermuted micro-average. This gate is bound into the canonical final
status; a missing or passing ablation makes Exp009 negative.

### 4. Runtime gate

The parent process generates the fixture, performs one untimed sequential read
of the entire WAV to establish the declared warm OS page-cache condition, and
then launches five sequential fresh `spawn` children. Before timing, each child
sets `OMP_NUM_THREADS=4`, `VECLIB_MAXIMUM_THREADS=4`, and
`MKL_NUM_THREADS=4`, imports dependencies, pins
`torch.set_num_threads(4)`/`torch.set_num_interop_threads(1)`, keeps the mel
layer/tensors on CPU, and warms layer construction with exactly one second of
zeros. It then runs `gc.collect()`, records its post-warm lifetime RSS high
water, and starts the timer immediately before opening the 600-second WAV.

Each child performs exactly one timed file path. Every run must be strictly
less than `5.000` seconds. Report all five raw values and nearest-rank p50/p90
(p90 is the maximum at `n=5`), stage times, pre-run RSS high water, final
lifetime peak RSS, and their difference. On the frozen macOS host,
`resource.getrusage(RUSAGE_SELF).ru_maxrss` is interpreted as bytes. Every
child's absolute lifetime peak must be at most 2 GiB and its increase over the
post-warm high water must be at most 1 GiB. Any child error or timeout is a
failed run, not a dropped denominator.

The runtime means post-prior Timing-v3 Family-B work: BeatThis model load and
inference, candidate generation, mapper control preparation, and UI/network
latency are excluded. Cold process import/layer construction is reported as a
secondary diagnostic. A later end-to-end card must budget candidate generation
and real codecs before making the owner's final five-second claim.

## Secondary Metric

- raw-score truth rank and best-equivalent-minus-best-invalid-decoy margin;
- class-stratified median/p90/max beat-time residual;
- constant BPM, jump-boundary, ramp-endpoint, and slope error distributions;
- unavailable-score count and reason;
- beat and half-beat support, worst 16-beat-window contrast, and coverage;
- result under the deterministic 200 ms block-permutation evidence ablation;
- feature/scoring wall time separately from decode/resample/mel wall time;
- cold vs warm runtime and added/lifetime RSS;
- deterministic replay: canonical result SHA-256 identical across three
  untimed repeats on the same platform; and
- current-v3 test count/runtime relative to the pre-card checkpoint.

## Verify Command / Evaluation Procedure

After explicit acceptance and implementation:

```bash
uv run --extra mps --group dev pytest -q \
  tests/timing/test_timing_v3_analytic_curve.py \
  tests/timing/test_timing_v3_audio_evidence.py \
  tests/timing/test_timing_v3_exp009_synthetic.py

.venv/bin/python -I -S -B \
  scripts/run_timing_v3_exp009.py \
  --output-dir /private/tmp/timing-v3-exp009-run

uv run --extra mps --group dev pytest -q \
  tests/timing/test_timing_v3_schema.py \
  tests/timing/test_timing_v3_global_constant_jump.py \
  tests/timing/test_timing_v3_local_frontier.py \
  tests/timing/test_timing_v3_fitter.py
```

The bootstrap and runner have no dataset-path, artifact-root, audio-path, seed,
threshold, or feature-parameter option. They generate only the source-frozen
synthetic manifest and require one absent-or-empty explicit output directory
whose resolved path is exactly `/private/tmp/timing-v3-exp009-run`; result,
pre-run identity lock, fixture, and manifest files all live below it. The one
filesystem-control exception is the cross-run advisory lock path specified
below; it is not a result artifact.

The experiment runner intentionally invokes the already-resolved `.venv`
interpreter directly; `uv run` is not in the audited execution boundary and
could perform environment synchronization/cache I/O before the Python guard is
installed. The preceding test command remains responsible for proving the
locked environment is present.

`scripts/run_timing_v3_exp009.py` imports only Python stdlib before installing
guards. It requires `sys.dont_write_bytecode is True`, redundantly sets it to
`True`, sets the frozen thread
environment, installs the audit hook and checked-filesystem wrappers, then
uses the absolute lexical `Path(sys.executable)` without resolving its expected
uv-managed interpreter symlink, requires
`executable.parent.parent / 'pyvenv.cfg'`, parses only the stdlib-defined UTF-8
`key = value` configuration after guard installation, requires that lexical
environment root to equal the resolved repository path joined with `.venv`,
and records/validates the executable symlink target separately. On the frozen Python 3.10
host it inserts exactly
`.venv/lib/python3.10/site-packages` after validating that directory and the
expected NumPy/Torch/nnAudio distributions; it does not use
`sysconfig.get_paths()` because `-S` leaves Python 3.10's `sys.prefix` at the
base uv interpreter. It then inserts the resolved repository `src/` path and
imports the experiment entry function, without importing `site` or processing
`.pth` files. Any package that depends on executable code from a `.pth` file is
an explicit bootstrap failure, not grounds to process it.

`python -I -S -B` prevents normal/user-site initialization, ambient
`PYTHONPATH` imports. The environment/source paths and expected repository root
are literal bootstrap constants bound by the accepted source hash. A
bootstrap-order test launches it in a fresh interpreter and proves
`sys.flags.no_site == 1`, `site` is absent, no non-stdlib `.pth` was processed,
and no `pulsefield_model`, NumPy, Torch, or nnAudio module is present in
`sys.modules` when guards become active. It then proves the guarded inserted
path is exactly the repository `.venv` site-packages and those three packages
become discoverable only after insertion. Every spawned runtime child uses the
same `-I -S -B` stdlib-only guard initializer before dependency import; it does
not import through the eager `pulsefield_model.timing.__init__` first.

The audit hook records `open`, directory-list, import, and subprocess events.
The stdlib-only guard additionally wraps `os.stat`, `os.lstat`, `os.scandir`,
`os.listdir`, `pathlib.Path.stat`, `exists`, `is_file`, `is_dir`, `glob`, and
`rglob` before project import, validating paths before delegating. Tests prove
that even a nonexistent `artifacts/cache/foo.npy` `exists()`/`stat()` probe
fails. The combined guard fails immediately on: any path below the repository
`artifacts/` or `dataset/` directories; any `.osu`; any audio suffix
`.wav/.mp3/.ogg/.flac/.m4a/.aac` outside the runner-created
output directory; any `.npy/.npz` outside the resolved Python environment or
that output directory; any write outside that output directory except an exact
`os.open(O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW, 0o600)` of
`/private/tmp/timing-v3-exp009-benchmark.lock` followed only by `fstat`,
nonblocking exclusive `flock`, and close; or an unapproved subprocess. Before
open, guarded `lstat` must either report absence or a nonsymlink regular file.
After open, `fstat` must prove a regular file owned by `os.getuid()`, link count
one, and permission bits exactly `0o600`; otherwise the descriptor closes and
preflight fails. `O_NOFOLLOW` plus descriptor validation owns the race-safe
decision. The lock is never truncated, content-written, chmodded, renamed, or
removed by the runner. Imported repository source, accepted card/tests,
stdlib, and environment-library reads remain allowed and logged.

The only subprocess allowlist entries are exactly `/usr/bin/pmset -g batt` and
`/usr/bin/pmset -g custom` for runtime preflight, the resolved pydub converter
with `-version` for provenance, and, if pydub actually spawns a decoder for the
PCM WAV, a validated command whose every input/output path lies in the output
directory. The preflight parses and records the exact `pmset` outputs, requires
an AC-power source and the active AC profile's `lowpowermode` to equal zero,
acquires a nonblocking exclusive `fcntl.flock` at
`/private/tmp/timing-v3-exp009-benchmark.lock`, and records
`os.getloadavg()`; failure to prove either power condition or acquire the lock
aborts before timing. If WAV decode uses no subprocess, the decode converter
command is canonical `null`. The canonical sorted access-event manifest and
its SHA-256 are published next to the result; zero denied events is a hard
gate. The hook cannot observe the converter's internal system calls, so
executable identity and full argument vectors are separately validated and
recorded.

Before execution, the runner writes a pre-run lock containing exact SHA-256
values for the accepted card, all Exp009 source/tests, the materialized
row/candidate manifest, the current read-only v3 comparator files, and the
Python/NumPy/Torch/nnAudio versions. The result must bind the identical lock;
any byte drift aborts before a score is computed.

## Guard Check

- No source outside the explicitly allowed files changes.
- No new direct or transitive dependency is added.
- No `artifacts/` path or real audio/cache/map identity is read.
- The access-event audit has zero denied events and its manifest/hash are
  present in the canonical result.
- Fresh-process tests prove guard installation precedes project/scientific
  imports and deny nonexistent `artifacts/`/`dataset/` stat and existence
  probes, cache creation, `.osu`, undeclared audio, and writes outside the
  output directory.
- Exact-command bootstrap tests require `sys.dont_write_bytecode is True`;
  runtime preflight records and gates AC power, Low Power Mode, and exclusive
  lock acquisition through the single outside-write exception, and reports
  the system load averages. A guard test proves no other operation or outside
  path is permitted by that exception and that a preexisting symlink,
  non-regular file, wrong owner/mode/link count, or target below
  `artifacts/`/`dataset/` fails before timing.
- Current Timing-v3 schema/fitter/source hashes remain unchanged.
- Existing focused Timing-v3 tests pass unchanged.
- Synthetic manifest, decoy construction, thresholds, reducers, and runtime
  boundary are final before the first aggregate is viewed.
- The implementation never reads class truth during scoring or tie-breaking.
- All failures publish an explicit status; no NaN, unavailable row, timeout,
  or exception becomes a best-so-far success.
- Every candidate is phase-continuous by construction and revalidated before
  scoring and after canonical round-trip.
- Exact click/frame alignment tests cover frame-center timestamps, padded-tail
  exclusion, events exactly at `+-50 ms`, and events one float64 ULP outside.

## Qualitative Check

Inspect only source, formulas, fixed synthetic manifests, canonical numeric
diagnostics, and aggregate tables. Plotting a few synthetic truth/decoy onset
curves is allowed only after the primary result is frozen and may explain but
not change the outcome. No real audio listening, waveform inspection, `.osu`
inspection, identity browsing, or hand correction is allowed.

## Positive Signal

- Every analytic representation gate passes.
- All three class-specific synthetic recognition gates pass without a changed
  parameter or excluded row.
- Every 600-second timed run is below five seconds and the memory guard passes.
- The block-permutation ablation fails every class gate and its micro-average
  success rate is at least 30 percentage points below the unpermuted result,
  showing that source order/fingerprints are not carrying the result.
- Deterministic replay and all existing focused v3 regression tests pass.

A positive result supports one next no-data card to integrate analytic ramps
and raw-score diagnostics into a bounded BeatThis candidate generator. It does
not authorize real 5,050-song execution.

## Negative Signal

- Any analytic, class-specific recognition, runtime, memory, deterministic, or
  regression gate fails.
- The selected method cannot separate tolerance-equivalent curves from invalid
  phase/slope decoys on the fixed synthetic stresses.
- Passing needs a new dependency, threshold relaxation, manifest change,
  post-result tuning, or real-data inspection.
- The five-second result depends on a mel cache hit or excludes file decode,
  resample, mel extraction, or candidate scoring.

## Kill Criteria

Kill Exp009 and do not open real data if:

- the final accepted card/source/test hashes are absent or differ at runtime;
- any real identity, cache, audio, `.osu`, label, metric, or `artifacts/` file
  is opened;
- the ramp definition changes from wall-clock-linear BPM after execution;
- a candidate owns an independent seam anchor or a seam exceeds `1e-6` ms;
- any class or evidence-destruction primary gate fails or a row is silently
  dropped;
- any one of the five warm timed runs reaches `5.000` seconds, absolute
  lifetime RSS exceeds 2 GiB, added RSS exceeds 1 GiB, or the full Exp009
  runner exceeds 15 minutes;
- a dependency or production/current-v3 source edit is required;
- a score/threshold/reducer/fixture is changed after aggregate exposure; or
- a later stage is proposed to rescue a negative/ambiguous Exp009 result.

After a kill, preserve the immutable result and choose at most one new no-data
hypothesis: waveform-envelope fallback for feature-cost failure, or a local
PLP/tempogram verifier for discrimination failure. Do not try both on real
data.

## Expected Failure Modes

- log-mel top-quartile flux may overreact to broadband noise or compression;
- half-beat contrast may punish genres with strong offbeats or double-kick;
- half/double tempo aliases may remain indistinguishable from raw onset energy;
- global percentile normalization may underweight quiet intros/outros;
- a 16-beat worst-window term may be unstable on short or sparse passages;
- time-linear BPM may not match beat-linear editor ramps or expressive rubato;
- 10 ms frames quantize raw evidence even though the analytic curve is
  continuous;
- pydub/ffmpeg decode and nnAudio mel creation may dominate the five-second
  budget;
- Python audit hooks alone do not cover every metadata syscall, so correctness
  also depends on the pre-import filesystem wrappers and source/test review;
- Apple-Silicon warm performance may not transfer to Linux/CUDA or a cold
  service process; and
- synthetic percussion may be much easier than full mixes.

## Confounders

- The real 5,050-song snapshot has no exact source-owned truth for the owner's
  99% constant, 80% jump, or 70% ramp claims. Existing `.osu` redlines and hit
  objects are correlated weak evidence, and the 48 permissive ramp candidates
  are explicitly not confirmed ramp truth.
- “Recognition” is not yet an established corpus metric. This card freezes a
  synthetic operational definition only; a later truth-contract card must
  define exact-recording labels, class denominators, abstention/fallback, and
  confidence intervals before real percentages can be claimed.
- The current corpus snapshot contains tracks longer than 600 seconds. Final
  full5050 execution must process/report all identities and its long stratum
  separately. Only the owner's five-second SLA is presently scoped to duration
  `<=600 s`; class-accuracy denominators remain intentionally unset until a
  later exact-truth card receives owner acceptance.
- The current live path decodes audio separately for mapper mel and BeatThis.
  Exp009 assumes the BeatThis logits prior already exists and measures one
  uncached raw-audio decode/mel/scoring path; it does not establish total live
  `SessionRuntime.prepare_audio()` latency.
- A WAV runtime fixture does not cover corpus codec tails. Real codec and
  duration strata must be frozen later.
- Synthetic source frequencies/clicks can align unusually well with mel bands.
  The ablation detects gross leakage, not ecological validity.

## Expected Runtime / Runtime Budget

- Source implementation plus targeted unit tests: no experiment claim; review
  before execution.
- Analytic and 72-row synthetic evaluation: target under five minutes.
- One parent page-cache priming read plus five fresh-child 600-second timed
  runs: target under one minute total; every measured child remains
  independently subject to the strict time and memory kills.
- Entire runner: hard stop at 15 minutes.
- Existing focused Timing-v3 regression suite: historical checkpoint was 162
  passing tests in about 118 seconds; any failure is a hard regression.

## Result Interpretation Plan

- Positive result would suggest: exact ramp math and candidate-conditioned
  raw-audio phase evidence are feasible enough to integrate, still disabled
  and without real-data claims.
- Negative result would suggest: preserve current v2/constant-jump v3 and test
  exactly one failure-matched alternative under a new no-data card.
- Ambiguous result would require: classify as negative for promotion; write a
  narrower source-only card only if the ambiguity has one identifiable cause
  not requiring real data.
- Human owner decides: whether to accept Exp009 for isolated code/synthetic
  execution; after a positive result, whether to accept the next integration
  card; later, whether the exact-truth annotation cost and production trade-off
  justify fresh/full-corpus evaluation.
- Next-loop action if positive: draft one Experiment 010 card for bounded
  BeatThis-candidate plus raw-score integration and the exact output/metric
  contract, still on source/synthetic inputs.
- Next-loop action if negative: retain current production v2; select only the
  failure-matched waveform-envelope or PLP alternative.
- Next-loop action if ambiguous: stop and preserve the result; do not open the
  5,050 identities.

## Result Log Template

- Experiment: Timing v3 Experiment 009
- Date:
- Accepted card SHA-256:
- Source/test closure SHA-256:
- Run id / canonical result SHA-256:
- Platform / OS / Python / NumPy / SciPy / Torch / nnAudio / pydub versions:
- AC-power proof / active Low Power Mode value / exact `pmset` outputs:
- Exclusive benchmark lock acquired / system load averages:
- ffmpeg executable, version, and full converter command:
- Torch intra-op/inter-op threads and thread environment:
- mel tensor/layer device:
- Dataset slice: fixed 72 48-second synthetic rows plus one 600-second WAV
  runtime row
- Baseline / comparator:
- Runtime five-run vector / p50 / p90 / max:
- Decode / mel / feature / score stage times:
- Added / lifetime RSS:
- Analytic oracle result:
- Constant successes / denominator:
- Jump successes / denominator:
- Ramp successes / denominator:
- Phase p50 / p90 / p95 / max by class:
- Tempo/boundary/slope errors by class:
- Unavailable count / reasons:
- Block-permutation class counts / micro delta / final gate:
- Access-event manifest SHA-256 / denied event count:
- Pre-run source/card/test/manifest lock SHA-256:
- Deterministic replay hashes:
- Verify command / result:
- Guard command / result:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: no; requires explicit owner
  acceptance of the final reviewed card bytes.
- Real-data execution allowed: no
- Closed loop complete: no
- Remaining ambiguity: independent review findings, final card hash, explicit
  owner acceptance, and then implementation/synthetic evidence.

## Next-Loop Action

- If positive: draft Experiment 010 for isolated candidate integration and
  exact metric/truth interfaces; do not open real data automatically.
- If negative: preserve the result and draft at most one failure-matched
  no-data alternative.
- If ambiguous: stop; ambiguity is not a pass.

## Novelty Notes

- Closest analogies: spectral-flux onset strength, phase-locked beat scoring,
  PLP/tempogram analysis, and analytic integration of tempo curves.
- Novelty layer, if any: candidate-warping scores a frozen analytic candidate
  interface with an independent deterministic audio feature on the same
  continuous phase axis; neural-prior proposal integration is not tested here.
- Representation novelty vs engineering variation: time-linear BPM sections
  extend the current schema; the onset feature itself is established signal
  processing assembled for a strict latency and continuity contract.
