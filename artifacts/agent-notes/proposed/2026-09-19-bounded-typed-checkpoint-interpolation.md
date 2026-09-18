# Agent Note: Does averaging adjacent O1 checkpoints retain LN organization with less burden?

Note ID: 2026-09-19-bounded-typed-checkpoint-interpolation
Status: proposed
Kind: research
Created: 2026-09-19
Updated: 2026-09-19
Product revision: 15d27db0b0723d7f606b1429c901d26c1f61e5ac
Scope: Inference-only interpolation between the paired O1 2M and 2.25M checkpoints
Related: 2026-09-18-bounded-typed-time-three-arm-comparison

## Question and evidence

The availability experiment found a much larger change from another250k training
onsets than from the proposed endpoint features. Original O1 changes from37.037%
LN heads at2M to10.262% at2.25M. Rapid attack burden falls, but the fixed LN-rich
core loses its independent coordination. Complete-suffix pooled NLL changes only
from1.906802358 to1.913932344. This leaves optimization/checkpoint sensitivity and
free-history mode changes as live concerns; no unique cause has been established.

The parameter midpoint is a cheap test of whether two behaviorally different
checkpoints share an interior with a useful tradeoff. A positive result motivates
a prospective averaging/LR experiment. A negative result stops this cheap branch;
it does not rule out a better-trained model, EMA, SWA or a different LR schedule.

## Experiment Card: bounded-typed-o1-interpolation-v1

### Identity and authority

- Revision:1. Owning Note: this Note ID. Accepted revision:none.
- The standing user instruction authorizes exploratory implementation and runs.
  This proposed Note does not constitute lifecycle acceptance or model adoption.

### Analogue and intervention

[Izmailov et al., Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
studies trajectory averaging under constant/cyclic learning rates. That supplies
the nearby mechanism, not evidence of mania generation quality. This probe uses
only two points from an AdamW trajectory, not a complete SWA/EMA training scheme.
It cannot show that averaging would stabilize all intervening checkpoints.

Set theta(alpha)=(1-alpha)*theta(A)+alpha*theta(B), with alpha=.25,.5,.75 fixed
before evaluation. Alpha=.5 is primary; .25/.75 describe the curve and cannot
replace a failed primary endpoint as an independently successful experiment.
Average all124 float32 state tensors in float64 and cast back to float32.
The model has no BatchNorm; its two integer joint-index buffers are nonpersistent
and rebuilt from the configuration. Do not average optimizer, RNG or raw caches.
Store distinctly formatted inference-only model files, without resumable training
metadata. Preserve both parents. No training, support, features or sampling change.

### Fixed comparison

All paths below are under `artifacts/bounded-typed-continuation/`.

- Clean execution source: the Product revision above. Both parents are loadable
  at that exact source with `endpoint_availability=none` and identical model config.
- A: `availability-20260919-v1/p0-init/checkpoint.pt`, SHA
  `83b67714e4eaf36300616e3250ea3883c6363bcb51543bac85980c4783cfdb4f`.
  These are the exact2M O1 weights; source-transition verification SHA
  `895b8cee87a2ee0d31e38be2b374313663b1d7e9b43930c1f68be9e90633885d`
  reproduces original complete native rows and NLL sums on three fixed cases.
- B: `availability-20260919-v1/p0-2250k/checkpoint.pt`, SHA
  `a4769bdc1498e178e24dd40050612ffcf67e0076f6879b61ac086a9964686f95`.
- Same model initialization171 and identical training-draw prefix. B has329
  additional updates; this probe does not retrain or estimate a seed distribution.
- Shared24 VAL development groups and seeds17/19/23 from
  `corpus-20260918-v1/screen-conditions.json`, SHA
  `f0ead07f41111c0413cae9b3f27b3a20c2abbedc2efea1b94f5c2b2707b808eb`.
- Parent aggregate evidence:
  `corpus-20260918-v1/development-readout-2000k.json`, SHA
  `262979f1deb1069f0ff9b401f5c2d83b55b79f0de4051b770b57d6ac64e828f3`;
  `availability-20260919-v1/development-readout-2250k.json`, SHA
  `d7502657f151ba72c32e6f09fb3455a2062becc472353078605e55b77ee8be77`.
- Primary group-balanced below40ms pair rate per1000 suffix heads:
  A2.593284659, B0.26781852. A has498 pairs,439 with one free lane, max run8;
  B has53,1 and3 respectively. Aggregate LN fractions are37.037% and10.262%.
  Existing full-suffix scores and generated rows are reused as fixed parents.

### Evidence and decision rule

For midpoint alpha=.5 to justify a prospective stabilization experiment:

1. Primary group-balanced rapid-pair rate is at most half of A, and its paired
   difference to A has a95% group-bootstrap interval wholly below zero. Average
   seeds within source group, resample24 groups10000 times, bootstrap seed672.
2. Pooled complete-suffix NLL is at most1.926802358 (better parent +.02).
3. Aggregate generated LN share is at least18.5185% (half of A). This is a gross
   mode-loss diagnostic, not a quality label or a per-source style requirement.
4. All216 interior-model generations pass mechanics and export/reparse, and
   selected raw-cache recoveries preserve results. The midpoint fixed LN-rich
   core source0447fb187bc3, seed17, retains independent LN coordination under the
   Foundation. Inspect sourcee6b273f7877d dense core too, without requiring LN.

Report ordinary/stress strata, durations, chord sizes, one-free-lane rapid pairs,
run lengths and every alpha. Burden reduction through lost LN organization fails
the combined gate. A model can miss the numeric gate while revealing a useful
curve; report that as descriptive REFINE, not a selected successful method.
No human preference/win-rate or overall playable-quality claim follows from this
diagnostic screen. Wider contexts and independent groups remain future work.

### Procedure and bounds

Fresh root: `artifacts/bounded-typed-continuation/stability-20260919-v1/`.
Pin drivers before execution. `derive_models.py` verifies parents/configuration,
constructs the three inference-only files and checks exact formula/state loading.
`development_screen.py` uses the existing native complete-suffix scoring,
generation, raw recovery, exporter and reparser at the frozen source. Driver
digests and derived-model digests are recorded before the screen starts.

Commands use `uv run --offline --python 3.10 --extra mps --group dev python`
followed by the named script. Python3.10.20, Torch2.11.0, NumPy1.26.4, Apple M5,
24GiB, CPU one thread. Derivation at most three minutes; screen at most one hour;
6GiB footprint/RSS,2GiB minimum available, at most128MiB swap growth,2GiB outputs
and1GiB free disk reserve. No downloads. Stop on a digest/configuration mismatch,
nonfinite parameter/probability, resource guard or mechanical failure. Fresh
destinations only; failed outputs are preserved, with any rerun explicitly named.

The R1 second-initialization CPU training may overlap this one-thread screen.
Record concurrency and do not infer causal throughput improvements. Same sampling
seeds provide pairing, not identical random choices after policies diverge.
All24 groups were repeatedly used in development, so interpolating here can
motivate a prospective experiment but cannot establish independent generalization.

## Next lifecycle condition

Append the exact run evidence and scoped semantic judgments. Keep this Note
proposed. The current overall goal needs sustained organization and playable
quality beyond this checkpoint diagnostic.

## Pinned derivation driver

`stability-20260919-v1/derive_models.py` SHA
`cbb85bfa74b91725fdfed706753115fb14adc77587b9d3fb35a7cc2436dae20b`.
It verifies exact alpha0/1 reconstruction, finite matching parameter tensors,
strict model loading and inference-only serialization before producing a manifest.

## Derived models and pinned native screen

Derivation completed in0.303s with all checks passing. `derivation.json` SHA
`dbce665415308ee1965f02beb1ba856887ba11acb8eade7ebc02ad31e6949fa6`.
All three models have2355835 parameters. Model SHAs for alpha.25/.5/.75:
`47290ebd7f89d1c7a749d9d42ced5b593a63bd5296854961c12e703a89d6d8b4`,
`8c6e51bb6815fab0581fa2a5a5d62d2b4274763a494595581aa310e14d77c3ae`,
`b68bc5030cb5d9fbba14993d56f9b8bc3b466ad3ed03cff5b07e48e17de8b260`.

`development_screen.py` SHA
`02c3073d035bf29354fee2b9c6e29ce1e7e5ada0baa4c4bd339448931c3f4104`.
It derives from the pinned availability screen
`b3a456635e34c444bdc36b4105bd97744c1a2bb8313cb105f320f4d880108a1a`,
changing only model provenance/loading, variant names and fresh output paths.
Scoring, native rollout, raw-state recovery and export/reparse procedures are
unchanged. R1 initialization172 training is already active on a separate CPU
thread when this screen starts; both retain their resource guards.

## Pinned analysis and inspection

`readout.py` SHA
`e1becc82254144700ea22b3c01adfb3abe4472cd1b0dd72345c46905c574a81e`
reuses the original parent outputs, independently recounts rapid pairs and
one-free-lane cases from complete rows, pools LN durations from matched objects,
and reports ordinary/stress/all strata with the declared group bootstrap.
`render_midpoint.py` SHA
`874e3155dd9c96fe165a5b83bfb576348cc9b62677c90987a9dd30f92a9063cd`
renders all context pages and paginates the complete action records for the two
fixed midpoint cores. These scripts do not alter models or select new cases.

## Result: better interior fit, incomplete generation tradeoff

The native screen completes216 generations and72 complete-suffix scores in
748.645s. All mechanics/export/reparse checks and three selected exact raw-cache
recoveries pass. Peak physical footprint233866200 bytes; swap growth zero. The
declared concurrent R1 training was active, so elapsed time is not a standalone
throughput comparison. No product source changes or checkpoint overwrites occurred.
Raw readout SHA
`d925bf1b58658f2911f8c12be2add5f7e6972b9507b570c76eb127dadde2693d`;
paired summary `stability-20260919-v1/readout.json` SHA
`273e17edfda7074681f0a02b02323c992be4c447331deff301e7ee68a68c954a`.

| Weight on2.25M parent | Pooled suffix NLL | Group mean rapid pairs/1000 heads | LN head share | Rapid pairs / one-free pairs | Longest rapid run |
| --- | --- | --- | --- | --- | --- |
| 0,2M parent | 1.906802 | 2.593285 | 37.037% | 498 /439 | 8 |
| .25 | 1.874042 | .753338 | 21.631% | 152 /129 | 5 |
| .5, primary | 1.863347 | .431977 | 14.000% | 90 /47 | 6 |
| .75 | 1.876252 | .168061 | 11.864% | 33 /9 | 2 |
| 1,2.25M parent | 1.913932 | .267819 | 10.262% | 53 /1 | 3 |

The midpoint primary rate difference to A is-2.161308,95% paired group-bootstrap
interval[-6.024613,-.085535]. The rapid-reduction and NLL gates pass. The aggregate
LN-share gate fails:14.000% is below18.5185%. Its remaining47 one-free-lane rapid
pairs and longest run6 also prevent interpreting the lower mean as elimination
of local occupancy burden. Alpha.25 has a more favorable aggregate LN/burden
tradeoff, but it was not the primary contrast and its semantic outcomes are
unreviewed. Do not replace the failed primary combined gate post hoc.

### Direct semantic inspection

All16 midpoint context images were viewed. Manifest SHA
`92e3111b4585ca6593505a8c01066008ddeb4e356cb164cff7a59068cdff06f4`;
`inspection-midpoint-v1/review.json` SHA
`12628bcd45fd37d18261ac83442f9ca82b647c7e9d4be7b9b1b259fba1299205`.
The Foundation, cached human packet and eight canonical human workflow documents
retain their pinned hashes. The previously viewed High LN positive, negative and
localized-supporting comparisons remain the calibration references.

The LN-rich core has33 LN heads and227 taps, with no entering hold. Its17305–20980ms
island contains independent starts/releases; for example, three holds begun19030
end19180/19330/19405, and the next column2 hold spans19330–19855 while the other
roles continue and restart. Later at20080, column0 remains held while columns1/2
begin holds and column3 taps; column1 survives a subsequent column2 tap. This is
LN coordination present/supporting, High. The longer TAP-led surroundings and
isolated later short holds do not make the whole core prominently coordinated.
The dense core has443 taps and no LN or entering hold: coordination absent, High.
Other tags remain unreviewed. The scoped semantic retention gate passes, although
the collection-level numeric coverage guard fails. No whole-output quality or
human-preference conclusion is claimed.

### Evaluation and decision

Evaluation: REFINE. The predeclared combined midpoint gate is not met. The three
interior points have lower teacher-forced NLL than either parent, giving a concrete
reason to investigate prospective trajectory averaging or LR policy after the R1
replication and broader organization review. This is a source-bounded diagnostic,
not proof that optimization is the unique cause or that SWA/EMA will preserve
generation modes. Averaging can improve fit while substantially changing free
generation; mode coverage and sustained organization must remain separate guards.

`curve-figure-v2/curve.png` and SVG show the five fixed parameter locations, fit,
group-balanced burden and LN share. The midpoint's localized coordination is
retained, so describing this result as complete LN disappearance would be wrong.
The current models remain experimental inference-only artifacts. No deployment,
promotion, additional alpha tuning, remote push or Note lifecycle transition is
authorized by this result. The overall playable-generation goal remains active.
