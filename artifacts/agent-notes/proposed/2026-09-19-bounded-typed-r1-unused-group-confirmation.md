# Agent Note: Does the frozen R1 candidate organize new development groups?

Note ID: 2026-09-19-bounded-typed-r1-unused-group-confirmation
Status: proposed
Kind: research
Created: 2026-09-19
Updated: 2026-09-19
Product revision: 15d27db0b0723d7f606b1429c901d26c1f61e5ac
Scope: Fixed-checkpoint R1 confirmation outside the bounded comparison's development groups
Related: 2026-09-19-bounded-typed-r1-second-initialization

## Question and current evidence

At 2M source-onset exposures, two R1 initializations pass the declared numerical
and fixed-core semantic checks. The preselected ordinary comparison inspected
four local and two overlapping wider scopes: five descriptive machine ties and
one limited preference for initialization 172. This establishes plausible
organization in those inspected outputs, but repeated use of the same 24 groups
can conceal development-set dependence. The next useful discriminator is a fixed
evaluation on additional groups, without changing training or decoding.

The closest analogue is the existing paired 24-group native screen and its
Foundation-calibrated inspection. This is an evaluation extension, with no claim
of a new representation, objective or model architecture. A positive result
supports a practical R1 inference handoff. A negative result must retain the
failures and narrow the quality claim before more training is selected.

## Experiment Card: bounded-typed-r1-unused-groups-v1

### Identity and authority

- Revision: 1. Owning Note: this Note ID. Accepted revision: none.
- Exploratory implementation and execution use the standing user instruction to
  optimize and experiment autonomously toward playable continuation. This Note
  remains proposed; execution does not imply lifecycle acceptance or adoption.

### Frozen candidates and baseline

Primary candidate: R1 initialization 172, 2M exposures, checkpoint
`r1-initialization-20260919-v1/train-2000k/exposure-2000000.pt`, SHA
`9be0adc00cea85058b59f0a2446884d43fc40cd412188ee123e866df6369d83a`.
Comparator: R1 initialization 171, 2M exposures,
`corpus-20260918-v1/r1-cpu-2000k/checkpoint.pt`, SHA
`302fae523cf1fe36afff463fe28f20739a2eb40cab5064d0941a588d8f70c564`.
Both paths are under the implementation worktree's
`artifacts/bounded-typed-continuation/` owner.

Use the clean Product revision above. The comparator's original source is
`1693d62ffaca04b2a6127d8e3a72d1988adf441f`; the current source is behavior-neutral
for R1 with availability `none`, as verified by the exact preflight SHA
`5803d32717be8e3b0dbc45f4b50555e3978a13ac41756c2eb7d8d6023405c164`.
No parameters, optimizer state, task support, temperature or sampling policy change.

On the existing 24 groups, initialization 172/171 has group macro suffix NLL
1.965768505/1.987622001 and group-balanced below-40-ms same-lane adjacent attack
pairs per 1000 suffix heads .338994232/.175522125. Both maximum attack runs are
two. Collection LN shares are 51.3234081%/27.6505147%. These are development
baselines, not numerical targets for a different source population. The prior
paired NLL difference interval [-.034605267,-.009672152] is conditional on the
two fixed initializations and reused groups, not a seed-population interval.

### Selection and intervention

The only change is the evaluation group set. Use the same pinned catalog,
split manifest, lossless source cache, eligibility and source descriptors as
`corpus-20260918-v1/prepare_screen.py`. Catalog SHA
`e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28`;
split SHA `15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.
Restrict to VAL with at least 128 post-seed onsets and 16 post-seed seconds.
Exclude every group in the prior conditions manifest, SHA
`f0ead07f41111c0413cae9b3f27b3a20c2abbedc2efea1b94f5c2b2707b808eb`.

Selection seed 741 chooses 12 groups uniformly without replacement, then one
eligible chart uniformly within each group. Select two additional distinct
remaining groups per stress descriptor, in order: LN fraction, number of onset
rows with different LN endpoints, maximum onsets per second, chord fraction,
largest onset gap, and density transition. Rank descending, breaking ties by
source SHA. Preserve group exclusion across ordinary and stress selections.
The result contains 24 distinct groups; freeze its manifest digest before model
scoring or generation. No generated outcome influences selection.

Draw each 16s core uniformly in post-seed physical time using a source-specific
SHA-derived RNG with prefix `scope-741:`. Derive its surrounding 64s scope by
the existing clamped start-minus-24s rule when the post-seed chart spans 64s.
These groups are unused by the bounded comparison's development screen and its
follow-ons. They are not asserted to be untouched by every earlier Pulsefield
research activity. TEST payload remains unread.

### Measurements and qualitative decision

Score both candidates on every complete suffix in 128-onset chunks. Include all
row and release decisions and normalize by actual source onsets. Generate each
suffix natively with seeds 17, 19 and 23: 48 scores and 144 generations. Do not
rank checkpoints or filter outputs after generation.

For the primary candidate, require all mechanics, export/reparse and selected
exact recoveries to pass. Its group-balanced rapid-pair rate must be at most .8
per 1000 suffix heads in ordinary and stress strata separately, with no generated
rapid run longer than four. This retains the existing engineering screen; it is
not a universal playability or difficulty definition. Its macro suffix NLL must
be no more than comparator +.05 on the same groups. Report pooled and group
statistics, each stress category, maximum runs and LN/TAP/chord distributions.
Use 10000 paired group-bootstrap resamples, seed 742, for NLL and rapid-rate
differences, reporting ordinary and stress strata separately. Do not claim
initialization-population uncertainty or turn these intervals into a human
preference test. Different LN shares alone do not fail quality: the weights are
unchanged and the evaluated source distribution differs from the prior screen.

Preselect qualitative pairs at ordinary indices 0/3/6/9, the first LN-rich source
(index 12), and the first dense source (index 16), all at their 16s cores and
generation seed 17. Also inspect 64s contexts for the first two ordinary indices
in manifest order that have a 64s scope. Require two such sources; stop and
record an insufficient-scope result if selection cannot provide them, without
silently replacing the selection rule. Render complete entry/exit contexts and
paginated action facts using the same canonical renderer. Mask initialization
labels separately for each pair and save all judgments before opening the map.

Use Foundation identity
`15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97`
and the previously verified human LN/Stream/Jack/Trill comparators. Record actual
column, type and release relationships; externally supplied timing gaps and
density changes are not learned composition. Each candidate/scope receives
plausible, unresolved or inadequate structural status, plus any confidently
assessed tags. Allow preference tie or both inadequate. Uninspected tags remain
unreviewed, and short LNs remain admissible.

A practical handoff requires the numerical guards, no inadequate primary scope,
plausible organization in both wider primary scopes, and concrete moving-flow
and independent LN organization somewhere among the preselected primary scopes.
These need not coexist everywhere. Any unresolved scope or uncertain pattern
evidence leaves the corresponding quality claim open. Any failure directs a
focused diagnosis before changing the model. Passing this limited confirmation
supports a bounded candidate-quality claim, not universal playability, human
preference or final adoption.

### Execution and resource bounds

Fresh output owner:
`artifacts/bounded-typed-continuation/confirmation-20260919-v1/`.
Pin selection, evaluation, summary and renderer drivers before running their
respective phases. First run `prepare_conditions.py`, then verify the manifest,
checkpoint digests and clean source before `development_screen.py`. Commands use
`uv run --offline --python 3.10 --extra mps --group dev python` followed by the
driver path. Python 3.10.20, Torch 2.11.0, NumPy 1.26.4; Apple M5 with 24 GiB;
CPU, one thread. No training and no external network work.

Selection has a five-minute bound. Evaluation has a one-hour bound, at most
6 GiB physical footprint/RSS, at least 2 GiB available memory, at most 128 MiB
swap growth and 2 GiB output. Recover raw generation state exactly at the first
eligible checkpoint after candidate 512 for indices 0 and 12, seed 17, for each
model. A selected source ending earlier has a recorded recovery-not-applicable
status and must receive an earlier nonterminal recovery check where possible.
Never overwrite outputs or resume under changed source/configuration. Stop on
resource, input identity, nonfinite score, mechanics, export or recovery failure;
retain partial evidence and its failure reason. Preparation errors must be fixed
and recorded before dependent execution.

Read-only inspection of the packaged inference APIs may overlap evaluation.
Keep product source unchanged throughout the frozen evaluation. A reusable
inference interface is engineering work to follow the source-pinned screen,
with targeted equivalence tests and no altered model behavior.

## Next lifecycle condition

Append exact selections, driver digests, execution ledgers and scoped judgments.
This proposed Card has no accepted revision. Its exploratory evidence may narrow
the next decision but does not change Note status or complete the overall goal.

## Pinned selection driver

`confirmation-20260919-v1/prepare_conditions.py` SHA
`ed48b9fa1eafb55f16192becc7f4be06b7f5df7be63383e8c07c020310bb7912`
adapts the prior source-only selector, SHA
`af8abee08a65ce07370b8fdd5009531a8b71a117cc4b3f0589133d4cd855209d`.
It checks clean product source, excludes the 24 prior groups before selection,
uses the declared RNG and scope rules, and requires two eligible wider scopes.
Its syntax compiles. No model inference or new training has run for this Card.

## Selection result and pinned evaluation driver

Selection completed in 3.705 s from 1511 eligible charts in 411 remaining groups.
The 24 selected groups are distinct and disjoint from all 24 prior screen groups.
`confirmation-20260919-v1/screen-conditions.json` SHA
`4200b2c27b8d21ba617889d75add008d3a90efff4cb194b805b5c0ca0d5ac5dd`.
The declared local indices are 0/3/6/9/12/16; the first two eligible ordinary wider
indices are 1/2. Ordinary index 0 has only 200 physical rows, so its declared
recovery uses an earlier nonterminal midpoint, with the before-512 condition
recorded explicitly. The LN-rich recovery retains the candidate-512 check.

`confirmation-20260919-v1/development_screen.py` SHA
`8904a3d3f093fcb429feb378139fcf15e10bbca973b21160538331980f1237b7`
adapts the previous completed native screen, SHA
`e29439dd9ed2aea55d568e46c4c9ea3cbada98c5287cc0057d3647bcb87c9eba`.
The reviewed changes load the two declared 2M checkpoints, verify this condition
manifest and the existing behavior-neutral preflight, and perform the four
declared recoveries. Generator inputs, likelihood, support, resource bounds and
export/reparse checks are unchanged. Syntax compilation passes. The generation
summary helper stays pinned at
`5b6f4286fa82d7ff9bee76284268fa761d0a099d127200f8ff6156aa174ea8c4`.

## Pinned summary and inspection drivers

`confirmation-20260919-v1/readout.py` SHA
`c0f67cc16353fc38894b3689c006afe2bab4b9ffde6f5a27b17718fae4a3d57f`
recounts complete generated rows, verifies exported bytes and requires all four
declared recoveries. It reports ordinary/stress aggregates and each stress
category, with the fixed bootstrap and numeric gates. The common row/statistics
helper stays at SHA
`e1becc82254144700ea22b3c01adfb3abe4472cd1b0dd72345c46905c574a81e`.

`confirmation-20260919-v1/render_masked.py` SHA
`e072e6e040fdc7806cf4e37c20a6fe4af661745567e106823a0c55bf541bd96c`
prepares six local pairs and two wider pairs at the fixed manifest indices.
It verifies the Foundation, canonical renderer and chart-reader digests, then
writes the public packet to `masked-inspection-v1/`. Its private A/B map must
remain unread until all eight judgments have been recorded. Both drivers compile.
The native screen is running; no product source or model parameters changed.

## Native-screen result

The screen completed in 394.136 s: 144 native generations, 48 complete-suffix
scores, all mechanical/export/reparse checks and four exact raw-state recoveries.
Both models recovered ordinary index 0 at candidate 113 and LN-rich index 12
at candidate 512, as declared for the short and longer sources. Peak physical
footprint was 233145328 bytes; swap growth was zero. The product worktree stayed
clean at the pinned revision. No new training or product edits occurred.

Raw `development-screen-v1/readout.json` SHA
`b46ff31e677ec370dfc5487da3b56896ad7fd27baaff23a2e8c2af04cb733765`;
recounted `readout.json` SHA
`e93382972b8ad556dc213b1d9a2455630cab0a57d3fb7f5e70bd9a72b5782952`.

| Scope | Initialization | Group macro NLL | Group mean rapid pairs/1000 heads | Rapid pairs | Maximum rapid run | LN head share |
| --- | --- | --- | --- | --- | --- | --- |
| 12 ordinary groups | 171 | 2.154512 | .015656 | 2 | 2 | 27.245% |
| 12 ordinary groups | 172 | 2.132558 | .079493 | 9 | 2 | 52.625% |
| 12 stress groups | 171 | 2.102821 | .097464 | 8 | 2 | 24.029% |
| 12 stress groups | 172 | 2.014975 | .409852 | 34 | 3 | 54.725% |
| All 24 groups | 171 | 2.128667 | .056560 | 10 | 2 | 25.342% |
| All 24 groups | 172 | 2.073766 | .244673 | 43 | 3 | 53.857% |

All four primary numeric guards pass. Complete-suffix pooled NLL is
1.970715593/1.920650907 for 171/172. Paired group macro NLL differences for
172 minus 171 are -.021954 in ordinary groups, 95% bootstrap interval
[-.044954,-.002709], and -.087846 in stress groups, interval
[-.175231,-.022753]. Corresponding rapid-rate differences are +.063837,
interval [0,.151312], and +.312388, interval [.049221,.627071]. These remain
conditional comparisons of two fixed checkpoints, not uncertainty over all
initializations or over player preference.

The higher rapid burden of 172 must remain visible alongside its better NLL.
In the two dense stress groups its mean rate is 1.588278 versus .249053 for 171,
and maximum run is three versus two. The .8 gate was prospectively defined for
the aggregate stress stratum, not each two-group category; it passes without
establishing low burden uniformly across categories. Overall, 25 of 172's 43
rapid pairs occur with one free lane, versus none of 171's 10. The six stress
category readouts are retained in the report. Both LN-rich sources have zero
rapid pairs, so the higher overall LN share is not by itself a burden diagnosis.

Shortest generated LN durations across all seeds are 18 ms for 171 and 8 ms for
172. These remain unfiltered observations, not a new minimum-duration policy or
a finding of player-specific impossibility. The preselected semantic review must
assess actual action relationships. After that masked review, targeted examination
of the dense rapid runs and shortest holds can clarify the limits of the aggregate
screen; any such examination must be labeled diagnostic, not substituted for the
preselected confirmation cases.

## Prepared masked packet and continuation state

`masked-inspection-v1/manifest.json` SHA
`b9bffadf6cc8d6ff24514e95708bdb81eb6fea1f48ebdb54032a40563c6d52bf`.
It contains 208 images: six local pairs with eight pages per variant and two wider
pairs with 28 pages per variant. The cases are ordinary 0/3/6/9, LN-rich 12,
dense 16, then ordinary 1/2 at 64s. These eight scopes belong to eight distinct
selected groups. All presentation and Foundation digest checks passed during
rendering. No image in this new packet has yet been inspected, and its private
initialization map remains unread. Record all eight masked judgments before
opening `masked-inspection-v1/private-mapping.json`.

Selection, evaluation, summary and rendering processes are terminal. There is no
process to restart or wait for. Next perform the preselected masked inspection,
then the declared diagnostic follow-up if needed. Numerical passage alone does
not establish the semantic gate or complete the playable-generation goal.

Read-only inference inspection confirms a reusable core in `generation.py`,
including seed-only initialization, exact physical state, finite raw history and
verified restoration. Packaged training already exists. A packaged bounded-task
generation runner is still absent; complete experiment execution currently uses
artifact drivers. A practical handoff should expose the frozen checkpoint and
external R/H/complete-seed condition, durable generation and verified osu! export,
with native behavior equivalence tests. Existing row logging, resource guards and
export helpers can be reused. Source playback metadata can accompany export, but
audio is not read by the model or bundled by the existing exporter. This is an
engineering direction to implement after the quality review, not a new model
contract or a completed inference feature.
