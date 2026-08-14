# Recall batteries

Use these searches as probes for the taxonomy in `../SKILL.md`. Every hit requires semantic judgment. The batteries over-match by design and under-match by nature, so pair them with an unpatterned read of the densest prose in scope.

## Invocation rules

- Replace `<scope>` with the explicit file or directory supplied by the task. Do not default it to the repository root.
- Use `--hidden --glob '!.git/**'` when `.agents/` belongs to scope; ripgrep otherwise skips hidden paths.
- Place exclusion globs after inclusion globs so a later include cannot re-admit them.
- Exclude `.venv/**`, `artifacts/**`, `ref-proj/**`, `**/__pycache__/**`, `.pytest_cache/**`, `*.egg-info/**`, and this skill's own directory unless the user explicitly targets one.
- Exclude `pulsf-prose-standard/references/examples.md` during a broad skill audit because it intentionally quotes bad prose for calibration.
- Do not broadly scan generated datasets, checkpoints, run snapshots, recorded outputs, or notebook outputs. Narrow to authored Markdown or source cells when a notebook is explicitly in scope.
- Use case-insensitive matching for natural-language probes, but keep identifier probes case-sensitive where capitalization reduces noise.
- Test a pattern against a known-positive string before trusting a zero-hit result. Zero hits never replace an unpatterned read.

A reusable exclusion tail is:

```sh
--glob '!.git/**' \
--glob '!.venv/**' \
--glob '!artifacts/**' \
--glob '!ref-proj/**' \
--glob '!**/__pycache__/**' \
--glob '!.pytest_cache/**' \
--glob '!*.egg-info/**' \
--glob '!.agents/skills/pulsf-trim-cot-leakage/**' \
--glob '!.agents/skills/pulsf-prose-standard/references/examples.md'
```

## Session-citation battery

```sh
rg -n --hidden '\(decision[ :]+[A-Za-z0-9-]+|\(audit[ :]+[A-Za-z0-9-]+|design §|plan §|the design ledger|the plan above|as discussed above|as requested|the user (asked|requested|said)' <scope> <exclusions>
rg -n --hidden '\b(T|W|P)[-_]?[0-9]+\b|\bphase [A-Z][0-9]+\b' <scope> <exclusions>
rg -n --hidden '§[0-9]' <scope> <exclusions>
```

Treat phase and section matches cautiously. `phase_b`, checkpoint phases, a committed document section, and an RFC section can be stable identifiers.

## Publication-viewpoint battery

```sh
rg -n --hidden -i 'this PR|this pull request|this branch|this stack|later PR|next PR|previous commit|this commit|in the diff|review round|reviewer confirmed|rejected in review|as of v[0-9]+' <scope> <exclusions>
```

Process documentation about how to write a PR may legitimately say “the PR.” The defect is durable product prose adopting one publication's viewpoint.

## Change-narration battery

```sh
rg -n --hidden -i 'used to |no longer|previously|the old (implementation|version|path|config)|was renamed|was moved|was removed|now |today|for now|this cut|current cut|roadmap' <scope> <exclusions>
rg -n --hidden -i '\bv[0-9]+ of (this|the) (doc|document|note|plan)' <scope> <exclusions>
```

Do not flag mapper version names, protocol versions, endpoint paths, checkpoint revisions, or live old/new runtime objects merely because they contain version vocabulary.

## Justification and derivation battery

```sh
rg -n --hidden -i 'this is (safe|correct) because|the cast is safe|it simply|obviously|clearly|first we |then we |next we |which is why|the code above|the test below' <scope> <exclusions>
```

Words such as “clearly” can be legitimate mathematical qualifiers, and ordered operational guides must use sequence language. Flag a derivation transcript only when it restates adjacent implementation or answers an absent reviewer.

## Hedge and planning battery

```sh
rg -n --hidden -i 'probably |should be enough|should suffice|good enough for now|later work|future PR|follow up later|candidate variant|selection pressure|next-loop action|kill criteria' <scope> <exclusions>
```

Research-triage templates legitimately contain planning labels. Finished analyses normally translate them into evidence, uncertainty, and a focused next diagnostic.

## Ephemeral-reference battery

```sh
rg -n --hidden -i 'terminal output above|result above|chat above|local run|notebook cell [0-9]+|/tmp/|artifacts/[^ )]+' <scope> <exclusions>
```

An exact artifact path can be valid in a local result log. In curated docs, preserve the claim's evidence and provenance without making a fresh clone depend on unretained state.

## Mixed-language and scaffolding battery

```sh
rg -n --hidden '(^|[^[:alnum:]_])(设计稿|评审|上一轮|旧版|本版|私有|用户说)([^[:alnum:]_]|$)' <scope> <exclusions>
rg -n --hidden '^(assistant|user|analysis|final):|BEGIN (PROMPT|RESPONSE)|END (PROMPT|RESPONSE)' <scope> <exclusions>
```

Keep deliberate translations and quoted evidence in a surface that owns them. Remove accidental working-language fragments and prompt scaffolding from otherwise finished English prose.

## Known false-positive families

- **Instrumental “used to”:** “the key used to sign requests” can describe purpose rather than past state.
- **Runtime old/new:** “the old session drains before the new one streams” describes concurrent live objects.
- **Model versions:** `v2`, `v2_1`, bundle IDs, schema versions, and `/v1/` paths are stable identifiers.
- **Training phases:** `phase_b`, warmup, forward, backward, and update name runtime or experiment stages.
- **External or committed sections:** RFC sections, paper sections, and stable headings in committed docs resolve.
- **PR process instructions:** “verify the PR base” in a pre-push skill is procedural, not leaked branch history.
- **Alternatives and postmortems:** “rejected” and historical language belong in sections whose genre owns the decision or incident story.
- **Measured “current”:** `current_allocated_memory()` and “current batch” can be API or runtime vocabulary, not an indexical timestamp.
- **Planner/model vocabulary:** a learned planner, control plan, or selection policy may be the technical subject rather than authoring workflow.
