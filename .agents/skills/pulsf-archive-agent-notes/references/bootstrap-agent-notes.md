# Bootstrap the Agent Note Store

Use this procedure only when the user explicitly asks to initialize or repair Agent Note storage. It creates Git state but never changes a product branch or publishes a remote ref.

## Resolve the existing state

From a product worktree, record the repository root and inspect all three identities before creating anything:

```sh
git show-ref --verify refs/heads/agent-notes
git remote get-url origin
git ls-remote --exit-code origin refs/heads/agent-notes
git worktree list --porcelain
```

Run the remote query only when `origin` exists; otherwise record that the store will remain local. For `git ls-remote --exit-code`, exit 2 with no matching ref confirms absence; authentication, network, server, or other query failure is inconclusive and blocks orphan creation. Use the explicit absolute `<notes-worktree>` path supplied or approved for this repository. It must be outside every existing worktree and must not already contain files. Do not reuse a stale directory, reset an existing ref, or invent a second notes branch.

- If the local ref and a valid registered worktree already exist, do not bootstrap.
- If the local ref exists without a worktree, validate the ref, then use `git worktree add <notes-worktree> agent-notes`.
- If only the remote ref exists, fetch it, create the local `agent-notes` branch at that exact OID, validate it, then add its worktree. Do not replace it with a new orphan history.
- If both refs exist, verify their ancestry. A local fast-forward descendant may be attached; divergence, remote-only commits, or a dirty or malformed existing worktree requires reconciliation before setup continues.

## Initialize a new orphan ref

Only when neither local nor remote ref exists, run:

```sh
git worktree add --orphan -b agent-notes <notes-worktree>
```

In that new worktree, create only this `.gitignore`:

```gitignore
/*
!/.gitignore
!/artifacts/
/artifacts/*
!/artifacts/agent-notes/
!/artifacts/agent-notes/**
```

Stage `.gitignore`, commit it as the initial `agent-notes` commit, and optionally lock the linked worktree against accidental pruning. Do not copy product files into the orphan worktree. A remote push remains a separate explicitly authorized action.

## Verify the store

- `git branch --show-current` in the new worktree reports `agent-notes`.
- `git merge-base main agent-notes` produces no OID and exits nonzero.
- The committed tree contains only `.gitignore` and Markdown under `artifacts/agent-notes/`.
- The allowlist ignores an unrelated root file and does not ignore a probe path below `artifacts/agent-notes/`.
- `git worktree list --porcelain` shows exactly one worktree for `refs/heads/agent-notes`.
- The invoking product worktree still ignores `artifacts/agent-notes/**` and has no product-branch change from setup.

Report the product worktree, notes worktree, initial note-branch OID, orphan check, allowlist check, and whether the ref remains local.
