"""Recover the snapshot's original .osu bytes without substituting revisions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from .dataset import ContractError, fetch_verified


def source_url(source: dict) -> str:
    ref = source["source_ref"]
    if ref["kind"] in ("osu", "url", "content-addressed"):
        url = ref["uri"]
    elif ref["kind"] == "hf" and ref.get("record_key") is None:
        commit = ref["commit"]
        if not commit or len(commit) != 40:
            raise ContractError("HF source requires an immutable full commit")
        url = f"https://huggingface.co/datasets/{ref['repository']}/resolve/{commit}/{quote(ref['path'])}"
    else:
        raise ContractError(f"Unsupported source locator (requires original .osu bytes): {ref}")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ContractError(f"Expected HTTPS source locator: {url}")
    return url


def recover_sources(sources: list[dict], cache_dir: Path, *, download: bool,
                    workers: int = 4, timeout: float = 30) -> list[dict]:
    """Return one explicit recovery status per source, including all failures.

    Only hash-verified bytes enter the cache. Offline mode checks exactly the
    content-addressed paths; it never scans unrelated datasets or artifacts.
    """
    from .dataset import checked_bytes

    def recover(source: dict) -> dict:
        sha = source["source_sha256"]
        path = cache_dir / f"{sha}.osu"
        result = {"source_sha256": sha, "path": str(path), "source_ref": source["source_ref"]}
        try:
            cached = path.exists()
            data = checked_bytes(path, sha) if cached or not download else fetch_verified(source_url(source), path, sha, timeout)
            if len(data) != source["byte_length"]:
                raise ContractError(f"{sha}: source byte_length differs from publication")
            result.update(status="verified", bytes=len(data), retrieval="cache" if cached else "download")
        except (OSError, ValueError) as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return result

    if workers < 1 or timeout <= 0:
        raise ContractError("Recovery workers and timeout must be positive")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(recover, sorted(sources, key=lambda s: s["source_sha256"])))
