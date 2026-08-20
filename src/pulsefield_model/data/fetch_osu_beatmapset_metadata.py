"""Fetch missing beatmapset metadata from the official osu! API v2.

The dataset layout is expected to be either::

    dataset/<shard>/<beatmapset_id>/*.osu

or a single shard root containing ``<beatmapset_id>`` directories. Metadata is
written to ``metadata.json`` in each beatmapset directory. Existing metadata is
left untouched unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TOKEN_URL = "https://osu.ppy.sh/oauth/token"
API_BASE_URL = "https://osu.ppy.sh/api/v2"
METADATA_FILENAME = "metadata.json"
METADATA_SCHEMA_VERSION = 1
DEFAULT_REQUEST_INTERVAL_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_REQUEST_ATTEMPTS = 3

# These constants are the local metadata contract. API fields not named here
# are deliberately not persisted, so downstream code has a stable schema.
BEATMAPSET_FIELDS = (
    "id",
    "artist",
    "artist_unicode",
    "title",
    "title_unicode",
    "creator",
    "user_id",
    "source",
    "status",
    "ranked",
    "ranked_date",
    "submitted_date",
    "last_updated",
    "bpm",
    "offset",
    "play_count",
    "favourite_count",
    "rating",
    "nsfw",
    "video",
    "storyboard",
    "spotlight",
    "is_scoreable",
    "preview_url",
    "track_id",
    "availability",
    "hype",
    "nominations_summary",
    "covers",
)

BEATMAP_FIELDS = (
    "id",
    "beatmapset_id",
    "mode",
    "mode_int",
    "status",
    "version",
    "difficulty_rating",
    "bpm",
    "total_length",
    "hit_length",
    "accuracy",
    "ar",
    "cs",
    "drain",
    "convert",
    "count_circles",
    "count_sliders",
    "count_spinners",
    "last_updated",
    "ranked",
    "is_scoreable",
    "playcount",
    "passcount",
    "checksum",
    "max_combo",
    "owners",
    "top_tag_ids",
)

RELATED_TAG_FIELDS = (
    "id",
    "name",
    "ruleset_id",
    "description",
)


class OsuApiError(RuntimeError):
    """An osu! OAuth or API request failed."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OsuApiClient:
    """Small client for the client-credentials flow and beatmapset endpoint."""

    def __init__(
        self,
        *,
        client_id: int,
        client_secret: str,
        request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self._access_token: str | None = None
        self._last_api_request_at: float | None = None

    def fetch_beatmapset(self, beatmapset_id: int) -> Mapping[str, Any]:
        if self._access_token is None:
            self._access_token = self._fetch_access_token()

        try:
            return self._fetch_beatmapset_with_token(beatmapset_id)
        except OsuApiError as exc:
            if exc.status != 401:
                raise

        # A full scan can outlive a token. Refresh once if the server rejects it.
        self._access_token = self._fetch_access_token()
        return self._fetch_beatmapset_with_token(beatmapset_id)

    def _fetch_access_token(self) -> str:
        body = urlencode(
            {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                "scope": "public",
            },
        ).encode("utf-8")
        request = Request(
            TOKEN_URL,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Pulsefield-model-metadata-fetcher/1",
            },
            method="POST",
        )
        payload = self._request_json(request, rate_limited=False)
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise OsuApiError("OAuth response did not contain an access_token")
        return token

    def _fetch_beatmapset_with_token(self, beatmapset_id: int) -> Mapping[str, Any]:
        assert self._access_token is not None
        request = Request(
            f"{API_BASE_URL}/beatmapsets/{beatmapset_id}",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._access_token}",
                "User-Agent": "Pulsefield-model-metadata-fetcher/1",
            },
            method="GET",
        )
        return self._request_json(request, rate_limited=True)

    def _request_json(self, request: Request, *, rate_limited: bool) -> Mapping[str, Any]:
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            if rate_limited:
                self._wait_for_rate_limit()
                self._last_api_request_at = time.monotonic()

            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.load(response)
            except HTTPError as exc:
                status = exc.code
                detail = _read_http_error_detail(exc)
                if _is_retryable_status(status) and attempt + 1 < MAX_REQUEST_ATTEMPTS:
                    time.sleep(_retry_delay_seconds(exc, attempt))
                    continue
                raise OsuApiError(
                    f"osu! request failed with HTTP {status}: {detail}",
                    status=status,
                ) from exc
            except (URLError, TimeoutError) as exc:
                if attempt + 1 < MAX_REQUEST_ATTEMPTS:
                    time.sleep(2**attempt)
                    continue
                raise OsuApiError(f"osu! request failed: {exc.reason if isinstance(exc, URLError) else exc}") from exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise OsuApiError("osu! returned an invalid JSON response") from exc

            if not isinstance(payload, Mapping):
                raise OsuApiError("osu! returned a JSON value that was not an object")
            return payload

        raise AssertionError("request retry loop exhausted unexpectedly")

    def _wait_for_rate_limit(self) -> None:
        if self._last_api_request_at is None:
            return
        elapsed = time.monotonic() - self._last_api_request_at
        remaining = self.request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)


def iter_beatmapset_directories(dataset_root: Path) -> Iterable[Path]:
    """Yield numeric directories containing at least one local .osu file."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    found: dict[Path, Path] = {}
    for first_level in dataset_root.iterdir():
        if not first_level.is_dir():
            continue
        if _is_beatmapset_directory(first_level):
            found[first_level.resolve()] = first_level
            continue
        for second_level in first_level.iterdir():
            if _is_beatmapset_directory(second_level):
                found[second_level.resolve()] = second_level

    yield from sorted(found.values(), key=lambda path: (int(path.name), str(path)))


def build_metadata(payload: Mapping[str, Any], *, fetched_at: str | None = None) -> dict[str, Any]:
    """Project an osu! BeatmapsetExtended response into the local schema."""
    raw_mapper_tags = payload.get("tags")
    if not isinstance(raw_mapper_tags, str):
        raw_mapper_tags = ""

    pack_tags = payload.get("pack_tags")
    if not isinstance(pack_tags, list):
        pack_tags = []

    related_tags = payload.get("related_tags")
    if not isinstance(related_tags, list):
        related_tags = []

    beatmaps = payload.get("beatmaps")
    if not isinstance(beatmaps, list):
        beatmaps = []

    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        "api_endpoint": f"{API_BASE_URL}/beatmapsets/{payload.get('id')}",
        **_select_fields(payload, BEATMAPSET_FIELDS),
        "tags": {
            "mapper_raw": raw_mapper_tags,
            "mapper": raw_mapper_tags.split(),
            "pack": [tag for tag in pack_tags if isinstance(tag, str)],
            "genre": _tag_descriptor(payload.get("genre")),
            "language": _tag_descriptor(payload.get("language")),
            "related": [
                _select_fields(tag, RELATED_TAG_FIELDS)
                for tag in related_tags
                if isinstance(tag, Mapping)
            ],
        },
        "beatmaps": [
            {
                **_select_fields(beatmap, BEATMAP_FIELDS),
                "top_tag_ids": (
                    beatmap.get("top_tag_ids") if isinstance(beatmap.get("top_tag_ids"), list) else []
                ),
            }
            for beatmap in beatmaps
            if isinstance(beatmap, Mapping)
        ],
    }
    return metadata


def write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """Write JSON atomically so an interrupted run never leaves a partial file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    temporary_path.replace(path)


def read_env_file(path: Path) -> dict[str, str]:
    """Read the simple KEY=VALUE form needed by this script."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f"invalid environment line {line_number} in {path}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def resolve_credentials(
    *,
    client_id: str | int | None,
    client_secret: str | None,
    env_file: Path,
) -> tuple[int, str]:
    """Resolve CLI, process environment, then local env-file credentials."""
    file_values = read_env_file(env_file)
    resolved_client_id = (
        client_id
        if client_id is not None
        else os.environ.get("OSU_CLIENT_ID") or file_values.get("OSU_CLIENT_ID")
    )
    resolved_client_secret = (
        client_secret
        if client_secret is not None
        else os.environ.get("OSU_CLIENT_SECRET") or file_values.get("OSU_CLIENT_SECRET")
    )

    if resolved_client_id is None or resolved_client_secret is None:
        raise ValueError(
            "osu! credentials are required: pass --client-id/--client-secret or set "
            "OSU_CLIENT_ID/OSU_CLIENT_SECRET",
        )
    try:
        numeric_client_id = int(resolved_client_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("OSU_CLIENT_ID must be an integer") from exc
    if numeric_client_id <= 0:
        raise ValueError("OSU_CLIENT_ID must be positive")
    if not resolved_client_secret:
        raise ValueError("OSU_CLIENT_SECRET must not be empty")
    return numeric_client_id, resolved_client_secret


def fetch_missing_metadata(
    *,
    dataset_root: Path,
    client: OsuApiClient,
    overwrite: bool = False,
    limit: int | None = None,
) -> tuple[int, int, list[tuple[Path, str]]]:
    """Fetch pending metadata and return (written, skipped, failures)."""
    directories = list(iter_beatmapset_directories(dataset_root))
    pending: list[Path] = []
    skipped = 0
    for directory in directories:
        if (directory / METADATA_FILENAME).exists() and not overwrite:
            skipped += 1
        else:
            pending.append(directory)

    if limit is not None:
        pending = pending[:limit]

    print(
        f"Found {len(directories)} beatmapsets; skipping {skipped} with existing metadata; "
        f"fetching {len(pending)}.",
    )
    written = 0
    failures: list[tuple[Path, str]] = []
    for index, directory in enumerate(pending, start=1):
        beatmapset_id = int(directory.name)
        try:
            payload = client.fetch_beatmapset(beatmapset_id)
            response_id = payload.get("id")
            if response_id != beatmapset_id:
                raise OsuApiError(
                    f"response beatmapset id {response_id!r} did not match requested id {beatmapset_id}",
                )
            write_metadata(directory / METADATA_FILENAME, build_metadata(payload))
        except (OSError, OsuApiError, ValueError) as exc:
            failures.append((directory, str(exc)))
            print(f"\nFailed beatmapset {beatmapset_id}: {exc}", file=sys.stderr)
            print_progress(index, len(pending), failed=len(failures))
            continue

        written += 1
        print_progress(index, len(pending), failed=len(failures))

    return written, skipped, failures


def format_progress(current: int, total: int, *, failed: int = 0, width: int = 30) -> str:
    """Render a dependency-free, fixed-width progress bar."""
    fraction = 1.0 if total == 0 else min(max(current / total, 0.0), 1.0)
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    suffix = f", {failed} failed" if failed else ""
    return f"[{bar}] {current}/{total} ({fraction:6.2%}){suffix}"


def print_progress(current: int, total: int, *, failed: int = 0) -> None:
    end = "\n" if current >= total else "\r"
    print(format_progress(current, total, failed=failed), end=end, flush=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path("dataset"),
        help="dataset root or a single shard root (default: dataset)",
    )
    parser.add_argument("--client-id", help="osu! OAuth client ID; overrides OSU_CLIENT_ID")
    parser.add_argument("--client-secret", help="osu! OAuth client secret; overrides OSU_CLIENT_SECRET")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="fallback KEY=VALUE credential file (default: .env)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing metadata.json files")
    parser.add_argument("--limit", type=int, help="fetch at most this many missing beatmapsets")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and report without authenticating or writing metadata",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return 2

    try:
        directories = list(iter_beatmapset_directories(args.dataset_root))
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    skipped = sum(
        (directory / METADATA_FILENAME).exists() and not args.overwrite for directory in directories
    )
    pending = len(directories) - skipped
    if args.limit is not None:
        pending = min(pending, args.limit)
    if args.dry_run:
        print(
            f"Found {len(directories)} beatmapsets; would skip {skipped} with existing metadata; "
            f"would fetch {pending}.",
        )
        return 0

    try:
        client_id, client_secret = resolve_credentials(
            client_id=args.client_id,
            client_secret=args.client_secret,
            env_file=args.env_file,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    client = OsuApiClient(client_id=client_id, client_secret=client_secret)
    written, skipped, failures = fetch_missing_metadata(
        dataset_root=args.dataset_root,
        client=client,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(f"Done: wrote {written}, skipped {skipped}, failed {len(failures)}.")
    return 1 if failures else 0


def _is_beatmapset_directory(path: Path) -> bool:
    return path.is_dir() and path.name.isdecimal() and next(path.glob("*.osu"), None) is not None


def _select_fields(payload: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: payload.get(field) for field in fields}


def _tag_descriptor(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"id": None, "name": None}
    return {"id": value.get("id"), "name": value.get("name")}


def _read_http_error_detail(error: HTTPError) -> str:
    try:
        raw_detail = error.read().decode("utf-8", errors="replace")
        payload = json.loads(raw_detail)
    except (OSError, json.JSONDecodeError):
        return error.reason or "unknown error"
    if isinstance(payload, Mapping):
        detail = payload.get("error") or payload.get("message")
        if detail:
            return str(detail)
    return error.reason or "unknown error"


def _is_retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def _retry_delay_seconds(error: HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass
    return float(2**attempt)


if __name__ == "__main__":
    raise SystemExit(main())
