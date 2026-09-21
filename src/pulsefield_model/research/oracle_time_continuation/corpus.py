"""Read pinned source allocations without creating a second song-group split."""
import json
from pathlib import Path

from ..scoped_style_modeling.dataset import ContractError, canonical_json, digest
from .storage import file_digest


def read_split(manifest_path, expected_sha):
    if not expected_sha:
        raise ContractError('Requires a pinned split_sha256')
    manifest = json.loads(Path(manifest_path).read_text())
    actual = digest(canonical_json({key: value for key, value in manifest.items() if key != 'sha256'}).encode())
    if actual != manifest.get('sha256') or actual != expected_sha:
        raise ContractError('Split manifest differs from its pinned SHA-256 identity')
    assignments = manifest['sources']
    groups = {}
    for assignment in assignments.values():
        group, assigned = assignment['group_id'], assignment['split']
        if assigned not in ('train', 'validation', 'test') or groups.setdefault(group, assigned) != assigned:
            raise ContractError('Split manifest has invalid or conflicting song-group assignments')
    return assignments


def source_assignments(manifest_path, expected_sha, selection, split):
    """Check the entire allocation before opening any selected payload."""
    if not selection:
        raise ContractError('Requires an explicit nonempty source_sha256 selection')
    assignments = read_split(manifest_path, expected_sha)
    for sha in selection:
        if sha not in assignments or assignments[sha]['split'] != split:
            raise ContractError(f'Selected source must belong to the pinned {split} split')
    return assignments


def catalog_entries(path, expected_sha, assignments, selection=(), *, split='train'):
    """Consume an existing admitted catalog, including its held-out allocations.

    The catalog owns transitive song groups and deduplication. Its exact bytes
    must be pinned. This adapter verifies identities and split consistency, and
    never reconstructs groups, reads test payloads or imports legacy features.
    An empty selection explicitly means all members of the requested split.
    """
    if not expected_sha or file_digest(Path(path), 32 * 1024**2) != expected_sha:
        raise ContractError('Catalog differs from its pinned SHA-256 identity')
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not entries or len(entries) > 100000:
        raise ContractError('Catalog requires 1 to 100000 admitted source identities')
    groups, arrangements, by_sha, published_groups = {}, {}, {}, {}
    for entry in entries:
        sha, arrangement = entry['source_sha256'], entry['arrangement_sha256']
        assigned, group = entry['split'], entry['group_id']
        if any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
               for value in (sha, arrangement)):
            raise ContractError('Catalog source and arrangement identities must be SHA-256 digests')
        if (assigned not in ('train', 'validation') or not group or
                type(entry['event_count']) is not int or entry['event_count'] < 1 or not entry['path'] or
                sha in by_sha or arrangement in arrangements or groups.setdefault(group, assigned) != assigned):
            raise ContractError('Catalog contains invalid, duplicate or conflicting source allocations')
        if sha in assignments:
            published = assignments[sha]
            if (published['split'] != assigned or
                    published_groups.setdefault(published['group_id'], group) != group):
                raise ContractError('Catalog conflicts with the pinned annotation allocation')
        by_sha[sha], arrangements[arrangement] = entry, assigned
    selected = list(selection) if selection else [sha for sha, entry in by_sha.items() if entry['split'] == split]
    if not selected or len(selected) != len(set(selected)):
        raise ContractError('Catalog selection requires distinct, nonempty source identities')
    if any(sha not in by_sha or by_sha[sha]['split'] != split for sha in selected):
        raise ContractError(f'Selected source must belong to the pinned {split} catalog allocation')
    return tuple(by_sha[sha] for sha in sorted(selected))


def admit_entry(store, entry):
    source = store.admit(Path(entry['path']), entry['source_sha256'],
                         group_id=entry['group_id'], split=entry['split'])
    if (source.identity.arrangement_sha256, source.row_count) != (entry['arrangement_sha256'], entry['event_count']):
        raise ContractError('Source arrangement or row count differs from the pinned catalog')
    return source
