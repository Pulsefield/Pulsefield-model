"""TRAIN-only joint arrangement representatives; no style or quality labels."""
from collections import Counter

import numpy as np

from ..scoped_style_modeling.dataset import ContractError

PROFILE_FORMAT = 'planned-audio/profiles-v1'
PROFILE_FIELDS = ('head_rows_per_second', 'heads_per_head_row', 'ln_head_fraction')


def arrangement_values(actions, duration_ms):
    """Measure a separate complete target, using the full decoded audio duration."""
    actions = np.asarray(actions)
    heads = np.isin(actions, (1, 2))
    h, n = int(heads.any(-1).sum()), int(heads.sum())
    if not h or not n or duration_ms <= 0:
        raise ContractError('An arrangement profile requires heads and a positive audio duration')
    return np.array([h * 1000 / duration_ms, n / h, (actions == 2).sum() / n], np.float64)


def transform(values):
    values = np.asarray(values, np.float64)
    if (values.shape[-1] != 3 or not np.isfinite(values).all() or
            np.any(values[..., 0] <= 0) or np.any(values[..., 1] < 1) or
            np.any(values[..., 1] > 4) or np.any(values[..., 2] < 0) or np.any(values[..., 2] > 1)):
        raise ContractError('Profiles require positive H rate, chord width in [1,4] and LN fraction in [0,1]')
    return np.stack((np.log(values[..., 0]), np.log(values[..., 1]),
                     np.arcsin(np.sqrt(values[..., 2]))), -1)


def normalized(values, bank):
    return (transform(values) - bank['mean']) / bank['std']


def chart_assignment(chart, bank):
    """Reference labels for training/evaluation only; never a native input."""
    code = normalized(arrangement_values(chart.source.rows['actions'], chart.duration_ms), bank)
    representatives = normalized(bank['profiles'], bank)
    return int(((representatives - code) ** 2).sum(-1).argmin())


def build_profile_bank(charts, count=16):
    """Fit deterministic weighted medoids on separate TRAIN chart descriptors.

    Weight each song group equally, then its charts equally. Medoids are actual
    joint profiles. Initialization order stays stable so profile IDs do not
    become sorted, independently recombined attributes. Validation never fits
    scales, representatives or class masses.
    """
    train = sorted((c for c in charts if c.split == 'train'), key=lambda c: c.entry['source_sha256'])
    if type(count) is not int or not 1 <= count <= len(train):
        raise ContractError('Profile count must fit the available TRAIN charts')
    groups = Counter(c.group_id for c in train)
    weight = np.array([1 / (len(groups) * groups[c.group_id]) for c in train])
    raw = np.stack([arrangement_values(c.source.rows['actions'], c.duration_ms) for c in train])
    values = transform(raw)
    mean = (values * weight[:, None]).sum(0)
    std = np.sqrt(((values - mean) ** 2 * weight[:, None]).sum(0))
    if not (std > 0).all():
        raise ContractError('All three profile dimensions need nonzero TRAIN variation')
    values = (values - mean) / std
    distance = ((values[:, None] - values[None]) ** 2).sum(-1)
    medoids = [int((weight @ distance).argmin())]
    for _ in range(1, count):
        scores = weight * distance[:, medoids].min(-1)
        scores[medoids] = -1
        if scores.max() <= 0:
            raise ContractError('Fewer distinct joint TRAIN profiles than requested representatives')
        medoids.append(int(scores.argmax()))
    initial = list(medoids)
    converged = False
    for iteration in range(50):
        assigned = distance[:, medoids].argmin(-1)
        updated = []
        for k in range(count):
            members = np.flatnonzero(assigned == k)
            if not len(members):
                raise ContractError('Profile preparation produced an empty cluster')
            centroid = np.average(values[members], axis=0, weights=weight[members])
            updated.append(int(members[((values[members] - centroid) ** 2).sum(-1).argmin()]))
        if updated == medoids:
            converged = True
            break
        medoids = updated
    assigned = distance[:, medoids].argmin(-1)
    masses = np.bincount(assigned, weights=weight, minlength=count)
    if len(set(medoids)) != count or not (masses > 0).all():
        raise ContractError('Every profile must have a distinct occupied TRAIN representative')
    error = distance[np.arange(len(train)), np.asarray(medoids)[assigned]]
    return dict(format=PROFILE_FORMAT, fields=list(PROFILE_FIELDS), count=count,
        transforms=['log', 'log', 'asin(sqrt)'], weighting='uniform group, then separate chart',
        mean=mean.tolist(), std=std.tolist(), profiles=raw[medoids].tolist(), masses=masses.tolist(),
        initial_source_sha256=[train[i].entry['source_sha256'] for i in initial],
        medoid_source_sha256=[train[i].entry['source_sha256'] for i in medoids],
        iterations=iteration + 1, converged=converged,
        weighted_quantization_mse=float(weight @ error),
        quantization_squared_distance_quantiles=np.quantile(error, [0, .5, .9, 1]).tolist(),
        charts=[dict(source_sha256=c.entry['source_sha256'], rows_sha256=c.entry.get('rows_sha256'),
            audio_sha256=c.entry.get('audio_sha256'), group_id=c.group_id, duration_ms=c.duration_ms,
            values=raw[i].tolist(), weight=float(weight[i]), profile_index=int(assigned[i]))
            for i, c in enumerate(train)])
