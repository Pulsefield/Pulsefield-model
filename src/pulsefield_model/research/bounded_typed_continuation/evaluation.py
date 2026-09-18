"""Complete-suffix likelihood with bounded exact-context recomputation."""
from __future__ import annotations

import torch

from ..scoped_style_modeling.dataset import ContractError
from .data import SourceInterval
from .train_run import add_metrics, measure


@torch.no_grad()
def suffix_likelihood(model, source, *, chunk_onsets=128, candidate_budget=8192, check=lambda _phase: None):
    """Pay every suffix factor once, including endpoint labels beyond each chunk.

    Chunks only bound computation. Complete exact facts and finite raw context are
    recomputed from the same source history at every boundary. R1/O1 therefore
    score the same complete Y conditional on the same external R/H/object seed.
    """
    if type(chunk_onsets) is not int or not 1 <= chunk_onsets <= 256:
        raise ContractError('Suffix scoring chunks must contain 1 to 256 onsets')
    previous_training = model.training
    model.eval()
    result = {}
    try:
        for first in range(0, len(source.onsets), chunk_onsets):
            interval = SourceInterval(source, first, min(chunk_onsets, len(source.onsets) - first))
            record = measure(model, [interval], candidate_budget=candidate_budget, backward=False,
                             denominator=interval.onset_count, check=check)
            add_metrics(result, record)
        result['nll_per_onset'] = (result['head_nll_sum'] + result['endpoint_nll_sum']) / result['source_onsets']
        return result
    finally:
        model.train(previous_training)
