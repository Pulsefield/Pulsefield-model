"""Learn scoped condition distinctions without an additional inference model."""
from torch.nn import functional as F


def condition_alignment(positive, negative, reference_positive, reference_negative, *, beta=.1):
    """Contrast summed neural row-factor scores against a frozen reference.

    Positive and negative scores describe the same factual rows on their own
    genuine history, under observed and mismatched conditions. They are not
    labels transplanted onto a generated prefix. Callers own scope selection,
    compatible negative conditions and exclusion of padded rows. References are
    detached even if supplied from an autograd-enabled computation.

    The symmetric loss cannot improve by shifting both relative scores equally.
    It is a training proxy; generated control and playability require separate
    evaluation. Empirically paired negatives do not imply independent-marginal
    noise-contrastive estimation or equivalence to guided sampling.
    """
    if beta <= 0:
        raise ValueError('Condition alignment requires a positive score scale')
    positive_gap = beta*(positive-reference_positive.detach())
    negative_gap = beta*(negative-reference_negative.detach())
    return (F.softplus(-positive_gap)+F.softplus(negative_gap)).mean()
