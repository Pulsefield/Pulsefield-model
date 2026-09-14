"""Fixed-information prefix routing diagnostics over complete joint-row distributions.

The unsummarized base observation is selected once. Revealing its prefix never
reselects a near neighborhood. This is an evaluation API,
not a consistency loss or a claim of globally compatible conditionals.
"""
from dataclasses import dataclass, replace
import math

import torch

from ..scoped_style_modeling.dataset import ContractError
from .diagnostics import batch_identity
from .observation import BlockExample, advance_occupancy, visible_states
from .tensors import collate

PATH_POLICY = "fixed-unsummarized-observation/prefix-index-routing-v1"


def _reveal_prefix(example: BlockExample, prefix_rows: int) -> BlockExample:
    obs = example.observation
    if obs.summaries:
        raise ContractError("Path diagnostics require an unsummarized base: far-summary support is not encoded")
    if type(prefix_rows) is not int or not 0 < prefix_rows < len(obs.target_indices):
        raise ContractError("A path comparison requires a nonempty prefix and suffix")
    if len(example.targets) != len(obs.target_indices):
        raise ContractError("Path comparison targets must align with the hidden block")
    observed_entry = visible_states(obs)[2]
    entry = observed_entry if example.query_entering_occupancy is None else example.query_entering_occupancy
    if len(entry) != 4 or any(v is not None and type(v) is not bool for v in entry):
        raise ContractError("Path comparison entry requires four bool-or-unknown values")
    if any(value is not None and value != known for value, known in zip(entry, observed_entry)):
        raise ContractError("Path query entry must be derivable from the base observation; external entry is outside this diagnostic")
    rows = list(obs.rows)
    for index, actions in zip(obs.target_indices[:prefix_rows], example.targets[:prefix_rows]):
        if len(actions) != 4 or not any(actions):
            raise ContractError("A revealed prefix requires complete non-silent source rows")
        entry = advance_occupancy(entry, actions)
        rows[index] = replace(rows[index], actions=actions)
    targets = example.targets[prefix_rows:]
    observation = replace(obs, rows=tuple(rows), target_indices=obs.target_indices[prefix_rows:])
    return BlockExample(observation, targets, sum(any(a & 3 for a in row) for row in targets), entry)


@dataclass(frozen=True)
class PrefixPathPair:
    decoder_history: BlockExample
    encoder_observation: BlockExample
    prefix_rows: int

    def __post_init__(self):
        if self.encoder_observation != _reveal_prefix(self.decoder_history, self.prefix_rows):
            raise ContractError("Path pair changed facts, skeleton, summaries, targets or prefix legality")


def prefix_path_pair(example: BlockExample, *, prefix_rows: int) -> PrefixPathPair:
    """Move the first K target rows into encoder observations, keeping other facts fixed.

    Choose the base block and K without reading hidden actions, or explicitly
    account for that selection in the population contract. The suffix decoder
    starts from the original query entry advanced through K, even if revealing
    K would permit a stronger occupation estimate from other encoder facts.
    Known query-entry values must follow from the base observation. Independently
    supplied entry conditions are outside this diagnostic's supported policy.
    Derived features/edges may change because K is now visible; supplied scope,
    context, time skeleton and other action facts cannot change. Far summaries
    are rejected because their source ranges are not explicit in model inputs;
    moving a target boundary can change the meaning of the same count vector.
    """
    return PrefixPathPair(example, _reveal_prefix(example, prefix_rows), prefix_rows)


def joint_distribution_metrics(history: torch.Tensor, observation: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-row JS, TV and both KL directions on CPU in float64.

    Inputs are normalized log probabilities with identical shape and finite
    support. Mismatched legality or nonfinite/unnormalized predictions raise
    ContractError. Illegal (-inf) classes contribute zero to every metric.
    """
    p_log, q_log = (x.detach().cpu().double() for x in (history, observation))
    if p_log.shape != q_log.shape or p_log.ndim < 2 or not p_log.numel():
        raise ContractError("Path distributions require aligned nonempty rows and classes")
    support = torch.isfinite(p_log)
    if not torch.equal(support, torch.isfinite(q_log)):
        raise ContractError("Path distributions have different legality support")
    for values in (p_log, q_log):
        if torch.isnan(values).any() or torch.isposinf(values).any() or not torch.allclose(
            values.logsumexp(-1), torch.zeros_like(values[..., 0]), atol=1e-5, rtol=0
        ):
            raise ContractError("Path distributions must contain normalized finite probabilities")
    p, q = p_log.exp(), q_log.exp()
    midpoint = torch.logaddexp(p_log, q_log) - math.log(2)
    p_log, q_log, midpoint = (x.masked_fill(~support, 0) for x in (p_log, q_log, midpoint))
    return {
        "js_nats": ((p * (p_log - midpoint) + q * (q_log - midpoint)).sum(-1) / 2).clamp_min(0),
        "tv": (p - q).abs().sum(-1) / 2,
        "kl_history_to_observation_nats": (p * (p_log - q_log)).sum(-1).clamp_min(0),
        "kl_observation_to_history_nats": (q * (q_log - p_log)).sum(-1).clamp_min(0),
    }


def evaluate_path_consistency(model, pairs: list[PrefixPathPair], *, batch_size: int = 8) -> dict:
    """Compare common suffix rows at identical teacher-forced prefixes and legality.

    Restore module modes and leave gradients/parameters untouched. Every record
    contains full-distribution differences and true-target costs on the common
    suffix; averages of row divergences are not block-joint divergences. The
    caller fixes population, base visibility and split indices before evaluation.
    """
    if not pairs or type(batch_size) is not int or batch_size < 1:
        raise ContractError("Path evaluation requires nonempty pairs and a positive batch size")
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    records = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start:start + batch_size]
                history = collate([p.decoder_history for p in chunk])
                observed = collate([p.encoder_observation for p in chunk])
                history_identity, observed_identity = batch_identity(history), batch_identity(observed)
                history_output, observed_output = model(history.to(device)), model(observed.to(device))
                for i, pair in enumerate(chunk):
                    count = len(pair.encoder_observation.targets)
                    lo, hi = pair.prefix_rows, pair.prefix_rows + count
                    p, q = history_output.log_probs[i, lo:hi], observed_output.log_probs[i, :count]
                    metrics = joint_distribution_metrics(p, q)
                    obs = pair.encoder_observation.observation
                    record = {"pair_index": start + i, "prefix_rows": lo, "suffix_rows": count,
                              "row_indices": list(obs.target_indices),
                              "history_decoder_positions": list(range(lo, hi)),
                              "observation_decoder_positions": list(range(count)),
                              "duration_ms": obs.rows[obs.target_indices[-1]].time_ms - obs.rows[obs.target_indices[0]].time_ms,
                              "history_batch_sha256": history_identity, "observation_batch_sha256": observed_identity,
                              "legal_classes": torch.isfinite(p).sum(-1).cpu().tolist()}
                    for name, values in metrics.items():
                        record[name] = values.tolist()
                        record[f"mean_{name}"] = float(values.mean())
                    for name, losses in (("history", history_output.row_nll[i, lo:hi]),
                                         ("observation", observed_output.row_nll[i, :count])):
                        record[f"{name}_row_nll"] = losses.cpu().tolist()
                        record[f"{name}_sequence_nll"] = float(losses.sum())
                        record[f"{name}_mean_row_nll"] = float(losses.mean())
                    records.append(record)
    finally:
        for module, training in modes:
            module.training = training
    return {"policy": PATH_POLICY, "model_policy": model.policy_identity, "nll_unit": "nats", "pairs": records}
