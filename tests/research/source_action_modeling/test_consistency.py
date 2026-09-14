from dataclasses import fields, replace
import math

import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from pulsefield_model.research.source_action_modeling.consistency import (
    PrefixPathPair, evaluate_path_consistency, joint_distribution_metrics, prefix_path_pair,
)
from pulsefield_model.research.source_action_modeling.model import initialize_comparison, initialize_model
from pulsefield_model.research.source_action_modeling.observation import ViewPolicy, advance_occupancy, paired_views, visible_states
from pulsefield_model.research.source_action_modeling.tensors import collate
from .conftest import example

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("view", ["near", "detailed"])
@pytest.mark.parametrize("prefix_rows", [1, 3])
def test_revealing_prefix_freezes_other_facts_and_common_entry(view, prefix_rows):
    original = paired_views(example(), ViewPolicy(0))[view]
    original = replace(original, observation=replace(original.observation, summaries=()))
    pair = prefix_path_pair(original, prefix_rows=prefix_rows)
    revealed = pair.encoder_observation
    a, b = original.observation, revealed.observation
    assert (a.scope, a.context, a.entering_occupancy, a.summaries) == (b.scope, b.context, b.entering_occupancy, b.summaries)
    prefix = dict(zip(a.target_indices[:prefix_rows], original.targets[:prefix_rows]))
    for i, (left, right) in enumerate(zip(a.rows, b.rows)):
        assert right == (replace(left, actions=prefix[i]) if i in prefix else left)
    assert revealed.targets == original.targets[prefix_rows:]
    entry = original.query_entering_occupancy
    for actions in original.targets[:prefix_rows]:
        entry = advance_occupancy(entry, actions)
    assert revealed.query_entering_occupancy == entry
    if view == "detailed" and prefix_rows == 1:
        # Detailed encoder facts could reveal more than the deliberately common
        # near-prefix legality. Moving K must not strengthen that decoder mask.
        assert entry != visible_states(b)[2]
    changed = replace(original, targets=(*original.targets[:-1], (0, 1, 0, 0)))
    assert changed.targets != original.targets
    other = prefix_path_pair(changed, prefix_rows=prefix_rows)
    for left, right in ((pair.decoder_history, other.decoder_history),
                        (pair.encoder_observation, other.encoder_observation)):
        x, y = collate([left]), collate([right])
        for field in fields(x.observation):
            assert torch.equal(getattr(x.observation, field.name), getattr(y.observation, field.name)), field.name
        assert torch.equal(x.queries.entering_occupancy, y.queries.entering_occupancy)


def test_path_pair_rejects_recomputed_views_and_extra_information():
    base = paired_views(example(), ViewPolicy(0))["detailed"]
    with pytest.raises(ContractError, match="unsummarized"):
        prefix_path_pair(base, prefix_rows=1)
    summaries = base.observation.summaries
    base = replace(base, observation=replace(base.observation, summaries=()))
    pair = prefix_path_pair(base, prefix_rows=1)
    observed = pair.encoder_observation
    with pytest.raises(ContractError, match="facts"):
        PrefixPathPair(base, replace(observed, query_entering_occupancy=(False,) * 4), 1)
    with pytest.raises(ContractError, match="facts"):
        PrefixPathPair(base, replace(observed, observation=replace(observed.observation, summaries=summaries)), 1)
    unsummarized = replace(observed, observation=replace(observed.observation, summaries=()), query_entering_occupancy=None)
    reselected = paired_views(unsummarized, ViewPolicy(0))["detailed"]
    with pytest.raises(ContractError, match="facts"):
        PrefixPathPair(base, reselected, 1)
    for size in (0, 4, -1, True):
        with pytest.raises(ContractError, match="prefix and suffix"):
            prefix_path_pair(base, prefix_rows=size)


def test_full_distribution_diagnostic_detects_changes_missed_by_target_nll():
    p = torch.tensor([[.5, .4, .1, 0.]], dtype=torch.float64).log()
    q = torch.tensor([[.5, .1, .4, 0.]], dtype=torch.float64).log()
    assert p[0, 0] == q[0, 0]
    result = joint_distribution_metrics(p, q)
    assert result["tv"].item() == pytest.approx(.3)
    assert result["js_nats"].item() == pytest.approx(.4 * math.log(1.6) + .1 * math.log(.4))
    assert result["kl_history_to_observation_nats"].item() == pytest.approx(.3 * math.log(4))
    assert all(value.item() == 0 for value in joint_distribution_metrics(p, p).values())
    with pytest.raises(ContractError, match="legality"):
        joint_distribution_metrics(p, torch.full_like(p, .25).log())
    with pytest.raises(ContractError, match="normalized"):
        joint_distribution_metrics(p, q + 1)


def test_path_pair_restricts_query_entry_to_base_observation():
    base = example()
    base = replace(base, observation=replace(base.observation, entering_occupancy=(None,) * 4))
    observed_entry = visible_states(base.observation)[2]
    assert observed_entry[0] is None
    extra_entry = (True, *observed_entry[1:])
    extra = replace(base, query_entering_occupancy=extra_entry)
    # Both entry encodings lead to the same legal mask. The diagnostic still
    # requires known entry values to follow from the base observation alone.
    a, b = extra_entry, observed_entry
    for actions in base.targets[:2]:
        a, b = advance_occupancy(a, actions), advance_occupancy(b, actions)
    assert a == b
    with pytest.raises(ContractError, match="external entry"):
        prefix_path_pair(extra, prefix_rows=2)
    assert prefix_path_pair(base, prefix_rows=2).encoder_observation.query_entering_occupancy == b


@pytest.mark.parametrize("device", DEVICES)
def test_path_evaluation_uses_suffix_alignment_and_restores_models_and_gradients(device):
    detailed = paired_views(example(), ViewPolicy(0))["detailed"]
    detailed = replace(detailed, observation=replace(detailed.observation, summaries=()))
    pairs = [prefix_path_pair(example(), prefix_rows=1),
             prefix_path_pair(example(long=True, size=16), prefix_rows=3),
             prefix_path_pair(detailed, prefix_rows=2)]
    models = {"reference": initialize_model(), **initialize_comparison()}
    for model in models.values():
        model.to(device).train()
        model.encoder.eval()
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        before = {n: t.clone() for n, t in model.state_dict().items()}
        result = evaluate_path_consistency(model, pairs, batch_size=2)
        assert model.training and not model.encoder.training
        assert torch.equal(parameter.grad, torch.ones_like(parameter))
        assert all(torch.equal(t, model.state_dict()[n]) for n, t in before.items())
        for record, pair in zip(result["pairs"], pairs):
            length = len(pair.encoder_observation.targets)
            assert len(record["js_nats"]) == length
            assert record["history_decoder_positions"][0] == pair.prefix_rows
            assert record["observation_decoder_positions"][0] == 0
            assert all(0 <= value <= math.log(2) + 1e-6 for value in record["js_nats"])
            for route in ("history", "observation"):
                assert record[f"{route}_sequence_nll"] == pytest.approx(sum(record[f"{route}_row_nll"]), rel=1e-6)
                assert record[f"{route}_mean_row_nll"] == pytest.approx(record[f"{route}_sequence_nll"] / length, rel=1e-6)


def test_matching_information_does_not_assert_the_model_is_already_consistent():
    model = initialize_model()
    pair = prefix_path_pair(example(), prefix_rows=1)
    result = evaluate_path_consistency(model, [pair])["pairs"][0]
    assert result["mean_js_nats"] > 0
    assert result["history_sequence_nll"] != result["observation_sequence_nll"]
