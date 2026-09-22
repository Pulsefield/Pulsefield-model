"""Query–relation score comparison at one fixed backbone schedule.

Both arms retain the combined representation, bias/value paths and decoder.
Only the query arm owns W_rel; all common tensors start identically.
"""
import torch

from ..scoped_style_modeling.dataset import ContractError
from .composition import CompositionEncoder, CompositionPredictor, ORDERS, initialize_composition
from .model import ModelConfig
from .representation_experiments import representation_arms

MATCHING_ARMS = ("additive", "query")


class RelationMatchingPredictor(CompositionPredictor):
    def __init__(self, matching, config, *, backbone_schedule, access):
        representation = representation_arms()["combined"]
        super().__init__(matching, config, representation, access=access)
        self.encoder = CompositionEncoder(config, representation, backbone_schedule, relation_matching=matching)

    @property
    def policy_identity(self):
        return {**super().policy_identity, "architecture": "fixed-order/query-relation-score-v1",
                "relation_matching": self.configuration, "backbone_schedule": self.encoder.arm,
                "relation_descriptor": "edge-projection+summed-relation-mlp+optional-extra",
                "relation_key_initialization": "zero" if self.configuration == "query" else "absent"}

    def initialize_untrained(self, seed):
        return initialize_relation_matching(self.config, seed, backbone_schedule=self.encoder.arm,
                                             access=self.access)[self.configuration]


def initialize_relation_matching(config: ModelConfig = ModelConfig(), seed: int = 17, *,
                                 backbone_schedule="serial", access="all"):
    """Copy every common tensor from the existing four-action composition model.

    The query arm adds one zero 64x64 matrix. Caller RNG and common initialization
    are independent of that addition. Neither arm imports training exposure.
    """
    if backbone_schedule not in ORDERS:
        raise ContractError("Relation matching requires one serial or interleaved backbone schedule")
    baseline = initialize_composition(config, seed, access=access)[backbone_schedule].state_dict()
    models = {}
    with torch.random.fork_rng(devices=[]):
        for matching in MATCHING_ARMS:
            model = RelationMatchingPredictor(matching, config, backbone_schedule=backbone_schedule, access=access)
            state = model.state_dict()
            state.update(baseline)
            model.load_state_dict(state)
            models[matching] = model
    return models
