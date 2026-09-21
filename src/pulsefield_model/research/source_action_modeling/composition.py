"""Equal-parameter local/relation order comparison with chronological storage.

S1..S4 identify computation positions, not semantic factors or local scales.
The two arms have identical tensors, reader identities and source information.
"""
from ..scoped_style_modeling.dataset import ContractError
from .local_representation import TimeLocalEncoder
from .model import ModelConfig
from .representation import COMPOSITION_LEVELS, RepresentationBank
from .representation_experiments import RepresentationPredictor, initialize_representation_comparison, representation_arms

ORDERS = {"serial": ("L1", "L2", "L3", "R"), "interleaved": ("L1", "L2", "R", "L3")}
COMPOSITION_POLICY = "same-modules/LLLR-vs-LLRL/chronological-bank-v1"


class CompositionEncoder(TimeLocalEncoder):
    def __init__(self, config, representation, arm, *, relation_matching="additive"):
        super().__init__(config, representation, relation_matching=relation_matching)
        self.arm = arm

    def forward(self, observation):
        states = self.computation_states(observation, relation_after=3 if self.arm == "serial" else 2)
        return RepresentationBank(COMPOSITION_LEVELS,
                                  (*states, self.contextual_state(states[-1], observation)), observation)


class CompositionPredictor(RepresentationPredictor):
    @property
    def policy_identity(self):
        return {**super().policy_identity, "architecture": COMPOSITION_POLICY,
                "order": list(ORDERS[self.encoder.arm]), "storage": list(COMPOSITION_LEVELS),
                "post_relation_support": "relation-conditioned; not strictly event-local"}

    def initialize_untrained(self, seed):
        return initialize_composition(self.config, seed, access=self.access)[self.configuration]


def initialize_composition(config: ModelConfig = ModelConfig(), seed: int = 17, *, access="all"):
    """Match every tensor to the combined time/row/source-access arm.

    Reordering the modules is the only intervention. Both variants retain three
    local kernels, one relation block, six readable states and the same decoder.
    """
    import torch
    if access not in ("all", "H"):
        raise ContractError("Composition access must be all or H")
    representation = representation_arms()["combined"]
    baseline = initialize_representation_comparison({"combined": representation}, config, seed, access=access)["combined"]
    models = {}
    with torch.random.fork_rng(devices=[]):
        for arm in ORDERS:
            model = CompositionPredictor(arm, config, representation, access=access)
            model.encoder = CompositionEncoder(config, representation, arm)
            model.load_state_dict(baseline.state_dict())
            models[arm] = model
    return models
