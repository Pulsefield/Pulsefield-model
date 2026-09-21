"""Matched early-representation arms for the existing training/evaluation APIs.

These controls isolate adjacency, time coordinates, row composition and source
access. They prescribe neither a research run budget nor an adoption decision.
"""
from dataclasses import asdict, replace
from hashlib import sha256
from typing import Mapping

import torch

from ..scoped_style_modeling.dataset import ContractError
from .local_representation import LOCAL_POLICY, LocalRepresentationConfig, TimeLocalEncoder
from .model import BankPredictor, DECODER_POLICY, LOSS_POLICY, ModelConfig, initialize_comparison
from .representation import ARCHITECTURE, READER_POLICY
from .time_basis import TIME_POLICY, TIME_SCALES_MS


def representation_arms() -> dict[str, LocalRepresentationConfig]:
    """Return independent, named controls; callers select a bounded subset.

    The smooth/row/time/row_time square separates the two computation changes.
    combined adds only direct raw-source access; state adds only prefix state.
    """
    baseline = LocalRepresentationConfig()
    event = replace(baseline, local_operator="event")
    scalar = replace(event, time_basis="scalar")
    smooth = replace(scalar, time_basis="smooth")
    row = replace(smooth, row_interaction=True)
    timed = replace(smooth, local_operator="time")
    row_time = replace(timed, row_interaction=True)
    combined = replace(row_time, source_skip=True)
    return {"baseline": baseline, "event": event, "scalar": scalar, "smooth": smooth,
            "row": row, "time": timed,
            "time_kernel": replace(timed, kernel_condition="time"),
            "action_kernel": replace(timed, kernel_condition="action"),
            "constant_kernel": replace(timed, kernel_condition="constant"),
            "row_time": row_time, "combined": combined,
            "state": replace(combined, state_condition=True)}


class RepresentationPredictor(BankPredictor):
    def __init__(self, name: str, config: ModelConfig, representation: LocalRepresentationConfig, *, access: str):
        super().__init__("composed_all", config)
        if access not in ("all", "H"):
            raise ContractError("Representation access must be all or H")
        self.configuration, self.access, self.representation = name, access, representation
        if representation != LocalRepresentationConfig():
            self.encoder = TimeLocalEncoder(config, representation)

    @property
    def policy_identity(self):
        return {"architecture": ARCHITECTURE if self.representation == LocalRepresentationConfig() else LOCAL_POLICY,
                "reader": READER_POLICY, "access": self.access, "configuration": self.configuration,
                "decoder": DECODER_POLICY, "loss": LOSS_POLICY, "representation": asdict(self.representation),
                "local_policy": LOCAL_POLICY, "time_policy": TIME_POLICY, "time_scales_ms": list(TIME_SCALES_MS)}

    def initialize_untrained(self, seed: int):
        return initialize_representation_comparison({self.configuration: self.representation}, self.config,
                                                    seed, access=self.access)[self.configuration]


def initialize_representation_comparison(arms: Mapping[str, LocalRepresentationConfig],
                                        config: ModelConfig = ModelConfig(), seed: int = 17, *,
                                        access: str = "all") -> dict[str, RepresentationPredictor]:
    """Match every unchanged tensor to composed_all, including both readers' width.

    Added modules use a private seed per module path, so selecting/reordering
    arms or toggling another intervention cannot alter their initialization.
    Shape-changing scalar/smooth projections are separately initialized.
    """
    if not arms or any(not isinstance(name, str) or not name or not isinstance(options, LocalRepresentationConfig)
                       for name, options in arms.items()):
        raise ContractError("Representation comparison requires named LocalRepresentationConfig arms")
    baseline = initialize_comparison(config, seed)["composed_all"].state_dict()
    result = {}
    with torch.random.fork_rng(devices=[]):
        for name, options in arms.items():
            torch.random.default_generator.manual_seed(seed)
            model = RepresentationPredictor(name, config, options, access=access)
            for path, module in model.named_modules():
                if hasattr(module, "reset_parameters"):
                    module_seed = int.from_bytes(sha256(f"{seed}:representation:{path}".encode()).digest()[:8], "little")
                    torch.random.default_generator.manual_seed(module_seed)
                    module.reset_parameters()
            state = model.state_dict()
            for key, value in baseline.items():
                if key in state and state[key].shape == value.shape:
                    state[key] = value.detach().clone()
            model.load_state_dict(state)
            result[name] = model
    return result


def train_representation_step(models, optimizers, sampler, *, blocks: int, gradient_cap: float = 1.0):
    """Use the same sampled blocks, visibility, legality and risk for selected arms."""
    from .comparison import train_paired_step
    if not models or any(not isinstance(m, RepresentationPredictor) for m in models.values()):
        raise ContractError("Representation updates require named representation predictors")
    if len({m.access for m in models.values()}) != 1:
        raise ContractError("Compare representation interventions at the same reader access")
    return train_paired_step(models, optimizers, sampler, blocks=blocks, gradient_cap=gradient_cap,
                             configurations=tuple(models))
