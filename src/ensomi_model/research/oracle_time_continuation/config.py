"""Validated model and memory capacities for the oracle-time research backbone."""
from dataclasses import dataclass

from ..scoped_style_modeling.dataset import ContractError


@dataclass(frozen=True)
class BackboneConfig:
    hidden: int = 128
    heads: int = 4
    temporal_layers: int = 2
    temporal_expansion: int = 1
    temporal_bias_hidden: int | None = None
    time_lookahead_rows: int = 0
    clock_readout_hidden: int = 0
    recent: int = 512
    coarse_group: int = 16
    coarse_capacity: int = 64
    attacks_per_lane: int = 12
    releases_per_lane: int = 4
    coupling_rank: int = 16
    max_chunk: int = 128

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name == 'clock_readout_hidden':
                if type(value) is not int or value < 0:
                    raise ContractError('clock_readout_hidden must be a nonnegative integer')
                continue
            if name == 'time_lookahead_rows':
                if type(value) is not int or not 0 <= value <= 16:
                    raise ContractError('time_lookahead_rows must be an integer in [0,16]')
                continue
            if name == 'temporal_bias_hidden' and value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ContractError(f"{name} must be a positive integer")
        if self.hidden % self.heads:
            raise ContractError("hidden must be divisible by heads")
        if self.max_chunk > 128:
            raise ContractError("Differentiable chunks support at most 128 rows")

    @property
    def recent_capacity(self) -> int:
        return self.recent + self.coarse_group - 1

    @property
    def temporal_hidden(self) -> int:
        return self.hidden * self.temporal_expansion

    @property
    def relation_capacity(self) -> int:
        return 4 * (self.attacks_per_lane + self.releases_per_lane) + 4
