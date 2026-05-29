from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from pulsefield_model.models.mapper.shared.batch import MapperTokenContract as _MapperTokenContract
from pulsefield_model.models.mapper.shared.loss import (
    MapperTupleLossConfig as _MapperTupleLossConfig,
    MapperTupleLossOutput as _MapperTupleLossOutput,
    MapperTupleModelLoss as _MapperTupleModelLoss,
)

from .vocab import MapperV21Vocab as _MapperV21Vocab


@dataclass(frozen=True)
class MapperV21LossConfig(_MapperTupleLossConfig):
    pass


@dataclass(frozen=True)
class MapperV21LossOutput(_MapperTupleLossOutput):
    pass


class MapperV21ModelLoss(_MapperTupleModelLoss):
    def __init__(self, config: MapperV21LossConfig | None = None, *, vocab: _MapperV21Vocab | None = None) -> None:
        resolved_config = MapperV21LossConfig() if config is None else config
        resolved_vocab = _MapperV21Vocab() if vocab is None else vocab
        super().__init__(
            resolved_config,
            vocab=resolved_vocab,
            token_contract=_MapperTokenContract(
                name="v2.1",
                vocab=resolved_vocab,
                requires_sparse_lane_state=True,
                uses_chart_end_for_terminal_windows=True,
            ),
        )
        self.config = resolved_config
        self.vocab = resolved_vocab

    def forward(self, output: Any, batch: Mapping[str, torch.Tensor]) -> MapperV21LossOutput:
        loss = super().forward(output, batch)
        return MapperV21LossOutput(
            total_loss=loss.total_loss,
            token_loss=loss.token_loss,
            ln_close_loss=loss.ln_close_loss,
            density_loss=loss.density_loss,
            adapter_reg_loss=loss.adapter_reg_loss,
            metrics=loss.metrics,
            metric_numerators=loss.metric_numerators,
            metric_denominators=loss.metric_denominators,
        )


__all__ = [
    "MapperV21LossConfig",
    "MapperV21LossOutput",
    "MapperV21ModelLoss",
]
