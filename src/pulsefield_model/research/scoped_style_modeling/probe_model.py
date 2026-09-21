"""Reusable temporal encoder and multiscale local/ordered assessment for B and C."""
from dataclasses import replace

import torch
from torch import nn

from .model import (AssessmentHead, ChartEncoder, initialize_model, mlp, packed_gru)
from .temporal import (SPAN_DIM, TEMPORAL_LANE_DIM, TEMPORAL_ROW_DIM, TEMPORAL_RELATION_DIM, TIME_DIM)


class TemporalEncoder(ChartEncoder):
    def __init__(self, config):
        super().__init__(config)
        dim = 2*config.hand_hidden
        self.time_lane = nn.Linear(TEMPORAL_LANE_DIM, config.lane_dim, bias=False)
        self.time_row = nn.Linear(TEMPORAL_ROW_DIM, 2*config.lane_dim+11, bias=False)
        self.time_edge = nn.Linear(TIME_DIM, dim, bias=False)
        self.time_relation = nn.Linear(TEMPORAL_RELATION_DIM, dim, bias=False)

    def forward(self, chart, sidecars):
        lanes = (self.lane(chart.lanes)+self.time_lane(sidecars.lanes)).flatten(-2)
        b, t = lanes.shape[:2]
        values = torch.cat((lanes, chart.rows[:, :, None].expand(-1, -1, 2, -1)), -1)
        values = values+self.time_row(sidecars.rows)[:, :, None]
        values = self.dropout(values.permute(0, 2, 1, 3).reshape(b*2, t, -1))
        h, _ = packed_gru(self.hand, values, chart.lengths.repeat_interleave(2))
        h = h.reshape(b, 2, t, -1).transpose(1, 2)
        extra = self.time_edge(sidecars.edges).index_add(0, chart.relation_edges,
                                                         self.time_relation(sidecars.relations))
        h = self.relations(h, chart, extra)
        valid = torch.arange(t, device=h.device)[None] < chart.lengths.to(h.device)[:, None]
        return h*valid[:, :, None, None]


class MultiscaleComposition(nn.Module):
    """Stride-one shared-hand composition, retaining incoming and every block state."""
    def __init__(self, config, dilations):
        super().__init__()
        dim = 2*config.hand_hidden
        self.blocks = nn.ModuleList(nn.Conv1d(dim, dim, 3, padding=d, dilation=d) for d in dilations)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in dilations)
        self.dropout = nn.Dropout(config.dropout)
        self.mixture = nn.Parameter(torch.zeros(len(dilations)+1))

    def forward(self, h, lengths):
        b, t, _, dim = h.shape
        valid = (torch.arange(t, device=h.device)[None] < lengths.to(h.device)[:, None])[:, :, None, None]
        h = h*valid
        states = [h]
        for block, norm in zip(self.blocks, self.norms):
            values = h.permute(0, 2, 3, 1).reshape(b*2, dim, t)
            values = block(values).reshape(b, 2, dim, t).permute(0, 3, 1, 2)
            h = norm(h+self.dropout(torch.nn.functional.gelu(values)))*valid
            states.append(h)
        return torch.stack(states, 2)

    def shared_memory(self, scales):
        """Concept-independent [batch,event,hand,dim] memory for ordered readout/selector."""
        return (scales*self.mixture.softmax(0)[None, None, :, None, None]).sum(2)


class MultiscaleAssessment(AssessmentHead):
    def __init__(self, config, scales, enhanced):
        super().__init__(config)
        self.span = nn.Linear(SPAN_DIM, config.row_dim, bias=False)
        self.time_metadata = nn.Linear(TEMPORAL_ROW_DIM, config.row_dim, bias=False) if enhanced else None
        self.query = nn.Linear(config.concept_dim, config.row_dim)
        self.response = nn.Linear(config.row_dim, 1)
        # The ordered head retains the R0 initialization. The additional head
        # jointly reads ordered context and all local summaries.
        self.local_head = mlp(4*config.section_hidden+2+scales*(config.row_dim+1), config.head_hidden, 3, config.dropout)

    def forward(self, memory, chart, concepts, scales, sidecars, *, inspect=False):
        ordered = self.summarize(memory, chart, concepts)
        vectors = (self.pair(scales.flatten(-2))+self.pair(scales.flip(-2).flatten(-2)))/2
        vectors = vectors+self.span(sidecars.spans)
        if self.time_metadata is not None:
            vectors = vectors+self.time_metadata(sidecars.rows)[:, :, None]
        responses = self.response(torch.tanh(vectors+self.query(self.concept(concepts))[:, None, None])).squeeze(-1)
        # Only source-event anchors inside the target section can be candidates.
        mask = sidecars.spans[:, :, 0, 7].bool()
        nonempty = mask.any(1)
        scores = responses.masked_fill(~mask[:, :, None], -torch.inf)
        safe = torch.where(nonempty[:, None, None], scores, torch.zeros_like(scores))
        weights = safe.softmax(1)*mask[:, :, None]
        strongest = torch.where(nonempty[:, None], safe.amax(1), torch.zeros_like(safe[:, 0]))
        attended = (weights[:, :, :, None]*vectors).sum(1)
        local = torch.cat((strongest[:, :, None], attended), -1).flatten(1)
        logits = self.head(ordered)+self.local_head(torch.cat((ordered, local), -1))
        if inspect:
            return logits, {'responses': responses, 'weights': weights, 'candidate_mask': mask,
                            'span_features': sidecars.spans}
        return logits


def readout_chart(chart, indices):
    return replace(chart, section_indices=chart.section_indices[indices],
                   section_lengths=chart.section_lengths[indices.cpu()],
                   section_features=chart.section_features[indices], section_events=chart.section_events[indices],
                   duration=chart.duration[indices])


class ProbeModel(nn.Module):
    def __init__(self, config, *, enhanced, multiscale, dilations):
        super().__init__()
        self.enhanced, self.multiscale = enhanced, multiscale
        self.encoder = TemporalEncoder(config) if enhanced else ChartEncoder(config)
        self.composition = MultiscaleComposition(config, dilations) if multiscale else None
        self.assessor = MultiscaleAssessment(config, len(dilations)+1, enhanced) if multiscale else AssessmentHead(config)

    def encode(self, chart, sidecars, frozen=None):
        h = frozen if frozen is not None else (self.encoder(chart, sidecars) if self.enhanced else self.encoder(chart))
        if self.composition is None:
            return h, None
        scales = self.composition(h, chart.lengths)
        return self.composition.shared_memory(scales), scales

    def read(self, encoded, chart, sidecars, concepts, indices, *, inspect=False):
        memory, scales = encoded
        selected_chart = readout_chart(chart, indices)
        if scales is None:
            logits = self.assessor(memory[indices], selected_chart, concepts)
            return (logits, None) if inspect else logits
        side = replace(sidecars, spans=sidecars.spans[indices], rows=sidecars.rows[indices])
        return self.assessor(memory[indices], selected_chart, concepts, scales[indices], side, inspect=inspect)


def initialize_probe(config, name, seed):
    """Explicitly copy every identical component; initialize extras independently."""
    enhanced = name in ('CT', 'CTM')
    multiscale = name in ('R1', 'CM', 'CTM')
    base = initialize_model(config.model, seed, auxiliary=False)
    # One seed per extension family keeps CM/CTM composition and readout equal
    # even though CTM constructs an additional encoder module first.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+10000)
        model = ProbeModel(config.model, enhanced=enhanced, multiscale=multiscale, dilations=config.dilations)
        if multiscale:
            torch.random.default_generator.manual_seed(seed+20000)
            model.composition = MultiscaleComposition(config.model, config.dilations)
            torch.random.default_generator.manual_seed(seed+30000)
            model.assessor = MultiscaleAssessment(config.model, len(config.dilations)+1, False)
            if enhanced:
                torch.random.default_generator.manual_seed(seed+40000)
                model.assessor.time_metadata = nn.Linear(TEMPORAL_ROW_DIM, config.model.row_dim, bias=False)
    state = model.state_dict()
    for key, value in base.state_dict().items():
        if key in state and state[key].shape == value.shape:
            state[key].copy_(value)
    model.load_state_dict(state)
    return model
