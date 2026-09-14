"""Modest frozen-bank assessment on original, eligible human review inputs."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..scoped_style_modeling.dataset import ASSESSMENTS, CONCEPTS, ContractError
from ..scoped_style_modeling.model import mlp
from ..scoped_style_modeling.probe_data import TARGET_POLICY, input_identity, sampled_cells, select_targets
from ..scoped_style_modeling.probe_metrics import prediction_rows, views, paired_changes
from .observation import observe_complete, declared_entering_occupancy
from .representation import RepresentationBank, level_descriptors, valid_rows
from .tensors import ObservationTensors, collate_observations

SEMANTIC_POLICY = f"source-action/human-only/{TARGET_POLICY}/complete-1x-v1"


def human_targets(records, issues, *, split: str) -> list[int]:
    """Apply the effective-human conflict/confidence policy, then exclude fallback."""
    if split not in ("train", "validation"):
        raise ContractError("Semantic probes admit only human training or validation inputs")
    selected, _ = select_targets(records, issues)
    return [i for i in selected if records[i]["layer"] == "human" and records[i]["split"] == split]


@dataclass(frozen=True)
class SemanticBatch:
    observation: ObservationTensors
    concepts: Tensor
    targets: Tensor
    input_indices: Tensor

    def to(self, device):
        return SemanticBatch(self.observation.to(device), self.concepts.to(device), self.targets.to(device),
                             self.input_indices.to(device))


class SemanticCorpus:
    """Thin adapter over verified PreparedCorpus; no altered inputs inherit a label.

    The frozen annotation adapter currently supports 1x only. A new rate needs
    a new target/input policy, rather than silently rescaling human judgments.
    """
    def __init__(self, corpus, issues):
        from .sampling import SPLIT_SHA256
        if corpus.summary["split_sha256"] != SPLIT_SHA256:
            raise ContractError("Semantic corpus requires the pinned source-group split")
        self.corpus, self.records = corpus, corpus.records
        self.train_indices = human_targets(self.records, issues, split="train")
        self.validation_indices = human_targets(self.records, issues, split="validation")
        self.eligible = set(self.train_indices + self.validation_indices)

    def batch(self, indices: list[int]) -> SemanticBatch:
        if not indices or not set(indices) <= self.eligible:
            raise ContractError("Semantic batches require eligible human cells")
        observations, inverse, positions = [], [], {}
        for i in indices:
            row = self.records[i]
            key = input_identity(row)
            if key not in positions:
                chart, _ = self.corpus.chart(row["chart_key"], row["chart_sha256"])
                if (vars(chart.inputs.scope), vars(chart.inputs.context)) != (row["scope"], row["context"]):
                    raise ContractError("Semantic scope/context differs from the original human judgment")
                positions[key] = len(observations)
                observations.append(observe_complete(chart, entering_occupancy=declared_entering_occupancy(chart)))
            inverse.append(positions[key])
        return SemanticBatch(collate_observations(observations),
                             torch.tensor([CONCEPTS.index(self.records[i]["concept"]) for i in indices]),
                             torch.tensor([ASSESSMENTS.index(self.records[i]["assessment"]) for i in indices]),
                             torch.tensor(inverse))


class ConceptReader(nn.Module):
    """Concept attention plus section-anchor mean, duration and a shared 3-class MLP.

    Hands combine symmetrically only here. Each concept receives an independent
    absent/supporting/prominent distribution; no across-concept normalization.
    """
    def __init__(self, access: str):
        super().__init__()
        if access not in ("H", "all"):
            raise ContractError("Concept access must be H or all")
        self.access = access
        self.fitted = False
        self.concept = nn.Embedding(len(CONCEPTS), 16)
        self.query = nn.Linear(17, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.head = mlp(2 * 64 + 16 + 2, 64, 3)

    @property
    def policy_identity(self):
        return {"targets": SEMANTIC_POLICY, "reader": "concept-attention/anchor-mean/64-v1", "access": self.access}

    def forward(self, bank: RepresentationBank, concepts: Tensor, input_indices: Tensor) -> Tensor:
        levels, values = bank.select(self.access)
        # Average hands after source and ordered composition have represented
        # both own/other arrangements. The encoder itself never pools hands.
        values = values[input_indices].mean(3)
        b, t, count, dim = values.shape
        obs = bank.observation
        anchors = (valid_rows(obs) & obs.rows[:, :, 6].bool())[input_indices]
        duration = obs.duration[input_indices]
        concept = self.concept(concepts)
        q = self.query(torch.cat((concept, duration), -1))
        keys = self.key(values) + level_descriptors(levels, dim, values)[None, None]
        scores = (keys * q[:, None, None]).sum(-1).flatten(1) / math.sqrt(dim)
        mask = anchors[:, :, None].expand(-1, -1, count).flatten(1)
        nonempty = mask.any(1)
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = torch.where(nonempty[:, None], scores, torch.zeros_like(scores)).softmax(-1) * mask
        attended = (self.value(values).reshape(b, t * count, dim) * weights[:, :, None]).sum(1)
        mean = (values * anchors[:, :, None, None]).sum((1, 2)) / (anchors.sum(1, keepdim=True) * count).clamp_min(1)
        return self.head(torch.cat((attended, mean, concept, duration, (~nonempty).to(values.dtype)[:, None]), -1))


def initialize_readout(access: str, seed: int = 17) -> ConceptReader:
    """Use identical weights for H-only and all-level access without consuming RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 30000)
        return ConceptReader(access)


def fit_readout(model, corpus: SemanticCorpus, *, steps: int, batch_size: int, learning_rate: float,
                seed: int = 17, on_step=None) -> tuple[ConceptReader, dict]:
    """Fit only the readout, with fixed concept/group/cell draws and encoder eval.

    Requires explicit update exposure and learning rate. The encoder's parameters,
    buffers, gradients and training mode are preserved. Training consumes only
    eligible human targets; all five concepts must have training support.
    Optional on_step(step, readout, optimizer) runs after each update for bounded
    execution and checkpoints; callback exceptions propagate after mode restore.
    """
    if any(type(v) is not int or v < 1 for v in (steps, batch_size)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ContractError("Readout fitting requires positive steps, batch size and learning rate")
    indices = sampled_cells(corpus.records, corpus.train_indices, seed, steps * batch_size)
    device = next(model.parameters()).device
    readout = initialize_readout(model.access, seed).to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=learning_rate)
    mode = model.encoder.training
    model.encoder.eval()
    losses = []
    try:
        for start in range(0, len(indices), batch_size):
            batch = corpus.batch(indices[start:start + batch_size]).to(device)
            with torch.no_grad():
                bank = model.encoder(batch.observation)
            optimizer.zero_grad(set_to_none=True)
            logits = readout(bank, batch.concepts, batch.input_indices)
            loss = nn.functional.cross_entropy(logits, batch.targets)
            if not torch.isfinite(loss):
                raise ContractError("Nonfinite frozen-probe loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if on_step is not None:
                on_step(start // batch_size + 1, readout, optimizer)
    finally:
        model.encoder.train(mode)
    readout.fitted = True
    return readout, {"policy": SEMANTIC_POLICY, "seed": seed, "sampled_indices": indices, "losses": losses,
                     "parameters": sum(p.numel() for p in readout.parameters())}


def evaluate_readout(model, readout: ConceptReader, corpus: SemanticCorpus, *, batch_size: int = 16) -> dict:
    """Report human validation NLL, ranking, 0.5 presence and paired Jack/Stream cases."""
    if not corpus.validation_indices or type(batch_size) is not int or batch_size < 1:
        raise ContractError("Semantic evaluation requires human validation cells and positive batch size")
    if model.access != readout.access:
        raise ContractError("Semantic reader access differs from its predictor configuration")
    device = next(model.parameters()).device
    mode, head_mode = model.encoder.training, readout.training
    model.encoder.eval()
    readout.eval()
    predictions = []
    try:
        with torch.no_grad():
            for start in range(0, len(corpus.validation_indices), batch_size):
                indices = corpus.validation_indices[start:start + batch_size]
                batch = corpus.batch(indices).to(device)
                logits = readout(model.encoder(batch.observation), batch.concepts, batch.input_indices)
                predictions.extend(prediction_rows([corpus.records[i] for i in indices], logits.cpu()))
    finally:
        model.encoder.train(mode)
        readout.train(head_mode)
    return {"policy": readout.policy_identity, "metrics": views(predictions), "predictions": predictions}


def fit_matched_probes(model, corpus: SemanticCorpus, *, encoder_seed: int, readout_seed: int,
                       steps: int, batch_size: int, learning_rate: float) -> dict:
    """Fit identical heads/draws on trained and matched untrained bank access.

    Supply the seed used to initialize the predictor. Both fits use complete
    original review contexts. Results measure frozen reuse, not generation.
    """
    untrained = model.initialize_untrained(encoder_seed).to(next(model.parameters()).device)
    result = {}
    for name, candidate in (("trained", model), ("untrained", untrained)):
        head, fit = fit_readout(candidate, corpus, steps=steps, batch_size=batch_size,
                                learning_rate=learning_rate, seed=readout_seed)
        result[name] = {"readout": head, "fit": fit, **evaluate_readout(candidate, head, corpus, batch_size=batch_size)}
    result["paired"] = paired_changes(result["untrained"]["predictions"], result["trained"]["predictions"])
    return result
