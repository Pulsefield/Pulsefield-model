"""Recorded uniform group/context/feasible-scale/start sampling without labels."""
from dataclasses import dataclass
import random

from ..scoped_style_modeling.dataset import ContractError, canonical_json, digest
from ..scoped_style_modeling.replay import PreparedChart
from .observation import BLOCK_SIZES, EventBlock, declared_entering_occupancy, observe

SAMPLING_POLICY = "uniform-group/context/feasible-{4,16,64}/event-start-v1"


@dataclass(frozen=True)
class TrainingContext:
    group_id: str
    key: str
    chart: PreparedChart

    @property
    def event_count(self):
        return sum(row.phase == "source" for row in self.chart.inputs.rows)


class BlockSampler:
    """Sample with replacement; feasible scales are uniform within each context.

    Context identities and source event counts pin the population. Sampling
    reads event positions, never lane actions, annotation labels or attack ranks.
    """
    def __init__(self, contexts: list[TrainingContext], seed: int = 17):
        self.contexts = sorted(contexts, key=lambda c: (c.group_id, c.key))
        if not self.contexts or len({c.key for c in self.contexts}) != len(self.contexts):
            raise ContractError("Sampling requires nonempty, distinct context identities")
        if any(c.event_count < min(BLOCK_SIZES) for c in self.contexts):
            raise ContractError("Every sampled context must contain at least four source-event rows")
        self.groups = {}
        for c in self.contexts:
            self.groups.setdefault(c.group_id, []).append(c)
        self.signature = digest(canonical_json([(c.group_id, c.key, c.event_count) for c in self.contexts]).encode())
        self.rng = random.Random(seed)
        self.position = 0

    def draw(self, count=1):
        if type(count) is not int or count < 1:
            raise ContractError("Sample count must be a positive integer")
        examples, records = [], []
        for _ in range(count):
            group = self.rng.choice(sorted(self.groups))
            context = self.rng.choice(self.groups[group])
            size = self.rng.choice([s for s in BLOCK_SIZES if s <= context.event_count])
            start = self.rng.randrange(context.event_count - size + 1)
            item = observe(context.chart, EventBlock(start, size),
                           entering_occupancy=declared_entering_occupancy(context.chart))
            obs = item.observation
            examples.append(item)
            records.append({"position": self.position, "group_id": group, "context_key": context.key,
                            "event_start": start, "rows": size, "attack_group_span": item.attack_group_span,
                            "duration_ms": obs.rows[obs.target_indices[-1]].time_ms - obs.rows[obs.target_indices[0]].time_ms})
            self.position += 1
        return examples, records

    def state_dict(self):
        return {"policy": SAMPLING_POLICY, "population": self.signature, "position": self.position,
                "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if state["policy"] != SAMPLING_POLICY or state["population"] != self.signature:
            raise ContractError("Sampler policy or context population differs from snapshot")
        if type(state["position"]) is not int or state["position"] < 0:
            raise ContractError("Invalid snapshot sampling position")
        self.rng.setstate(state["rng"])
        self.position = state["position"]
