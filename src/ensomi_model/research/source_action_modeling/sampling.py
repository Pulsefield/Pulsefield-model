"""Recorded uniform group/context/feasible-scale/start sampling without labels."""
from dataclasses import asdict, dataclass
import random

from ..scoped_style_modeling.dataset import ContractError, canonical_json, digest
from ..scoped_style_modeling.replay import PreparedChart
from .actions import ACTION_SCHEMA, source_actions
from .observation import BLOCK_SIZES, EventBlock, ViewPolicy, declared_entering_occupancy, observe, paired_views

SAMPLING_POLICY = "uniform-group/context/feasible-{4,16,64}/event-start-v1"
SPLIT_SHA256 = "15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a"
VIEWS = ("near", "detailed", "coarse")


@dataclass(frozen=True)
class TrainingContext:
    group_id: str
    key: str
    chart: PreparedChart

    def __post_init__(self):
        for row in self.chart.inputs.rows:
            source_actions(row)

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
        self.signature = digest(canonical_json({"action_schema": ACTION_SCHEMA,
            "contexts": [(c.group_id, c.key, c.event_count) for c in self.contexts]}).encode())
        self.rng = random.Random(seed)
        self.position = 0

    def draw(self, count=1):
        if type(count) is not int or count < 1:
            raise ContractError("Sample count must be a positive integer")
        examples, records = [], []
        for _ in range(count):
            group = self.rng.choice(sorted(self.groups))
            context = self.rng.choice(self.groups[group])
            scales = [s for s in BLOCK_SIZES if s <= context.event_count]
            size = self.rng.choice(scales)
            start = self.rng.randrange(context.event_count - size + 1)
            item = observe(context.chart, EventBlock(start, size),
                           entering_occupancy=declared_entering_occupancy(context.chart))
            obs = item.observation
            examples.append(item)
            records.append({"position": self.position, "group_id": group, "context_key": context.key,
                            "event_start": start, "rows": size, "attack_group_span": item.attack_group_span,
                            "sampling_probability": 1 / (len(self.groups) * len(self.groups[group]) *
                                                         len(scales) * (context.event_count - size + 1)),
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


class PairedBlockSampler:
    """One target draw exposes every evaluated view to every model configuration."""
    sampling_policy = SAMPLING_POLICY

    def __init__(self, contexts: list[TrainingContext], seed: int = 17, *, policy: ViewPolicy = ViewPolicy()):
        self.blocks = BlockSampler(contexts, seed)
        self.policy = policy

    def draw(self, count=1):
        examples, records = self.blocks.draw(count)
        paired = [paired_views(item, self.policy) for item in examples]
        return paired, [{**r, "views": list(VIEWS), "view_policy": asdict(self.policy)} for r in records]

    def state_dict(self):
        return {"policy": "paired-all-three-views-v1", "views": list(VIEWS), "view_policy": asdict(self.policy),
                "blocks": self.blocks.state_dict()}

    def load_state_dict(self, state):
        if state["policy"] != "paired-all-three-views-v1" or state["views"] != list(VIEWS) or state["view_policy"] != asdict(self.policy):
            raise ContractError("Paired sampler view policy differs from snapshot")
        self.blocks.load_state_dict(state["blocks"])
