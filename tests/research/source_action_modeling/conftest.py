import pytest
import torch
import random

import numpy as np

from ensomi_model.research.scoped_style_modeling.dataset import Interval, NoteRef
from ensomi_model.research.scoped_style_modeling.replay import mirror_objects, prepare_chart
from ensomi_model.research.source_action_modeling.observation import EventBlock, declared_entering_occupancy, observe


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def fixture_chart(*, changed=False, mirror=False, long=False):
    def note(line, lane, start, end=None):
        return NoteRef(line, lane, "normal" if end is None else "long", start, start if end is None else end)
    objects = (note(1, 0, -100, 300 if changed else 200), note(2, 1, 0, 200 if changed else 100),
               note(3, 1, 300 if changed else 200, 400), note(4, 2, 100), note(5, 3, 200),
               note(6, 2, 400 if changed else 300, 500), note(7, 0, 400), note(8, 3, 500),
               note(9, 1, 600), note(10, 0, 700, 2200), note(11, 2, 800))
    if long:
        objects += tuple(note(i + 20, 1 + i % 3, 900 + i * 50) for i in range(25))
    if mirror:
        objects = mirror_objects(objects)
    return prepare_chart(objects, Interval(100, 2150 if long else 850), Interval(0, 2200 if long else 900))


def example(*, start=1, size=4, **kwargs):
    chart = fixture_chart(**kwargs)
    return observe(chart, EventBlock(start, size), entering_occupancy=declared_entering_occupancy(chart))
