"""Scoped supervision from the existing mania strain calculation.

This is an offline target/readout, not a causal player frontier or an official
fragment star rating. Construct it from complete source objects: slicing first
would erase incoming strain and change LN relationships. Known tail endpoints
are used by the underlying difficulty calculation, so this trace must not be
fed back as if those endpoints were known during incremental generation.
"""
from dataclasses import dataclass

import numpy as np

from ...osu_core.difficulty import ManiaStrain, create_difficulty_hit_objects


SPEC = 'full-prefix-mania-strain-scope-v1'


@dataclass(frozen=True)
class ScopeStrainTrace:
    times: np.ndarray
    individual: np.ndarray
    overall: np.ndarray

    @classmethod
    def from_objects(cls, objects):
        """Replay complete native-1.0x 4K objects using the shared strain owner."""
        skill = ManiaStrain(total_columns=4)
        times, individual, overall = [], [], []
        for obj in create_difficulty_hit_objects(list(objects), 4, 1.):
            skill.process(obj)
            times.append(obj.start_time)
            individual.append(skill.highest_individual_strain)
            overall.append(skill.overall_strain)
        return cls(*(np.asarray(x, dtype=np.float64) for x in (times, individual, overall)))

    def at(self, times):
        """Right-continuous strain; zero before the first processed object."""
        times = np.asarray(times, dtype=np.float64)
        result = np.zeros_like(times)
        if not len(self.times):
            return result
        index = np.searchsorted(self.times, times, side='right')-1
        valid = index >= 0
        delta = (times[valid]-self.times[index[valid]])/1000
        result[valid] = (self.individual[index[valid]]*ManiaStrain.individual_decay_base**delta
                         + self.overall[index[valid]]*ManiaStrain.overall_decay_base**delta)
        return result

    def peaks(self, start_ms, end_ms):
        """Peak per at-most-400ms cell, aligned to this half-open scope.

        Each cell includes the inherited decaying state and only heads inside
        that cell. Neither a future head nor a peak before the scope is copied
        across its boundary. The final partial cell is retained.
        """
        if end_ms <= start_ms:
            raise ValueError('Difficulty scope must have positive duration')
        edges = np.r_[np.arange(start_ms, end_ms, ManiaStrain.section_length), end_ms]
        peaks = self.at(edges[:-1])
        indices = np.searchsorted(self.times, edges, side='left')
        values = self.individual+self.overall
        for i, (a, b) in enumerate(zip(indices, indices[1:])):
            if b > a:
                peaks[i] = max(peaks[i], float(values[a:b].max()))
        return peaks

    def level(self, start_ms, end_ms):
        """Duration-normalized weighted peak level in approximate star units.

        The 0.018 multiplier and 0.9 rank decay come from the shared whole-chart
        rating. Dividing by 1-0.9**cell_count removes the small-scope truncation
        of the weight sum. A constant peak has the same level at every duration;
        more variable passages can still differ. This is a declared proxy,
        especially incomplete for release execution and coordination.
        """
        peaks = np.sort(self.peaks(start_ms, end_ms))[::-1]
        decay = ManiaStrain.decay_weight
        return float(.018*np.dot(peaks, decay**np.arange(len(peaks)))/(1-decay**len(peaks)))

    def describe(self, start_ms, end_ms):
        return dict(spec=SPEC, start_ms=start_ms, end_ms=end_ms,
                    level=self.level(start_ms, end_ms),
                    meaning='Full-prefix strain proxy; not official local stars or a playability verdict')
