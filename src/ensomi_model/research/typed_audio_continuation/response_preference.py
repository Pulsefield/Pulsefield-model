"""Empirical recovery preferences at the factor that owns each action.

The reference is the native 2–6-star ranked TRAIN population, weighted by
song group, chart and audio seconds. These are soft rarity preferences, not
physiological limits or a complete definition of the V3 gameplay frontier.
"""
from dataclasses import dataclass

import numpy as np

from .program import ACTIONS, RELEASE_BITS


@dataclass(frozen=True)
class RecoveryPreference:
    reference: str = 'ranked-train-native-recovery-q01-v1'
    stars: tuple = (2., 3., 4., 5., 6.)
    hh_ms: tuple = (169., 143., 107., 89., 83.)
    rh_ms: tuple = (107., 83., 65., 55., 49.)
    hr_ms: tuple = (89., 75., 53., 44., 43.)
    strength: float = 4.
    maximum_cost: float = 4.
    head_pressure: float = 0.

    def head_cost(self, state, recent_heads, times, counts, stars):
        """Marginal cost of adding heads to a short occupied-key workload.

        Holds older than the HH lookback occupy keys throughout that interval.
        Other recent heads share the remaining keys. This coarse overload
        potential cannot identify the actual repeated column or certify that
        a particular candidate is comfortable; it reads no R1 column history.
        """
        times, counts = np.asarray(times), np.asarray(counts)
        window = np.interp(stars, self.stars, self.hh_ms)
        if self.head_pressure == 0:
            return np.zeros_like(times+counts+window, dtype=float)
        used = np.zeros_like(times+window, dtype=float)
        for previous, heads in recent_heads:
            used += heads*((times >= previous) & (times-previous < window))
        ages = times[..., None]-np.asarray(state.starts, dtype=float)
        occupied_throughout = (ages >= np.asarray(window)[..., None]).sum(-1)
        capacity = 4-occupied_throughout
        before = np.maximum(0., used-capacity)
        after = np.maximum(0., used+counts-capacity)
        marginal = self.head_pressure*(after**2-before**2)/np.maximum(capacity, 1)
        return np.where(np.isfinite(window), np.minimum(marginal, self.maximum_cost), 0.)

    def cost(self, gap_ms, stars, kind):
        """Finite cost below the interpolated reference's first percentile.

        The cost is zero above that reference, and missing gaps/requests have
        no preference. Its finite cap leaves every legal action in support.
        Difficulty outside the reference range uses its nearest endpoint.
        """
        gap = np.asarray(gap_ms, dtype=np.float64)
        threshold = np.interp(stars, self.stars, getattr(self, kind+'_ms'))
        raw = self.strength*np.maximum(0., np.log(threshold/np.maximum(gap, 1.)))
        return np.where(np.isfinite(raw), np.minimum(raw, self.maximum_cost), 0.)

    def release_clock_cost(self, state, times, stars, recovery, duration_ms):
        """Cost of the easiest currently eligible LN release at each time.

        The release clock has not chosen a subset. An older eligible hold can
        therefore keep the release clock open while the mark preference below
        still discourages closing a younger hold. True audio-end closure is
        mandatory and receives no delaying preference.
        """
        times = np.asarray(times)
        ages = times[..., None]-np.asarray(state.starts, dtype=float)
        costs = self.cost(ages, np.asarray(stars)[..., None], 'hr')
        best = np.where(ages >= recovery.hr, costs, np.inf).min(-1)
        return np.where(np.isfinite(best) & (times < duration_ms), best, 0.)

    def mark_cost(self, state, now, stars, duration_ms):
        """Score the proposed release subset, before LN identities are fixed."""
        if now == duration_ms:
            return np.zeros(len(RELEASE_BITS))
        ages = now-np.asarray(state.starts, dtype=float)
        return RELEASE_BITS @ self.cost(ages, stars, 'hr')

    def row_cost(self, replay, now, stars):
        """Score proposed attacks using their actual column's recovery.

        RH applies only before the first head following a release. Taking the
        larger HH/RH cost avoids counting the same attack twice; complete-row
        scoring still handles coupled choices across columns.
        """
        heads = np.asarray(replay.last_lane_attack_ms, dtype=float)
        releases = np.asarray(replay.last_lane_release_ms, dtype=float)
        hh = self.cost(now-heads, stars, 'hh')
        rh = self.cost(np.where(releases > heads, now-releases, np.nan), stars, 'rh')
        return np.isin(ACTIONS, (1, 2)) @ np.maximum(hh, rh)
