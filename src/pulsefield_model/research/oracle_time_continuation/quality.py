"""Streaming organization diagnostics; these are not a learned playability score."""
import json
import math
from pathlib import Path


class ChartMetrics:
    """Constant-size counts for matched source/generated chart comparisons.

    Long holds, repetition and dense attacks are measured, not prohibited: their
    musical role and difficulty require source comparison and human playtesting.
    """
    def __init__(self):
        self.rows = self.notes = self.closes = 0
        self.presses = [0] * 5
        self.occupation_ms = [0.] * 5
        self.open_heads = [None] * 4
        self.last_attack = [None] * 4
        self.min_lane_attack_gap_ms = None
        self.last_time = None
        self.first_time = None
        self.last_actions = None
        self.identical_run = self.longest_identical_run = 0
        self.ln_bins = [0] * 9
        self.ln_edges_ms = (125, 250, 500, 1000, 2000, 4000, 16000, 64000)
        self.max_ln_ms = self.sum_ln_ms = 0.
        self.max_occupied_silence_ms = 0.
        self.raw_surprisal = self.policy_surprisal = 0.
        self.sampled = self.close_pruned_lane_rows = 0

    def consume(self, time, actions, sampling=None):
        if self.first_time is None:
            self.first_time = time
        if self.last_time is not None:
            gap = time - self.last_time
            held = sum(t is not None for t in self.open_heads)
            self.occupation_ms[held] += gap
            if held:
                self.max_occupied_silence_ms = max(self.max_occupied_silence_ms, gap)
        press_count = sum(a in (1, 2) for a in actions)
        self.presses[press_count] += 1
        self.rows += 1
        self.notes += press_count
        self.identical_run = self.identical_run + 1 if tuple(actions) == self.last_actions else 1
        self.longest_identical_run = max(self.longest_identical_run, self.identical_run)
        for lane, action in enumerate(actions):
            if action in (1, 2):
                if self.last_attack[lane] is not None:
                    gap = time - self.last_attack[lane]
                    self.min_lane_attack_gap_ms = gap if self.min_lane_attack_gap_ms is None else min(gap, self.min_lane_attack_gap_ms)
                self.last_attack[lane] = time
            if action == 2:
                self.open_heads[lane] = time
            elif action == 3:
                duration = time - self.open_heads[lane]
                self.closes += 1
                self.max_ln_ms = max(self.max_ln_ms, duration)
                self.sum_ln_ms += duration
                self.ln_bins[sum(duration > edge for edge in self.ln_edges_ms)] += 1
                self.open_heads[lane] = None
        if sampling is not None and not sampling.get('seed', False):
            self.sampled += 1
            self.raw_surprisal -= sampling['raw_model_log_probability']
            self.policy_surprisal -= sampling['decode_log_probability']
            self.close_pruned_lane_rows += sum(raw > 0 and policy == 0
                                               for raw, policy in zip(sampling['raw_close_probability'],
                                                                      sampling['policy_close_probability']))
        self.last_time, self.last_actions = time, tuple(actions)

    def report(self):
        return dict(rows=self.rows, notes=self.notes, ln_closes=self.closes,
                    press_histogram=self.presses, occupancy_duration_ms=self.occupation_ms,
                    span_ms=0. if self.last_time is None else self.last_time - self.first_time,
                    min_same_lane_attack_gap_ms=self.min_lane_attack_gap_ms,
                    longest_identical_row_run=self.longest_identical_run,
                    ln_duration_upper_edges_ms=self.ln_edges_ms, ln_duration_histogram=self.ln_bins,
                    max_ln_ms=self.max_ln_ms, mean_ln_ms=self.sum_ln_ms / max(1, self.closes),
                    max_gap_while_occupied_ms=self.max_occupied_silence_ms,
                    generated_rows=self.sampled,
                    mean_selected_raw_surprisal=self.raw_surprisal / max(1, self.sampled),
                    mean_selected_policy_surprisal=self.policy_surprisal / max(1, self.sampled),
                    close_pruned_lane_rows=self.close_pruned_lane_rows)


def source_metrics(rows):
    metrics = ChartMetrics()
    for row in rows:
        metrics.consume(row.time_ms, row.actions)
    return metrics.report()


def generated_metrics(path: Path):
    metrics = ChartMetrics()
    with Path(path).open('rb') as stream:
        for line in stream:
            value = json.loads(line)
            metrics.consume(value['time_ms'], value['actions'], value)
    return metrics.report()
