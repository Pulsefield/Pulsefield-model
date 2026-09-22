"""Time-parallel local/relation frontiers, checked against the online reference.

Exact replay and relation identity selection remain sequential CPU operations.
All learned writes are post-commit; query facts and neighbors use the preceding
state. A bounded chunk is the only extra storage, including paired Q-by-K reads.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .features import (HistoryStatus, TIME_DIM, clock_features, relative_lanes)
from .local import DILATIONS, LocalEntry, LocalState


def history_batch(encoder, inputs):
    times, extras = [], []
    for history, time, pace, gap, terminal in inputs:
        clocks = history.clocks_at(time)
        for hand in range(2):
            lanes = relative_lanes(hand)
            times.extend(value for lane in lanes for value in
                         (clocks.ln_age_ms[lane], clocks.lane_attack_ms[lane], clocks.lane_release_ms[lane]))
            times.extend(value for side in (hand, 1 - hand) for value in
                         (clocks.hand_attack_ms[side], clocks.hand_release_ms[side]))
            times.extend((gap, clocks.since_first_row_ms, pace.mean_ms, sum(pace.gaps_ms) if pace.gaps_ms else None))
            last = ([float(history.last_row.actions[lane] == action) for lane in lanes for action in range(4)]
                    if history.last_row else [0.] * 16)
            status = HistoryStatus.PRESENT if history.row_count else HistoryStatus.BOS
            extras.append(last + [history.occupancy[lane] for lane in lanes] + [len(pace.gaps_ms) / 32, terminal] +
                          [status == v for v in HistoryStatus])
    like = encoder.projection[0].weight
    features = torch.cat((clock_features(times, like).reshape(len(inputs), 2, 20 * TIME_DIM),
                          like.new_tensor(extras).reshape(len(inputs), 2, 26)), -1)
    return encoder.projection(features)


def fuse_local(encoder, raw, levels):
    values, metadata = [], []
    for entries in levels:
        times = [None if entry is None else entry.span_ms for entry in entries]
        tags = [[0 if entry is None else entry.count,
                 *(value == (HistoryStatus.BOS if entry is None else entry.status) for value in HistoryStatus)]
                for entry in entries]
        values.append(torch.stack([encoder.boundary.expand(2, -1) if entry is None else entry.value for entry in entries]))
        metadata.append(torch.cat((clock_features(times, raw), raw.new_tensor(tags)), -1)[:, None].expand(-1, 2, -1))
    return encoder.fusion(torch.cat((raw, *values, *metadata), -1))


def local_batch(encoder, initial, rows, first_id, raw, pace):
    entries = [LocalEntry(first_id + i, row, raw[i], ((first_id + i, row.time_ms),)) for i, row in enumerate(rows)]
    levels, buffers = [], []
    for layer, dilation, previous in zip(encoder.layers, DILATIONS, initial.buffers):
        inputs = list(previous) + entries
        current_values = torch.stack([entry.value for entry in entries])
        total = torch.zeros_like(current_values)
        supports = [{} for _ in rows]
        for slot, offset in enumerate((0, dilation, 2 * dilation)):
            indices = [len(previous) + i - offset for i in range(len(rows))]
            valid = [index >= 0 for index in indices]
            selected = [inputs[index] if index >= 0 else None for index in indices]
            values = torch.stack([torch.zeros_like(current_values[0]) if entry is None else entry.value
                                  for entry in selected])
            times, actions = [], []
            for i, (current, old) in enumerate(zip(entries, selected)):
                times.append(None if old is None else current.row.time_ms - old.row.time_ms)
                for hand in range(2):
                    lanes = relative_lanes(hand)
                    actions.append([float(current.row.actions[lane] == action) for lane in lanes for action in range(4)] +
                                   [float(old is not None and old.row.actions[lane] == action)
                                    for lane in lanes for action in range(4)])
                if old is not None:
                    supports[i].update(old.support)
            edge = torch.cat((clock_features(times, raw)[:, None].expand(-1, 2, -1),
                              raw.new_tensor(actions).reshape(len(rows), 2, 32)), -1)
            total = total + ((1 + layer.gates[slot](edge).tanh()) * layer.maps[slot](layer.norm(values)) *
                             raw.new_tensor(valid)[:, None, None])
        supports = [tuple(sorted(support.items())) for support in supports]
        times = [support[-1][1] - support[0][1] for support in supports]
        tags = [[len(support), *(v == (HistoryStatus.TRUNCATED if support[0][0] > 1 else HistoryStatus.PRESENT)
                                for v in HistoryStatus)] for support in supports]
        metadata = torch.cat((clock_features(times, raw), raw.new_tensor(tags)), -1)
        output = current_values + layer.update(total + layer.metadata(metadata)[:, None])
        buffers.append(tuple(inputs[-2 * dilation:]))
        entries = [LocalEntry(first_id + i, row, output[i], supports[i]) for i, row in enumerate(rows)]
        levels.append(entries)
    state = LocalState(tuple(buffers), tuple(level[-1] for level in levels), pace)
    post = fuse_local(encoder, raw, levels)
    before = [[initial.latest[level] if initial.latest else None, *entries[:-1]] for level, entries in enumerate(levels)]
    return state, post, before


def relation_batch(encoder, frontiers, states, times, row_ids):
    count, width = len(states), max(1, max(len(state.nodes) for state in states))
    payloads, all_times, all_actions, all_tags, visible = [], [], [], [], []
    for state, time, row_id in zip(states, times, row_ids):
        ranks = [{index: rank + 1 for rank, index in enumerate(reversed(ids))}
                 for ids in (*state.attacks, *state.releases)]
        hand_latest = [max((state.attacks[lane][-1] for lane in relative_lanes(hand)[:2] if state.attacks[lane]),
                           default=None) for hand in range(2)]
        padding = frontiers.new_zeros(2, encoder.config.hidden) if state.nodes else encoder.bos.expand(2, -1)
        payloads.append(torch.stack([node.payload for node in state.nodes] +
                                   [padding] * (width - len(state.nodes))))
        visible.append([index < max(1, len(state.nodes)) for index in range(width)])
        for index in range(width):
            node = state.nodes[index] if index < len(state.nodes) else None
            for hand in range(2):
                lanes = relative_lanes(hand)
                if node is None:
                    all_times.extend([None] * 11)
                    all_actions.append([0.] * 16)
                    all_tags.append([0.] * 23)
                    continue
                all_times.extend([time - node.row.time_ms] + [node.lane_intervals_ms[lane] for lane in lanes] +
                                 [node.hand_intervals_ms[side] for side in (hand, 1 - hand)] +
                                 [node.closed_durations_ms[lane] for lane in lanes])
                all_actions.append([float(node.row.actions[lane] == action) for lane in lanes for action in range(4)])
                tags = [value for lane in lanes for value in
                        (ranks[lane].get(node.row_id, 0) / encoder.config.attacks_per_lane,
                         ranks[4 + lane].get(node.row_id, 0) / encoder.config.releases_per_lane,
                         state.active_heads[lane] == node.row_id, node.closed_heads[lane] is not None,
                         node.lane_predecessors[lane] is not None)]
                tags += [hand_latest[side] == node.row_id for side in (hand, 1 - hand)]
                tags += [(row_id - node.row_id) / 32]
                all_tags.append(tags)
    edges = torch.cat((clock_features(all_times, frontiers).reshape(count, width, 2, 11 * TIME_DIM),
                       frontiers.new_tensor(all_actions).reshape(count, width, 2, 16),
                       frontiers.new_tensor(all_tags).reshape(count, width, 2, 23)), -1)
    attention = encoder.attention
    # Flatten learned projections before the paired SDPA layout. MPS linear
    # backward cannot consume channels-last gradients through this 4-D bank.
    bank = attention.norm(torch.stack(payloads).reshape(-1, encoder.config.hidden))
    def split(value):
        return (value.reshape(count, width, 2, attention.heads, attention.width)
                .permute(0, 2, 3, 1, 4).reshape(2 * count, attention.heads, width, attention.width).contiguous())
    query = attention.query(attention.norm(frontiers)).reshape(2 * count, attention.heads, 1, attention.width)
    bias = (attention.bias(edges.reshape(-1, edges.shape[-1])).reshape(count, width, 2, attention.heads)
            .permute(0, 2, 3, 1).reshape(2 * count, attention.heads, 1, width))
    mask = torch.tensor(visible, dtype=torch.bool, device=frontiers.device)
    mask = mask[:, None].expand(-1, 2, -1).reshape(2 * count, 1, 1, width)
    bias = bias.masked_fill(~mask, -torch.inf).contiguous()
    result = F.scaled_dot_product_attention(query, split(attention.key(bank)), split(attention.value(bank)),
                                           attn_mask=bias, dropout_p=0., is_causal=False)
    result = result.reshape_as(frontiers)
    output = frontiers + attention.output(result)
    return output + attention.ff(attention.ff_norm(output))


def frontiers(model, state, rows, *, score_queries):
    execution, pace = state.execution, state.local.pace
    inputs, content_inputs, query_inputs = [], [], []
    for row in rows:
        query = execution.query(model.config.time_lookahead_rows)
        following = execution.commit(row)
        gap = query.clocks.previous_row_ms
        inputs.append(query)
        query_inputs.append((query.history, query.time_ms, pace, gap, query.is_terminal))
        pace = pace.commit(gap)
        content_inputs.append((following.replay, row.time_ms, pace, gap, query.is_terminal))
        execution = following
    raw = history_batch(model.facts, content_inputs)
    timing = None if model.timing is None else model.timing([q.future_offsets_ms for q in inputs])[:, None]
    if timing is not None:
        raw = raw + timing
    local, post_frontiers, before_levels = local_batch(model.local, state.local, rows,
                                                     state.execution.next_index + 1, raw, pace)
    relation, before, after = state.relation, [], []
    for i, row in enumerate(rows):
        before.append(relation)
        relation = model.relation.commit(relation, row, state.execution.next_index + i + 1, post_frontiers[i])
        after.append(relation)
    times = [row.time_ms for row in rows]
    row_ids = [query.history.row_count + 1 for query in inputs]
    contents = relation_batch(model.relation, post_frontiers, after, times, row_ids)
    queries = None
    if score_queries:
        raw_queries = history_batch(model.facts, query_inputs)
        if timing is not None:
            raw_queries = raw_queries + timing
        query_frontiers = fuse_local(model.local, raw_queries, before_levels)
        queries = relation_batch(model.relation, query_frontiers, before, times, row_ids)
    return queries, contents, execution, local, relation, tuple(inputs), before, after
