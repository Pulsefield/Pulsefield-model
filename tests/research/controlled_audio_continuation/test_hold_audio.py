from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel, load_model
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, interval_losses, score_interval
from ensomi_model.research.planned_audio_continuation.release import conditioned_release_logits
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from planned_audio_continuation.test_distribution import chart, config


def setup(width=8):
    torch.set_num_threads(1)
    model = ControlledAudioModel(replace(config(), lookahead=16, bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=60),
        style_names=('ln-coordination',), hold_audio_width=width).eval()
    return model


def held(release=1501, lane=0):
    close = [0]*4;close[lane] = 3
    tap = [0]*4;tap[lane] = 1
    rest = [3]*4;rest[lane] = 0
    return chart([0, release, 1800, 1900], [(2, 2, 2, 2), close, tap, rest], 2000)


def controls(model, end=2001):
    return ControlSchedule((ControlSpan(0, end, stars=3, ln_fraction=.7),), model.style_names)


def activate(model):
    with torch.no_grad():
        model.hold_cues.release.weight.normal_(std=.05)
        model.hold_cues.row.weight.normal_(std=.05)


def test_old_hold_origin_is_read_from_full_audio_and_future_tails_are_not_inputs():
    torch.manual_seed(140)
    net = setup();activate(net)
    c = held()
    # The 0-ms origins precede this crop; its hypothetical release normalizer
    # still reaches the deadline before the later H at 1800 ms.
    batch = collate_interval(IntervalExample(c, 10, 100), net.config, recovery=net.recovery)
    assert int(batch.inputs.base.mel_start[0]) > 0
    assert batch.inputs.release_waits
    with pytest.raises(ContractError, match='full-song encoding'):
        score_interval(net, batch.inputs, net.encode_coarse(torch.from_numpy(c.mel)[None]), controls=controls(net))
    encoded = net.encode_audio(torch.from_numpy(c.mel)[None]).detach().requires_grad_()
    scores = score_interval(net, batch.inputs, None, controls=controls(net), encoded_full=encoded)
    release_loss = interval_losses(scores, batch)[1]
    gradient = torch.autograd.grad(release_loss, encoded)[0]
    assert gradient[0, 0].abs().sum() > 0
    # The query is after 1000 ms. Only origin retrieval reads frame 0 in this
    # detached encoding; current-audio/global-context computation is not a path.
    changed = held(1101, 1)
    other = collate_interval(IntervalExample(changed, 10, 100), net.config, recovery=net.recovery)
    compared = score_interval(net, other.inputs, None, controls=controls(net), encoded_full=encoded)
    for a, b in zip(vars(scores).values(), vars(compared).values()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_cues_receive_both_losses_and_preserve_mirror_and_tap_independence():
    torch.manual_seed(213)
    net = setup();activate(net)
    c = held()
    encoded = net.encode_audio(torch.from_numpy(c.mel)[None])
    batch = collate_interval(IntervalExample(c, 0, 2001), net.config, recovery=net.recovery)
    losses = interval_losses(score_interval(net, batch.inputs, None, controls=controls(net), encoded_full=encoded), batch)
    for loss, output in ((losses[1], net.hold_cues.release.weight), (losses[2], net.hold_cues.row.weight)):
        grad = torch.autograd.grad(loss, (output, net.hold_cues.slots[0].weight), retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad)
    audio_grads = torch.autograd.grad(losses[1]+losses[2],
        (net.audio_input.weight, net.context.project.weight), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in audio_grads)
    mirror = chart(c.source.rows['time'], c.source.rows['actions'][:, ::-1], c.duration_ms)
    mb = collate_interval(IntervalExample(mirror, 0, 2001), net.config, recovery=net.recovery)
    a, b = [score_interval(net, x.inputs, None, controls=controls(net), encoded_full=encoded) for x in (batch, mb)]
    reverse = [ROW_ACTIONS.index(row[::-1]) for row in ROW_ACTIONS]
    torch.testing.assert_close(a.head, b.head, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(a.release, b.release, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(a.row, b.row[:, reverse], atol=2e-6, rtol=2e-6)

    left = chart([0,100,250,400], [(1,0,0,0),(0,2,0,0),(0,0,0,1),(0,3,0,0)], 500)
    right = chart([0,100,250,400], [(0,1,1,1),(0,2,0,0),(1,0,1,0),(0,3,0,0)], 500)
    encoded = net.encode_audio(torch.from_numpy(left.mel)[None])
    bs = [collate_interval(IntervalExample(c,0,501),net.config,recovery=net.recovery) for c in (left,right)]
    a,b = [score_interval(net,x.inputs,None,controls=controls(net,501),encoded_full=encoded) for x in bs]
    torch.testing.assert_close(a.head,b.head,atol=0,rtol=0)
    torch.testing.assert_close(a.release,b.release,atol=0,rtol=0)


def test_zero_initialized_cues_and_disabled_checkpoints_preserve_the_original_model(tmp_path):
    torch.manual_seed(711)
    base = setup(0)
    cue = setup()
    receipt = cue.load_state_dict(base.state_dict(), strict=False)
    assert not receipt.unexpected_keys and all(k.startswith('hold_cues.') for k in receipt.missing_keys)
    c = held();mel = torch.from_numpy(c.mel)[None]
    batch = collate_interval(IntervalExample(c,0,2001),base.config,recovery=base.recovery)
    encoded = base.encode_audio(mel)
    ordinary = score_interval(base,batch.inputs,base.encode_coarse(mel),controls=controls(base))
    full = score_interval(base,batch.inputs,None,controls=controls(base),encoded_full=encoded)
    new = score_interval(cue,batch.inputs,None,controls=controls(cue),encoded_full=encoded)
    for old, same, added in zip(vars(ordinary).values(),vars(full).values(),vars(new).values()):
        torch.testing.assert_close(old,same,atol=2e-5,rtol=2e-6)
        torch.testing.assert_close(same,added,atol=0,rtol=0)
    for model, legacy in ((base, True),(cue,False)):
        options = model.probability_options()
        if legacy:
            del options['hold_audio_width']
        path = tmp_path/('legacy.pt' if legacy else 'cues.pt')
        torch.save(dict(format='controlled-audio/v1',model_config=asdict(model.config),
                        probability_options=options,model=model.state_dict()),path)
        restored = load_model(path)
        assert restored.hold_audio_width == model.hold_audio_width
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[key],atol=0,rtol=0)


def test_interval_partition_preserves_probability_with_old_origins_and_conditional_waits():
    torch.manual_seed(329)
    net = setup();activate(net)
    c = held();encoded = net.encode_audio(torch.from_numpy(c.mel)[None])
    def loss(width):
        result = []
        for i in range(IntervalExample(c,0,width).count):
            batch = collate_interval(IntervalExample(c,i,width),net.config,recovery=net.recovery)
            result.append(torch.stack(interval_losses(score_interval(net,batch.inputs,None,
                controls=controls(net),encoded_full=encoded),batch)))
        return torch.stack(result).sum(0)
    expected, actual = loss(2001), loss(137)
    torch.testing.assert_close(expected,actual,atol=5e-5,rtol=5e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_full_audio_cue_losses_and_shared_gradients_match_mps():
    torch.manual_seed(586)
    cpu = setup();activate(cpu)
    mps = setup().to('mps');mps.load_state_dict(cpu.state_dict())
    c = held();values = []
    for device, net in (('cpu',cpu),('mps',mps)):
        encoded = net.encode_audio(torch.from_numpy(c.mel)[None].to(device))
        batch = collate_interval(IntervalExample(c,0,2001),net.config,device,recovery=net.recovery)
        loss = torch.stack(interval_losses(score_interval(net,batch.inputs,None,
            controls=controls(net),encoded_full=encoded),batch))
        gradient = torch.autograd.grad(loss[-1],(net.hold_cues.slots[0].weight,
            net.hold_cues.release.weight,net.hold_cues.row.weight))
        assert torch.isfinite(loss).all() and all(torch.isfinite(g).all() for g in gradient)
        values.append((loss.detach().cpu(),*(g.cpu() for g in gradient)))
    for a,b in zip(*values):
        torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-4)


def test_cached_native_rows_and_release_queries_match_training_and_publication_partition():
    torch.manual_seed(932)
    net = setup();activate(net)
    mel = np.random.default_rng(32).normal(size=(100,128)).astype(np.float32)
    schedule = controls(net,1001)
    heads = (0,100,200,300,500,750,900)
    original_row, original_release = net.planned_row_log_probs, net.release_logits
    row_records, release_records = [], {}
    def row(*args, **kwargs):
        value = original_row(*args,**kwargs)
        row_records.append(value.detach().clone())
        return value
    def release(audio, history, clocks, **kwargs):
        value = original_release(audio,history,clocks,**kwargs)
        for clock, q in zip(clocks.detach().numpy(),value.detach()):
            release_records[clock.tobytes()] = q.clone()
        return value
    net.planned_row_log_probs,net.release_logits = row,release
    with torch.inference_mode():
        a = ControlledSession(net,mel,1000,schedule,head_times=heads,seed=78)
        a.publish_to(1000)
    net.planned_row_log_probs,net.release_logits = original_row,original_release
    with torch.inference_mode():
        b = ControlledSession(net,mel,1000,schedule,head_times=heads,seed=78)
        for end in range(137,1000,137):
            b.publish_to(end)
        b.publish_to(1000)
    assert a.rows == b.rows and not any(a.replay.occupancy)
    generated = chart([r.time_ms for r in a.rows],[r.actions for r in a.rows],1000)
    batch = collate_interval(IntervalExample(generated,0,1001),net.config,recovery=net.recovery)
    with torch.no_grad():
        encoded = net.encode_audio(torch.from_numpy(mel)[None])
        scores = score_interval(net,batch.inputs,None,controls=schedule,encoded_full=encoded)
    torch.testing.assert_close(scores.row[:len(a.rows)],torch.cat(row_records),atol=3e-5,rtol=3e-6)
    ordinary = batch.inputs.release_valid.clone().flatten()
    for wait in batch.inputs.release_waits:
        ordinary[wait.destinations] = False
        raw = torch.stack([release_records[x.tobytes()] for x in wait.clocks.numpy()])
        expected = conditioned_release_logits(raw.flatten()[wait.native_indices])[wait.offsets]
        torch.testing.assert_close(scores.release.flatten()[wait.destinations],expected,atol=3e-5,rtol=3e-6)
    ordinary = ordinary.reshape_as(batch.inputs.release_valid)
    for i in torch.where(ordinary.any(-1))[0]:
        torch.testing.assert_close(scores.release[i],release_records[batch.inputs.release_clock[i].numpy().tobytes()],
                                   atol=3e-5,rtol=3e-6)
