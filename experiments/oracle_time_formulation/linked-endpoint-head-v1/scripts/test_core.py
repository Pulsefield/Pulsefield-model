import unittest
from types import SimpleNamespace

import numpy as np
import torch

from core import EndpointHead, candidate_features, context_features, linked_rows, reconstruct, time_tensor
from pulsefield_model.research.oracle_time_continuation.storage import ROW_DTYPE


class RepresentationChecks(unittest.TestCase):
    def test_structured_disk_times_have_owned_exact_storage(self):
        rows=np.array([(100.5,[1,0,0,0]),(100000.25,[0,1,0,0])],dtype=ROW_DTYPE)
        times=time_tensor(rows)
        self.assertEqual(times.dtype,torch.float64)
        self.assertEqual(times.tolist(),[100.5,100000.25])
        self.assertTrue(times.is_contiguous())
        rows['time'][0]=200
        self.assertEqual(times[0].item(),100.5)

    def test_crossing_holds_roundtrip_and_no_current_target_input(self):
        rows=np.array([(0,[2,2,0,0]),(40,[0,0,1,0]),(100,[3,0,2,0]),
                       (160,[1,3,0,0]),(5000,[0,0,3,0])],dtype=ROW_DTYPE)
        onsets,ends,pending=linked_rows(rows)
        np.testing.assert_array_equal(reconstruct(rows['actions'],ends),rows['actions'])
        self.assertEqual(ends[0,0],2);self.assertEqual(ends[0,1],3)
        self.assertTrue(np.all(pending[0]==-1))
        self.assertEqual(pending[2,0],2);self.assertEqual(pending[2,1],3)
        self.assertEqual(pending[2,2],-1)
        self.assertFalse(onsets[4])

    def test_mirror_preserves_context(self):
        clocks=SimpleNamespace(ln_age_ms=(None,20,None,40),lane_attack_ms=(1,2,3,4),
            lane_release_ms=(5,6,None,8),previous_row_ms=30,since_first_row_ms=1000)
        query=SimpleNamespace(clocks=clocks,time_ms=100)
        mirror_clocks=SimpleNamespace(**{k:(tuple(reversed(v)) if isinstance(v,tuple) else v) for k,v in vars(clocks).items()})
        mirror=SimpleNamespace(clocks=mirror_clocks,time_ms=100)
        hidden=torch.arange(16).reshape(2,8).float()
        actions=(2,0,1,0); pending=(None,250,None,300)
        left=context_features(hidden,query,actions,pending,0)
        right=context_features(hidden.flip(0),mirror,actions[::-1],pending[::-1],3)
        torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_all_future_candidates_and_masked_padding(self):
        feature=candidate_features(np.arange(301)*100,np.ones(301,bool),0)
        self.assertEqual(feature.shape,(300,20))
        model=EndpointHead(24)
        context=torch.randn(2,24)
        candidates=torch.stack((feature,feature))
        valid=torch.ones(2,300,dtype=torch.bool);valid[1,200:]=False
        logp=model(context,candidates,valid,use_context=True)
        self.assertTrue(torch.isfinite(logp[0,299]))
        self.assertTrue(torch.isneginf(logp[1,200:]).all())
        torch.testing.assert_close(logp.exp().sum(-1),torch.ones(2))
        (-logp[0,299]).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_prior_cannot_read_history(self):
        model=EndpointHead(24)
        torch.nn.init.normal_(model.context[-1].weight,std=.1)
        candidates=torch.randn(2,9,20);valid=torch.ones(2,9,dtype=torch.bool)
        a=model(torch.randn(2,24),candidates,valid,use_context=False)
        b=model(torch.randn(2,24)*10,candidates,valid,use_context=False)
        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
