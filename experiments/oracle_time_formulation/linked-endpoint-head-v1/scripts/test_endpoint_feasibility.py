import itertools
import unittest

import torch

from endpoint_feasibility import conditional_branch, requires_early_release


class FeasibilityChecks(unittest.TestCase):
    def test_strict_pre_onset_release_and_free_lane(self):
        self.assertTrue(requires_early_release((None,5,7,8),{0},5))
        self.assertFalse(requires_early_release((None,4,7,8),{0},5))
        self.assertFalse(requires_early_release((None,None,7,8),{0},5))
        self.assertFalse(requires_early_release((None,5,7,8),{0},None))

    def test_all_joint_paths_match_independent_law_conditioned_on_feasibility(self):
        p=torch.tensor([[.2,.3,.5],[.6,.1,.3],[.1,.7,.2]],dtype=torch.float64)
        early=torch.tensor([True,False,False])
        norm=1-float((1-p[:,0]).prod())
        total=0.
        for path in itertools.product(range(3),repeat=3):
            if not any(early[x] for x in path):
                continue
            probability=1.;satisfied=False
            for i,choice in enumerate(path):
                branch=conditional_branch(p.log(),early,i,satisfied)
                probability*=float(branch[choice].exp())
                satisfied=satisfied or bool(early[choice])
            expected=float(torch.tensor([p[i,x] for i,x in enumerate(path)]).prod())/norm
            self.assertAlmostEqual(probability,expected,places=13)
            total+=probability
        self.assertAlmostEqual(total,1.,places=13)

    def test_already_feasible_preserves_original_distribution(self):
        logp=torch.tensor([[.1,.2,.7],[.7,.2,.1]],dtype=torch.float64).log()
        torch.testing.assert_close(conditional_branch(logp,torch.tensor([True,False,False]),0,True),logp[0],atol=0,rtol=0)

    def test_impossible_assignment_is_explicit(self):
        with self.assertRaises(ValueError):
            conditional_branch(torch.tensor([[0.,-torch.inf]]),torch.tensor([False,True]),0,False)

    def test_rare_feasible_mass_does_not_turn_into_forced_first_head(self):
        logp=torch.tensor([[-100.,0.]]*3,dtype=torch.float64).log_softmax(-1)
        branch=conditional_branch(logp,torch.tensor([True,False]),0,False)
        self.assertAlmostEqual(float(branch[0].exp()),1./3.,places=13)


if __name__=='__main__':
    unittest.main()
