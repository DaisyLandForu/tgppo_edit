import unittest
from pathlib import Path
from unittest.mock import Mock
import copy
import numpy as np
import torch
from project.search_features import frontier_features, discounted_returns, SearchRecorder
from project.policy.critic import Critic
from project.policy.search_critic import SearchCritic
from pyscipopt import Model, Branchrule, SCIP_RESULT
from project.utils.settings import init_params, settings


def snap(bounds, depths):
    return dict(bounds=bounds, depths=depths, nleaves=len(bounds), nchildren=0,
                nsiblings=0, solver_open_nodes=len(bounds), infinity=1e20)


class SearchChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_empty_degenerate_and_infinities(self):
        x, d = frontier_features(snap([], []))
        np.testing.assert_array_equal(x[:64], 0)
        np.testing.assert_array_equal(x[64:], [0, 1, 0, 0, 0])
        x, d = frontier_features(snap([2, 2, 1e20, -1e20], [0, 0, 0, 0]))
        self.assertEqual(x[0], 1)
        np.testing.assert_allclose(x[66:], [.5, .25, .25])
        self.assertEqual(d['association_l1'], 0.)
        x, _ = frontier_features(snap([np.inf, -np.inf], [1, 2]))
        self.assertEqual(x[:64].sum(), 0.)

    def test_identical_marginals_different_association(self):
        x, dx = frontier_features(snap([0, 1], [1, 8]))
        y, dy = frontier_features(snap([0, 1], [8, 1]))
        h, k = x[:64].reshape(8, 8), y[:64].reshape(8, 8)
        np.testing.assert_array_equal(h.sum(0), k.sum(0))
        np.testing.assert_array_equal(h.sum(1), k.sum(1))
        self.assertFalse(np.array_equal(h, k))
        self.assertEqual(dx['association_l1'], 1.)
        self.assertTrue(dx['both_axes_nondegenerate'])
        np.testing.assert_array_equal(x, frontier_features(snap([1, 0], [8, 1]))[0])

    def test_count_nan_and_negative_depth_rejected(self):
        bad = snap([1], [1]); bad['solver_open_nodes'] = 2
        with self.assertRaises(ValueError): frontier_features(bad)
        with self.assertRaises(ValueError): frontier_features(snap([np.nan], [1]))
        with self.assertRaises(ValueError): frontier_features(snap([1], [-1]))

    def test_return_labels_have_terminal_boundary(self):
        np.testing.assert_allclose(discounted_returns([1, 2, 3], .5), [2.75, 3.5, 3])
        np.testing.assert_allclose(discounted_returns([7], .5), [7])

    def test_sidecar_alignment_and_action_before_branch(self):
        model = Mock()
        model.getOpenNodeSnapshot.return_value = dict(snap([1], [2]), solving_time=30.)
        recorder = SearchRecorder(120)
        recorder.capture(model, dict(cands_state=np.zeros((3, 25))))
        self.assertEqual(recorder.pending['remaining_time'], .75)
        recorder.commit(2)
        trajectory = dict(episodes=[dict(transitions=[dict(action=2, cands_state=np.zeros((3, 25)))])])
        self.assertEqual(len(recorder.export(trajectory)['rows']), 1)
        trajectory['episodes'][0]['transitions'][0]['action'] = 1
        with self.assertRaises(ValueError): recorder.export(trajectory)

    def test_cached_and_full_forward_preserve_base_value_and_gradients(self):
        torch.manual_seed(11)
        base = Critic(25, 8, 53, 32, 4, 1, .05).eval()
        c, n, m = torch.randn(2, 3, 25), torch.randn(2, 8), torch.randn(2, 53)
        time, frontier = torch.rand(2, 1), torch.randn(2, 69)
        expected = base(c, n, m).reshape(-1)
        for group in ('R0', 'Rcap', 'Rprod', 'Rjoint'):
            torch.manual_seed(23)
            model = SearchCritic(base, group).eval()
            actual = model(c, n, m, time, frontier)
            torch.testing.assert_close(actual, expected)
            a, t = model.encode(c, n, m)
            torch.testing.assert_close(actual, model.forward_cached(a, t, torch.cat([n, m], -1), time, frontier))
            model.freeze_encoder()
            model(c, n, m, time, frontier).sum().backward()
            self.assertIsNone(model.var_embedding[1].weight.grad)
            self.assertIsNotNone(model.time_projection.weight.grad)
            if group != 'R0':
                self.assertGreater(model.extra_projection.weight.grad.abs().sum(), 0)

    def test_parameter_matching_and_common_initialization(self):
        base = Critic(25, 8, 53, 32, 4, 1, .05)
        for loss in ('mse', 'hl_gauss'):
            variants = []
            for group in ('R0', 'Rcap', 'Rprod', 'Rjoint'):
                torch.manual_seed(11)
                variants.append(SearchCritic(base, group, loss))
            counts = [sum(p.numel() for p in v.parameters()) for v in variants]
            self.assertEqual(len(set(counts[1:])), 1)
            for key, tensor in variants[0].value_head.state_dict().items():
                for v in variants[1:]: torch.testing.assert_close(tensor, v.value_head.state_dict()[key])

    def test_hl_gauss_normalization_and_expectation(self):
        base = Critic(25, 8, 53, 32, 4, 1, .05)
        model = SearchCritic(base, 'Rjoint', 'hl_gauss')
        target = torch.tensor([-36., -5., 0., 15., 52.])
        probabilities = model.gaussian_targets(target)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(5))
        self.assertTrue((probabilities >= 0).all())
        decoded = (probabilities*model.return_centers).sum(-1)
        torch.testing.assert_close(decoded[1:4], target[1:4], atol=.01, rtol=0)
        self.assertGreater(decoded[0], -36.)  # Truncated Gaussian edge bias reported.
        with self.assertRaises(ValueError): model.gaussian_targets(torch.tensor([53.]))

    @unittest.skipUnless(torch.cuda.is_available(), 'Cached tensor fit GPU unavailable')
    def test_gpu_cached_fit_and_trainable_adapter_restore(self):
        torch.manual_seed(11)
        base = Critic(25, 8, 53, 32, 4, 1, .05)
        for loss in ('mse', 'hl_gauss'):
            model = SearchCritic(base, 'Rjoint', loss)
            model.freeze_encoder(); model.cuda().train()
            args = [torch.randn(8, 32, device='cuda'), torch.randn(8, 32, device='cuda'),
                    torch.randn(8, 61, device='cuda'), torch.rand(8, 1, device='cuda'),
                    torch.rand(8, 69, device='cuda')]
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.00012)
            optimizer.zero_grad()
            objective = model.training_loss(model.logits_cached(*args), torch.randn(8,device='cuda'))
            objective.backward(); optimizer.step()
            state = {k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}
            restored = SearchCritic(base, 'Rjoint', loss).eval()
            restored.load_state_dict(state,strict=False)
            model.eval()
            torch.testing.assert_close(model.forward_cached(*args).cpu(),
                restored.forward_cached(*[x.cpu() for x in args]),atol=1e-4,rtol=1e-4)


class RealFrontierChecks(unittest.TestCase):
    def test_native_search_read_only_and_internal_objective_direction(self):
        class Probe(Branchrule):
            def __init__(self): self.snapshots = []
            def branchexeclp(self, allowaddcons):
                before = (self.model.getNNodes(), self.model.getPrimalbound(), self.model.getDualbound())
                snapshot = self.model.getOpenNodeSnapshot()
                frontier_features(snapshot)  # Every category count must match SCIP.
                after = (self.model.getNNodes(), self.model.getPrimalbound(), self.model.getDualbound())
                if before != after: raise AssertionError('Snapshot changed solver bounds/nodes')
                self.snapshots.append(snapshot)
                return {'result': SCIP_RESULT.DIDNOTRUN}
        def run(probe_enabled, maximize=False):
            m = Model()
            init_params(m, dict(node_limit=150, time_limit=120), settings['sandbox'])
            m.readProblem(str(Path(__file__).resolve().parents[4] / 'tgppo/datasets/train_instances/lseu.mps.gz'))
            if maximize:
                m.setObjective(-m.getObjective(), 'maximize')
                m.setObjlimit(-1120.)
            else: m.setObjlimit(1120.)
            p = Probe()
            if probe_enabled: m.includeBranchrule(p, 'Probe', 'read-only native fallback check', 999999, -1, 1.)
            m.optimize()
            result = (m.getNNodes(), str(m.getStatus()), m.getPrimalbound(), m.getDualbound())
            m.freeProb()
            return result, p.snapshots
        plain, _ = run(False)
        instrumented, snapshots = run(True)
        self.assertEqual(plain, instrumented)
        maximum, reverse = run(True, True)
        self.assertEqual(maximum[:2], instrumented[:2])
        self.assertAlmostEqual(maximum[2], -instrumented[2])
        self.assertAlmostEqual(maximum[3], -instrumented[3])
        self.assertEqual(len(snapshots), len(reverse))
        self.assertTrue(any(len(s['bounds']) for s in snapshots))
        for a,b in zip(snapshots, reverse):
            np.testing.assert_allclose(a['bounds'], b['bounds'])
            np.testing.assert_array_equal(a['depths'], b['depths'])


if __name__ == '__main__':
    unittest.main()
