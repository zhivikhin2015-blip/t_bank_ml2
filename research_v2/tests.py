"""Проверки научных инвариантов; не обращаются к отложенным исходам."""
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .core import (OUTPUT, PROJECT, SOURCE, UserSampler, affine_calibration, cache_array,
                   capture_rng, load_npz, paired_interval, restore_rng, save_npz,
                   save_torch, sha, user_weights, write_json)


class ScientificChecks(unittest.TestCase):
    def test_macro_weights_and_sampler(self):
        uid = np.array([10, 10, 20])
        weights = user_weights(uid)
        np.testing.assert_allclose(weights, [.25, .25, .5])
        self.assertAlmostEqual(np.dot(weights, [1., 0., 0.]), .25)
        sampled = UserSampler(uid).sample(np.random.default_rng(7), 20000)
        self.assertLess(abs(np.mean(uid[sampled] == 20) - .5), .015)

    def test_analytic_optimum_and_identity(self):
        ref = torch.tensor([.6, .3, .1], dtype=torch.float64)
        reward = torch.tensor([.1, .9, .5], dtype=torch.float64)
        beta = .3
        optimal = (ref.log() + reward / beta).softmax(-1)
        log_z = torch.logsumexp(ref.log() + reward / beta, 0)
        for q in [ref, optimal, torch.tensor([.2, .2, .6], dtype=torch.float64)]:
            objective = (q * reward).sum() - beta * (q * (q.log() - ref.log())).sum()
            rhs = beta * log_z - beta * (q * (q.log() - optimal.log())).sum()
            torch.testing.assert_close(objective, rhs, atol=1e-12, rtol=0)

    def test_multistep_policy_gradient_including_future_kl(self):
        theta = torch.tensor([[.2, -.1], [-.4, .3], [.5, -.2]],
                             dtype=torch.float64, requires_grad=True)
        ref = torch.tensor([[.55, .45], [.7, .3], [.2, .8]], dtype=torch.float64)
        reward0 = torch.tensor([.1, .4], dtype=torch.float64)
        reward1 = torch.tensor([[.2, .9], [.8, .3]], dtype=torch.float64)
        lp = theta.log_softmax(-1)
        p = lp.exp()
        kl = (p * (lp - ref.log())).sum(-1)
        beta, gamma = .2, .9
        exact, surrogate = 0., 0.
        for a0, a1 in itertools.product(range(2), repeat=2):
            state1 = a0 + 1
            probability = p[0, a0] * p[state1, a1]
            g1 = reward1[a0, a1] - beta * kl[state1]
            g0 = reward0[a0] - beta * kl[0] + gamma * g1
            exact = exact + probability * g0
            sampled_surrogate = (lp[0, a0] * g0.detach() + gamma * lp[state1, a1] * g1.detach()
                                 - beta * (kl[0] + gamma * kl[state1]))
            surrogate = surrogate + probability.detach() * sampled_surrogate
        exact_grad = torch.autograd.grad(exact, theta, retain_graph=True)[0]
        estimated_grad = torch.autograd.grad(surrogate, theta)[0]
        torch.testing.assert_close(exact_grad, estimated_grad, atol=1e-12, rtol=0)

    def test_leave_one_out_unbiased(self):
        logits = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
        lp = logits.log_softmax(-1)
        p = lp.exp()
        r = torch.tensor([.1, .8], dtype=torch.float64)
        expected = (p * r).sum()
        surrogate = 0.
        for a, b in itertools.product(range(2), repeat=2):
            sampled = ((r[a] - r[b]) * lp[a] + (r[b] - r[a]) * lp[b]) / 2
            surrogate = surrogate + (p[a] * p[b]).detach() * sampled
        torch.testing.assert_close(torch.autograd.grad(expected, logits, retain_graph=True)[0],
                                   torch.autograd.grad(surrogate, logits)[0], atol=1e-12, rtol=0)

    def test_affine_calibration_is_nonnegative_and_user_weighted(self):
        x = np.array([.1, .3, .8])
        uid = np.array([1, 1, 2])
        fit = affine_calibration(x, .2 + .5 * x, uid)
        self.assertAlmostEqual(fit['slope'], .5)
        self.assertAlmostEqual(fit['intercept'], .2)
        reversed_fit = affine_calibration(x, 1 - x, uid)
        self.assertEqual(reversed_fit['slope'], 0)

    def test_noninferiority_keeps_reference_in_paired_resampling(self):
        uid = np.array([1, 1, 2, 3])
        base = np.array([.2, .4, .7, .1])
        same = paired_interval(base, base, uid, factor=.95, resamples=100, one_sided=True)
        self.assertGreater(same['lower_one_sided95'], 0)
        worse = paired_interval(.9 * base, base, uid, factor=.95, resamples=100, one_sided=True)
        self.assertLess(worse['lower_one_sided95'], 0)
        order = np.array([3, 0, 2, 1])
        permuted = paired_interval(base[order], base[order], uid[order], factor=.95, resamples=100)
        self.assertEqual(same['ci95'], permuted['ci95'])

    def test_rng_optimizer_and_accumulated_time_survive_resume(self):
        torch.manual_seed(17)
        rng = np.random.default_rng(17)
        model = torch.nn.Linear(2, 1).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        def update(m, opt, gen):
            x = torch.tensor(gen.normal(size=(4, 2)), dtype=torch.float64)
            y = torch.randn(4, 1, dtype=torch.float64)
            opt.zero_grad()
            ((m(x) - y) ** 2).mean().backward()
            opt.step()
        update(model, optimizer, rng)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'state.pt'
            save_torch(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'rng': capture_rng(rng), 'training_seconds': 11.5, 'step': 1})
            update(model, optimizer, rng)
            expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
            loaded = torch.load(path, weights_only=True)
            other = torch.nn.Linear(2, 1).double()
            other_opt = torch.optim.AdamW(other.parameters(), lr=.01)
            other.load_state_dict(loaded['model'])
            other_opt.load_state_dict(loaded['optimizer'])
            restore_rng(loaded['rng'], rng)
            update(other, other_opt, rng)
            for name, value in other.state_dict().items():
                torch.testing.assert_close(expected[name], value, atol=0, rtol=0)
            self.assertEqual(loaded['training_seconds'], 11.5)

    def test_actual_training_session_resumes_model_history_and_time(self):
        from . import train
        torch.manual_seed(33)
        rng = np.random.default_rng(33)
        model = torch.nn.Linear(2, 1).double()
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        config = {'spec': {'steps': 2}, 'gpt2': {}}
        def make_loss(m, gen):
            x = torch.tensor(gen.normal(size=(4, 2)), dtype=torch.float64)
            y = torch.randn(4, 1, dtype=torch.float64)
            return ((m(x)-y)**2).mean(), {}
        with tempfile.TemporaryDirectory() as d, patch.object(train, 'OUTPUT', Path(d)):
            session = train.TrainingSession('test', model, opt, rng, config, torch.device('cpu'))
            loss, details = session.update(lambda: make_loss(model, rng))
            session.checkpoint(loss, details, mse=.2)
            saved_seconds = session.training_seconds
            session.update(lambda: make_loss(model, rng))
            expected = train.cpu_state(model)
            other = torch.nn.Linear(2, 1).double()
            other_opt = torch.optim.AdamW(other.parameters(), lr=.01)
            other_rng = np.random.default_rng(999)
            resumed = train.TrainingSession('test', other, other_opt, other_rng, config, torch.device('cpu'))
            self.assertEqual(resumed.step, 1)
            self.assertEqual(resumed.training_seconds, saved_seconds)
            self.assertEqual(resumed.best_mse, .2)
            self.assertEqual(len(resumed.history), 1)
            self.assertIsNotNone(resumed.best_model)
            resumed.update(lambda: make_loss(other, other_rng))
            self.assertGreater(resumed.training_seconds, saved_seconds)
            for name, value in other.state_dict().items():
                torch.testing.assert_close(expected[name], value, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                train.TrainingSession('test', other, other_opt, other_rng,
                                      {'spec': {'steps': 3}, 'gpt2': {}}, torch.device('cpu'))

    def test_cache_rejects_wrong_inputs_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cache.npy'
            value = cache_array(path, {'input': 'a'}, lambda: np.array([1., 2.]))
            np.testing.assert_array_equal(value, [1, 2])
            with self.assertRaises(ValueError):
                cache_array(path, {'input': 'b'}, lambda: np.zeros(2))
            path.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                cache_array(path, {'input': 'a'}, lambda: np.zeros(2))

    def test_npz_roundtrip_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'data.npz'
            save_npz(path, uid=np.array([1, 2]), history=np.array([[0, 2], [1, 3]]))
            value = load_npz(path)
            np.testing.assert_array_equal(value['history'], [[0, 2], [1, 3]])
            self.assertFalse(path.with_name('data.npz.tmp').exists())

    def test_factorized_full_catalog_scores_equal_direct_scores(self):
        from .core import CalibratedReward
        from .evaluate import all_scores
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(71)
        cat = {'duration': np.array([0, 0, 40, 100, 300, 505], dtype=np.float32)}
        for variant in ['history_duration', 'item_duration', 'history_no_duration']:
            model = CalibratedReward(GPT2LMHeadModel(GPT2Config(vocab_size=6, n_embd=8,
                         n_layer=1, n_head=2, n_positions=8)), cat, 290, variant).eval()
            model.calibration_slope.fill_(.7)
            model.calibration_intercept.fill_(.2)
            context = torch.randn(3, 8)
            calibrated, raw = all_scores(model, context)
            item = torch.arange(6).repeat(3)
            direct = model.raw_score(context.repeat_interleave(6, dim=0), item).reshape(3, 6)
            torch.testing.assert_close(raw, direct, atol=1e-7, rtol=0)
            torch.testing.assert_close(calibrated, (.2+.7*direct).clamp(0,1), atol=1e-7, rtol=0)

    def test_calibration_bins_preserve_full_population_weights(self):
        from .metrics import calibration
        result = calibration(np.array([.1, .2, 1.]), np.array([.2, .2, 0.]), np.array([1, 1, 2]))
        self.assertAlmostEqual(sum(row['weight_mass'] for row in result['bins']), 1.)
        high = result['bins'][-1]
        self.assertEqual(high['examples'], 1)
        self.assertAlmostEqual(high['weight_mass'], .5)

    def test_evaluation_resumes_after_interrupt_and_rejects_changed_inputs(self):
        from . import evaluate as ev
        from .core import CalibratedReward, canonical_hash
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(71)
        temporary_root=PROJECT/'.tmp'/'research_v2'
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as d:
            output=Path(d)
            cat={'duration':np.r_[0.,0.,np.arange(20)+40].astype(np.float32),
                 'popularity':np.r_[0.,0.,np.arange(20)+1].astype(np.float32),
                 'first_seen':np.zeros(22,dtype=np.int64)}
            data={'history':np.array([[0,0,0,0,0,0,0,2],[0,0,0,0,0,0,2,3],[0,0,0,0,0,2,3,4]]),
                  'target':np.array([2,3,4]),'uid':np.array([10,20,20]),
                  'timestamp':np.array([100,120,150]),'ratio':np.array([.2,.8,1.]),'organic':np.array([1,0,1])}
            cat_path=output/'catalog.npz';data_path=output/'policy_val.npz'
            save_npz(cat_path,**cat);save_npz(data_path,**data)
            meta={'duration_cap_seconds':290,'files':{'catalog.npz':sha(cat_path)},'new_holdout_kind':'synthetic integration check'}
            save_npz(output/'reward_baselines.npz',global_mean=np.array(.7),item_mean=np.full(22,.7))
            config=GPT2Config(vocab_size=22,n_embd=8,n_layer=1,n_head=2,n_positions=8,use_cache=False)
            actor=GPT2LMHeadModel(config).eval()
            actor_path=output/'actor.pt'
            save_torch(actor_path,{'model':actor.state_dict(),'config':{'gpt2':config.to_dict()}})
            reward=CalibratedReward(actor,cat,290,'history_duration').eval()
            reward_path=output/'reward.pt'
            save_torch(reward_path,{'model':reward.state_dict(),
                       'config':{'gpt2':config.to_dict(),'variant':'history_duration'}})
            specs=[ev.descriptor('baseline_s11',actor_path),ev.descriptor('synthetic_policy',actor_path)]
            original_evaluate=ev.evaluate_one
            calls=[]
            def interrupted(spec,*args):
                calls.append(spec['name'])
                if len(calls)==2:
                    raise RuntimeError('Synthetic process interruption')
                return original_evaluate(spec,*args)
            with patch.object(ev,'OUTPUT',output),patch.object(ev,'data_manifest',lambda:meta),\
                 patch.object(ev,'catalog',lambda:cat),patch.object(ev,'load_data',lambda *args:data),\
                 patch.object(ev,'data_hash',lambda split:sha(data_path)),\
                 patch.object(ev,'actor_path',lambda seed:actor_path),\
                 patch.object(ev,'device',lambda:torch.device('cpu')),\
                 patch.object(ev,'log_access',lambda *args,**kwargs:None),\
                 patch.object(ev,'evaluate_one',interrupted):
                with self.assertRaises(RuntimeError):
                    ev.evaluate('policy_val',specs,reward_path,reward_path,'synthetic_resume')
                result=ev.evaluate('policy_val',specs,reward_path,reward_path,'synthetic_resume')
                self.assertTrue(result['completed'])
                self.assertEqual(calls,['baseline_s11','synthetic_policy','synthetic_policy'])
                ev.evaluate('policy_val',specs,reward_path,reward_path,'synthetic_resume')
                self.assertEqual(len(calls),3)
                actor_path.write_bytes(b'changed checkpoint')
                with self.assertRaises(ValueError):
                    ev.evaluate('policy_val',specs,reward_path,reward_path,'synthetic_resume')

    def test_holdout_preparation_cannot_read_raw_before_freeze(self):
        from . import core, data
        with tempfile.TemporaryDirectory() as d, patch.object(core,'OUTPUT',Path(d)),\
             patch.object(data.ds,'dataset') as raw_reader:
            with self.assertRaises(FileNotFoundError):
                data.prepare_holdout()
            raw_reader.assert_not_called()


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(OUTPUT / 'scientific_checks.json', {'passed': result.wasSuccessful(),
               'checks_run': result.testsRun, 'failures': len(result.failures),
               'errors': len(result.errors), 'heldout_outcomes_read': False,
               'sources':{str(p.relative_to(PROJECT)):sha(p) for p in
                 [SOURCE/'tests.py',SOURCE/'core.py',SOURCE/'train.py',SOURCE/'evaluate.py',SOURCE/'metrics.py']}})
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
