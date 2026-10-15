import logging
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from methods.FEAT import FEAT, _build_projector_from_W, aggregate_ecc_priors_from_clients, gsa_distill_loss
from methods.FEAT_repo import FEAT as RepoFEAT, FLDataSelector
from main import args_parser, get_learner
from utils.dmc import DummyDataset
from utils.fedcbdr_replay import global_leverage_scores, select_fedcbdr_indices
from utils.inc_net import ETFClassifier, IncrementalNet_CausalETF, simplex_etf


class TinyBackbone(nn.Module):
    out_dim = 8

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8 * 8 * 3, self.out_dim)

    def forward(self, images):
        return {"features": self.linear(images.flatten(1))}


def small_network(args):
    # Keep the actual FEAT network forward/classifier while replacing the
    # expensive ResNet backbone in the two-task training smoke test.
    network = IncrementalNet_CausalETF(args)
    network.convnet = TinyBackbone()
    network.feature_dim = 8
    network.fc = ETFClassifier(8)
    return network


class TinyDataManager:
    nb_tasks = 2

    def __init__(self):
        rng = np.random.default_rng(7)
        self.images = rng.integers(0, 256, (32, 8, 8, 3), dtype=np.uint8)
        self.labels = np.repeat(np.arange(4), 8)

    def get_task_size(self, task):
        return 2

    def get_dataset(self, indices, source, mode):
        selected = np.isin(self.labels, indices)
        return DummyDataset(self.images[selected], self.labels[selected],
                            transforms.ToTensor()), np.flatnonzero(selected)


class FEATTests(unittest.TestCase):
    def test_implementation_argument_and_dispatch(self):
        with patch("sys.argv", ["main.py", "--implementation", "repo"]):
            self.assertEqual(args_parser().implementation, "repo")
        with patch("sys.argv", ["main.py", "--implementation", "paper"]):
            self.assertEqual(args_parser().implementation, "paper")
        with patch("sys.argv", ["main.py"]):
            self.assertEqual(args_parser().implementation, "paper")

        args = {
            "implementation": "repo", "method": "feat", "dataset": "cifar10",
            "tasks": 2, "beta": 0, "num_users": 2, "frac": 1.0,
            "mem_size": 1, "k_value": 0, "p_value": 0, "seed": 2,
            "local_ep": 1, "net": "resnet18", "gpu": 0,
            "init_cls": 2, "increment": 2, "class_order": [0, 1, 2, 3],
            "wandb": 0, "save_dir": "", "nums": 0,
            "r": 0.015, "ltc": 5, "num_worker": 0, "mulc": None,
            "local_bs": 16, "local_lr": 0.01, "weight_decay": 1e-5,
            "com_round": 1, "kd": 0.1, "ema_decay": 0.7,
            "n": 1, "tttt": 255,
        }
        with patch("methods.FEAT_repo.IncrementalNet",
                   side_effect=lambda model_args, pretrained=False: small_network(model_args)), \
             patch.object(RepoFEAT, "_get_norm_and_transform",
                          return_value=(transforms.ToTensor(), None)), \
             patch("main.logging.basicConfig"):
            learner, logger = get_learner("feat", args)
        self.assertIsInstance(learner, RepoFEAT)
        manager = TinyDataManager()
        learner.incremental_train(manager, logger)
        self.assertTrue(any(learner.retained_ds_all))
        learner.after_task()
        learner.incremental_train(manager, logger)
        result = learner.eval_task(args, 2, learner.PH, learner.PT,
                                   learner.bar_eH_G, learner.r_H, learner.r_T)
        self.assertIn("top1", result[0])
        args["implementation"] = "paper"
        with patch("methods.FEAT.IncrementalNet_CausalETF", side_effect=small_network), \
             patch("main.logging.basicConfig"):
            paper_learner, _ = get_learner("feat", args)
        self.assertIsInstance(paper_learner, FEAT)

    def test_repo_selector_returns_local_indices(self):
        selector = FLDataSelector(
            2, [torch.eye(3), torch.eye(3)], query_budget=3
        )
        masked = selector.mask_local_data()
        probabilities = selector.aggregate_leverage_scores(
            selector.compute_local_leverage_scores(masked)
        )
        selected = selector.sample_data(probabilities)
        self.assertEqual(sum(len(rows) for rows in selected.values()), 3)
        self.assertTrue(all(0 <= index < 3
                            for rows in selected.values() for index in rows))

    def test_fedcbdr_global_leverage_and_balanced_unique_selection(self):
        features = [
            torch.tensor([[1., 0.], [1., 0.], [10., 0.], [0., 1.]]),
            torch.tensor([[0., 1.], [0., 2.], [1., 0.], [0., 3.]]),
        ]
        labels = [np.array([0, 0, 0, 1]), np.array([1, 1, 0, 1])]
        scores = global_leverage_scores(features, seed=4)
        self.assertGreater(float(scores[0][2]), float(scores[0][0]))
        self.assertAlmostEqual(sum(float(item.sum()) for item in scores), 2.0, places=4)
        selected = select_fedcbdr_indices(features, labels, per_class_budget=1, seed=4)
        self.assertEqual(sum(map(len, selected.values())), 4)
        self.assertTrue(all(len(rows) <= 2 and len(rows) == len(set(rows))
                            for rows in selected.values()))
        chosen_labels = [labels[client][rows] for client, rows in selected.items()]
        self.assertEqual(np.bincount(np.concatenate(chosen_labels), minlength=2).tolist(), [2, 2])

    def test_weighted_tail_priors(self):
        global_prior = aggregate_ecc_priors_from_clients([
            {"bar_eH_T": 1.0, "bar_eT_T": 3.0, "nT": 1},
            {"bar_eH_T": 3.0, "bar_eT_T": 1.0, "nT": 3},
        ])
        self.assertAlmostEqual(global_prior["bar_eH_G"], 2.5)
        self.assertAlmostEqual(global_prior["bar_eT_G"], 1.5)

    def test_etf_and_gsa(self):
        weights = simplex_etf(4, 8)
        gram = weights @ weights.T
        target = (torch.eye(4) - torch.ones(4, 4) / 4) * 4 / 3
        torch.testing.assert_close(gram, target, atol=1e-5, rtol=1e-5)
        labels = torch.tensor([0, 0, 0, 1])
        features = weights[labels].clone().requires_grad_(True)
        loss = gsa_distill_loss(features, labels, weights, tau=0.5)
        self.assertLess(float(loss.detach()), 1e-5)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_projector_and_egc_forward(self):
        args = {"net": "resnet18", "dataset": "cifar10", "init_cls": 2, "increment": 2}
        network = small_network(args)
        network.update_fc(4)
        weights = network.fc.weight.T
        tail = _build_projector_from_W(weights[:, :2])
        head = _build_projector_from_W(weights[:, 2:])
        torch.testing.assert_close(head @ head, head, atol=1e-5, rtol=1e-5)
        inputs = torch.rand(3, 3, 8, 8)
        original = network(inputs)["logits"]
        corrected = network(inputs, head, tail, 0.0, 1, 1)["logits"]
        self.assertEqual(original.shape, corrected.shape)
        self.assertTrue(torch.isfinite(corrected).all())

    def test_two_task_federated_smoke(self):
        args = {
            "net": "resnet18", "dataset": "cifar10", "init_cls": 2,
            "increment": 2, "gpu": 0, "seed": 2, "tasks": 2,
            "wandb": 0, "save_dir": "", "nums": 0, "num_users": 2,
            "num_worker": 0, "mulc": None, "local_bs": 16,
            "local_ep": 1, "local_lr": 0.01, "weight_decay": 1e-5,
            "mem_size": 1, "beta": 0, "com_round": 1,
            "lambda": 0.1, "temp": 0.5, "ema_decay": 0.7,
        }
        with patch("methods.FEAT.IncrementalNet_CausalETF", side_effect=small_network):
            learner = FEAT(args)
        manager = TinyDataManager()
        logger = logging.getLogger("feat_test")
        learner.incremental_train(manager, logger)
        self.assertEqual(learner._total_classes, 2)
        self.assertIsNone(learner.bar_eH_G)
        learner.after_task()
        learner.incremental_train(manager, logger)
        self.assertEqual(learner._total_classes, 4)
        self.assertIsNotNone(learner.bar_eH_G)
        self.assertIsNotNone(learner.PH)
        result = learner.eval_task(args, 2, learner.PH, learner.PT,
                                   learner.bar_eH_G, learner.r_H, learner.r_T)
        self.assertIn("top1", result[0])


if __name__ == "__main__":
    unittest.main()
