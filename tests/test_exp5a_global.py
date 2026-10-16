import contextlib
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision import transforms

from main import args_parser, get_learner, train
from methods.FEAT_exp5a_global import FEATExp5aGlobal
from methods.FEAT_repo_exp5a_global import RepoFEATExp5aGlobal
from methods.exp5a_global import Exp5aGlobal
from utils.dmc import DummyDataset
from utils.inc_net import ETFClassifier, IncrementalNet_CausalETF


class Exp5aGlobalIntegrationTests(unittest.TestCase):
    def test_repo_dispatch_and_two_task_replay(self):
        class TinyManager:
            nb_tasks = 2

            def __init__(self, *_args):
                import numpy as np
                rng = np.random.default_rng(11)
                self.images = rng.integers(0, 256, (16, 8, 8, 3), dtype=np.uint8)
                self.labels = np.repeat([0, 1, 2, 3], 4)

            def get_class_order(self):
                return [0, 1, 2, 3]

            def get_task_size(self, _task):
                return 2

            def get_dataset(self, indices, source, mode):
                import numpy as np
                mask = np.isin(self.labels, indices)
                return DummyDataset(self.images[mask], self.labels[mask],
                                    transforms.ToTensor()), np.flatnonzero(mask)

        with patch('sys.argv', [
            'main.py', '--method=exp5a_global', '--implementation=repo',
            '--dataset=cifar10', '--tasks=2', '--num_users=2', '--com_round=2',
            '--local_ep=1', '--local_bs=4', '--num_worker=0',
            '--mem_size=2', '--exp5a_rank=2', '--beta=0', '--n=2',
        ]):
            args = vars(args_parser())
        args.update(num_class=4, init_cls=2, increment=2, save_dir='')
        records, lrs, replay_counts, loader_sizes = [], [], [], []
        original_update = RepoFEATExp5aGlobal._local_update_with_trajectory

        def track_update(self, model, loader, lr, client):
            lrs.append((self._cur_task, lr))
            if self._cur_task == 1:
                replay_counts.append(sum(len(ds) for ds in self.retained_ds_all[client]))
                loader_sizes.append((loader.dataset.local_len, len(loader.dataset)))
            return original_update(self, model, loader, lr, client)

        deterministic = torch.are_deterministic_algorithms_enabled()
        output = io.StringIO()
        try:
            with patch('main.DataManager', TinyManager), \
                 patch('methods.FEAT_repo.IncrementalNet',
                       side_effect=lambda _args, _pretrained=False: TinyFEATNet()), \
                 patch.object(RepoFEATExp5aGlobal, '_get_norm_and_transform',
                              return_value=(transforms.ToTensor(), None)), \
                 patch.object(RepoFEATExp5aGlobal, '_write_exp5_diag_jsonl',
                              lambda self, rows: records.extend(rows)), \
                 patch.object(RepoFEATExp5aGlobal, '_local_update_with_trajectory', track_update), \
                 patch('methods.FEAT_repo.FLDataSelector',
                       side_effect=AssertionError('Repo selector must not run')), \
                 patch('main.logging.basicConfig'), \
                 contextlib.redirect_stdout(output), \
                 contextlib.redirect_stderr(io.StringIO()):
                train(args)
        finally:
            torch.use_deterministic_algorithms(deterministic)

        self.assertEqual(sum(row['type'] == 'step' for row in records), 4)
        self.assertEqual(replay_counts, [2, 2, 2, 2])
        self.assertEqual(loader_sizes, [(4, 8)] * 4)
        self.assertEqual(len(lrs), 8)
        self.assertNotEqual(lrs[0][1], lrs[2][1])
        self.assertEqual(output.getvalue().count('[Attribution Invariant Gate] PASSED'), 2)

    def test_final_stage_tracks_all_five_resnet_batchnorms(self):
        model = IncrementalNet_CausalETF({
            'net': 'resnet18', 'dataset': 'cifar10', 'init_cls': 2, 'increment': 2,
        })
        layers = FEATExp5aGlobal._trajectory_layers(model)
        self.assertEqual(len(layers), 5)
        self.assertEqual(sum(layer.weight.numel() + layer.bias.numel() for layer in layers),
                         5120)

    def test_dispatch_uses_feat_with_exp5_replay_and_defaults(self):
        with patch('sys.argv', ['main.py', '--method=exp5a_global']):
            args = args_parser()
        self.assertEqual(args.beta, 0.5)
        self.assertEqual(args.wandb, 0)
        self.assertEqual(args.exp5a_rank, 128)
        self.assertEqual(args.mem_size, 50)
        self.assertNotIn('gdr_task_budget', vars(args))
        with patch('sys.argv', ['main.py', '--method=exp5a_global', '--gdr_task_budget=4']), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                args_parser()

        config = vars(args)
        config.update(init_cls=20, increment=20, save_dir='')
        with patch('methods.FEAT.IncrementalNet_CausalETF',
                   side_effect=lambda _args: TinyFEATNet()), \
             patch('main.logging.basicConfig'):
            learner, _ = get_learner('exp5a_global', config)
        self.assertIsInstance(learner, FEATExp5aGlobal)
        self.assertIsInstance(learner._network, TinyFEATNet)
        self.assertIs(FEATExp5aGlobal._greedy_selection, Exp5aGlobal._greedy_selection)
        self.assertIs(FEATExp5aGlobal._compute_projected_trajectories,
                      Exp5aGlobal._compute_projected_trajectories)

        with patch('sys.argv', ['main.py', '--method=feat']):
            feat_args = args_parser()
        self.assertEqual((feat_args.beta, feat_args.wandb, feat_args.w_new),
                         (0.5, 1, 1.0))

    def test_shared_projection_and_greedy_replay(self):
        learner = object.__new__(Exp5aGlobal)
        learner.num_users = 2
        learner.exp5a_rank = 2
        learner.exp5a_svd_oversampling = 0
        learner.seed = 2023
        learner._cur_task = 0
        learner.args = {}
        learner.wandb = 0
        learner.target_mode = 'budget_scaled_sum'
        learner.trajectory_matrix = torch.tensor([
            [1.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.9, 0.0], [0.0, -1.0, 0.0],
        ])
        groups = {0: [0, 1, 2], 1: [3, 4, 5]}
        offsets = [0, 3]
        rows = [{'label': i // 3} for i in range(6)]
        records = []
        learner._write_exp5_diag_jsonl = records.extend

        with contextlib.redirect_stdout(io.StringIO()):
            projected, bases, energies = learner._compute_projected_trajectories(
                groups, offsets, rows)
            selected = learner._greedy_selection(projected, groups, 2, offsets, rows)

        self.assertIs(bases[0], bases[1])
        self.assertTrue(torch.allclose(projected[0], learner.trajectory_matrix[:3] @ bases[0]))
        self.assertTrue(torch.allclose(projected[1], learner.trajectory_matrix[3:] @ bases[1]))
        self.assertAlmostEqual(energies[0], 1.0, places=4)
        self.assertEqual(selected, {0: [1], 1: [1]})
        self.assertEqual(sum(record['type'] == 'step' for record in records), 2)

    def test_two_task_feat_training_attribution_and_replay_on_cpu(self):
        class TinyManager:
            nb_tasks = 2

            def __init__(self, *_args):
                import numpy as np
                rng = np.random.default_rng(7)
                self.images = rng.integers(0, 256, (16, 8, 8, 3), dtype=np.uint8)
                self.labels = np.repeat([0, 1, 2, 3], 4)

            def get_class_order(self):
                return [0, 1, 2, 3]

            def get_task_size(self, _task):
                return 2

            def get_dataset(self, indices, source, mode):
                import numpy as np
                mask = np.isin(self.labels, indices)
                return DummyDataset(self.images[mask], self.labels[mask],
                                    transforms.ToTensor()), np.flatnonzero(mask)

        with patch('sys.argv', [
            'main.py', '--method=exp5a_global', '--dataset=cifar10', '--tasks=2',
            '--num_users=2', '--com_round=1', '--local_ep=1', '--local_bs=4',
            '--num_worker=0', '--mem_size=2', '--exp5a_rank=2', '--beta=0',
        ]):
            args = vars(args_parser())
        args.update(num_class=4, init_cls=2, increment=2, save_dir='')
        records = []
        replay_counts = []
        original_replay_dataset = FEATExp5aGlobal._replay_dataset

        def track_replay(self, client, dataset):
            if self._cur_task == 1:
                replay_counts.append(sum(len(labels) for _, labels in self.retained_ds_all[client]))
            return original_replay_dataset(self, client, dataset)

        deterministic = torch.are_deterministic_algorithms_enabled()
        output = io.StringIO()
        try:
            with patch('main.DataManager', TinyManager), \
                 patch('methods.FEAT.IncrementalNet_CausalETF',
                       side_effect=lambda _args: TinyFEATNet()), \
                 patch.object(FEATExp5aGlobal, '_write_exp5_diag_jsonl',
                              lambda self, rows: records.extend(rows)), \
                 patch.object(FEATExp5aGlobal, '_replay_dataset', track_replay), \
                 patch('methods.FEAT.select_fedcbdr_indices',
                       side_effect=AssertionError('FedCBDR selector must not run')), \
                 patch('main.logging.basicConfig'), \
                 contextlib.redirect_stdout(output), \
                 contextlib.redirect_stderr(io.StringIO()):
                train(args)
        finally:
            torch.use_deterministic_algorithms(deterministic)
        self.assertEqual(sum(row['type'] == 'step' for row in records), 8)
        self.assertEqual(replay_counts, [2, 2])
        self.assertIn('[Attribution Invariant Gate] PASSED successfully.', output.getvalue())
        self.assertIn('Exp5 Replay Evaluation', output.getvalue())


class TinyBackbone(nn.Module):
    out_dim = 8

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8 * 8 * 3, 8)
        self.layer4 = nn.Sequential(nn.BatchNorm1d(8), nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(8))

    def forward(self, images):
        return {'features': self.layer4(self.linear(images.flatten(1)))}


class TinyFEATNet(IncrementalNet_CausalETF):
    def __init__(self):
        nn.Module.__init__(self)
        self.convnet = TinyBackbone()
        self.feature_dim = 8
        self.fc = ETFClassifier(8)


if __name__ == '__main__':
    unittest.main()
