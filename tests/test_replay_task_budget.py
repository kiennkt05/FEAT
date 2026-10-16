"""Check the equal task-wide replay budget across multiple FEAT tasks."""

import contextlib
import io
import logging
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision import transforms

from main import args_parser, get_learner
from methods.FEAT_repo import FEAT as RepoFEAT
from methods.FEAT_exp5a_global import FEATExp5aGlobal
from methods.FEAT_repo_exp5a_global import RepoFEATExp5aGlobal
from tests.test_exp5a_global import TinyFEATNet
from utils.dmc import DummyDataset
from utils.replay_budget import check_fedcbdr_mem_size


class ThreeTaskManager:
    nb_tasks = 3

    def __init__(self):
        rng = np.random.default_rng(13)
        self.images = rng.integers(0, 256, (24, 8, 8, 3), dtype=np.uint8)
        self.labels = np.repeat(np.arange(6), 4)

    def get_task_size(self, _task):
        return 2

    def get_dataset(self, indices, source, mode):
        mask = np.isin(self.labels, indices)
        return DummyDataset(self.images[mask], self.labels[mask],
                            transforms.ToTensor()), np.flatnonzero(mask)


class ReplayTaskBudgetTests(unittest.TestCase):
    def test_repo_selection_uses_slots_with_replacement(self):
        learner = object.__new__(RepoFEAT)
        learner.args = {'mem_size': 4, 'num_users': 2}
        learner.user_groups = {0: [0], 1: [1]}
        learner._extract_client_features = lambda _client: (torch.ones(1, 2), [0])
        learner._select_data_for_retention()
        self.assertEqual(learner.selected_data_indices, {0: [0, 0], 1: [0, 0]})

    def test_optional_fedcbdr_setting_check(self):
        settings = {
            ('cifar10', 3): 450, ('cifar10', 5): 300,
            ('cifar100', 5): 1000, ('cifar100', 10): 500,
            ('tiny_imagenet', 10): 2000, ('tiny_imagenet', 20): 1000,
        }
        for (dataset, tasks), budget in settings.items():
            with self.subTest(dataset=dataset, tasks=tasks):
                self.assertEqual(check_fedcbdr_mem_size(dataset, tasks, budget), budget)
                with self.assertRaisesRegex(ValueError, f'mem_size={budget}'):
                    check_fedcbdr_mem_size(dataset, tasks, budget + 1)
        with self.assertRaisesRegex(ValueError, 'no FedCBDR mem_size setting'):
            check_fedcbdr_mem_size('tiny_imagenet', 5, 500)

        with patch('sys.argv', ['main.py', '--dataset=cifar10', '--tasks=3',
                                '--num_users=5', '--mem_size=450',
                                '--check_fedcbdr_mem_size']):
            self.assertEqual(args_parser().mem_size, 450)
        with patch('sys.argv', ['main.py', '--dataset=cifar10', '--tasks=3',
                                '--num_users=5', '--mem_size=300',
                                '--check_fedcbdr_mem_size']), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                args_parser()

    def test_three_task_equal_client_selection_counts(self):
        # mem_size=4 is shared by two clients: two retained slots per client
        # from each nonfinal task; old task selections accumulate.
        for method in ('feat', 'exp5a_global'):
            for implementation in ('paper', 'repo'):
                with self.subTest(method=method, implementation=implementation):
                    argv = [
                        'main.py', f'--method={method}',
                        f'--implementation={implementation}', '--dataset=cifar10',
                        '--tasks=3', '--num_users=2', '--com_round=1',
                        '--local_ep=1', '--local_bs=4', '--num_worker=0',
                        '--mem_size=4', '--beta=0', '--n=1', '--exp5a_rank=2',
                    ]
                    with patch('sys.argv', argv):
                        args = vars(args_parser())
                    args.update(num_class=6, init_cls=2, increment=2,
                                class_order=list(range(6)), save_dir='')
                    manager = ThreeTaskManager()
                    deterministic = torch.are_deterministic_algorithms_enabled()
                    try:
                        with patch('methods.FEAT.IncrementalNet_CausalETF',
                                   side_effect=lambda _args: TinyFEATNet()), \
                             patch('methods.FEAT_repo.IncrementalNet',
                                   side_effect=lambda _args, _pretrained=False: TinyFEATNet()), \
                             patch('methods.FEAT_repo.FEAT._get_norm_and_transform',
                                   return_value=(transforms.ToTensor(), None)), \
                             patch.object(FEATExp5aGlobal, '_write_exp5_diag_jsonl'), \
                             patch.object(RepoFEATExp5aGlobal, '_write_exp5_diag_jsonl'), \
                             patch('main.logging.basicConfig'), \
                             contextlib.redirect_stdout(io.StringIO()), \
                             contextlib.redirect_stderr(io.StringIO()):
                            learner, _ = get_learner(method, args)
                            for task, expected in enumerate((2, 4, 4)):
                                learner.incremental_train(manager, logging.getLogger(__name__))
                                if implementation == 'paper':
                                    counts = [sum(len(labels) for _, labels in parts)
                                              for parts in learner.retained_ds_all]
                                else:
                                    counts = [sum(len(part) for part in parts)
                                              for parts in learner.retained_ds_all]
                                self.assertEqual(counts, [expected, expected],
                                                 f'{method}/{implementation} task {task}')
                                if task == manager.nb_tasks - 1 and method == 'exp5a_global':
                                    self.assertIsNone(learner.trajectory_matrix)
                                learner.after_task()
                    finally:
                        torch.use_deterministic_algorithms(deterministic)


if __name__ == '__main__':
    unittest.main()
