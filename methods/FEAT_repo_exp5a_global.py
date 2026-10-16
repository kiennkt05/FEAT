"""Repository FEAT training with LANDER Exp5a-Global replay selection.

The repository training loop, loss, scheduler, best-model snapshot, and replay
composition are retained. Only the memory selector is replaced. Trajectories
attribute local SGD updates to examples using the final stage's BN affine
parameters, since FEAT's ETF classifier is fixed.
"""

import copy
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.FEAT_exp5a_global import FEATExp5aGlobal
from methods.FEAT_repo import (FEAT as RepoFEAT, CustomConcatDataset,
                               aggregate_ecc_priors_from_clients,
                               estimate_ecc_priors, gsa_distill_loss)
from methods.exp5a_global import Exp5aGlobal
from utils.dmc import DatasetSplit, average_weights, partition_data
from utils.replay_budget import equal_client_quota


class RepoFEATExp5aGlobal(RepoFEAT):
    _trajectory_layers = staticmethod(FEATExp5aGlobal._trajectory_layers)
    _affine_vector = staticmethod(FEATExp5aGlobal._affine_vector)
    _compute_projected_trajectories = Exp5aGlobal._compute_projected_trajectories
    _greedy_selection = Exp5aGlobal._greedy_selection
    _evaluate_selection = Exp5aGlobal._evaluate_selection
    _log_global_z_error = Exp5aGlobal._log_global_z_error
    _safe_percentile_leq = staticmethod(Exp5aGlobal._safe_percentile_leq)
    _write_exp5_diag_jsonl = Exp5aGlobal._write_exp5_diag_jsonl

    def __init__(self, args):
        super().__init__(args)
        self.num_users = args["num_users"]
        self.exp5a_rank = args.get("exp5a_rank", 128)
        self.exp5a_svd_oversampling = args.get("exp5a_svd_oversampling", 16)
        self.target_mode = args.get("exp5a_target_mode", "budget_scaled_sum")
        self.tau = args.get("exp5a_tau", 0.05)
        self.wandb = args.get("wandb", 0)
        self.trajectory_matrix = None
        self.exp5_diag_path = os.path.abspath(
            f"exp5_greedy_diag_{args.get('exp_name', 'feat_repo_exp5a_global')}_seed{self.seed}.jsonl"
        )
        self._exp5_diag_initialized = False

    def _prepare_trajectories(self, train_dataset):
        offsets, rows, local_ids = [], [], []
        for client in range(self.num_users):
            offsets.append(len(rows))
            client_ids = {}
            for local_id, dataset_idx in enumerate(self.user_groups[client]):
                dataset_idx = int(dataset_idx)
                client_ids[dataset_idx] = local_id
                rows.append({"client_id": client, "local_id": local_id,
                             "global_id": len(rows), "dataset_idx": dataset_idx,
                             "label": int(train_dataset.labels[dataset_idx])})
            local_ids.append(client_ids)
        self._exp5_offsets = offsets
        self._exp5_rows = rows
        self._exp5_local_ids = local_ids
        layers = self._trajectory_layers(self._network)
        width = sum(layer.weight.numel() + layer.bias.numel() for layer in layers)
        self.trajectory_matrix = torch.zeros((len(rows), width), dtype=torch.float32)
        self._task_pred_delta = torch.zeros(width)
        self._task_actual_delta = torch.zeros(width)

    def _fl_train(self, train_dataset, train_data_indices, test_loader):
        self._network.to(self.device)
        self.best_model = None
        self.lowest_loss = np.inf
        optimizer = torch.optim.SGD(self._network.parameters(), lr=self.args["local_lr"],
                                    momentum=0.9, weight_decay=self.args["weight_decay"])
        if self.args["dataset"] in ("tiny_imagenet", "imagenet"):
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.args["com_round"], eta_min=1e-3)

        self.user_groups, _ = partition_data(train_dataset.labels, beta=self.args["beta"],
                                              n_parties=self.num_users)
        self._prepare_trajectories(train_dataset)
        layers = self._trajectory_layers(self._network)
        round_errors = []

        prog_bar = tqdm(range(self.args["com_round"]))
        for com in prog_bar:
            before = self._affine_vector(layers)
            self._round_pred_delta = torch.zeros_like(before)
            local_weights, loss_weight, priors_list = [], [], []
            idxs_users = range(self.num_users)
            for idx in idxs_users:
                local_train_ds_i = DatasetSplit(train_dataset, self.user_groups[idx])
                if self._cur_task > 0:
                    local_train_ds_i = CustomConcatDataset(
                        [[local_train_ds_i], self.retained_ds_all[idx]],
                        self.transform, self.args, self.logger)
                    print('####', local_train_ds_i.labels.shape)
                local_train_loader = DataLoader(
                    local_train_ds_i, batch_size=self.args["local_bs"], shuffle=True,
                    pin_memory=self.device.type == "cuda", **self._worker_options())
                local_model = copy.deepcopy(self._network)
                w, total_loss = self._local_update_with_trajectory(
                    local_model, local_train_loader, scheduler.get_last_lr()[0], idx)
                if self._cur_task > 0:
                    head_ids = list(range(self._known_classes, self._total_classes))
                    tail_ids = list(range(self._known_classes))
                    priors, P_H, P_T = estimate_ecc_priors(
                        model=local_model, data_loader=local_train_loader,
                        head_class_ids=head_ids, tail_class_ids=tail_ids,
                        ema_decay=float(self.args["ema_decay"]), feature_key="features")
                    priors_list.append(priors)
                local_weights.append(copy.deepcopy(w))
                loss_weight.append(total_loss)
                del local_train_loader, local_model, w
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # The repo scheduler is stepped with a global optimizer that has no
            # gradients; the client copies own every actual SGD update.
            optimizer.step()
            scheduler.step()
            sum_loss = sum(loss_weight)
            if sum_loss < self.lowest_loss:
                self.lowest_loss = sum_loss
                self.best_model = copy.deepcopy(self._network.state_dict())
            self._network.load_state_dict(average_weights(local_weights))

            actual = self._affine_vector(layers) - before
            predicted = self._round_pred_delta
            error = (torch.linalg.vector_norm(predicted - actual) /
                     (torch.linalg.vector_norm(actual) + 1e-8)).item()
            round_errors.append(error)
            self._task_pred_delta += predicted
            self._task_actual_delta += actual

            test_acc = self._compute_accuracy(self._network, test_loader)
            if self._cur_task > 0:
                test_old_acc = self._compute_accuracy(copy.deepcopy(self._network), self.old_loader)
                test_new_acc = self._compute_accuracy(copy.deepcopy(self._network), self.new_loader)
                print("Task {}, Test_accy {:.2f} O {} N {}".format(
                    self._cur_task, test_acc, test_old_acc, test_new_acc))
            info = "Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                self._cur_task, com + 1, self.args["com_round"], test_acc)
            self.logger.info(info)
            prog_bar.set_description(info)

        task_error = (torch.linalg.vector_norm(self._task_pred_delta - self._task_actual_delta) /
                      (torch.linalg.vector_norm(self._task_actual_delta) + 1e-8)).item()
        max_error = max(round_errors, default=0.0)
        print(f"[Attribution Invariant Gate] Task {self._cur_task}: "
              f"E_task = {task_error:.2e}, max E_a = {max_error:.2e} (tau = {self.tau})")
        if task_error > self.tau or max_error > self.tau:
            raise RuntimeError("Trajectory attribution invariant gate FAILED: "
                               f"E_task={task_error:.4e}, max_E_a={max_error:.4e}, tau={self.tau}")
        print("[Attribution Invariant Gate] PASSED successfully.")

        if self._cur_task > 0:
            global_priors = aggregate_ecc_priors_from_clients(priors_list, eps=1e-8)
            self.bar_eH_G = global_priors["bar_eH_G"]
            self.PH, self.PT = P_H, P_T
            self.r_H = self._total_classes - self._known_classes - 1
            self.r_T = self._known_classes - 1

        self._network.load_state_dict(self.best_model)
        if self._cur_task < self.tasks - 1:
            self._select_data_for_retention()
            for idx in idxs_users:
                self.retained_ds_all[idx].append(self._get_retained_dataset(idx))
        self.trajectory_matrix = None
        del self.best_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _local_update_with_trajectory(self, model, loader, lr, client):
        model.train()
        layers = self._trajectory_layers(model)
        captures = [None] * len(layers)

        def record_affine(index):
            def hook(_module, inputs, output):
                captures[index] = (inputs[0], output)
                return output.clone()  # Later residual additions and ReLUs can mutate BN output.
            return hook

        hooks = [layer.register_forward_hook(record_affine(index))
                 for index, layer in enumerate(layers)]
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=self.args["weight_decay"])
        steps = self.args["local_ep"] * len(loader)
        step = 0
        total_loss = 0
        try:
            for epoch in range(self.args["local_ep"]):
                epoch_lossce_collector, epoch_lossrd_collector = [], []
                for sample_ids, images, labels in loader:
                    step += 1
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = model(images)
                    loss_ce = F.cross_entropy(output["logits"], labels)
                    loss_rd = gsa_distill_loss(output["features"], labels, model.fc.weight)
                    loss = loss_ce + self.args["kd"] * loss_rd

                    grad_outputs = torch.autograd.grad(
                        loss, tuple(value for _, value in captures), retain_graph=True)
                    parts = []
                    for layer, (affine_input, _), grad_out in zip(layers, captures, grad_outputs):
                        reduce_dims = (0,) + tuple(range(2, affine_input.ndim))
                        mean = affine_input.mean(dim=reduce_dims, keepdim=True)
                        var = affine_input.var(dim=reduce_dims, unbiased=False, keepdim=True)
                        normalized = (affine_input - mean) * torch.rsqrt(var + layer.eps)
                        spatial_dims = tuple(range(2, grad_out.ndim))
                        grad_weight = grad_out * normalized
                        if spatial_dims:
                            grad_weight = grad_weight.sum(dim=spatial_dims)
                            grad_bias = grad_out.sum(dim=spatial_dims)
                        else:
                            grad_bias = grad_out
                        parts.extend((grad_weight, grad_bias))
                    per_sample = torch.cat(parts, dim=1).detach().float().cpu()

                    multiplier = (1.0 - 0.9 ** (steps - step + 1)) / (1.0 - 0.9)
                    coefficient = -lr * multiplier / self.num_users
                    if self._cur_task == 0:
                        positions = [self._exp5_local_ids[client][int(idx)] for idx in sample_ids]
                        selected = torch.arange(len(positions))
                    else:
                        selected = torch.nonzero(sample_ids < len(self.user_groups[client])).flatten()
                        positions = sample_ids[selected].tolist()
                    if positions:
                        candidates = self._exp5_offsets[client] + np.asarray(positions, dtype=int)
                        self.trajectory_matrix[candidates] += coefficient * per_sample[selected]
                    self._round_pred_delta += coefficient * (
                        per_sample.sum(dim=0) +
                        self.args["weight_decay"] * self._affine_vector(layers))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if epoch == 0:
                        total_loss += loss.detach()
                    epoch_lossce_collector.append(loss_ce.item())
                    epoch_lossrd_collector.append(loss_rd.item())
                self.logger.info('Epoch: %d Loss CE: %f  Loss RD: %f ' % (
                    epoch, sum(epoch_lossce_collector) / len(epoch_lossce_collector),
                    sum(epoch_lossrd_collector) / len(epoch_lossrd_collector)))
        finally:
            for hook in hooks:
                hook.remove()
        return model.state_dict(), total_loss

    def _select_data_for_retention(self):
        # One replay budget for all clients' candidates in this task.
        budget = self.args["mem_size"]
        rows, offsets, groups = self._exp5_rows, self._exp5_offsets, self.user_groups
        if budget <= 0:
            self.selected_data_indices = {client: [] for client in range(self.num_users)}
            return
        equal_client_quota(budget, self.num_users, [len(groups[k]) for k in range(self.num_users)])
        if budget > len(rows):
            raise ValueError(f"Exp5a replay budget {budget} exceeds {len(rows)} task candidates")
        projected, _, _ = self._compute_projected_trajectories(groups, offsets, rows)
        selected = self._greedy_selection(projected, groups, budget, offsets, rows)
        self._evaluate_selection(selected, projected, groups, offsets,
                                 self.trajectory_matrix.shape[1], rows)
        self.selected_data_indices = {client: sorted(selected[client])
                                      for client in range(self.num_users)}
