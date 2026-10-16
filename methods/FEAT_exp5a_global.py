"""FEAT training with LANDER Exp5a-Global replay selection.

FEAT fixes its ETF classifier, so the trainable BatchNorm affine parameters in
the final encoder stage provide the exact SGD trajectory for selection. LANDER's
pooled uncentered PCA, client quotas, residual greedy rule, and diagnostics are
reused without changing their implementation.
"""

import copy
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from methods.FEAT import FEAT, aggregate_ecc_priors_from_clients, estimate_ecc_priors, gsa_distill_loss
from methods.exp5a_global import Exp5aGlobal
from utils.dmc import DummyDataset, average_weights


class FEATExp5aGlobal(FEAT):
    _compute_projected_trajectories = Exp5aGlobal._compute_projected_trajectories
    _greedy_selection = Exp5aGlobal._greedy_selection
    _evaluate_selection = Exp5aGlobal._evaluate_selection
    _log_global_z_error = Exp5aGlobal._log_global_z_error
    _safe_percentile_leq = staticmethod(Exp5aGlobal._safe_percentile_leq)
    _write_exp5_diag_jsonl = Exp5aGlobal._write_exp5_diag_jsonl

    def __init__(self, args):
        super().__init__(args)
        self.num_users = args["num_users"]
        self.select_on_last_task = True
        self.exp5a_rank = args.get("exp5a_rank", 128)
        self.exp5a_svd_oversampling = args.get("exp5a_svd_oversampling", 16)
        self.target_mode = args.get("exp5a_target_mode", "budget_scaled_sum")
        self.tau = args.get("exp5a_tau", 0.05)
        self.wandb = args.get("wandb", 0)
        self.trajectory_matrix = None
        self.exp5_diag_path = os.path.abspath(
            f"exp5_greedy_diag_{args.get('exp_name', 'feat_exp5a_global')}_seed{self.seed}.jsonl"
        )
        self._exp5_diag_initialized = False

    @staticmethod
    def _trajectory_layers(model):
        stage = getattr(model.convnet, "layer4", None)
        if stage is None:
            raise RuntimeError("Exp5a-Global FEAT needs a final encoder stage (layer4)")
        layers = tuple(module for module in stage.modules()
                       if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and module.affine)
        if not layers:
            raise RuntimeError("Exp5a-Global FEAT needs trainable BatchNorm layers in layer4")
        return layers

    @staticmethod
    def _affine_vector(layers):
        return torch.cat([parameter.detach().flatten()
                          for layer in layers for parameter in (layer.weight, layer.bias)]).float().cpu()

    def _fl_train(self, train_dataset):
        groups = self.user_groups
        offsets, rows = [], []
        for client in range(self.num_users):
            offsets.append(len(rows))
            for local_id, dataset_idx in enumerate(groups[client]):
                rows.append({"client_id": client, "local_id": local_id,
                             "global_id": len(rows), "dataset_idx": int(dataset_idx),
                             "label": int(train_dataset.labels[dataset_idx])})
        self._exp5_offsets = offsets
        self._exp5_rows = rows

        layers = self._trajectory_layers(self._network)
        width = sum(layer.weight.numel() + layer.bias.numel() for layer in layers)
        self.trajectory_matrix = torch.zeros((len(rows), width), dtype=torch.float32)
        self._task_pred_delta = torch.zeros(width, dtype=torch.float32)
        self._task_actual_delta = torch.zeros(width, dtype=torch.float32)
        round_errors = []

        for round_id in tqdm(range(self.args["com_round"]), desc=f"Task {self._cur_task + 1}"):
            affine_before = self._affine_vector(layers)
            self._round_pred_delta = torch.zeros(width, dtype=torch.float32)
            local_weights, local_priors = [], []
            lr = self.args["local_lr"]
            active_clients = sum(
                bool(len(groups[client]) + sum(len(part[0]) for part in self.retained_ds_all[client]))
                for client in range(self.num_users)
            )
            for client in range(self.num_users):
                client_data = self._replay_dataset(client, train_dataset)
                if not len(client_data):
                    continue
                local_model = copy.deepcopy(self._network)
                self._local_update_with_trajectory(
                    local_model, self._loader(client_data, shuffle=True), lr,
                    client, len(groups[client]), 1.0 / active_clients,
                )
                local_weights.append({key: value.detach().cpu().clone()
                                      for key, value in local_model.state_dict().items()})
                if self._cur_task > 0 and self.retained_ds_all[client]:
                    images = np.concatenate([part[0] for part in self.retained_ds_all[client]])
                    labels = np.concatenate([part[1] for part in self.retained_ds_all[client]])
                    replay = DummyDataset(images, labels, train_dataset.trsf, train_dataset.use_path)
                    local_priors.append(estimate_ecc_priors(
                        local_model, self._loader(replay), self._known_classes,
                        self.args["ema_decay"],
                    ))
                del local_model

            if not local_weights:
                raise RuntimeError("No client received training data")
            fixed_etf = self._network.fc.weight.detach().clone()
            self._network.load_state_dict(average_weights(local_weights))
            with torch.no_grad():
                self._network.fc.weight.copy_(fixed_etf)
            if self._cur_task > 0:
                priors = aggregate_ecc_priors_from_clients(local_priors)
                self.bar_eH_G = None if priors is None else priors["bar_eH_G"]
                self.bar_eT_G = None if priors is None else priors["bar_eT_G"]
                self._update_projectors()

            actual = self._affine_vector(layers) - affine_before
            predicted = self._round_pred_delta
            error = (torch.linalg.vector_norm(predicted - actual) /
                     (torch.linalg.vector_norm(actual) + 1e-8)).item()
            round_errors.append(error)
            self._task_pred_delta += predicted
            self._task_actual_delta += actual

            if self.args.get("eval_each_round", False):
                accuracy = self._compute_accuracy(
                    self._network, self.test_loader, self.PH, self.PT,
                    self.bar_eH_G, self.r_H, self.r_T,
                )
                self.logger.info("Task %d round %d: %.2f%%", self._cur_task + 1,
                                 round_id + 1, accuracy)

        task_error = (torch.linalg.vector_norm(self._task_pred_delta - self._task_actual_delta) /
                      (torch.linalg.vector_norm(self._task_actual_delta) + 1e-8)).item()
        max_error = max(round_errors, default=0.0)
        print(f"[Attribution Invariant Gate] Task {self._cur_task}: "
              f"E_task = {task_error:.2e}, max E_a = {max_error:.2e} (tau = {self.tau})")
        if task_error > self.tau or max_error > self.tau:
            raise RuntimeError("Trajectory attribution invariant gate FAILED: "
                               f"E_task={task_error:.4e}, max_E_a={max_error:.4e}, tau={self.tau}")
        print("[Attribution Invariant Gate] PASSED successfully.")

    def _local_update_with_trajectory(self, model, loader, lr, client, num_current, client_weight):
        model.train()
        layers = self._trajectory_layers(model)
        captures = [None] * len(layers)

        def record_affine(index):
            def hook(_module, inputs, output):
                captures[index] = (inputs[0], output)
                # Residual addition and ReLU can modify BN outputs in place.
                # Preserve the unmodified output for exact affine gradients.
                return output.clone()
            return hook

        hooks = [layer.register_forward_hook(record_affine(index))
                 for index, layer in enumerate(layers)]
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=self.args["weight_decay"])
        steps = self.args["local_ep"] * len(loader)
        step = 0
        try:
            for epoch in range(self.args["local_ep"]):
                total, batches = 0.0, 0
                for sample_ids, images, labels in loader:
                    step += 1
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = model(images)
                    loss = F.cross_entropy(output["logits"], labels)
                    if self._cur_task > 0:
                        loss = loss + self.args["lambda"] * gsa_distill_loss(
                            output["features"], labels, model.fc.weight,
                            tau=self.args["temp"],
                        )

                    # Each BN's sample gradients sum to the optimized scalar
                    # loss gradient on that layer's affine parameters.
                    grad_outputs = torch.autograd.grad(
                        loss, tuple(output for _, output in captures), retain_graph=True
                    )
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
                    coefficient = -lr * client_weight * multiplier
                    current_mask = sample_ids < num_current
                    if current_mask.any():
                        candidates = self._exp5_offsets[client] + sample_ids[current_mask].numpy()
                        self.trajectory_matrix[candidates] += coefficient * per_sample[current_mask]
                    self._round_pred_delta += coefficient * (
                        per_sample.sum(dim=0) + self.args["weight_decay"] * self._affine_vector(layers)
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())
                    batches += 1
                self.logger.info("Task %d local epoch %d loss %.4f", self._cur_task + 1,
                                 epoch + 1, total / max(batches, 1))
        finally:
            for hook in hooks:
                hook.remove()

    def _retain_exemplars(self, train_dataset):
        # FEAT and FedCBDR specify a per-client, per-task replay allowance.
        budget = self.num_users * self.args["mem_size"]
        if budget <= 0:
            self.trajectory_matrix = None
            return
        groups, offsets, rows = self.user_groups, self._exp5_offsets, self._exp5_rows
        if budget > len(rows):
            raise ValueError(f"Exp5a replay budget {budget} exceeds {len(rows)} task candidates")
        projected, _, _ = self._compute_projected_trajectories(groups, offsets, rows)
        selected = self._greedy_selection(projected, groups, budget, offsets, rows)
        self._evaluate_selection(selected, projected, groups, offsets,
                                 self.trajectory_matrix.shape[1], rows)
        for client, local_ids in selected.items():
            indices = np.asarray(groups[client], dtype=int)[np.asarray(sorted(local_ids), dtype=int)]
            self.retained_ds_all[client].append(
                (train_dataset.images[indices].copy(), train_dataset.labels[indices].copy())
            )
            self.logger.info("Task %d client %d retained %d replay samples",
                             self._cur_task, client, len(indices))
        self.trajectory_matrix = None
