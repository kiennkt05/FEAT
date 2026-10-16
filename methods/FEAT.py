"""Federated Geometry-Aware Correction (FEAT), arXiv:2604.08617.

GSA follows equations 3-5 and 15; EGC follows equations 8-14.  The replay
policy is deliberately independent of FEAT, as in the paper.  This runner
uses the FedCBDR global-perspective replay selector.
"""

import copy
import logging
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.base_cvpr import BaseLearner
from utils.dmc import DummyDataset, average_weights, partition_data, setup_seed
from utils.fedcbdr_replay import select_fedcbdr_indices
from utils.inc_net import IncrementalNet_CausalETF


def gsa_distill_loss(features, labels, prototypes, tau=0.5):
    """Class-balanced KL(P_feature || P_prototype) over batch relation rows."""
    if tau <= 0:
        raise ValueError("GSA temperature must be positive")
    f = F.normalize(features, dim=1)
    p = F.normalize(prototypes[labels], dim=1).detach()
    log_feature = F.log_softmax((f @ f.T) / tau, dim=1)
    log_target = F.log_softmax((p @ p.T) / tau, dim=1)
    row_kl = (log_feature.exp() * (log_feature - log_target)).sum(dim=1)
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    per_class = torch.zeros(len(counts), device=features.device, dtype=features.dtype)
    per_class.scatter_add_(0, inverse, row_kl)
    return (per_class / counts).mean()


def _build_projector_from_W(prototypes):
    """Equation 8: orthogonal projector onto the prototype column space."""
    projector = prototypes @ torch.linalg.pinv(prototypes.T @ prototypes) @ prototypes.T
    return (projector + projector.T) / 2


@torch.no_grad()
def estimate_ecc_priors(model, loader, num_tail, rho):
    """Estimate local tail-sample EMA energies (equation 9)."""
    weights = F.normalize(model.fc.weight.detach(), dim=1).T
    P_T = _build_projector_from_W(weights[:, :num_tail])
    P_H = _build_projector_from_W(weights[:, num_tail:])
    r_T = max(num_tail - 1, 1)
    r_H = max(weights.shape[1] - num_tail - 1, 1)
    head_ema = tail_ema = None
    n_tail = 0
    model.eval()
    for _, images, labels in loader:
        tail_mask = labels < num_tail
        if not tail_mask.any():
            continue
        features = F.normalize(model.extract_vector(images.to(weights.device))[
            tail_mask.to(weights.device)
        ], dim=1)
        head_energy = (features @ P_H).square().sum(dim=1).mean() / r_H
        tail_energy = (features @ P_T).square().sum(dim=1).mean() / r_T
        head_ema = rho * head_energy if head_ema is None else (1 - rho) * head_ema + rho * head_energy
        tail_ema = rho * tail_energy if tail_ema is None else (1 - rho) * tail_ema + rho * tail_energy
        n_tail += int(tail_mask.sum())
    return {
        "bar_eH_T": None if head_ema is None else float(head_ema),
        "bar_eT_T": None if tail_ema is None else float(tail_ema),
        "nT": n_tail,
    }


def aggregate_ecc_priors_from_clients(priors):
    """Sample-size-weighted global priors (equation 10)."""
    count = sum(p["nT"] for p in priors)
    if count == 0:
        return None
    return {
        "bar_eH_G": sum(p["bar_eH_T"] * p["nT"] for p in priors) / count,
        "bar_eT_G": sum(p["bar_eT_T"] * p["nT"] for p in priors) / count,
        "nT_sum": count,
    }


class FEAT(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet_CausalETF(args)
        self.device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
        self._network.to(self.device)
        self.retained_ds_all = [[] for _ in range(args["num_users"])]
        self.PH = self.PT = self.bar_eH_G = self.bar_eT_G = self.r_H = self.r_T = None
        self.logger = logging.getLogger(__name__)

    def _loader(self, dataset, *, shuffle=False, batch_size=None):
        workers = self.args["num_worker"]
        options = dict(batch_size=batch_size or self.args["local_bs"], shuffle=shuffle,
                       num_workers=workers, pin_memory=self.device.type == "cuda")
        if workers:
            options["persistent_workers"] = True
            if self.args.get("mulc") and not sys.platform.startswith("win"):
                options["multiprocessing_context"] = self.args["mulc"]
        return DataLoader(dataset, **options)

    def _replay_dataset(self, client, train_dataset):
        indices = np.asarray(self.user_groups[client], dtype=int)
        images, labels = train_dataset.images[indices], train_dataset.labels[indices]
        for old_images, old_labels in self.retained_ds_all[client]:
            images = np.concatenate((images, old_images))
            labels = np.concatenate((labels, old_labels))
        return DummyDataset(images, labels, train_dataset.trsf, train_dataset.use_path)

    @torch.no_grad()
    def _retain_exemplars(self, train_dataset):
        """FedCBDR global pseudo-feature and leverage-score replay selection."""
        budget = self.args["mem_size"]
        if budget <= 0:
            return
        # Use the deterministic test transform so feature rows correspond to
        # stable local sample IDs, even when training uses random augmentation.
        selection_dataset, _ = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test"
        )
        if not np.array_equal(selection_dataset.labels, train_dataset.labels):
            raise RuntimeError("Selection and training datasets have different sample order")
        self._network.eval()
        local_features = []
        local_labels = []
        for client in range(self.args["num_users"]):
            client_indices = np.asarray(self.user_groups[client], dtype=int)
            client_dataset = DummyDataset(
                selection_dataset.images[client_indices],
                selection_dataset.labels[client_indices],
                selection_dataset.trsf, selection_dataset.use_path
            )
            features = []
            for _, images, _ in self._loader(client_dataset):
                features.append(self._network.extract_vector(images.to(self.device)).cpu())
            local_features.append(
                torch.cat(features) if features else torch.empty(0, self._network.feature_dim)
            )
            local_labels.append(train_dataset.labels[client_indices])

        selected = select_fedcbdr_indices(
            local_features, local_labels, per_client_budget=budget,
            seed=self.seed + self._cur_task
        )
        for client, positions in selected.items():
            client_indices = np.asarray(self.user_groups[client], dtype=int)
            chosen = client_indices[np.asarray(positions, dtype=int)]
            if len(chosen):
                self.retained_ds_all[client].append(
                    (train_dataset.images[chosen].copy(), train_dataset.labels[chosen].copy())
                )

    def incremental_train(self, data_manager, logger):
        self.logger = logger
        self.data_manager = data_manager
        setup_seed(self.seed + self._cur_task + 1)
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._network.to(self.device)
        self.PH = self.PT = self.bar_eH_G = self.bar_eT_G = self.r_H = self.r_T = None

        train_dataset, _ = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        test_dataset, _ = data_manager.get_dataset(
            np.arange(self._total_classes), source="test", mode="test"
        )
        self.test_loader = self._loader(test_dataset, batch_size=256)
        self.user_groups, _ = partition_data(
            train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"]
        )
        self._fl_train(train_dataset)
        if self._cur_task < data_manager.nb_tasks - 1 or getattr(self, "select_on_last_task", False):
            self._retain_exemplars(train_dataset)

    def _fl_train(self, train_dataset):
        rounds = self.args["com_round"]
        for round_id in tqdm(range(rounds), desc=f"Task {self._cur_task + 1}"):
            local_weights = []
            local_priors = []
            lr = self.args["local_lr"]
            for client in range(self.args["num_users"]):
                client_data = self._replay_dataset(client, train_dataset)
                if not len(client_data):
                    continue
                local_model = copy.deepcopy(self._network)
                loader = self._loader(client_data, shuffle=True)
                self._local_update(local_model, loader, lr)
                local_weights.append({key: value.detach().cpu().clone()
                                      for key, value in local_model.state_dict().items()})
                if self._cur_task > 0:
                    old_data = self.retained_ds_all[client]
                    if old_data:
                        images = np.concatenate([part[0] for part in old_data])
                        labels = np.concatenate([part[1] for part in old_data])
                        replay = DummyDataset(images, labels, train_dataset.trsf, train_dataset.use_path)
                        local_priors.append(estimate_ecc_priors(
                            local_model, self._loader(replay), self._known_classes,
                            self.args["ema_decay"]
                        ))
                del local_model

            if not local_weights:
                raise RuntimeError("No client received training data")
            fixed_etf = self._network.fc.weight.detach().clone()
            self._network.load_state_dict(average_weights(local_weights))
            with torch.no_grad():
                self._network.fc.weight.copy_(fixed_etf)
            if self._cur_task > 0:
                global_priors = aggregate_ecc_priors_from_clients(local_priors)
                self.bar_eH_G = None if global_priors is None else global_priors["bar_eH_G"]
                self.bar_eT_G = None if global_priors is None else global_priors["bar_eT_G"]
                self._update_projectors()
            if self.args.get("eval_each_round", False):
                accuracy = self._compute_accuracy(
                    self._network, self.test_loader, self.PH, self.PT,
                    self.bar_eH_G, self.r_H, self.r_T
                )
                self.logger.info("Task %d round %d: %.2f%%", self._cur_task + 1, round_id + 1, accuracy)

    def _update_projectors(self):
        weights = F.normalize(self._network.fc.weight.detach(), dim=1).T
        self.PT = _build_projector_from_W(weights[:, :self._known_classes])
        self.PH = _build_projector_from_W(weights[:, self._known_classes:])
        self.r_T = max(self._known_classes - 1, 1)
        self.r_H = max(self._total_classes - self._known_classes - 1, 1)

    def _local_update(self, model, loader, lr):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=self.args["weight_decay"])
        for epoch in range(self.args["local_ep"]):
            total = 0.0
            batches = 0
            for _, images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = model(images)
                loss = F.cross_entropy(output["logits"], labels)
                if self._cur_task > 0:
                    loss = loss + self.args["lambda"] * gsa_distill_loss(
                        output["features"], labels, model.fc.weight,
                        tau=self.args["temp"]
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            self.logger.info("Task %d local epoch %d loss %.4f", self._cur_task + 1,
                             epoch + 1, total / max(batches, 1))

    def after_task(self):
        self._known_classes = self._total_classes
