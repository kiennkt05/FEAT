"""FedCBDR global-perspective replay selection (arXiv:2507.07712, Sec. 4.1).

The signed row permutation is an implicit orthogonal P_k.  It preserves a
one-to-one correspondence between masked rows and local sample IDs while
avoiding a dense n_k x n_k matrix for each client.  Q is shared across clients.
"""

from collections import defaultdict

import numpy as np
import torch


def global_leverage_scores(local_features, *, seed=0):
    """Return one global-SVD leverage vector per client in original row order."""
    if not local_features:
        return []
    nonempty = [features for features in local_features if len(features)]
    if not nonempty:
        return [torch.empty(0) for _ in local_features]
    width = nonempty[0].shape[1]
    if any(features.ndim != 2 or features.shape[1] != width for features in nonempty):
        raise ValueError("All client feature matrices must have the same width")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    shared_q, _ = torch.linalg.qr(torch.randn(width, width, generator=generator))
    masked = []
    permutations = []
    sizes = []
    for features in local_features:
        count = len(features)
        sizes.append(count)
        if count == 0:
            permutations.append(torch.empty(0, dtype=torch.long))
            continue
        features = features.detach().to(device="cpu", dtype=torch.float32)
        permutation = torch.randperm(count, generator=generator)
        signs = 2 * torch.randint(0, 2, (count, 1), generator=generator) - 1
        masked.append((features[permutation] * signs) @ shared_q)
        permutations.append(permutation)

    # Economical SVD supplies the left singular vectors of the pooled matrix.
    # Ignore null-space vectors: they contain no feature information.
    pooled = torch.cat(masked)
    U, singular_values, _ = torch.linalg.svd(pooled, full_matrices=False)
    tolerance = singular_values.max() * max(pooled.shape) * torch.finfo(pooled.dtype).eps
    rank = max(int((singular_values > tolerance).sum()), 1)
    masked_scores = U[:, :rank].square().sum(dim=1)
    scores = []
    offset = 0
    for count, permutation in zip(sizes, permutations):
        local = torch.empty(count)
        if count:
            local[permutation] = masked_scores[offset:offset + count]
            offset += count
        scores.append(local)
    return scores


def select_fedcbdr_indices(local_features, local_labels, *, per_class_budget, seed=0):
    """Select globally informed, class-balanced unique IDs under client caps.

    Returns a dict of client ID to row positions within that client's current
    task dataset.  The total target is K * C_task * per_class_budget and each
    client's cap is C_task * per_class_budget.  A round-robin over classes
    balances the global buffer; within a class, exponential-race keys perform
    leverage-weighted sampling without replacement.
    """
    if len(local_features) != len(local_labels):
        raise ValueError("Each feature matrix needs its matching labels")
    if per_class_budget < 0:
        raise ValueError("per_class_budget must be nonnegative")
    client_count = len(local_features)
    selected = {client: [] for client in range(client_count)}
    if not client_count or per_class_budget == 0:
        return selected

    labels = [np.asarray(item, dtype=np.int64) for item in local_labels]
    if any(len(features) != len(classes) for features, classes in zip(local_features, labels)):
        raise ValueError("Feature rows and labels must align")
    class_ids = sorted(set(np.concatenate(labels).tolist())) if any(len(item) for item in labels) else []
    if not class_ids:
        return selected
    client_cap = len(class_ids) * per_class_budget
    total_cap = client_cap * client_count
    leverage = global_leverage_scores(local_features, seed=seed)
    generator = np.random.default_rng(seed)
    queues = defaultdict(list)
    for client, (classes, scores) in enumerate(zip(labels, leverage)):
        # FedCBDR Eq. 6 first normalizes leverage within each client; the
        # common client factor cancels after global normalization.
        normalizer = max(float(scores.sum()), 1e-12)
        for position, (class_id, score) in enumerate(zip(classes, scores.tolist())):
            # Exponential race yields weighted sampling without replacement.
            weight = max(float(score) / normalizer, 1e-12)
            priority = -np.log(max(generator.random(), 1e-12)) / weight
            queues[int(class_id)].append((priority, client, position))
    for class_id in class_ids:
        queues[class_id].sort()

    offsets = {class_id: 0 for class_id in class_ids}
    total_selected = 0
    while total_selected < total_cap:
        made_progress = False
        for class_id in class_ids:
            queue = queues[class_id]
            pointer = offsets[class_id]
            while pointer < len(queue) and len(selected[queue[pointer][1]]) >= client_cap:
                pointer += 1
            offsets[class_id] = pointer
            if pointer == len(queue):
                continue
            _, client, position = queue[pointer]
            selected[client].append(position)
            total_selected += 1
            offsets[class_id] = pointer + 1
            made_progress = True
            if total_selected >= total_cap:
                break
        if not made_progress:
            break
    return selected
