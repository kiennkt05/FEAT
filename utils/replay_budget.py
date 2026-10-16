"""Validate an equal per-client allocation of a task-wide replay budget."""

FEDCBDR_MEM_SIZE_BY_SETTING = {
    ("cifar10", 3): 450,
    ("cifar10", 5): 300,
    ("cifar100", 5): 1000,
    ("cifar100", 10): 500,
    ("tiny_imagenet", 10): 2000,
    ("tiny_imagenet", 20): 1000,
}


def check_fedcbdr_mem_size(dataset, tasks, mem_size):
    """Check the per-task budget used in FedCBDR's main experiment settings."""
    dataset = dataset.lower().replace("-", "_")
    if dataset == "tinyimagenet":
        dataset = "tiny_imagenet"
    expected = FEDCBDR_MEM_SIZE_BY_SETTING.get((dataset, tasks))
    if expected is None:
        raise ValueError(
            f"no FedCBDR mem_size setting for dataset={dataset}, tasks={tasks}"
        )
    if mem_size != expected:
        raise ValueError(
            f"FedCBDR dataset={dataset}, tasks={tasks} requires "
            f"mem_size={expected} samples per task; got {mem_size}"
        )
    return expected


def equal_client_quota(total_budget, num_clients, candidate_counts=None, *, replacement=False):
    if num_clients <= 0:
        raise ValueError("num_users must be positive")
    if total_budget < 0:
        raise ValueError("mem_size must be nonnegative")
    if total_budget % num_clients:
        raise ValueError("mem_size must be divisible by num_users for equal replay allocation")
    quota = total_budget // num_clients
    if candidate_counts is not None:
        if len(candidate_counts) != num_clients:
            raise ValueError("candidate counts must include every client")
        for client, count in enumerate(candidate_counts):
            if quota and (count == 0 if replacement else count < quota):
                raise ValueError(
                    f"client {client} has {count} task candidates, fewer than its replay quota {quota}"
                )
    return quota
