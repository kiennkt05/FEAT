import torch
from torch import nn
from torch.nn import functional as F


class TaskAwareTemperatureScalingLoss(nn.Module):
    """FedCBDR task-aware temperature scaling audit modes."""

    def __init__(
        self,
        num_old_classes,
        tau_old,
        tau_new,
        weight_old,
        weight_new,
        mode="paper_eq",
        correction_mode="none",
    ):
        super().__init__()

        if tau_old <= 0 or tau_new <= 0:
            raise ValueError("TTS temperatures must be positive")
        if mode not in {"paper_eq", "repo_dual"}:
            raise ValueError("invalid TTS mode")

        self.num_old_classes = int(num_old_classes)
        self.tau_old = float(tau_old)
        self.tau_new = float(tau_new)
        self.weight_old = float(weight_old)
        self.weight_new = float(weight_new)
        self.mode = mode
        self.correction_mode = correction_mode

    def forward(
        self,
        logits,
        labels,
        is_replay=None,
        replay_weights=None,
    ):
        if self.num_old_classes <= 0:
            return F.cross_entropy(logits, labels)
        if self.num_old_classes >= logits.shape[1]:
            raise ValueError(
                "num_old_classes must be smaller than output dimension"
            )

        scaled = torch.cat(
            (
                logits[:, : self.num_old_classes] / self.tau_old,
                logits[:, self.num_old_classes :] / self.tau_new,
            ),
            dim=1,
        )
        old = labels < self.num_old_classes

        if self.mode == "repo_dual":
            reference = logits[:, 0]
            sample_temperatures = torch.where(
                old,
                torch.full_like(reference, self.tau_old),
                torch.full_like(reference, self.tau_new),
            )
            scaled = scaled / sample_temperatures[:, None]

        losses = F.cross_entropy(scaled, labels, reduction="none")
        self.last_sample_ce = losses.detach()

        if (
            self.correction_mode == "replay_loss_experimental"
            and is_replay is not None
            and replay_weights is not None
        ):
            losses = torch.where(
                is_replay.bool(),
                losses * replay_weights.to(losses.device),
                losses,
            )

        if self.mode == "repo_dual":
            task_weights = torch.where(
                old,
                torch.full_like(losses, self.weight_old),
                torch.full_like(losses, self.weight_new),
            )
            return (losses * task_weights).mean()

        output = logits.sum() * 0.0
        if old.any():
            output = output + self.weight_old * losses[old].mean()
        if (~old).any():
            output = output + self.weight_new * losses[~old].mean()
        return output


