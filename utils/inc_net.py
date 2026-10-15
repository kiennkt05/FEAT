"""ResNet-18 with the fixed simplex ETF classifier used by FEAT."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from convs.resnet import resnet18


def simplex_etf(num_classes, feature_dim, *, device=None, dtype=None):
    """Return unit length ETF prototypes as rows (C, d)."""
    if num_classes < 1 or num_classes > feature_dim:
        raise ValueError("ETF requires 1 <= num_classes <= feature_dim")
    basis, _ = torch.linalg.qr(
        torch.randn(feature_dim, num_classes, device=device, dtype=dtype), mode="reduced"
    )
    centered = torch.eye(num_classes, device=device, dtype=dtype)
    centered -= torch.ones_like(centered) / num_classes
    if num_classes == 1:
        return basis.T.contiguous()
    return (num_classes / (num_classes - 1)) ** 0.5 * (basis @ centered).T.contiguous()


class ETFClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("weight", torch.empty(0, feature_dim))

    def update(self, num_classes):
        self.weight = simplex_etf(
            num_classes, self.feature_dim, device=self.weight.device, dtype=self.weight.dtype
        )

    def forward(self, features):
        return F.linear(features, self.weight)


class IncrementalNet_CausalETF(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()
        if args.get("net", "resnet18").lower() != "resnet18":
            raise ValueError("FEAT currently supports the ResNet-18 backbone")
        self.convnet = resnet18(pretrained=pretrained, args=args)
        self.feature_dim = self.convnet.out_dim
        self.fc = ETFClassifier(self.feature_dim)

    def update_fc(self, num_classes):
        self.fc.update(num_classes)

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x, P_H=None, P_T=None, bar_eH_G=None, r_H=None, r_T=None):
        output = self.convnet(x)
        features = output["features"]
        if P_H is not None and P_T is not None and bar_eH_G is not None:
            normalized = F.normalize(features, dim=1)
            head = normalized @ P_H
            tail = normalized @ P_T
            e_head = head.square().sum(dim=1) / max(int(r_H), 1)
            e_tail = tail.square().sum(dim=1) / max(int(r_T), 1)
            gate = ((e_head - bar_eH_G) / (e_head + e_tail + 1e-8)).clamp_min(0)
            corrected = F.normalize(normalized - gate[:, None] * head + gate[:, None] * tail, dim=1)
            logits = self.fc(corrected)
        else:
            logits = self.fc(features)
        output["logits"] = logits
        return output

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
