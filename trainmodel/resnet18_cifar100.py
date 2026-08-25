from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF


FEDMIA_CIFAR100_MEAN = (0.4914, 0.4822, 0.4465)
FEDMIA_CIFAR100_STD = (0.2023, 0.1994, 0.2010)


class BasicBlock(nn.Module):
    """CIFAR-sized ResNet basic block used by the FedMIA reference model."""

    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(32, channels)
        if stride == 1 and in_channels == channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(32, channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.relu(self.norm1(self.conv1(inputs)), inplace=True)
        outputs = self.norm2(self.conv2(outputs))
        return F.relu(outputs + self.shortcut(inputs), inplace=True)


class FedMIAResNet18(nn.Module):
    """ResNet18 architecture used for the FedMIA CIFAR-100 experiments.

    This intentionally uses a 3x3, stride-one CIFAR stem and GroupNorm rather
    than torchvision's ImageNet stem and BatchNorm.  GroupNorm also means the
    federated state contains no client-local running-statistic buffers.
    """

    model_type = "resnet18"
    trainable_state_filename = "final_resnet18.pt"

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self._in_channels = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(32, 64)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(
        self, channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride, *([1] * (blocks - 1))]
        layers = []
        for block_stride in strides:
            layers.append(
                BasicBlock(self._in_channels, channels, stride=block_stride)
            )
            self._in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.relu(self.norm1(self.conv1(inputs)), inplace=True)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, output_size=1).flatten(1)
        return self.linear(outputs)


def _to_fedmia_cifar100_tensor(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone().float()
        if tensor.ndim != 3:
            raise ValueError("CIFAR-100 images must have shape [C, H, W].")
        if not tensor.is_floating_point() or tensor.max() > 1.0:
            tensor = tensor / 255.0
    else:
        tensor = TF.to_tensor(image)
    return TF.normalize(tensor, FEDMIA_CIFAR100_MEAN, FEDMIA_CIFAR100_STD)


def collate_fedmia_cifar100(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the paper's deterministic, no-augmentation CIFAR transform."""

    images, labels = zip(*batch)
    return (
        torch.stack([_to_fedmia_cifar100_tensor(image) for image in images]),
        torch.as_tensor(labels, dtype=torch.long),
    )
