import torch
import torch.nn as nn
import torchvision


class DGBaselineResNet(nn.Module):
    """
    DG baseline model for leave-one-domain-out (LODO) evaluation.

    Input:
        audio spectrogram tensor of shape [B, F, T]

    Output:
        deception logits of shape [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        if pretrained:
            backbone = torchvision.models.resnet50(
                weights="ResNet50_Weights.IMAGENET1K_V1"
            )
        else:
            backbone = torchvision.models.resnet50(weights=None)

        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.audio_backbone = backbone
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        features = self.audio_backbone(x.unsqueeze(1))
        logits = self.classifier(features)
        return logits
