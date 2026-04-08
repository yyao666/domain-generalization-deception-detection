import torch
import torch.nn as nn
import torchvision


class ContrastiveResNet(nn.Module):
    """
    DG model with:
    - deception classifier
    - contrastive projection head
    """

    def __init__(
        self,
        num_classes: int = 2,
        projection_dim: int = 128,
        dropout: float = 0.5,
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

        self.deception_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        self.contrastive_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        features = self.audio_backbone(x.unsqueeze(1))

        deception_logits = self.deception_classifier(features)
        projection = self.contrastive_projection(features)

        return deception_logits, projection
