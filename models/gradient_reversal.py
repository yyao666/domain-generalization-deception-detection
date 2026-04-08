import torch
import torch.nn as nn
import torchvision
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_grl)


class GradientReversalResNet(nn.Module):
    """
    Domain-adversarial DG model with:
    - deception classifier
    - domain classifier
    - gradient reversal layer
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_domains: int = 3,
        dropout: float = 0.5,
        lambda_grl: float = 1.0,
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

        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            GradientReversalLayer(lambda_grl=lambda_grl),
            nn.Linear(in_features, num_domains),
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        features = self.audio_backbone(x.unsqueeze(1))

        deception_logits = self.deception_classifier(features)
        domain_logits = self.domain_classifier(features)

        return deception_logits, domain_logits
