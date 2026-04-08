import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.baseline import DGBaselineResNet
from models.gradient_reversal import GradientReversalResNet
from models.contrastive import ContrastiveResNet
from models.combined import CombinedObjectiveResNet

from data.dataset import SpectrogramDataset
from data.collate import (
    collate_with_labels,
    collate_with_domain_labels
)

from losses.focal_loss import FocalLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.ntxent_loss import NTXentLoss


# ===============================
# CONFIG
# ===============================

METHOD = "baseline"   # baseline | grl | contrastive | combined

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# LODO PROTOCOL
# ===============================

protocols = [
    [["CHINESE", "MALAY"], ["HINDI"]],
    [["CHINESE", "HINDI"], ["MALAY"]],
    [["MALAY", "HINDI"], ["CHINESE"]],
]


# ===============================
# MODEL SELECTION
# ===============================

def build_model():

    if METHOD == "baseline":
        model = DGBaselineResNet()

    elif METHOD == "grl":
        model = GradientReversalResNet()

    elif METHOD == "contrastive":
        model = ContrastiveResNet()

    elif METHOD == "combined":
        model = CombinedObjectiveResNet()

    else:
        raise ValueError("Unknown method")

    return model.to(DEVICE)


# ===============================
# TRAIN LOOP
# ===============================

def train_one_epoch(
    dataloader,
    model,
    optimizer,
    loss_fn,
    extra_loss=None
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:

        optimizer.zero_grad()

        if METHOD == "grl":
            spec, labels, domain = batch
            spec = spec.to(DEVICE)
            labels = labels.to(DEVICE)
            domain = domain.to(DEVICE)

            pred, domain_pred = model(spec)

            loss = loss_fn(pred, labels)
            domain_loss = nn.CrossEntropyLoss()(domain_pred, domain)

            loss = loss + domain_loss

        else:
            spec, labels = batch
            spec = spec.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(spec)

            if METHOD in ["contrastive", "combined"]:
                pred, projection = pred

                loss = loss_fn(pred, labels)

                if extra_loss:
                    loss += extra_loss(projection, labels)

            else:
                loss = loss_fn(pred, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        correct += (
            torch.argmax(pred, dim=1) == labels
        ).sum().item()

        total += labels.size(0)

    acc = 100 * correct / total
    return total_loss / len(dataloader), acc


# ===============================
# MAIN TRAINING
# ===============================

for train_domains, test_domains in protocols:

    print("Train:", train_domains)
    print("Test:", test_domains)

    return_domain = METHOD == "grl"

    train_dataset = SpectrogramDataset(
        annotations_file="path/to/all_samples.csv",
        spec_dir="path/to/spectrograms",
        domains=train_domains,
        return_domain_label=return_domain
    )

    test_dataset = SpectrogramDataset(
        annotations_file="path/to/all_samples.csv",
        spec_dir="path/to/spectrograms",
        domains=test_domains,
        return_domain_label=return_domain
    )

    if return_domain:
        collate_fn = collate_with_domain_labels
    else:
        collate_fn = collate_with_labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_fn = nn.CrossEntropyLoss()

    extra_loss = None

    if METHOD == "contrastive":
        extra_loss = ContrastiveLoss()

    if METHOD == "combined":
        extra_loss = NTXentLoss()
        loss_fn = FocalLoss()

    best_acc = 0

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(
            train_loader,
            model,
            optimizer,
            loss_fn,
            extra_loss
        )

        print(
            f"Epoch {epoch+1} "
            f"Loss {train_loss:.4f} "
            f"Acc {train_acc:.2f}"
        )

        best_acc = max(best_acc, train_acc)

    print("Best Accuracy:", best_acc)
