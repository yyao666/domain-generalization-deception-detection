import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.baseline import DGBaselineResNet
from models.gradient_reversal import GradientReversalResNet
from models.contrastive import ContrastiveResNet
from models.combined import CombinedObjectiveResNet

from data.dataset import SpectrogramDataset
from data.collate import collate_with_labels, collate_with_domain_labels

from losses.focal_loss import FocalLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.ntxent_loss import NTXentLoss


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(config, device):
    method = config["method"]
    model_cfg = config["model"]
    pretrained = config["training"]["pretrained"]

    if method == "baseline":
        model = DGBaselineResNet(
            num_classes=model_cfg["num_classes"],
            dropout=model_cfg["dropout"],
            pretrained=pretrained,
        )

    elif method == "grl":
        model = GradientReversalResNet(
            num_classes=model_cfg["num_classes"],
            num_domains=model_cfg["num_domains"],
            dropout=model_cfg["dropout"],
            lambda_grl=model_cfg["lambda_grl"],
            pretrained=pretrained,
        )

    elif method == "contrastive":
        model = ContrastiveResNet(
            num_classes=model_cfg["num_classes"],
            projection_dim=model_cfg["projection_dim"],
            dropout=model_cfg["dropout"],
            pretrained=pretrained,
        )

    elif method == "combined":
        model = CombinedObjectiveResNet(
            num_classes=model_cfg["num_classes"],
            projection_dim=model_cfg["projection_dim"],
            dropout=model_cfg["dropout"],
            pretrained=pretrained,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return model.to(device)


def build_losses(config):
    method = config["method"]
    loss_cfg = config["loss"]

    main_loss = nn.CrossEntropyLoss()
    extra_losses = {}

    if method == "contrastive":
        extra_losses["contrastive"] = ContrastiveLoss(
            temperature=loss_cfg["temperature"]
        )

    elif method == "combined":
        main_loss = nn.CrossEntropyLoss()
        extra_losses["focal"] = FocalLoss(
            alpha=loss_cfg["alpha"],
            gamma=loss_cfg["gamma"],
        )
        extra_losses["ntxent"] = NTXentLoss(
            temperature=loss_cfg["temperature"]
        )

    elif method == "grl":
        extra_losses["domain"] = nn.CrossEntropyLoss()

    return main_loss, extra_losses


def train_one_epoch(dataloader, model, optimizer, method, main_loss_fn, extra_losses, config, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    alpha = config["loss"]["alpha"]
    beta = config["loss"]["beta"]

    for batch in dataloader:
        optimizer.zero_grad()

        if method == "grl":
            specs, labels, domain_labels = batch
            specs = specs.to(device)
            labels = labels.to(device)
            domain_labels = domain_labels.to(device)

            deception_logits, domain_logits = model(specs)

            deception_loss = main_loss_fn(deception_logits, labels)
            domain_loss = extra_losses["domain"](domain_logits, domain_labels)

            loss = alpha * deception_loss + (1.0 - alpha) * domain_loss
            logits_for_acc = deception_logits

        elif method == "contrastive":
            specs, labels = batch
            specs = specs.to(device)
            labels = labels.to(device)

            deception_logits, projection = model(specs)

            deception_loss = main_loss_fn(deception_logits, labels)
            contrastive_loss = extra_losses["contrastive"](projection, labels)

            loss = deception_loss + beta * contrastive_loss
            logits_for_acc = deception_logits

        elif method == "combined":
            specs, labels = batch
            specs = specs.to(device)
            labels = labels.to(device)

            deception_logits, projection = model(specs)

            deception_loss = main_loss_fn(deception_logits, labels)
            focal_loss = extra_losses["focal"](deception_logits, labels)


            ntxent_loss = extra_losses["ntxent"](projection, projection)

            loss = deception_loss + alpha * focal_loss + beta * ntxent_loss
            logits_for_acc = deception_logits

        else:  # baseline
            specs, labels = batch
            specs = specs.to(device)
            labels = labels.to(device)

            deception_logits = model(specs)
            loss = main_loss_fn(deception_logits, labels)
            logits_for_acc = deception_logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (torch.argmax(logits_for_acc, dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def evaluate_one_epoch(dataloader, model, method, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if method == "grl":
                specs, labels, _ = batch
            else:
                specs, labels = batch

            specs = specs.to(device)
            labels = labels.to(device)

            outputs = model(specs)

            if method in ["grl", "contrastive", "combined"]:
                logits = outputs[0]
            else:
                logits = outputs

            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def main():
    config = load_config("config.yaml")

    method = config["method"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    protocols = config["protocols"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)

    for protocol in protocols:
        train_domains = protocol["train"]
        test_domains = protocol["test"]

        print(f"\nTrain domains: {train_domains}")
        print(f"Test domains: {test_domains}")

        return_domain_label = method == "grl"
        collate_fn = collate_with_domain_labels if return_domain_label else collate_with_labels

        train_dataset = SpectrogramDataset(
            annotations_file=data_cfg["annotations_file"],
            spec_dir=data_cfg["spectrogram_dir"],
            domains=train_domains,
            language_mode=data_cfg["language_mode"],
            return_domain_label=return_domain_label,
        )

        test_dataset = SpectrogramDataset(
            annotations_file=data_cfg["annotations_file"],
            spec_dir=data_cfg["spectrogram_dir"],
            domains=test_domains,
            language_mode=data_cfg["language_mode"],
            return_domain_label=return_domain_label,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg["num_workers"],
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
            collate_fn=collate_fn,
        )

        model = build_model(config, device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
        )
        main_loss_fn, extra_losses = build_losses(config)

        best_val_acc = 0.0
        heldout_name = "_".join(test_domains).lower()
        checkpoint_path = f"checkpoints/{method}_{heldout_name}_best.pth"

        for epoch in range(train_cfg["num_epochs"]):
            train_loss, train_acc = train_one_epoch(
                train_loader,
                model,
                optimizer,
                method,
                main_loss_fn,
                extra_losses,
                config,
                device,
            )

            val_loss, val_acc = evaluate_one_epoch(
                test_loader,
                model,
                method,
                nn.CrossEntropyLoss(),
                device,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)

            print(
                f"Epoch {epoch + 1:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

        print(f"Best Val Accuracy ({heldout_name}): {best_val_acc:.2f}%")
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
