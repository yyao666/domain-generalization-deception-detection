import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.baseline import DGBaselineResNet
from models.gradient_reversal import GradientReversalResNet
from models.contrastive import ContrastiveResNet
from models.combined import CombinedObjectiveResNet

from data.dataset import SpectrogramDataset
from data.collate import collate_with_labels, collate_with_domain_labels


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


def evaluate_one_epoch(dataloader, model, method, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

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

            if method in ["contrastive", "combined", "grl"]:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = config["method"]

    data_cfg = config["data"]
    train_cfg = config["training"]

    model = build_model(config, device)

    checkpoint_path = "best_model.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    return_domain_label = method == "grl"
    collate_fn = collate_with_domain_labels if return_domain_label else collate_with_labels

    for protocol in config["protocols"]:
        test_domains = protocol["test"]

        test_dataset = SpectrogramDataset(
            annotations_file=data_cfg["annotations_file"],
            spec_dir=data_cfg["spectrogram_dir"],
            domains=test_domains,
            language_mode=data_cfg["language_mode"],
            return_domain_label=return_domain_label,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
            collate_fn=collate_fn,
        )

        test_loss, test_acc = evaluate_one_epoch(
            test_loader,
            model,
            method,
            device,
        )

        print(f"Test domains: {test_domains}")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.2f}%")
        print("-" * 40)


if __name__ == "__main__":
    main() 
