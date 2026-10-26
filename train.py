import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGE_SIZE = 224
SUPPORTED_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an educational ResNet-18 brain MRI image classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/brain_mri_resnet18.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_transforms() -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), normalize]
        ),
        "test": transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), normalize]
        ),
    }


def check_dataset(data_dir: Path) -> list[str]:
    for split in SUPPORTED_SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise SystemExit(f"Missing dataset split directory: {split_dir}")
    train = datasets.ImageFolder(data_dir / "train")
    validation = datasets.ImageFolder(data_dir / "val")
    test = datasets.ImageFolder(data_dir / "test")
    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise SystemExit("The train, val, and test splits must each contain images.")
    if train.class_to_idx != validation.class_to_idx or train.class_to_idx != test.class_to_idx:
        raise SystemExit("All dataset splits must contain the same class folder names.")
    if len(train.classes) < 2:
        raise SystemExit("At least two class folders are required for classification.")
    return train.classes


def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    return (
        total_loss / len(loader.dataset),
        float(accuracy_score(targets, predictions)),
        float(precision),
        float(recall),
        float(f1),
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive integers.")
    torch.manual_seed(args.seed)
    classes = check_dataset(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transforms = make_transforms()
    datasets_by_split = {
        split: datasets.ImageFolder(args.data_dir / split, transform=image_transforms[split])
        for split in SUPPORTED_SPLITS
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets_by_split.items()
    }

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy, _, _, _ = evaluate(
            model, loaders["val"], loss_fn, device
        )
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"validation_loss={val_loss:.4f} validation_accuracy={val_accuracy:.4f}"
        )
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    test_loss, accuracy, precision, recall, f1 = evaluate(
        model, loaders["test"], loss_fn, device
    )
    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "labels": classes,
            "metrics": metrics,
            "model_version": "resnet18-custom-v1",
        },
        args.output,
    )
    print(f"Test metrics: {metrics}")
    print(f"Saved model checkpoint to {args.output.resolve()}")


if __name__ == "__main__":
    main()
