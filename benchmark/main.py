#!/usr/bin/env python3
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    Build a ResNet-18 model for CIFAR-10.
    Adjust this if your .pth was trained with a different architecture.
    """
    model = models.resnet18(weights=None)  # no pre-trained weights
    # Change the final layer to match CIFAR-10 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """
    Load weights from a .pth file into the model.
    Handles common cases: plain state_dict, or dict with 'state_dict' key,
    and optional 'module.' prefix from DataParallel.
    """
    state = torch.load(weights_path, map_location=device)

    # If saved as {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip 'module.' prefix if present (from nn.DataParallel)
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_key = k
            if k.startswith("module."):
                new_key = k[len("module.") :]
            new_state[new_key] = v
        state = new_state

    model.load_state_dict(state, strict=True)
    return model


def get_cifar10_test_loader(data_root: str, batch_size: int = 128, num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for CIFAR-10 test set.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,  # set True if you want it to auto-download
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader


def measure_inference_time(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Measure average inference time over the dataloader.
    Returns:
        total_time (seconds),
        avg_time_per_batch (seconds),
        avg_time_per_image (seconds)
    """
    model.eval()
    model.to(device)

    total_time = 0.0
    total_images = 0
    num_batches = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            batch_size = images.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            elapsed = end - start

            total_time += elapsed
            total_images += batch_size
            num_batches += 1

    avg_time_per_batch = total_time / num_batches
    avg_time_per_image = total_time / total_images

    return total_time, avg_time_per_batch, avg_time_per_image


def main():
    parser = argparse.ArgumentParser(
        description="Measure average inference time of ResNet-18 on CIFAR-10 using a .pth weights file."
    )
    parser.add_argument("weights_path", type=str, help="Path to the .pth weights file")
    parser.add_argument("data_root", type=str, help="Root directory of CIFAR-10 dataset")

    # Optional args (you can ignore these if you want strictly 2 arguments)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to use: "cuda" or "cpu"',
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Using device: {device}")

    # Build model and load weights
    model = build_resnet18_cifar10(num_classes=10)
    model = load_weights(model, args.weights_path, device)
    print("Model and weights loaded successfully.")

    # Data loader
    test_loader = get_cifar10_test_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Test dataset size: {len(test_loader.dataset)} images")

    # Measure inference time
    total_time, avg_batch, avg_image = measure_inference_time(model, test_loader, device)

    images_per_sec = 1.0 / avg_image if avg_image > 0 else float("inf")

    print("\n===== Inference Time Results =====")
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average time per batch: {avg_batch:.6f} seconds")
    print(f"Average time per image: {avg_image:.8f} seconds")
    print(f"Throughput: {images_per_sec:.2f} images/second")


if __name__ == "__main__":
    main()
