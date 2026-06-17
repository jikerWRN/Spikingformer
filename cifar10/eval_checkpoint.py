import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from spikingjelly.clock_driven import functional
from timm.models import create_model
from timm.utils import accuracy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import model  # noqa: F401 registers Spikingformer with timm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Spikingformer on CIFAR-10")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .pth.tar checkpoint")
    parser.add_argument("--data-dir", required=True, type=str, help="CIFAR-10 root directory")
    parser.add_argument("--config", default="cifar10.yml", type=str, help="Model/eval config YAML")
    parser.add_argument("--batch-size", default=None, type=int, help="Validation batch size")
    parser.add_argument("--workers", default=None, type=int, help="DataLoader workers")
    parser.add_argument("--device", default="cuda", type=str, help="Device, for example cuda or cuda:0")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if not present")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "state_dict_ema"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        state_dict[key] = value
    return state_dict


def build_model(cfg):
    return create_model(
        "Spikingformer",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.2,
        drop_block_rate=None,
        img_size_h=cfg["img_size"],
        img_size_w=cfg["img_size"],
        patch_size=cfg["patch_size"],
        embed_dims=cfg["dim"],
        num_heads=cfg["num_heads"],
        mlp_ratios=cfg["mlp_ratio"],
        in_channels=3,
        num_classes=cfg["num_classes"],
        qkv_bias=False,
        depths=cfg["depths"],
        sr_ratios=1,
        T=cfg["time_step"],
    )


def build_loader(cfg, args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=args.download,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size or cfg["val_batch_size"],
        shuffle=False,
        num_workers=args.workers if args.workers is not None else cfg["workers"],
        pin_memory=True,
    )


def evaluate(net, loader, device):
    criterion = nn.CrossEntropyLoss().to(device)
    net.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = net(images)
            loss = criterion(outputs, targets)
            functional.reset_net(net)

            top1, top5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_top1 += top1.item() * batch_size
            total_top5 += top5.item() * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    net = build_model(cfg).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    loader = build_loader(cfg, args)
    metrics = evaluate(net, loader, device)
    print(f"loss: {metrics['loss']:.4f}")
    print(f"top1: {metrics['top1']:.4f}")
    print(f"top5: {metrics['top5']:.4f}")


if __name__ == "__main__":
    main()
