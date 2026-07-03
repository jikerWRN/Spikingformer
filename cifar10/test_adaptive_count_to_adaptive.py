#!/usr/bin/env python3
"""Evaluate a checkpoint produced by train_adaptive_count_to_adaptive.py."""

import argparse
import importlib
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from spikingjelly.clock_driven import functional
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.utils import accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Test adaptive Spikingformer on CIFAR-10")
    parser.add_argument("--checkpoint", default="/home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count_to_adaptive_fix_UPB_sj_back-20260702-224147/model_best.pth.tar", help="Path to model_best.pth.tar")


    parser.add_argument("--data-dir", default="/home/wangyufei/dataset", help="CIFAR-10 root (overrides the config)")
    parser.add_argument("--config", default=str(Path(__file__).with_name("cifar10.yml")))
    parser.add_argument("--model-file", default="model_4layers_baseline_ILIF01_count_to_adaptive_fix_UPB_sj_back",
                        help="Model module/path; defaults to the value saved in the checkpoint")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision")
    parser.add_argument("--use-ema", action="store_true", help="Load state_dict_ema when available")
    parser.add_argument("--strict", action="store_true", help="Require an exact checkpoint/model match")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--spike-percentile", type=float, nargs="+", default=None,
                        help="Positive activation percentiles used by the adaptive spike upper bound")
    parser.add_argument("--spike-selected-percentile", type=float, default=None,
                        help="Selected percentile used as the adaptive spike upper bound")
    return parser.parse_args()


def as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return vars(value)


def import_model(model_file):
    path = Path(model_file)
    if path.suffix == ".py" or path.parent != Path("."):
        path = path.resolve()
        sys.path.insert(0, str(path.parent))
        module_name = path.stem
    else:
        # Match the training script's usual execution from the cifar10 directory.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        module_name = model_file
    importlib.import_module(module_name)  # registers Spikingformer with timm


def build_model(cfg):
    return create_model(
        "Spikingformer", pretrained=False, drop_rate=0.0, drop_path_rate=0.2,
        drop_block_rate=None, img_size_h=cfg["img_size"], img_size_w=cfg["img_size"],
        patch_size=cfg["patch_size"], embed_dims=cfg["dim"], num_heads=cfg["num_heads"],
        mlp_ratios=cfg["mlp_ratio"], in_channels=3, num_classes=cfg["num_classes"],
        qkv_bias=False, depths=cfg["depths"], sr_ratios=1, T=cfg["time_step"],
        spike_percentile=cfg.get("spike_percentile", [0.7, 0.8, 0.9, 0.99]),
        spike_selected_percentile=cfg.get("spike_selected_percentile", 0.99),
    )


def normalize_state_dict(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def evaluate(model, loader, device, amp, log_interval):
    criterion = nn.CrossEntropyLoss().to(device)
    totals = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "samples": 0}
    model.eval()
    amp_context = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) \
        if amp and device.type == "cuda" else nullcontext

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with amp_context():
                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss = criterion(outputs, targets)
            top1, top5 = accuracy(outputs, targets, topk=(1, 5))
            n = targets.size(0)
            totals["loss"] += loss.item() * n
            totals["top1"] += top1.item() * n
            totals["top5"] += top5.item() * n
            totals["samples"] += n
            functional.reset_net(model)
            if batch_idx % log_interval == 0:
                print(f"Test [{batch_idx:4d}/{len(loader)}]  "
                      f"loss {loss.item():.4f}  acc@1 {top1.item():.2f}  acc@5 {top5.item():.2f}")

    n = totals.pop("samples")
    return {name: value / n for name, value in totals.items()}


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    # Saved training arguments take precedence and reproduce the trained architecture.
    cfg.update(as_dict(checkpoint.get("args") if isinstance(checkpoint, dict) else None))
    if args.spike_percentile is not None:
        cfg["spike_percentile"] = args.spike_percentile
    if args.spike_selected_percentile is not None:
        cfg["spike_selected_percentile"] = args.spike_selected_percentile
    model_file = args.model_file or cfg.get("model_file")
    if not model_file:
        raise ValueError("Model module is unavailable; pass --model-file")
    import_model(model_file)
    model = build_model(cfg).to(device)

    state_key = "state_dict_ema" if args.use_ema and "state_dict_ema" in checkpoint else "state_dict"
    state_dict = checkpoint.get(state_key, checkpoint)
    incompatible = model.load_state_dict(normalize_state_dict(state_dict), strict=args.strict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"Checkpoint mismatch: {len(incompatible.missing_keys)} missing, "
              f"{len(incompatible.unexpected_keys)} unexpected keys")

    data_dir = args.data_dir or cfg.get("data_dir")
    if not data_dir:
        raise ValueError("Dataset root is unavailable; pass --data-dir")
    data_cfg = resolve_data_config(cfg, model=model, verbose=True)
    dataset = create_dataset(cfg.get("dataset", "torch/cifar10"), root=data_dir,
                             split=cfg.get("val_split", "validation"), is_training=False)
    loader = create_loader(
        dataset, input_size=data_cfg["input_size"], batch_size=args.batch_size or cfg.get("val_batch_size", 64),
        is_training=False, use_prefetcher=False, interpolation=data_cfg["interpolation"],
        mean=data_cfg["mean"], std=data_cfg["std"],
        num_workers=args.workers if args.workers is not None else cfg.get("workers", 4),
        distributed=False, crop_pct=data_cfg["crop_pct"], pin_memory=device.type == "cuda")

    metrics = evaluate(model, loader, device, args.amp, args.log_interval)
    epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
    print(f"Checkpoint epoch: {epoch}")
    print(f"Loss: {metrics['loss']:.4f}  Acc@1: {metrics['top1']:.4f}  Acc@5: {metrics['top5']:.4f}")


if __name__ == "__main__":
    main()
