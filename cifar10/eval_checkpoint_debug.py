import argparse
import importlib
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from spikingjelly.clock_driven import functional
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.utils import accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Spikingformer on CIFAR-10")
    parser.add_argument("--checkpoint", default="/home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_FLIF01_no_sto-20260621-003238/model_best.pth.tar", type=str, help="Path to .pth.tar checkpoint")
    parser.add_argument("--data-dir", default="/home/wangyufei/dataset", type=str, help="CIFAR-10 root directory")
    parser.add_argument("--model-file", default="model_4layers_baseline_FLIF01_no_sto", type=str,
                        help="Python model module to import, e.g. model or model_4layers_random")
    parser.add_argument("--config", default="/home/wangyufei/code/SNNTransformer/Spikingformer/cifar10/cifar10.yml", type=str, help="Model/eval config YAML")
    parser.add_argument("--dataset", default=None, type=str, help="Dataset name, defaults to config dataset")
    parser.add_argument("--val-split", default=None, type=str, help="Validation split, defaults to config val_split")
    parser.add_argument("--batch-size", default=None, type=int, help="Validation batch size")
    parser.add_argument("--workers", default=None, type=int, help="DataLoader workers")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device, for example cuda or cuda:0")
    parser.add_argument("--no-prefetcher", action="store_true", help="Disable timm prefetcher")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory layout")
    parser.add_argument("--tta", default=0, type=int, help="Test-time augmentation reduction factor")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even if enabled in config")
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


def import_model_module(model_file):
    module_name = model_file[:-3] if model_file.endswith(".py") else model_file
    return importlib.import_module(module_name)


def build_model(cfg, model_module):
    return model_module.vit_snn(
        drop_rate=0.0,
        drop_path_rate=0.2,
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


def build_loader(cfg, args, net, device):
    eval_args = dict(cfg)
    eval_args["dataset"] = args.dataset or cfg.get("dataset", "torch/cifar10")
    eval_args["val_split"] = args.val_split or cfg.get("val_split", "validation")
    eval_args["batch_size"] = args.batch_size or cfg["val_batch_size"]
    eval_args["input_size"] = cfg.get("input_size", None)
    eval_args["crop_pct"] = cfg.get("crop_pct", None)
    eval_args["interpolation"] = cfg.get("interpolation", "")
    eval_args["mean"] = cfg.get("mean", None)
    eval_args["std"] = cfg.get("std", None)

    data_config = resolve_data_config(eval_args, model=net, verbose=True)
    dataset_eval = create_dataset(
        eval_args["dataset"],
        root=args.data_dir,
        split=eval_args["val_split"],
        is_training=False,
        batch_size=eval_args["batch_size"],
    )
    use_prefetcher = (not args.no_prefetcher) and device.type == "cuda"

    return create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=eval_args["batch_size"],
        is_training=False,
        use_prefetcher=use_prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers if args.workers is not None else cfg["workers"],
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=True,
    )


def evaluate(net, loader, device, args, amp_autocast=suppress):
    criterion = nn.CrossEntropyLoss().to(device)
    net.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            if not ((not args.no_prefetcher) and device.type == "cuda"):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                outputs = net(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if args.tta > 1:
                outputs = outputs.unfold(0, args.tta, args.tta).mean(dim=2)
                targets = targets[0:targets.size(0):args.tta]
            loss = criterion(outputs, targets)
            functional.reset_net(net)

            top1, top5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_top1 += top1.item() * batch_size
            total_top5 += top5.item() * batch_size
            total_samples += batch_size

            if device.type == "cuda":
                torch.cuda.synchronize()

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model_module = import_model_module(args.model_file)
    net = build_model(cfg, model_module).to(device)
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)

    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    loader = build_loader(cfg, args, net, device)
    use_amp = bool(cfg.get("amp", False)) and not args.no_amp and device.type == "cuda"
    amp_autocast = torch.cuda.amp.autocast if use_amp else suppress
    metrics = evaluate(net, loader, device, args, amp_autocast=amp_autocast)
    print(f"loss: {metrics['loss']:.4f}")
    print(f"top1: {metrics['top1']:.4f}")
    print(f"top5: {metrics['top5']:.4f}")


if __name__ == "__main__":
    main()
