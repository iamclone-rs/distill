"""Fine-tune only LayerNorm parameters of an OpenCLIP teacher for SBIR.

Protocol follows the project: train on all seen classes, validate on unseen
classes after every epoch, and select the best checkpoint by unseen mAP.
No feature adapter is used and no image features are precomputed because the
backbone LayerNorm parameters change during training.
"""

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data_config import UNSEEN_CLASSES
from src.eval_laion_sketchy import (
    classification_metrics,
    parse_map_k,
    resolve_metric_config,
    retrieval_at_k,
)
from src.finetune_laion_adapter import (
    PathDataset,
    collect_seen,
    collect_unseen,
    get_all_classes,
    multi_positive_cross_modal_loss,
    semantic_loss,
)


class ImagePairDataset(Dataset):
    """One sketch with a randomly sampled same-class photo."""

    def __init__(self, sketch_samples, photo_samples, transform):
        self.sketch_samples = sketch_samples
        self.transform = transform
        self.photo_paths = {}
        for path, label in photo_samples:
            self.photo_paths.setdefault(int(label), []).append(path)

    def __len__(self):
        return len(self.sketch_samples)

    def __getitem__(self, index):
        sketch_path, label = self.sketch_samples[index]
        photo_path = random.choice(self.photo_paths[int(label)])
        with Image.open(sketch_path) as image:
            sketch = self.transform(image.convert("RGB"))
        with Image.open(photo_path) as image:
            photo = self.transform(image.convert("RGB"))
        return sketch, photo, int(label)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_autocast(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def enable_layernorm_only(model):
    model.requires_grad_(False)
    layernorm_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.requires_grad_(True)
            layernorm_modules.append(name)
    if not layernorm_modules:
        raise RuntimeError("No torch.nn.LayerNorm modules found in the teacher model.")
    return layernorm_modules


@torch.inference_mode()
def encode_images(model, loader, device, precision, description):
    features = []
    labels = []
    for images, targets in tqdm(
        loader, desc=description, unit="batch", leave=False
    ):
        images = images.to(device, non_blocking=True)
        with make_autocast(device, precision):
            image_features = model.encode_image(images)
        features.append(F.normalize(image_features.float(), dim=-1).cpu())
        labels.append(targets.long())
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def encode_class_text(model, tokens, class_count, device, precision):
    with make_autocast(device, precision):
        features = model.encode_text(tokens)
    features = F.normalize(features.float(), dim=-1)
    return {
        "sketch": features[:class_count].cpu(),
        "photo": features[class_count:].cpu(),
    }


def build_text_tokens(tokenizer, classnames, device):
    sketch_prompts = [
        f"a sketch of a {name.replace('_', ' ')}." for name in classnames
    ]
    photo_prompts = [
        f"a photo of a {name.replace('_', ' ')}." for name in classnames
    ]
    return tokenizer(sketch_prompts + photo_prompts).to(device)


def evaluate(
    model,
    sketch_loader,
    photo_loader,
    text_tokens,
    class_count,
    device,
    precision,
    map_k,
    precision_k,
    retrieval_chunk_size,
    epoch,
):
    model.eval()
    sketch_features, sketch_labels = encode_images(
        model,
        sketch_loader,
        device,
        precision,
        f"Epoch {epoch} · val sketch",
    )
    photo_features, photo_labels = encode_images(
        model,
        photo_loader,
        device,
        precision,
        f"Epoch {epoch} · val photo",
    )
    text_features = encode_class_text(
        model, text_tokens, class_count, device, precision
    )
    result = {
        "sketch_zero_shot": classification_metrics(
            sketch_features, sketch_labels, text_features["sketch"], device
        ),
        "photo_zero_shot": classification_metrics(
            photo_features, photo_labels, text_features["photo"], device
        ),
        "sketch_to_photo": retrieval_at_k(
            sketch_features,
            sketch_labels,
            photo_features,
            photo_labels,
            device,
            map_k=map_k,
            precision_k=precision_k,
            chunk_size=retrieval_chunk_size,
            description=f"Epoch {epoch} · val retrieval",
            show_progress=True,
        ),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def make_scheduler(optimizer, total_steps, warmup_fraction):
    warmup_steps = int(total_steps * warmup_fraction)

    def schedule(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def layernorm_state_dict(model):
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    path,
    model,
    args,
    epoch,
    metrics,
    seen_classes,
    unseen_classes,
    layernorm_modules,
):
    torch.save(
        {
            "epoch": epoch,
            "layernorm_state_dict": layernorm_state_dict(model),
            "layernorm_modules": layernorm_modules,
            "model": args.model,
            "pretrained": args.pretrained,
            "dataset": args.dataset,
            "seen_classes": seen_classes,
            "unseen_classes": unseen_classes,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="sketchy_2", choices=sorted(UNSEEN_CLASSES))
    parser.add_argument("--model", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_retrieval", type=float, default=1.0)
    parser.add_argument("--lambda_semantic", type=float, default=0.5)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--map_k", type=parse_map_k, default="auto")
    parser.add_argument("--precision_k", type=int, default=0)
    parser.add_argument("--retrieval_chunk_size", type=int, default=256)
    parser.add_argument("--max_train_per_class", type=int, default=None)
    parser.add_argument("--max_eval_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="teacher_layernorm_runs/laion_h_sketchy2")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch_size must be at least 2 for cross-modal retrieval loss")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.precision != "fp32":
        print(f"Precision {args.precision} requested without CUDA; using fp32.")
        args.precision = "fp32"

    map_k, precision_k = resolve_metric_config(
        args.dataset, args.map_k, args.precision_k
    )
    map_name = "all" if map_k is None else str(map_k)
    map_metric_key = f"mAP@{map_name}"
    precision_metric_key = f"P@{precision_k}_project_compatible"

    unseen_classes = UNSEEN_CLASSES[args.dataset]
    all_classes = get_all_classes(args.root)
    seen_classes = sorted(set(all_classes) - set(unseen_classes))
    missing_unseen = sorted(set(unseen_classes) - set(all_classes))
    if missing_unseen:
        raise RuntimeError(f"Unseen class directories are missing: {missing_unseen}")

    train_samples = collect_seen(
        args.root,
        seen_classes,
        args.seed,
        max_train_per_class=args.max_train_per_class,
    )
    unseen_samples = collect_unseen(
        args.root,
        unseen_classes,
        max_eval_per_class=args.max_eval_per_class,
        seed=args.seed,
    )

    print(
        f"Protocol: LayerNorm-only, full {len(seen_classes)} seen classes → "
        f"{len(unseen_classes)} unseen validation classes. "
        f"Metrics: {map_metric_key}, P@{precision_k}."
    )
    print(f"Loading {args.model} ({args.pretrained})...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    layernorm_modules = enable_layernorm_only(model)
    if args.gradient_checkpointing:
        if not hasattr(model, "set_grad_checkpointing"):
            raise RuntimeError("This OpenCLIP model does not support gradient checkpointing.")
        model.set_grad_checkpointing(True)
    model = model.to(device)

    train_dataset = ImagePairDataset(
        train_samples["sketch"], train_samples["photo"], preprocess_train
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    sketch_val_loader = DataLoader(
        PathDataset(unseen_samples["sketch"], preprocess_val),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    photo_val_loader = DataLoader(
        PathDataset(unseen_samples["photo"], preprocess_val),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    seen_tokens = build_text_tokens(tokenizer, seen_classes, device)
    unseen_tokens = build_text_tokens(tokenizer, unseen_classes, device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(
        f"Trainable LayerNorm modules: {len(layernorm_modules)}, "
        f"parameters: {trainable_count:,} ({trainable_count / 1e6:.3f}M)"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(
        optimizer,
        args.epochs * len(train_loader),
        args.warmup_fraction,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and args.precision == "fp16"
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    initial_metrics = evaluate(
        model,
        sketch_val_loader,
        photo_val_loader,
        unseen_tokens,
        len(unseen_classes),
        device,
        args.precision,
        map_k,
        precision_k,
        args.retrieval_chunk_size,
        0,
    )
    initial_record = {"epoch": 0, "train": None, "unseen_validation": initial_metrics}
    metrics_path.write_text(json.dumps(initial_record) + "\n", encoding="utf-8")
    save_checkpoint(
        output_dir / "initial.pt",
        model,
        args,
        0,
        initial_record,
        seen_classes,
        unseen_classes,
        layernorm_modules,
    )
    save_checkpoint(
        output_dir / "best.pt",
        model,
        args,
        0,
        initial_record,
        seen_classes,
        unseen_classes,
        layernorm_modules,
    )
    initial_retrieval = initial_metrics["sketch_to_photo"]
    best_map = initial_retrieval[map_metric_key]
    print(
        f"Epoch 0/{args.epochs} | val {map_metric_key}={best_map:.4f} "
        f"P@{precision_k}={initial_retrieval[precision_metric_key]:.4f} "
        f"sketch@1={initial_metrics['sketch_zero_shot']['top1']:.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"total": 0.0, "retrieval": 0.0, "semantic": 0.0}
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} · LayerNorm train",
            unit="batch",
        )

        for sketches, photos, labels in progress:
            sketches = sketches.to(device, non_blocking=True)
            photos = photos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with make_autocast(device, args.precision):
                sketch_features = F.normalize(model.encode_image(sketches), dim=-1)
                photo_features = F.normalize(model.encode_image(photos), dim=-1)
                text_features = F.normalize(model.encode_text(seen_tokens), dim=-1)
                sketch_text = text_features[:len(seen_classes)]
                photo_text = text_features[len(seen_classes):]
                loss_retrieval = multi_positive_cross_modal_loss(
                    sketch_features, photo_features, labels, args.temperature
                )
                loss_semantic = semantic_loss(
                    sketch_features,
                    photo_features,
                    labels,
                    sketch_text,
                    photo_text,
                    args.temperature,
                )
                loss = (
                    args.lambda_retrieval * loss_retrieval
                    + args.lambda_semantic * loss_semantic
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            totals["total"] += loss.item()
            totals["retrieval"] += loss_retrieval.item()
            totals["semantic"] += loss_semantic.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_metrics = {key: value / len(train_loader) for key, value in totals.items()}
        validation_metrics = evaluate(
            model,
            sketch_val_loader,
            photo_val_loader,
            unseen_tokens,
            len(unseen_classes),
            device,
            args.precision,
            map_k,
            precision_k,
            args.retrieval_chunk_size,
            epoch,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "unseen_validation": validation_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        save_checkpoint(
            output_dir / "last.pt",
            model,
            args,
            epoch,
            record,
            seen_classes,
            unseen_classes,
            layernorm_modules,
        )

        retrieval = validation_metrics["sketch_to_photo"]
        current_map = retrieval[map_metric_key]
        is_best = current_map > best_map
        if is_best:
            best_map = current_map
            save_checkpoint(
                output_dir / "best.pt",
                model,
                args,
                epoch,
                record,
                seen_classes,
                unseen_classes,
                layernorm_modules,
            )
        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_metrics['total']:.4f} "
            f"ret={train_metrics['retrieval']:.4f} "
            f"sem={train_metrics['semantic']:.4f} | "
            f"val {map_metric_key}={current_map:.4f} "
            f"P@{precision_k}={retrieval[precision_metric_key]:.4f} "
            f"sketch@1={validation_metrics['sketch_zero_shot']['top1']:.4f}"
            f"{' *' if is_best else ''}"
        )

    print(f"Training complete. Metrics and LayerNorm checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
