"""Lightweight modality-adapter fine-tuning for an OpenCLIP teacher on Sketchy.

The adapter is trained only on seen classes. Unseen classes are evaluated after
every epoch, but are never used for optimization or best-checkpoint selection.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data_config import UNSEEN_CLASSES
from src.eval_laion_sketchy import (
    IMAGE_EXTENSIONS,
    classification_metrics,
    encode_images,
    encode_text,
    make_loader,
    retrieval_at_k,
)


class PathDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, label


class FeaturePairDataset(Dataset):
    """Pairs every sketch feature with a random same-class photo feature."""

    def __init__(self, sketch_features, sketch_labels, photo_features, photo_labels):
        self.sketch_features = sketch_features
        self.sketch_labels = sketch_labels
        self.photo_features = photo_features
        self.photo_indices = {}
        for label in photo_labels.unique().tolist():
            self.photo_indices[int(label)] = torch.where(photo_labels == label)[0]

    def __len__(self):
        return len(self.sketch_features)

    def __getitem__(self, index):
        label = int(self.sketch_labels[index])
        candidates = self.photo_indices[label]
        photo_index = candidates[torch.randint(len(candidates), (1,)).item()]
        return (
            self.sketch_features[index],
            self.photo_features[photo_index],
            label,
        )


class ResidualAdapter(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        residual = self.up(F.gelu(self.down(self.norm(features))))
        return F.normalize(features + residual, dim=-1)


class ModalityAdapters(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim):
        super().__init__()
        self.sketch = ResidualAdapter(feature_dim, bottleneck_dim)
        self.photo = ResidualAdapter(feature_dim, bottleneck_dim)


def list_images(class_dir):
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def limit_paths(paths, limit, rng):
    if limit is None or len(paths) <= limit:
        return paths
    return sorted(rng.sample(paths, limit))


def collect_seen_splits(root, classnames, val_fraction, seed, max_train_per_class=None):
    root = Path(root)
    rng = random.Random(seed)
    train = {"sketch": [], "photo": []}
    val = {"sketch": [], "photo": []}

    for label, classname in enumerate(classnames):
        for modality in ("sketch", "photo"):
            paths = list_images(root / modality / classname)
            if len(paths) < 2:
                raise RuntimeError(
                    f"Need at least two {modality} files for class '{classname}', found {len(paths)}"
                )
            rng.shuffle(paths)
            val_count = max(1, int(round(len(paths) * val_fraction)))
            val_count = min(val_count, len(paths) - 1)
            val_paths = sorted(paths[:val_count])
            train_paths = limit_paths(sorted(paths[val_count:]), max_train_per_class, rng)
            train[modality].extend((path, label) for path in train_paths)
            val[modality].extend((path, label) for path in val_paths)

    return train, val


def collect_unseen(root, classnames, max_eval_per_class=None, seed=42):
    root = Path(root)
    rng = random.Random(seed)
    samples = {"sketch": [], "photo": []}
    for label, classname in enumerate(classnames):
        for modality in ("sketch", "photo"):
            paths = list_images(root / modality / classname)
            if not paths:
                raise RuntimeError(f"No {modality} files found for unseen class '{classname}'")
            paths = limit_paths(paths, max_eval_per_class, rng)
            samples[modality].extend((path, label) for path in paths)
    return samples


def get_all_classes(root):
    sketch_root = Path(root) / "sketch"
    photo_root = Path(root) / "photo"
    if not sketch_root.is_dir() or not photo_root.is_dir():
        raise FileNotFoundError(f"Expected sketch/ and photo/ under {root}")
    return sorted(
        path.name for path in sketch_root.iterdir()
        if path.is_dir() and (photo_root / path.name).is_dir()
    )


def multi_positive_cross_modal_loss(sketch_features, photo_features, labels, temperature):
    logits = sketch_features @ photo_features.t() / temperature
    positive_mask = labels[:, None].eq(labels[None, :]).float()
    targets_sketch = positive_mask / positive_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    targets_photo = positive_mask.t() / positive_mask.t().sum(dim=-1, keepdim=True).clamp(min=1)
    loss_sketch = -(targets_sketch * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss_photo = -(targets_photo * F.log_softmax(logits.t(), dim=-1)).sum(dim=-1).mean()
    return 0.5 * (loss_sketch + loss_photo)


def semantic_loss(sketch_features, photo_features, labels, sketch_text, photo_text, temperature):
    sketch_logits = sketch_features @ sketch_text.t() / temperature
    photo_logits = photo_features @ photo_text.t() / temperature
    return 0.5 * (
        F.cross_entropy(sketch_logits, labels)
        + F.cross_entropy(photo_logits, labels)
    )


def retention_loss(adapted_sketch, adapted_photo, base_sketch, base_photo):
    return 0.5 * (
        (1.0 - F.cosine_similarity(adapted_sketch, base_sketch, dim=-1)).mean()
        + (1.0 - F.cosine_similarity(adapted_photo, base_photo, dim=-1)).mean()
    )


@torch.inference_mode()
def adapt_features(adapter, features, device, batch_size=4096):
    adapter.eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].to(device).float()
        outputs.append(adapter(batch).cpu())
    return torch.cat(outputs)


def evaluate_split(adapters, feature_set, text_set, device, top_k, retrieval_chunk_size):
    sketch_features = adapt_features(adapters.sketch, feature_set["sketch"][0], device)
    photo_features = adapt_features(adapters.photo, feature_set["photo"][0], device)
    sketch_labels = feature_set["sketch"][1]
    photo_labels = feature_set["photo"][1]

    result = {
        "sketch_zero_shot": classification_metrics(
            sketch_features, sketch_labels, text_set["sketch"], device
        ),
        "photo_zero_shot": classification_metrics(
            photo_features, photo_labels, text_set["photo"], device
        ),
        "sketch_to_photo": retrieval_at_k(
            sketch_features,
            sketch_labels,
            photo_features,
            photo_labels,
            device,
            top_k=top_k,
            chunk_size=retrieval_chunk_size,
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
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(path, adapters, args, epoch, metrics, seen_classes, unseen_classes):
    torch.save(
        {
            "epoch": epoch,
            "adapter_state_dict": {
                key: value.detach().cpu() for key, value in adapters.state_dict().items()
            },
            "feature_dim": adapters.sketch.norm.normalized_shape[0],
            "bottleneck_dim": args.bottleneck_dim,
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
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_retrieval", type=float, default=1.0)
    parser.add_argument("--lambda_semantic", type=float, default=0.5)
    parser.add_argument("--lambda_retain", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--retrieval_chunk_size", type=int, default=256)
    parser.add_argument("--fp16_backbone", action="store_true")
    parser.add_argument("--max_train_per_class", type=int, default=None)
    parser.add_argument("--max_eval_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="teacher_adapter_runs/laion_h_sketchy2")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val_fraction must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16_backbone and device.type == "cuda"

    unseen_classes = UNSEEN_CLASSES[args.dataset]
    all_classes = get_all_classes(args.root)
    seen_classes = sorted(set(all_classes) - set(unseen_classes))
    missing_unseen = sorted(set(unseen_classes) - set(all_classes))
    if missing_unseen:
        raise RuntimeError(f"Unseen class directories are missing: {missing_unseen}")
    print(
        f"Protocol: train adapter on {len(seen_classes)} seen classes; "
        f"evaluate only on {len(unseen_classes)} unseen classes."
    )

    train_samples, seen_val_samples = collect_seen_splits(
        args.root,
        seen_classes,
        args.val_fraction,
        args.seed,
        max_train_per_class=args.max_train_per_class,
    )
    unseen_samples = collect_unseen(
        args.root,
        unseen_classes,
        max_eval_per_class=args.max_eval_per_class,
        seed=args.seed,
    )

    print(f"Loading frozen backbone {args.model} ({args.pretrained})...")
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    backbone = backbone.eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if use_fp16:
        backbone = backbone.half()

    def encode_sample_group(name, samples):
        result = {}
        for modality in ("sketch", "photo"):
            print(f"Encoding {name}/{modality}: {len(samples[modality])} files")
            result[modality] = encode_images(
                backbone,
                make_loader(
                    PathDataset(samples[modality], preprocess),
                    args.encode_batch_size,
                    args.workers,
                ),
                device,
                use_fp16,
            )
        return result

    train_features = encode_sample_group("seen_train", train_samples)
    seen_val_features = encode_sample_group("seen_val", seen_val_samples)
    unseen_features = encode_sample_group("unseen_eval", unseen_samples)

    seen_text = {
        "sketch": encode_text(
            backbone, tokenizer, seen_classes, "a sketch of a {}.", device, use_fp16
        ),
        "photo": encode_text(
            backbone, tokenizer, seen_classes, "a photo of a {}.", device, use_fp16
        ),
    }
    unseen_text = {
        "sketch": encode_text(
            backbone, tokenizer, unseen_classes, "a sketch of a {}.", device, use_fp16
        ),
        "photo": encode_text(
            backbone, tokenizer, unseen_classes, "a photo of a {}.", device, use_fp16
        ),
    }

    feature_dim = train_features["sketch"][0].shape[-1]
    backbone = backbone.cpu()
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # FP16 storage halves CPU memory; adapter batches are converted back to FP32.
    for feature_group in (train_features, seen_val_features, unseen_features):
        for modality in ("sketch", "photo"):
            feature_group[modality] = (
                feature_group[modality][0].half(),
                feature_group[modality][1],
            )

    pair_dataset = FeaturePairDataset(
        train_features["sketch"][0],
        train_features["sketch"][1],
        train_features["photo"][0],
        train_features["photo"][1],
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    adapters = ModalityAdapters(feature_dim, args.bottleneck_dim).to(device)
    trainable = sum(parameter.numel() for parameter in adapters.parameters())
    print(f"Adapter trainable parameters: {trainable:,} ({trainable / 1e6:.3f}M)")
    optimizer = torch.optim.AdamW(
        adapters.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(
        optimizer, args.epochs * len(train_loader), args.warmup_fraction
    )
    seen_text_gpu = {key: value.to(device) for key, value in seen_text.items()}

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    global_step = 0

    print("Evaluating epoch 0 identity adapters on held-out seen images...")
    initial_seen_metrics = evaluate_split(
        adapters,
        seen_val_features,
        seen_text,
        device,
        args.top_k,
        args.retrieval_chunk_size,
    )
    print("Evaluating epoch 0 identity adapters on unseen classes...")
    initial_unseen_metrics = evaluate_split(
        adapters,
        unseen_features,
        unseen_text,
        device,
        args.top_k,
        args.retrieval_chunk_size,
    )
    initial_metrics = {
        "epoch": 0,
        "global_step": 0,
        "train": None,
        "seen_validation": initial_seen_metrics,
        "unseen_evaluation": initial_unseen_metrics,
    }
    print(json.dumps(initial_metrics, indent=2))
    metrics_path.write_text(json.dumps(initial_metrics) + "\n", encoding="utf-8")
    save_checkpoint(
        output_dir / "initial.pt",
        adapters,
        args,
        0,
        initial_metrics,
        seen_classes,
        unseen_classes,
    )
    save_checkpoint(
        output_dir / "best_seen.pt",
        adapters,
        args,
        0,
        initial_metrics,
        seen_classes,
        unseen_classes,
    )
    best_seen_map = initial_seen_metrics["sketch_to_photo"][f"mAP@{args.top_k}"]

    for epoch in range(1, args.epochs + 1):
        adapters.train()
        totals = {"total": 0.0, "retrieval": 0.0, "semantic": 0.0, "retain": 0.0}

        for batch_index, (base_sketch, base_photo, labels) in enumerate(train_loader, start=1):
            base_sketch = F.normalize(
                base_sketch.to(device, non_blocking=True).float(), dim=-1
            )
            base_photo = F.normalize(
                base_photo.to(device, non_blocking=True).float(), dim=-1
            )
            labels = labels.to(device, non_blocking=True)
            adapted_sketch = adapters.sketch(base_sketch)
            adapted_photo = adapters.photo(base_photo)

            loss_retrieval = multi_positive_cross_modal_loss(
                adapted_sketch, adapted_photo, labels, args.temperature
            )
            loss_semantic = semantic_loss(
                adapted_sketch,
                adapted_photo,
                labels,
                seen_text_gpu["sketch"],
                seen_text_gpu["photo"],
                args.temperature,
            )
            loss_retain = retention_loss(
                adapted_sketch, adapted_photo, base_sketch, base_photo
            )
            loss = (
                args.lambda_retrieval * loss_retrieval
                + args.lambda_semantic * loss_semantic
                + args.lambda_retain * loss_retain
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            totals["total"] += loss.item()
            totals["retrieval"] += loss_retrieval.item()
            totals["semantic"] += loss_semantic.item()
            totals["retain"] += loss_retain.item()
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                    f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        train_metrics = {key: value / len(train_loader) for key, value in totals.items()}
        print(f"Evaluating epoch {epoch} on held-out seen images...")
        seen_metrics = evaluate_split(
            adapters,
            seen_val_features,
            seen_text,
            device,
            args.top_k,
            args.retrieval_chunk_size,
        )
        print(f"Evaluating epoch {epoch} on unseen classes (report only)...")
        unseen_metrics = evaluate_split(
            adapters,
            unseen_features,
            unseen_text,
            device,
            args.top_k,
            args.retrieval_chunk_size,
        )

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "seen_validation": seen_metrics,
            "unseen_evaluation": unseen_metrics,
        }
        print(json.dumps(epoch_metrics, indent=2))
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(epoch_metrics) + "\n")

        save_checkpoint(
            output_dir / "last.pt",
            adapters,
            args,
            epoch,
            epoch_metrics,
            seen_classes,
            unseen_classes,
        )
        seen_map = seen_metrics["sketch_to_photo"][f"mAP@{args.top_k}"]
        if seen_map > best_seen_map:
            best_seen_map = seen_map
            save_checkpoint(
                output_dir / "best_seen.pt",
                adapters,
                args,
                epoch,
                epoch_metrics,
                seen_classes,
                unseen_classes,
            )
            print(f"New best held-out seen mAP@{args.top_k}: {best_seen_map:.6f}")

    print(f"Training complete. Metrics and checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
