"""Fine-tune residual sketch/photo adapters on frozen DFN5B features.

This follows the repository protocol: train on all seen classes, validate on
unseen classes after every epoch, and select the best checkpoint by unseen mAP.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data_config import UNSEEN_CLASSES
from src.eval_teacher import (
    IMAGE_EXTENSIONS,
    classification_metrics,
    encode_images,
    encode_text,
    make_loader,
    retrieval_at_k,
    resolve_metric_config,
)
from src.teacher_adapter import (
    DFN_MODEL_NAME,
    DFN_PRETRAINED,
    ModalityAdapters,
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


def list_images(class_dir):
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_seen(root, classnames):
    root = Path(root)
    train = {"sketch": [], "photo": []}

    for label, classname in enumerate(classnames):
        for modality in ("sketch", "photo"):
            paths = list_images(root / modality / classname)
            if not paths:
                raise RuntimeError(f"No {modality} files found for seen class '{classname}'")
            train[modality].extend((path, label) for path in paths)

    return train


def collect_unseen(root, classnames):
    root = Path(root)
    samples = {"sketch": [], "photo": []}
    for label, classname in enumerate(classnames):
        for modality in ("sketch", "photo"):
            paths = list_images(root / modality / classname)
            if not paths:
                raise RuntimeError(f"No {modality} files found for unseen class '{classname}'")
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


@torch.inference_mode()
def adapt_features(adapter, features, device, batch_size=4096):
    adapter.eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size].to(device).float()
        outputs.append(adapter(batch).cpu())
    return torch.cat(outputs)


def evaluate_split(
    adapters,
    feature_set,
    text_set,
    device,
    map_k,
    precision_k,
    retrieval_chunk_size,
    description,
):
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
            map_k=map_k,
            precision_k=precision_k,
            chunk_size=retrieval_chunk_size,
            description=description,
            show_progress=False,
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
            "adapter_mode": "residual",
            "model": DFN_MODEL_NAME,
            "pretrained": DFN_PRETRAINED,
            "dataset": args.dataset,
            "seen_classes": seen_classes,
            "unseen_classes": unseen_classes,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(UNSEEN_CLASSES))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bottleneck_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_retrieval", type=float, default=1.0)
    parser.add_argument("--lambda_semantic", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    map_k, precision_k = resolve_metric_config(args.dataset)
    map_name = "all" if map_k is None else str(map_k)
    map_metric_key = f"mAP@{map_name}"
    precision_metric_key = f"P@{precision_k}_project_compatible"

    unseen_classes = UNSEEN_CLASSES[args.dataset]
    all_classes = get_all_classes(args.root)
    seen_classes = sorted(set(all_classes) - set(unseen_classes))
    missing_unseen = sorted(set(unseen_classes) - set(all_classes))
    if missing_unseen:
        raise RuntimeError(f"Unseen class directories are missing: {missing_unseen}")
    print(
        f"Protocol: train on all {len(seen_classes)} seen classes; "
        f"validate on {len(unseen_classes)} unseen classes. "
        f"Metrics: {map_metric_key}, P@{precision_k}."
    )

    train_samples = collect_seen(args.root, seen_classes)
    unseen_samples = collect_unseen(args.root, unseen_classes)

    print(f"Loading frozen backbone {DFN_MODEL_NAME} ({DFN_PRETRAINED})...")
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        DFN_MODEL_NAME, pretrained=DFN_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(DFN_MODEL_NAME)
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
                description=f"Encode {name}/{modality}",
            )
        return result

    train_features = encode_sample_group("seen_train", train_samples)
    unseen_features = encode_sample_group("unseen_validation", unseen_samples)

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
    for feature_group in (train_features, unseen_features):
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
    print(
        "Residual adapter trainable parameters: "
        f"{trainable:,} ({trainable / 1e6:.3f}M)"
    )
    optimizer = torch.optim.AdamW(
        adapters.parameters(), lr=args.lr, weight_decay=1e-2
    )
    scheduler = make_scheduler(
        optimizer, args.epochs * len(train_loader), warmup_fraction=0.05
    )
    seen_text_gpu = {key: value.to(device) for key, value in seen_text.items()}

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    global_step = 0

    initial_unseen_metrics = evaluate_split(
        adapters,
        unseen_features,
        unseen_text,
        device,
        map_k,
        precision_k,
        256,
        "Epoch 0 - unseen evaluation",
    )
    initial_metrics = {
        "epoch": 0,
        "global_step": 0,
        "train": None,
        "unseen_validation": initial_unseen_metrics,
    }
    initial_unseen_retrieval = initial_unseen_metrics["sketch_to_photo"]
    print(
        f"Epoch 0/{args.epochs} | "
        f"val {map_metric_key}={initial_unseen_retrieval[map_metric_key]:.4f} "
        f"P@{precision_k}={initial_unseen_retrieval[precision_metric_key]:.4f} "
        f"sketch@1={initial_unseen_metrics['sketch_zero_shot']['top1']:.4f}"
    )
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
        output_dir / "best.pt",
        adapters,
        args,
        0,
        initial_metrics,
        seen_classes,
        unseen_classes,
    )
    best_map = initial_unseen_metrics["sketch_to_photo"][map_metric_key]

    for epoch in range(1, args.epochs + 1):
        adapters.train()
        totals = {"total": 0.0, "retrieval": 0.0, "semantic": 0.0}

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} - train",
            unit="batch",
        )
        for base_sketch, base_photo, labels in progress:
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
            loss = (
                args.lambda_retrieval * loss_retrieval
                + args.lambda_semantic * loss_semantic
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            totals["total"] += loss.item()
            totals["retrieval"] += loss_retrieval.item()
            totals["semantic"] += loss_semantic.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_metrics = {key: value / len(train_loader) for key, value in totals.items()}
        unseen_metrics = evaluate_split(
            adapters,
            unseen_features,
            unseen_text,
            device,
            map_k,
            precision_k,
            256,
            f"Epoch {epoch} - unseen evaluation",
        )

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "unseen_validation": unseen_metrics,
        }
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
        current_map = unseen_metrics["sketch_to_photo"][map_metric_key]
        is_best = current_map > best_map
        if is_best:
            best_map = current_map
            save_checkpoint(
                output_dir / "best.pt",
                adapters,
                args,
                epoch,
                epoch_metrics,
                seen_classes,
                unseen_classes,
            )
        unseen_retrieval = unseen_metrics["sketch_to_photo"]
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_metrics['total']:.4f} "
            f"ret={train_metrics['retrieval']:.4f} "
            f"sem={train_metrics['semantic']:.4f} | "
            f"val {map_metric_key}={unseen_retrieval[map_metric_key]:.4f} "
            f"P@{precision_k}={unseen_retrieval[precision_metric_key]:.4f} "
            f"sketch@1={unseen_metrics['sketch_zero_shot']['top1']:.4f}"
            f"{' *' if is_best else ''}"
        )

    print(f"Training complete. Metrics and checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
