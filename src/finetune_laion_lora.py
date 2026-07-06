"""Jointly fine-tune modality-specific Q/V LoRA and the output adapter."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_config import UNSEEN_CLASSES
from src.eval_laion_sketchy import (
    classification_metrics,
    parse_map_k,
    resolve_metric_config,
    retrieval_at_k,
)
from src.finetune_laion_adapter import (
    ModalityAdapters,
    PathDataset,
    collect_seen,
    collect_unseen,
    get_all_classes,
    multi_positive_cross_modal_loss,
    retention_loss,
    semantic_loss,
)
from src.finetune_laion_layernorm import (
    ImagePairDataset,
    build_text_tokens,
    encode_class_text,
    make_autocast,
    make_scheduler,
    seed_worker,
)
from src.laion_lora import (
    inject_visual_qv_lora,
    load_lora_state_dict,
    lora_named_parameters,
    lora_parameter_count,
    lora_state_dict,
    set_lora_modality,
)


def load_adapter(path, model_name, pretrained, output_dim, device):
    checkpoint = torch.load(path, map_location="cpu")
    required = {"adapter_state_dict", "feature_dim", "bottleneck_dim"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(f"Adapter checkpoint is missing keys: {sorted(missing)}")
    if checkpoint.get("model") not in (None, model_name):
        raise RuntimeError(
            f"Adapter model mismatch: {checkpoint.get('model')} != {model_name}"
        )
    if checkpoint.get("pretrained") not in (None, pretrained):
        raise RuntimeError(
            "Adapter pretrained mismatch: "
            f"{checkpoint.get('pretrained')} != {pretrained}"
        )
    feature_dim = int(checkpoint["feature_dim"])
    if feature_dim != output_dim:
        raise RuntimeError(
            f"Adapter feature_dim={feature_dim}, teacher output_dim={output_dim}."
        )
    adapters = ModalityAdapters(
        feature_dim=feature_dim,
        bottleneck_dim=int(checkpoint["bottleneck_dim"]),
    )
    adapters.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    return adapters.to(device=device, dtype=torch.float32), checkpoint


@torch.inference_mode()
def encode_images(model, adapter, loader, modality, device, precision, description):
    model.eval()
    adapter.eval()
    set_lora_modality(model, modality)
    features = []
    labels = []
    for images, targets in tqdm(loader, desc=description, unit="batch", leave=False):
        images = images.to(device, non_blocking=True)
        with make_autocast(device, precision):
            base_features = model.encode_image(images)
        base_features = F.normalize(base_features.float(), dim=-1)
        features.append(adapter(base_features).cpu())
        labels.append(targets.long())
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def evaluate(
    model,
    adapters,
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
    sketch_features, sketch_labels = encode_images(
        model,
        adapters.sketch,
        sketch_loader,
        "sketch",
        device,
        precision,
        f"Epoch {epoch} · val sketch",
    )
    photo_features, photo_labels = encode_images(
        model,
        adapters.photo,
        photo_loader,
        "photo",
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


def save_checkpoint(
    path,
    model,
    adapters,
    args,
    epoch,
    metrics,
    seen_classes,
    unseen_classes,
):
    torch.save(
        {
            "epoch": epoch,
            "lora_state_dict": lora_state_dict(model),
            "adapter_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in adapters.state_dict().items()
            },
            "feature_dim": adapters.sketch.norm.normalized_shape[0],
            "bottleneck_dim": adapters.sketch.down.out_features,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.lora_dropout,
            "num_layers": args.lora_layers,
            "target": "visual_qv",
            "modalities": ["sketch", "photo"],
            "model": args.model,
            "pretrained": args.pretrained,
            "dataset": args.dataset,
            "adapter_ckpt": args.adapter_ckpt,
            "joint_adapter": not args.freeze_adapter,
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
    parser.add_argument("--adapter_ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lora_layers", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter_lr", type=float, default=1e-5)
    parser.add_argument("--freeze_adapter", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_retrieval", type=float, default=1.0)
    parser.add_argument("--lambda_semantic", type=float, default=0.5)
    parser.add_argument("--lambda_retain", type=float, default=0.0)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_augmentation", action="store_true", default=False)
    parser.add_argument("--map_k", type=parse_map_k, default="auto")
    parser.add_argument("--precision_k", type=int, default=0)
    parser.add_argument("--retrieval_chunk_size", type=int, default=256)
    parser.add_argument("--max_train_per_class", type=int, default=None)
    parser.add_argument("--max_eval_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="teacher_lora_runs/laion_h_sketchy2_r16_l4")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch_size must be at least 2")
    if args.rank != 16:
        print(f"Warning: this experiment was designed for rank 16, got rank={args.rank}.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.precision != "fp32":
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

    print(f"Loading {args.model} ({args.pretrained})...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.requires_grad_(False)
    layer_indices = inject_visual_qv_lora(
        model,
        num_layers=args.lora_layers,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
    )
    if args.gradient_checkpointing:
        if not hasattr(model, "set_grad_checkpointing"):
            raise RuntimeError("This OpenCLIP model does not support gradient checkpointing.")
        model.set_grad_checkpointing(True)
    model = model.to(device)

    output_dim = int(
        getattr(
            model.visual,
            "output_dim",
            model.text_projection.shape[-1],
        )
    )

    adapters, adapter_checkpoint = load_adapter(
        args.adapter_ckpt,
        args.model,
        args.pretrained,
        output_dim,
        device,
    )
    adapter_count = sum(parameter.numel() for parameter in adapters.parameters())
    adapters.requires_grad_(not args.freeze_adapter)
    lora_count = lora_parameter_count(model)
    visual_width = int(model.visual.transformer.width)
    expected_count = (
        args.lora_layers * 2 * 2 * args.rank * (visual_width + visual_width)
    )
    print(
        f"LoRA visual blocks={layer_indices}, rank={args.rank}, alpha={args.alpha}, "
        f"trainable={lora_count:,} ({lora_count / 1e6:.3f}M)"
    )
    if lora_count != expected_count:
        raise RuntimeError(f"Unexpected LoRA count: {lora_count} != {expected_count}")
    print(
        f"Output adapter={adapter_count:,} "
        f"({'frozen' if args.freeze_adapter else f'trainable, lr={args.adapter_lr:g}'}); "
        f"total PEFT storage={adapter_count + lora_count:,}. Adapter source epoch="
        f"{adapter_checkpoint.get('epoch', 'unknown')}"
    )

    train_transform = preprocess_train if args.train_augmentation else preprocess_val
    train_dataset = ImagePairDataset(
        train_samples["sketch"], train_samples["photo"], train_transform
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
    model.eval()
    with torch.inference_mode(), make_autocast(device, args.precision):
        seen_text = encode_class_text(
            model, seen_tokens, len(seen_classes), device, args.precision
        )
    seen_text = {key: value.to(device) for key, value in seen_text.items()}

    lora_parameters = [parameter for _, parameter in lora_named_parameters(model)]
    adapter_parameters = list(adapters.parameters()) if not args.freeze_adapter else []
    trainable_parameters = lora_parameters + adapter_parameters
    parameter_groups = [
        {"params": lora_parameters, "lr": args.lr},
    ]
    if adapter_parameters:
        parameter_groups.append(
            {"params": adapter_parameters, "lr": args.adapter_lr}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer, args.epochs * len(train_loader), args.warmup_fraction
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and args.precision == "fp16"
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    initial_metrics = evaluate(
        model,
        adapters,
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
        adapters,
        args,
        0,
        initial_record,
        seen_classes,
        unseen_classes,
    )
    save_checkpoint(
        output_dir / "best.pt",
        model,
        adapters,
        args,
        0,
        initial_record,
        seen_classes,
        unseen_classes,
    )
    best_map = initial_metrics["sketch_to_photo"][map_metric_key]
    print(
        f"Epoch 0/{args.epochs} | val {map_metric_key}={best_map:.4f} "
        f"P@{precision_k}="
        f"{initial_metrics['sketch_to_photo'][precision_metric_key]:.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        adapters.eval() if args.freeze_adapter else adapters.train()
        totals = {"total": 0.0, "retrieval": 0.0, "semantic": 0.0, "retain": 0.0}
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} · LoRA train",
            unit="batch",
        )
        for sketches, photos, labels in progress:
            sketches = sketches.to(device, non_blocking=True)
            photos = photos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with make_autocast(device, args.precision):
                # A single paired forward is required for correct modality routing
                # when OpenCLIP recomputes transformer blocks during checkpointed
                # backward. The first half is sketch, the second half is photo.
                set_lora_modality(model, "paired")
                raw_features = model.encode_image(torch.cat((sketches, photos), dim=0))
                raw_sketch, raw_photo = raw_features.float().chunk(2, dim=0)
                raw_sketch = F.normalize(raw_sketch, dim=-1)
                raw_photo = F.normalize(raw_photo, dim=-1)
                adapted_sketch = adapters.sketch(raw_sketch)
                adapted_photo = adapters.photo(raw_photo)
                loss_retrieval = multi_positive_cross_modal_loss(
                    adapted_sketch, adapted_photo, labels, args.temperature
                )
                loss_semantic = semantic_loss(
                    adapted_sketch,
                    adapted_photo,
                    labels,
                    seen_text["sketch"],
                    seen_text["photo"],
                    args.temperature,
                )
                loss_retain = retention_loss(
                    adapted_sketch, adapted_photo, raw_sketch, raw_photo
                )
                loss = (
                    args.lambda_retrieval * loss_retrieval
                    + args.lambda_semantic * loss_semantic
                    + args.lambda_retain * loss_retain
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
            totals["retain"] += loss_retain.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_metrics = {key: value / len(train_loader) for key, value in totals.items()}
        validation_metrics = evaluate(
            model,
            adapters,
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
            adapters,
            args,
            epoch,
            record,
            seen_classes,
            unseen_classes,
        )
        retrieval = validation_metrics["sketch_to_photo"]
        current_map = retrieval[map_metric_key]
        is_best = current_map > best_map
        if is_best:
            best_map = current_map
            save_checkpoint(
                output_dir / "best.pt",
                model,
                adapters,
                args,
                epoch,
                record,
                seen_classes,
                unseen_classes,
            )
        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_metrics['total']:.4f} "
            f"ret={train_metrics['retrieval']:.4f} "
            f"sem={train_metrics['semantic']:.4f} | "
            f"val {map_metric_key}={current_map:.4f} "
            f"P@{precision_k}={retrieval[precision_metric_key]:.4f}"
            f"{' *' if is_best else ''}"
        )

    print(f"Training complete. LoRA checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
