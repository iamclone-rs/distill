"""Zero-shot semantic and SBIR evaluation for an OpenCLIP teacher on Sketchy."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data_config import UNSEEN_CLASSES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderDataset(Dataset):
    def __init__(self, root, modality, classnames, transform, max_per_class=None, seed=42):
        self.transform = transform
        self.samples = []
        rng = random.Random(seed)

        for label, classname in enumerate(classnames):
            class_dir = Path(root) / modality / classname
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            paths = sorted(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not paths:
                raise RuntimeError(f"No images found in: {class_dir}")
            if max_per_class is not None and len(paths) > max_per_class:
                paths = sorted(rng.sample(paths, max_per_class))

            self.samples.extend((path, label) for path in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, label


def make_loader(dataset, batch_size, workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


@torch.inference_mode()
def encode_images(model, loader, device, use_fp16):
    all_features = []
    all_labels = []

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if use_fp16:
            images = images.half()
        features = F.normalize(model.encode_image(images), dim=-1)
        all_features.append(features.float().cpu())
        all_labels.append(labels.long())

        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"  encoded {batch_index}/{len(loader)} batches", flush=True)

    return torch.cat(all_features), torch.cat(all_labels)


@torch.inference_mode()
def encode_text(model, tokenizer, classnames, template, device, use_fp16):
    prompts = [template.format(name.replace("_", " ")) for name in classnames]
    tokens = tokenizer(prompts).to(device)
    features = model.encode_text(tokens)
    return F.normalize(features.float(), dim=-1).cpu()


def classification_metrics(image_features, labels, text_features, device, chunk_size=2048):
    text_features = text_features.to(device)
    top1_correct = 0
    top5_correct = 0
    top5 = min(5, text_features.shape[0])

    for start in range(0, len(image_features), chunk_size):
        features = image_features[start:start + chunk_size].to(device)
        targets = labels[start:start + chunk_size].to(device)
        predictions = (features @ text_features.t()).topk(top5, dim=-1).indices
        top1_correct += (predictions[:, 0] == targets).sum().item()
        top5_correct += (predictions == targets[:, None]).any(dim=1).sum().item()

    count = len(labels)
    return {
        "top1": top1_correct / count,
        "top5": top5_correct / count,
    }


def retrieval_at_k(
    query_features,
    query_labels,
    gallery_features,
    gallery_labels,
    device,
    top_k=200,
    chunk_size=256,
):
    """Category-level mAP@K and P@K with sketch queries and photo gallery."""
    gallery_features = gallery_features.to(device)
    gallery_labels = gallery_labels.to(device)
    actual_k = min(top_k, len(gallery_features))
    ap_values = []
    precision_values = []

    for start in range(0, len(query_features), chunk_size):
        queries = query_features[start:start + chunk_size].to(device)
        labels = query_labels[start:start + chunk_size].to(device)
        similarities = queries @ gallery_features.t()
        indices = similarities.topk(actual_k, dim=-1).indices
        relevant = gallery_labels[indices].eq(labels[:, None])

        cumulative_hits = relevant.cumsum(dim=-1)
        ranks = torch.arange(1, actual_k + 1, device=device, dtype=torch.float32)
        precision_at_rank = cumulative_hits / ranks
        relevant_total = gallery_labels[None, :].eq(labels[:, None]).sum(dim=-1)
        denominator = relevant_total.clamp(max=actual_k).clamp(min=1)
        average_precision = (precision_at_rank * relevant).sum(dim=-1) / denominator

        ap_values.append(average_precision.cpu())
        precision_values.append(relevant.float().mean(dim=-1).cpu())

        completed = min(start + chunk_size, len(query_features))
        print(f"  ranked {completed}/{len(query_features)} sketches", flush=True)

    return {
        f"mAP@{top_k}": torch.cat(ap_values).mean().item(),
        f"P@{top_k}": torch.cat(precision_values).mean().item(),
        "queries": len(query_features),
        "gallery": len(gallery_features),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root containing sketch/ and photo/.")
    parser.add_argument("--dataset", default="sketchy_2", choices=sorted(UNSEEN_CLASSES))
    parser.add_argument("--model", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retrieval_chunk_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Optional quick-test limit applied independently to sketch and photo classes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        print("FP16 requested without CUDA; using FP32.")

    print(f"Loading {args.model} ({args.pretrained}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.eval().to(device)
    if use_fp16:
        model = model.half()

    classnames = UNSEEN_CLASSES[args.dataset]
    sketch_dataset = FolderDataset(
        args.root,
        "sketch",
        classnames,
        preprocess,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    photo_dataset = FolderDataset(
        args.root,
        "photo",
        classnames,
        preprocess,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    print(
        f"Split={args.dataset}: {len(classnames)} classes, "
        f"{len(sketch_dataset)} sketches, {len(photo_dataset)} photos"
    )

    print("Encoding sketches...")
    sketch_features, sketch_labels = encode_images(
        model,
        make_loader(sketch_dataset, args.batch_size, args.workers),
        device,
        use_fp16,
    )
    print("Encoding photos...")
    photo_features, photo_labels = encode_images(
        model,
        make_loader(photo_dataset, args.batch_size, args.workers),
        device,
        use_fp16,
    )

    sketch_text = encode_text(
        model,
        tokenizer,
        classnames,
        "a sketch of a {}.",
        device,
        use_fp16,
    )
    photo_text = encode_text(
        model,
        tokenizer,
        classnames,
        "a photo of a {}.",
        device,
        use_fp16,
    )

    # Image encoding is complete; release the large teacher before ranking the gallery.
    model = model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "model": args.model,
        "pretrained": args.pretrained,
        "dataset": args.dataset,
        "classes": len(classnames),
        "sketch_zero_shot": classification_metrics(
            sketch_features, sketch_labels, sketch_text, device
        ),
        "photo_zero_shot": classification_metrics(
            photo_features, photo_labels, photo_text, device
        ),
        "sketch_to_photo": retrieval_at_k(
            sketch_features,
            sketch_labels,
            photo_features,
            photo_labels,
            device,
            top_k=args.top_k,
            chunk_size=args.retrieval_chunk_size,
        ),
    }

    print("\nResults")
    print(json.dumps(results, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
