"""
train_teacher.py
================
Stage 1: Fine-tune DFN5B-CoPrompt teacher tren SBIR task.
    - Loss = CE(cls) + Triplet + NT-Xent  (y het student)
    - In baseline metrics truoc khi train
    - Sau moi epoch validate va in mAP / Precision@K
    - Save checkpoint tot nhat: dfn5b_finetuned.pt

Vi du chay (Google Colab):
    !python -m train_teacher \\
        --root /content/sketchy/Sketchy \\
        --dataset sketchy_2 \\
        --epochs 5 \\
        --batch_size 32 \\
        --lr 2e-5 \\
        --workers 8
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
from torchmetrics.functional import retrieval_average_precision

from src.sketchy_dataset import TrainDataset, ValidDataset
from src.utils import get_all_categories, retrieval_precision
from src.losses import nt_xent
from src.dfn_coprompt_teacher import DFNCoPromptTeacher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(wrapper: DFNCoPromptTeacher, batch: tuple, classnames: list, args) -> tuple:
    photo, sketch, _ph_aug, _sk_aug, neg, label = batch
    photo  = photo.to(device)
    sketch = sketch.to(device)
    neg    = neg.to(device)
    label  = label.to(device)

    ph_logits, ph_norm, ph_feat = wrapper.get_logits(photo, classnames, modality="photo")
    sk_logits, sk_norm, sk_feat = wrapper.get_logits(sketch, classnames, modality="sketch")
    _, ng_norm, _ng_feat = wrapper.get_logits(neg, classnames, modality="photo")

    loss_cls  = (
        F.cross_entropy(ph_logits, label)
        + F.cross_entropy(sk_logits, label)
    )

    # Triplet loss
    dist_fn      = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet_fn   = nn.TripletMarginWithDistanceLoss(distance_function=dist_fn, margin=0.2)
    loss_triplet = triplet_fn(sk_norm, ph_norm, ng_norm)

    # NT-Xent
    loss_nt = nt_xent(ph_feat, sk_feat)

    total = (
        args.lambda_cls     * loss_cls
        + args.lambda_triplet * loss_triplet
        + args.lambda_nt_xent * loss_nt
    )
    return total, loss_cls.item(), loss_triplet.item(), loss_nt.item()


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(wrapper: DFNCoPromptTeacher, sk_loader: DataLoader, ph_loader: DataLoader, classnames: list, args) -> float:
    wrapper.eval()
    use_amp = device.type == "cuda"

    sk_feats, sk_labels = [], []
    ph_feats, ph_labels = [], []

    # ── Extract features (FP16 autocast for 2-3x speedup on GPU) ──────────────
    for images, labels in tqdm(sk_loader, desc="  Extract sketch", leave=False):
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            feat = wrapper.encode_sketch(images.to(device), classnames)
        feat = F.normalize(feat.float(), dim=-1)
        sk_feats.append(feat.cpu())
        sk_labels.append(labels)

    for images, labels in tqdm(ph_loader, desc="  Extract photo ", leave=False):
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            feat = wrapper.encode_photo(images.to(device), classnames)
        feat = F.normalize(feat.float(), dim=-1)
        ph_feats.append(feat.cpu())
        ph_labels.append(labels)

    sk_feats  = torch.cat(sk_feats)            # (Nq, D)
    ph_feats  = torch.cat(ph_feats)            # (Ng, D)
    sk_labels = torch.cat(sk_labels).numpy()
    ph_labels = torch.cat(ph_labels).numpy()

    map_k = 200 if args.dataset == "sketchy_2" else 0
    p_k   = 200 if args.dataset in ("sketchy_2", "quickdraw") else 100

    # ── Retrieval: batched matmul thay vì vòng lặp per-query ──────────────────
    # sim_matrix: (Nq, Ng)  — cosine similarity vì cả 2 đã normalize
    sim_matrix = sk_feats @ ph_feats.T         # (Nq, Ng)

    aps, precs = [], []
    for idx in range(len(sk_feats)):
        cat    = sk_labels[idx]
        dist   = sim_matrix[idx]               # (Ng,) — đã tính sẵn
        target = torch.tensor(ph_labels == cat, dtype=torch.bool)

        ap = (
            retrieval_average_precision(dist, target, top_k=min(map_k, len(ph_feats)))
            if map_k else
            retrieval_average_precision(dist, target)
        )
        aps.append(ap)
        precs.append(retrieval_precision(dist, target, top_k=p_k))

    mAP  = torch.stack(aps).mean().item()
    prec = torch.stack(precs).mean().item()
    lbl  = f"@{map_k}" if map_k else "@all"
    print(f"  [Val] mAP{lbl}: {mAP:.4f}   P@{p_k}: {prec:.4f}")
    wrapper.train()
    return mAP


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Fine-tune DFN5B teacher on SBIR")

    # Dataset
    parser.add_argument("--root",            type=str,   default="../datasets/Sketchy")
    parser.add_argument("--dataset",         type=str,   default="sketchy_2")
    parser.add_argument("--use_classes",     type=int,   default=104)
    parser.add_argument("--max_size",        type=int,   default=224)
    parser.add_argument("--n_ctx",           type=int,   default=1)
    parser.add_argument("--prompt_depth",    type=int,   default=12)
    parser.add_argument("--proportion",      type=float, default=1.0)
    parser.add_argument("--data_split",      type=int,   default=-1)
    parser.add_argument("--gzs",             action="store_true", default=False)
    parser.add_argument("--no_aug",          action="store_true", default=False)
    parser.add_argument("--visualize",       action="store_true", default=False)

    # Training
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--test_batch_size", type=int,   default=512)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--workers",         type=int,   default=4)
    parser.add_argument("--log_every",       type=int,   default=50)
    parser.add_argument("--seed",            type=int,   default=42)

    # Loss
    parser.add_argument("--lambda_cls",      type=float, default=1.0)
    parser.add_argument("--lambda_triplet",  type=float, default=1.0)
    parser.add_argument("--lambda_nt_xent",  type=float, default=1.0)

    # Output
    parser.add_argument("--save_path",       type=str,   default="dfn5b_finetuned.pt")

    args = parser.parse_args()

    seed_everything(args.seed)

    print("=" * 60)
    print(f"  dataset={args.dataset}  epochs={args.epochs}  "
          f"n_ctx={args.n_ctx}  prompt_depth={args.prompt_depth}  "
          f"lr={args.lr}  bs={args.batch_size}  seed={args.seed}")
    print(f"  loss -> cls={args.lambda_cls}  "
          f"triplet={args.lambda_triplet}  nt_xent={args.lambda_nt_xent}")
    print("=" * 60)

    # Load DFN5B
    print("\n[Teacher] Loading DFN5B-CLIP-H/14 (ViT-H-14-quickgelu, dfn5b)...")
    dfn_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", pretrained="dfn5b"
    )
    tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
    dfn_model = dfn_model.to(device)

    wrapper = DFNCoPromptTeacher(args, dfn_model, tokenizer).to(device)
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    print(f"[Teacher] DFN-CoPrompt trainable params: {trainable / 1e6:.2f}M")

    # Datasets
    train_ds   = TrainDataset(args, args.proportion)
    val_sketch = ValidDataset(args, mode="sketch")
    val_photo  = ValidDataset(args)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    lkw = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **lkw
    )
    sk_loader = DataLoader(val_sketch, batch_size=args.test_batch_size, shuffle=False, **lkw)
    ph_loader = DataLoader(val_photo,  batch_size=args.test_batch_size, shuffle=False, **lkw)

    classnames = get_all_categories(args)
    print(f"\n[Data] Seen classes: {len(classnames)}"
          f"  |  Train: {len(train_ds)}"
          f"  |  Val sketch: {len(val_sketch)}"
          f"  |  Val photo: {len(val_photo)}\n")

    # Optimizer + Scheduler
    trainable_params = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_map = 0.0

    # Baseline truoc khi train
    print("[Baseline] DFN5B zero-shot (chua fine-tune):")
    validate(wrapper, sk_loader, ph_loader, classnames, args)
    print()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, lc, lt, ln = compute_loss(wrapper, batch, classnames, args)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{lc:.3f}",
                             tri=f"{lt:.3f}", ntx=f"{ln:.3f}")

        avg = epoch_loss / len(train_loader)
        print(f"\n[Epoch {epoch}] avg_loss={avg:.4f}")
        mAP = validate(wrapper, sk_loader, ph_loader, classnames, args)

        if mAP > best_map:
            best_map = mAP
            torch.save(
                {
                    "epoch":   epoch,
                    "mAP":     mAP,
                    "args":    vars(args),
                    "teacher_type": "dfn_coprompt",
                    "teacher_coprompt_state_dict": wrapper.state_dict(),
                },
                args.save_path,
            )
            print(f"  Best saved -> {args.save_path}  (mAP={mAP:.4f})")

        scheduler.step()
        print()

    print("=" * 60)
    print(f"[Done] Best mAP = {best_map:.4f}")
    print(f"       Checkpoint -> {os.path.abspath(args.save_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
