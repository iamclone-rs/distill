"""
train_teacher.py
================
Stage 1: Fine-tune DFN5B (last N visual blocks) tren SBIR task.
    - Loss = CE(cls) + Triplet + NT-Xent  (y het student)
    - In baseline metrics truoc khi train
    - Sau moi epoch validate va in mAP / Precision@K
    - Save checkpoint tot nhat: dfn5b_finetuned.pt

Vi du chay (Google Colab):
    !python -m train_teacher \\
        --root /content/sketchy/Sketchy \\
        --dataset sketchy_2 \\
        --epochs 5 \\
        --n_blocks 4 \\
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Freeze helpers ───────────────────────────────────────────────────────────

def _freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze_layernorms(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)


def unfreeze_last_n_blocks(model: nn.Module, n: int = 4) -> int:
    """
    Freeze toan bo DFN5B, sau do mo:
      - Tat ca LayerNorm (moi noi trong model)
      - n transformer blocks cuoi cung cua visual encoder
      - ln_post va proj cua visual encoder (neu co)
      - logit_scale
    """
    _freeze_all(model)
    _unfreeze_layernorms(model)

    blocks = model.visual.transformer.resblocks
    total  = len(blocks)
    for i, block in enumerate(blocks):
        if i >= total - n:
            for p in block.parameters():
                p.requires_grad_(True)

    for attr in ("ln_post", "proj"):
        obj = getattr(model.visual, attr, None)
        if obj is None:
            continue
        if isinstance(obj, nn.Module):
            for p in obj.parameters():
                p.requires_grad_(True)
        elif isinstance(obj, torch.Tensor):
            obj.requires_grad_(True)

    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Teacher] Unfreeze last {n}/{total} blocks + all LayerNorm + proj"
        f" -> {trainable / 1e6:.1f}M trainable params"
    )
    return trainable


# ─── Wrapper model ────────────────────────────────────────────────────────────

class TeacherWrapper(nn.Module):
    """
    Boc DFN5B voi:
      - adapter rieng cho photo branch va sketch branch (residual, alpha=0.1)
      - cache text features theo classnames
    """

    DIM = 1024  # DFN5B output dim

    def __init__(self, dfn_model: nn.Module, tokenizer):
        super().__init__()
        self.model     = dfn_model
        self.tokenizer = tokenizer
        self._text_cache: dict = {}

        def _make_adapter():
            return nn.Sequential(
                nn.Linear(self.DIM, self.DIM // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(self.DIM // 4, self.DIM, bias=False),
            ).float().to(device)

        self.adapter_photo  = _make_adapter()
        self.adapter_sketch = _make_adapter()
        self.alpha = 0.1

    def _base_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x).float()

    def encode_photo(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._base_encode(x)
        return self.alpha * self.adapter_photo(feat) + (1 - self.alpha) * feat

    def encode_sketch(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._base_encode(x)
        return self.alpha * self.adapter_sketch(feat) + (1 - self.alpha) * feat

    @torch.no_grad()
    def get_text_features(self, classnames: list) -> torch.Tensor:
        key = tuple(classnames)
        if key not in self._text_cache:
            prompts = [
                "a photo/sketch of " + n.replace("_", " ") + "."
                for n in classnames
            ]
            tokens = self.tokenizer(prompts).to(device)
            tf = self.model.encode_text(tokens).float()
            tf = F.normalize(tf, dim=-1)
            self._text_cache[key] = tf
        return self._text_cache[key]

    def get_logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp()


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(wrapper: TeacherWrapper, batch: tuple, classnames: list, args) -> tuple:
    photo, sketch, _ph_aug, _sk_aug, neg, label = batch
    photo  = photo.to(device)
    sketch = sketch.to(device)
    neg    = neg.to(device)
    label  = label.to(device)

    ph_feat = wrapper.encode_photo(photo)
    sk_feat = wrapper.encode_sketch(sketch)
    ng_feat = wrapper.encode_photo(neg)

    ph_norm = F.normalize(ph_feat, dim=-1)
    sk_norm = F.normalize(sk_feat, dim=-1)
    ng_norm = F.normalize(ng_feat, dim=-1)

    # Classification loss
    text_feat = wrapper.get_text_features(classnames)
    ls        = wrapper.get_logit_scale()
    ph_logits = ls * ph_norm @ text_feat.T
    sk_logits = ls * sk_norm @ text_feat.T
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
def validate(wrapper: TeacherWrapper, sk_loader: DataLoader, ph_loader: DataLoader, args) -> float:
    wrapper.eval()
    use_amp = device.type == "cuda"

    sk_feats, sk_labels = [], []
    ph_feats, ph_labels = [], []

    # ── Extract features (FP16 autocast for 2-3x speedup on GPU) ──────────────
    for images, labels in tqdm(sk_loader, desc="  Extract sketch", leave=False):
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            feat = wrapper.encode_sketch(images.to(device))
        feat = F.normalize(feat.float(), dim=-1)
        sk_feats.append(feat.cpu())
        sk_labels.append(labels)

    for images, labels in tqdm(ph_loader, desc="  Extract photo ", leave=False):
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            feat = wrapper.encode_photo(images.to(device))
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
    parser.add_argument("--n_blocks",        type=int,   default=4,
                        help="So blocks cuoi DFN5B se unfreeze (mac dinh=4)")
    parser.add_argument("--log_every",       type=int,   default=50)

    # Loss
    parser.add_argument("--lambda_cls",      type=float, default=1.0)
    parser.add_argument("--lambda_triplet",  type=float, default=1.0)
    parser.add_argument("--lambda_nt_xent",  type=float, default=1.0)

    # Output
    parser.add_argument("--save_path",       type=str,   default="dfn5b_finetuned.pt")

    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("=" * 60)
    print(f"  dataset={args.dataset}  epochs={args.epochs}  "
          f"n_blocks={args.n_blocks}  lr={args.lr}  bs={args.batch_size}")
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

    unfreeze_last_n_blocks(dfn_model, n=args.n_blocks)
    wrapper = TeacherWrapper(dfn_model, tokenizer).to(device)

    # Datasets
    train_ds   = TrainDataset(args, args.proportion)
    val_sketch = ValidDataset(args, mode="sketch")
    val_photo  = ValidDataset(args)

    lkw = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
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
    validate(wrapper, sk_loader, ph_loader, args)
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
        mAP = validate(wrapper, sk_loader, ph_loader, args)

        if mAP > best_map:
            best_map = mAP
            torch.save(
                {
                    "epoch":   epoch,
                    "mAP":     mAP,
                    "args":    vars(args),
                    "dfn5b_state_dict":          dfn_model.state_dict(),
                    "adapter_photo_state_dict":  wrapper.adapter_photo.state_dict(),
                    "adapter_sketch_state_dict": wrapper.adapter_sketch.state_dict(),
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
