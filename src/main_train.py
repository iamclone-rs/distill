"""Train the ViT-B/32 CoPrompt student with DFN5B sketch-to-photo KL."""

import argparse
import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.data_config import UNSEEN_CLASSES
from src.model import IMAGE_SIZE, ZS_SBIR
from src.sketchy_dataset import TrainDataset, ValidDataset
from src.utils import get_all_categories


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


def _loader_generator(seed):
    return torch.Generator().manual_seed(seed)


def get_dataloaders(args):
    train_dataset = TrainDataset(args.root, args.dataset, IMAGE_SIZE)
    val_sketch = ValidDataset(
        args.root,
        args.dataset,
        modality="sketch",
        visualize=args.visualize,
        image_size=IMAGE_SIZE,
    )
    val_photo = ValidDataset(
        args.root,
        args.dataset,
        modality="photo",
        generalized=args.gzs,
        visualize=args.visualize,
        image_size=IMAGE_SIZE,
    )

    common = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    if args.workers > 0:
        common["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=_loader_generator(args.seed),
        **common,
    )
    sketch_loader = DataLoader(
        val_sketch,
        batch_size=args.test_batch_size,
        shuffle=False,
        generator=_loader_generator(args.seed + 1),
        **common,
    )
    photo_loader = DataLoader(
        val_photo,
        batch_size=args.test_batch_size,
        shuffle=False,
        generator=_loader_generator(args.seed + 2),
        **common,
    )
    return train_loader, sketch_loader, photo_loader


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(UNSEEN_CLASSES))
    parser.add_argument("--teacher_adapter_ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--lambda_kd", type=float, default=2.5)
    parser.add_argument("--kd_temperature", type=float, default=0.07)
    parser.add_argument("--n_ctx", type=int, default=1)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_triplet", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--gzs", action="store_true")
    parser.add_argument("--exp_name", default="coprompt_dfn5b_kd")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.lambda_kd <= 0:
        raise ValueError("--lambda_kd must be positive in the cleaned KD pipeline.")
    if args.kd_temperature <= 0:
        raise ValueError("--kd_temperature must be positive.")

    seed_everything(args.seed)
    print(
        "[Run] DFN5B residual adapter -> ViT-B/32 student | "
        f"dataset={args.dataset}, batch={args.batch_size}, lr={args.lr}, "
        f"kd={args.lambda_kd}, temp={args.kd_temperature}, seed={args.seed}"
    )
    train_loader, sketch_loader, photo_loader = get_dataloaders(args)
    classnames = get_all_categories(args)
    model = ZS_SBIR(args=args, classname=classnames)

    logger = TensorBoardLogger("tb_logs", name=args.exp_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="mAP",
        dirpath=f"saved_models/{args.exp_name}",
        filename="{epoch:02d}-{mAP:.4f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        max_epochs=args.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=20)],
    )
    trainer.fit(model, train_loader, [sketch_loader, photo_loader])


if __name__ == "__main__":
    main()
