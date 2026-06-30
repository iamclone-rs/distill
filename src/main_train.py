import os
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import ModelCheckpoint 

from src.sketchy_dataset import TrainDataset, ValidDataset
from src.model import ZS_SBIR
from src.utils import get_all_categories


def print_run_config(args):
    print(
        "[Distill] branch weights -> "
        f"photo={args.lambda_photo_distill}, "
        f"sketch={args.lambda_sketch_distill}, "
        f"text={args.lambda_text_distill}"
    )

    print(
        "[Loss] base weights -> "
        f"cls={args.lambda_cls}, "
        f"triplet={args.lambda_triplet}, "
        f"nt_xent={args.lambda_nt_xent}"
    )
    print(
        "[Run] "
        f"dataset={args.dataset}, teacher={args.teacher}, "
        f"use_distill_proj={args.use_distill_proj}, "
        f"quantize_fp16={args.quantize_fp16}, seed={args.seed}"
    )


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


def get_datasets(args):
    seed_everything(args.seed)
    
    train_dataset = TrainDataset(args, args.proportion)
    val_sketch = ValidDataset(args, mode='sketch')
    val_photo = ValidDataset(args)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,           # Transfer CPU→GPU nhanh hơn (non-blocking)
        persistent_workers=args.workers > 0,  # Giữ worker sống giữa các epoch
        prefetch_factor=4 if args.workers > 0 else None,  # Pre-load 4 batch trước
        worker_init_fn=seed_worker,
        generator=generator,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # Tránh batch lẻ ở cuối gây vấn đề với RKD (cần B>=2)
        **loader_kwargs,
    )
    val_sketch_loader = DataLoader(
        dataset=val_sketch,
        batch_size=args.test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_photo_loader = DataLoader(
        dataset=val_photo,
        batch_size=args.test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_sketch_loader, val_photo_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../datasets/tuberlin", help="path to dataset")
    parser.add_argument("--ckpt_path", type=str, default="", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="tuberlin", help="type of dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=1)
    parser.add_argument("--img_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--use_classes", type=int, default=104)
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed cho Python/NumPy/PyTorch/DataLoader workers.")
    parser.add_argument("--lambda_cls", type=float, default=1.0,
                        help="Trọng số cho classification loss: CE(photo,text)+CE(sketch,text).")
    parser.add_argument("--lambda_triplet", type=float, default=1.0,
                        help="Trọng số cho triplet loss sketch-photo-negative.")
    parser.add_argument("--lambda_nt_xent", type=float, default=1.0,
                        help="Trọng số cho NT-Xent loss giữa photo và sketch student.")
    
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--use_adapt_sk', type=bool, default=True)
    parser.add_argument('--use_adapt_ph', type=bool, default=True)
    parser.add_argument('--use_adapt_txt', type=bool, default=True)
    parser.add_argument('--use_co_sk', type=bool, default=True)
    parser.add_argument('--use_co_ph', type=bool, default=True)
    parser.add_argument('--progress', action='store_true', default=False,
                        help='Hiện tqdm progress bar trong lúc train')
    parser.add_argument('--no_aug', action='store_true', default=False,
                        help='Tắt augmentation cho photo_aug/sketch_aug, dùng transform thường.')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--gzs', action='store_true', default=False)
    parser.add_argument('--teacher', type=str, default='clip32',
                        choices=['clip32', 'dfn5b', 'laion_h'],
                        help=(
                            "Teacher model cho distillation:\n"
                            "  clip32 → CLIP ViT-B/32 (mặc định, cross_loss)\n"
                            "  dfn5b  → DFN5B-CLIP-H/14 1024-dim (cần open-clip-torch)\n"
                            "  laion_h → LAION CLIP-H/14 1024-dim (cần open-clip-torch)"
                        ))
    parser.add_argument('--quantize_fp16', action='store_true', default=False,
                        help='Chạy strong teacher dfn5b/laion_h ở FP16 để giảm VRAM và tăng tốc.')
    parser.add_argument('--lambda_photo_distill', type=float, default=0.0,
                        help='Trọng số trực tiếp cho photo distillation loss.')
    parser.add_argument('--use_distill_proj', action='store_true', default=False,
                        help='Thêm linear projection student sang teacher dim rồi distill bằng cross_loss InfoNCE.')
    parser.add_argument('--lambda_sketch_distill', type=float, default=0.0,
                        help='Trọng số trực tiếp cho sketch distillation loss.')
    parser.add_argument('--lambda_text_distill', type=float, default=0.0,
                        help='Trọng số riêng cho text distillation loss.')
    parser.add_argument('--exp_name', type=str, default='Co_prompt')


    
    args = parser.parse_args()
    print_run_config(args)
    logger = TensorBoardLogger('tb_logs', name=args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        dirpath='saved_models/%s'%args.exp_name,
        filename="{epoch:02d}-{mAP:.4f}",
        save_top_k=1,
        mode='max',
        save_last=True)
    
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    train_loader, val_sketch_loader, val_photo_loader = get_datasets(args)
    trainer = Trainer(accelerator='gpu', devices=1, 
        min_epochs=1, max_epochs=args.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        callbacks=[checkpoint_callback]
    )

    classnames = get_all_categories(args)
 
    if ckpt_path is None:
        model = ZS_SBIR(args=args, classname=classnames)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]

        skip = [
            "model.prompt_learner_photo.token_prefix",
            "model.prompt_learner_photo.token_suffix",
            "model.prompt_learner_sketch.token_prefix",
            "model.prompt_learner_sketch.token_suffix",
        ]
        for k in skip:
            sd.pop(k, None)

        model = ZS_SBIR(args=args, classname=classnames)  # classnames = 220
        missing, unexpected = model.load_state_dict(sd, strict=False)

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])
