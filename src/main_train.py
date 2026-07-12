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
from src.data_config import UNSEEN_CLASSES


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
    
    train_dataset = TrainDataset(args)
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
    parser.add_argument("--root", type=str, required=True, help="dataset root containing sketch/ and photo/")
    parser.add_argument("--ckpt_path", type=str, default="", help="student checkpoint to resume")
    parser.add_argument("--dataset", type=str, default="sketchy_1",
                        choices=sorted(UNSEEN_CLASSES), help="zero-shot split")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed cho Python/NumPy/PyTorch/DataLoader workers.")
    parser.add_argument("--lambda_cls", type=float, default=1.0,
                        help="Trọng số cho classification loss: CE(photo,text)+CE(sketch,text).")
    parser.add_argument("--lambda_triplet", type=float, default=1.0,
                        help="Trọng số cho triplet loss sketch-photo-negative.")
    
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--progress', action='store_true', default=True,
                        help='Hiện tqdm progress bar trong lúc train')
    parser.add_argument('--no_progress', action='store_false', dest='progress',
                        help='Tắt tqdm progress bar.')
    parser.add_argument('--quantize_fp16', action='store_true', default=True,
                        help='Chạy DFN5B teacher ở FP16 để giảm VRAM và tăng tốc.')
    parser.add_argument('--no_quantize_fp16', action='store_false', dest='quantize_fp16',
                        help='Giữ DFN5B teacher ở FP32.')
    parser.add_argument('--teacher_adapter_ckpt', type=str, default='',
                        help='Checkpoint modality adapter đã fine-tune cho DFN5B.')
    parser.add_argument('--joint_teacher_adapter', action='store_true', default=True,
                        help='Train DFN5B sketch/photo adapters jointly with the student.')
    parser.add_argument('--no_joint_teacher_adapter', action='store_false', dest='joint_teacher_adapter',
                        help='Tắt joint teacher adapter để chạy ablation/no-teacher.')
    parser.add_argument('--teacher_adapter_bottleneck', type=int, default=64)
    parser.add_argument('--teacher_adapter_lr', type=float, default=2e-5)
    parser.add_argument('--lambda_teacher_retrieval', type=float, default=1.0,
                        help='Trọng số teacher-adapter triplet loss trên nhánh ablation này.')
    parser.add_argument('--lambda_teacher_semantic', type=float, default=1.0)
    parser.add_argument('--teacher_temperature', type=float, default=0.07)
    parser.add_argument('--teacher_triplet_margin', type=float, default=0.2)
    parser.add_argument('--lambda_kd', type=float, default=3.0,
                        help='Trọng số relational KD sketch-photo.')
    parser.add_argument('--kd_temperature', type=float, default=0.07,
                        help='Temperature cho phân phối similarity sketch-photo.')
                        
    parser.add_argument('--exp_name', type=str, default='teacher_adapter_triplet_baseline')


    
    args = parser.parse_args()
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
    from pytorch_lightning.callbacks import TQDMProgressBar
    progress_bar = TQDMProgressBar(refresh_rate=20)

    trainer = Trainer(accelerator='gpu', devices=1, 
        min_epochs=1, max_epochs=args.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        callbacks=[checkpoint_callback, progress_bar]
    )

    classnames = get_all_categories(args)
 
    if ckpt_path is None:
        model = ZS_SBIR(args=args, classname=classnames)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ZS_SBIR(args=args, classname=classnames)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])
