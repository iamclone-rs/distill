import os
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import ModelCheckpoint 

from src.sketchy_dataset import TrainDataset, ValidDataset
from src.model import ZS_SBIR
from src.utils import get_all_categories

def get_datasets(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_dataset = TrainDataset(args, args.proportion)
    val_sketch = ValidDataset(args, mode='sketch')
    val_photo = ValidDataset(args)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,           # Transfer CPU→GPU nhanh hơn (non-blocking)
        persistent_workers=args.workers > 0,  # Giữ worker sống giữa các epoch
        prefetch_factor=4 if args.workers > 0 else None,  # Pre-load 4 batch trước
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
    parser.add_argument("--distill", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    
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
    parser.add_argument('--train_full_student', action='store_true', default=False,
                        help='Không freeze student CLIP visual encoders; fine-tune toàn bộ student.')
    parser.add_argument('--train_attn_out_proj', action='store_true', default=False,
                        help='Mở thêm attention out_proj trong student visual encoders.')
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
    parser.add_argument('--lambda_distill', type=float, default=1.0,
                        help=(
                            "Trọng số cho distillation loss:\n"
                            "  clip32: 1.0 (mặc định, scale tương đương cls/nt_xent)\n"
                            "  dfn5b:  khuyến nghị 10.0–50.0 (RKD scale nhỏ hơn ~10–50x)"
                        ))
    parser.add_argument('--lambda_photo_distill', type=float, default=None,
                        help=(
                            "Trọng số riêng cho photo image distill. "
                            "Nếu set flag này, image distill dùng "
                            "lambda_photo_distill * photo_loss + lambda_sketch_distill * sketch_loss."
                        ))
    parser.add_argument('--use_distill_proj', action='store_true', default=False,
                        help='Thêm linear projection student sang teacher dim rồi distill bằng cross_loss InfoNCE.')
    parser.add_argument('--image_distill_mode', type=str, default='auto',
                        choices=['auto', 'none', 'infonce', 'rkd', 'cosine', 'mse', 'mae', 'smoothl1'],
                        help=(
                            "Cách distill image feature. "
                            "auto giữ hành vi cũ; none tắt image distill; "
                            "infonce dùng cross_loss; rkd dùng relational KD; "
                            "cosine/mse/mae/smoothl1 dùng feature regression."
                        ))
    parser.add_argument('--distill_photo_only', action='store_true', default=False,
                        help='Chỉ distill nhánh photo từ teacher; sketch học qua CE/triplet/NT-Xent.')
    parser.add_argument('--lambda_sketch_distill', type=float, default=0.0,
                        help='Trọng số riêng cho sketch image distill khi dùng lambda_photo_distill hoặc distill_photo_only.')
    parser.add_argument('--lambda_photo_rkd', type=float, default=0.0,
                        help='Trọng số RKD phụ cho photo image, cộng song song với image_distill_mode hiện tại.')
    parser.add_argument('--lambda_sketch_rkd', type=float, default=0.0,
                        help='Trọng số RKD phụ cho sketch image, cộng song song với image_distill_mode hiện tại.')
    parser.add_argument('--distill_text', action='store_true', default=False,
                        help='Distill text features từ teacher text encoder sang student text prompts.')
    parser.add_argument('--text_distill_mode', type=str, default='infonce',
                        choices=['infonce', 'cosine', 'mse', 'mae', 'smoothl1', 'rkd'],
                        help=(
                            "Cách distill text feature khi bật --distill_text. "
                            "infonce giữ hành vi cũ; cosine/mse/mae/smoothl1 dùng feature regression; "
                            "rkd học quan hệ giữa các class text."
                        ))
    parser.add_argument('--lambda_text_distill', type=float, default=1.0,
                        help='Trọng số riêng cho text distillation loss.')
    parser.add_argument('--lambda_text_rkd', type=float, default=0.0,
                        help='Trọng số RKD phụ cho text prompt features, cộng song song với text_distill_mode hiện tại.')
    parser.add_argument('--distill_xsupcon', action='store_true', default=False,
                        help='Bật cross-modal supervised contrastive loss cho sketch->photo theo class label.')
    parser.add_argument('--lambda_xsupcon', type=float, default=1.0,
                        help='Trọng số cho cross-modal supervised contrastive loss.')
    parser.add_argument('--xsupcon_temperature', type=float, default=0.07,
                        help='Temperature cho cross-modal supervised contrastive loss.')
    parser.add_argument('--xsupcon_bidirectional', action='store_true', default=False,
                        help='Thêm chiều photo->sketch cho cross-modal supervised contrastive loss.')
    parser.add_argument('--distill_rank', action='store_true', default=False,
                        help='Bật text-guided ranking distillation cho sketch->photo retrieval.')
    parser.add_argument('--lambda_rank_distill', type=float, default=1.0,
                        help='Trọng số cho ranking distillation loss.')
    parser.add_argument('--rank_distill_temperature', type=float, default=0.07,
                        help='Temperature cho student ranking distribution.')
    parser.add_argument('--teacher_rank_temperature', type=float, default=0.07,
                        help='Temperature cho teacher ranking distribution.')
    parser.add_argument('--distill_xmodal', action='store_true', default=False,
                        help='Bật cross-modal similarity matrix distillation cho sketch->photo retrieval.')
    parser.add_argument('--lambda_xmodal_distill', type=float, default=1.0,
                        help='Trọng số cho cross-modal similarity matrix distillation.')
    parser.add_argument('--xmodal_distill_mode', type=str, default='smoothl1',
                        choices=['mse', 'smoothl1'],
                        help='Loss dùng để match ma trận similarity sketch->photo.')
    parser.add_argument('--distill_listwise', action='store_true', default=False,
                        help='Bật class-aware listwise distillation cho sketch->photo retrieval.')
    parser.add_argument('--lambda_listwise_distill', type=float, default=1.0,
                        help='Trọng số cho class-aware listwise distillation.')
    parser.add_argument('--listwise_teacher_weight', type=float, default=0.5,
                        help='Tỷ lệ teacher soft target trong listwise loss; phần còn lại là class-positive target.')
    parser.add_argument('--listwise_distill_temperature', type=float, default=0.07,
                        help='Temperature cho student listwise logits.')
    parser.add_argument('--teacher_listwise_temperature', type=float, default=0.07,
                        help='Temperature cho teacher listwise logits.')
    parser.add_argument('--listwise_bidirectional', action='store_true', default=False,
                        help='Thêm chiều photo->sketch cho listwise distillation.')
    parser.add_argument('--distill_method', type=str, default='manual',
                        choices=['manual', 'rank', 'xmodal', 'listwise'],
                        help=(
                            "Preset distill. manual dùng các flag riêng; "
                            "rank bật ranking distill; "
                            "xmodal bật cross-modal matrix distill; "
                            "listwise bật class-aware listwise distill."
                        ))
    parser.add_argument('--infer_with_distill_proj', action='store_true', default=False,
                        help='Dùng projected feature cho validation/inference retrieval.')
    parser.add_argument('--rkd_weight', type=float, default=0.5,
                        help='Trọng số RKD phụ nếu dùng projected_kd_loss thử nghiệm.')
    parser.add_argument('--train_teacher_ln', action='store_true', default=False,
                        help=(
                            "Train visual LayerNorm của teacher bằng sketch teacher branch. "
                            "Photo teacher target vẫn chạy no_grad."
                        ))
    
    parser.add_argument('--exp_name', type=str, default='Co_prompt')


    
    args = parser.parse_args()
    if args.distill_method == 'rank':
        args.use_distill_proj = False
        args.distill_rank = True
        args.distill_photo_only = True
        args.image_distill_mode = 'none'
    elif args.distill_method == 'xmodal':
        args.use_distill_proj = False
        args.distill_xmodal = True
        args.distill_photo_only = True
        args.image_distill_mode = 'none'
    elif args.distill_method == 'listwise':
        args.use_distill_proj = False
        args.distill_listwise = True
        args.distill_photo_only = True
        args.image_distill_mode = 'none'

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
