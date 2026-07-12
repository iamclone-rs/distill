import copy
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics.functional import retrieval_average_precision #, retrieval_precision
import open_clip

from src.coprompt import MultiModalPromptLearner, TextEncoder
from src.utils import load_clip_to_cpu, retrieval_precision
from src.losses import loss_fn
from src.teacher_adapters import ModalityAdapters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# DFN5B teacher loader
# ---------------------------------------------------------------------------
DFN5B_MODEL = "ViT-H-14-quickgelu"
DFN5B_PRETRAINED = "dfn5b"


def _freeze_teacher(teacher):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _load_teacher_adapters(args, strong_teacher):
    ckpt_path = getattr(args, "teacher_adapter_ckpt", "")
    joint_training = getattr(args, "joint_teacher_adapter", False)
    if not ckpt_path:
        if not joint_training:
            return None
        adapters = ModalityAdapters(
            feature_dim=int(strong_teacher.output_dim),
            bottleneck_dim=args.teacher_adapter_bottleneck,
        ).to(device=device, dtype=torch.float32)
        print(
            "[Teacher Adapter] initialized for joint training "
            f"(feature_dim={strong_teacher.output_dim}, "
            f"bottleneck={args.teacher_adapter_bottleneck})"
        )
        return adapters
    if strong_teacher is None:
        raise ValueError("--teacher_adapter_ckpt requires the DFN5B teacher.")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    required = {"adapter_state_dict", "feature_dim", "bottleneck_dim"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"Invalid teacher adapter checkpoint '{ckpt_path}'; missing keys: {sorted(missing)}"
        )

    saved_model = checkpoint.get("model")
    saved_pretrained = checkpoint.get("pretrained")
    if saved_model and saved_model != DFN5B_MODEL:
        raise RuntimeError(
            f"Adapter model mismatch: checkpoint={saved_model}, teacher={DFN5B_MODEL}"
        )
    if saved_pretrained and saved_pretrained != DFN5B_PRETRAINED:
        raise RuntimeError(
            "Adapter pretrained-weight mismatch: "
            f"checkpoint={saved_pretrained}, teacher={DFN5B_PRETRAINED}"
        )

    feature_dim = int(checkpoint["feature_dim"])
    teacher_dim = int(getattr(strong_teacher, "output_dim", feature_dim))
    if feature_dim != teacher_dim:
        raise RuntimeError(
            f"Adapter feature_dim={feature_dim} does not match teacher output_dim={teacher_dim}."
        )

    if checkpoint.get("adapter_mode", "residual") != "residual":
        raise RuntimeError("Only residual DFN5B adapters are supported.")
    adapters = ModalityAdapters(
        feature_dim=feature_dim,
        bottleneck_dim=int(checkpoint["bottleneck_dim"]),
    )
    adapters.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapters.requires_grad_(joint_training)
    adapters.train(joint_training)
    adapters = adapters.to(device=device, dtype=torch.float32)
    print(
        f"[Teacher Adapter] loaded {ckpt_path} "
        f"(epoch={checkpoint.get('epoch', 'unknown')}, feature_dim={feature_dim}, "
        f"bottleneck={checkpoint['bottleneck_dim']}, "
        f"mode={checkpoint.get('adapter_mode', 'residual')}, "
        f"trainable={joint_training})"
    )
    return adapters


def _infer_teacher_image_size(teacher):
    visual = getattr(teacher, "visual", None)
    if visual is None:
        return None

    for attr in ("image_size", "input_resolution"):
        value = getattr(visual, attr, None)
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            return int(value[0])
        return int(value)

    return None


def _load_teacher(args):
    """Load the frozen DFN5B teacher when relational KD is enabled."""
    if args.lambda_kd <= 0 and not args.joint_teacher_adapter:
        print("[Teacher] KD và joint adapter đều tắt -> bỏ qua DFN5B teacher")
        return None

    print(f"[Teacher] Đang load DFN5B ({DFN5B_MODEL})...")
    teacher, _, _ = open_clip.create_model_and_transforms(
        DFN5B_MODEL, pretrained=DFN5B_PRETRAINED
    )
    teacher.text_tokenizer = open_clip.get_tokenizer(DFN5B_MODEL)
    teacher = _freeze_teacher(teacher)
    teacher = teacher.to(device)
    if getattr(args, "quantize_fp16", False):
        if device.type != "cuda":
            print("[Teacher] quantize_fp16=True nhưng không có CUDA; giữ teacher ở FP32.")
        else:
            teacher = teacher.half()
            print("[Teacher] DFN5B chạy FP16")
    teacher.output_dim = 1024
    teacher.image_size = _infer_teacher_image_size(teacher)
    print(
        "[Teacher] DFN5B đã sẵn sàng "
        f"(frozen, output {teacher.output_dim}-dim, image_size={teacher.image_size or 'unknown'})"
    )
    return teacher

# ---------------------------------------------------------------------------

def freeze_all_but_ln(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, clip_model, clip_model_distill, strong_teacher=None
    ):
        super().__init__()
        self.cfg = cfg
        clip_model.apply(freeze_all_but_ln)
        clip_model_distill.apply(freeze_all_but_ln)
        self.dtype = clip_model.dtype
        self.prompt_learner_photo = MultiModalPromptLearner(cfg, clip_model_distill, type='photo')
        self.prompt_learner_sketch = MultiModalPromptLearner(cfg, clip_model_distill, type='sketch')
        
        self.ph_encoder = copy.deepcopy(clip_model.visual)
        self.sk_encoder = copy.deepcopy(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model_distill)
        self.logit_scale = clip_model.logit_scale
        
        self.model_distill = strong_teacher
        self.teacher_active = strong_teacher is not None
        self.joint_teacher_adapter = getattr(cfg, "joint_teacher_adapter", False)
        self.teacher_adapters = _load_teacher_adapters(cfg, strong_teacher)
        self._teacher_text_cache = {}
        self._teacher_fp16 = (
            self.teacher_active
            and getattr(cfg, "quantize_fp16", False)
            and device.type == "cuda"
        )
        print(
            "[Relational KD] sketch-photo branch -> "
            f"active={self.teacher_active}, lambda={cfg.lambda_kd}, "
            f"temperature={cfg.kd_temperature}"
        )

    def train(self, mode=True):
        super().train(mode)
        if self.model_distill is not None:
            self.model_distill.eval()
        if self.teacher_adapters is not None:
            self.teacher_adapters.train(mode and self.joint_teacher_adapter)
        return self
    
    def teacher_image_input(self, image):
        teacher_size = getattr(self.model_distill, "image_size", None)
        if teacher_size is not None and tuple(image.shape[-2:]) != (teacher_size, teacher_size):
            image = F.interpolate(
                image.float(),
                size=(teacher_size, teacher_size),
                mode="bicubic",
                align_corners=False,
            )
        return image.half() if self._teacher_fp16 else image.float()

    def adapt_teacher_feature(self, feature, modality):
        if self.teacher_adapters is None:
            return feature
        feature = F.normalize(feature.float(), dim=-1)
        adapter = (
            self.teacher_adapters.photo
            if modality == "photo"
            else self.teacher_adapters.sketch
        )
        return adapter(feature)

    def get_teacher_text_features(self, classnames):
        cache_key = tuple(classnames)
        if cache_key in self._teacher_text_cache:
            return self._teacher_text_cache[cache_key]

        sketch_prompts = [
            f"a sketch of a {name.replace('_', ' ')}." for name in classnames
        ]
        photo_prompts = [
            f"a photo of a {name.replace('_', ' ')}." for name in classnames
        ]
        tokens = self.model_distill.text_tokenizer(
            sketch_prompts + photo_prompts
        ).to(device)
        with torch.no_grad():
            text_features = F.normalize(
                self.model_distill.encode_text(tokens).float(), dim=-1
            )
        class_count = len(classnames)
        result = (
            text_features[:class_count],
            text_features[class_count:],
        )
        self._teacher_text_cache[cache_key] = result
        return result

    def get_logits(self, img_tensor, classnames, type='photo'):
        if type=='photo':
            prompt_learner = self.prompt_learner_photo
            image_encoder = self.ph_encoder
        else:
            image_encoder = self.sk_encoder
            prompt_learner = self.prompt_learner_sketch
            
        logit_scale = self.logit_scale.exp()
        (
            tokenized_prompts,
            prompts,
            visual_ctx,
        ) = prompt_learner(classnames)
        
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_encoder(
                img_tensor.type(self.dtype), visual_ctx, []
            ) # (batch_size, 768)
        
        image_features_normalize = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features_normalize @ text_features.t()
        
        return logits, image_features_normalize
        
    def forward(self, x, classnames):
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label = x
        pos_logits, photo_features = self.get_logits(photo_tensor, classnames)
        sk_logits, sketch_features = self.get_logits(
            sk_tensor, classnames, type='sketch'
        )
        _, negative_features = self.get_logits(neg_tensor, classnames)

        teacher_photo_features = photo_features.detach()
        teacher_sketch_features = sketch_features.detach()
        teacher_sketch_text = None
        teacher_photo_text = None
        if self.teacher_active:
            with torch.no_grad():
                teacher_photo_base = self.model_distill.encode_image(
                    self.teacher_image_input(photo_aug_tensor)
                )
                teacher_sketch_base = self.model_distill.encode_image(
                    self.teacher_image_input(sk_aug_tensor)
                )
            teacher_photo_features = self.adapt_teacher_feature(
                teacher_photo_base, "photo"
            )
            teacher_sketch_features = self.adapt_teacher_feature(
                teacher_sketch_base, "sketch"
            )
            if self.joint_teacher_adapter:
                teacher_sketch_text, teacher_photo_text = (
                    self.get_teacher_text_features(classnames)
                )

        return (
            photo_features,
            sketch_features,
            teacher_photo_features,
            teacher_sketch_features,
            negative_features,
            label,
            pos_logits,
            sk_logits,
            self.teacher_active,
            self.joint_teacher_adapter,
            teacher_sketch_text,
            teacher_photo_text,
        )
        
    def extract_feature(self, image, classname, type='photo'):
        _, feature = self.get_logits(image, classnames=classname, type=type)
        return feature


class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super(ZS_SBIR, self).__init__()
        self.args = args
        self.classname = classname
        clip_model = load_clip_to_cpu(args)
        
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = load_clip_to_cpu(args, design_details=design_details)
        
        self.distance_fn = lambda x, y: F.cosine_similarity(x, y)
        self.best_metric = 1e-3

        strong_teacher = _load_teacher(args)
        self.model = CustomCLIP(
            cfg=args,
            clip_model=clip_model,
            clip_model_distill=clip_model_distill,
            strong_teacher=strong_teacher,
        )

        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        
    def configure_optimizers(self):
        adapter_params = (
            [p for p in self.model.teacher_adapters.parameters() if p.requires_grad]
            if self.model.teacher_adapters is not None
            else []
        )
        adapter_param_ids = {id(p) for p in adapter_params}
        student_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in adapter_param_ids
        ]
        param_groups = [{"params": student_params, "lr": self.args.lr}]
        if adapter_params:
            param_groups.append(
                {"params": adapter_params, "lr": self.args.teacher_adapter_lr}
            )
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=self.args.lr,
            weight_decay=1e-3,
            momentum=0.9,
        )
        trainable = sum(p.numel() for group in optimizer.param_groups for p in group["params"] if p.requires_grad)
        print(
            "[Optimizer] SGD "
            f"lr={self.args.lr}, momentum=0.9, weight_decay=1e-3, "
            f"teacher_adapter_lr={self.args.teacher_adapter_lr if adapter_params else 'off'}, "
            f"trainable_params={trainable:,}"
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.1
        )
        
        return [optimizer] , [scheduler]
    
    def forward(self, data, classname):
        return self.model(data, classname)
    
    def training_step(self, batch, batch_idx):
        features = self.forward(batch, self.classname)
        loss, loss_dict = loss_fn(self.args, features)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        for k, v in loss_dict.items():
            bar_names = {
                "kd_sketch_photo": "KD_SP",
                "teacher_triplet": "T_TRI",
                "teacher_semantic": "T_SEM",
            }
            show_on_bar = k in bar_names
            bar_name = bar_names.get(k, k)
            self.log(bar_name, v, on_step=True, on_epoch=False, prog_bar=show_on_bar)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        image_tensor, label = batch
        if dataloader_idx == 0:
            feat = self.model.extract_feature(image_tensor, classname=self.classname, type='sketch')
            self.val_step_outputs_sk.append((feat, label))
        else:
            feat = self.model.extract_feature(image_tensor, classname=self.classname, type='photo')
            self.val_step_outputs_ph.append((feat, label))

    def on_validation_epoch_end(self):
        query_len = len(self.val_step_outputs_sk)
        gallery_len = len(self.val_step_outputs_ph)

        query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
        gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])

        all_sketch_category = np.array(sum([list(self.val_step_outputs_sk[i][1].detach().cpu().numpy()) for i in range(query_len)], []))
        all_photo_category = np.array(sum([list(self.val_step_outputs_ph[i][1].detach().cpu().numpy()) for i in range(gallery_len)], []))

        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        precision = torch.zeros(len(query_feat_all))
        if self.args.dataset == "sketchy_2":
            map_k = 200
            p_k = 200
        else:
            map_k = 0
            if self.args.dataset == "quickdraw":
                p_k = 200
            else:
                p_k = 100

        for idx, sk_feat in enumerate(query_feat_all):
            category = all_sketch_category[idx]
            distance = self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool, device=device)
            target[np.where(all_photo_category == category)] = True

            if map_k != 0:
                top_k_actual = min(map_k, len(gallery))
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=top_k_actual)
            else:
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())

            precision[idx] = retrieval_precision(distance.cpu(), target.cpu(), top_k=p_k)

        mAP = torch.mean(ap)
        precision = torch.mean(precision)
        self.log("mAP", mAP, on_step=False, on_epoch=True)
        if self.global_step > 0:
            self.best_metric = max(self.best_metric, mAP.item())

        if map_k != 0:
            print('mAP@{}: {}, P@{}: {}, Best mAP: {}'.format(map_k, mAP.item(), p_k, precision, self.best_metric))
        else:
            print('mAP@all: {}, P@{}: {}, Best mAP: {}'.format(mAP.item(), p_k, precision, self.best_metric))
        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            print(f"Train loss (epoch avg): {train_loss.item():.6f}")

        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
