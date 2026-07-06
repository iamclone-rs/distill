"""Student CoPrompt model distilled from a DFN5B residual-adapter teacher."""

import copy
from collections import defaultdict

import numpy as np
import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision

from src.coprompt import MultiModalPromptLearner, TextEncoder
from src.data_config import VISUALIZE_CLASSES
from src.losses import TrainingFeatures, compute_training_loss
from src.teacher_adapter import (
    DFN_MODEL_NAME,
    DFN_PRETRAINED,
    load_adapter_checkpoint,
)
from src.utils import (
    get_all_categories,
    load_clip_to_cpu,
    retrieval_precision,
    visualize_tsne,
)


STUDENT_BACKBONE = "ViT-B/32"
IMAGE_SIZE = 224


def freeze_all_but_layernorm(module):
    if isinstance(module, nn.LayerNorm):
        return
    if getattr(module, "weight", None) is not None:
        module.weight.requires_grad_(False)
    if getattr(module, "bias", None) is not None:
        module.bias.requires_grad_(False)


def _teacher_image_size(teacher) -> int:
    value = getattr(teacher.visual, "image_size", IMAGE_SIZE)
    if isinstance(value, (tuple, list)):
        return int(value[0])
    return int(value)


def load_dfn_teacher(adapter_checkpoint: str):
    if not adapter_checkpoint:
        raise ValueError("--teacher_adapter_ckpt is required.")

    print(f"[Teacher] Loading {DFN_MODEL_NAME} ({DFN_PRETRAINED})...")
    teacher, _, _ = open_clip.create_model_and_transforms(
        DFN_MODEL_NAME,
        pretrained=DFN_PRETRAINED,
    )
    teacher.eval().requires_grad_(False)
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(target_device)
    if target_device.type == "cuda":
        teacher = teacher.half()

    feature_dim = int(teacher.visual.output_dim)
    adapters, checkpoint = load_adapter_checkpoint(
        adapter_checkpoint,
        expected_feature_dim=feature_dim,
        device=target_device,
    )
    print(
        f"[Teacher] frozen DFN5B + residual adapter "
        f"(epoch={checkpoint.get('epoch', 'unknown')}, "
        f"bottleneck={checkpoint['bottleneck_dim']}, "
        f"precision={'fp16' if target_device.type == 'cuda' else 'fp32'})"
    )
    return teacher, adapters


class CustomCLIP(nn.Module):
    def __init__(self, cfg, image_clip, text_clip, teacher, teacher_adapters):
        super().__init__()
        self.cfg = cfg
        image_clip.apply(freeze_all_but_layernorm)
        text_clip.apply(freeze_all_but_layernorm)
        self.dtype = image_clip.dtype

        self.prompt_learner_photo = MultiModalPromptLearner(
            text_clip, cfg.n_ctx, IMAGE_SIZE
        )
        self.prompt_learner_sketch = MultiModalPromptLearner(
            text_clip, cfg.n_ctx, IMAGE_SIZE
        )
        self.ph_encoder = copy.deepcopy(image_clip.visual)
        self.sk_encoder = copy.deepcopy(image_clip.visual)
        self.text_encoder = TextEncoder(text_clip)
        self.logit_scale = image_clip.logit_scale

        self.model_distill = teacher
        self.teacher_adapters = teacher_adapters
        self.teacher_image_size = _teacher_image_size(teacher)
        self.teacher_fp16 = next(teacher.parameters()).dtype == torch.float16

    def train(self, mode=True):
        super().train(mode)
        self.model_distill.eval()
        self.teacher_adapters.eval()
        return self

    def teacher_image_input(self, image):
        if tuple(image.shape[-2:]) != (
            self.teacher_image_size,
            self.teacher_image_size,
        ):
            image = F.interpolate(
                image.float(),
                size=(self.teacher_image_size, self.teacher_image_size),
                mode="bicubic",
                align_corners=False,
            )
        return image.half() if self.teacher_fp16 else image.float()

    def adapt_teacher_feature(self, feature, modality):
        feature = F.normalize(feature.float(), dim=-1)
        adapter = (
            self.teacher_adapters.photo
            if modality == "photo"
            else self.teacher_adapters.sketch
        )
        return adapter(feature)

    def get_logits(self, image, classnames, modality="photo"):
        if modality == "photo":
            prompt_learner = self.prompt_learner_photo
            image_encoder = self.ph_encoder
        else:
            prompt_learner = self.prompt_learner_sketch
            image_encoder = self.sk_encoder

        tokenized, prompts, visual_context = prompt_learner(classnames)
        text_features = self.text_encoder(prompts, tokenized)
        image_features = image_encoder(
            image.type(self.dtype),
            visual_context,
            [],
        )
        image_normalized = F.normalize(image_features, dim=-1)
        text_normalized = F.normalize(text_features, dim=-1)
        logits = self.logit_scale.exp() * image_normalized @ text_normalized.t()
        return logits, image_normalized, image_features

    def forward(self, batch, classnames):
        photo, sketch, photo_aug, sketch_aug, negative, labels = batch
        photo_logits, photo_normalized, student_photo = self.get_logits(
            photo, classnames, "photo"
        )
        sketch_logits, sketch_normalized, student_sketch = self.get_logits(
            sketch, classnames, "sketch"
        )
        _, negative_normalized, _ = self.get_logits(
            negative, classnames, "photo"
        )

        with torch.no_grad():
            teacher_photo = self.model_distill.encode_image(
                self.teacher_image_input(photo_aug)
            )
            teacher_sketch = self.model_distill.encode_image(
                self.teacher_image_input(sketch_aug)
            )
            teacher_photo = self.adapt_teacher_feature(teacher_photo, "photo")
            teacher_sketch = self.adapt_teacher_feature(teacher_sketch, "sketch")

        return TrainingFeatures(
            photo_normalized=photo_normalized,
            sketch_normalized=sketch_normalized,
            negative_normalized=negative_normalized,
            photo_logits=photo_logits,
            sketch_logits=sketch_logits,
            student_photo=student_photo,
            student_sketch=student_sketch,
            teacher_photo=teacher_photo,
            teacher_sketch=teacher_sketch,
            labels=labels,
        )

    def extract_feature(self, image, classnames, modality="photo"):
        _, normalized, _ = self.get_logits(image, classnames, modality)
        return normalized


def _count_params(module, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def _fmt_params(count):
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _grad_stats(module):
    parameter_count = 0
    element_count = 0
    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm().item()
        if norm == 0.0:
            continue
        parameter_count += 1
        element_count += parameter.numel()
        squared_norm += norm**2
    return parameter_count, element_count, squared_norm**0.5


def _tensor_debug(name, value):
    text = (
        f"  {name:<24} shape={tuple(value.shape)} "
        f"dtype={value.dtype} grad={value.requires_grad}"
    )
    if value.ndim >= 2 and value.shape[-1] > 1:
        text += f" mean_norm={value.detach().float().norm(dim=-1).mean().item():.4f}"
    return text


class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super().__init__()
        self.args = args
        self.classname = classname

        image_clip = load_clip_to_cpu(STUDENT_BACKBONE, args.n_ctx)
        text_clip = load_clip_to_cpu(
            STUDENT_BACKBONE,
            args.n_ctx,
            design_details={
                "trainer": "CoOp",
                "vision_depth": 0,
                "language_depth": 0,
                "vision_ctx": 0,
                "language_ctx": 0,
            },
        )
        teacher, adapters = load_dfn_teacher(args.teacher_adapter_ckpt)
        self.model = CustomCLIP(
            args,
            image_clip,
            text_clip,
            teacher,
            adapters,
        )
        self.distance_fn = lambda x, y: F.cosine_similarity(x, y)
        self.best_metric = 1e-3
        self.feature_debug_printed = False
        self.grad_debug_printed = False
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        self.saved_features = defaultdict(lambda: {"sketch": [], "photo": []})
        self._print_model_debug_summary()

    def _print_param_row(self, name, module):
        print(
            f"  {name:<24} trainable="
            f"{_fmt_params(_count_params(module, True)):>8} / "
            f"total={_fmt_params(_count_params(module)):>8}"
        )

    def _print_grad_row(self, name, module):
        count, elements, norm = _grad_stats(module)
        print(
            f"  {name:<24} grad_params={count:>4} "
            f"grad_elems={_fmt_params(elements):>8} grad_norm={norm:.4e}"
        )

    def _print_model_debug_summary(self):
        print("=" * 78)
        print("[CoPrompt Debug] DFN5B adapter -> sketch-photo KL")
        self._print_param_row("CustomCLIP", self.model)
        self._print_param_row("prompt_photo", self.model.prompt_learner_photo)
        self._print_param_row("prompt_sketch", self.model.prompt_learner_sketch)
        self._print_param_row("ph_encoder", self.model.ph_encoder)
        self._print_param_row("sk_encoder", self.model.sk_encoder)
        self._print_param_row("text_encoder", self.model.text_encoder)
        self._print_param_row("DFN5B teacher", self.model.model_distill)
        print(
            "[Loss] "
            f"cls={self.args.lambda_cls}, triplet={self.args.lambda_triplet}, "
            f"kd={self.args.lambda_kd}, temp={self.args.kd_temperature}"
        )
        print("=" * 78)

    def configure_optimizers(self):
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(
            trainable,
            lr=self.args.lr,
            weight_decay=1e-3,
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        print(
            f"[Optimizer] SGD lr={self.args.lr}, momentum=0.9, "
            f"weight_decay=1e-3, trainable={_fmt_params(sum(p.numel() for p in trainable))}"
        )
        return [optimizer], [scheduler]

    def forward(self, data, classnames):
        return self.model(data, classnames)

    def training_step(self, batch, batch_idx):
        classnames = get_all_categories(self.args)
        features = self(batch, classnames)
        if (
            not self.feature_debug_printed
            and batch_idx == 0
            and getattr(self.trainer, "is_global_zero", True)
        ):
            print("=" * 78)
            print("[CoPrompt Debug] First train batch feature contract")
            for name, value in vars(features).items():
                print(_tensor_debug(name, value))
            print("=" * 78)
            self.feature_debug_printed = True

        loss, losses = compute_training_loss(
            features,
            lambda_cls=self.args.lambda_cls,
            lambda_triplet=self.args.lambda_triplet,
            lambda_kd=self.args.lambda_kd,
            kd_temperature=self.args.kd_temperature,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        for name, value in losses.items():
            self.log(
                "KD_SP" if name == "kd_sk_ph" else name,
                value,
                on_step=name == "kd_sk_ph",
                on_epoch=name != "kd_sk_ph",
                prog_bar=name == "kd_sk_ph",
            )
        return loss

    def on_after_backward(self):
        if self.grad_debug_printed or not getattr(self.trainer, "is_global_zero", True):
            return
        print("=" * 78)
        print("[CoPrompt Debug] First backward non-zero gradient summary")
        self._print_grad_row("CustomCLIP", self.model)
        self._print_grad_row("prompt_photo", self.model.prompt_learner_photo)
        self._print_grad_row("prompt_sketch", self.model.prompt_learner_sketch)
        self._print_grad_row("ph_encoder", self.model.ph_encoder)
        self._print_grad_row("sk_encoder", self.model.sk_encoder)
        self._print_grad_row("text_encoder", self.model.text_encoder)
        self._print_grad_row("DFN5B teacher", self.model.model_distill)
        print("=" * 78)
        self.grad_debug_printed = True

    def validation_step(self, batch, batch_idx, dataloader_idx):
        classnames = get_all_categories(self.args)
        images, labels = batch
        if dataloader_idx == 0:
            features = self.model.extract_feature(images, classnames, "sketch")
            self.val_step_outputs_sk.append((features, labels))
            modality = "sketch"
        else:
            features = self.model.extract_feature(images, classnames, "photo")
            self.val_step_outputs_ph.append((features, labels))
            modality = "photo"

        if self.args.visualize:
            for feature, label in zip(features.detach().cpu(), labels.detach().cpu()):
                self.saved_features[str(int(label))][modality].append(feature)

    def on_validation_epoch_end(self):
        if self.args.visualize:
            visualize_classes = VISUALIZE_CLASSES[self.args.dataset]
            visualize_tsne(visualize_classes, self.saved_features, mode="photo")
            visualize_tsne(visualize_classes, self.saved_features, mode="sketch")
        else:
            sketch_features = torch.cat([item[0] for item in self.val_step_outputs_sk])
            photo_features = torch.cat([item[0] for item in self.val_step_outputs_ph])
            sketch_labels = np.concatenate(
                [item[1].detach().cpu().numpy() for item in self.val_step_outputs_sk]
            )
            photo_labels = np.concatenate(
                [item[1].detach().cpu().numpy() for item in self.val_step_outputs_ph]
            )

            map_k = 200 if self.args.dataset == "sketchy_2" else None
            precision_k = 200 if self.args.dataset in ("sketchy_2", "quickdraw") else 100
            average_precision = torch.zeros(len(sketch_features))
            precision = torch.zeros(len(sketch_features))
            for index, sketch_feature in enumerate(sketch_features):
                similarities = self.distance_fn(sketch_feature.unsqueeze(0), photo_features)
                target = torch.as_tensor(
                    photo_labels == sketch_labels[index],
                    dtype=torch.bool,
                    device=similarities.device,
                )
                average_precision[index] = retrieval_average_precision(
                    similarities.cpu(),
                    target.cpu(),
                    top_k=min(map_k, len(photo_features)) if map_k else None,
                )
                precision[index] = retrieval_precision(
                    similarities.cpu(), target.cpu(), top_k=precision_k
                )

            mean_ap = average_precision.mean()
            mean_precision = precision.mean()
            self.log("mAP", mean_ap, on_step=False, on_epoch=True)
            if self.global_step > 0:
                self.best_metric = max(self.best_metric, mean_ap.item())
            map_name = f"@{map_k}" if map_k else "@all"
            print(
                f"mAP{map_name}: {mean_ap.item()}, P@{precision_k}: "
                f"{mean_precision.item()}, Best mAP: {self.best_metric}"
            )
            train_loss = self.trainer.callback_metrics.get("train_loss")
            if train_loss is not None:
                print(f"Train loss (epoch avg): {train_loss.item():.6f}")

        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
        self.saved_features.clear()
