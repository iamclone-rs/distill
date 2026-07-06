import copy
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from collections import defaultdict
from torchmetrics.functional import retrieval_average_precision #, retrieval_precision
import open_clip
from clip import clip

from src.coprompt import MultiModalPromptLearner, TextEncoder
from src.utils import load_clip_to_cpu, get_all_categories, retrieval_precision, visualize_tsne
from src.losses import loss_fn
from src.data_config import VISUALIZE_CLASSES, UNSEEN_CLASSES
from src.finetune_laion_adapter import ModalityAdapters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------
_TEACHER_REGISTRY = {
    # key             : (open_clip model name,        pretrained tag)
    "dfn5b"           : ("ViT-H-14-quickgelu",       "dfn5b"),
    "laion_h"         : ("ViT-H-14",                 "laion2b_s32b_b79k"),
}
TEACHER_CHOICES = ["clip32", *_TEACHER_REGISTRY.keys()]


def _needs_strong_teacher(args):
    if getattr(args, "teacher", "clip32") == "clip32":
        return False
    distill_mode = getattr(args, "distill_mode", "kd_div")
    if distill_mode == "linear_infonce":
        return (
            getattr(args, "lambda_infonce_photo", 0.0) > 0
            or getattr(args, "lambda_infonce_sketch", 0.0) > 0
            or getattr(args, "lambda_infonce_text", 0.0) > 0
        )
    if distill_mode == "teacher_weighted_ntxent":
        return getattr(args, "lambda_tw_ntxent", 0.0) > 0
    if distill_mode == "independent_cosine":
        return (
            getattr(args, "lambda_ind_photo", 0.0) > 0
            or getattr(args, "lambda_ind_sketch", 0.0) > 0
            or getattr(args, "lambda_ind_sketch_photo", 0.0) > 0
            or getattr(args, "lambda_ind_text", 0.0) > 0
        )
    return (
        getattr(args, "lambda_rkd_sk_ph", 0.0) > 0
        or getattr(args, "lambda_rkd_ph_txt", 0.0) > 0
        or getattr(args, "lambda_rkd_sk_txt", 0.0) > 0
    )


def _extract_teacher_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in (
        "dfn5b_state_dict",
        "teacher_state_dict",
        "model_state_dict",
        "state_dict",
    ):
        if key in checkpoint:
            return checkpoint[key]

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    return checkpoint


def _strip_teacher_prefix(key):
    prefixes = (
        "module.",
        "model.model_distill.",
        "model.teacher.",
        "model_distill.",
        "teacher.",
        "dfn5b.",
        "model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key


def _load_teacher_checkpoint(teacher, ckpt_path):
    if not ckpt_path:
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_teacher_state_dict(checkpoint)
    target_state = teacher.state_dict()
    loadable = {}
    skipped = 0

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            skipped += 1
            continue
        stripped_key = _strip_teacher_prefix(key)
        if stripped_key in target_state and target_state[stripped_key].shape == value.shape:
            loadable[stripped_key] = value
        else:
            skipped += 1

    if not loadable:
        raise RuntimeError(
            f"Không tìm thấy tensor nào khớp để load teacher_ckpt='{ckpt_path}'. "
            "Kiểm tra checkpoint có đúng backbone teacher không."
        )

    missing, unexpected = teacher.load_state_dict(loadable, strict=False)
    print(
        "[Teacher] loaded checkpoint "
        f"{ckpt_path} -> loaded={len(loadable)}, skipped={skipped}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def _load_teacher_layernorm_checkpoint(teacher, args):
    ckpt_path = getattr(args, "teacher_layernorm_ckpt", "")
    if not ckpt_path:
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("layernorm_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(
            f"Invalid LayerNorm teacher checkpoint '{ckpt_path}': "
            "missing layernorm_state_dict."
        )

    teacher_key = getattr(args, "teacher", "")
    if teacher_key in _TEACHER_REGISTRY:
        expected_model, expected_pretrained = _TEACHER_REGISTRY[teacher_key]
        saved_model = checkpoint.get("model")
        saved_pretrained = checkpoint.get("pretrained")
        if saved_model and saved_model != expected_model:
            raise RuntimeError(
                f"LayerNorm model mismatch: checkpoint={saved_model}, teacher={expected_model}"
            )
        if saved_pretrained and saved_pretrained != expected_pretrained:
            raise RuntimeError(
                "LayerNorm pretrained-weight mismatch: "
                f"checkpoint={saved_pretrained}, teacher={expected_pretrained}"
            )

    target_state = teacher.state_dict()
    invalid = [
        key for key, value in state_dict.items()
        if key not in target_state or target_state[key].shape != value.shape
    ]
    if invalid:
        raise RuntimeError(
            f"LayerNorm checkpoint has {len(invalid)} incompatible tensors; "
            f"first keys: {invalid[:5]}"
        )
    teacher.load_state_dict(state_dict, strict=False)
    print(
        f"[Teacher LayerNorm] loaded {ckpt_path} "
        f"(epoch={checkpoint.get('epoch', 'unknown')}, tensors={len(state_dict)})"
    )


def _freeze_teacher(teacher):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _load_teacher_adapters(args, strong_teacher):
    ckpt_path = getattr(args, "teacher_adapter_ckpt", "")
    if not ckpt_path:
        return None
    if strong_teacher is None:
        raise ValueError("--teacher_adapter_ckpt requires a strong teacher, not teacher=clip32.")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    required = {"adapter_state_dict", "feature_dim", "bottleneck_dim"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"Invalid teacher adapter checkpoint '{ckpt_path}'; missing keys: {sorted(missing)}"
        )

    teacher_key = getattr(args, "teacher", "")
    if teacher_key in _TEACHER_REGISTRY:
        expected_model, expected_pretrained = _TEACHER_REGISTRY[teacher_key]
        saved_model = checkpoint.get("model")
        saved_pretrained = checkpoint.get("pretrained")
        if saved_model and saved_model != expected_model:
            raise RuntimeError(
                f"Adapter model mismatch: checkpoint={saved_model}, teacher={expected_model}"
            )
        if saved_pretrained and saved_pretrained != expected_pretrained:
            raise RuntimeError(
                "Adapter pretrained-weight mismatch: "
                f"checkpoint={saved_pretrained}, teacher={expected_pretrained}"
            )

    feature_dim = int(checkpoint["feature_dim"])
    teacher_dim = int(getattr(strong_teacher, "output_dim", feature_dim))
    if feature_dim != teacher_dim:
        raise RuntimeError(
            f"Adapter feature_dim={feature_dim} does not match teacher output_dim={teacher_dim}."
        )

    adapters = ModalityAdapters(
        feature_dim=feature_dim,
        bottleneck_dim=int(checkpoint["bottleneck_dim"]),
    )
    adapters.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapters.eval().requires_grad_(False)
    adapters = adapters.to(device=device, dtype=torch.float32)
    print(
        f"[Teacher Adapter] loaded {ckpt_path} "
        f"(epoch={checkpoint.get('epoch', 'unknown')}, feature_dim={feature_dim}, "
        f"bottleneck={checkpoint['bottleneck_dim']})"
    )
    return adapters


def _infer_teacher_output_dim(teacher):
    tokenizer = getattr(teacher, "text_tokenizer", None)
    if tokenizer is None:
        return 1024

    tokenized = tokenizer(["a photo of object."]).to(device)
    with torch.no_grad():
        text_features = teacher.encode_text(tokenized)
    return int(text_features.shape[-1])


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
    """
    Trả về strong_teacher model (frozen) hoặc None.

    args.teacher:
        'clip32'  → None  (không dùng strong teacher)
        key trong _TEACHER_REGISTRY → open_clip teacher frozen

    """
    teacher_key = getattr(args, "teacher", "clip32")

    if teacher_key == "clip32":
        print("[Teacher] clip32 (ViT-B/32) -> không dùng strong teacher")
        return None

    if not _needs_strong_teacher(args):
        print(f"[Teacher] {teacher_key} được chọn nhưng distill weight = 0 -> bỏ qua strong teacher")
        return None

    if teacher_key not in _TEACHER_REGISTRY:
        raise ValueError(
            f"Teacher '{teacher_key}' không hợp lệ. "
            f"Chọn một trong: clip32, {', '.join(_TEACHER_REGISTRY)}"
        )

    model_name, pretrained = _TEACHER_REGISTRY[teacher_key]
    print(f"[Teacher] Đang load {teacher_key} ({model_name}, pretrained={pretrained})...")
    teacher, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    teacher.text_tokenizer = open_clip.get_tokenizer(model_name)
    _load_teacher_checkpoint(teacher, getattr(args, "teacher_ckpt", ""))
    _load_teacher_layernorm_checkpoint(teacher, args)
    teacher = _freeze_teacher(teacher)
    teacher = teacher.to(device)
    if getattr(args, "quantize_fp16", False):
        if device.type != "cuda":
            print("[Teacher] quantize_fp16=True nhưng không có CUDA; giữ teacher ở FP32.")
        else:
            teacher = teacher.half()
            print(f"[Teacher] {teacher_key} quantize_fp16=True -> teacher chạy FP16")
    teacher.output_dim = _infer_teacher_output_dim(teacher)
    teacher.image_size = _infer_teacher_image_size(teacher)
    print(
        f"[Teacher] {teacher_key} đã sẵn sàng "
        f"(frozen, output {teacher.output_dim}-dim, image_size={teacher.image_size or 'unknown'})"
    )
    return teacher

# ---------------------------------------------------------------------------

def freeze_model(m):
    m.requires_grad_(False)
    

def freeze_all_but_ln(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


def unfreeze_layernorm_params(m):
    num_params = 0
    for module in m.modules():
        if isinstance(module, torch.nn.LayerNorm):
            for p in module.parameters(recurse=False):
                p.requires_grad_(True)
                num_params += p.numel()
    return num_params


def unfreeze_attention_out_proj(m):
    num_params = 0
    for module in m.modules():
        if isinstance(module, nn.MultiheadAttention):
            for p in module.out_proj.parameters():
                p.requires_grad_(True)
                num_params += p.numel()
    return num_params
            
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
        self.text_encoder = TextEncoder(clip_model_distill, cfg)
        self.logit_scale = clip_model.logit_scale
        
        # strong_teacher=<model> -> DFN5B frozen, dùng distillation target
        if strong_teacher is not None:
            self.model_distill = strong_teacher
            self._use_strong_teacher = True
        else:
            self.model_distill = clip_model_distill
            self._use_strong_teacher = False
        self.teacher_adapters = _load_teacher_adapters(cfg, strong_teacher)
        
        self._distill_mode = getattr(cfg, "distill_mode", "kd_div")
        lambda_rkd_sk_ph = getattr(cfg, "lambda_rkd_sk_ph", 0.0)
        lambda_rkd_ph_txt = getattr(cfg, "lambda_rkd_ph_txt", 0.0)
        lambda_rkd_sk_txt = getattr(cfg, "lambda_rkd_sk_txt", 0.0)
        lambda_infonce_photo = getattr(cfg, "lambda_infonce_photo", 0.0)
        lambda_infonce_sketch = getattr(cfg, "lambda_infonce_sketch", 0.0)
        lambda_infonce_text = getattr(cfg, "lambda_infonce_text", 0.0)
        lambda_tw_ntxent = getattr(cfg, "lambda_tw_ntxent", 0.0)
        lambda_ind_photo = getattr(cfg, "lambda_ind_photo", 0.0)
        lambda_ind_sketch = getattr(cfg, "lambda_ind_sketch", 0.0)
        lambda_ind_sketch_photo = getattr(cfg, "lambda_ind_sketch_photo", 0.0)
        lambda_ind_text = getattr(cfg, "lambda_ind_text", 0.0)

        self._kd_image_distill_active = (
            lambda_rkd_sk_ph > 0 or lambda_rkd_ph_txt > 0 or lambda_rkd_sk_txt > 0
        )
        self._infonce_image_distill_active = (
            lambda_infonce_photo > 0 or lambda_infonce_sketch > 0
        )
        self._tw_ntxent_active = lambda_tw_ntxent > 0
        self._independent_image_distill_active = (
            lambda_ind_photo > 0
            or lambda_ind_sketch > 0
            or lambda_ind_sketch_photo > 0
        )
        if self._distill_mode == "kd_div":
            self._image_distill_active = self._kd_image_distill_active
            self._need_teacher_text = lambda_rkd_ph_txt > 0 or lambda_rkd_sk_txt > 0
        elif self._distill_mode == "linear_infonce":
            self._image_distill_active = self._infonce_image_distill_active
            self._need_teacher_text = lambda_infonce_text > 0
        elif self._distill_mode == "teacher_weighted_ntxent":
            self._image_distill_active = self._tw_ntxent_active
            self._need_teacher_text = False
        elif self._distill_mode == "independent_cosine":
            self._image_distill_active = self._independent_image_distill_active
            self._need_teacher_text = lambda_ind_text > 0
        else:
            raise ValueError(f"Unknown distill_mode: {self._distill_mode}")
        self._teacher_fp16 = (
            self._use_strong_teacher
            and getattr(cfg, "quantize_fp16", False)
            and device.type == "cuda"
        )
        self._teacher_output_dim = getattr(strong_teacher, "output_dim", 512)
        self._project_to_teacher_dim = (
            self._distill_mode in ("linear_infonce", "independent_cosine")
            and self._use_strong_teacher
        )
        if self._project_to_teacher_dim:
            self.distill_proj = nn.Linear(512, self._teacher_output_dim, bias=False).to(clip_model.dtype)
            if self._need_teacher_text:
                self.text_distill_proj = nn.Linear(512, self._teacher_output_dim, bias=False).to(clip_model.dtype)
        if self._need_teacher_text:
            self._teacher_text_cache = {}
        if self._distill_mode == "kd_div":
            print(
                "[KD-div] active branches -> "
                f"sk_ph={lambda_rkd_sk_ph > 0} ({lambda_rkd_sk_ph}), "
                f"ph_txt={lambda_rkd_ph_txt > 0} ({lambda_rkd_ph_txt}), "
                f"sk_txt={lambda_rkd_sk_txt > 0} ({lambda_rkd_sk_txt})"
            )
        elif self._distill_mode == "linear_infonce":
            print(
                "[Linear InfoNCE] active branches -> "
                f"photo={lambda_infonce_photo > 0} ({lambda_infonce_photo}), "
                f"sketch={lambda_infonce_sketch > 0} ({lambda_infonce_sketch}), "
                f"text={lambda_infonce_text > 0} ({lambda_infonce_text}), "
                f"project_512_to_{self._teacher_output_dim}={self._project_to_teacher_dim}"
            )
        elif self._distill_mode == "teacher_weighted_ntxent":
            print(
                "[Teacher-Weighted NT-Xent] active -> "
                f"{self._tw_ntxent_active} ({lambda_tw_ntxent}), "
                f"alpha={getattr(cfg, 'tw_alpha', 0.3)}, "
                f"temp={getattr(cfg, 'tw_temperature', 0.08)}"
            )
        else:
            print(
                "[Independent Cosine] active branches -> "
                f"photo={lambda_ind_photo > 0} ({lambda_ind_photo}), "
                f"sketch={lambda_ind_sketch > 0} ({lambda_ind_sketch}), "
                f"sketch_photo={lambda_ind_sketch_photo > 0} ({lambda_ind_sketch_photo}), "
                f"text={lambda_ind_text > 0} ({lambda_ind_text}), "
                f"project_512_to_{self._teacher_output_dim}={self._project_to_teacher_dim}"
            )
        self.saved_features = defaultdict(lambda: {"sketch": [], "photo": []})

    def project_image_distill_feature(self, feature):
        if not self._project_to_teacher_dim:
            return feature
        return self.distill_proj(feature.type(self.dtype))

    def project_text_distill_feature(self, feature):
        if not self._project_to_teacher_dim or not hasattr(self, "text_distill_proj"):
            return feature
        return self.text_distill_proj(feature.type(self.dtype))
    
    def teacher_image_input(self, image):
        if not self._use_strong_teacher:
            return image
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
        if not self._need_teacher_text:
            return None

        cache_key = tuple(classnames)
        if cache_key in self._teacher_text_cache:
            return self._teacher_text_cache[cache_key]

        prompts = [
            "a photo/sketch of " + name.replace("_", " ") + "."
            for name in classnames
        ]
        if self._use_strong_teacher:
            tokenizer = getattr(self.model_distill, "text_tokenizer", None)
            if tokenizer is None:
                return None
            tokenized = tokenizer(prompts).to(device)
        else:
            tokenized = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_features = self.model_distill.encode_text(tokenized)
        self._teacher_text_cache[cache_key] = text_features
        return text_features

    def get_logits(self, img_tensor, classnames, type='photo', return_text=False):
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
        
        text_features, txt_guided_prompts = self.text_encoder(
            prompts, tokenized_prompts, return_all=True
        )  # text_features: (n_classes, 512)

        # Dùng hidden context tokens của từng text block để dẫn dắt visual block kế tiếp.
        # Trung bình qua các class để thu được prompt dùng chung có shape (n_ctx, 768).
        # prompt_depth=1 → compound_prompt_projections=[] → visual_deep_prompts=[] (như cũ)
        visual_deep_prompts = []
        for index, layer in enumerate(prompt_learner.compound_prompt_projections):
            text_prompt = txt_guided_prompts[index]
            text_prompt = layer(text_prompt).mean(dim=0)
            visual_deep_prompts.append(text_prompt)

        image_features = image_encoder(
                img_tensor.type(self.dtype), visual_ctx, visual_deep_prompts
            ) # (batch_size, 768)
        
        image_features_normalize = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features_normalize @ text_features.t()
        
        if return_text:
            return logits, image_features_normalize, image_features, text_features
        return logits, image_features_normalize, image_features
        
    def forward(self, x, classnames):
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label = x
        pos_logits, photo_features_norm, photo_feature, photo_text_feature = self.get_logits(
            photo_tensor, classnames, return_text=True
        )
        sk_logits, sk_feature_norm, sk_feature, sk_text_feature = self.get_logits(
            sk_tensor, classnames, type='sketch', return_text=True
        )
        _, neg_feature, neg_raw_feature = self.get_logits(neg_tensor, classnames)
        
        if self._distill_mode == "kd_div":
            lambda_rkd_sk_ph = getattr(self.cfg, "lambda_rkd_sk_ph", 0.0)
            lambda_rkd_ph_txt = getattr(self.cfg, "lambda_rkd_ph_txt", 0.0)
            lambda_rkd_sk_txt = getattr(self.cfg, "lambda_rkd_sk_txt", 0.0)
            train_photo_distill = lambda_rkd_sk_ph > 0 or lambda_rkd_ph_txt > 0
            train_sketch_distill = lambda_rkd_sk_ph > 0 or lambda_rkd_sk_txt > 0
        elif self._distill_mode == "linear_infonce":
            train_photo_distill = getattr(self.cfg, "lambda_infonce_photo", 0.0) > 0
            train_sketch_distill = getattr(self.cfg, "lambda_infonce_sketch", 0.0) > 0
        elif self._distill_mode == "teacher_weighted_ntxent":
            train_photo_distill = getattr(self.cfg, "lambda_tw_ntxent", 0.0) > 0
            train_sketch_distill = train_photo_distill
        elif self._distill_mode == "independent_cosine":
            train_photo_distill = (
                getattr(self.cfg, "lambda_ind_photo", 0.0) > 0
                or getattr(self.cfg, "lambda_ind_sketch_photo", 0.0) > 0
            )
            train_sketch_distill = getattr(self.cfg, "lambda_ind_sketch", 0.0) > 0
        else:
            raise ValueError(f"Unknown distill_mode: {self._distill_mode}")
        photo_aug_features = photo_feature.detach()
        sk_aug_features = sk_feature.detach()

        if self._image_distill_active:
            with torch.no_grad():
                if train_photo_distill:
                    teacher_input = self.teacher_image_input(photo_aug_tensor)
                    photo_aug_features = self.model_distill.encode_image(teacher_input)
                    photo_aug_features = self.adapt_teacher_feature(
                        photo_aug_features, "photo"
                    )
                if train_sketch_distill:
                    teacher_input = self.teacher_image_input(sk_aug_tensor)
                    sk_aug_features = self.model_distill.encode_image(teacher_input)
                    sk_aug_features = self.adapt_teacher_feature(
                        sk_aug_features, "sketch"
                    )
        photo_distill_feature = self.project_image_distill_feature(photo_feature)
        sk_distill_feature = self.project_image_distill_feature(sk_feature)
        neg_distill_feature = self.project_image_distill_feature(neg_raw_feature)
        photo_text_distill_feature = self.project_text_distill_feature(photo_text_feature)
        sk_text_distill_feature = self.project_text_distill_feature(sk_text_feature)
        teacher_text_feature = self.get_teacher_text_features(classnames)
            
        return photo_features_norm, sk_feature_norm, photo_aug_features, sk_aug_features, \
            neg_feature, label, pos_logits, sk_logits, photo_feature, sk_feature, \
            photo_distill_feature, sk_distill_feature, neg_distill_feature, \
            photo_text_distill_feature, sk_text_distill_feature, teacher_text_feature
        
    def extract_feature(self, image, classname, type='photo'):
        _, feature, raw_feature = self.get_logits(image, classnames=classname, type=type)
        return feature


def _count_params(module, trainable_only=False):
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


def _fmt_params(num_params):
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def _prompt_local_params(prompt_learner):
    total = prompt_learner.ctx.numel()
    trainable = prompt_learner.ctx.numel() if prompt_learner.ctx.requires_grad else 0
    total += _count_params(prompt_learner.proj)
    trainable += _count_params(prompt_learner.proj, trainable_only=True)
    return trainable, total


def _grad_stats(module):
    params_with_grad = 0
    elems_with_grad = 0
    grad_sq_sum = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach().float()
        grad_norm = grad.norm().item()
        if grad_norm == 0.0:
            continue
        params_with_grad += 1
        elems_with_grad += p.numel()
        grad_sq_sum += grad_norm ** 2
    return params_with_grad, elems_with_grad, grad_sq_sum ** 0.5


def _prompt_local_grad_stats(prompt_learner):
    params = [prompt_learner.ctx, *prompt_learner.proj.parameters()]
    params_with_grad = 0
    elems_with_grad = 0
    grad_sq_sum = 0.0
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad.detach().float()
        grad_norm = grad.norm().item()
        if grad_norm == 0.0:
            continue
        params_with_grad += 1
        elems_with_grad += p.numel()
        grad_sq_sum += grad_norm ** 2
    return params_with_grad, elems_with_grad, grad_sq_sum ** 0.5


def _tensor_debug(name, value):
    if value is None:
        return f"  {name:<28} None"
    if not torch.is_tensor(value):
        return f"  {name:<28} {type(value).__name__}"

    text = (
        f"  {name:<28} shape={tuple(value.shape)} "
        f"dtype={value.dtype} grad={value.requires_grad}"
    )
    if value.ndim >= 2 and value.shape[-1] > 1:
        with torch.no_grad():
            norm = value.detach().float().norm(dim=-1).mean().item()
        text += f" mean_norm={norm:.4f}"
    return text
            
class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super(ZS_SBIR, self).__init__()
        if getattr(args, "teacher_adapter_ckpt", "") and getattr(
            args, "teacher_layernorm_ckpt", ""
        ):
            raise ValueError(
                "Use either --teacher_adapter_ckpt or --teacher_layernorm_ckpt, not both."
            )
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
        self._feature_debug_printed = False
        self._grad_debug_printed = False
        self._print_model_debug_summary()
    
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []
        self.saved_features = defaultdict(lambda: {"sketch": [], "photo": []})

    def _print_module_param_row(self, name, module):
        total = _count_params(module)
        trainable = _count_params(module, trainable_only=True)
        print(
            f"  {name:<24} trainable={_fmt_params(trainable):>8} / "
            f"total={_fmt_params(total):>8}"
        )

    def _print_prompt_local_row(self, name, prompt_learner):
        trainable, total = _prompt_local_params(prompt_learner)
        print(
            f"  {name:<24} trainable={_fmt_params(trainable):>8} / "
            f"total={_fmt_params(total):>8}"
        )

    def _print_grad_row(self, name, module):
        params_with_grad, elems_with_grad, grad_norm = _grad_stats(module)
        print(
            f"  {name:<24} grad_params={params_with_grad:>4} "
            f"grad_elems={_fmt_params(elems_with_grad):>8} grad_norm={grad_norm:.4e}"
        )

    def _print_prompt_local_grad_row(self, name, prompt_learner):
        params_with_grad, elems_with_grad, grad_norm = _prompt_local_grad_stats(prompt_learner)
        print(
            f"  {name:<24} grad_params={params_with_grad:>4} "
            f"grad_elems={_fmt_params(elems_with_grad):>8} grad_norm={grad_norm:.4e}"
        )

    def _print_model_debug_summary(self):
        print("=" * 78)
        print("[CoPrompt Debug] Module parameter summary")
        print("[CoPrompt Debug] Note: prompt_learner_* rows include its registered CLIP reference.")
        self._print_module_param_row("CustomCLIP", self.model)
        self._print_module_param_row("prompt_learner_photo", self.model.prompt_learner_photo)
        self._print_module_param_row("prompt_learner_sketch", self.model.prompt_learner_sketch)
        self._print_prompt_local_row("photo ctx+proj only", self.model.prompt_learner_photo)
        self._print_prompt_local_row("sketch ctx+proj only", self.model.prompt_learner_sketch)
        self._print_module_param_row("ph_encoder", self.model.ph_encoder)
        self._print_module_param_row("sk_encoder", self.model.sk_encoder)
        self._print_module_param_row("text_encoder", self.model.text_encoder)
        self._print_module_param_row("model_distill", self.model.model_distill)
        if hasattr(self.model, "distill_proj"):
            self._print_module_param_row("distill_proj", self.model.distill_proj)
        if hasattr(self.model, "text_distill_proj"):
            self._print_module_param_row("text_distill_proj", self.model.text_distill_proj)

        total_trainable = _count_params(self.model, trainable_only=True)
        print(f"[CoPrompt Debug] Total trainable params: {_fmt_params(total_trainable)}")
        print(
            "[CoPrompt Debug] Loss weights: "
            f"cls={getattr(self.args, 'lambda_cls', 1.0)}, "
            f"triplet={getattr(self.args, 'lambda_triplet', 1.0)}, "
            f"nt_xent={getattr(self.args, 'lambda_nt_xent', 1.0)}, "
            f"distill_mode={getattr(self.args, 'distill_mode', 'kd_div')}"
        )
        print("=" * 78)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-3, momentum=0.9)
        trainable = sum(p.numel() for group in optimizer.param_groups for p in group["params"] if p.requires_grad)
        print(
            "[CoPrompt Debug] Optimizer: SGD "
            f"lr={self.args.lr}, momentum=0.9, weight_decay=1e-3, "
            f"trainable_params={_fmt_params(trainable)}"
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
        classname = get_all_categories(self.args)
        features = self.forward(batch, classname)
        if (
            not self._feature_debug_printed
            and batch_idx == 0
            and getattr(self.trainer, "is_global_zero", True)
        ):
            feature_names = [
                "photo_features_norm",
                "sk_feature_norm",
                "photo_aug_features",
                "sk_aug_features",
                "neg_features",
                "label",
                "pos_logits",
                "sk_logits",
                "photo_features_raw",
                "sk_features_raw",
                "photo_distill_features",
                "sk_distill_features",
                "neg_distill_features",
                "photo_text_distill_features",
                "sk_text_distill_features",
                "teacher_text_features",
            ]
            print("=" * 78)
            print("[CoPrompt Debug] First train batch feature contract")
            for name, value in zip(feature_names, features):
                print(_tensor_debug(name, value))
            print("=" * 78)
            self._feature_debug_printed = True
        
        loss, loss_dict = loss_fn(self.args, self.model, features=features, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        for k, v in loss_dict.items():
            show_on_bar = (
                k.startswith('kd_')
                or k.startswith('infonce_')
                or k.startswith('tw_')
                or k.startswith('ind_')
            )
            bar_name = (
                k.replace("kd_sk_ph", "KD_SP")
                .replace("kd_ph_txt", "KD_PT")
                .replace("kd_sk_txt", "KD_ST")
                .replace("infonce_photo_text", "I_PT")
                .replace("infonce_sketch_text", "I_ST")
                .replace("infonce_photo", "I_PH")
                .replace("infonce_sketch", "I_SK")
                .replace("tw_ntxent", "TW_NTX")
                .replace("ind_photo", "IND_PH")
                .replace("ind_sketch_photo", "IND_SP")
                .replace("ind_sketch", "IND_SK")
                .replace("ind_text", "IND_TXT")
            )
            self.log(bar_name, v, on_step=True, on_epoch=False, prog_bar=show_on_bar)
        return loss

    def on_after_backward(self):
        if self._grad_debug_printed or not getattr(self.trainer, "is_global_zero", True):
            return

        print("=" * 78)
        print("[CoPrompt Debug] First backward non-zero gradient summary")
        self._print_grad_row("CustomCLIP", self.model)
        self._print_prompt_local_grad_row("photo ctx+proj only", self.model.prompt_learner_photo)
        self._print_prompt_local_grad_row("sketch ctx+proj only", self.model.prompt_learner_sketch)
        self._print_grad_row("ph_encoder", self.model.ph_encoder)
        self._print_grad_row("sk_encoder", self.model.sk_encoder)
        self._print_grad_row("text_encoder", self.model.text_encoder)
        self._print_grad_row("model_distill", self.model.model_distill)
        if hasattr(self.model, "distill_proj"):
            self._print_grad_row("distill_proj", self.model.distill_proj)
        if hasattr(self.model, "text_distill_proj"):
            self._print_grad_row("text_distill_proj", self.model.text_distill_proj)
        print("=" * 78)
        self._grad_debug_printed = True
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        # classnames = get_all_categories(self.args, mode="test")
        classnames = get_all_categories(self.args, mode="train")
        image_tensor, label = batch
        if dataloader_idx == 0:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='sketch')
            self.val_step_outputs_sk.append((feat, label))
            modality = "sketch"
        else:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='photo')
            self.val_step_outputs_ph.append((feat, label))
            modality = "photo"
        
        if self.args.visualize:
            feat = feat.detach().cpu()
            label = label.detach().cpu()

            for f, l in zip(feat, label):
                self.saved_features[str(int(l))][modality].append(f)
    
    def on_validation_epoch_end(self):
        if self.args.visualize:
            # visualize_classes = UNSEEN_CLASSES[self.args.dataset]
            visualize_classes = VISUALIZE_CLASSES[self.args.dataset]
            visualize_tsne(visualize_classes, self.saved_features, mode="photo")
            visualize_tsne(visualize_classes, self.saved_features, mode="sketch")
        else:
            query_len = len(self.val_step_outputs_sk)
            gallery_len = len(self.val_step_outputs_ph)
            
            query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
            gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])
            
            all_sketch_category = np.array(sum([list(self.val_step_outputs_sk[i][1].detach().cpu().numpy()) for i in range(query_len)], []))
            all_photo_category = np.array(sum([list(self.val_step_outputs_ph[i][1].detach().cpu().numpy()) for i in range(gallery_len)], []))
            
            ## mAP category-level SBIR Metrics
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
                self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
            
            if map_k != 0:
                print('mAP@{}: {}, P@{}: {}, Best mAP: {}'.format(map_k, mAP.item(), p_k, precision, self.best_metric))
            else:
                print('mAP@all: {}, P@{}: {}, Best mAP: {}'.format(mAP.item(), p_k, precision, self.best_metric))
            train_loss = self.trainer.callback_metrics.get("train_loss", None)

            if train_loss is not None:
                print(f"Train loss (epoch avg): {train_loss.item():.6f}")
                
        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
        self.saved_features.clear()
