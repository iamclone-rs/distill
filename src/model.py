import copy
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from collections import defaultdict
from torchmetrics.functional import retrieval_average_precision #, retrieval_precision

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

from src.coprompt import MultiModalPromptLearner, Adapter, TextEncoder
from src.dfn_coprompt_teacher import DFNCoPromptTeacher
from src.utils import load_clip_to_cpu, get_all_categories, retrieval_precision, visualize_tsne
from src.losses import loss_fn
from src.data_config import VISUALIZE_CLASSES, UNSEEN_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------
_TEACHER_REGISTRY = {
    # key         : (open_clip model name,    pretrained tag)
    "dfn5b"       : ("ViT-H-14-quickgelu",   "dfn5b"),
}


def _needs_strong_teacher(args):
    if getattr(args, "teacher", "clip32") == "clip32":
        return False
    if getattr(args, "use_rkd", False):
        return (
            getattr(args, "lambda_rkd_sk_ph", 0.0) > 0
            or getattr(args, "lambda_rkd_ph_txt", 0.0) > 0
            or getattr(args, "lambda_rkd_sk_txt", 0.0) > 0
        )

    return (
        getattr(args, "lambda_photo_distill", 0.0) > 0
        or getattr(args, "lambda_sketch_distill", 0.0) > 0
        or getattr(args, "lambda_text_distill", 0.0) > 0
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


def _freeze_teacher(teacher):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _load_teacher(args):
    """
    Trả về strong_teacher model (frozen) hoặc None.

    args.teacher:
        'clip32'  → None  (dùng clip_model_distill ViT-B/32, hành vi gốc)
        'dfn5b'   → DFN5B-CLIP-H/14 (1024-dim, frozen, via open_clip)

    """
    teacher_key = getattr(args, "teacher", "clip32")

    if teacher_key == "clip32":
        print("[Teacher] clip32 (ViT-B/32) — hành vi gốc, cross_loss")
        return None

    if not _needs_strong_teacher(args):
        print(f"[Teacher] {teacher_key} được chọn nhưng distill weight = 0 -> bỏ qua strong teacher")
        return None

    if teacher_key not in _TEACHER_REGISTRY:
        raise ValueError(
            f"Teacher '{teacher_key}' không hợp lệ. "
            f"Chọn một trong: clip32, {', '.join(_TEACHER_REGISTRY)}"
        )

    if not OPEN_CLIP_AVAILABLE:
        raise ImportError(
            f"Teacher '{teacher_key}' yêu cầu open_clip. "
            "Cài bằng: pip install open-clip-torch"
        )

    model_name, pretrained = _TEACHER_REGISTRY[teacher_key]
    print(f"[Teacher] Đang load {teacher_key} ({model_name}, pretrained={pretrained})...")
    teacher, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    teacher.text_tokenizer = open_clip.get_tokenizer(model_name)
    ckpt_path = getattr(args, "teacher_ckpt", "")
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "teacher_coprompt_state_dict" in checkpoint:
            print(f"[Teacher] Loading DFN-CoPrompt teacher checkpoint: {ckpt_path}")
            teacher = DFNCoPromptTeacher(args, teacher, teacher.text_tokenizer)
            missing, unexpected = teacher.load_state_dict(
                checkpoint["teacher_coprompt_state_dict"],
                strict=False,
            )
            print(
                "[Teacher] DFN-CoPrompt checkpoint loaded -> "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        else:
            _load_teacher_checkpoint(teacher, ckpt_path)
    teacher = _freeze_teacher(teacher)
    teacher = teacher.to(device)
    if getattr(args, "quantize_fp16", False):
        if device.type != "cuda":
            print("[Teacher] quantize_fp16=True nhưng không có CUDA; giữ teacher ở FP32.")
        else:
            teacher = teacher.half()
            print(f"[Teacher] {teacher_key} quantize_fp16=True -> teacher chạy FP16")
    print(f"[Teacher] {teacher_key} đã sẵn sàng (frozen, output 1024-dim)")
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
        
        self.adapter_photo = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)
        
        # strong_teacher=None  → clip32 (giữ nguyên hành vi gốc)
        # strong_teacher=<model> → DFN5B, output 1024-dim, dùng RKD loss
        if strong_teacher is not None:
            self.model_distill = strong_teacher
            self._use_strong_teacher = True
        else:
            self.model_distill = clip_model_distill
            self._use_strong_teacher = False
        self._lambda_photo_distill = getattr(cfg, "lambda_photo_distill", 0.0)
        self._lambda_sketch_distill = getattr(cfg, "lambda_sketch_distill", 0.0)
        self._lambda_text_distill = getattr(cfg, "lambda_text_distill", 0.0)
        
        use_rkd = getattr(cfg, "use_rkd", False)
        lambda_rkd_sk_ph = getattr(cfg, "lambda_rkd_sk_ph", 0.0)
        lambda_rkd_ph_txt = getattr(cfg, "lambda_rkd_ph_txt", 0.0)
        lambda_rkd_sk_txt = getattr(cfg, "lambda_rkd_sk_txt", 0.0)

        self._image_distill_active = (
            self._lambda_photo_distill > 0 or self._lambda_sketch_distill > 0 or
            (use_rkd and (lambda_rkd_sk_ph > 0 or lambda_rkd_ph_txt > 0 or lambda_rkd_sk_txt > 0))
        )
        self._distill_text = (
            self._lambda_text_distill > 0 or
            (use_rkd and (lambda_rkd_ph_txt > 0 or lambda_rkd_sk_txt > 0))
        )
        self._distill_proj_requested = getattr(cfg, "use_distill_proj", False)
        self._teacher_fp16 = (
            self._use_strong_teacher
            and getattr(cfg, "quantize_fp16", False)
            and device.type == "cuda"
        )
        self._distill_proj_dim = 1024 if self._use_strong_teacher else 512
        
        old_distill_active = (self._lambda_photo_distill > 0 or self._lambda_sketch_distill > 0 or self._lambda_text_distill > 0)
        if self._use_strong_teacher and old_distill_active and not self._distill_proj_requested:
            raise ValueError(
                "Strong teacher output 1024-dim, student output 512-dim. "
                "Hãy bật --use_distill_proj khi dùng thuật toán InfoNCE cũ."
            )
        self._use_distill_proj = self._distill_proj_requested and (
            self._image_distill_active
        )
        if self._use_distill_proj:
            self.distill_proj = nn.Linear(512, self._distill_proj_dim, bias=False).to(clip_model.dtype)
            print(
                "[Distill] use_distill_proj=True -> "
                f"student feature 512 -> {self._distill_proj_dim}"
            )
        self._need_teacher_text = self._distill_text
        if self._distill_text:
            self.text_distill_proj = nn.Linear(512, self._distill_proj_dim, bias=False).to(clip_model.dtype)
        if self._need_teacher_text:
            self._teacher_text_cache = {}
        if self._distill_text:
            print(
                "[Distill] distill_text=True -> "
                f"student text 512 -> {self._distill_proj_dim}"
            )
        print(
            "[Distill] active branches -> "
            f"photo={self._lambda_photo_distill > 0} ({self._lambda_photo_distill}), "
            f"sketch={self._lambda_sketch_distill > 0} ({self._lambda_sketch_distill}), "
            f"text={self._lambda_text_distill > 0} ({self._lambda_text_distill})"
        )
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.1
        self.saved_features = defaultdict(lambda: {"sketch": [], "photo": []})

        # init token random
        text_dim = clip_model_distill.positional_embedding.shape[-1]
        num_guided_layers = len(
            self.prompt_learner_photo.compound_prompt_projections
        )
        txt_guided_prompts = torch.empty(
            num_guided_layers,
            cfg.n_ctx,
            text_dim
        )
        nn.init.normal_(txt_guided_prompts, std=0.02)

        # register_buffer: không học, không gradient
        self.register_buffer(
            "txt_guided_prompts",
            txt_guided_prompts
        )
    
    def project_distill_feature(self, feature):
        if not self._use_distill_proj:
            return feature
        return self.distill_proj(feature.type(self.dtype))

    def project_text_distill_feature(self, feature):
        if not self._distill_text:
            return feature
        return self.text_distill_proj(feature.type(self.dtype))

    def teacher_image_input(self, image):
        if not self._use_strong_teacher:
            return image
        return image.half() if self._teacher_fp16 else image.float()

    def get_teacher_text_features(self, classnames):
        if not self._need_teacher_text or not self._use_strong_teacher:
            return None

        cache_key = tuple(classnames)
        if cache_key in self._teacher_text_cache:
            return self._teacher_text_cache[cache_key]

        if hasattr(self.model_distill, "get_text_features"):
            with torch.no_grad():
                text_features = self.model_distill.get_text_features(classnames)
            self._teacher_text_cache[cache_key] = text_features
            return text_features

        tokenizer = getattr(self.model_distill, "text_tokenizer", None)
        if tokenizer is None:
            return None

        prompts = [
            "a photo/sketch of " + name.replace("_", " ") + "."
            for name in classnames
        ]
        tokenized = tokenizer(prompts).to(device)
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
            shared_ctx,
            # deep_compound_prompts_text,
            # deep_compound_prompts_vision,
        ) = prompt_learner(classnames)
        
        text_features, _ = self.text_encoder(
            prompts, tokenized_prompts, return_all=True
        ) # (n_classes, 512)

        # init token random
        txt_guided_prompts = self.txt_guided_prompts

        visual_deep_prompts = []
        for index, layer in enumerate(prompt_learner.compound_prompt_projections):
            text_prompt = txt_guided_prompts[index]
            text_prompt = layer(text_prompt.half())
            # text_prompt = text_prompt.mean(dim=0)
            visual_deep_prompts.append(text_prompt)

        image_features = image_encoder(
                img_tensor.type(self.dtype), shared_ctx, visual_deep_prompts
            ) # (batch_size, 768)
        
        x_a = self.adapter_photo(image_features)
        image_features = (
            self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
        )

        x_b = self.adapter_text(text_features)
        text_features = (
            self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features
        )

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
        
        use_rkd = getattr(self.cfg, "use_rkd", False)
        lambda_rkd_sk_ph = getattr(self.cfg, "lambda_rkd_sk_ph", 0.0)
        lambda_rkd_ph_txt = getattr(self.cfg, "lambda_rkd_ph_txt", 0.0)
        lambda_rkd_sk_txt = getattr(self.cfg, "lambda_rkd_sk_txt", 0.0)
        
        train_photo_distill = self._lambda_photo_distill > 0 or (use_rkd and (lambda_rkd_sk_ph > 0 or lambda_rkd_ph_txt > 0))
        train_sketch_distill = self._lambda_sketch_distill > 0 or (use_rkd and (lambda_rkd_sk_ph > 0 or lambda_rkd_sk_txt > 0))
        photo_aug_features = photo_feature.detach()
        sk_aug_features = sk_feature.detach()

        if self._image_distill_active:
            with torch.no_grad():
                if train_photo_distill:
                    teacher_input = self.teacher_image_input(photo_aug_tensor)
                    if hasattr(self.model_distill, "encode_photo"):
                        photo_aug_features = self.model_distill.encode_photo(teacher_input, classnames)
                    else:
                        photo_aug_features = self.model_distill.encode_image(teacher_input)
                if train_sketch_distill:
                    teacher_input = self.teacher_image_input(sk_aug_tensor)
                    if hasattr(self.model_distill, "encode_sketch"):
                        sk_aug_features = self.model_distill.encode_sketch(teacher_input, classnames)
                    else:
                        sk_aug_features = self.model_distill.encode_image(teacher_input)
        photo_distill_feature = self.project_distill_feature(photo_feature)
        sk_distill_feature = self.project_distill_feature(sk_feature)
        neg_distill_feature = self.project_distill_feature(neg_raw_feature)
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
        self.saved_features = defaultdict(lambda: {"sketch": [], "photo": []})
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-3, momentum=0.9)
        
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
        
        loss, loss_dict = loss_fn(self.args, self.model, features=features, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        for k, v in loss_dict.items():
            show_on_bar = k.startswith('rkd')
            # Rút gọn tên để không bị vỡ giao diện
            bar_name = k.replace("rkd_sk_ph", "R_SP").replace("rkd_ph_txt", "R_PT").replace("rkd_sk_txt", "R_ST")
            self.log(bar_name, v, on_step=True, on_epoch=False, prog_bar=show_on_bar)
        return loss
    
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
