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
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(device)
    print(f"[Teacher] {teacher_key} đã sẵn sàng (frozen, output 1024-dim) → RKD loss")
    return teacher

# ---------------------------------------------------------------------------

def freeze_model(m):
    m.requires_grad_(False)
    

def freeze_all_but_bn(m):
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
            
class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, clip_model, clip_model_distill, strong_teacher=None
    ):
        super().__init__()
        self.cfg = cfg
        clip_model.apply(freeze_all_but_bn)
        clip_model_distill.apply(freeze_all_but_bn)
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
        self._train_teacher_ln = getattr(cfg, "train_teacher_ln", False)
        if self._train_teacher_ln:
            teacher_visual = getattr(self.model_distill, "visual", self.model_distill)
            num_ln_params = unfreeze_layernorm_params(teacher_visual)
            print(
                "[Teacher] train_teacher_ln=True -> "
                f"unfreeze {num_ln_params} visual LayerNorm params for sketch teacher adaptation"
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
        
        return logits, image_features_normalize, image_features
        
    def forward(self, x, classnames):
        photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label = x
        pos_logits, photo_features_norm, photo_feature = self.get_logits(photo_tensor, classnames)
        sk_logits, sk_feature_norm, sk_feature = self.get_logits(sk_tensor, classnames, type='sketch')
        _, neg_feature, _ = self.get_logits(neg_tensor, classnames)
        
        if self._train_teacher_ln:
            # Keep photo teacher target fixed, but let sketch target update
            # visual LayerNorm params for sketch-domain adaptation.
            if self._use_strong_teacher:
                photo_teacher_input = photo_aug_tensor.float()
                sketch_teacher_input = sk_aug_tensor.float()
            else:
                photo_teacher_input = photo_aug_tensor
                sketch_teacher_input = sk_aug_tensor

            with torch.no_grad():
                photo_aug_features = self.model_distill.encode_image(photo_teacher_input)
            sk_aug_features = self.model_distill.encode_image(sketch_teacher_input)
        else:
            # Gộp photo_aug + sketch_aug thành 1 batch → 1 teacher forward pass thay vì 2
            # Tiết kiệm ~50% thời gian teacher inference (ViT-H/14 rất chậm nếu gọi 2 lần)
            with torch.no_grad():
                if self._use_strong_teacher:
                    aug_cat = torch.cat([photo_aug_tensor, sk_aug_tensor], dim=0).float()
                    aug_feats = self.model_distill.encode_image(aug_cat)          # (2B, 1024)
                else:
                    aug_cat = torch.cat([photo_aug_tensor, sk_aug_tensor], dim=0)
                    aug_feats = self.model_distill.encode_image(aug_cat)          # (2B, 512)
            
            B = photo_aug_tensor.shape[0]
            photo_aug_features = aug_feats[:B]   # (B, D)
            sk_aug_features    = aug_feats[B:]   # (B, D)
            
        return photo_features_norm, sk_feature_norm, photo_aug_features, sk_aug_features, \
            neg_feature, label, pos_logits, sk_logits, photo_feature, sk_feature
        
    def extract_feature(self, image, classname, type='photo'):
        _, feature, _ = self.get_logits(image, classnames=classname, type=type)
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
        
        loss = loss_fn(self.args, self.model, features=features, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
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
