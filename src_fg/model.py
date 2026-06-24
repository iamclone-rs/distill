import copy
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from collections import defaultdict

from src_fg.coprompt import MultiModalPromptLearner, Adapter, TextEncoder
from src.utils import load_clip_to_cpu, get_all_categories
from src_fg.losses import loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_model(m):
    m.requires_grad_(False)
    
def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)
            
class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, clip_model, clip_model_distill
    ):
        super().__init__()
        clip_model.apply(freeze_all_but_bn)
        clip_model_distill.apply(freeze_all_but_bn)
        self.dtype = clip_model.dtype
        self.prompt_learner_photo = MultiModalPromptLearner(cfg, clip_model, type='photo')
        self.prompt_learner_sketch = MultiModalPromptLearner(cfg, clip_model, type='sketch')
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
        self.adapter_photo = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)
        
        self.model_distill = clip_model_distill
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.1
    
    def get_logits(self, img_tensor, classnames, type='photo'):
        if type=='photo':
            prompt_learner = self.prompt_learner_photo
        else:
            prompt_learner = self.prompt_learner_sketch
        # tokenized_prompts = prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        (
            tokenized_prompts,
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = prompt_learner(classnames)
        
        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        ) # (n_classes, 512)
        
        image_features = self.image_encoder(
                img_tensor.type(self.dtype), shared_ctx, deep_compound_prompts_vision
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
        _, neg_feature_norm, neg_feature = self.get_logits(neg_tensor, classnames)
        
        photo_aug_features = self.model_distill.encode_image(photo_aug_tensor)
        sk_aug_features = self.model_distill.encode_image(sk_aug_tensor)
        
        return photo_features_norm, sk_feature_norm, neg_feature_norm, photo_aug_features, sk_aug_features, \
            label, pos_logits, sk_logits, photo_feature, sk_feature, neg_feature
            
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
        
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.best_metric = 1e-3
        
        self.model = CustomCLIP(cfg=args, clip_model=clip_model, clip_model_distill=clip_model_distill)
        self.val = defaultdict(lambda: {
            "val_img_features": [],
            "val_img_names": [],
            "val_sk_features": [],
            "val_sk_names": [],
        })
        self._seen_img_names = defaultdict(set)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-3, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=7,
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
    
    def validation_step(self, batch, batch_idx):
        classnames = get_all_categories(self.args, mode="test")
        sk_tensor, sk_name, img_tensor, pos_name, label = batch
        
        sk_feature = self.model.extract_feature(sk_tensor, classname=classnames, type='sketch')
        img_feature = self.model.extract_feature(img_tensor, classname=classnames, type='photo')
        
        if torch.is_tensor(label):
            label_list = label.detach().cpu().tolist()
        else:
            label_list = list(label)
            
        for i in range(len(label_list)):
            lab = label_list[i]
            
            self.val[lab]["val_sk_features"].append(sk_feature[i].detach().cpu())
            self.val[lab]["val_sk_names"].append(sk_name[i])
            
            p_name = pos_name[i]
            if p_name not in self.val[lab]["val_img_names"]:
                self.val[lab]["val_img_names"].append(p_name)
                self.val[lab]["val_img_features"].append(img_feature[i].detach().cpu())
                    
    def on_validation_epoch_end(self):
        top1_total, top5_total, total_sk = 0, 0, 0
        
        for category, bucket in self.val.items():
            rank = torch.zeros(len(bucket["val_sk_names"]))
            val_img_feature = torch.stack(bucket["val_img_features"])
            
            if len(bucket["val_img_features"]) == 0:
                print("rank: ", rank)
                print(category)
                continue
            for num, sketch_feature in enumerate(bucket["val_sk_features"]):
                s_name = bucket["val_sk_names"][num]
                sk_query_name = s_name.split('/')[-1].split('-')[:-1][0]
                
                position_query = bucket["val_img_names"].index(sk_query_name)
                
                distance = self.distance_fn(sketch_feature.unsqueeze(0), val_img_feature)
                target_distance = self.distance_fn(
                    sketch_feature.unsqueeze(0),
                    val_img_feature[position_query].unsqueeze(0)
                )
                
                rank[num] = distance.le(target_distance).sum()
                
            top1_total = top1_total + rank.le(1).sum().numpy()
            top5_total = top5_total + rank.le(5).sum().numpy()
            total_sk = total_sk + rank.shape[0]
                
        top1 = top1_total / total_sk
        top5 = top5_total / total_sk
        
        self.log("top1", top1, on_step=False, on_epoch=True)
        print('top1: {}, top5: {}'.format(top1, top5))
        self.val.clear()