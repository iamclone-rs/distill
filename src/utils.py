import os
import torch
import torch.nn as nn
from torch import Tensor, tensor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from torch.nn import functional as F

from clip import clip
from src.data_config import UNSEEN_CLASSES

def retrieval_average_precision(preds, target, top_k = None):
    top_k = top_k or preds.shape[-1]
    if not isinstance(top_k, int) and top_k <= 0:
        raise ValueError(f"Argument ``top_k`` has to be a positive integer or None, but got {top_k}.")

    target = target[preds.topk(min(top_k, preds.shape[-1]), sorted=True, dim=-1)[1]]

    if not target.sum():
        return tensor(0.0, device=preds.device)

    positions = torch.arange(1, len(target) + 1, device=target.device, dtype=torch.float32)[target > 0]
    return torch.div((torch.arange(len(positions), device=positions.device, dtype=torch.float32) + 1), positions).mean()


def get_all_categories(args, mode="train"):
    all_categories = os.listdir(os.path.join(args.root, 'sketch'))
    unseen_classes = UNSEEN_CLASSES[args.dataset]
    if '.ipynb_checkpoints' in all_categories:
        all_categories.remove('.ipynb_checkpoints')
    if mode=="train":
        all_categories = sorted(list(set(all_categories) - set(unseen_classes)))
    else:
        all_categories = sorted(unseen_classes)
        # all_categories = sorted(list(set(all_categories)))
    return all_categories

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def load_clip_to_cpu(cfg, design_details=None):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if design_details is None:
        design_details = {
            "trainer": "CoPrompt",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": cfg.n_ctx,
        }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def retrieval_precision(preds, target, top_k):
    sorted_idx = preds.argsort(dim=-1, descending=True)
    sorted_target = target[sorted_idx]

    tot_pos = sorted_target.sum().item()

    if tot_pos == 0:
        return torch.tensor(0.0, device=preds.device)

    if top_k is not None:
        top = min(top_k, int(tot_pos))
    else:
        top = int(tot_pos)

    return sorted_target[:top].float().mean()

def visualize_tsne(visualize_classes, saved_features, mode="photo"):
    label_to_color = {
        "cow": "#E2514A",
        "raccoon": "#F5AA53",
        "scissors": "#FCE283",
        "seagull": "#EAF890",
        "sword": "#8ACA8F",
        "tree": "#4DA3B5",
    }


    if mode == "sketch":
        X = np.concatenate([torch.stack(v["sketch"]).cpu().numpy()
                            for v in saved_features.values() if len(v["sketch"]) > 0], axis=0)
        y = sum([[k] * len(v["sketch"])
                for k, v in saved_features.items() if len(v["sketch"]) > 0], [])

        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1)).fit_transform(X)

        plt.figure(figsize=(8, 6))
        for cls in sorted(set(y)):
            idx = [i for i, t in enumerate(y) if t == cls]
            name = visualize_classes[int(cls)]
            plt.scatter(
                Z[idx, 0], Z[idx, 1],
                s=20,
                c=label_to_color[name],
                marker="o",              
                label=name,  # đổi số -> chữ
                edgecolors="white",
                linewidths=0.5
            )

        ax = plt.gca()
        ax.set_xticks([])   # bỏ trục tọa độ
        ax.set_yticks([])
        for spine in ax.spines.values():   # bỏ đường viền
            spine.set_visible(False)

        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig("our_sketch.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
    
    else:
        X = np.concatenate([torch.stack(v["photo"]).cpu().numpy()
                            for v in saved_features.values() if len(v["photo"]) > 0], axis=0)
        y = sum([[k] * len(v["photo"])
                for k, v in saved_features.items() if len(v["photo"]) > 0], [])

        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1)).fit_transform(X)

        plt.figure(figsize=(8, 6))
        for cls in sorted(set(y)):
            idx = [i for i, t in enumerate(y) if t == cls]
            name = visualize_classes[int(cls)]
            plt.scatter(
                Z[idx, 0], Z[idx, 1],
                s=20,
                c=label_to_color[name],
                marker="o",              
                label=name,  # đổi số -> chữ
                edgecolors="white",
                linewidths=0.5
            )

        ax = plt.gca()
        ax.set_xticks([])   # bỏ trục tọa độ
        ax.set_yticks([])
        for spine in ax.spines.values():   # bỏ đường viền
            spine.set_visible(False)

        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig("our_photo.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
