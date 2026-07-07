import os

import torch

from clip import clip
from src.data_config import UNSEEN_CLASSES


def get_all_categories(args):
    categories = os.listdir(os.path.join(args.root, "sketch"))
    categories = [name for name in categories if name != ".ipynb_checkpoints"]
    return sorted(set(categories) - set(UNSEEN_CLASSES[args.dataset]))


def load_clip_to_cpu(cfg, design_details=None):
    model_path = clip._download(clip._MODELS[cfg.backbone])
    try:
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
    return clip.build_model(state_dict or model.state_dict(), design_details)


def retrieval_precision(preds, target, top_k):
    ranked_target = target[preds.argsort(dim=-1, descending=True)]
    relevant_count = int(ranked_target.sum().item())
    if relevant_count == 0:
        return torch.tensor(0.0, device=preds.device)
    return ranked_target[:min(top_k, relevant_count)].float().mean()
