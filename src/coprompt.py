"""Shallow CoPrompt components used by the student model."""

import torch
import torch.nn as nn

from clip import clip


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.resblocks = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        for block in self.resblocks:
            x = block(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return (
            x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )


class MultiModalPromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx: int, image_size: int = 224):
        super().__init__()
        if n_ctx <= 0 or n_ctx > 4:
            raise ValueError("The cleaned shallow prompt supports n_ctx in [1, 4].")
        if image_size != clip_model.visual.input_resolution:
            raise ValueError(
                f"image_size={image_size}, CLIP expects {clip_model.visual.input_resolution}."
            )

        self.n_ctx = n_ctx
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.token_embedding.requires_grad_(False)
        self.dropout = nn.Dropout(p=0.1)

        prompt_prefix = "a photo/sketch of "
        prompt = clip.tokenize(prompt_prefix)
        with torch.no_grad():
            embedding = self.token_embedding(prompt).type(self.dtype)
        self.ctx = nn.Parameter(embedding[0, 1 : 1 + n_ctx, :])
        self.prompt_prefix = prompt_prefix
        self.visual_projection = nn.Linear(self.ctx.shape[-1], 768)
        if self.dtype == torch.float16:
            self.visual_projection.half()

    def forward(self, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        raw_prompts = [f"{self.prompt_prefix} {name}." for name in classnames]
        tokenized = torch.cat([clip.tokenize(prompt) for prompt in raw_prompts])
        tokenized = tokenized.to(self.ctx.device)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized).type(self.dtype)

        context = self.dropout(self.ctx) if self.training else self.ctx
        context = context.unsqueeze(0).expand(len(classnames), -1, -1)
        prompts = torch.cat(
            (embedding[:, :1, :], context, embedding[:, 1 + self.n_ctx :, :]),
            dim=1,
        )
        return tokenized, prompts, self.visual_projection(self.ctx)
