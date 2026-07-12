import torch
import torch.nn as nn

from clip import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        x = x.permute(1, 0, 2)  # NLD -> LND
        for block in self.resblocks:
            x = block(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x
    
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model, type='photo'):
        super().__init__()
        self.clip_model = clip_model
        self.cfg = cfg
        n_ctx = cfg.n_ctx
        ctx_init = "a photo/sketch of "
            
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.max_size
        
        self.dropout_layer = nn.Dropout(p=0.1)
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors)
            prompt_prefix = ctx_init
        
        self.prompt_prefix = prompt_prefix
        self.proj = nn.Linear(ctx_dim, 768)

        # Keep the exact RNG sequence of the recorded shallow-prompt benchmark.
        # The original implementation initialized this projection even when
        # prompt_depth=1, then discarded it because there were no deep prompts.
        # Removing that unused allocation changes the sketch prompt projection
        # initialization and makes seed=42 reproduce a different run.
        _benchmark_rng_compat = nn.Linear(ctx_dim, 768)

        if dtype == torch.float16:
            self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)

    def forward(self, classnames):
        n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        raw_prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in raw_prompts])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts.to(device)).type(self.clip_model.dtype)
        
        ctx = self.ctx
        if self.training:
            ctx = self.dropout_layer(ctx)
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.cfg.n_ctx :, :]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        
        return (
            tokenized_prompts,
            prompts,
            self.proj(self.ctx),
        )
