import torch
import torch.nn as nn
import torch.nn.functional as F


class DFNPromptLearner(nn.Module):
    def __init__(self, cfg, dfn_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.n_ctx = cfg.n_ctx
        self.prompt_prefix = "a photo/sketch of "
        self.dropout_layer = nn.Dropout(p=0.1)
        self._token_embedding = [dfn_model.token_embedding]

        text_dim = dfn_model.ln_final.weight.shape[0]
        visual_dim = dfn_model.visual.conv1.out_channels
        prompt = tokenizer(self.prompt_prefix)
        with torch.no_grad():
            embedding = self._token_embedding[0](prompt.to(next(dfn_model.parameters()).device))
        ctx_vectors = embedding[0, 1:1 + self.n_ctx, :].detach().clone()
        if ctx_vectors.shape[0] != self.n_ctx:
            ctx_vectors = torch.empty(self.n_ctx, text_dim, device=embedding.device, dtype=embedding.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors.float())
        self.proj = nn.Linear(text_dim, visual_dim)
        single_layer = nn.Linear(text_dim, visual_dim)
        self.compound_prompt_projections = nn.ModuleList(
            [nn.Linear(text_dim, visual_dim) for _ in range(max(cfg.prompt_depth - 1, 0))]
        )
        for layer in self.compound_prompt_projections:
            layer.load_state_dict(single_layer.state_dict())

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        raw_prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = self.tokenizer(raw_prompts).to(self.ctx.device)

        with torch.no_grad():
            embedding = self._token_embedding[0](tokenized_prompts)

        ctx = self.ctx
        if self.training:
            ctx = self.dropout_layer(ctx)
        ctx = ctx.to(dtype=embedding.dtype)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(classnames), -1, -1)

        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx:, :]
        prompts = self.construct_prompts(ctx, prefix, suffix)
        shared_ctx = self.proj(self.ctx.to(dtype=self.proj.weight.dtype))
        return tokenized_prompts, prompts, shared_ctx


class OpenCLIPPromptTextEncoder(nn.Module):
    def __init__(self, dfn_model):
        super().__init__()
        object.__setattr__(self, "transformer_ref", dfn_model.transformer)
        object.__setattr__(self, "ln_final_ref", dfn_model.ln_final)
        self._positional_embedding = [dfn_model.positional_embedding]
        self._text_projection = [dfn_model.text_projection]
        self._attn_mask = [dfn_model.attn_mask]
        self.text_eos_id = getattr(dfn_model, "text_eos_id", None)

    def forward(self, prompts, tokenized_prompts):
        cast_dtype = self.transformer_ref.get_cast_dtype()
        x = prompts.to(cast_dtype) + self._positional_embedding[0].to(cast_dtype)
        attn_mask = self._attn_mask[0]
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x.device)

        for block in self.transformer_ref.resblocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_final_ref(x)
        if self.text_eos_id is None:
            pooled = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)]
        else:
            eos = tokenized_prompts.eq(self.text_eos_id).int().argmax(dim=-1)
            pooled = x[torch.arange(x.shape[0], device=x.device), eos]

        text_projection = self._text_projection[0]
        if text_projection is not None:
            if isinstance(text_projection, nn.Linear):
                pooled = text_projection(pooled)
            else:
                pooled = pooled @ text_projection
        return pooled


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
        )

    def forward(self, x):
        return self.fc(x)


def freeze_all_but_layernorm(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters(recurse=False):
                p.requires_grad_(True)


class DFNCoPromptTeacher(nn.Module):
    def __init__(self, cfg, dfn_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.model = dfn_model
        self.tokenizer = tokenizer
        self.dtype = next(dfn_model.parameters()).dtype
        self.output_dim = dfn_model.visual.output_dim
        self.visual_dim = dfn_model.visual.conv1.out_channels
        self.text_dim = dfn_model.ln_final.weight.shape[0]

        freeze_all_but_layernorm(self.model)

        self.prompt_learner_photo = DFNPromptLearner(cfg, dfn_model, tokenizer)
        self.prompt_learner_sketch = DFNPromptLearner(cfg, dfn_model, tokenizer)
        self.text_encoder = OpenCLIPPromptTextEncoder(dfn_model)
        self._logit_scale = [dfn_model.logit_scale]

        self.adapter_photo = Adapter(self.output_dim, 4)
        self.adapter_text = Adapter(self.output_dim, 4)
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.1

        txt_guided_prompts = torch.empty(
            len(self.prompt_learner_photo.compound_prompt_projections),
            cfg.n_ctx,
            self.text_dim,
        )
        nn.init.normal_(txt_guided_prompts, std=0.02)
        self.register_buffer("txt_guided_prompts", txt_guided_prompts)

    def _visual_forward(self, image, shared_ctx, visual_deep_prompts):
        visual = self.model.visual
        x = visual.conv1(image.to(dtype=visual.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)

        visual_ctx = shared_ctx.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1)
        x = torch.cat([x, visual_ctx], dim=1)
        x = visual.ln_pre(x)

        n_ctx = visual_ctx.shape[1]
        counter = 0
        for idx, block in enumerate(visual.transformer.resblocks):
            if idx > 0 and counter < len(visual_deep_prompts):
                prefix = x[:, :x.shape[1] - n_ctx, :]
                visual_context = visual_deep_prompts[counter].to(device=x.device, dtype=x.dtype)
                visual_context = visual_context.expand(x.shape[0], -1, -1)
                x = torch.cat([prefix, visual_context], dim=1)
                counter += 1
            x = block(x)

        pooled, _ = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled

    def _deep_visual_prompts(self, prompt_learner):
        prompts = []
        for index, layer in enumerate(prompt_learner.compound_prompt_projections):
            text_prompt = self.txt_guided_prompts[index].to(layer.weight.device, dtype=layer.weight.dtype)
            prompts.append(layer(text_prompt))
        return prompts

    def get_logits(self, image, classnames, modality="photo", return_text=False):
        if modality == "photo":
            prompt_learner = self.prompt_learner_photo
        else:
            prompt_learner = self.prompt_learner_sketch

        tokenized_prompts, prompts, shared_ctx = prompt_learner(classnames)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        visual_deep_prompts = self._deep_visual_prompts(prompt_learner)
        image_features = self._visual_forward(image, shared_ctx, visual_deep_prompts)

        adapter_photo_dtype = self.adapter_photo.fc[0].weight.dtype
        adapter_text_dtype = self.adapter_text.fc[0].weight.dtype

        image_features = (
            self.image_adapter_m * self.adapter_photo(image_features.to(adapter_photo_dtype))
            + (1 - self.image_adapter_m) * image_features.to(adapter_photo_dtype)
        )
        text_features = (
            self.text_adapter_m * self.adapter_text(text_features.to(adapter_text_dtype))
            + (1 - self.text_adapter_m) * text_features.to(adapter_text_dtype)
        )

        image_norm = F.normalize(image_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        logits = self._logit_scale[0].exp() * image_norm @ text_norm.t()

        if return_text:
            return logits, image_norm, image_features, text_norm, text_features
        return logits, image_norm, image_features

    def encode_photo(self, image, classnames=None):
        if classnames is None:
            return self.model.encode_image(image)
        _, _, image_features = self.get_logits(image, classnames, modality="photo")
        return image_features

    def encode_sketch(self, image, classnames=None):
        if classnames is None:
            return self.model.encode_image(image)
        _, _, image_features = self.get_logits(image, classnames, modality="sketch")
        return image_features

    def encode_image(self, image):
        return self.model.encode_image(image)

    def encode_text(self, tokenized_prompts):
        return self.model.encode_text(tokenized_prompts)

    def get_text_features(self, classnames):
        photo_tokens, photo_prompts, _ = self.prompt_learner_photo(classnames)
        sketch_tokens, sketch_prompts, _ = self.prompt_learner_sketch(classnames)
        photo_text = self.text_encoder(photo_prompts, photo_tokens)
        sketch_text = self.text_encoder(sketch_prompts, sketch_tokens)
        text_features = 0.5 * (photo_text.float() + sketch_text.float())
        adapter_text_dtype = self.adapter_text.fc[0].weight.dtype
        text_features = (
            self.text_adapter_m * self.adapter_text(text_features.to(adapter_text_dtype))
            + (1 - self.text_adapter_m) * text_features.to(adapter_text_dtype)
        )
        return F.normalize(text_features, dim=-1)
