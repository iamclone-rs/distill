import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from src.utils import get_clones
from src.data_config import UNSEEN_CLASSES

_tokenizer = _Tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [
            x,
            compound_prompts_deeper_text,
            0,
        ]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
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
        txt_ctx = cfg.txt_ctx
        n_ctx = cfg.n_ctx
        ctx_init = "visual representation of "
            
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.max_size
        
        self.dropout_layer = nn.Dropout(p=0.1)
        self.compound_prompts_depth = (
            cfg.prompt_depth
        )  # max=12, but will create 11 such shared prompts
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (txt_ctx) <= 4:
            # use given words to initialize context vectors
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            ctx_prompt = embedding[0, 1 : 1 + txt_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            ctx_prompt = torch.empty(txt_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors)
            prompt_prefix = ctx_init
        
        # self.register_buffer("ctx_prompt", ctx_prompt)
        self.ctx_prompt = nn.Parameter(ctx_prompt)
        self.prompt_prefix = prompt_prefix
        self.proj = nn.Linear(ctx_dim, 768)
        self.bv = nn.Linear(ctx_dim, 768)
        
        if dtype == torch.float16:
            self.proj.half()
            self.bv.half()
        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, 512))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for single_para in self.compound_prompts_text:
            nn.init.kaiming_uniform_(single_para)
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix, label=None):
            
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, classnames, label=None):
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
        
        prompts = self.construct_prompts(ctx, prefix, suffix, label)
        
        # Before returning, need to transform prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            text_prompt = self.compound_prompts_text[index]
            if self.training:
                text_prompt = self.dropout_layer(text_prompt)
            visual_deep_prompts.append(layer(text_prompt))
            
        ctx_prompt_proj = self.bv(self.ctx_prompt)

        combined_prompts = [
            torch.cat([vdp, ctx_prompt_proj], dim=0)
            for vdp in visual_deep_prompts
        ]
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        
        return (
            tokenized_prompts,
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text, # deep_compound_prompts_text
            # visual_deep_prompts,        # deep_compound_prompts_vision
            combined_prompts,        # deep_compound_prompts_vision
        )  # pass here original, as for visual 768 is required

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x