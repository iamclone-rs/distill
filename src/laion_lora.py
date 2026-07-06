"""Modality-specific Q/V LoRA for OpenCLIP visual transformer blocks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityQVMultiheadAttention(nn.Module):
    """Frozen MultiheadAttention plus separate sketch/photo Q/V LoRA paths."""

    def __init__(self, base_attn, rank=16, alpha=32.0, dropout=0.05):
        super().__init__()
        if not isinstance(base_attn, nn.MultiheadAttention):
            raise TypeError("LoRA wrapper expects torch.nn.MultiheadAttention.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_attn = base_attn
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.embed_dim = int(base_attn.embed_dim)
        self.num_heads = int(base_attn.num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = bool(base_attn.batch_first)
        self.dropout = nn.Dropout(float(dropout))
        self.active_modality = "photo"

        dtype = base_attn.in_proj_weight.dtype
        device = base_attn.in_proj_weight.device
        for modality in ("sketch", "photo"):
            for projection in ("q", "v"):
                a = nn.Parameter(
                    torch.empty(self.rank, self.embed_dim, device=device, dtype=dtype)
                )
                b = nn.Parameter(
                    torch.zeros(self.embed_dim, self.rank, device=device, dtype=dtype)
                )
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                setattr(self, f"lora_{modality}_{projection}_a", a)
                setattr(self, f"lora_{modality}_{projection}_b", b)

        self.base_attn.requires_grad_(False)

    def set_modality(self, modality):
        if modality not in ("sketch", "photo", "paired"):
            raise ValueError(f"Unknown LoRA modality: {modality}")
        self.active_modality = modality

    def _single_lora(self, inputs, projection, modality):
        a = getattr(self, f"lora_{modality}_{projection}_a")
        b = getattr(self, f"lora_{modality}_{projection}_b")
        return F.linear(F.linear(self.dropout(inputs), a), b) * self.scaling

    def _lora(self, inputs, projection):
        if self.active_modality != "paired":
            return self._single_lora(inputs, projection, self.active_modality)
        batch_dim = 0 if self.batch_first else 1
        if inputs.shape[batch_dim] % 2:
            raise RuntimeError(
                "Paired LoRA expects an even batch: sketches first, photos second."
            )
        split = inputs.shape[batch_dim] // 2
        sketch_inputs, photo_inputs = inputs.split(split, dim=batch_dim)
        return torch.cat(
            (
                self._single_lora(sketch_inputs, projection, "sketch"),
                self._single_lora(photo_inputs, projection, "photo"),
            ),
            dim=batch_dim,
        )

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
        **kwargs,
    ):
        del average_attn_weights, kwargs
        if key_padding_mask is not None:
            raise NotImplementedError("Visual LoRA does not support key_padding_mask.")

        weight_q, weight_k, weight_v = self.base_attn.in_proj_weight.chunk(3, dim=0)
        if self.base_attn.in_proj_bias is None:
            bias_q = bias_k = bias_v = None
        else:
            bias_q, bias_k, bias_v = self.base_attn.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, weight_q, bias_q) + self._lora(query, "q")
        k = F.linear(key, weight_k, bias_k)
        v = F.linear(value, weight_v, bias_v) + self._lora(value, "v")

        if self.batch_first:
            batch_size, query_length, _ = q.shape
            key_length = k.shape[1]
        else:
            query_length, batch_size, _ = q.shape
            key_length = k.shape[0]

        def split_heads(tensor, sequence_length):
            if self.batch_first:
                return tensor.view(
                    batch_size, sequence_length, self.num_heads, self.head_dim
                ).permute(0, 2, 1, 3)
            return tensor.view(
                sequence_length, batch_size, self.num_heads, self.head_dim
            ).permute(1, 2, 0, 3)

        q = split_heads(q, query_length)
        k = split_heads(k, key_length)
        v = split_heads(v, key_length)

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.view(
                    batch_size, self.num_heads, query_length, key_length
                )
            attn_mask = attn_mask.to(device=q.device, dtype=q.dtype)

        dropout_p = self.base_attn.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        if self.batch_first:
            output = output.permute(0, 2, 1, 3).reshape(
                batch_size, query_length, self.embed_dim
            )
        else:
            output = output.permute(2, 0, 1, 3).reshape(
                query_length, batch_size, self.embed_dim
            )
        output = self.base_attn.out_proj(output)
        return output, None if not need_weights else None


def inject_visual_qv_lora(model, num_layers=4, rank=16, alpha=32.0, dropout=0.05):
    blocks = model.visual.transformer.resblocks
    total_layers = len(blocks)
    if num_layers <= 0 or num_layers > total_layers:
        raise ValueError(
            f"LoRA num_layers must be in [1, {total_layers}], got {num_layers}."
        )
    layer_indices = list(range(total_layers - num_layers, total_layers))
    for index in layer_indices:
        if isinstance(blocks[index].attn, ModalityQVMultiheadAttention):
            raise RuntimeError(f"Visual block {index} already has LoRA.")
        blocks[index].attn = ModalityQVMultiheadAttention(
            blocks[index].attn,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    return layer_indices


def set_lora_modality(model, modality):
    for module in model.modules():
        if isinstance(module, ModalityQVMultiheadAttention):
            module.set_modality(modality)


def lora_named_parameters(model):
    for module_name, module in model.named_modules():
        if not isinstance(module, ModalityQVMultiheadAttention):
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name.startswith("lora_"):
                full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                yield full_name, parameter


def lora_state_dict(model):
    return {
        name: parameter.detach().cpu()
        for name, parameter in lora_named_parameters(model)
    }


def load_lora_state_dict(model, state_dict):
    target = dict(lora_named_parameters(model))
    missing = sorted(set(target) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(target))
    incompatible = sorted(
        name
        for name in set(target) & set(state_dict)
        if target[name].shape != state_dict[name].shape
    )
    if missing or unexpected or incompatible:
        raise RuntimeError(
            "Incompatible LoRA state: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={incompatible[:5]}"
        )
    with torch.no_grad():
        for name, parameter in target.items():
            parameter.copy_(state_dict[name].to(parameter))


def lora_parameter_count(model):
    return sum(parameter.numel() for _, parameter in lora_named_parameters(model))
