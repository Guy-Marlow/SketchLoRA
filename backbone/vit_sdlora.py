"""SD-LoRA backbone, built on backbone/vit_lora.py's VisionTransformer/Block/
Attention_LoRA plus one learnable per-task magnitude scalar, shared across
every block, that rescales each frozen past task's normalized LoRA term."""
import math
from functools import partial

import torch
from torch import nn

from backbone.vit_lora import (
    Attention_LoRA, Block, PatchEmbed, VisionTransformer, _load_pretrained_split_qkv,
)

SCALE_INIT = 0.8


class Attention_LoRA_SDLoRA(Attention_LoRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scales_ref = None   # set via set_task_scales(), a plain list not an nn.Module attr

    def set_task_scales(self, scales):
        self._scales_ref = [scales]

    @torch.no_grad()
    def add_task_slot(self, rank=None):
        """Same as the base add_task_slot, except the new slot may be allocated at a different rank (SD-LoRA-RR); earlier slots keep their own rank."""
        r = rank if rank is not None else self.lora_rank
        ref = self.frozen_delta_q
        device, dtype = ref.device, ref.dtype
        new_A_q = nn.Linear(self.dim, r, bias=False).to(device=device, dtype=dtype)
        new_B_q = nn.Linear(r, self.dim, bias=False).to(device=device, dtype=dtype)
        new_A_v = nn.Linear(self.dim, r, bias=False).to(device=device, dtype=dtype)
        new_B_v = nn.Linear(r, self.dim, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(new_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(new_B_q.weight)
        nn.init.zeros_(new_B_v.weight)
        self.lora_A_q.append(new_A_q)
        self.lora_B_q.append(new_B_q)
        self.lora_A_v.append(new_A_v)
        self.lora_B_v.append(new_B_v)
        self.n_tasks += 1

    def _lora_delta(self, x, A_list, B_list, frozen_delta):
        task, merge = self._task, self._merge
        if task < 0:
            return 0.0
        scales = self._scales_ref[0]
        if not merge:
            # Task-routed forward: this task's own term alone, unnormalized.
            return scales[task] * B_list[task](A_list[task](x))
        delta = scales[task] * B_list[task](A_list[task](x))
        for s in range(task):
            B_s, A_s = B_list[s].weight, A_list[s].weight
            norm = torch.linalg.norm(B_s) * torch.linalg.norm(A_s)
            delta = delta + scales[s] * (B_list[s](A_list[s](x)) / norm)
        return delta


class Block_SDLoRA(Block):
    def __init__(self, dim, num_heads=12, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 n_tasks=1, lora_rank=10, lora_alpha=None):
        super().__init__(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                          act_layer=act_layer, norm_layer=norm_layer,
                          n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha)
        # Replace the plain Attention_LoRA super().__init__ built, to swap in the SD-LoRA attention class.
        self.attn = Attention_LoRA_SDLoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                          attn_drop=attn_drop, proj_drop=drop, n_tasks=n_tasks,
                                          lora_rank=lora_rank, lora_alpha=lora_alpha)


class VisionTransformerSDLoRA(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, n_tasks=1, lora_rank=10, lora_alpha=None, n_lora_blocks=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                          num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                          num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                          drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                          norm_layer=norm_layer, act_layer=act_layer, n_tasks=n_tasks,
                          lora_rank=lora_rank, lora_alpha=lora_alpha, n_lora_blocks=n_lora_blocks)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # Replace the plain-Block self.blocks super().__init__ built, with SD-LoRA blocks.
        self.blocks = nn.ModuleList([
            Block_SDLoRA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                        norm_layer=norm_layer, act_layer=act_layer, n_tasks=n_tasks,
                        lora_rank=lora_rank, lora_alpha=lora_alpha)
            for i in range(depth)])
        self.task_scales = nn.ParameterList([nn.Parameter(torch.tensor([SCALE_INIT]))])
        for attn in self.attn_modules():
            attn.set_task_scales(self.task_scales)

    def add_task_slot(self, rank=None):
        # Not calling super().add_task_slot(): re-implemented so the per-task
        # rank override reaches each block's new slot.
        for blk in self.blocks:
            blk.attn.add_task_slot(rank=rank)
        self.n_tasks += 1
        ref = self.task_scales[0]
        self.task_scales.append(nn.Parameter(torch.tensor([SCALE_INIT], device=ref.device, dtype=ref.dtype)))
        # Every block's _scales_ref already points at this same ParameterList,
        # so append() needs no re-propagation to the blocks.

    def freeze_to_task(self, task, train_a=True):
        super().freeze_to_task(task, train_a=train_a)
        # SD-LoRA re-activates every past task's magnitude scalar alongside
        # the new task's own A/B.
        for s in range(task + 1):
            self.task_scales[s].requires_grad = True


def vit_base_patch16_224_sdlora(pretrained=False, n_tasks=1, lora_rank=10, lora_alpha=None, **kwargs):
    model = VisionTransformerSDLoRA(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                    n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_sdlora(pretrained=False, n_tasks=1, lora_rank=10, lora_alpha=None, **kwargs):
    model = VisionTransformerSDLoRA(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                    n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224_in21k")
