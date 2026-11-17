"""SD-LoRA+SketchLoRA backbone: SD-LoRA's magnitude/direction decoupling
applied to a SketchLoRA fixed-size sketch, with the sketch's singular values
made per-task re-learnable via a rescaling vector. Each block keeps an
explicit factored SVD of the folded history plus one reused new-adapter slot
per projection."""

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from backbone.vit_lora import (
    Attention_LoRA, Block, PatchEmbed, VisionTransformer, _load_pretrained_split_qkv,
)

NEW_ADAPTER_INIT_STD = 0.02

_PROJS = ("q", "v")


class Attention_SDLoRASketch(Attention_LoRA):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 lora_rank=10, lora_alpha=None, svd_rank=10, freeze_rescale=False):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                          attn_drop=attn_drop, proj_drop=proj_drop,
                          n_tasks=1, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.svd_rank = svd_rank
        self.freeze_rescale = freeze_rescale
        for proj in _PROJS:
            self.register_buffer("sketch_U_{}".format(proj), torch.zeros(dim, svd_rank))
            self.register_buffer("sketch_S_{}".format(proj), torch.zeros(svd_rank))
            self.register_buffer("sketch_Vh_{}".format(proj), torch.zeros(svd_rank, dim))
            setattr(self, "rescale_{}".format(proj), nn.Parameter(torch.zeros(svd_rank)))
            setattr(self, "mag_{}".format(proj), nn.Parameter(torch.zeros(1)))
            getattr(self, "rescale_{}".format(proj)).requires_grad = not freeze_rescale
        self._new_task_params()

    @torch.no_grad()
    def _new_task_params(self):
        """Reset the new adapter's direction factors, magnitude and rescale vector; called at construction and at every task boundary."""
        for proj in _PROJS:
            A = getattr(self, "lora_A_{}".format(proj))[0].weight
            B = getattr(self, "lora_B_{}".format(proj))[0].weight
            nn.init.normal_(A, std=NEW_ADAPTER_INIT_STD)
            nn.init.normal_(B, std=NEW_ADAPTER_INIT_STD)
            getattr(self, "rescale_{}".format(proj)).data.zero_()
            getattr(self, "mag_{}".format(proj)).data.zero_()

    @staticmethod
    def _apply_rescale(S, rescale):
        """Effective singular values S*(1+rescale)."""
        return S * (1.0 + rescale)

    def _sketch_term(self, x, proj):
        """Apply the sketch as project-in, scale, project-out, without materializing the dense delta."""
        U = getattr(self, "sketch_U_{}".format(proj))
        S = getattr(self, "sketch_S_{}".format(proj))
        Vh = getattr(self, "sketch_Vh_{}".format(proj))
        rescale = getattr(self, "rescale_{}".format(proj))
        S_eff = self._apply_rescale(S, rescale)
        proj_in = F.linear(x, Vh)       # [..., r_hat] = x @ Vh^T
        scaled = proj_in * S_eff         # elementwise scale by the (rescaled) singular values
        return F.linear(scaled, U)       # [..., dim] = scaled @ U^T

    def _new_adapter_term(self, x, proj):
        """mag * normalize(B_new @ A_new) applied to x, computing the norm via the trace trick without forming the dense product."""
        A = getattr(self, "lora_A_{}".format(proj))[0]
        B = getattr(self, "lora_B_{}".format(proj))[0]
        mag = getattr(self, "mag_{}".format(proj))
        Aw, Bw = A.weight, B.weight          # [r,dim], [dim,r]
        AAt = Aw @ Aw.t()
        BtB = Bw.t() @ Bw
        normsq = (AAt * BtB).sum().clamp_min(1e-12)
        scale = mag / normsq.sqrt()
        return scale * B(A(x))

    def _lora_delta(self, x, A_list, B_list, frozen_delta):
        proj = "q" if A_list is self.lora_A_q else "v"
        return self._sketch_term(x, proj) + self._new_adapter_term(x, proj)

    @torch.no_grad()
    def fold(self):
        """Task-boundary merge: bake this task's rescale into S and its magnitude into the new adapter, QR-SVD-merge both into the sketch at svd_rank in double precision, then reset for the next task. Returns the retained energy for the q and v projections."""
        retained = {}
        for proj in _PROJS:
            U = getattr(self, "sketch_U_{}".format(proj))
            S = getattr(self, "sketch_S_{}".format(proj))
            Vh = getattr(self, "sketch_Vh_{}".format(proj))
            rescale = getattr(self, "rescale_{}".format(proj))
            mag = getattr(self, "mag_{}".format(proj))
            A = getattr(self, "lora_A_{}".format(proj))[0].weight
            B = getattr(self, "lora_B_{}".format(proj))[0].weight

            S_eff = S.double() * (1.0 + rescale.double())
            sign = torch.sign(S_eff)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            sqrt_abs = S_eff.abs().sqrt()
            B_sk = U.double() * sqrt_abs                            # [dim, r_hat]
            A_sk = (sign * sqrt_abs).unsqueeze(1) * Vh.double()     # [r_hat, dim]

            Ad, Bd = A.double(), B.double()
            AAt = Ad @ Ad.t()
            BtB = Bd.t() @ Bd
            normsq = (AAt * BtB).sum().clamp_min(1e-12)
            adapter_scale = mag.double() / normsq.sqrt()
            B_ad = Bd * adapter_scale                                # fold the scalar magnitude into one factor
            A_ad = Ad

            delta_true = (B_sk @ A_sk) + (B_ad @ A_ad)               # transient, diagnostic-only

            B_cat = torch.cat([B_sk, B_ad], dim=1)                   # [dim, r_hat+r]
            A_cat = torch.cat([A_sk, A_ad], dim=0)                   # [r_hat+r, dim]
            Q_B, R_B = torch.linalg.qr(B_cat)
            Q_A, R_A = torch.linalg.qr(A_cat.t())
            M_small = R_B @ R_A.t()
            U_small, S_new, Vh_small = torch.linalg.svd(M_small)
            U_full = Q_B @ U_small
            Vh_full = Vh_small @ Q_A.t()

            r = min(self.svd_rank, S_new.shape[0])
            recon = (U_full[:, :r] * S_new[:r]) @ Vh_full[:r, :]
            fro_true = delta_true.norm().clamp_min(1e-12)
            retained_energy = (1.0 - ((delta_true - recon).norm() / fro_true) ** 2).item()
            retained[proj] = retained_energy

            U.zero_()
            S.zero_()
            Vh.zero_()
            U[:, :r] = U_full[:, :r].to(U.dtype)
            S[:r] = S_new[:r].to(S.dtype)
            Vh[:r, :] = Vh_full[:r, :].to(Vh.dtype)

        self._new_task_params()
        return retained


class Block_SDLoRASketch(Block):
    def __init__(self, dim, num_heads=12, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 lora_rank=10, lora_alpha=None, svd_rank=10, freeze_rescale=False):
        super().__init__(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                          act_layer=act_layer, norm_layer=norm_layer,
                          n_tasks=1, lora_rank=lora_rank, lora_alpha=lora_alpha)
        # Replace the plain Attention_LoRA super().__init__ built, with the SD-LoRA-Sketch attention class.
        self.attn = Attention_SDLoRASketch(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop,
                                           lora_rank=lora_rank, lora_alpha=lora_alpha,
                                           svd_rank=svd_rank, freeze_rescale=freeze_rescale)


class VisionTransformerSDLoRASketch(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, lora_rank=10, lora_alpha=None, n_lora_blocks=None,
                 svd_rank=10, freeze_rescale=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                          num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                          num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                          drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                          norm_layer=norm_layer, act_layer=act_layer, n_tasks=1,
                          lora_rank=lora_rank, lora_alpha=lora_alpha, n_lora_blocks=n_lora_blocks)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # Replace the plain-Block self.blocks super().__init__ built, with SD-LoRA-Sketch blocks.
        self.blocks = nn.ModuleList([
            Block_SDLoRASketch(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                               norm_layer=norm_layer, act_layer=act_layer,
                               lora_rank=lora_rank, lora_alpha=lora_alpha,
                               svd_rank=svd_rank, freeze_rescale=freeze_rescale)
            for i in range(depth)])
        self.svd_rank = svd_rank

    def freeze_to_task(self, task, train_a=True):
        """Freeze everything then re-enable slot 0's A/B and the magnitude/rescale params."""
        for p in self.parameters():
            p.requires_grad = False
        train_blocks = self.blocks if self.n_lora_blocks is None else self.blocks[:self.n_lora_blocks]
        for blk in train_blocks:
            attn = blk.attn
            for proj in _PROJS:
                getattr(attn, "lora_A_{}".format(proj))[0].weight.requires_grad = True
                getattr(attn, "lora_B_{}".format(proj))[0].weight.requires_grad = True
                getattr(attn, "mag_{}".format(proj)).requires_grad = True
                getattr(attn, "rescale_{}".format(proj)).requires_grad = not attn.freeze_rescale

    def add_task_slot(self):
        raise NotImplementedError(
            "SD-LoRA-Sketch is fixed-size and never grows a per-task slot bank.")

    def fold(self):
        """Fold every block's current new-adapter into its sketch. Returns
        each block's retained energy for the q and v projections, for the
        learner to aggregate and log."""
        return {i: blk.attn.fold() for i, blk in enumerate(self.blocks)}


def vit_base_patch16_224_sdlora_sketch(pretrained=False, lora_rank=10, lora_alpha=None,
                                        svd_rank=10, freeze_rescale=False, **kwargs):
    model = VisionTransformerSDLoRASketch(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                          mlp_ratio=4, qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          lora_rank=lora_rank, lora_alpha=lora_alpha,
                                          svd_rank=svd_rank, freeze_rescale=freeze_rescale, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_sdlora_sketch(pretrained=False, lora_rank=10, lora_alpha=None,
                                              svd_rank=10, freeze_rescale=False, **kwargs):
    model = VisionTransformerSDLoRASketch(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                          mlp_ratio=4, qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          lora_rank=lora_rank, lora_alpha=lora_alpha,
                                          svd_rank=svd_rank, freeze_rescale=freeze_rescale, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224_in21k")
