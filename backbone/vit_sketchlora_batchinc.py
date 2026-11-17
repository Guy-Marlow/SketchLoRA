"""ViT backbone for the batch-incremental drift-detection SketchLoRA setting.
Decouples spawning a new live adapter from folding it into the fixed sketch,
allowing any number of frozen, not-yet-folded slots in between."""
import math
from functools import partial

import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed


class Attention_LoRABatchInc(nn.Module):
    """Multi-head attention with Q,V LoRA: one fixed sketch plus a growing list of slots, of which one is live and zero or more are pending."""

    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 lora_rank=10, lora_alpha=None, svd_rank=10):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lora_rank = lora_rank
        self.svd_rank = svd_rank
        self.lora_scaling = (lora_alpha if lora_alpha is not None else lora_rank) / lora_rank

        # the one live slot, preallocated at construction (slot 0).
        self.lora_A_q = nn.ModuleList([nn.Linear(dim, lora_rank, bias=False)])
        self.lora_B_q = nn.ModuleList([nn.Linear(lora_rank, dim, bias=False)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, lora_rank, bias=False)])
        self.lora_B_v = nn.ModuleList([nn.Linear(lora_rank, dim, bias=False)])
        nn.init.kaiming_uniform_(self.lora_A_q[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q[0].weight)
        nn.init.zeros_(self.lora_B_v[0].weight)

        # The one fixed sketch, starting at an exact zero map (zero B).
        self.sketch_A_q = nn.Linear(dim, svd_rank, bias=False)
        self.sketch_B_q = nn.Linear(svd_rank, dim, bias=False)
        self.sketch_A_v = nn.Linear(dim, svd_rank, bias=False)
        self.sketch_B_v = nn.Linear(svd_rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.sketch_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.sketch_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.sketch_B_q.weight)
        nn.init.zeros_(self.sketch_B_v.weight)

        # Routing state, broadcast from VisionTransformerBatchInc.set_routing() once per forward.
        self._pending = []
        self._live_idx = 0
        # Frozen-feature probe mode for the KL detector: when True, _delta
        # returns exactly 0 regardless of pending/live/sketch state.
        self._adapters_disabled = False

    def set_routing(self, pending, live_idx):
        self._pending = pending
        self._live_idx = live_idx

    def set_adapters_disabled(self, flag):
        self._adapters_disabled = flag

    @torch.no_grad()
    def add_task_slot(self):
        """Append one more (A,B) pair for q and v; pending/live bookkeeping is the caller's responsibility."""
        ref = self.sketch_B_q.weight
        device, dtype = ref.device, ref.dtype
        new_A_q = nn.Linear(self.dim, self.lora_rank, bias=False).to(device=device, dtype=dtype)
        new_B_q = nn.Linear(self.lora_rank, self.dim, bias=False).to(device=device, dtype=dtype)
        new_A_v = nn.Linear(self.dim, self.lora_rank, bias=False).to(device=device, dtype=dtype)
        new_B_v = nn.Linear(self.lora_rank, self.dim, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(new_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(new_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(new_B_q.weight)
        nn.init.zeros_(new_B_v.weight)
        self.lora_A_q.append(new_A_q)
        self.lora_B_q.append(new_B_q)
        self.lora_A_v.append(new_A_v)
        self.lora_B_v.append(new_B_v)

    @torch.no_grad()
    def fold_pending(self, pending_indices):
        """Merge the sketch plus every pending slot into a new svd_rank-sized sketch via QR-SVD truncation, then free the folded slots. Returns the retained energy for the q and v projections."""
        retained = {}
        for proj, A_list, B_list, sketch_A, sketch_B in (
                ("q", self.lora_A_q, self.lora_B_q, self.sketch_A_q, self.sketch_B_q),
                ("v", self.lora_A_v, self.lora_B_v, self.sketch_A_v, self.sketch_B_v)):
            A_sk, B_sk = sketch_A.weight, sketch_B.weight
            dt = B_sk.dtype
            parts_A = [A_sk.float()] + [A_list[i].weight.float() for i in pending_indices]
            parts_B = [B_sk.float()] + [B_list[i].weight.float() for i in pending_indices]

            delta_true = sum(B @ A for A, B in zip(parts_A, parts_B))
            B_cat = torch.cat(parts_B, dim=1)     # [dim, svd_rank + sum(pending ranks)]
            A_cat = torch.cat(parts_A, dim=0)     # [svd_rank + sum(pending ranks), dim]
            Q_B, R_B = torch.linalg.qr(B_cat)
            Q_A, R_A = torch.linalg.qr(A_cat.t())
            M_small = R_B @ R_A.t()
            U_small, S, Vh_small = torch.linalg.svd(M_small)
            U = Q_B @ U_small
            Vh = Vh_small @ Q_A.t()

            r = min(self.svd_rank, S.shape[0])
            sqrt_s = S[:r].sqrt()
            A_new = (sqrt_s.unsqueeze(1) * Vh[:r, :]).to(dt)
            B_new = (U[:, :r] * sqrt_s).to(dt)

            recon = B_new.float() @ A_new.float()
            fro_true = torch.linalg.norm(delta_true).clamp_min(1e-12)
            retained[proj] = (1.0 - (torch.linalg.norm(delta_true - recon) / fro_true) ** 2).item()

            sketch_A.weight.data.copy_(A_new)
            sketch_B.weight.data.copy_(B_new)

        for i in pending_indices:
            self.lora_A_q[i] = nn.Identity()
            self.lora_B_q[i] = nn.Identity()
            self.lora_A_v[i] = nn.Identity()
            self.lora_B_v[i] = nn.Identity()
        return retained

    def orth_loss(self):
        """Soft orthogonality penalty between the live slot's down-projection and the detached sketch's, plus one detached term per pending slot."""
        orth = 0.0
        for A_list, sketch_A in ((self.lora_A_q, self.sketch_A_q), (self.lora_A_v, self.sketch_A_v)):
            A_live = A_list[self._live_idx].weight
            A_sketch = sketch_A.weight.detach()
            orth = orth + torch.abs(A_live @ A_sketch.t()).sum()
            for i in self._pending:
                A_other = A_list[i].weight.detach()
                orth = orth + torch.abs(A_live @ A_other.t()).sum()
        return orth

    def _delta(self, x, A_list, B_list, sketch_A, sketch_B):
        if self._adapters_disabled:
            return 0.0
        delta = sketch_B(sketch_A(x))
        for i in self._pending:
            delta = delta + B_list[i](A_list[i](x))
        delta = delta + B_list[self._live_idx](A_list[self._live_idx](x))
        return delta * self.lora_scaling

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x) + self._delta(x, self.lora_A_q, self.lora_B_q, self.sketch_A_q, self.sketch_B_q)
        k = self.k_proj(x)
        v = self.v_proj(x) + self._delta(x, self.lora_A_v, self.lora_B_v, self.sketch_A_v, self.sketch_B_v)

        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)
        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 lora_rank=10, lora_alpha=None, svd_rank=10):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRABatchInc(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop,
                                           lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def set_routing(self, pending, live_idx):
        self.attn.set_routing(pending, live_idx)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))
        x = residual + x
        return x


class VisionTransformerBatchInc(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, lora_rank=10, lora_alpha=None, svd_rank=10, pending_cap=4):
        super().__init__()
        print("Initialized ViT-B/16 backbone with batch-incremental sketch+pending+live LoRA (Q, V).")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.lora_rank = lora_rank
        self.svd_rank = svd_rank
        self.pending_cap = pending_cap
        # n_tasks counts slots ever spawned, not CIL tasks.
        self.n_tasks = 1
        self._pending = []     # list of frozen, not-yet-folded slot indices
        self._live_idx = 0     # the one slot currently receiving gradients

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  act_layer=act_layer, lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    def attn_modules(self):
        return [blk.attn for blk in self.blocks]

    # Spawn and fold mechanics; when to call these is the driver's decision.
    def add_task_slot(self):
        """Freeze the current live slot into pending and spawn a fresh live slot; does not check or enforce pending_cap itself."""
        if self._live_idx is not None:
            self._pending.append(self._live_idx)
        new_idx = self.n_tasks
        for blk in self.blocks:
            blk.attn.add_task_slot()
        self.n_tasks += 1
        self._live_idx = new_idx

    def fold_pending(self):
        """Fold every pending slot, never the live one, into the sketch for every block. Returns each block's retained energy per projection."""
        pending_indices = list(self._pending)
        retained = {}
        for li, attn in enumerate(self.attn_modules()):
            r = attn.fold_pending(pending_indices)
            retained[(li, "q")] = r["q"]
            retained[(li, "v")] = r["v"]
        self._pending = []
        return retained

    def orth_loss(self):
        return sum(attn.orth_loss() for attn in self.attn_modules())

    # Routing.
    def set_routing(self):
        for blk in self.blocks:
            blk.set_routing(self._pending, self._live_idx)

    def set_adapters_disabled(self, flag):
        """For the KL detector's frozen-feature probe: when True, every block's sketch/pending/live contribution is forced to 0; the caller must reset it to False afterward."""
        for attn in self.attn_modules():
            attn.set_adapters_disabled(flag)

    # Memory accounting: sketch, pending and live tracked separately.
    def persistent_bytes(self):
        sketch_bytes = pending_bytes = live_bytes = 0
        for attn in self.attn_modules():
            for sketch_mod in (attn.sketch_A_q, attn.sketch_B_q, attn.sketch_A_v, attn.sketch_B_v):
                sketch_bytes += sum(p.numel() * p.element_size() for p in sketch_mod.parameters())
            for i in self._pending:
                for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                    pending_bytes += sum(p.numel() * p.element_size() for p in mlist[i].parameters())
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                live_bytes += sum(p.numel() * p.element_size() for p in mlist[self._live_idx].parameters())
        return {"sketch": sketch_bytes, "pending": pending_bytes, "live": live_bytes}

    # Forward pass.
    def forward_features(self, x):
        self.set_routing()
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        return self.forward_features(x)


def _load_pretrained_split_qkv(model, timm_name):
    """Load a timm ViT-B/16 checkpoint into the split-qkv batch-incremental model."""
    checkpoint_model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            w = state_dict.pop(key)
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = w[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = w[768:768 * 2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = w[768 * 2:]
        elif 'qkv.bias' in key:
            b = state_dict.pop(key)
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = b[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = b[768:768 * 2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = b[768 * 2:]
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_sketchlora_batchinc(pretrained=False, lora_rank=10, lora_alpha=None,
                                             svd_rank=10, pending_cap=4, **kwargs):
    model = VisionTransformerBatchInc(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                      lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank,
                                      pending_cap=pending_cap, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_sketchlora_batchinc(pretrained=False, lora_rank=10, lora_alpha=None,
                                                   svd_rank=10, pending_cap=4, **kwargs):
    model = VisionTransformerBatchInc(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                      lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank,
                                      pending_cap=pending_cap, **kwargs)
    return _load_pretrained_split_qkv(model, "vit_base_patch16_224_in21k")
