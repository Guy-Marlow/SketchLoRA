"""RainbowPromptModule implements the base-knowledge pool, task-conditioning,
Prompt_Evolution and stored-prompt inference for RainbowPrompt. Task-id inference
at test time is batch-level: it sums each candidate task's key similarity over the
whole batch before taking the argmax."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RainbowPromptModule(nn.Module):
    def __init__(self, num_layers, embed_dim, n_tasks, num_heads=12, length=20,
                 self_attn_idx=None, KI_iter=10, use_linear=True, D1=28, D2=56,
                 evolve_dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.num_heads = num_heads
        self.length = length
        self.self_attn_idx = set(self_attn_idx if self_attn_idx is not None else [])
        self.KI_iter = KI_iter
        self.use_linear = use_linear
        self.evolve_dropout = evolve_dropout

        if self.use_linear:
            self.query_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in range(num_layers)])
            self.key_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in range(num_layers)])
            self.value_matcher = nn.ModuleList([nn.Linear(embed_dim, D2) for _ in range(num_layers)])
            self.dense = nn.ModuleList([nn.Linear(D2, embed_dim) for _ in range(num_layers)])
            self.fc1 = nn.ModuleList([nn.Linear(embed_dim, D1) for _ in range(num_layers)])
            self.fc2 = nn.ModuleList([nn.Linear(D1, embed_dim) for _ in range(num_layers)])

        # One base-knowledge entry per task per layer, [n_tasks, length, embed_dim],
        # initialized uniform on [0, 1) to match the reference.
        self.base_knowledge = nn.ParameterList([
            nn.Parameter(torch.empty(n_tasks, length, embed_dim).uniform_())
            for _ in range(num_layers)])
        self.base_key = nn.Parameter(torch.empty(n_tasks, embed_dim).uniform_())

        self.register_buffer('stored_prompts', torch.zeros(n_tasks, num_layers, length, embed_dim))

    @torch.no_grad()
    def add_task_slot(self):
        """Append one more task's row to base_knowledge, base_key and stored_prompts, using the constructor's own init convention. Used by the streaming setting to grow slot count beyond what was preallocated."""
        device, dtype = self.base_key.device, self.base_key.dtype
        for layer in range(self.num_layers):
            new_row = torch.empty(1, self.length, self.embed_dim, device=device, dtype=dtype).uniform_()
            self.base_knowledge[layer] = nn.Parameter(
                torch.cat([self.base_knowledge[layer].data, new_row], dim=0))
        new_key_row = torch.empty(1, self.embed_dim, device=device, dtype=dtype).uniform_()
        self.base_key = nn.Parameter(torch.cat([self.base_key.data, new_key_row], dim=0))
        new_stored = torch.zeros(1, self.num_layers, self.length, self.embed_dim,
                                  device=device, dtype=dtype)
        self.stored_prompts = torch.cat([self.stored_prompts, new_stored], dim=0)
        self.n_tasks += 1

    @staticmethod
    def l2n(x, dim=-1):
        return F.normalize(x, p=2, dim=dim, eps=1e-12)

    def task_conditioning(self, base_knowledge, task_key):
        """Attention-based relevance reweighting of base_knowledge against task_key."""
        key_exp = task_key.view(1, 1, -1).expand(base_knowledge.size(0), base_knowledge.size(1), -1)
        scores = torch.matmul(base_knowledge, key_exp.transpose(-1, -2))
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, base_knowledge)

    def evolve_step(self, prev, curr, layer):
        """One Prompt_Evolution update step combining attention and, if use_linear, an FFN sublayer."""
        d = curr.shape[-1]
        if not self.use_linear:
            attn = torch.softmax(torch.matmul(curr, prev.transpose(-1, -2)) / math.sqrt(d), dim=-1)
            out = torch.matmul(attn, prev)
            return F.layer_norm(prev + out, [d])

        # use_linear=True: q from curr, k/v from prev.
        q = self.query_matcher[layer](curr)   # [length, D2]
        k = self.key_matcher[layer](prev)     # [length, D2]
        v = self.value_matcher[layer](prev)   # [length, D2]

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(weights, v)                                    # [length, D2]

        # Second attention pass along the projected-feature axis, same q/k.
        q_t, k_t = q.transpose(-1, -2), k.transpose(-1, -2)                # [D2, length]
        t_logits = torch.matmul(q_t, k_t.transpose(-1, -2)) / math.sqrt(q_t.shape[-1])
        t_weights = torch.softmax(t_logits, dim=-1)
        out = torch.matmul(t_weights, out.transpose(-1, -2)).transpose(-1, -2)  # [length, D2]
        attn_out = self.dense[layer](out)                                  # [length, embed_dim]

        attn_out = F.dropout(attn_out, self.evolve_dropout, training=True)
        out1 = F.layer_norm(prev + attn_out, [d])

        ffn_out = self.fc2[layer](F.relu(self.fc1[layer](out1)))
        ffn_out = F.dropout(ffn_out, self.evolve_dropout, training=True)
        return F.layer_norm(out1 + ffn_out, [d])

    def _inject_shape(self, evolved, batch_size):
        """[length, embed_dim] -> key/value prefix pair, each [B, num_heads, length/2, head_dim]."""
        half = self.length // 2
        head_dim = self.embed_dim // self.num_heads
        key_prompt = evolved[:half].view(half, self.num_heads, head_dim).permute(1, 0, 2)
        value_prompt = evolved[half:].view(half, self.num_heads, head_dim).permute(1, 0, 2)
        key_prompt = key_prompt.unsqueeze(0).expand(batch_size, -1, -1, -1)
        value_prompt = value_prompt.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return key_prompt, value_prompt

    def forward(self, layer, task_id, cls_feat, train, batch_size, known_task=None):
        """If known_task is set and train is False, retrieve that task's stored prompt directly instead of matching keys."""
        if not train and known_task is not None:
            evolved = self.stored_prompts[known_task, layer]
            key_prompt, value_prompt = self._inject_shape(evolved, batch_size)
            return key_prompt, value_prompt, None

        base_knowledge = self.base_knowledge[layer]   # [n_tasks, length, embed_dim]
        if train:
            task_key = self.base_key[task_id]
            key_n = self.l2n(task_key, dim=0)
            feat_n = self.l2n(cls_feat, dim=-1)
            sim_loss = torch.matmul(key_n, feat_n.t()).sum() / feat_n.shape[0]

            curr_bk = base_knowledge[task_id:task_id + 1]                # [1, length, embed_dim]
            attended_curr = self.task_conditioning(curr_bk, key_n)[0]    # [length, embed_dim]

            if layer in self.self_attn_idx:
                p = attended_curr
                for _ in range(self.KI_iter):
                    p = self.evolve_step(p, p, layer)
                evolved = p
            elif task_id == 0:
                # Task 0 still runs one evolve_step on itself rather than passing
                # the task-conditioned base knowledge through unchanged.
                evolved = self.evolve_step(attended_curr, attended_curr, layer)
            else:
                prev_bk = base_knowledge[0:task_id].detach()             # [task_id, length, embed_dim]
                attended_prev = self.task_conditioning(prev_bk, key_n)   # [task_id, length, embed_dim]
                evolved_list = [self.evolve_step(attended_prev[t], attended_curr, layer) for t in range(task_id)]
                evolved = torch.stack(evolved_list, dim=0).mean(dim=0)

            with torch.no_grad():
                self.stored_prompts[task_id, layer] = evolved.detach()
        else:
            key_n = self.l2n(self.base_key[:task_id + 1], dim=-1)     # [task_id+1, embed_dim]
            feat_n = self.l2n(cls_feat, dim=-1)                       # [B, embed_dim]
            sims = torch.matmul(key_n, feat_n.t()).sum(dim=1) / feat_n.shape[0]   # [task_id+1]
            pred_task = int(sims.argmax().item())
            evolved = self.stored_prompts[pred_task, layer]
            sim_loss = None

        key_prompt, value_prompt = self._inject_shape(evolved, batch_size)
        return key_prompt, value_prompt, sim_loss
