"""O-LoRA: in addition to the task-local cross-entropy, each task's LoRA-A is
pushed orthogonal to every previous task's frozen LoRA-A and L2-regularized."""

import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from models.lora import Learner as LoRALearner
from utils.toolkit import tensor2numpy


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        self.lamda_1 = args.get("lamda_1", 0.5)   # orthogonality weight
        self.lamda_2 = args.get("lamda_2", 0.0)   # L2 weight on current LoRA
        # accumulate (sum) previous frozen + current LoRA by default
        self.train_merge = bool(args.get("lora_train_merge", True))

    # sample-boundary streaming hooks: one adapter slot per chunk, with the
    # orthogonality penalty snapshotting against previous chunk adapters
    def _stream_init(self):
        self._stream_chunk = -1

    def _stream_slot(self):
        return self._stream_chunk

    def _stream_begin_chunk(self, loader):
        self._stream_chunk += 1
        if self._stream_chunk > 0:
            self._network.add_task_slot()
        self._cur_task = self._stream_chunk        # _orth_and_l2 reads _cur_task as current slot
        self._network.freeze_to_task(self._stream_chunk, train_a=True)
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._stream_new_optimizer()
        self._refresh_orth_cache()

    def _stream_extra_loss(self, lo, hi):
        orth, l2 = self._orth_and_l2()
        return self.lamda_1 * orth + self.lamda_2 * l2

    def _refresh_orth_cache(self):
        """Rebuild the per-(block, projection) concatenated frozen-A cache used by the
        orthogonality penalty, once per task rather than per step. No-op on task 0 and
        on any later call within the same task."""
        t = self._train_adapter()
        if t <= 0:
            return
        for attn in self._network.attn_modules():
            for proj, A_list in (("q", attn.lora_A_q), ("v", attn.lora_A_v)):
                cache_attr, task_attr = "_orth_prev_" + proj, "_orth_prev_task_" + proj
                if getattr(attn, task_attr, None) != t:
                    stacked = torch.cat([A_list[s].weight.detach() for s in range(t)], dim=0)
                    setattr(attn, cache_attr, stacked)
                    setattr(attn, task_attr, t)

    def _orth_and_l2(self):
        t = self._train_adapter()
        orth = 0.0
        l2 = 0.0
        for attn in self._network.attn_modules():
            for proj, (A_list, B_list) in (("q", (attn.lora_A_q, attn.lora_B_q)),
                                           ("v", (attn.lora_A_v, attn.lora_B_v))):
                A_t = A_list[t].weight                       # [r, dim]
                l2 = l2 + torch.norm(A_t, p=2) + torch.norm(B_list[t].weight, p=2)
                if t > 0:
                    cache_attr = "_orth_prev_" + proj
                    A_prev = getattr(attn, cache_attr)       # [(t*r), dim] (frozen)
                    orth = orth + torch.abs(A_t @ A_prev.t()).sum()
        return orth, l2

    # persistent_state() falls back to models/lora.py's, which sums every
    # currently-allocated slot's A+B+head (no separate fold accounting here
    # since O-LoRA uses factored, non-folded forward)
    def _train(self, train_loader):
        self._refresh_orth_cache()

        self._network.to(self._device)
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, ce_run, orth_run, correct, total = 0.0, 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._train_adapter(), merge=self.train_merge)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)

                orth, l2 = self._orth_and_l2()
                loss = ce + self.lamda_1 * orth + self.lamda_2 * l2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_run += ce.item()
                orth_run += float(orth)

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, Orth {:.3f}), Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs,
                    losses / len(train_loader), ce_run / len(train_loader),
                    orth_run / len(train_loader), train_acc))
        logging.info("[O-LoRA] Task {} done. Acc {:.2f}, final Orth {:.3f}".format(
            self._cur_task, train_acc, orth_run / len(train_loader)))
