"""SketchLoRA's orthogonality/alignment penalty between the trainable
residual and the frozen sketch, via independently selectable align/orth
modes on the A and B factors."""

import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from models.sketchlora import Learner as SketchLoRALearner
from models.sketchlora import SKETCH
from utils.toolkit import tensor2numpy


class Learner(SketchLoRALearner):
    def __init__(self, args):
        super().__init__(args)
        self.align_weight = args.get("sketchlora_align_weight", 0.5)
        # Independent B weight; align/orth losses scale differently.
        self.align_weight_b = args.get("sketchlora_align_weight_b", None)
        self.align_mode = args.get("sketchlora_align_mode", "align")
        assert self.align_mode in ("align", "orth", "none"), \
            "sketchlora_align_mode must be 'align' (pull toward sketch), 'orth' " \
            "(push away from sketch, O-LoRA's own formula), or 'none' (no " \
            "regularizer on A at all, only useful combined with sketchlora_align_b_mode)"
        # Lets B use a different mode than A.
        self.align_b_mode = args.get("sketchlora_align_b_mode", "none")
        assert self.align_b_mode in ("align", "orth", "none"), \
            "sketchlora_align_b_mode must be 'align', 'orth', or 'none'"
        # Also orthogonalizes B's columns, "orth" mode only.
        self.align_orth_ab = bool(args.get("sketchlora_align_orth_ab", False))
        if self.align_orth_ab and self.align_mode != "orth":
            logging.warning(
                "[SketchLoRA-Align] sketchlora_align_orth_ab=True has no effect "
                "under align_mode=%s. It only applies to align_mode='orth'.",
                self.align_mode)
        assert not (self.align_orth_ab and self.align_b_mode != "none"), \
            "sketchlora_align_orth_ab and sketchlora_align_b_mode both specify B's " \
            "treatment."

    def _attns(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net.attn_modules()

    def _align_loss(self):
        """Align mode: signed cosine similarity, A toward the sketch."""
        t = self._train_adapter()
        align = 0.0
        for attn in self._attns():
            for A_list in (attn.lora_A_q, attn.lora_A_v):
                A_new = A_list[t].weight
                A_sketch = A_list[SKETCH].weight.detach()
                A_new_n = F.normalize(A_new, dim=1)
                A_sketch_n = F.normalize(A_sketch, dim=1)
                align = align + (A_new_n @ A_sketch_n.t()).sum()
        return align

    def _orth_loss(self):
        """Orth mode: O-LoRA's penalty, A against the sketch."""
        t = self._train_adapter()
        orth = 0.0
        for attn in self._attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                    (attn.lora_A_v, attn.lora_B_v)):
                A_new = A_list[t].weight
                A_sketch = A_list[SKETCH].weight.detach()
                orth = orth + torch.abs(A_new @ A_sketch.t()).sum()
                if self.align_orth_ab:
                    orth = orth + self._orth_loss_b_term(attn, A_list, B_list)
        return orth

    def _orth_loss_b_term(self, attn, A_list, B_list):
        """Per-module B-orth term shared by two callers."""
        t = self._train_adapter()
        B_new = B_list[t].weight
        B_sketch = B_list[SKETCH].weight.detach()
        return torch.abs(B_new.t() @ B_sketch).sum()

    def _orth_loss_b(self):
        """Standalone B-only orth penalty."""
        orth = 0.0
        for attn in self._attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                    (attn.lora_A_v, attn.lora_B_v)):
                orth = orth + self._orth_loss_b_term(attn, A_list, B_list)
        return orth

    def _align_loss_b(self):
        """B-space counterpart of _align_loss."""
        t = self._train_adapter()
        align = 0.0
        for attn in self._attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                    (attn.lora_A_v, attn.lora_B_v)):
                B_new = B_list[t].weight
                B_sketch = B_list[SKETCH].weight.detach()
                B_new_n = F.normalize(B_new, dim=0)
                B_sketch_n = F.normalize(B_sketch, dim=0)
                align = align + (B_new_n.t() @ B_sketch_n).sum()
        return align

    def _current_align_weight(self):
        return self.align_weight

    def _reg_terms(self):
        """A's and B's regularizer values and signs, independently."""
        if self._cur_task <= 0:
            return 0.0, 1.0, 0.0, 1.0
        if self.align_mode == "none":
            reg_a, sign_a = 0.0, 1.0
        elif self.align_mode == "align":
            reg_a, sign_a = self._align_loss(), -1.0
        else:
            reg_a, sign_a = self._orth_loss(), 1.0
        if self.align_b_mode == "none":
            reg_b, sign_b = 0.0, 1.0
        elif self.align_b_mode == "align":
            reg_b, sign_b = self._align_loss_b(), -1.0
        else:
            reg_b, sign_b = self._orth_loss_b(), 1.0
        return reg_a, sign_a, reg_b, sign_b

    def _stream_extra_loss(self, lo, hi):
        """Streaming-driver hook providing the same regularizer term."""
        reg_a, sign_a, reg_b, sign_b = self._reg_terms()
        w_a = self._current_align_weight()
        w_b = self.align_weight_b if self.align_weight_b is not None else w_a
        return w_a * sign_a * reg_a + w_b * sign_b * reg_b

    def _train_core(self, train_loader):
        """Training loop plus the align-to-sketch term."""
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        b_active = self.align_b_mode != "none"
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, ce_run, rega_run, regb_run, correct, total = 0.0, 0.0, 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._train_adapter(), merge=self.train_merge)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)

                reg_a, sign_a, reg_b, sign_b = self._reg_terms()
                w_a = self._current_align_weight()
                w_b = self.align_weight_b if self.align_weight_b is not None else w_a
                loss = ce + w_a * sign_a * reg_a + w_b * sign_b * reg_b

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_run += ce.item()
                rega_run += float(reg_a)
                regb_run += float(reg_b)

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            n = len(train_loader)
            if b_active:
                prog_bar.set_description(
                    "[SketchLoRA-Align:{}+{}] Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, "
                    "A-{} {:.3f} (wA={:.4f}), B-{} {:.3f} (wB={:.4f})), Acc {:.2f}".format(
                        self.align_mode, self.align_b_mode, self._cur_task, epoch + 1, self.epochs,
                        losses / n, ce_run / n, self.align_mode, rega_run / n, w_a,
                        self.align_b_mode, regb_run / n, w_b, train_acc))
            else:
                prog_bar.set_description(
                    "[SketchLoRA-Align:{}] Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, {} {:.3f}, w={:.3f}), Acc {:.2f}".format(
                        self.align_mode, self._cur_task, epoch + 1, self.epochs,
                        losses / n, ce_run / n,
                        self.align_mode, rega_run / n, w_a, train_acc))
        if b_active:
            logging.info(
                "[SketchLoRA-Align:{}+{}] Task {} done. Acc {:.2f}, final A-{} {:.3f} (wA={:.4f}), "
                "final B-{} {:.3f} (wB={:.4f})".format(
                    self.align_mode, self.align_b_mode, self._cur_task, train_acc,
                    self.align_mode, rega_run / n, w_a, self.align_b_mode, regb_run / n, w_b))
        else:
            logging.info("[SketchLoRA-Align:{}] Task {} done. Acc {:.2f}, final {} {:.3f}, align_weight {:.3f}".format(
                self.align_mode, self._cur_task, train_acc, self.align_mode, rega_run / n, w_a))

    def _train(self, train_loader):
        """Mirrors base _train, calling _train_core for the align term."""
        self._task_class_ranges[self._cur_task] = (self._known_classes, self._total_classes)
        self._freeze_inactive_blocks()

        self._train_core(train_loader)

        at_period_boundary = (self._cur_task + 1) % self.svd_period == 0
        at_last_task = (self._cur_task + 1) >= self._n_run_effective
        if at_period_boundary or at_last_task:
            self._maybe_extract_drift_exact()
            self._compress()
            self._maybe_extract_drift_sketch()
            self._maybe_save_adapter_snapshot()
            net = self._network.module if hasattr(self._network, "module") else self._network
            net.default_task = SKETCH
