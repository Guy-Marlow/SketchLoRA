"""SD-LoRA + SketchLoRA: SD-LoRA's magnitude/direction decoupling applied to a
fixed-size, QR-SVD-folded sketch, with the sketch's singular values made
per-task re-learnable via a small rescaling vector."""

import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lora import Learner as LoRALearner, num_workers
from utils.toolkit import tensor2numpy

_PROJS = ("q", "v")


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # Forward always combines sketch + new adapter; no single-slot mode.
        self.train_merge = True
        self._network.merge = True
        self.svd_rank = args.get("svd_rank", args.get("lora_rank", 10))
        self.reg_strength = args.get("sdsketch_reg_strength", 0.0)
        self.reg_power = args.get("sdsketch_reg_power", 1.0)
        # when set, freezes the rescale vector at 0 forever, degrading to a
        # plain QR-SVD-folded sketch with no per-task rescale term
        self.freeze_rescale = bool(args.get("sdsketch_freeze_rescale", False))

    # single reused slot, forever
    def _train_adapter(self):
        return 0

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = 0
        logging.info("[SD-LoRA-Sketch] Learning on {}-{}".format(self._known_classes, self._total_classes))

        self._network.freeze_to_task(0)  # backbone's freeze_to_task ignores the task arg here
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._log_trainable()

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # fold this task's just-trained new adapter into the sketch now,
        # before eval_task() runs, unconditionally including task 0
        self._fold_sketch()

    def _rescale_reg_loss(self):
        if self._cur_task <= 0 or self.reg_strength == 0.0:
            return 0.0
        net = self._network.module if hasattr(self._network, "module") else self._network
        reg = 0.0
        for attn in net.attn_modules():
            for proj in _PROJS:
                S = getattr(attn, "sketch_S_{}".format(proj))
                rescale = getattr(attn, "rescale_{}".format(proj))
                energy = S.pow(2)
                share = energy / energy.sum().clamp_min(1e-12)
                reg = reg + (share.pow(self.reg_power) * rescale.pow(2)).sum()
        return reg

    def _train(self, train_loader):
        """Same training loop as models/lora.py's base _train, with the
        rescale regularizer term added to the loss."""
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, ce_run, reg_run, correct, total = 0.0, 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=0, merge=True)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)
                reg = self._rescale_reg_loss()
                loss = ce + self.reg_strength * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_run += ce.item()
                reg_run += float(reg)

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            n = len(train_loader)
            prog_bar.set_description(
                "[SD-LoRA-Sketch] Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, reg {:.5f}), Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs, losses / n, ce_run / n, reg_run / n, train_acc))
        logging.info("[SD-LoRA-Sketch] Task {} finished. Train_accy {:.2f}".format(self._cur_task, train_acc))
        self._log_memory_accounting()

    @torch.no_grad()
    def _fold_sketch(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        per_block = net.fold()
        vals = [v for blk in per_block.values() for v in blk.values()]
        logging.info("[SD-LoRA-Sketch] fold at task {}: retained_energy mean={:.4f} min={:.4f}".format(
            self._cur_task, sum(vals) / len(vals), min(vals)))

    # memory accounting: fixed-size sketch, the transient new adapter, its
    # rescale and magnitude parameters, and the head
    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        sketch_bytes = adapter_bytes = regmag_bytes = 0
        for attn in net.attn_modules():
            for proj in _PROJS:
                for name in ("sketch_U_{}".format(proj), "sketch_S_{}".format(proj), "sketch_Vh_{}".format(proj)):
                    t = getattr(attn, name)
                    sketch_bytes += t.numel() * t.element_size()
                for mod_name in ("lora_A_{}".format(proj), "lora_B_{}".format(proj)):
                    p = getattr(attn, mod_name)[0].weight
                    adapter_bytes += p.numel() * p.element_size()
                for name in ("rescale_{}".format(proj), "mag_{}".format(proj)):
                    p = getattr(attn, name)
                    regmag_bytes += p.numel() * p.element_size()
        head_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        total_bytes = sketch_bytes + adapter_bytes + regmag_bytes + head_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"sketch": sketch_bytes, "new_adapter": adapter_bytes,
                             "rescale_mag": regmag_bytes, "head": head_bytes}}

    def _log_memory_accounting(self):
        state = self.persistent_state()
        logging.info(
            "[SD-LoRA-Sketch] task {} persistent bank: fixed sketch (U+S+Vh, Q+V, every "
            "block) = {:,} bytes (sketch={:,}, new_adapter={:,}, rescale_mag={:,}, "
            "head={:,})".format(
                self._cur_task, state["bytes"], state["breakdown"]["sketch"],
                state["breakdown"]["new_adapter"], state["breakdown"]["rescale_mag"],
                state["breakdown"]["head"]))
