"""Batch-incremental training with KL-drift-based adapter spawning
(boundary_mode="kl_batch_inc"), isolated from models/sketchlora.py and
models/sketchlora_align.py."""
import copy
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.base import BaseLearner
from backbone.linears import SimpleLinear
from utils.inc_net import BaseNet
from utils.online_detectors import KLDriftDetector
from utils.toolkit import tensor2numpy

num_workers = 8


class BatchIncNet(BaseNet):
    """Minimal net wrapper for backbone/vit_sketchlora_batchinc.py, whose
    forward takes no task/merge kwargs (it always includes sketch, pending
    and live adapters unconditionally), so only update_fc/generate_fc need
    overriding here."""

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)


class Learner(BaseLearner):
    """Task-major training where a KL-drift detector on the frozen backbone's
    features decides when to spawn a new live adapter, independently of a
    fixed-interval clock that folds pending adapters into the sketch."""

    def __init__(self, args):
        super().__init__(args)
        self._network = BatchIncNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay") or 0.0
        self.min_lr = args.get("min_lr") or 0.0
        self.epochs_per_task = args["tuned_epoch"]

        self.lora_rank = args.get("lora_rank", 10)
        self.svd_rank = args.get("svd_rank", self.lora_rank)
        self.pending_cap = args.get("batchinc_pending_cap", 4)
        self.sketch_interval_steps = args.get("batchinc_sketch_interval_steps", 200)
        self.orth_weight = args.get("batchinc_orth_weight", 0.5)
        # Warmup steps for a freshly spawned adapter's LR.
        self.lr_warmup_steps = args.get("batchinc_lr_warmup_steps", 20)

        kl_window_samples = args.get("batchinc_kl_window_samples", 256)
        kl_window_batches = max(4, -(-kl_window_samples // self.batch_size))  # ceil div
        self.kl_detector = KLDriftDetector(
            window_batches=kl_window_batches,
            warmup_multiplier=args.get("batchinc_kl_warmup_multiplier", 2),
            tau=args.get("batchinc_kl_tau", 4.0),
            consecutive=args.get("batchinc_kl_consecutive", 3),
            var_eps=args.get("batchinc_kl_var_eps", 1e-4),
            max_chunk_samples=args.get("batchinc_kl_max_chunk_samples"),
        )
        self.kl_window_batches = kl_window_batches
        # No detector; spawn follows the fixed-interval clock instead.
        self.fixed_interval_only = bool(args.get("batchinc_fixed_interval_only", False))

        self._optimizer = None
        self._global_step = 0
        self._adapter_step_counts = []
        self._cur_adapter_steps = 0

    def _net(self):
        return self._network.module if hasattr(self._network, "module") else self._network

    def _new_optimizer_and_schedule(self):
        """Rebuilt fresh at every spawn; lr set by hand per step."""
        net = self._net()
        params = [p for p in net.parameters() if p.requires_grad]
        self._optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        self._cur_adapter_steps = 0

    def _current_lr_scale(self):
        """Linear warmup ramp, then flat."""
        if self._cur_adapter_steps >= self.lr_warmup_steps:
            return 1.0
        return (self._cur_adapter_steps + 1) / self.lr_warmup_steps

    def _freeze_to_live(self):
        """Only the live slot and head are trainable."""
        net = self._net()
        for p in net.backbone.parameters():
            p.requires_grad = False
        live = net.backbone._live_idx
        for attn in net.backbone.attn_modules():
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[live].parameters():
                    p.requires_grad = True
        for p in net.fc.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _frozen_cls_features(self, inputs):
        net = self._net()
        net.backbone.set_adapters_disabled(True)
        feats = net.backbone(inputs)
        net.backbone.set_adapters_disabled(False)
        return feats

    def _maybe_spawn(self, inputs):
        """True iff a spawn happened this step."""
        if self.fixed_interval_only:
            due = (self._global_step > 0) and (self._global_step % self.sketch_interval_steps == 0)
            if due:
                self._do_spawn()
                return True
            return False
        feats = self._frozen_cls_features(inputs)
        fired = self.kl_detector.step(feats, inputs.shape[0])
        if fired:
            logging.info("[kl_batch_inc] KL detector fired at step {} (reason={}, kl={}, z={})".format(
                self._global_step, self.kl_detector.last_fire_reason,
                self.kl_detector.last_kl, self.kl_detector.last_z))
            self._do_spawn()
            self.kl_detector.reset()
        return fired

    def _do_spawn(self):
        net = self._net()
        self._adapter_step_counts.append(self._cur_adapter_steps)
        net.backbone.add_task_slot()
        self._freeze_to_live()
        self._new_optimizer_and_schedule()
        if len(net.backbone._pending) >= self.pending_cap:
            logging.info("[kl_batch_inc] pending cap ({}) reached immediately after spawn, "
                         "folding before any training on the new live slot".format(self.pending_cap))
            self._do_fold()

    def _maybe_fold(self):
        net = self._net()
        interval_due = (self._global_step > 0) and (self._global_step % self.sketch_interval_steps == 0)
        cap_due = len(net.backbone._pending) >= self.pending_cap
        if interval_due or cap_due:
            self._do_fold()

    def _do_fold(self):
        net = self._net()
        n_pending = len(net.backbone._pending)
        if n_pending == 0:
            return
        retained = net.backbone.fold_pending()
        vals = list(retained.values())
        logging.info("[kl_batch_inc] fold at step {}: {} pending slot(s) merged, "
                     "retained_energy mean={:.4f} min={:.4f}".format(
                         self._global_step, n_pending, sum(vals) / len(vals), min(vals)))

    @torch.no_grad()
    def _eval_cil(self, test_loader, known_classes, total_classes):
        net = self._net()
        net.eval()
        correct, total = 0, 0
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = net(inputs)["logits"]
            preds = logits.argmax(dim=1)
            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        return float(tensor2numpy(correct) * 100 / total) if total else 0.0

    def kl_batch_inc_run(self, data_manager, args):
        nb_tasks = data_manager.nb_tasks
        n_run = args.get("stop_after_tasks") or nb_tasks
        n_run = min(n_run, nb_tasks)
        if n_run < nb_tasks:
            logging.info("[kl_batch_inc] stop_after_tasks={} (of {} total)".format(n_run, nb_tasks))

        min_detectable = 3 * self.kl_window_batches

        self._known_classes = 0
        self._total_classes = 0
        self._global_step = 0
        self._adapter_step_counts = []
        self._cur_adapter_steps = 0
        self._net().to(self._device)

        results = []
        known = 0
        for t in range(n_run):
            task_size = data_manager.get_task_size(t)
            lo, hi = known, known + task_size
            known = hi
            if hi > self._total_classes:
                self._total_classes = hi
                self._net().update_fc(self._total_classes)
                self._net().to(self._device)
                # New fc as a param group, no rebuild.
                if self._optimizer is not None:
                    for p in self._net().fc.parameters():
                        p.requires_grad = True
                    self._optimizer.add_param_group(
                        {"params": list(self._net().fc.parameters()), "lr": self.init_lr})

            if t == 0:
                # One-time bootstrap; later spawns rebuild via _do_spawn().
                self._freeze_to_live()
                self._new_optimizer_and_schedule()

            train_set = data_manager.get_dataset(np.arange(lo, hi), source="train", mode="train")
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                      num_workers=num_workers)
            test_set = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
            test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                     num_workers=num_workers)

            task_steps = len(train_loader) * self.epochs_per_task
            if task_steps < min_detectable:
                logging.warning(
                    "[kl_batch_inc] task {} is only {} steps ({} batches x {} epochs), below the "
                    "detector's own ~{}-step minimum detectable segment (window_batches={}). The "
                    "detector may never fire within this task at this config; consider a smaller "
                    "batchinc_kl_window_samples".format(
                        t, task_steps, len(train_loader), self.epochs_per_task,
                        min_detectable, self.kl_window_batches))

            self._net().train()
            for epoch in range(self.epochs_per_task):
                losses, ce_run, orth_run, correct, total = 0.0, 0.0, 0.0, 0, 0
                for _, inputs, targets in train_loader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    self._maybe_spawn(inputs)

                    logits = self._net()(inputs)["logits"]
                    local_logits = logits[:, lo:hi]
                    ce = F.cross_entropy(local_logits, targets - lo)
                    orth = self._net().backbone.orth_loss()
                    loss = ce + self.orth_weight * orth

                    lr_scale = self._current_lr_scale()
                    for g in self._optimizer.param_groups:
                        g["lr"] = self.init_lr * lr_scale

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    self._global_step += 1
                    self._cur_adapter_steps += 1
                    self._maybe_fold()

                    losses += loss.item()
                    ce_run += ce.item()
                    orth_run += float(orth)
                    preds = local_logits.argmax(dim=1)
                    correct += preds.eq(targets - lo).cpu().sum()
                    total += len(targets)
                train_acc = tensor2numpy(correct) * 100 / total if total else 0.0
                logging.info(
                    "[kl_batch_inc] Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, orth {:.5f}), "
                    "Train_accy {:.2f}, global_step={}".format(
                        t, epoch + 1, self.epochs_per_task, losses / len(train_loader),
                        ce_run / len(train_loader), orth_run / len(train_loader), train_acc,
                        self._global_step))

            with torch.no_grad():
                self._replace_fc_head(train_loader=None)
            cil = self._eval_cil(test_loader, self._known_classes, self._total_classes)
            logging.info("[kl_batch_inc] Task {} boundary eval: CIL top1 = {:.2f} "
                         "(pending={}, live={})".format(
                             t, cil, list(self._net().backbone._pending), self._net().backbone._live_idx))
            results.append({"completed": t + 1, "cil": cil, "til": None,
                            "global_step": self._global_step,
                            "n_pending": len(self._net().backbone._pending),
                            "n_spawned": self._net().backbone.n_tasks})
            self._known_classes = self._total_classes

        self._adapter_step_counts.append(self._cur_adapter_steps)
        logging.info("[kl_batch_inc] FINAL: {} total steps, {} adapters spawned, "
                     "per-adapter step counts: {}".format(
                         self._global_step, self._net().backbone.n_tasks, self._adapter_step_counts))
        return results

    def _replace_fc_head(self, train_loader):
        """No-op: fc is gradient-trained, not a prototype head."""
        return
