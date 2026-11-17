"""Baseline per-task LoRA continual learner: freezes the ViT-B/16 backbone and
trains one LoRA pair per task on the query and value projections of every
block."""

import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.inc_net import LoRAVitNet
from models.base import BaseLearner
from models.stream_mixin import StreamMixin
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(StreamMixin, BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = LoRAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args.get("weight_decay") is not None else 5e-4
        self.epochs = args["tuned_epoch"]
        self.min_lr = args["min_lr"] if args.get("min_lr") is not None else 1e-8
        # train forward routing: per-task (False) vs accumulated sum (True)
        self.train_merge = bool(args.get("lora_train_merge", False))
        # LR schedule: cosine anneal (default) vs constant init_lr
        self.lr_anneal = bool(args.get("lr_anneal", True))

    def after_task(self):
        self._known_classes = self._total_classes

    # adapter selection, overridden by SeqLoRA to pin a single adapter
    def _train_adapter(self):
        """Which LoRA index to train/route on the current task."""
        return self._cur_task

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._train_adapter()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Grow one adapter slot if this task needs an index that isn't allocated
        # yet (construction only preallocates slot 0, see utils/inc_net.py).
        # No-op for SeqLoRA (_train_adapter() always returns 0) and for
        # sketchlora (bounded by its fixed lora_n_slots, never reaches it).
        if self._train_adapter() >= self._network.backbone.n_tasks:
            self._network.add_task_slot()

        # only the active LoRA + head are trainable
        self._network.freeze_to_task(self._train_adapter())
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

    # flat list of trainable parameters, uniform weight_decay for everything;
    # a subclass needing per-group weight_decay can return a list of
    # optim.AdamW-style param-group dicts instead
    def _optimizer_param_groups(self):
        return [p for p in self._network.parameters() if p.requires_grad]

    def _train(self, train_loader):
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._train_adapter(), merge=self.train_merge)["logits"]
                # task-local cross-entropy: only this task's class slice
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc))
        logging.info("Task {} finished. Train_accy {:.2f}".format(self._cur_task, train_acc))

    # deployed (CIL) forward, used by utils/metrics_logger.py's latency measurement
    def _deployed_forward(self, inputs):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=self._train_adapter(), merge=self.train_merge)

    # persistent state: every allocated adapter slot's bytes, regardless of
    # which are currently trainable, plus the classifier head
    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        total_params, total_bytes = 0, 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for m in mlist:
                    for p in m.parameters():
                        total_params += p.numel()
                        total_bytes += p.numel() * p.element_size()
        total_params += sum(p.numel() for p in net.fc.parameters())
        total_bytes += sum(p.numel() * p.element_size() for p in net.fc.parameters())
        return {"params": total_params, "bytes": total_bytes,
                "breakdown": {"lora_slots": total_params - sum(p.numel() for p in net.fc.parameters()),
                             "head": sum(p.numel() for p in net.fc.parameters())}}

    def _log_trainable(self):
        net = self._network
        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logging.info("Trainable params this task: {:,} / {:,}".format(trainable, total))
