"""SD-LoRA: per task, allocates a new LoRA pair plus one shared learnable
magnitude scalar; the forward sums every past task's Frobenius-normalized
contribution rescaled by its own scalar, plus the current task's raw term."""
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.lora import Learner as LoRALearner, num_workers
from utils.metrics_logger import log_bank_effective_rank


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        self.train_merge = True
        self._network.merge = True
        self.lora_rank = args.get("lora_rank", 10)
        # SD-LoRA-RR: optional [[task_threshold, rank], ...] schedule, sorted
        # ascending by threshold. None (default) = constant lora_rank every
        # task (base method, unmodified).
        self.rank_schedule = args.get("sdlora_rank_schedule", None)
        if self.rank_schedule is not None:
            self.rank_schedule = sorted(self.rank_schedule, key=lambda pair: pair[0])
        # opt-in summed-bank effective-rank diagnostic, off by default
        self.log_bank_rank = bool(args.get("log_bank_rank", False))
        self.bank_rank_energy_target = args.get("bank_rank_energy_target", 0.99)

    def _rank_for_task(self, cur_task):
        if not self.rank_schedule:
            return self.lora_rank
        rank = self.lora_rank
        for threshold, r in self.rank_schedule:
            if cur_task >= threshold:
                rank = r
        return rank

    def incremental_train(self, data_manager):
        """Same as models/lora.py's base incremental_train, except
        add_task_slot() passes this task's rank from the SD-LoRA-RR
        schedule."""
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._train_adapter()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        rank = self._rank_for_task(self._cur_task)
        if self._train_adapter() >= self._network.backbone.n_tasks:
            self._network.add_task_slot(rank=rank)
            if self.rank_schedule is not None:
                logging.info("[SD-LoRA-RR] task {}: new adapter slot allocated at rank={}".format(
                    self._cur_task, rank))

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

    def _train(self, train_loader):
        super()._train(train_loader)
        net = self._network.module if hasattr(self._network, "module") else self._network
        scales = [round(s.item(), 4) for s in net.backbone.task_scales]
        logging.info("[SD-LoRA] task {} done. task_scales so far (m_0..m_{}): {}".format(
            self._cur_task, self._cur_task, scales))
        self._log_bank_effective_rank()

    @torch.no_grad()
    def _log_bank_effective_rank(self):
        """Log the summed bank's effective rank via SVD, built from SD-LoRA's
        own normalized-and-scaled merge formula rather than a plain sum.
        No-op unless log_bank_rank is set."""
        if not self.log_bank_rank:
            return
        net = self._network.module if hasattr(self._network, "module") else self._network
        t = self._train_adapter()
        scales = net.backbone.task_scales
        deltas = {}
        for li, attn in enumerate(net.attn_modules()):
            for proj, (A_list, B_list) in (("q", (attn.lora_A_q, attn.lora_B_q)),
                                            ("v", (attn.lora_A_v, attn.lora_B_v))):
                delta = scales[t] * (B_list[t].weight @ A_list[t].weight)
                for s in range(t):
                    B_s, A_s = B_list[s].weight, A_list[s].weight
                    norm = torch.linalg.norm(B_s) * torch.linalg.norm(A_s)
                    delta = delta + scales[s] * (B_s @ A_s) / norm
                deltas[(li, proj)] = delta
        log_bank_effective_rank("SD-LoRA", self._cur_task, deltas, self.bank_rank_energy_target)

    # persistent state: extends models/lora.py's base direction-bank count
    # with the per-task magnitude scalars as their own breakdown key
    def persistent_state(self):
        base = super().persistent_state()
        net = self._network.module if hasattr(self._network, "module") else self._network
        scale_params = sum(p.numel() for p in net.backbone.task_scales)
        scale_bytes = sum(p.numel() * p.element_size() for p in net.backbone.task_scales)
        base["params"] += scale_params
        base["bytes"] += scale_bytes
        base["breakdown"]["scale_params"] = scale_params
        if self.rank_schedule is not None:
            attn0 = next(iter(net.attn_modules()))
            base["breakdown"]["slot_ranks"] = [m.out_features for m in attn0.lora_A_q]
        return base
