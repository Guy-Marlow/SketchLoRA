"""RainbowPrompt: a per-task evolved prompt trained with task-local
cross-entropy plus a pull constraint on task-key/CLS-feature similarity,
retrieved at CIL time via nearest stored task-key match."""

import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.inc_net import RainbowPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = RainbowPromptVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay") or 5e-4
        self.epochs = args["tuned_epoch"]
        self.min_lr = args.get("min_lr") or 1e-8
        self.pull_constraint_coeff = args.get("pull_constraint_coeff", 0.1)
        self.clip_grad = args.get("clip_grad", 1.0)
        # Load-bearing warmup; init_lr from step 1 diverges.
        self.warmup_lr = args.get("warmup_lr", 1e-6)
        self.warmup_epochs = args.get("warmup_epochs", 5)
        self._global_step = 0
        self._warmup_steps = None   # set on first _train call from that loader's length
        self.data_manager = None

    def after_task(self):
        self._known_classes = self._total_classes

    def _rp_slot(self):
        """Which row to train/route into."""
        return self._cur_task

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._rp_slot()
        logging.info("[RainbowPrompt] Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._rp_slot() >= self._network.backbone.prompt_module.n_tasks:
            self._network.backbone.prompt_module.add_task_slot()

        for p in self._network.parameters():
            p.requires_grad = False
        for p in self._network.fc.parameters():
            p.requires_grad = True
        # Whole tensor trainable; only current row gets gradient.
        for bk in self._network.backbone.prompt_module.base_knowledge:
            bk.requires_grad_(True)
        self._network.backbone.prompt_module.base_key.requires_grad_(True)
        pm = self._network.backbone.prompt_module
        if pm.use_linear:
            for module_list in (pm.query_matcher, pm.key_matcher, pm.value_matcher,
                                 pm.dense, pm.fc1, pm.fc2):
                for p in module_list.parameters():
                    p.requires_grad_(True)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        self._network.to(self._device)
        self._train(self.train_loader)

    def _train(self, train_loader):
        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.warmup_lr, weight_decay=self.weight_decay)
        if self._warmup_steps is None:
            self._warmup_steps = self.warmup_epochs * len(train_loader)

        lo, hi = self._ce_slice()
        t = self._rp_slot()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                lr = self.warmup_lr + (self.init_lr - self.warmup_lr) * min(
                    self._global_step, self._warmup_steps) / self._warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr
                self._global_step += 1

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs, task_id=t, train=True)
                local_logits = out["logits"][:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)
                loss = loss - self.pull_constraint_coeff * out["sim_loss"]

                optimizer.zero_grad()
                loss.backward()
                # Clipping is load-bearing; omitting it diverges.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self._network.parameters() if p.requires_grad], self.clip_grad)
                optimizer.step()
                losses += loss.item()

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "[RainbowPrompt] Task {}, Epoch {}/{} => Loss {:.3f}, Acc {:.2f}".format(
                    t, epoch + 1, self.epochs, losses / len(train_loader), train_acc))
        logging.info("[RainbowPrompt] Task {} done. Acc {:.2f}".format(t, train_acc))

    @torch.no_grad()
    def _eval_cnn(self, loader):
        """Deployed eval: predicts its own task id, not a forced one."""
        self._network.eval()
        y_pred, y_true = [], []
        k = min(self.topk, self._total_classes)
        slot = self._rp_slot()
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            logits = self._network(inputs, task_id=slot, train=False)["logits"]
            predicts = torch.topk(logits, k=k, dim=1, largest=True, sorted=True)[1]
            if k < self.topk:
                pad = predicts[:, -1:].expand(-1, self.topk - k)
                predicts = torch.cat([predicts, pad], dim=1)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def persistent_state(self):
        net = self._network
        pm = net.backbone.prompt_module
        bk_bytes = sum(bk.numel() * bk.element_size() for bk in pm.base_knowledge)
        key_bytes = pm.base_key.numel() * pm.base_key.element_size()
        stored_bytes = pm.stored_prompts.numel() * pm.stored_prompts.element_size()
        head_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        evolve_bytes = 0
        if pm.use_linear:
            for module_list in (pm.query_matcher, pm.key_matcher, pm.value_matcher,
                                 pm.dense, pm.fc1, pm.fc2):
                evolve_bytes += sum(p.numel() * p.element_size() for p in module_list.parameters())
        total_bytes = bk_bytes + key_bytes + stored_bytes + head_bytes + evolve_bytes
        total_params = total_bytes // 4
        return {"params": int(total_params), "bytes": int(total_bytes),
                "breakdown": {"base_knowledge": bk_bytes, "base_key": key_bytes,
                             "stored_prompts": stored_bytes, "head": head_bytes,
                             "evolve_sublayers": evolve_bytes}}

    @torch.no_grad()
    def _deployed_forward(self, inputs):
        """Deployed CIL forward, for inference-cost measurement."""
        return self._network(inputs, task_id=self._rp_slot(), train=False)
