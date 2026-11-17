"""SketchLoRA: a single fixed-rank sketch adapter that replaces a growing
per-task LoRA bank by periodically folding a trainable residual adapter into
a frozen, SVD-compressed sketch. Memory stays bounded at the sketch rank no
matter how many tasks arrive.
"""

import json
import logging
import math
import os

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lora import Learner as LoRALearner, num_workers
from utils.toolkit import tensor2numpy

from utils.merge_algs import (rand_svd, rand_svd_probe, factors_from_probe,
                               random_factors_from_probe, countsketch_compress,
                               freqdir_input_factor, freqdir_output_factor, fd_shrink_probe)
from utils.metrics_logger import save_adapter_snapshot

# slot 0 = sketch, slot 1 = residual
SKETCH = 0
RESIDUAL = 1


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        self.lora_rank = args.get("lora_rank", 10)
        self.svd_rank = args.get("svd_rank", self.lora_rank)
        self.oversampling = args.get("svd_oversampling", 10)
        self.n_lora_blocks = args.get("n_lora_blocks", None)
        # P>1: folds P residual slots together.
        self.svd_period = args.get("svd_period", 1)
        assert self.svd_period >= 1
        # Delays the fold past the boundary, training a lead-in chunk into
        # the previous task's still-open residual first.
        self.oracle_fold_delay_epochs = args.get("oracle_fold_delay_epochs", 0)
        assert self.oracle_fold_delay_epochs >= 0
        if self.oracle_fold_delay_epochs > 0:
            assert self.svd_period == 1, \
                "oracle_fold_delay_epochs doesn't combine with svd_period>1 since that mechanism " \
                "already gives each task its own separate residual slot jointly compressed at " \
                "the period boundary, a different mechanism from continuously training one " \
                "shared residual across a task boundary. Got svd_period={}".format(self.svd_period)
        self._fold_delay_pending = False
        self._fold_delay_cached_eval = None
        # Merge algorithm at each fold boundary.
        self.merge_op = args.get("merge_op", "randsvd")
        assert self.merge_op in ("randsvd", "exactsvd", "countsketch",
                                  "freqdir", "freqdir_dense", "qrsvd")
        # Keeps a random subset of singular directions instead of the top-energy ones.
        self.rank_selection = args.get("sketchlora_rank_selection", "top")
        assert self.rank_selection in ("top", "random")
        if self.rank_selection == "random":
            assert self.merge_op == "randsvd", \
                "sketchlora_rank_selection='random' only applies to merge_op='randsvd'."
        if self.rank_selection == "random":
            assert args.get("sketch_diag", False), \
                "sketchlora_rank_selection='random' requires sketch_diag=True. It needs " \
                "the randomized probe's full spectrum (U_probe/S/Vh_probe), which is only " \
                "computed when need_svdvals is True; the raw rand_svd() fallback path has " \
                "no comparable pool of directions to sample from."
        # "period" aliases svd_period under the streaming driver.
        raw_lazy = args.get("lazy_merge", "off")
        if isinstance(raw_lazy, bool):
            self.lazy_merge_mode = "off"
            assert not raw_lazy, "the legacy saturation-based lazy_merge=True mode has been removed; use lazy_merge='period' with lazy_merge_period instead"
        else:
            assert raw_lazy in ("off", "period"), \
                "lazy_merge must be 'off' or 'period'"
            self.lazy_merge_mode = raw_lazy
        self.lazy_merge = self.lazy_merge_mode != "off"
        self.lazy_merge_period = args.get("lazy_merge_period", None)
        if self.lazy_merge_mode == "period":
            assert self.lazy_merge_period is not None and self.lazy_merge_period >= 1, \
                "lazy_merge=period requires lazy_merge_period >= 1"
            configured_svd_period = args.get("svd_period", 1)
            if configured_svd_period != 1 and configured_svd_period != self.lazy_merge_period:
                raise ValueError(
                    "svd_period and lazy_merge_period are the same underlying knob, "
                    "set only one, or set both to the same value")
            self.svd_period = self.lazy_merge_period
        # Merges factors directly, skipping the dense delta_W.
        if self.merge_op == "freqdir":
            assert self.svd_period == 1, \
                "merge_op='freqdir' requires exactly one residual slot per boundary"
        # Shrinks the dense probe instead of factoring it directly.
        if self.merge_op == "freqdir_dense":
            assert self.svd_period == 1, \
                "merge_op='freqdir_dense' requires exactly one residual slot per boundary"
        self.lora_weight_decay = args.get("sketchlora_lora_wd", None)
        # Per-boundary test-set embeddings, for offline drift analysis.
        self.embed_drift_dir = args.get("embed_drift_dir", None)
        if self.embed_drift_dir is not None:
            os.makedirs(self.embed_drift_dir, exist_ok=True)
        self.adapter_snapshot_dir = args.get("adapter_snapshot_dir", None)
        if self.adapter_snapshot_dir is not None:
            os.makedirs(self.adapter_snapshot_dir, exist_ok=True)
        self._task_class_ranges = {}
        self.cs_rank = args.get("cs_rank", self.svd_rank)
        self.cs_seed = args.get("cs_seed", args["seed"] if not isinstance(args.get("seed"), list)
                                 else args["seed"][0])
        self.rank_selection_seed = args.get(
            "sketchlora_rank_selection_seed",
            args["seed"] if not isinstance(args.get("seed"), list) else args["seed"][0])
        # Forces a final compress on the last task.
        self._n_run_effective = min(args.get("stop_after_tasks") or args["nb_tasks"], args["nb_tasks"])
        self.train_merge = True
        self._network.merge = True
        for attn in self._all_attns():
            for A_list in (attn.lora_A_q, attn.lora_A_v):
                nn.init.zeros_(A_list[SKETCH].weight)
        self._sketch_populated = False
        self.sketch_diag = bool(args.get("sketch_diag", True))
        if self.merge_op == "freqdir" and self.sketch_diag:
            logging.warning(
                "[SketchLoRA] merge_op='freqdir' auto-disabling sketch_diag.")
            self.sketch_diag = False
        self._diag_records = []
        seed = args["seed"] if not isinstance(args.get("seed"), list) else args["seed"][0]
        split = "ic{}i{}".format(args.get("init_cls"), args.get("increment"))
        diag_dir = args.get("sketchlora_diag_dir")
        if diag_dir is not None:
            prefix = args.get("prefix") or "sketchlora"
            self._diag_path = os.path.join(diag_dir, "sketchlora_diag_{}_seed{}.json".format(prefix, seed))
        else:
            dataset_tag = args.get("dataset", "unknown")
            tag = "{}_r{}_b{}_{}".format(dataset_tag, self.svd_rank, self.n_lora_blocks or "all", split)
            self._diag_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "run_logs", "sketchlora_diag_{}_seed{}.json".format(tag, seed))

    # Attention blocks carrying LoRA (all, or first n).
    def _all_attns(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net.attn_modules()

    def _active_attns(self):
        attns = self._all_attns()
        return attns if self.n_lora_blocks is None else attns[:self.n_lora_blocks]

    def _residual_slots(self):
        """Slot indices 1..P."""
        return list(range(RESIDUAL, RESIDUAL + self.svd_period))

    def _optimizer_param_groups(self):
        """Separate weight_decay for LoRA params when sketchlora_lora_wd is set."""
        if self.lora_weight_decay is None:
            return super()._optimizer_param_groups()
        lora_params, other_params = [], []
        for name, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            (lora_params if ("lora_A" in name or "lora_B" in name) else other_params).append(p)
        groups = [{"params": lora_params, "weight_decay": self.lora_weight_decay}]
        if other_params:
            groups.append({"params": other_params, "weight_decay": self.weight_decay})
        return groups

    def _freeze_inactive_blocks(self):
        """Freezes residual slots on blocks beyond n_lora_blocks."""
        if self.n_lora_blocks is None:
            return
        for attn in self._all_attns()[self.n_lora_blocks:]:
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for slot in self._residual_slots():
                    for p in mlist[slot].parameters():
                        p.requires_grad = False

    def _stream_init(self):
        self._cur_task = -1

    def _stream_slot(self):
        return RESIDUAL + (self._cur_task % self.svd_period)

    def _stream_begin_chunk(self, loader):
        self._cur_task += 1
        self._freeze_inactive_blocks()
        super()._stream_begin_chunk(loader)

    def _stream_end_chunk(self, loader):
        self._freeze_inactive_blocks()
        at_period_boundary = (self._cur_task + 1) % self.svd_period == 0
        if at_period_boundary:
            self._compress()

    def _train_adapter(self):
        return RESIDUAL + (self._cur_task % self.svd_period)

    def _deployed_forward(self, inputs):
        """Deployed forward: sketch only."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=SKETCH, merge=True)

    def _drift_test_loader(self, task_idx):
        known, total = self._task_class_ranges[task_idx]
        dataset = self.data_manager.get_dataset(np.arange(known, total), source="test", mode="test")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    @torch.no_grad()
    def _extract_features(self, loader, forward_fn):
        """Returns the extracted features and their labels, in loader order."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.eval()
        feats, labels = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            out = forward_fn(net, inputs)
            feats.append(out["features"].cpu().numpy())
            labels.append(targets.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    def _save_drift_group(self, fname, features, labels, **meta):
        path = os.path.join(self.embed_drift_dir, fname)
        np.savez(path, features=features, labels=labels, **meta)
        logging.info("[embed_drift] wrote {} ({} points)".format(path, features.shape[0]))

    def _maybe_extract_drift_exact(self):
        """This task's embeddings before compression."""
        if self.embed_drift_dir is None:
            return
        n = self._cur_task
        loader = self._drift_test_loader(n)
        feats, labels = self._extract_features(
            loader, lambda net, x: net(x, task=self._train_adapter(), merge=self.train_merge))
        self._save_drift_group("exact_task{}.npz".format(n), feats, labels,
                                task=n, boundary=n, kind="exact")

    def _maybe_extract_drift_sketch(self):
        """Every past task's embeddings after compression."""
        if self.embed_drift_dir is None:
            return
        n = self._cur_task
        for r in range(n + 1):
            loader = self._drift_test_loader(r)
            feats, labels = self._extract_features(
                loader, lambda net, x: net(x, task=SKETCH, merge=True))
            self._save_drift_group("sketch_task{}_boundary{}.npz".format(r, n), feats, labels,
                                    task=r, boundary=n, kind="sketch")

    def _maybe_save_adapter_snapshot(self):
        """Persists the sketch slot's (B, A) per module after _compress()."""
        if self.adapter_snapshot_dir is None:
            return
        n = self._cur_task
        layer_proj_to_BA = {}
        for li, attn in enumerate(self._active_attns()):
            for proj, (A_list, B_list) in (("q", (attn.lora_A_q, attn.lora_B_q)),
                                            ("v", (attn.lora_A_v, attn.lora_B_v))):
                layer_proj_to_BA[(li, proj)] = (B_list[SKETCH].weight, A_list[SKETCH].weight)
        save_adapter_snapshot(self.adapter_snapshot_dir, "sketch_boundary{}.pt".format(n),
                               layer_proj_to_BA, boundary=n, merge_op=self.merge_op)

    def _train_n_epochs(self, train_loader, n_epochs, lo, hi, epoch_label_offset=0):
        """Runs n_epochs of local-window CE training with a fresh optimizer."""
        if n_epochs <= 0:
            return
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=self.min_lr) if self.lr_anneal else None

        prog_bar = tqdm(range(n_epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._train_adapter(), merge=self.train_merge)["logits"]
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
                    self._cur_task, epoch + 1 + epoch_label_offset, self.epochs,
                    losses / len(train_loader), train_acc))
        logging.info("Task {} ({} epochs this chunk). Train_accy {:.2f}".format(
            self._cur_task, n_epochs, train_acc))

    def _fold_delay_eval_now(self):
        """Evaluates now and caches the result, so eval_task() never reports
        a task that finished training but hasn't been folded yet."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.default_task = SKETCH
        self._fold_delay_cached_eval = super().eval_task()

    def eval_task(self):
        """Returns the cached fold-completion eval under fold delay."""
        if self.oracle_fold_delay_epochs > 0 and self._fold_delay_cached_eval is not None:
            return self._fold_delay_cached_eval
        return super().eval_task()

    def _train_with_fold_delay(self, train_loader):
        """Trains a lead-in chunk into the previous task's still-open
        residual, folds, then trains this task's own remainder on the fresh
        residual."""
        lo, hi = self._ce_slice()
        delay = min(self.oracle_fold_delay_epochs, self.epochs)
        is_last_task = (self._cur_task + 1) >= self._n_run_effective

        if self._fold_delay_pending:
            logging.info("[fold_delay] task {}: lead-in {} epoch(s) on the still-open residual "
                         "from task {}".format(self._cur_task, delay, self._cur_task - 1))
            self._train_n_epochs(train_loader, delay, lo, hi, epoch_label_offset=0)
            logging.info("[fold_delay] task {}: firing deferred fold from task {}".format(
                self._cur_task, self._cur_task - 1))
            self._compress()
            self._fold_delay_pending = False
            self._fold_delay_eval_now()
            remaining = self.epochs - delay
            remaining_offset = delay
        else:
            remaining = self.epochs
            remaining_offset = 0

        if remaining > 0:
            logging.info("[fold_delay] task {}: training {} own epoch(s) on the {} residual".format(
                self._cur_task, remaining, "fresh" if remaining_offset else "freshly-initialized"))
            self._train_n_epochs(train_loader, remaining, lo, hi, epoch_label_offset=remaining_offset)

        if is_last_task:
            logging.info("[fold_delay] task {} (last task): forcing final fold, no defer".format(
                self._cur_task))
            self._compress()
            self._fold_delay_pending = False
            self._fold_delay_eval_now()
        else:
            self._fold_delay_pending = True

        net = self._network.module if hasattr(self._network, "module") else self._network
        net.default_task = SKETCH if is_last_task else RESIDUAL

    def _train(self, train_loader):
        if self.oracle_fold_delay_epochs > 0:
            self._task_class_ranges[self._cur_task] = (self._known_classes, self._total_classes)
            self._freeze_inactive_blocks()
            self._train_with_fold_delay(train_loader)
            return
        self._task_class_ranges[self._cur_task] = (self._known_classes, self._total_classes)
        self._freeze_inactive_blocks()
        super()._train(train_loader)

        at_period_boundary = (self._cur_task + 1) % self.svd_period == 0
        at_last_task = (self._cur_task + 1) >= self._n_run_effective
        if at_period_boundary or at_last_task:
            self._maybe_extract_drift_exact()
            self._compress()
            self._maybe_extract_drift_sketch()
            self._maybe_save_adapter_snapshot()
            # Routes eval to the sketch (the adapter used at inference time).
            net = self._network.module if hasattr(self._network, "module") else self._network
            net.default_task = SKETCH

    @torch.no_grad()
    def _compress(self):
        """Folds (sketch + residuals) into the sketch, per layer/projection.
        merge_op selects the algorithm: randsvd (randomized SVD),
        exactsvd (dense exact SVD), qrsvd (exact SVD via QR factoring,
        avoiding the dense product), countsketch (CountSketch), freqdir
        (frequent-directions on the factors, skipping the dense delta_W),
        freqdir_dense (frequent-directions shrinkage on the dense probe).
        Always truncates to the fixed svd_rank."""
        retained, sigma_next, fro, rhat = [], [], [], []
        fd_dense_stats = []
        residual_slots = self._residual_slots()
        skip_compression = (not self._sketch_populated) and len(residual_slots) == 1
        full_svd_needed = (not skip_compression) and self.merge_op in ("exactsvd", "qrsvd")
        need_svdvals = (not skip_compression) and (not full_svd_needed) and \
            self.merge_op != "freqdir" and self.sketch_diag
        module_idx = 0
        for attn in self._active_attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                   (attn.lora_A_v, attn.lora_B_v)):
                A_s, B_s = A_list[SKETCH].weight, B_list[SKETCH].weight
                dev, dt = B_s.device, B_s.dtype
                if self.merge_op == "freqdir":
                    delta_W = None
                else:
                    delta_W = B_s @ A_s
                    for slot in residual_slots:
                        A_r, B_r = A_list[slot].weight, B_list[slot].weight
                        delta_W = delta_W + B_r @ A_r

                if skip_compression:
                    # Task 0: copy the residual, no SVD needed.
                    only_slot = residual_slots[0]
                    B_hat = B_list[only_slot].weight.clone()
                    A_hat = A_list[only_slot].weight.clone()
                    final_rank = B_hat.shape[1]
                    if self.sketch_diag:
                        fro_delta = delta_W.float().norm()
                        retained.append(1.0)
                        sigma_next.append(0.0)
                        fro.append(fro_delta.item())
                        rhat.append(final_rank)
                    B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)
                    if final_rank == B_s.shape[1]:
                        B_s.data.copy_(B_hat)
                        A_s.data.copy_(A_hat)
                    else:
                        newA = nn.Linear(delta_W.shape[0], final_rank, bias=False).to(dev, dt)
                        newB = nn.Linear(final_rank, delta_W.shape[0], bias=False).to(dev, dt)
                        newA.weight.data.copy_(A_hat)
                        newB.weight.data.copy_(B_hat)
                        for p in list(newA.parameters()) + list(newB.parameters()):
                            p.requires_grad = False
                        A_list[SKETCH] = newA
                        B_list[SKETCH] = newB
                    for slot in residual_slots:
                        nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                        nn.init.zeros_(B_list[slot].weight)
                    module_idx += 1
                    continue

                if self.merge_op == "freqdir_dense":
                    prev_rank = A_s.shape[0] if self._sketch_populated else 0
                    residual_total = sum(A_list[slot].weight.shape[0] for slot in residual_slots)
                    composite_rank = prev_rank + residual_total
                    working_rank = max(composite_rank, self.svd_rank)
                    U_probe, S_probe, Vh_probe = rand_svd_probe(delta_W, working_rank, self.oversampling)
                    S_real = S_probe[:composite_rank]
                    S, fd_stats = fd_shrink_probe(S_real)
                    final_rank = min(self.svd_rank, S.numel())
                    root_S = S[:final_rank].sqrt()
                    B_hat = (U_probe[:, :final_rank].to(dt) * root_S.to(dt).unsqueeze(0))
                    A_hat = (root_S.to(dt).unsqueeze(1) * Vh_probe[:final_rank, :].to(dt))
                    fd_dense_stats.append(fd_stats)
                    B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)

                    if self.sketch_diag:
                        recon_err = (delta_W.float() - (B_hat.float() @ A_hat.float())).norm()
                        fro_delta = delta_W.float().norm()
                        retained.append(1.0 - (recon_err / fro_delta).item() ** 2 if fro_delta > 0 else 1.0)
                        sigma_next.append(S[final_rank].item() if S.numel() > final_rank else 0.0)
                        fro.append(fro_delta.item())
                        rhat.append(final_rank)

                    if final_rank == B_s.shape[1]:
                        B_s.data.copy_(B_hat)
                        A_s.data.copy_(A_hat)
                    else:
                        newA = nn.Linear(delta_W.shape[0], final_rank, bias=False).to(dev, dt)
                        newB = nn.Linear(final_rank, delta_W.shape[0], bias=False).to(dev, dt)
                        newA.weight.data.copy_(A_hat)
                        newB.weight.data.copy_(B_hat)
                        for p in list(newA.parameters()) + list(newB.parameters()):
                            p.requires_grad = False
                        A_list[SKETCH] = newA
                        B_list[SKETCH] = newB
                    for slot in residual_slots:
                        nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                        nn.init.zeros_(B_list[slot].weight)
                    module_idx += 1
                    continue

                S = None
                U_probe = Vh_probe = None
                prev_rank = A_s.shape[0] if self._sketch_populated else 0
                residual_total = sum(A_list[slot].weight.shape[0] for slot in residual_slots)
                composite_rank = prev_rank + residual_total
                if self.merge_op == "qrsvd" and full_svd_needed:
                    # Exact SVD via QR, not dense.
                    B_cat = torch.cat([B_s] + [B_list[slot].weight for slot in residual_slots],
                                      dim=1).float()
                    A_cat = torch.cat([A_s] + [A_list[slot].weight for slot in residual_slots],
                                      dim=0).float()
                    Q_B, R_B = torch.linalg.qr(B_cat)
                    Q_A, R_A = torch.linalg.qr(A_cat.t())
                    M_small = R_B @ R_A.t()
                    U_small, S, Vh_small = torch.linalg.svd(M_small)
                    U = Q_B @ U_small
                    Vh = Vh_small @ Q_A.t()
                elif full_svd_needed:
                    U, S, Vh = torch.linalg.svd(delta_W.float())
                elif self.merge_op == "randsvd" and need_svdvals:
                    _working_rank = max(composite_rank, self.svd_rank)
                    U_probe, S, Vh_probe = rand_svd_probe(delta_W, _working_rank, self.oversampling)
                elif need_svdvals:
                    S = torch.linalg.svdvals(delta_W.float())

                # target rank for this compression: fixed at svd_rank (cs_rank
                # for countsketch, which has its own target-rank knob)
                if self.merge_op == "countsketch":
                    r_hat_t = self.cs_rank
                else:
                    r_hat_t = self.svd_rank

                # merges (B_hat, A_hat) per the chosen algorithm
                if self.merge_op == "countsketch":
                    B_ws = [B_s] + [B_list[slot].weight for slot in residual_slots]
                    A_ws = [A_s] + [A_list[slot].weight for slot in residual_slots]
                    seed = (int(self.cs_seed) * 1_000_003 + (self._cur_task + 1) * 9176
                            + module_idx) % (2 ** 63 - 1)
                    B_hat, A_hat = countsketch_compress(B_ws, A_ws, r_hat_t, seed)
                elif self.merge_op == "freqdir":
                    A_r = A_list[residual_slots[0]].weight
                    B_r = B_list[residual_slots[0]].weight
                    M_A = torch.cat([A_s, A_r], dim=0).float()
                    M_B = torch.cat([B_s, B_r], dim=1).float()
                    A_hat = freqdir_input_factor(M_A, r_hat_t)
                    B_hat = freqdir_output_factor(M_B, r_hat_t)
                elif full_svd_needed:
                    root_S = S[:r_hat_t].sqrt()
                    B_hat = U[:, :r_hat_t].to(dt) * root_S.to(dt).unsqueeze(0)
                    A_hat = root_S.to(dt).unsqueeze(1) * Vh[:r_hat_t, :].to(dt)
                else:
                    # Reuses the probe above, if computed.
                    if U_probe is not None and self.rank_selection == "random":
                        # Keeps a random subset of singular directions instead of the top ones.
                        _rs_seed = (int(self.rank_selection_seed) * 1_000_003
                                    + (self._cur_task + 1) * 9176 + module_idx) % (2 ** 63 - 1)
                        B_hat, A_hat = random_factors_from_probe(
                            U_probe, S, Vh_probe, r_hat_t, composite_rank, _rs_seed)
                    elif U_probe is not None:
                        B_hat, A_hat = factors_from_probe(U_probe, S, Vh_probe, r_hat_t)
                    else:
                        B_hat, A_hat = rand_svd(delta_W, r_hat_t, self.oversampling)
                B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)
                final_rank = B_hat.shape[1]

                if self.sketch_diag:
                    recon_err = (delta_W.float() - (B_hat.float() @ A_hat.float())).norm()
                    fro_delta = delta_W.float().norm()
                    retained.append(1.0 - (recon_err / fro_delta).item() ** 2 if fro_delta > 0 else 1.0)
                    sigma_next.append(S[final_rank].item() if (S is not None and S.numel() > final_rank) else 0.0)
                    fro.append(fro_delta.item())
                    rhat.append(final_rank)

                if final_rank == B_s.shape[1]:
                    B_s.data.copy_(B_hat)
                    A_s.data.copy_(A_hat)
                else:
                    dim = B_s.shape[0]
                    newA = nn.Linear(dim, final_rank, bias=False).to(dev, dt)
                    newB = nn.Linear(final_rank, dim, bias=False).to(dev, dt)
                    newA.weight.data.copy_(A_hat)
                    newB.weight.data.copy_(B_hat)
                    for p in list(newA.parameters()) + list(newB.parameters()):
                        p.requires_grad = False
                    A_list[SKETCH] = newA
                    B_list[SKETCH] = newB
                # All P residual slots reset, not just one.
                for slot in residual_slots:
                    nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                    nn.init.zeros_(B_list[slot].weight)
                module_idx += 1
        self._sketch_populated = True
        if self.sketch_diag:
            self._record_diag(retained, sigma_next, fro, rhat, fd_dense_stats)

    def _record_diag(self, retained, sigma_next, fro, rhat, fd_dense_stats=None):
        """Aggregates and persists the per-compression singular-spectrum stats."""
        import numpy as np
        rec = {
            "task": self._cur_task,
            "retained_energy": retained,
            "sigma_next": sigma_next,
            "fro": fro,
            "r_hat": rhat,
            "retained_mean": float(np.mean(retained)),
            "retained_min": float(np.min(retained)),
            "fro_mean": float(np.mean(fro)),
            "r_hat_mean": float(np.mean(rhat)) if rhat else None,
            "r_hat_max": int(np.max(rhat)) if rhat else None,
            "r_hat_total": int(np.sum(rhat)) if rhat else None,
        }
        if fd_dense_stats:
            rec["freqdir_dense_delta"] = [r["delta"] for r in fd_dense_stats]
            rec["freqdir_dense_pre_shrink_energy"] = [r["pre_shrink_energy"] for r in fd_dense_stats]
            rec["freqdir_dense_post_shrink_energy"] = [r["post_shrink_energy"] for r in fd_dense_stats]
        self._diag_records.append(rec)
        os.makedirs(os.path.dirname(self._diag_path), exist_ok=True)
        with open(self._diag_path, "w") as f:
            json.dump(self._diag_records, f, indent=2)
        rh = (" | r_hat mean={:.1f} total={}".format(rec["r_hat_mean"], rec["r_hat_total"])
              if rec["r_hat_mean"] is not None else "")
        logging.info(
            "[SketchDiag] task {}: retained-energy mean={:.3f} min={:.3f} | "
            "||DeltaW||_F mean={:.3f}{}".format(
                self._cur_task, rec["retained_mean"], rec["retained_min"], rec["fro_mean"], rh))
