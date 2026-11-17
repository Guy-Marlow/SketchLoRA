"""Boundary-agnostic training driver for the LoRA continual-learning methods.

Provides the per-method hook contract (adapter slot selection, chunk begin/end,
deployed forward) plus ``offset_consolidation_run``, which decouples a method's
own adapter bookkeeping from the ground-truth task boundaries used by the
standard per-task loop in trainer.py, folding adapters at a fixed epoch-based
cadence and phase instead of at every task boundary.
"""

import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# DataLoader worker count for every streaming driver, matching models/lora.py's
# oracle-mode worker count.
num_workers = 8


class StreamMixin:
    # ---- per-method hooks (defaults = SeqLoRA-style single slot) ----
    def _stream_init(self):
        """Per-method state set up once before the first chunk."""
        pass

    def _stream_slot(self):
        """Backbone slot index to train/route during the current chunk."""
        return 0

    def _stream_train_merge(self):
        return bool(getattr(self, "train_merge", False))

    def _stream_begin_chunk(self, loader):
        """Start a new adapter chunk: (advance slot,) set trainability, fresh optimizer."""
        self._network.freeze_to_task(self._stream_slot(),
                                     train_a=getattr(self, "_stream_train_a", True))
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._stream_new_optimizer()

    def _stream_end_chunk(self, loader):
        """Adapter boundary action, such as fold or snapshot. Default: nothing."""
        pass

    def _stream_end_task(self, ct):
        """Fired once per class-task after its epochs finish, for bookkeeping
        that is task-scoped rather than chunk-scoped. Default: nothing."""
        pass

    def _stream_extra_loss(self, lo, hi):
        """Optional added loss (O-LoRA orthogonality). Default: 0."""
        return 0.0

    def _stream_cil_forward(self, inputs):
        """Deployed (CIL) forward: route current slot with the method's merge flag."""
        return self._network(inputs, task=self._stream_slot(), merge=self._stream_train_merge())

    # ---- shared machinery ----
    def _stream_new_optimizer(self):
        """Build a fresh optimizer and cosine schedule for a chunk, using
        self.epochs as the schedule's period."""
        params = [p for p in self._network.parameters() if p.requires_grad]
        self._stream_optim = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        self._stream_sched = optim.lr_scheduler.CosineAnnealingLR(
            self._stream_optim, T_max=self.epochs, eta_min=self.min_lr) \
            if getattr(self, "lr_anneal", True) else None

    def _stream_train_epoch(self, loader, lo, hi):
        """Train one epoch over the given loader with the current chunk's
        optimizer, using the task-local class slice [lo, hi) for cross-entropy."""
        self._network.train()
        slot, merge = self._stream_slot(), self._stream_train_merge()
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs, task=slot, merge=merge)["logits"]
            local_logits = logits[:, lo:hi]
            loss = F.cross_entropy(local_logits, targets - lo)
            extra = self._stream_extra_loss(lo, hi)
            if not (isinstance(extra, float) and extra == 0.0):
                loss = loss + extra
            self._stream_optim.zero_grad()
            loss.backward()
            self._stream_optim.step()
        if self._stream_sched is not None:
            self._stream_sched.step()

    # offset-consolidation: fold at a configurable epoch cadence and phase,
    # decoupled from task boundaries
    def offset_consolidation_run(self, data_manager, args):
        """Folds at a fixed epoch-slot cadence (offset_k, offset_phi) instead
        of once per task boundary; offset_k=epochs and offset_phi=0 reproduces
        per-task folding exactly. The optimizer and LR schedule reset only at
        a fold. Evaluation still happens at every task boundary, on whichever
        state is live, checked before any coincident fold."""
        k = int(args["offset_k"])
        phi = int(args.get("offset_phi", 0))
        assert k > 0, "offset_k must be positive"
        assert 0 <= phi < k, "offset_phi must be in [0, offset_k)"
        E = self.epochs
        nb_tasks = data_manager.nb_tasks
        _n_run = args.get("stop_after_tasks") or nb_tasks
        _n_run = min(_n_run, nb_tasks)
        if _n_run < nb_tasks:
            logging.info("[offset] stop_after_tasks={} (of {} total)".format(_n_run, nb_tasks))

        total_slots = _n_run * E
        merge_after = {s for s in range(total_slots) if (s + 1 - phi) % k == 0}
        merge_after.add(total_slots - 1)
        merge_sorted = sorted(merge_after)
        logging.info("[offset] k={} phi={} E={} total_slots={} merge points at slots {}".format(
            k, phi, E, total_slots, merge_sorted))

        self.data_manager = data_manager
        self._task_ranges = []
        known = 0
        for t in range(_n_run):
            task_size = data_manager.get_task_size(t)
            self._task_ranges.append((known, known + task_size))
            known += task_size

        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1
        self._stream_chunk = -1
        self._stream_init()

        results = []
        slot_opened = False
        cur_task_idx = -1
        loader, lo, hi = None, 0, 0

        for s in range(total_slots):
            t = s // E
            if t != cur_task_idx:
                cur_task_idx = t
                lo, hi = self._task_ranges[t]
                if hi > self._total_classes:
                    self._total_classes = hi
                    self._network.update_fc(self._total_classes)
                task_data, task_targets, _ = data_manager.get_dataset(
                    np.arange(lo, hi), source="train", mode="train", ret_data=True)
                train_set = data_manager.get_dataset(
                    [], source="train", mode="train", appendent=(task_data, task_targets))
                # persistent_workers=True: this loader is reused across however
                # many slots of this task fall in the current segment.
                loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=num_workers, persistent_workers=True)
                self._network.to(self._device)

            if not slot_opened:
                self._stream_begin_chunk(loader)
                # the new schedule's T_max is always self.epochs, not the segment
                # length, so a short segment is cut off before annealing finishes and a
                # long one sits at the floor LR for the remainder
                slot_opened = True

            self._stream_train_epoch(loader, lo, hi)

            task_boundary = (s + 1) % E == 0
            if task_boundary:
                results.append(self._stream_eval(t + 1))

            if s in merge_after:
                self._stream_end_chunk(loader)
                slot_opened = False
                if s < total_slots - 1:
                    self._stream_begin_chunk(loader)   # fresh flat-T_max=self.epochs schedule
                    slot_opened = True

            if task_boundary:
                self._stream_end_task(t)

        # forgetting: mean over past tasks of (that task's peak accuracy minus its
        # accuracy at the final checkpoint), matching trainer.py's oracle-mode formula
        if args.get("print_forget") and results:
            n_checkpoints = len(results)
            final_n_tasks = results[-1]["completed"]
            np_acctable = np.zeros([final_n_tasks, n_checkpoints])
            for col, r in enumerate(results):
                row = r["cil_per_task"]
                np_acctable[:len(row), col] = np.array(row)
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:final_n_tasks - 1])
            logging.info("[offset] Accuracy Matrix (CNN, tasks x checkpoints):\n{}".format(
                np.array2string(np_acctable, max_line_width=100000, precision=2, suppress_small=True)))
            logging.info("Forgetting (CNN): {}".format(forgetting))
            results[-1]["forgetting"] = float(forgetting)

        self._stream_results = results
        return results

    # ---- streaming evaluation: completed tasks only, folded adapter ----
    @torch.no_grad()
    def _stream_eval(self, completed):
        """CIL accuracy (argmax over completed classes) on the first `completed`
        class-tasks, using the method's deployed adapter."""
        self._network.eval()
        ranges = self._task_ranges[:completed]
        hi_total = ranges[-1][1]
        cil_correct = cil_n = 0
        # per-task CIL accuracy (global argmax restricted to each task's own test
        # images), needed for the forgetting computation
        cil_per_task = []
        for lo, hi in ranges:
            ds = self.data_manager.get_dataset(np.arange(lo, hi), source="test", mode="test")
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
            cil_t_corr, cil_t_n = 0, 0
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device); tnp = targets.numpy()
                cil_logits = self._stream_cil_forward(inputs)["logits"][:, :hi_total]
                cil_pred = cil_logits.argmax(1).cpu().numpy()
                cil_correct += int((cil_pred == tnp).sum()); cil_n += len(tnp)
                cil_t_corr += int((cil_pred == tnp).sum()); cil_t_n += len(tnp)
            cil_per_task.append(round(100.0 * cil_t_corr / max(cil_t_n, 1), 2))
        cil = round(100.0 * cil_correct / max(cil_n, 1), 2)
        logging.info("[stream eval] completed {:>2} | chunk {:>3} | CIL {:.2f}".format(
            completed, self._stream_slot(), cil))
        return {"completed": completed, "cil": cil, "cil_per_task": cil_per_task}
