"""InfLoRA: per task, sets the new LoRA down-projection A analytically from an
SVD of the input covariance, projected to remove the subspace already spanned
by past tasks (DualGPM), then trains only the up-projection B. Task updates are
folded directly into q_proj/v_proj's own weight after training."""

import logging
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lora import Learner as LoRALearner

num_workers = 8


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # InfLoRA folds directly into q_proj/v_proj's own weight rather than a
        # separate frozen_delta buffer, since those are pre-existing tensors
        # every backbone already allocates
        self.rank = args.get("lora_rank", 10)
        self.lamb = args.get("lamb", 0.95)        # DualGPM threshold lower bound
        self.lame = args.get("lame", 1.0)         # DualGPM threshold upper bound
        self.total_sessions = args["nb_tasks"]
        # q_proj/v_proj already carry every past task's folded contribution, so
        # forward only needs to add the current task's own unfolded term
        self.train_merge = False
        self._network.merge = False               # read by base.py::_eval_cnn's bare forward
        self.feature_list = []                    # per-layer orthonormal bases (np)
        self.project_type = []                    # 'remove' | 'retain' per layer
        self.feature_mat = []                     # per-layer projection matrices

    # persistent state beyond the shared base model: the DualGPM bases that
    # seed future tasks' analytic A, the current unfolded task's own slot, and
    # fc; folding into q_proj/v_proj itself costs zero additional memory
    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        cur_slot_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[self._cur_task].parameters():
                    cur_slot_bytes += p.numel() * p.element_size()
        # feature_mat is a cached derivative of feature_list, recomputable, but
        # counted here since it is actually held in memory
        dualgpm_bytes = 0
        for f in self.feature_list:
            dualgpm_bytes += f.numel() * f.element_size()
        for m in self.feature_mat:
            dualgpm_bytes += m.numel() * m.element_size()
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters()) if net.fc is not None else 0
        total_bytes = cur_slot_bytes + dualgpm_bytes + fc_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"frozen_delta": 0, "current_slot": cur_slot_bytes,
                             "dualgpm_bases": dualgpm_bytes, "fc": fc_bytes}}

    # sample-boundary streaming hooks: one DualGPM slot per chunk, with A set
    # analytically from the currently-available chunk's input covariance
    def _stream_init(self):
        self._stream_chunk = -1
        self._stream_train_a = False             # InfLoRA: A analytic+frozen, train only B

    def _stream_slot(self):
        return self._stream_chunk

    def _stream_begin_chunk(self, loader):
        self._stream_chunk += 1
        if self._stream_chunk > 0:
            self._network.add_task_slot()
        self._cur_task = self._stream_chunk      # _init_lora_A / DualGPM read _cur_task as slot
        self._network.freeze_to_task(self._stream_chunk, train_a=False)
        self._fold_into_base(self._stream_chunk - 1)
        self._network.free_folded_slots(self._stream_chunk - 1)
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._network.to(self._device)
        self._init_lora_A(loader)                # analytic, DualGPM-projected A for this chunk
        self._stream_new_optimizer()

    def _stream_end_chunk(self, loader):
        self._update_dualgpm(loader)             # grow DualGPM feature memory

    # task loop
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = self._cur_task
        logging.info("[InfLoRA] Learning on {}-{}".format(self._known_classes, self._total_classes))

        # grow one adapter slot if this task needs an index that isn't allocated yet
        if self._cur_task >= self._network.backbone.n_tasks:
            self._network.add_task_slot()

        # only B (+ head) trainable; A is set analytically below. freeze_to_task
        # is kept only for its trainability-toggle side effect; the fold itself
        # happens next, directly into q_proj/v_proj
        self._network.freeze_to_task(self._cur_task, train_a=False)
        self._fold_into_base(self._cur_task - 1)
        self._network.free_folded_slots(self._cur_task - 1)
        for p in self._network.fc.parameters():
            p.requires_grad = True

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

        self._network.to(self._device)
        self._init_lora_A(self.train_loader)
        self._log_trainable()
        self._train(self.train_loader)           # train B + head (inherited)
        self._update_dualgpm(self.train_loader)

    @torch.no_grad()
    def _fold_into_base(self, task):
        """Fold `task`'s trained (A, B) directly into q_proj/v_proj's own
        weight, in place and mathematically exact. Safe because InfLoRA never
        reads a folded slot again, and TIL is never evaluated. No-op for
        task < 0."""
        if task < 0:
            return
        net = self._network.module if hasattr(self._network, "module") else self._network
        for attn in net.attn_modules():
            # lora_scaling applied explicitly since _lora_delta's forward applies it
            # after summing, and it is only ever 1.0 while lora_alpha stays unset
            attn.q_proj.weight.data += attn.lora_scaling * (attn.lora_B_q[task].weight @ attn.lora_A_q[task].weight)
            attn.v_proj.weight.data += attn.lora_scaling * (attn.lora_B_v[task].weight @ attn.lora_A_v[task].weight)

    # analytic init of the down-projection A
    @torch.no_grad()
    def _init_lora_A(self, train_loader):
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        # full forward pass to accumulate the input covariance; merge=False since
        # q_proj/v_proj already carry every past task's folded contribution and
        # this task's own freshly allocated slot has not trained yet
        for _, inputs, _t in train_loader:
            net(inputs.to(self._device), task=self._cur_task, merge=False)
        net.set_collect(False)

        for kk, attn in enumerate(net.attn_modules()):
            cur = attn.cur_matrix.clone().double()           # [dim, dim] on cpu
            # treat "no projector banked for this block yet" the same as task 0
            if self._cur_task == 0 or kk >= len(self.feature_mat):
                U, S, _ = torch.linalg.svd(cur)
                basis = U[:, :self.rank]
            else:
                fmat = self.feature_mat[kk].double()
                if self.project_type[kk] == "remove":
                    cur = cur - fmat @ cur
                else:
                    assert self.project_type[kk] == "retain"
                    cur = fmat @ cur
                U, S, _ = torch.linalg.svd(cur, full_matrices=False)
                basis = U[:, :self.rank]
            A = (basis.t() / math.sqrt(3)).float().to(self._device)   # [rank, dim]
            attn.lora_A_q[self._cur_task].weight.data.copy_(A)
            attn.lora_A_v[self._cur_task].weight.data.copy_(A)
            attn.reset_cur_matrix()

    # grow the DualGPM feature memory
    @torch.no_grad()
    def _update_dualgpm(self, train_loader):
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        # merge=False: runs after _train, so this task's own slot holds its
        # just-trained B, and q_proj/v_proj already carry the folded past
        for _, inputs, _t in train_loader:
            net(inputs.to(self._device), task=self._cur_task, merge=False)
        net.set_collect(False)

        mat_list = [attn.cur_matrix.clone() for attn in net.attn_modules()]
        for attn in net.attn_modules():
            attn.reset_cur_matrix()
        self.update_DualGPM(mat_list)

        # feature_mat is the cached projector _init_lora_A reads next cycle,
        # rebuilt after update_DualGPM has grown feature_list for this cycle
        self.feature_mat = []
        for p in range(len(self.feature_list)):
            Uf = self.feature_list[p] @ self.feature_list[p].t()
            self.feature_mat.append(Uf)

    # GPU port of InfLoRA's own update_DualGPM
    def update_DualGPM(self, mat_list):
        threshold = (self.lame - self.lamb) * self._cur_task / self.total_sessions + self.lamb
        logging.info("[InfLoRA] DualGPM threshold: {:.4f}".format(threshold))
        if len(self.feature_list) == 0:
            for activation in mat_list:
                U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = int(torch.sum(torch.cumsum(sval_ratio, 0) < threshold).item())
                self.feature_list.append(U[:, 0:max(r, 1)])
                self.project_type.append('remove' if r < (activation.shape[0] / 2) else 'retain')
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                if self.project_type[i] == 'remove':
                    U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    act_hat = activation - self.feature_list[i] @ (self.feature_list[i].t() @ activation)
                    U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = (sval_total - sval_hat) / sval_total
                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        continue
                    Ui = torch.cat([self.feature_list[i], U[:, 0:r]], dim=1)
                    self.feature_list[i] = Ui[:, 0:Ui.shape[0]] if Ui.shape[1] > Ui.shape[0] else Ui
                else:
                    assert self.project_type[i] == 'retain'
                    U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    act_hat = self.feature_list[i] @ (self.feature_list[i].t() @ activation)
                    U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = sval_hat / sval_total
                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval >= (1 - threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        continue
                    act_feature = self.feature_list[i] - U[:, 0:r] @ (U[:, 0:r].t() @ self.feature_list[i])
                    Ui, Si, Vi = torch.linalg.svd(act_feature, full_matrices=True)
                    self.feature_list[i] = Ui[:, :self.feature_list[i].shape[1] - r]

        # convert any over-grown 'remove' layer to its 'retain' complement
        for i in range(len(self.feature_list)):
            if self.project_type[i] == 'remove' and \
                    (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0] / 2)):
                feature = self.feature_list[i]
                U, S, V = torch.linalg.svd(feature, full_matrices=True)
                self.feature_list[i] = U[:, feature.shape[1]:]
                self.project_type[i] = 'retain'
            elif self.project_type[i] == 'retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0] / 2)
            logging.info("[InfLoRA] Layer {}: {}/{} type {}".format(
                i + 1, self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
