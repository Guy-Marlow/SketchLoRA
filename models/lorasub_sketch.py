"""LoRA-Subtraction-Sketch: bounds models/lorasub.py's per-task
K/V adapter bank at a fixed 2 slots (sketch and residual) with the same
exact QR-SVD merge SketchLoRA uses, and no orthogonality penalty, since
the merge handles overlapping directions exactly rather than needing one."""

import logging
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.lorasub import Learner as LoRASubLearner
from models.sketchlora import SKETCH, RESIDUAL

num_workers = 8


class Learner(LoRASubLearner):
    """Folds the residual slot into the sketch via QR-SVD each task
    instead of growing a bank."""

    def __init__(self, args):
        super().__init__(args)
        assert args.get("lora_n_slots", 2) == 2, \
            "LoRA-Subtraction-Sketch is a fixed 2-slot method (SKETCH + RESIDUAL). " \
            "It never grows a per-task slot bank. Got lora_n_slots={}".format(
                args.get("lora_n_slots", 2))
        self.svd_rank = args.get("svd_rank", args.get("lora_rank", 10))
        self.merge_op = args.get("merge_op", "qrsvd")
        assert self.merge_op == "qrsvd", \
            "LoRA-Subtraction-Sketch only implements merge_op='qrsvd'. Got {}".format(self.merge_op)
        self._sketch_populated = False

    def _train_adapter(self):
        return RESIDUAL

    # Streaming uses plain AdamW/CE, not the DRS loop.
    def _stream_slot(self):
        return RESIDUAL

    def _stream_begin_chunk(self, loader):
        self._cur_task += 1
        super()._stream_begin_chunk(loader)

    def _stream_end_chunk(self, loader):
        self._sketch_fold()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        self._register_task_range()
        self._network.default_task = RESIDUAL
        logging.info("[LoRA-Sub-Sketch] Learning on {}-{}".format(self._known_classes, self._total_classes))

        self._network.freeze_to_task(RESIDUAL)
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

        self._sketch_fold()

    @torch.no_grad()
    def _sketch_fold(self):
        """Exact QR-SVD merge of RESIDUAL into SKETCH; no orthogonality
        penalty needed."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        retained_energies = []
        for attn in net.attn_modules():
            for A_list, B_list in ((attn.lora_A_k, attn.lora_B_k), (attn.lora_A_v, attn.lora_B_v)):
                A_sk, B_sk = A_list[SKETCH].weight, B_list[SKETCH].weight
                A_res, B_res = A_list[RESIDUAL].weight, B_list[RESIDUAL].weight
                dt = B_sk.dtype

                if not self._sketch_populated:
                    # First task: copy RESIDUAL into SKETCH directly.
                    assert A_res.shape[0] == self.svd_rank, \
                        "first-fold transplant needs svd_rank == lora_rank for an exact, " \
                        "lossless copy. Got lora_rank={} svd_rank={}".format(
                            A_res.shape[0], self.svd_rank)
                    A_list[SKETCH].weight.data.copy_(A_res)
                    B_list[SKETCH].weight.data.copy_(B_res)
                    retained_energies.append(1.0)
                else:
                    delta_true = (B_sk.float() @ A_sk.float()) + (B_res.float() @ A_res.float())
                    B_cat = torch.cat([B_sk, B_res], dim=1).float()      # [d, r_sk + rank]
                    A_cat = torch.cat([A_sk, A_res], dim=0).float()      # [r_sk + rank, d]
                    Q_B, R_B = torch.linalg.qr(B_cat)
                    Q_A, R_A = torch.linalg.qr(A_cat.t())
                    M_small = R_B @ R_A.t()
                    U_small, S, Vh_small = torch.linalg.svd(M_small)
                    U = Q_B @ U_small
                    Vh = Vh_small @ Q_A.t()

                    r = min(self.svd_rank, S.shape[0])
                    sqrtS = S[:r].sqrt()
                    B_new = (U[:, :r] * sqrtS).to(dt)
                    A_new = (sqrtS.unsqueeze(1) * Vh[:r, :]).to(dt)

                    recon = B_new.float() @ A_new.float()
                    fro_true = torch.linalg.norm(delta_true)
                    retained = 1.0 - (torch.linalg.norm(delta_true - recon) / fro_true.clamp_min(1e-12)) ** 2
                    retained_energies.append(retained.item())

                    A_list[SKETCH].weight.data.copy_(A_new)
                    B_list[SKETCH].weight.data.copy_(B_new)

                nn.init.kaiming_uniform_(A_list[RESIDUAL].weight, a=math.sqrt(5))
                kr = 1.0 / attn.dim
                nn.init.sparse_(B_list[RESIDUAL].weight, sparsity=(1 - kr))

        logging.info("[LoRA-Sub-Sketch] fold at task {}: retained_energy mean={:.4f} min={:.4f}".format(
            self._cur_task, sum(retained_energies) / len(retained_energies), min(retained_energies)))
        self._sketch_populated = True

    def persistent_state(self):
        """Counts sketch, residual, head, and prototype bytes."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        sketch_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_k, attn.lora_B_k, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[SKETCH].parameters():
                    sketch_bytes += p.numel() * p.element_size()
        residual_bytes = 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_k, attn.lora_B_k, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[RESIDUAL].parameters():
                    residual_bytes += p.numel() * p.element_size()
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        proto_bytes = sum(p.numel() * p.element_size() for p in self._protos.values())
        total_bytes = sketch_bytes + residual_bytes + fc_bytes + proto_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"sketch": sketch_bytes, "residual": residual_bytes,
                             "head": fc_bytes, "protos": proto_bytes}}

    def _log_memory_accounting(self):
        state = self.persistent_state()
        logging.info(
            "[LoRA-Sub-Sketch] task {} persistent bank: fixed 2 slots (K+V, A+B, every "
            "block) = {:,} bytes (sketch={:,}, residual={:,}, head={:,}, protos={:,} "
            "[{} classes])".format(
                self._cur_task, state["bytes"], state["breakdown"]["sketch"],
                state["breakdown"]["residual"], state["breakdown"]["head"],
                state["breakdown"]["protos"], len(self._protos)))
