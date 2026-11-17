"""LoRA-Subtraction / LoRA-minus: a per-task K/V adapter
bank trained with Adam steps projected onto each task's own drift-resistant
input subspace, plus a prototype-repulsion loss pushing new features away
from past-task class means to keep the additive bank from interfering
across tasks."""

import logging
import math
from collections import defaultdict

import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.lora import Learner as LoRALearner
from utils.toolkit import tensor2numpy
from utils.metrics_logger import log_bank_effective_rank


class _ProjectedAdam(torch.optim.Optimizer):
    """Adam with an optional projection of each step's update onto a
    per-parameter subspace."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, svd=False,
                 thres=1.001, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        svd=svd, thres=thres)
        super().__init__(params, defaults)
        self.eigens = defaultdict(dict)
        self.transforms = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("svd", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            svd = group["svd"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = self._get_update(group, p.grad, p)
                if svd and p in self.transforms:
                    transform = self.transforms[p]
                    if transform.shape[0] == update.shape[0]:
                        update = transform @ update
                    else:
                        update = update @ transform
                p.add_(update)
        return loss

    def _get_update(self, group, grad, p):
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        state["step"] += 1

        if group["weight_decay"] != 0:
            grad = grad.add(p, alpha=group["weight_decay"])

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.sqrt().add_(group["eps"])

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
        return -step_size * exp_avg / denom

    def get_eigens(self, fea_in):
        """Eigendecomposes each parameter's input covariance."""
        for group in self.param_groups:
            if not group["svd"]:
                continue
            for p in group["params"]:
                if not p.requires_grad or p not in fea_in:
                    continue
                U, S, _ = torch.linalg.svd(fea_in[p].double())
                self.eigens[p] = {"eigen_value": S.float(), "eigen_vector": U.float()}

    def get_transforms(self):
        """Builds each parameter's projection matrix from its eigenbasis."""
        for group in self.param_groups:
            if not group["svd"]:
                continue
            thres = group["thres"]
            for p in group["params"]:
                if not p.requires_grad or thres == 1.0 or p not in self.eigens:
                    continue
                eigen_value = self.eigens[p]["eigen_value"]
                eigen_vector = self.eigens[p]["eigen_vector"]
                cumulative = eigen_value.cumsum(dim=0) / eigen_value.sum()
                num_vectors = int((cumulative >= thres).nonzero(as_tuple=True)[0][0].item()) + 1
                basis = eigen_vector[:, :num_vectors]
                transform = basis @ basis.t()
                self.transforms[p] = transform / torch.norm(transform)
        total = sum(1 for g in self.param_groups if g["svd"] for _ in g["params"])
        logging.debug("[LoRA-Sub] DRS transforms built for {} parameter(s)".format(total))


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        self.train_merge = True
        self.svd_thres = args.get("svd_thres", 0.99)
        self.adam_betas = tuple(args.get("adam_betas", (0.9, 0.999)))
        self.adam_eps = args.get("adam_eps", 1e-8)
        self._fea_in = {}
        self.use_proto_repulsion = bool(args.get("use_proto_repulsion", True))
        self.proto_repulsion_weight = args.get("proto_repulsion_weight", 0.2)
        self.proto_margin = args.get("proto_margin", 2.0)
        self._protos = {}
        self.log_bank_rank = bool(args.get("log_bank_rank", False))
        self.bank_rank_energy_target = args.get("bank_rank_energy_target", 0.99)

    @torch.no_grad()
    def _collect_subtracted_covariance(self, train_loader):
        """The "subtraction" probe: collects each block's input covariance
        with past tasks' contribution removed."""
        net = self._network
        net.eval()
        net.reset_cur_matrix()
        net.set_collect(True)
        net.set_subtract_mode(True)
        task = self._train_adapter()
        for _, inputs, _t in train_loader:
            net(inputs.to(self._device), task=task, merge=True)
        net.set_collect(False)
        net.set_subtract_mode(False)

        for attn in net.attn_modules():
            cov = attn.cur_matrix.clone()
            self._fea_in[attn.lora_A_k[task].weight] = cov
            self._fea_in[attn.lora_B_k[task].weight] = cov
            self._fea_in[attn.lora_A_v[task].weight] = cov
            self._fea_in[attn.lora_B_v[task].weight] = cov
            attn.reset_cur_matrix()

    @torch.no_grad()
    def _update_protos(self, train_loader):
        """Stores each class's mean feature for the next task's repulsion term."""
        net = self._network
        net.eval()
        adapter = self._train_adapter()
        feats_by_class = defaultdict(list)
        for _, inputs, targets in train_loader:
            feats = net.extract_vector(inputs.to(self._device), task=adapter, merge=True).cpu()
            for f, y in zip(feats, targets):
                feats_by_class[int(y.item())].append(f)
        for cls, flist in feats_by_class.items():
            self._protos[cls] = torch.stack(flist, dim=0).mean(dim=0)

    def _proto_repulsion_loss(self, features, protos_tensor):
        """Margin hinge pushing features away from the nearest past prototype."""
        feat_norm = F.normalize(features, dim=-1)
        dists = torch.cdist(feat_norm, protos_tensor)
        min_dist, _ = dists.min(dim=1)
        return F.relu(self.proto_margin - min_dist).mean()

    def _optimizer_param_groups_split(self):
        net = self._network
        task = self._train_adapter()
        feature_params = []
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_k, attn.lora_B_k, attn.lora_A_v, attn.lora_B_v):
                feature_params += [p for p in mlist[task].parameters() if p.requires_grad]
        head_params = [p for p in net.fc.parameters() if p.requires_grad]
        return feature_params, head_params

    def _train(self, train_loader):
        net = self._network
        net.to(self._device)
        # adapter differs from task for models/lorasub_sketch.py.
        task = self._cur_task
        adapter = self._train_adapter()

        if task > 0:
            self._collect_subtracted_covariance(train_loader)

        feature_params, head_params = self._optimizer_param_groups_split()
        optimizer = _ProjectedAdam(
            [{"params": feature_params, "svd": True, "thres": self.svd_thres, "lr": self.init_lr},
             {"params": head_params, "svd": False, "lr": self.init_lr}],
            betas=self.adam_betas, eps=self.adam_eps, weight_decay=self.weight_decay)

        if task > 0:
            optimizer.get_eigens(self._fea_in)
            optimizer.get_transforms()
            self._fea_in = {}

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        use_proto = self.use_proto_repulsion and task > 0 and len(self._protos) > 0
        protos_tensor = (F.normalize(torch.stack(list(self._protos.values()), dim=0), dim=-1).to(self._device)
                          if use_proto else None)

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            net.train()
            losses, proto_losses, correct, total = 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                ret = net(inputs, task=adapter, merge=True)
                logits, features = ret["logits"], ret["features"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)
                if use_proto:
                    proto_loss = self._proto_repulsion_loss(features, protos_tensor)
                    loss = loss + self.proto_repulsion_weight * proto_loss
                    proto_losses += proto_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = tensor2numpy(correct) * 100 / total if total else 0.0
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f} (proto {:.3f}), Train_accy {:.2f}".format(
                    task, epoch + 1, self.epochs, losses / len(train_loader),
                    proto_losses / len(train_loader), train_acc))
        logging.info("[LoRA-Sub] Task {} finished. Train_accy {:.2f}".format(task, train_acc))
        self._update_protos(train_loader)
        self._log_memory_accounting()
        self._log_bank_effective_rank()

    def persistent_state(self):
        """K+V adapter bank, head, and prototype store."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        total_params, total_bytes = 0, 0
        for attn in net.attn_modules():
            for mlist in (attn.lora_A_k, attn.lora_B_k, attn.lora_A_v, attn.lora_B_v):
                for m in mlist:
                    for p in m.parameters():
                        total_params += p.numel()
                        total_bytes += p.numel() * p.element_size()
        lora_bytes = total_bytes
        head_params = sum(p.numel() for p in net.fc.parameters())
        head_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters())
        proto_params = sum(p.numel() for p in self._protos.values())
        proto_bytes = sum(p.numel() * p.element_size() for p in self._protos.values())
        total_params += head_params + proto_params
        total_bytes += head_bytes + proto_bytes
        return {"params": total_params, "bytes": total_bytes,
                "breakdown": {"lora_slots": lora_bytes, "head": head_bytes, "protos": proto_bytes}}

    def _log_memory_accounting(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        state = self.persistent_state()
        logging.info(
            "[LoRA-Sub] task {} persistent bank: {} slot(s) allocated (K+V, A+B, every "
            "block) = {:,} bytes (adapters={:,}, head={:,}, protos={:,} [{} classes])".format(
                self._cur_task, net.backbone.n_tasks, state["bytes"],
                state["breakdown"]["lora_slots"], state["breakdown"]["head"],
                state["breakdown"]["protos"], len(self._protos)))

    @torch.no_grad()
    def _log_bank_effective_rank(self):
        """The summed bank's true combined rank, not each slot's own."""
        if not self.log_bank_rank:
            return
        net = self._network.module if hasattr(self._network, "module") else self._network
        t = self._train_adapter()
        deltas = {}
        for li, attn in enumerate(net.attn_modules()):
            for proj, (A_list, B_list) in (("k", (attn.lora_A_k, attn.lora_B_k)),
                                            ("v", (attn.lora_A_v, attn.lora_B_v))):
                B_cat = torch.cat([B_list[s].weight for s in range(t + 1)], dim=1).float()
                A_cat = torch.cat([A_list[s].weight for s in range(t + 1)], dim=0).float()
                deltas[(li, proj)] = B_cat @ A_cat
        log_bank_effective_rank("LoRA-Sub", self._cur_task, deltas, self.bank_rank_energy_target)
