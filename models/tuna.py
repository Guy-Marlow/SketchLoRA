import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import TUNANet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, target2onehot
from torch.distributions.multivariate_normal import MultivariateNormal

num_workers = 8

# Isotropic covariance floor, applied at sample time.
_CLS_COV_EPS = 1e-4


def _cls_cov_value_bytes(v):
    """Byte cost of one stored cls_cov entry."""
    if isinstance(v, tuple):
        return sum(t.numel() * t.element_size() for t in v)
    return v.numel() * v.element_size()


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=20, m=0):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m

        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf * self.s, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = TUNANet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()
        self.use_orth = args["use_orth"]
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.args['tuned_epoch'] = args['tuned_epoch']
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.ca_cov_rank = args.get("ca_cov_rank", 5)

        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def persistent_state(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        adapter_bytes = sum(p.numel() * p.element_size() for p in net.backbone.adapter_list.parameters())
        fc_bytes = sum(p.numel() * p.element_size() for p in net.fc.parameters()) if net.fc is not None else 0
        cls_stats_bytes = (sum(v.numel() * v.element_size() for v in self.cls_mean.values())
                            + sum(_cls_cov_value_bytes(v) for v in self.cls_cov.values()))
        total_bytes = adapter_bytes + cls_stats_bytes + fc_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes),
                "breakdown": {"adapters": adapter_bytes, "cls_stats": cls_stats_bytes, "fc": fc_bytes}}

    @torch.no_grad()
    def _deployed_forward(self, inputs):
        """Deployed forward: an entropy-based ensemble over banked adapters."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        if self._cur_task == 0:
            features = net.backbone(inputs, adapter_id=0, train=False)["features"]
            logits = net.fc(features)["logits"][:, :self._total_classes]
            return {"logits": logits}
        n_banked = len(net.backbone.adapter_list)
        if n_banked == 0:
            features = net.backbone(inputs, adapter_id=0, train=False)["features"]
            logits = net.fc(features)["logits"][:, :self._total_classes]
            return {"logits": logits}
        all_logits, all_entropies = [], []
        for i in range(n_banked):
            features = net.backbone(inputs, adapter_id=i, train=False)["features"]
            logits = net.fc(features)["logits"][:, :self._total_classes] * self.args['scale']
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            all_logits.append(logits)
            all_entropies.append(entropy)
        all_logits = torch.stack(all_logits)
        all_entropies = torch.stack(all_entropies)
        min_entropy_indices = torch.argmin(all_entropies, dim=0)
        min_entropy_logits = all_logits[min_entropy_indices, torch.arange(inputs.shape[0], device=inputs.device)]
        features = net.backbone(inputs, adapter_id=n_banked, train=False)["features"]
        general_logits = net.fc(features)["logits"][:, :self._total_classes] * self.args['scale']
        outputs = F.softmax(general_logits, dim=1) + F.softmax(min_entropy_logits, dim=1)
        return {"logits": outputs}

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        self._network.update_fc(self._total_classes - self._known_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source="train", mode="train")
        self.data_manager = data_manager
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)
        self._network.fc.to(self._device)
        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)

        self._init_train(train_loader, test_loader, optimizer, scheduler)

        self._network.backbone.adapter_update()
        if self._cur_task > 0:
            self._network.backbone.merge()
        self._compute_mean(self._network.backbone)
        if self._cur_task > 0:
            self.classifer_align(self._network.backbone)

    def get_optimizer(self, model):
        base_params = [p for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad]
        base_params = {'params': base_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': self._network.fc.parameters(), 'lr': self.init_lr,
                          'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                network_params,
                momentum=0.9,
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                network_params,
            )

        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                network_params,
            )

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'],
                                                             eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["m"])
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["features"]
                logits = self._network.fc(features)["logits"]

                # Excludes carryover-tail samples: cosface can't use ignore_index.
                new_mask = targets >= self._known_classes
                if new_mask.any():
                    loss = loss_cos(logits[new_mask][:, self._known_classes:],
                                    targets[new_mask] - self._known_classes)
                else:
                    loss = torch.zeros((), device=self._device, requires_grad=True)
                if self._cur_task > 0 and self.use_orth:
                    loss += self.orth_loss(features) * self.args["reg"] * torch.exp(-torch.tensor(self._cur_task+1, dtype=torch.float32, device=self._device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds[new_mask].eq(targets[new_mask]).cpu().sum()
                total += int(new_mask.sum())

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
    def orth_loss(self, features):
        final_loss = 0

        n_banked = min(self._cur_task, len(self._network.backbone.adapter_list))
        for i in range(n_banked):
            loss = 0
            for j in range(12):
                cur_up_proj = self._network.backbone.cur_adapter[j].up_proj.weight
                prev_up_proj = self._network.backbone.adapter_list[i][j].up_proj.weight
                cur_up_proj_normalized = F.normalize(cur_up_proj, p=2, dim=1)
                prev_up_proj_normalized = F.normalize(prev_up_proj, p=2, dim=1)

                dot_product = torch.mean(torch.matmul(cur_up_proj_normalized, prev_up_proj_normalized.transpose(1, 0)))

                loss += (torch.abs(dot_product)) / 12
            final_loss += loss  
       
        return final_loss
    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4
            )

            vectors = []
            for _, _inputs, _targets in idx_loader:
                _vectors = model(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (
                        torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.diag(
                    torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(
                        self._device))
            elif self.args["ca_storage_efficient_method"] == 'lowrank_covariance':
                # Randomized SVD; oversamples r+6 before truncating to r.
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                cov = torch.cov(features_per_cls.T)
                r = self.ca_cov_rank
                q = min(r + 6, cov.shape[-1])
                U, S, _ = torch.svd_lowrank(cov, q=q, niter=2)
                self.cls_cov[class_idx] = (U[:, :r].contiguous(), S[:r].clamp(min=0).contiguous())


    def classifer_align(self, model):
        for p in self._network.fc.parameters():
            p.requires_grad = True

        run_epochs = self.crct_epochs
        network_params = [
            {'params': self._network.fc.parameters(), 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)

        prog_bar = tqdm(range(run_epochs))
        task_size = self._known_classes - self._total_classes
        self._network.eval()

        n_ca_classes = min(self._total_classes, len(self.cls_mean))
        if n_ca_classes == 0:
            return

        # Distributions built once; only sampling repeats every epoch.
        lowrank_mode = self.args["ca_storage_efficient_method"] == 'lowrank_covariance'
        if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
            distributions = {}
            for class_idx in range(n_ca_classes):
                if self.args["decay"]:
                    t_id = class_idx // task_size
                    decay = (t_id + 1) / (self._cur_task + 1) * 0.1

                    mean = torch.tensor(self.cls_mean[class_idx], dtype=torch.float64).to(self._device) * (
                            0.9 + decay)
                else:
                    mean = self.cls_mean[class_idx].to(self._device)
                cov = self.cls_cov[class_idx].to(self._device)
                if self.args["ca_storage_efficient_method"] == 'variance':
                    cov = torch.diag(cov)
                distributions[class_idx] = MultivariateNormal(mean.float(), cov.float())
        elif lowrank_mode:
            # Rank-r covariance is singular; sample via U/S.
            lowrank_factors = {}
            for class_idx in range(n_ca_classes):
                mean = self.cls_mean[class_idx].to(self._device).float()
                U, S = self.cls_cov[class_idx]
                lowrank_factors[class_idx] = (mean, U.to(self._device).float(), S.to(self._device).float().sqrt())
        else:
            raise NotImplementedError

        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            for class_idx in range(n_ca_classes):
                if lowrank_mode:
                    mean, U, sqrtS = lowrank_factors[class_idx]
                    r = U.shape[1]
                    z1 = torch.randn(num_sampled_pcls, r, device=self._device)
                    z2 = torch.randn(num_sampled_pcls, mean.shape[-1], device=self._device)
                    sampled_data_single = (mean.unsqueeze(0) + (z1 * sqrtS) @ U.t()
                                           + (_CLS_COV_EPS ** 0.5) * z2)
                else:
                    m = distributions[class_idx]
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)

                sampled_label.extend([class_idx] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(n_ca_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = self._network.fc(inp)["logits"]
                logits = self.args['scale'] * outputs

                loss = F.cross_entropy(logits, tgt)

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn1(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                features = self._network.backbone(inputs, adapter_id=0, train=False)["features"]
                logits = self._network.fc(features)["logits"][:, :self._total_classes]

            k = min(self.topk, self._total_classes)
            predicts = torch.topk(logits, k=k, dim=1, largest=True, sorted=True)[1]
            if k < self.topk:
                pad = predicts[:, -1:].expand(-1, self.topk - k)
                predicts = torch.cat([predicts, pad], dim=1)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_cnn(self, loader):
        if self._cur_task == 0:
            y_pred, y_true = self._eval_cnn1(loader)
            return y_pred, y_true
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_specific, y_pred_general = [], []
        k = min(self.topk, self._total_classes)
        n_banked = len(self._network.backbone.adapter_list)
        if n_banked == 0:
            return self._eval_cnn1(loader)
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            all_predicts = []
            all_entropies = []
            all_logits = []
            for i in range(n_banked):
                with torch.no_grad():
                    features = self._network.backbone(inputs, adapter_id=i, train=False)["features"]
                    logits = self._network.fc(features)["logits"][:, :self._total_classes]*self.args['scale']
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                predicts = torch.topk(logits, k=k, dim=1, largest=True, sorted=True)[1]
                all_predicts.append(predicts.cpu().numpy())
                all_entropies.append(entropy.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
            all_predicts = np.array(all_predicts)
            all_entropies = torch.tensor(all_entropies)
            all_logits = torch.tensor(all_logits)
            min_entropy_indices = torch.argmin(all_entropies, axis=0)

            min_entropy_logits = all_logits[min_entropy_indices, torch.arange(len(min_entropy_indices))].to(
                self._device)
            with torch.no_grad():
                features = self._network.backbone(inputs, adapter_id=n_banked, train=False)["features"]
                logits = self._network.fc(features)["logits"][:, :self._total_classes]*self.args['scale']
                logits = F.softmax(logits, dim=1)
            min_entropy_logits = F.softmax(min_entropy_logits, dim=1)

            outputs = logits + min_entropy_logits
            predicts = torch.topk(outputs, k=k, dim=1, largest=True, sorted=True)[1]
            if k < self.topk:
                pad = predicts[:, -1:].expand(-1, self.topk - k)
                predicts = torch.cat([predicts, pad], dim=1)
            pred_specific = torch.max(min_entropy_logits,dim=1)[1]
            pred_general = torch.max(logits, dim=1)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            y_pred_specific.append(pred_specific.cpu().numpy())
            y_pred_general.append(pred_general.cpu().numpy())
       
        return np.concatenate(y_pred), np.concatenate(y_true)