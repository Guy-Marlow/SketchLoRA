import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, EaseCosineLinear, TunaLinear, CosineLinearFeature
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    elif '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ease" :
            from backbone import vit_ease
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_ease":
                model = vit_ease.vit_base_patch16_224_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_ease":
                model = vit_ease.vit_base_patch16_224_in21k_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    elif '_tuna' in name:
        # Fixed rather than configurable: TUNA's adapter width comes from `r`
        # below, not ffn_num, which is only carried for shape-parity with the
        # EASE/CL-LoRA tuning_config template.
        ffn_num = 16
        from backbone import vit_tuna
        from easydict import EasyDict
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            _device=args["device"][0],
            # VPT related
            vpt_on=False,
            vpt_num=0,
        )
        if name == "vit_base_patch16_224_tuna":
            model = vit_tuna.vit_base_patch16_224_adapter(num_classes=args["nb_classes"],
                                                             global_pool=False, drop_path_rate=0.0,
                                                             tuning_config=tuning_config,
                                                             r=args["r"])
        elif name == "vit_base_patch16_224_in21k_tuna":
            model = vit_tuna.vit_base_patch16_224_in21k_adapter(num_classes=args["nb_classes"],
                                                                   global_pool=False, drop_path_rate=0.0,
                                                                   tuning_config=tuning_config,
                                                                   r=args["r"])
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        return model
    # SD-LoRA+SketchLoRA backbone. Checked before the plain '_sdlora' branch
    # below since "vit_base_patch16_224_sdlora_sketch" contains "_sdlora" as a
    # substring. Fixed-size: one sketch slot plus one reused transient slot
    # per block, never growing.
    elif '_sdlora_sketch' in name:
        from backbone import vit_sdlora_sketch
        lora_rank = args.get("lora_rank", 10)
        lora_alpha = args.get("lora_alpha", None)
        n_lora_blocks = args.get("n_lora_blocks", None)
        svd_rank = args.get("svd_rank", lora_rank)
        freeze_rescale = bool(args.get("sdsketch_freeze_rescale", False))
        if name == "vit_base_patch16_224_sdlora_sketch":
            model = vit_sdlora_sketch.vit_base_patch16_224_sdlora_sketch(
                lora_rank=lora_rank, lora_alpha=lora_alpha, n_lora_blocks=n_lora_blocks,
                svd_rank=svd_rank, freeze_rescale=freeze_rescale, num_classes=0, drop_path_rate=0.0)
        elif name == "vit_base_patch16_224_in21k_sdlora_sketch":
            model = vit_sdlora_sketch.vit_base_patch16_224_in21k_sdlora_sketch(
                lora_rank=lora_rank, lora_alpha=lora_alpha, n_lora_blocks=n_lora_blocks,
                svd_rank=svd_rank, freeze_rescale=freeze_rescale, num_classes=0, drop_path_rate=0.0)
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        model.out_dim = 768
        return model
    # SD-LoRA backbone. Same growable-slot convention as the plain '_lora'
    # branch below.
    elif '_sdlora' in name:
        from backbone import vit_sdlora
        n_tasks = args.get("lora_n_slots", 1)
        lora_rank = args.get("lora_rank", 10)
        lora_alpha = args.get("lora_alpha", None)
        n_lora_blocks = args.get("n_lora_blocks", None)
        if name == "vit_base_patch16_224_sdlora":
            model = vit_sdlora.vit_base_patch16_224_sdlora(
                n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha,
                n_lora_blocks=n_lora_blocks, num_classes=0, drop_path_rate=0.0)
        elif name == "vit_base_patch16_224_in21k_sdlora":
            model = vit_sdlora.vit_base_patch16_224_in21k_sdlora(
                n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha,
                n_lora_blocks=n_lora_blocks, num_classes=0, drop_path_rate=0.0)
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        model.out_dim = 768
        return model
    # LoRA-Subtraction (k,v) backbone. Checked before the generic '_lora'
    # branch below since "lorasub" contains "_lora" as a substring.
    elif '_lorasub' in name:
        from backbone import vit_lorasub
        n_tasks = args.get("lora_n_slots", 1)
        lora_rank = args.get("lora_rank", 10)
        lora_alpha = args.get("lora_alpha", None)
        n_lora_blocks = args.get("n_lora_blocks", None)
        if name == "vit_base_patch16_224_lorasub":
            model = vit_lorasub.vit_base_patch16_224_lorasub(
                n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha,
                n_lora_blocks=n_lora_blocks, num_classes=0, drop_path_rate=0.0)
        elif name == "vit_base_patch16_224_in21k_lorasub":
            model = vit_lorasub.vit_base_patch16_224_in21k_lorasub(
                n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha,
                n_lora_blocks=n_lora_blocks, num_classes=0, drop_path_rate=0.0)
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        model.out_dim = 768
        return model
    # Batch-incremental, boundary-agnostic (q,v) backbone. Always starts with
    # one live slot and grows through its own add_task_slot(), driven by the
    # drift detector or a fixed-interval clock rather than task count.
    elif '_sketchlora_batchinc' in name:
        from backbone import vit_sketchlora_batchinc
        lora_rank = args.get("lora_rank", 10)
        lora_alpha = args.get("lora_alpha", None)
        svd_rank = args.get("svd_rank", lora_rank)
        pending_cap = args.get("batchinc_pending_cap", 4)
        if name == "vit_base_patch16_224_sketchlora_batchinc":
            model = vit_sketchlora_batchinc.vit_base_patch16_224_sketchlora_batchinc(
                lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank,
                pending_cap=pending_cap, num_classes=0, drop_path_rate=0.0)
        elif name == "vit_base_patch16_224_in21k_sketchlora_batchinc":
            model = vit_sketchlora_batchinc.vit_base_patch16_224_in21k_sketchlora_batchinc(
                lora_rank=lora_rank, lora_alpha=lora_alpha, svd_rank=svd_rank,
                pending_cap=pending_cap, num_classes=0, drop_path_rate=0.0)
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        model.out_dim = 768
        return model
    # Baseline per-task LoRA (q,v) backbone.
    elif '_lora' in name:
        from backbone import vit_lora
        # Starts with one preallocated slot and grows one at a time via
        # add_task_slot(), called from each Learner's incremental_train.
        # Methods with constant memory (e.g. sketchlora) instead request a
        # fixed slot count through "lora_n_slots".
        n_tasks = args.get("lora_n_slots", 1)
        lora_rank = args.get("lora_rank", 10)
        lora_alpha = args.get("lora_alpha", None)
        # Restricts trainable LoRA to the first n blocks; None means all blocks.
        n_lora_blocks = args.get("n_lora_blocks", None)
        if name == "vit_base_patch16_224_lora":
            model = vit_lora.vit_base_patch16_224_lora(
                n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha,
                n_lora_blocks=n_lora_blocks, num_classes=0, drop_path_rate=0.0)
        elif name == "vit_base_patch16_224_in21k_lora":
            model = vit_lora.vit_base_patch16_224_in21k_lora(
                n_tasks=n_tasks, lora_rank=lora_rank, lora_alpha=lora_alpha,
                n_lora_blocks=n_lora_blocks, num_classes=0, drop_path_rate=0.0)
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        model.out_dim = 768
        return model
    # RainbowPrompt backbone.
    elif '_rainbowprompt' in name:
        from backbone import vit_rainbowprompt
        from backbone.rainbowprompt_module import RainbowPromptModule
        # Same growable-slot convention as the '_lora' branch above: starts
        # with one slot and grows via add_task_slot().
        n_tasks = args.get("lora_n_slots", 1)
        prompt_module = RainbowPromptModule(
            num_layers=12, embed_dim=768, n_tasks=n_tasks,
            length=args.get("rp_length", 20),
            self_attn_idx=args.get("self_attn_idx", [0, 1, 2, 3, 4, 5]),
            KI_iter=args.get("KI_iter", 10),
            use_linear=args.get("use_linear", True),
            D1=args.get("rp_D1", 28), D2=args.get("rp_D2", 56),
            evolve_dropout=args.get("rp_evolve_dropout", 0.1)).to(args["device"][0])
        model = vit_rainbowprompt.vit_base_patch16_224_rainbowprompt(prompt_module=prompt_module)
        model.out_dim = 768
        return model
    elif '_cllora' in name:
        from types import SimpleNamespace
        from backbone import vit_cllora
        ffn_num = args["ffn_num"]
        tuning_config = SimpleNamespace(
            use_distillation=args["use_distillation"],
            use_block_weight=args["use_block_weight"],
            msa_adapt=args["msa_adapt"],
            msa=args["msa"],
            specfic_pos=args["specfic_pos"],
            general_pos=args["general_pos"],
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            vpt_on=False,
            vpt_num=0,
            _device=args["device"][0],
        )
        if name == "vit_base_patch16_224_cllora":
            model = vit_cllora.vit_base_patch16_224_cllora(
                num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
        elif name == "vit_base_patch16_224_in21k_cllora":
            model = vit_cllora.vit_base_patch16_224_in21k_cllora(
                num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        model.out_dim = 768
        return model
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            # cnn path returns an fmaps/features/logits dict
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        # Uses the number of banked adapters, not the task count, so the classifier
        # width stays correct once a capped bank stops growing.
        n_banked = len(self.backbone.adapter_list)
        if self.use_init_ptm:
            return self.out_dim * (n_banked + 2)
        else:
            return self.out_dim * (n_banked + 1)

    def update_fc(self, nb_classes):
        self._cur_task += 1

        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            # Copies into the old weight's actual width, not an assumed growth-by-out_dim slice.
            old_width = weight.shape[1]
            fc.weight.data[: old_nb_classes, :old_width] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def generate_fc(self, in_dim, out_dim):
        fc = EaseCosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                # cur_task matches the number of banked adapters, as in feature_dim above.
                out = self.fc.forward_reweight(x, cur_task=len(self.backbone.adapter_list), alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
            
        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class OurNet(BaseNet):
    """Network wrapper for CL-LoRA: growing per-task classifier blocks over a general/specific-LoRA ViT backbone, with test-time forward always using the diagonal reweighted scoring path."""
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim = self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = False
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.fc_list = nn.ModuleList()
        self.adapter_list = nn.ModuleList()
        self.init_proto = None

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    @property
    def feature_dim(self):
        # Uses the number of banked adapters, not the task count, so the classifier
        # width stays correct once a capped bank stops growing.
        n_banked = len(self.backbone.adapter_list)
        if self.use_init_ptm:
            return self.out_dim * (n_banked + 2)
        return self.out_dim * (n_banked + 1)

    def update_fc(self, nb_classes):
        self._cur_task += 1
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)
        init_proto = self.generate_fc(self.out_dim, nb_classes).to(self._device)
        if self.init_proto is not None:
            old_nb_classes = self.init_proto.out_features
            weight = copy.deepcopy(self.init_proto.weight.data)
            init_proto.weight.data[: old_nb_classes, :] = nn.Parameter(weight)
        del self.init_proto
        self.init_proto = init_proto

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            # Copies into the old weight's actual width, not an assumed growth-by-out_dim slice.
            old_width = weight.shape[1]
            fc.weight.data[: old_nb_classes, :old_width] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        self.fc.requires_grad_(False)

    def add_fc(self):
        self.fc_list.append(self.proxy_fc.requires_grad_(False))
        del self.proxy_fc

    def generate_fc(self, in_dim, out_dim):
        return CosineLinearFeature(in_dim, out_dim)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward_kd(self, x, t_idx):
        x_new, x_teacher = self.backbone.forward_general_cls(x, t_idx)
        out_new, out_teacher = self.proxy_fc(x_new), self.proxy_fc(x_teacher)
        return out_new, out_teacher

    def forward(self, x, test=False):
        if not test:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
            out.update({"features": x})
            return out
        x_input = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
        # cur_task matches the number of banked adapters, as in feature_dim above.
        out = self.fc.forward_diagonal(x_input, cur_task=len(self.backbone.adapter_list), alpha=self.alpha,
                                       init_cls=self.init_cls, inc=self.inc,
                                       use_init_ptm=self.use_init_ptm, beta=self.beta)
        out.update({"features": x_input})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class TUNANet(nn.Module):
    def __init__(self, args, pretrained):
        super(TUNANet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]
       

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes,nextperiod_initialization=None):
      
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=False)

    def generate_fc(self, in_dim, out_dim): 
        fc = TunaLinear(in_dim, out_dim)
        return fc

    def forward_orig(self, x):
        features = self.backbone(x, adapter_id=0)['features']

        res = dict()
        res['features'] = features
        res['logits'] = self.fc(features)['logits']

        return res

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        res = self.backbone(x, adapter_id, train, fc_only)
        return res


class RainbowPromptVitNet(BaseNet):
    """Network wrapper for RainbowPrompt: a single growing linear head, with forward threading a task id and train flag into the backbone's evolving-prompt injection."""

    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.default_task = 0

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
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

    def forward(self, x, task_id=None, train=False, known_task=None):
        task_id = self.default_task if task_id is None else task_id
        out = self.backbone(x, task_id=task_id, train=train, known_task=known_task)
        logits = self.fc(out["features"])
        logits.update({"features": out["features"], "sim_loss": out["sim_loss"]})
        return logits


class LoRAVitNet(BaseNet):
    """Network wrapper for the baseline per-task LoRA ViT: a single linear head grows with the class space, and forward threads a task id and merge flag into the backbone to route to one LoRA or sum LoRAs for CIL."""

    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        # merge=True sums LoRAs 0..task for CIL inference.
        self.merge = bool(args.get("lora_merge", True))
        self.default_task = -1

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.weight.data[:nb_output] = weight
            # Some classifiers (e.g. CosineLinear) have no bias, so only copy one when both exist.
            if getattr(self.fc, "bias", None) is not None and getattr(fc, "bias", None) is not None:
                bias = copy.deepcopy(self.fc.bias.data)
                fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def _resolve(self, task, merge):
        return (self.default_task if task < 0 else task,
                self.merge if merge is None else merge)

    def extract_vector(self, x, task=-1, merge=None):
        task, merge = self._resolve(task, merge)
        return self.backbone(x, task=task, merge=merge)

    def forward(self, x, task=-1, merge=None):
        task, merge = self._resolve(task, merge)
        features = self.backbone(x, task=task, merge=merge)
        out = self.fc(features)
        out.update({"features": features})
        return out

    def freeze_to_task(self, task, train_a=True):
        self.backbone.freeze_to_task(task, train_a=train_a)

    # Passes through a new LoRA slot; some backbones take a rank override, others take none.
    def add_task_slot(self, rank=None):
        if rank is not None:
            self.backbone.add_task_slot(rank=rank)
        else:
            self.backbone.add_task_slot()

    # Passes through frozen-slot folding (InfLoRA only).
    def enable_frozen_folding(self):
        self.backbone.enable_frozen_folding()

    # Passes through freeing folded slots for a task (InfLoRA only).
    def free_folded_slots(self, task):
        self.backbone.free_folded_slots(task)

    # Passes through InfLoRA's input-covariance collection controls.
    def set_collect(self, flag):
        self.backbone.set_collect(flag)

    def reset_cur_matrix(self):
        self.backbone.reset_cur_matrix()

    def attn_modules(self):
        return self.backbone.attn_modules()

    # Passes through the LoRA-Subtraction probe-mode flag; a no-op for other backbones.
    def set_subtract_mode(self, flag):
        self.backbone.set_subtract_mode(flag)

    # Passes through SD-LoRA-Sketch's task-boundary fold.
    def fold(self):
        return self.backbone.fold()