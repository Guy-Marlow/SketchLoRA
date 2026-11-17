import json
import os
import subprocess
import sys

PY = sys.executable
SEEDS = [1993, 1996, 1999]
DATASETS = {
    "cifar224": (10, 10),
    "imagenetr": (10, 10),
    "sun397": (37, 40),
    "food101": (6, 5),
    "omnibenchmark1k": (10, 10),
}

METHODS = {
    "sketchlora": {
        "model_name": "sketchlora_align",
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 10, "svd_rank": 10, "lora_alpha": None,
        "lora_merge": True, "lora_train_merge": True,
        "svd_oversampling": 10, "lora_n_slots": 2, "sketch_diag": True,
        "sketchlora_lora_wd": 0.0, "sketchlora_align_mode": "none",
        "sketchlora_align_weight": 0.0, "merge_op": "qrsvd",
        "tuned_epoch": 20, "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "init_lr": {d: 0.001 for d in DATASETS},
        "diag": True,
    },
    "sketchlora_align": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 10, "svd_rank": 10, "lora_alpha": None,
        "lora_merge": True, "lora_train_merge": True,
        "svd_oversampling": 10, "lora_n_slots": 2, "sketch_diag": True,
        "sketchlora_lora_wd": 0.0, "sketchlora_align_mode": "orth",
        "sketchlora_align_weight": 0.5, "merge_op": "qrsvd",
        "tuned_epoch": 20, "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "init_lr": {d: 0.001 for d in DATASETS},
        "diag": True,
    },
    "lorasub_sketch": {
        "backbone_type": "vit_base_patch16_224_lorasub",
        "lora_rank": 10, "svd_rank": 10, "lora_alpha": None,
        "lora_n_slots": 2, "merge_op": "qrsvd", "svd_thres": 0.99,
        "use_proto_repulsion": True, "proto_repulsion_weight": 0.2, "proto_margin": 2.0,
        "tuned_epoch": 20, "batch_size": 48, "weight_decay": 0.0, "min_lr": 0.0,
        "init_lr": {"cifar224": 0.001, "imagenetr": 0.003, "sun397": 0.003,
                    "food101": 0.003, "omnibenchmark1k": 0.003},
        "diag": False,
    },
    "sdlora_sketch": {
        "backbone_type": "vit_base_patch16_224_sdlora_sketch",
        "lora_rank": 10, "lora_alpha": None, "svd_rank": 10,
        "sdsketch_reg_strength": 0.1, "sdsketch_reg_power": 1.0,
        "sdsketch_freeze_rescale": False,
        "tuned_epoch": 20, "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "init_lr": {"cifar224": 0.005, "imagenetr": 0.008, "sun397": 0.005,
                    "food101": 0.005, "omnibenchmark1k": 0.005},
        "diag": False,
    },
}

OUT_DIR = "exps/sketch_baselines"
os.makedirs(OUT_DIR, exist_ok=True)

for key, spec in METHODS.items():
    diag = spec["diag"]
    model_name = spec.get("model_name", key)
    for dataset, (init_cls, increment) in DATASETS.items():
        for seed in SEEDS:
            args = {k: v for k, v in spec.items() if k not in ("init_lr", "diag", "model_name")}
            args["init_lr"] = spec["init_lr"][dataset]
            args.update({
                "memory_size": 0, "memory_per_class": 0, "fixed_memory": False,
                "shuffle": True, "scenario": "cil", "pretrained": True,
                "print_forget": True, "final_metrics": True,
                "model_name": model_name, "dataset": dataset,
                "init_cls": init_cls, "increment": increment,
                "seed": [seed], "device": ["0"],
                "prefix": "sketch_baselines_{}_{}_s{}".format(key, dataset, seed),
            })
            if diag:
                args["sketchlora_diag_dir"] = "run_logs/sketch_baselines/{}/{}/diag".format(key, dataset)
            cfg_path = os.path.join(OUT_DIR, "{}_{}_s{}.json".format(key, dataset, seed))
            with open(cfg_path, "w") as f:
                json.dump(args, f, indent=2)
            subprocess.run([PY, "main.py", "--config", cfg_path], check=True)
