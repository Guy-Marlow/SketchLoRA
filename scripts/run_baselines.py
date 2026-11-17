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
    "seqlora": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 20, "lora_alpha": None, "lora_merge": False,
        "tuned_epoch": 20, "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "init_lr": {"cifar224": 0.0003, "imagenetr": 0.001, "sun397": 0.0003,
                    "food101": 0.0003, "omnibenchmark1k": 0.0003},
    },
    "olora": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 10, "lora_alpha": None, "lora_merge": True,
        "lamda_1": 0.5, "lamda_2": 0.0,
        "log_bank_rank": True, "bank_rank_energy_target": 0.99,
        "tuned_epoch": 20, "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "init_lr": {d: 0.001 for d in DATASETS},
    },
    "lorasub": {
        "backbone_type": "vit_base_patch16_224_lorasub",
        "lora_rank": 10, "lora_alpha": None,
        "svd_thres": 0.99, "use_proto_repulsion": True,
        "proto_repulsion_weight": 0.2, "proto_margin": 2.0,
        "log_bank_rank": True, "bank_rank_energy_target": 0.99,
        "tuned_epoch": 20, "batch_size": 48, "weight_decay": 0.0, "min_lr": 0.0,
        "init_lr": {"cifar224": 0.001, "imagenetr": 0.003, "sun397": 0.003,
                    "food101": 0.003, "omnibenchmark1k": 0.003},
    },
    "sdlora": {
        "backbone_type": "vit_base_patch16_224_sdlora",
        "lora_rank": 10, "lora_alpha": None, "lora_merge": True, "lora_train_merge": True,
        "log_bank_rank": True, "bank_rank_energy_target": 0.99,
        "tuned_epoch": 20, "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "init_lr": {"cifar224": 0.005, "imagenetr": 0.008, "sun397": 0.005,
                    "food101": 0.005, "omnibenchmark1k": 0.005},
    },
}

OUT_DIR = "exps/baselines"
os.makedirs(OUT_DIR, exist_ok=True)

for model_name, spec in METHODS.items():
    for dataset, (init_cls, increment) in DATASETS.items():
        for seed in SEEDS:
            args = {k: v for k, v in spec.items() if k != "init_lr"}
            args["init_lr"] = spec["init_lr"][dataset]
            args.update({
                "memory_size": 0, "memory_per_class": 0, "fixed_memory": False,
                "shuffle": True, "scenario": "cil", "pretrained": True,
                "print_forget": True, "final_metrics": True,
                "model_name": model_name, "dataset": dataset,
                "init_cls": init_cls, "increment": increment,
                "seed": [seed], "device": ["0"],
                "prefix": "baselines_{}_{}_s{}".format(model_name, dataset, seed),
            })
            cfg_path = os.path.join(OUT_DIR, "{}_{}_s{}.json".format(model_name, dataset, seed))
            with open(cfg_path, "w") as f:
                json.dump(args, f, indent=2)
            subprocess.run([PY, "main.py", "--config", cfg_path], check=True)
