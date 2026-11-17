import sys
import logging
import copy
import torch
from torch.nn import functional as F
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.metrics_logger import MetricsLogger
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    # Epoch-offset consolidation: folds at a configurable epoch cadence/phase
    # (offset_k/offset_phi) instead of at every task boundary. See
    # models/stream_mixin.py::offset_consolidation_run's own docstring for
    # the mechanism.
    if args.get("boundary_mode") == "offset_consolidation":
        _run_stream(model, data_manager, args, method_name="offset_consolidation_run", tag="offset")
        return

    # Batch-incremental + KL-drift-detection setting: decouples adapter spawning
    # (a KL-drift detector on frozen-backbone features, or a fixed-interval
    # clock) from sketch consolidation (its own schedule/cap). kl_batch_inc_run
    # lives on the Learner itself, not a shared mixin, but _run_stream only
    # calls getattr(model, method_name)(...), so it works unchanged.
    if args.get("boundary_mode") == "kl_batch_inc":
        _run_stream(model, data_manager, args, method_name="kl_batch_inc_run", tag="kl_batch_inc")
        return

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    # stop_after_tasks: optionally train/eval only the first N tasks of the regime
    # (smoke/LR-probe runs). args["nb_tasks"] stays the full count so per-method
    # schedules (e.g. InfLoRA's DualGPM threshold ramp over total_sessions) see the
    # full regime; only the loop is shortened. None/absent = full run.
    _n_run = args.get("stop_after_tasks") or data_manager.nb_tasks
    _n_run = min(_n_run, data_manager.nb_tasks)
    if _n_run < data_manager.nb_tasks:
        logging.info("stop_after_tasks={} (of {} total)".format(_n_run, data_manager.nb_tasks))

    # Opt-in detailed metrics logging via "final_metrics": true in the config,
    # so every other track sharing this trainer.py (icarl/der/foster/aper_*/
    # ranpac/...) is unaffected. Additive only: writes its own JSON, changes
    # no existing printed/logged output.
    mlog = None
    if args.get("final_metrics"):
        _tag = "{}_{}_{}_s{}".format(
            args["model_name"], args["dataset"],
            args.get("prefix") or (args.get("boundary_mode") or "task"), args["seed"])
        mlog = MetricsLogger(os.path.join("run_logs", "final", args["model_name"]), _tag, args)

    for task in range(_n_run):
        if mlog:
            mlog.begin_task()
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        model.incremental_train(data_manager)

        if mlog:
            mlog.mark_train_done()

        # OmniBenchmark-1K (100 tasks) evaluates every 5th task (plus the final
        # one) to keep eval from dominating wall-clock; every other dataset
        # evaluates every task. cnn_matrix stores (task_idx, row) tuples so
        # downstream forgetting/BWT computation can index by true task rather
        # than row position.
        is_checkpoint = (
            ((task + 1) % 5 == 0 or task == _n_run - 1)
            if args["dataset"] == "omnibenchmark1k" else True
        )
        if is_checkpoint:
            cnn_accy, nme_accy = model.eval_task()
        else:
            cnn_accy, nme_accy = None, None

        model.after_task()
        if mlog:
            mlog.record_task(model, task, cnn_accy)

        if not is_checkpoint:
            continue

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append((task, cnn_values))

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append((task, cnn_values))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if mlog:
        test_loader = getattr(model, "test_loader", None)
        if test_loader is not None:
            mlog.record_inference_cost(model, test_loader)
        mlog.finalize(cnn_matrix, task)

    if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in cnn_matrix:  # (task_idx, row): true index, not position
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np.array2string(np_acctable, max_line_width=100000, precision=2, suppress_small=True))
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np.array2string(np_acctable, max_line_width=100000, precision=2, suppress_small=True))
        logging.info('Forgetting (NME): {}'.format(forgetting))


def _run_stream(model, data_manager, args, method_name="offset_consolidation_run", tag="offset"):
    """Drive a boundary-decoupled streaming run and report the CIL curve over
    completed-task checkpoints. method_name/tag let this same reporting logic
    serve both offset_consolidation_run and kl_batch_inc_run without
    duplication."""
    results = getattr(model, method_name)(data_manager, args)
    comp = [r["completed"] for r in results]
    cil = [r["cil"] for r in results]
    logging.info("[{}] completed-task checkpoints: {}".format(tag, comp))
    logging.info("[{}] CIL curve: {}".format(tag, cil))
    print("[{}] checkpoints (completed tasks):".format(tag), comp)
    print("[{}] CIL:".format(tag), cil)
    logging.info("[{}] FINAL CIL {:.2f} (over {} tasks)".format(tag, cil[-1], comp[-1]))
    print("[{}] FINAL CIL {:.2f}".format(tag, cil[-1]))
    out = "run_logs/{}_{}_{}_s{}.json".format(
        tag, args["model_name"], args.get("prefix", "run"), args["seed"])
    os.makedirs("run_logs", exist_ok=True)
    with open(out, "w") as f:
        import json as _json
        _json.dump({"args_subset": {k: args.get(k) for k in
                    ("model_name", "dataset", "init_cls", "increment", "stream_budget_mb",
                     "n_lora_blocks", "init_lr", "svd_energy_target", "lamda_1", "lamb", "lame",
                     "offset_k", "offset_phi")},
                    "results": results, "partial": False}, f, indent=2)
    print("[{}] wrote".format(tag), out)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))