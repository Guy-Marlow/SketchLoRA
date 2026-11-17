"""Metrics logger wrapping trainer.py's task loop to record timing, peak memory, persistent
storage, FLOPs/latency and boundary overhead, and to derive FAA/AIA/forgetting/BWT from the same
accuracy matrices trainer.py assembles. Writes incrementally to a per-run JSON file.
"""

import io
import json
import logging
import os
import time

import torch


def measure_inference_flops(learner, sample_input):
    """Return the total FLOPs for one forward pass of learner._deployed_forward(sample_input),
    or None if unavailable or profiling fails. Measured via PyTorch's profiler (with_flops=True)
    on the method's own deployed forward call, counting whatever ops actually run rather than
    applying a per-architecture formula."""
    if not hasattr(learner, "_deployed_forward"):
        logging.warning("[flops] {} has no _deployed_forward; skipping FLOPs "
                        "measurement".format(type(learner).__name__))
        return None
    learner._network.eval()
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad(), torch.profiler.profile(
                activities=activities, with_flops=True) as prof:
            learner._deployed_forward(sample_input)
        total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops)
        return int(total_flops) if total_flops > 0 else None
    except Exception as e:
        logging.warning("[flops] measurement failed ({}); skipping".format(e))
        return None


def default_persistent_state(learner):
    """Fallback persistent_state heuristic: total bytes of every parameter whose name does not
    look like it belongs to the frozen backbone."""
    net = learner._network.module if hasattr(learner._network, "module") else learner._network
    backbone_markers = ("patch_embed", "cls_token", "pos_embed", "norm.", "q_proj.weight",
                        "k_proj.weight", "v_proj.weight", "proj.weight", "proj.bias",
                        "fc1.", "fc2.")
    total_params, total_bytes = 0, 0
    for name, p in net.named_parameters():
        is_backbone = any(m in name for m in backbone_markers) and "lora" not in name.lower()
        if is_backbone:
            continue
        total_params += p.numel()
        total_bytes += p.numel() * p.element_size()
    return {"params": total_params, "bytes": total_bytes,
            "breakdown": {"heuristic_non_backbone": total_bytes}}


def safe_persistent_state(learner):
    if hasattr(learner, "persistent_state"):
        try:
            return learner.persistent_state()
        except Exception as e:
            logging.warning("persistent_state() failed ({}); falling back to heuristic".format(e))
    return default_persistent_state(learner)


def serialized_bytes(state_dict_like):
    """Return the torch.save-serialized byte size of a state dict or tensor collection."""
    buf = io.BytesIO()
    torch.save(state_dict_like, buf)
    return buf.tell()


def save_adapter_snapshot(out_dir, fname, layer_proj_to_BA, **meta):
    """Save a dict of {(block_idx, proj): (B_tensor, A_tensor)} LoRA factors, detached and moved
    to CPU, as a single torch.save payload -- a boundary-time snapshot of each wrapped (block,
    projection) module's current LoRA factors, so the dense update can be reconstructed offline
    without re-running training."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    payload = {
        "B": {"{}_{}".format(li, p): B.detach().cpu() for (li, p), (B, A) in layer_proj_to_BA.items()},
        "A": {"{}_{}".format(li, p): A.detach().cpu() for (li, p), (B, A) in layer_proj_to_BA.items()},
        "meta": meta,
    }
    torch.save(payload, path)
    one_shape = next(iter(payload["B"].values())).shape
    logging.info("[adapter_snapshot] wrote {} ({} modules, e.g. B{} )".format(
        path, len(payload["B"]), tuple(one_shape)))


@torch.no_grad()
def effective_rank(delta, energy_target=0.99):
    """Return (r_eff, full_rank, singular_values) for a dense [out, in] delta, where r_eff is
    the fewest singular values whose squared sum covers energy_target of the total energy."""
    S = torch.linalg.svdvals(delta.double())
    energy = S ** 2
    total = energy.sum().clamp_min(1e-12)
    cum_ratio = torch.cumsum(energy, dim=0) / total
    r_eff = int((cum_ratio < energy_target).sum().item()) + 1
    r_eff = min(r_eff, S.shape[0])
    return r_eff, S.shape[0], S


@torch.no_grad()
def log_bank_effective_rank(tag, cur_task, deltas_by_block_proj, energy_target=0.99):
    """Effective-rank diagnostic for growing-bank LoRA methods: compute and log the effective
    rank of each (block, projection) summed-bank delta plus a mean/min/max summary, and return
    the per-key rank breakdown. Opt-in via log_bank_rank, since it costs a full SVD per
    (block, projection) per task."""
    full_rank = None
    ranks = {}
    for key, delta in deltas_by_block_proj.items():
        r_eff, r_full, _ = effective_rank(delta, energy_target)
        ranks[key] = r_eff
        full_rank = r_full
    vals = list(ranks.values())
    mean_r = sum(vals) / len(vals)
    per_block_str = ", ".join(
        "{}.{}={}".format(li, proj, r) for (li, proj), r in sorted(ranks.items()))
    logging.info(
        "[{}] task {} summed-bank effective rank ({:.0f}% energy): "
        "mean={:.2f} min={} max={} (full_rank={}) | {}".format(
            tag, cur_task, energy_target * 100, mean_r, min(vals), max(vals),
            full_rank, per_block_str))
    return ranks


def compute_cl_summary(matrix, last_task_idx):
    """Compute FAA, AIA, forgetting, BWT and last-task accuracy from a list of (task_idx, row)
    checkpoint accuracy rows, using each task's first-recorded accuracy for BWT so sparse eval
    cadences (e.g. every-5-tasks) are handled the same as dense ones."""
    import numpy as np
    n = last_task_idx + 1
    acc = np.zeros((n, n))
    for task_idx, row in matrix:
        acc[task_idx, :len(row)] = np.array(row)
    acc = acc.T  # [task_bucket, checkpoint_task_idx]: acc[j, i] = accuracy of task j at checkpoint i

    faa = float(np.mean(acc[:, -1]))                                    # final checkpoint's row
    aia = float(np.mean([np.mean(row) for _, row in matrix]))           # avg over CHECKPOINTS only
    last_task_acc = float(acc[-1, -1])
    if n > 1 and len(matrix) > 1:
        forgetting = float(np.mean((np.max(acc, axis=1) - acc[:, -1])[:-1]))
        checkpoint_idxs = sorted(idx for idx, _ in matrix)
        first_seen = np.full(n, np.nan)
        for j in range(n - 1):
            for ci in checkpoint_idxs:
                if ci >= j:
                    first_seen[j] = acc[j, ci]
                    break
        valid = ~np.isnan(first_seen[:-1])
        bwt = float(np.mean(acc[:-1, -1][valid] - first_seen[:-1][valid])) if valid.any() else None
    else:
        forgetting = None
        bwt = None
    return {"final_average_accuracy": round(faa, 4),
            "average_incremental_accuracy": round(aia, 4),
            "forgetting": round(forgetting, 4) if forgetting is not None else None,
            "backward_transfer": round(bwt, 4) if bwt is not None else None,
            "last_task_accuracy": round(last_task_acc, 4)}


class MetricsLogger:
    def __init__(self, out_dir, tag, args):
        os.makedirs(out_dir, exist_ok=True)
        self.out_path = os.path.join(out_dir, "metrics_{}.json".format(tag))
        keep = ("model_name", "dataset", "init_cls", "increment", "seed", "scenario",
                "boundary_mode", "n_lora_blocks", "lora_rank",
                "lora_alpha", "svd_energy_target", "svd_rank", "svd_period", "merge_op",
                "cs_rank", "lamda_1", "lamda_2", "lamb", "lame")
        self.meta = {
            "args_subset": {k: args.get(k) for k in keep},
            "status": "running",
            "per_task": [],
            "train_seconds_total": 0.0,
            "eval_seconds_total": 0.0,
            "boundary_seconds_total": 0.0,
        }
        # args["device"][0] is a concrete torch.device set by trainer.py before
        # MetricsLogger is constructed, so memory stats are reset and queried
        # on this run's own device rather than the process default.
        self._device = args.get("device", [None])[0]
        self._device_stats_ok = False
        if torch.cuda.is_available() and self._device is not None and self._device.type == "cuda":
            try:
                # a trivial allocation forces this device's CUDA context to exist
                # before reset_peak_memory_stats is allowed to query it
                torch.zeros(1, device=self._device)
                torch.cuda.reset_peak_memory_stats(self._device)
                self._device_stats_ok = True
            except Exception as e:
                logging.warning("[metrics] could not init per-device memory stats "
                                "for {} ({}); peak_vram_mb will be null".format(self._device, e))
        self._write()
        self._t_task_start = None
        self._t_train_end = None

    def _write(self):
        with open(self.out_path, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)

    # ---- per-task timing hooks (call around the existing trainer.py loop body) ----
    def begin_task(self):
        self._t_task_start = time.time()
        self._t_train_end = None

    def mark_train_done(self):
        """Call immediately after incremental_train() returns, before eval_task()."""
        self._t_train_end = time.time()

    def record_task(self, learner, task_idx, cnn_accy, boundary_seconds=0.0):
        """Call once per task/chunk, right after eval_task() + after_task(). Returns the
        record it wrote (also appended to self.meta['per_task'])."""
        now = time.time()
        train_s = (self._t_train_end - self._t_task_start) if self._t_train_end else None
        eval_s = (now - self._t_train_end) if self._t_train_end else None
        pstate = safe_persistent_state(learner)
        peak_mb = (round(torch.cuda.max_memory_allocated(self._device) / 1024 / 1024, 1)
                  if self._device_stats_ok else None)
        net = learner._network.module if hasattr(learner._network, "module") else learner._network
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        rec = {
            "task": task_idx,
            "train_seconds": round(train_s, 3) if train_s is not None else None,
            "eval_seconds": round(eval_s, 3) if eval_s is not None else None,
            "boundary_seconds": round(boundary_seconds, 4),
            "cil_top1": cnn_accy["top1"] if cnn_accy else None,
            "cil_top5": cnn_accy["top5"] if cnn_accy else None,
            # trainable = currently-unfrozen params this task, distinct from persistent below
            "trainable_params": int(trainable_params),
            "persistent_params": pstate["params"],
            "persistent_mb": round(pstate["bytes"] / 1024 / 1024, 4),
            "persistent_state_breakdown": pstate.get("breakdown"),
            "peak_vram_mb": peak_mb,
        }
        self.meta["per_task"].append(rec)
        if train_s is not None:
            self.meta["train_seconds_total"] += train_s
        if eval_s is not None:
            self.meta["eval_seconds_total"] += eval_s
        self.meta["boundary_seconds_total"] += boundary_seconds
        self._write()
        return rec

    # ---- inference cost (call once, after the final task) ----
    def record_inference_cost(self, learner, test_loader, n_batches=10):
        """Measure latency over up to n_batches of the deployed eval path plus FLOPs/image,
        skipping the metric if the learner has no _deployed_forward."""
        if not hasattr(learner, "_deployed_forward"):
            logging.warning("[metrics] {} has no _deployed_forward; skipping inference-cost "
                            "measurement".format(type(learner).__name__))
            return
        net = learner._network
        net.eval()
        device = learner._device
        times = []
        n_images = 0
        sample_batch = None
        try:
            with torch.no_grad():
                for i, (_, inputs, _t) in enumerate(test_loader):
                    if i >= n_batches:
                        break
                    inputs = inputs.to(device)
                    if sample_batch is None:
                        sample_batch = inputs
                    if device != "cpu" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.time()
                    learner._deployed_forward(inputs)
                    if device != "cpu" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - t0)
                    n_images += inputs.shape[0]
        except Exception as e:
            logging.warning("[metrics] inference-cost measurement failed ({}); skipping".format(e))
            return
        ms_per_image = (sum(times) / max(n_images, 1)) * 1000.0 if times else None
        flops_total = measure_inference_flops(learner, sample_batch) if sample_batch is not None else None
        flops_per_image = (flops_total / sample_batch.shape[0]) if flops_total is not None else None
        self.meta["inference_ms_per_image"] = round(ms_per_image, 4) if ms_per_image else None
        self.meta["inference_flops_per_image"] = round(flops_per_image, 1) if flops_per_image else None
        self._write()

    # ---- finalize (call once, after the loop ends) ----
    def finalize(self, cnn_matrix, last_task_idx):
        if cnn_matrix:
            self.meta["cil_summary"] = compute_cl_summary(cnn_matrix, last_task_idx)
        self.meta["status"] = "done"
        self._write()
        logging.info("[metrics] wrote {}".format(self.out_path))
