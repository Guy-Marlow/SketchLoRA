"""Online, task-boundary-free fold/spawn signal detector for SketchLoRA's batch-incremental,
KL-divergence-drift streaming setting.
"""
import math
from collections import deque

import torch


class KLDriftDetector:
    """Fold/spawn trigger that compares a recent ring-buffer window of frozen-backbone CLS
    features against a lagged running reference via diagonal-Gaussian KL divergence, firing once
    the self-calibrated z-score of the KL value stays above a threshold for several steps."""

    def __init__(self, window_batches, warmup_multiplier=2, tau=4.0, consecutive=3,
                 var_eps=1e-4, max_chunk_samples=None):
        assert window_batches >= 4, \
            "window_batches must be at least 4 -- got {}".format(window_batches)
        self.window_batches = window_batches
        self.warmup_steps = warmup_multiplier * window_batches
        self.tau = tau
        self.consecutive = consecutive
        self.var_eps = var_eps
        self.max_chunk_samples = max_chunk_samples

        self._recent_buf = deque(maxlen=window_batches)
        self._ref_n = None
        self._ref_mean = None
        self._ref_M2 = None
        self._kl_calib_n = None
        self._kl_calib_mean = None
        self._kl_calib_M2 = None
        self._consec = 0
        self._step_count = 0
        self.images_since_fire = 0
        self.last_kl = None
        self.last_z = None
        self.last_fire_reason = None

    def reset(self):
        """Clear detector state after spawning a new adapter; the caller calls this, not the
        detector itself, and the same warmup gate then doubles as the refractory period."""
        self._recent_buf.clear()
        self._ref_n = None
        self._ref_mean = None
        self._ref_M2 = None
        self._kl_calib_n = None
        self._kl_calib_mean = None
        self._kl_calib_M2 = None
        self._consec = 0
        self._step_count = 0
        self.images_since_fire = 0
        self.last_kl = None
        self.last_z = None

    def step(self, features, batch_size):
        """features: [B, D] pooled frozen-backbone CLS features for this batch, already
        detached. Returns True exactly on the step that should spawn a new adapter."""
        self.images_since_fire += batch_size
        if self.max_chunk_samples is not None and self.images_since_fire >= self.max_chunk_samples:
            self._fire("cap")
            return True

        self._step_count += 1
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0, unbiased=False)
        n = features.shape[0]

        # the entry about to be evicted feeds the reference before it is dropped, so the
        # reference lags the window instead of overlapping it
        if len(self._recent_buf) == self.window_batches:
            old_mean, old_var, old_n = self._recent_buf[0]
            self._update_reference(old_mean, old_var, old_n)
        self._recent_buf.append((batch_mean, batch_var, n))

        self.last_kl = None
        self.last_z = None
        if self._ref_n is None:
            return False

        recent_mean, recent_var, _ = self._pooled_recent_stats()
        ref_var = self._ref_M2 / self._ref_n
        kl = self._kl_diag_gaussian(recent_mean, recent_var, self._ref_mean, ref_var, self.var_eps)
        self.last_kl = kl

        z = None
        if self._kl_calib_n is not None and self._kl_calib_n >= 2:
            std = math.sqrt(max(self._kl_calib_M2 / self._kl_calib_n, 1e-12))
            z = (kl - self._kl_calib_mean) / max(std, 1e-6)
        self.last_z = z

        above = (z is not None) and (z >= self.tau)
        if above:
            self._consec += 1
        else:
            self._consec = 0
            self._update_kl_calib(kl)   # bootstrap (z is None) or normal (z < tau)

        warm = self._step_count >= self.warmup_steps
        if warm and above and self._consec >= self.consecutive:
            self._fire("kl")
            return True
        return False

    def _update_reference(self, mean, var, n):
        """Parallel Welford merge of one batch's own (mean, var, n) into the running reference."""
        m2 = var * n
        if self._ref_n is None:
            self._ref_n, self._ref_mean, self._ref_M2 = n, mean.clone(), m2.clone()
            return
        n_a, mean_a, M2_a = self._ref_n, self._ref_mean, self._ref_M2
        n_b, mean_b, M2_b = n, mean, m2
        n_ab = n_a + n_b
        delta = mean_b - mean_a
        mean_ab = mean_a + delta * (n_b / n_ab)
        M2_ab = M2_a + M2_b + (delta ** 2) * (n_a * n_b / n_ab)
        self._ref_n, self._ref_mean, self._ref_M2 = n_ab, mean_ab, M2_ab

    def _pooled_recent_stats(self):
        total_n = sum(n for _, _, n in self._recent_buf)
        pooled_mean = sum(mean * n for mean, _, n in self._recent_buf) / total_n
        pooled_var = sum(n * (var + (mean - pooled_mean) ** 2)
                          for mean, var, n in self._recent_buf) / total_n
        return pooled_mean, pooled_var, total_n

    def _update_kl_calib(self, kl):
        """Scalar Welford update of the running KL mean/std used for self-calibration."""
        if self._kl_calib_n is None:
            self._kl_calib_n, self._kl_calib_mean, self._kl_calib_M2 = 1, kl, 0.0
            return
        self._kl_calib_n += 1
        delta = kl - self._kl_calib_mean
        self._kl_calib_mean += delta / self._kl_calib_n
        delta2 = kl - self._kl_calib_mean
        self._kl_calib_M2 += delta * delta2

    @staticmethod
    def _kl_diag_gaussian(mean_p, var_p, mean_q, var_q, eps):
        """KL(P || Q) for diagonal Gaussians P=(mean_p,var_p), Q=(mean_q,var_q), summed over dims."""
        var_p = var_p.clamp_min(eps)
        var_q = var_q.clamp_min(eps)
        kl_per_dim = 0.5 * (var_p / var_q + (mean_q - mean_p) ** 2 / var_q - 1.0
                            + torch.log(var_q / var_p))
        return float(kl_per_dim.sum().item())

    def _fire(self, reason):
        self.last_fire_reason = reason
