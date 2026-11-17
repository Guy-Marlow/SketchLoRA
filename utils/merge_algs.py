"""Merge algorithms for SketchLoRA's merge_op ablations: randomized SVD (randsvd/exactsvd,
plus the probe/truncate split used for adaptive-rank selection and random-rank keeping),
CountSketch hashing (countsketch), and two Frequent Directions variants -- decoupled
input/output-factor shrinkage (freqdir, "FD Separate") and dense composite-probe shrinkage
(freqdir_dense, "FD Sum").
"""

import torch
import numpy as np


def rand_svd(M: np.ndarray | torch.Tensor, target_rank: int, oversampling: int):
    """Compute a rank target_rank randomized SVD of M and return it as LoRA factors (B_hat, A_hat)."""
    if(isinstance(M, np.ndarray)):
        omega = np.random.randn(M.shape[1], target_rank + oversampling)
        Y = M @ omega
        Q = np.linalg.qr(Y).Q
        M_bar = np.transpose(Q) @ M

        U_bar, S, Vh = np.linalg.svd(M_bar)
        S = np.diag(S)
        U = (Q @ U_bar)[:, 0:target_rank]
        S_root = np.power(S, 0.5)[0:target_rank, 0:target_rank]

        B_hat = U @ S_root
        A_hat = S_root @ Vh[0:target_rank, :]

        return (B_hat, A_hat)
    else:
        M = M.cuda()

        omega = torch.randn(M.shape[1], target_rank + oversampling, device=M.device)
        Y = M @ omega
        Q, _ = torch.linalg.qr(Y)
        M_bar = Q.t() @ M
        try:
            U_bar, S, Vh = torch.linalg.svd(M_bar)
        except torch._C._LinAlgError:
            # fall back to the slower but more numerically robust classical driver
            U_bar, S, Vh = torch.linalg.svd(M_bar, driver="gesvd")

        S = torch.diag(S)
        U = (Q @ U_bar)[:, 0:target_rank]
        S_root = torch.pow(S, 0.5)[0:target_rank, 0:target_rank]

        B_hat = U @ S_root
        A_hat = S_root @ Vh[0:target_rank, :]

        return (B_hat, A_hat)


def rand_svd_probe(M: torch.Tensor, working_rank: int, oversampling: int):
    """Run the randomized SVD projection and return the full, untruncated (U, S, Vh) so the
    caller can pick a target rank from the probed spectrum before truncating."""
    M = M.cuda()
    omega = torch.randn(M.shape[1], working_rank + oversampling, device=M.device)
    Y = M @ omega
    Q, _ = torch.linalg.qr(Y)
    M_bar = Q.t() @ M
    try:
        U_bar, S, Vh = torch.linalg.svd(M_bar)
    except torch._C._LinAlgError:
        U_bar, S, Vh = torch.linalg.svd(M_bar, driver="gesvd")
    U = Q @ U_bar   # not truncated, caller picks the rank and slices
    return U, S, Vh


def factors_from_probe(U, S, Vh, r_hat):
    """Truncate a rand_svd_probe decomposition to r_hat directions and build the (B_hat, A_hat)
    LoRA factor pair."""
    root_S = S[:r_hat].sqrt()
    B_hat = U[:, :r_hat] * root_S.unsqueeze(0)
    A_hat = root_S.unsqueeze(1) * Vh[:r_hat, :]
    return B_hat, A_hat


def random_factors_from_probe(U, S, Vh, r_hat, composite_rank, seed):
    """Ablation counterpart to factors_from_probe for the "Random Select" baseline, keeping a
    random r_hat-sized subset of the first composite_rank directions instead of the top ones."""
    assert r_hat <= composite_rank, \
        "cannot keep {} random directions out of only {} meaningful ones".format(
            r_hat, composite_rank)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(composite_rank, generator=gen)
    keep = perm[:r_hat].to(U.device)
    root_S = S[keep].sqrt()
    B_hat = U[:, keep] * root_S.unsqueeze(0)
    A_hat = root_S.unsqueeze(1) * Vh[keep, :]
    return B_hat, A_hat


def countsketch_compress(B_list, A_list, cs_rank, seed):
    """Merge a list of (B, A) LoRA factor pairs into a single rank-k factor pair via CountSketch
    hashing, k = min(cs_rank, out_features, in_features)."""
    dev, dt = B_list[0].device, B_list[0].dtype
    F_B = torch.cat([b.float() for b in B_list], dim=1)   # [out, m]
    F_A = torch.cat([a.float() for a in A_list], dim=0)   # [m, in]

    out_features, in_features = F_B.shape[0], F_A.shape[1]
    k = min(cs_rank, out_features, in_features)

    nb, na = F_B.norm(dim=0), F_A.norm(dim=1)
    keep = (nb * na) > 0
    if not bool(keep.any()):
        return (torch.zeros(out_features, k, dtype=dt, device=dev),
                torch.zeros(k, in_features, dtype=dt, device=dev))

    F_B, F_A = F_B[:, keep], F_A[keep, :]
    nb, na = nb[keep], na[keep]
    # rebalance each column so its B and A norms match, product unchanged
    t = torch.sqrt(na / nb)
    F_B = F_B * t.unsqueeze(0)
    F_A = F_A / t.unsqueeze(1)
    m = F_B.shape[1]

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) % (2 ** 63 - 1))
    h = torch.randint(0, k, (m,), generator=g).to(F_B.device)
    sgn = (torch.randint(0, 2, (m,), generator=g) * 2 - 1).to(F_B.dtype).to(F_B.device)

    B_new = torch.zeros(out_features, k, dtype=F_B.dtype, device=F_B.device)
    A_new = torch.zeros(k, in_features, dtype=F_A.dtype, device=F_A.device)
    B_new.index_add_(1, h, F_B * sgn.unsqueeze(0))
    A_new.index_add_(0, h, F_A * sgn.unsqueeze(1))

    return B_new.to(dt).to(dev), A_new.to(dt).to(dev)


def freqdir_shrink(S):
    """Shrink a descending singular value spectrum by its smallest squared value, the Frequent
    Directions rule, without truncating to a target rank."""
    S_sq = S.pow(2)
    rent = S_sq[-1]
    return torch.clamp(S_sq - rent, min=0.0).sqrt()


def freqdir_input_factor(M_A, r_hat):
    """Compress the concatenated input-space factor matrix to a rank r_hat input factor via
    Frequent Directions shrinkage and truncation."""
    _, S, Vh = torch.linalg.svd(M_A, full_matrices=False)
    S_shrunk = freqdir_shrink(S)
    assert S_shrunk[r_hat - 1] > 0, (
        "freqdir: target rank r_hat={} falls inside the shrinkage-zeroed tail "
        "of the input spectrum (length {})".format(r_hat, S.numel()))
    root = S_shrunk[:r_hat].sqrt()
    return root.unsqueeze(1) * Vh[:r_hat, :]


def freqdir_output_factor(M_B, r_hat):
    """Compress the concatenated output-space factor matrix to a rank r_hat output factor via
    Frequent Directions shrinkage and truncation."""
    U, S, _ = torch.linalg.svd(M_B, full_matrices=False)
    S_shrunk = freqdir_shrink(S)
    assert S_shrunk[r_hat - 1] > 0, (
        "freqdir: target rank r_hat={} falls inside the shrinkage-zeroed tail "
        "of the output spectrum (length {})".format(r_hat, S.numel()))
    root = S_shrunk[:r_hat].sqrt()
    return U[:, :r_hat] * root.unsqueeze(0)


def fd_shrink_probe(S):
    """Shrink a descending, untruncated candidate singular value spectrum by its smallest
    squared value so a later truncation to any rank keeps the Frequent Directions error bound."""
    sq = S.pow(2)
    delta = sq.min()
    shrunk_sq = torch.clamp(sq - delta, min=0.0)
    S_shrunk = shrunk_sq.sqrt()
    stats = {
        "delta": delta.item(),
        "pre_shrink_energy": sq.sum().item(),
        "post_shrink_energy": shrunk_sq.sum().item(),
        "n_candidates": int(S.numel()),
    }
    return S_shrunk, stats
