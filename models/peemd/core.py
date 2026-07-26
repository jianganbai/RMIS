import math
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


# ============================================================
# Thread / process control helpers
# ============================================================
def _limit_blas_threads_to_one() -> None:
    """Avoid oversubscription when using multi-process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================================================
# Minimal PE + (E)MD utilities
# ============================================================
def permutation_entropy_1d(x: np.ndarray, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    """Permutation entropy: higher => more random/noise-like."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < (order - 1) * delay + 2:
        return 0.0

    m, k = order, delay
    n_emb = n - (m - 1) * k
    if n_emb <= 1:
        return 0.0

    emb = np.empty((n_emb, m), dtype=np.float64)
    for i in range(m):
        emb[:, i] = x[i * k : i * k + n_emb]

    patterns = np.argsort(emb, axis=1, kind="mergesort")
    counts: Dict[Tuple[int, ...], int] = {}
    for row in patterns:
        key = tuple(row.tolist())
        counts[key] = counts.get(key, 0) + 1

    p = np.array(list(counts.values()), dtype=np.float64)
    p /= p.sum()
    pe = -np.sum(p * np.log(p + 1e-12))
    if normalize:
        pe /= np.log(math.factorial(m) + 1e-12)
    return float(pe)


def find_extrema(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Discrete local maxima/minima indices and values."""
    dx1 = x[1:-1] - x[:-2]
    dx2 = x[1:-1] - x[2:]
    max_mask = (dx1 > 0) & (dx2 > 0)
    min_mask = (dx1 < 0) & (dx2 < 0)
    max_idx = np.where(max_mask)[0] + 1
    min_idx = np.where(min_mask)[0] + 1
    return max_idx, x[max_idx], min_idx, x[min_idx]


def interp_envelope(t: np.ndarray, idx: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Cubic spline envelope with simple endpoint padding."""
    n = t.size
    if idx.size < 2:
        return np.full(n, val[0] if val.size > 0 else 0.0, dtype=np.float64)

    idx2 = idx.copy()
    val2 = val.copy()
    if idx2[0] != 0:
        idx2 = np.insert(idx2, 0, 0)
        val2 = np.insert(val2, 0, val2[0])
    if idx2[-1] != n - 1:
        idx2 = np.append(idx2, n - 1)
        val2 = np.append(val2, val2[-1])

    cs = CubicSpline(t[idx2], val2, bc_type="natural")
    return cs(t)


def is_imf(h: np.ndarray, mean_env: np.ndarray, tol: float = 0.05) -> bool:
    """Baseline IMF stopping check."""
    zc = np.sum((h[:-1] * h[1:]) < 0)
    max_idx, _, min_idx, _ = find_extrema(h)
    ne = max_idx.size + min_idx.size
    if abs(ne - zc) > 1:
        return False
    denom = np.mean(np.abs(h)) + 1e-12
    ratio = np.mean(np.abs(mean_env)) / denom
    return ratio < tol


def emd_decompose(x: np.ndarray, max_imfs: int = 8, max_sift: int = 50, tol: float = 0.05) -> List[np.ndarray]:
    """Baseline EMD sifting producing up to max_imfs IMFs."""
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(x.size, dtype=np.float64)

    residue = x.copy()
    imfs: List[np.ndarray] = []

    for _ in range(max_imfs):
        h = residue.copy()
        max_idx, _, min_idx, _ = find_extrema(h)
        if (max_idx.size + min_idx.size) < 4:
            break

        for _ in range(max_sift):
            max_idx, max_val, min_idx, min_val = find_extrema(h)
            if (max_idx.size < 2) or (min_idx.size < 2):
                break
            upper = interp_envelope(t, max_idx, max_val)
            lower = interp_envelope(t, min_idx, min_val)
            mean_env = 0.5 * (upper + lower)
            h1 = h - mean_env
            if is_imf(h1, mean_env, tol=tol):
                h = h1
                break
            h = h1

        imfs.append(h)
        residue = residue - h

    return imfs


def eemd_decompose(
    x: np.ndarray,
    ensembles: int = 20,
    noise_std: float = 0.2,
    max_imfs: int = 8,
    max_sift: int = 50,
    tol: float = 0.05,
    seed: int = 0,
) -> List[np.ndarray]:
    """Baseline EEMD: repeat EMD with added noise and average IMFs by index."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    sig_std = np.std(x) + 1e-12
    nstd = noise_std * sig_std

    all_imfs: List[List[np.ndarray]] = []
    max_k = 0
    for _ in range(ensembles):
        xn = x + rng.normal(0.0, nstd, size=x.shape)
        imfs = emd_decompose(xn, max_imfs=max_imfs, max_sift=max_sift, tol=tol)
        all_imfs.append(imfs)
        max_k = max(max_k, len(imfs))

    out: List[np.ndarray] = []
    for k in range(max_k):
        stack = []
        for imfs in all_imfs:
            stack.append(imfs[k] if k < len(imfs) else np.zeros_like(x))
        out.append(np.mean(np.stack(stack, axis=0), axis=0))
    return out


def peemd_style_decompose(
    x: np.ndarray,
    ensembles: int = 20,
    noise_std: float = 0.2,
    max_imfs: int = 8,
    max_sift: int = 50,
    tol: float = 0.05,
    pe_order: int = 3,
    pe_delay: int = 1,
    pe_th: float = 0.7,
    seed: int = 0,
) -> List[np.ndarray]:
    """
    True "partial ensemble" (faster than "EEMD(x) then select"):
      1) EMD(x) -> IMFs_emd
      2) Select noisy IMFs by permutation entropy >= pe_th
      3) If none noisy: return IMFs_emd
      4) noisy_component = sum(noisy IMFs)
      5) EEMD(noisy_component) with max_imfs reduced to (#noisy IMFs)
      6) Replace those noisy IMF slots; keep order
    """
    x = np.asarray(x, dtype=np.float64)

    imfs_emd = emd_decompose(x, max_imfs=max_imfs, max_sift=max_sift, tol=tol)
    if len(imfs_emd) == 0:
        return []

    pe_vals = [permutation_entropy_1d(imf, order=pe_order, delay=pe_delay, normalize=True) for imf in imfs_emd]
    noisy_idx = [i for i, v in enumerate(pe_vals) if v >= pe_th]

    if len(noisy_idx) == 0:
        return imfs_emd

    noisy_component = np.sum(np.stack([imfs_emd[i] for i in noisy_idx], axis=0), axis=0)
    max_imfs_noisy = min(max_imfs, max(1, len(noisy_idx)))

    imfs_noisy = eemd_decompose(
        noisy_component,
        ensembles=ensembles,
        noise_std=noise_std,
        max_imfs=max_imfs_noisy,
        max_sift=max_sift,
        tol=tol,
        seed=seed,
    )

    noisy_set = set(noisy_idx)
    out_imfs: List[np.ndarray] = []
    p = 0
    for i in range(len(imfs_emd)):
        if i in noisy_set:
            if p < len(imfs_noisy):
                out_imfs.append(imfs_noisy[p])
                p += 1
            else:
                out_imfs.append(np.zeros_like(x))
        else:
            out_imfs.append(imfs_emd[i])

    return out_imfs


def imf_stats(imf: np.ndarray) -> np.ndarray:
    """
    Per-IMF stats (8 dims):
      [energy, rms, meanabs, std, skew, kurt, zcr, maxabs]
    """
    x = imf.astype(np.float64)
    eps = 1e-12
    energy = np.mean(x * x)
    rms = np.sqrt(energy + eps)
    meanabs = np.mean(np.abs(x))
    std = np.std(x) + eps
    m3 = np.mean(((x - np.mean(x)) / std) ** 3)
    m4 = np.mean(((x - np.mean(x)) / std) ** 4)
    zcr = np.mean((x[:-1] * x[1:]) < 0) if x.size > 1 else 0.0
    maxabs = np.max(np.abs(x)) if x.size > 0 else 0.0
    return np.array([energy, rms, meanabs, std, m3, m4, zcr, maxabs], dtype=np.float32)


# ============================================================
# Multiprocessing worker (top-level, picklable)
# ============================================================
def _extract_one_worker(args) -> np.ndarray:
    wav_1d, seed, cfg = args
    imfs = peemd_style_decompose(
        wav_1d,
        ensembles=cfg["ensembles"],
        noise_std=cfg["noise_std"],
        max_imfs=cfg["max_imfs"],
        max_sift=cfg["max_sift"],
        tol=cfg["tol"],
        pe_order=cfg["pe_order"],
        pe_delay=cfg["pe_delay"],
        pe_th=cfg["pe_th"],
        seed=seed,
    )
    out = []
    k0 = cfg["k0"]
    stats_per_imf = cfg["stats_per_imf"]
    for i in range(k0):
        out.append(imf_stats(imfs[i]) if i < len(imfs) else np.zeros(stats_per_imf, dtype=np.float32))
    return np.concatenate(out, axis=0).astype(np.float32)
