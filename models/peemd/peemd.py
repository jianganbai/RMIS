import atexit
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .core import (
    _extract_one_worker,
    _limit_blas_threads_to_one,
    imf_stats,
    peemd_style_decompose,
)


# RMIS adapter for PEEMD. The signal decomposition/statistics implementation
# lives in core.py; this file keeps the benchmark-facing embedding contract.
class PEEMDModel(nn.Module):
    """
    Interface unchanged:
      forward(x, valid_len=None, label=None, out_emb=False, **kwargs) -> dict

    NUM_PROC is now an init parameter:
      - num_proc <= 1  -> serial feature extraction
      - num_proc >  1  -> parallelize within batch using ProcessPoolExecutor

    IMPORTANT:
      - If your DataLoader uses num_workers>0, set num_proc=1 to avoid nested processes.
      - If you set num_proc>1, also set BLAS/OMP threads to 1 (we do it once here).
    """

    def __init__(
        self,
        loss: Optional[nn.Module],
        emb_size: int,  # kept for compatibility; not used for projection by default
        k0: int = 8,
        stats_per_imf: int = 8,
        ensembles: int = 20,
        noise_std: float = 0.2,
        max_imfs: int = 8,
        max_sift: int = 50,
        tol: float = 0.05,
        pe_order: int = 3,
        pe_delay: int = 1,
        pe_th: float = 0.7,
        log_compress: bool = True,
        per_sample_zscore: bool = True,
        l2_normalize: bool = True,
        weight_ckpt: Optional[str] = None,
        num_proc: int = 1,     # <= 1: no MP; >1: batch-level multiprocessing
        base_seed: int = 0,    # seed offset for per-sample randomness (EEMD noise)
    ) -> None:
        super().__init__()
        self.loss = loss

        self.k0 = int(k0)
        self.stats_per_imf = int(stats_per_imf)
        assert self.stats_per_imf == 8, "imf_stats is fixed to 8 dims; change imf_stats if you want another size."
        self.feat_dim = self.k0 * self.stats_per_imf

        self.ensembles = int(ensembles)
        self.noise_std = float(noise_std)
        self.max_imfs = int(max_imfs)
        self.max_sift = int(max_sift)
        self.tol = float(tol)
        self.pe_order = int(pe_order)
        self.pe_delay = int(pe_delay)
        self.pe_th = float(pe_th)

        self.emb_size = int(emb_size)  # kept, unused by default

        self.log_compress = bool(log_compress)
        self.per_sample_zscore = bool(per_sample_zscore)
        self.l2_normalize = bool(l2_normalize)

        self.num_proc = int(num_proc)
        self.base_seed = int(base_seed)

        # config bundle for multiprocessing worker
        self._mp_cfg = dict(
            ensembles=self.ensembles,
            noise_std=self.noise_std,
            max_imfs=self.max_imfs,
            max_sift=self.max_sift,
            tol=self.tol,
            pe_order=self.pe_order,
            pe_delay=self.pe_delay,
            pe_th=self.pe_th,
            k0=self.k0,
            stats_per_imf=self.stats_per_imf,
        )

        # pool is created lazily (only if needed)
        self._pool: Optional[ProcessPoolExecutor] = None
        atexit.register(self.close)

        if weight_ckpt is not None:
            weights = torch.load(weight_ckpt, weights_only=True, map_location="cpu")
            missing_unexpected = self.load_state_dict(weights, strict=False)
            print(missing_unexpected)

    def _ensure_pool(self) -> Optional[ProcessPoolExecutor]:
        if self.num_proc <= 1:
            return None
        if self._pool is not None:
            return self._pool
        _limit_blas_threads_to_one()
        self._pool = ProcessPoolExecutor(max_workers=self.num_proc)
        return self._pool

    def close(self) -> None:
        """Optional: manually close process pool to free resources early."""
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        self._pool = None

    def __del__(self):
        # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _get_wav(self, x: torch.Tensor, valid_len: Optional[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.ndim != 2:
            raise ValueError(f"PEEMDModel expects waveform x with shape (B,N) or (N,), got {tuple(x.shape)}")

        if valid_len is None:
            return x

        if not torch.is_tensor(valid_len):
            valid_len = torch.tensor(valid_len, device=x.device)
        else:
            valid_len = valid_len.to(device=x.device)

        if valid_len.ndim != 1 or valid_len.numel() != x.size(0):
            raise ValueError(f"valid_len shape mismatch: expected (B,), got {tuple(valid_len.shape)}")

        wav_list: List[torch.Tensor] = []
        for i in range(x.size(0)):
            L = int(valid_len[i].item())
            L = max(0, min(L, x.size(1)))
            wav_list.append(x[i, :L])
        return wav_list

    @torch.no_grad()
    def _extract_one(self, wav_1d: np.ndarray, seed: int) -> np.ndarray:
        imfs = peemd_style_decompose(
            wav_1d,
            ensembles=self.ensembles,
            noise_std=self.noise_std,
            max_imfs=self.max_imfs,
            max_sift=self.max_sift,
            tol=self.tol,
            pe_order=self.pe_order,
            pe_delay=self.pe_delay,
            pe_th=self.pe_th,
            seed=seed,
        )
        out = []
        for i in range(self.k0):
            out.append(imf_stats(imfs[i]) if i < len(imfs) else np.zeros(self.stats_per_imf, dtype=np.float32))
        return np.concatenate(out, axis=0).astype(np.float32)

    @torch.no_grad()
    def _batch_extract(self, wav: Any) -> torch.Tensor:
        """
        wav can be:
          - Tensor(B,N)
          - list[Tensor] variable lengths (cropped by valid_len)

        Multiprocessing behavior:
          - num_proc <= 1: serial
          - num_proc > 1 : parallel within batch using self._pool
        """
        if isinstance(wav, list):
            device = wav[0].device
            wav_np_list = [w.detach().float().cpu().numpy() for w in wav]
        else:
            device = wav.device
            wav_np = wav.detach().float().cpu().numpy()
            wav_np_list = [wav_np[i] for i in range(wav_np.shape[0])]

        # serial
        if self.num_proc <= 1 or len(wav_np_list) <= 1:
            feats = [self._extract_one(w, seed=self.base_seed + i) for i, w in enumerate(wav_np_list)]
            feats = np.stack(feats, axis=0)
            return torch.from_numpy(feats).to(device=device)

        # multiprocessing
        pool = self._ensure_pool()
        if pool is None:
            feats = [self._extract_one(w, seed=self.base_seed + i) for i, w in enumerate(wav_np_list)]
            feats = np.stack(feats, axis=0)
            return torch.from_numpy(feats).to(device=device)

        tasks = [(w, self.base_seed + i, self._mp_cfg) for i, w in enumerate(wav_np_list)]
        feats = list(pool.map(_extract_one_worker, tasks))
        feats = np.stack(feats, axis=0)
        return torch.from_numpy(feats).to(device=device)

    def _normalize_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 2:
            raise ValueError(f"Expected feat (B,D), got {tuple(feat.shape)}")
        if feat.size(1) != self.feat_dim:
            raise ValueError(f"feat dim mismatch: got {feat.size(1)}, expect {self.feat_dim}")

        out = feat

        # per IMF stats: [energy, rms, meanabs, std, skew, kurt, zcr, maxabs]
        if self.log_compress:
            out = out.clone()
            per_imf = self.stats_per_imf  # 8
            log_ids = [0, 1, 2, 3, 7]
            idx = []
            for i in range(self.k0):
                base = i * per_imf
                idx.extend([base + j for j in log_ids])
            idx = torch.tensor(idx, device=out.device, dtype=torch.long)
            out[:, idx] = torch.log1p(torch.clamp(out[:, idx], min=0.0))

        if self.per_sample_zscore:
            m = out.mean(dim=1, keepdim=True)
            s = out.std(dim=1, keepdim=True)
            out = (out - m) / (s + 1e-6)

        if self.l2_normalize:
            out = out / (out.norm(dim=1, keepdim=True) + 1e-6)

        return out

    def forward(
        self,
        x: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        out_emb: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        wav = self._get_wav(x, valid_len)
        feat = self._batch_extract(wav)     # (B, feat_dim)
        emb = self._normalize_feat(feat)    # cosine-kNN friendly

        if out_emb or self.loss is None:
            return {"embedding": emb}

        return self.loss(emb, label)
