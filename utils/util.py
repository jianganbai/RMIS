import torch
import random
import numpy as np

from scipy.stats import mode


def ensemble_embs(embs: np.ndarray, ensem_mode: str) -> np.ndarray:
    if ensem_mode == 'sum':
        r = np.sum(embs, axis=0)
    elif ensem_mode == 'mean':
        r = np.mean(embs, axis=0)
    elif ensem_mode == 'std':
        r = np.std(embs, axis=0)
    elif ensem_mode == 'mean_std':
        r = np.concatenate([np.mean(embs, axis=0), np.std(embs, axis=0)])
    elif ensem_mode == 'max':
        r = np.max(embs, axis=0)
    elif ensem_mode == 'min':
        r = np.min(embs, axis=0)
    elif ensem_mode == 'raw':
        r = embs
    elif ensem_mode == 'hard':  # embs: [num_seg, num_class]
        val, count = mode(np.argmax(embs, axis=-1))
        r = np.array([val])
    elif ensem_mode == 'soft':  # embs: [num_seg, num_class]
        r = np.argmax(embs.sum(axis=0), keepdims=True)
    else:
        raise NotImplementedError(f'Aggregation {ensem_mode} not implemented!')
    return r


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)
