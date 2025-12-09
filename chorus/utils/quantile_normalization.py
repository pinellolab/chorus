import numpy as np 
import os 
import json 
import logging 

logger = logging.getLogger(__name__)

def non_zero_sort(distr: dict[str, np.memmap], outdir: str, cut_nz_q: float | None = 0.999,) -> dict[str, str]:
    out_distr = {}
    for k, d in distr.items():
        v = d[d>0]
        if cut_nz_q is not None:
            q = np.quantile(v, cut_nz_q)
            v[v > q] = q 

        v = np.sort(v)
        outpath = os.path.join(outdir, os.path.basename(d.filename))
        out = np.memmap(outpath,
                        mode="w+",
                        dtype=np.float32,
                        shape=v.shape)
        out[:] = v
        out_distr[k] = os.path.realpath(outpath)
    with open(os.path.join(outdir, 'info.json'), 'w') as outp:
        json.dump(out_distr, outp, indent=4)
    return out_distr

def quantile_unimap(arr: np.ndarray, points_cnt: int = 10_000_000 + 1) -> np.ndarray:
    quantiles_cnt = np.linspace(0.0, 1.0, num=points_cnt)
    arr_s = np.sort(arr)
    float_indices = quantiles_cnt * (arr.shape[0] - 1)
    lower_indices = np.asarray(np.floor(float_indices), dtype=np.int64)
    upper_indices = np.asarray(np.ceil(float_indices), dtype=np.int64)
    
    weights = float_indices - lower_indices
    weights[float_indices == upper_indices] = 1 
    quantiles = arr_s[lower_indices] * (1-weights) + arr_s[upper_indices] * weights
    return quantiles

def build_support_distr(distr: dict[str, np.ndarray], points_cnt: int) -> np.ndarray:
    support = 0
    for d in distr.values():
        unimap = quantile_unimap(d, points_cnt=points_cnt)
        support += unimap
    support /= len(distr)

    return support

def quantile_map(values: np.ndarray, initial_distr: np.ndarray, support_distr: np.ndarray) -> np.ndarray:
    N = support_distr.shape[0] - 1
    M = initial_distr.shape[0] - 1

    initial_distr_sorted = np.sort(initial_distr)

    left_rank = np.searchsorted(initial_distr_sorted, values, side='left')
    right_rank = np.searchsorted(initial_distr_sorted, values, side='right')

    left_point = np.asarray(np.floor((left_rank / M) * N ), dtype=np.int64)
    right_point = np.asarray(np.ceil((right_rank / M) * N ), dtype=np.int64)
    left_point[left_point > N] = N 
    right_point[right_point > N] = N

    mapped_values = (support_distr[left_point] + support_distr[right_point]) / 2

    return mapped_values

def quantile_map_singlev(v: np.float32, initial_distr: np.ndarray, support_distr: np.ndarray) -> np.ndarray:
    N = support_distr.shape[0] - 1
    M = initial_distr.shape[0] - 1
    if np.isclose(v, 0):
        return v

    left_rank = np.searchsorted(initial_distr, v, side='left')
    right_rank = np.searchsorted(initial_distr, v, side='right')

    left_point = np.int64(np.floor((left_rank / M) * N ))
    right_point = np.int64(np.ceil((right_rank / M) * N ))
    if left_point > N:
        left_point = N
    if right_point > N:
        right_point = N

    return (support_distr[left_point] + support_distr[right_point]) / 2