import numpy as np
import tacuda
from utils import bench


def cpu_wma(x, w):
    """Weighted moving average with linear weights."""
    weights = np.arange(1, w + 1, dtype=x.dtype)
    y = np.full_like(x, np.nan)
    conv = np.convolve(x, weights / weights.sum(), mode="valid")
    y[w - 1 :] = conv
    return y


if __name__ == "__main__":
    n = 1_000_000
    x = np.random.rand(n).astype(np.float32)
    for w in [5, 14, 50, 200]:
        t_cpu = bench(cpu_wma, x, w)
        t_gpu = bench(tacuda.wma, x, w)
        speedup = t_cpu / max(t_gpu, 1e-9)
        print(f"w={w}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}")
