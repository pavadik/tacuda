import numpy as np
import tacuda
from utils import bench


def cpu_sma(x, w):
    """Pure NumPy implementation of simple moving average."""
    y = np.full_like(x, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    y[w - 1 :] = (c[w:] - c[:-w]) / w
    return y


if __name__ == "__main__":
    n = 1_000_000
    x = np.random.rand(n).astype(np.float32)
    for w in [5, 14, 50, 200]:
        t_cpu = bench(cpu_sma, x, w)
        t_gpu = bench(tacuda.sma, x, w)
        speedup = t_cpu / max(t_gpu, 1e-9)
        print(f"w={w}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}")
