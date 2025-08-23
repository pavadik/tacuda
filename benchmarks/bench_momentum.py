import numpy as np
import tacuda
from utils import bench


def cpu_momentum(x, p):
    """Simple momentum: difference from lagged value."""
    y = np.full_like(x, np.nan)
    y[p:] = x[p:] - x[:-p]
    return y


if __name__ == "__main__":
    n = 1_000_000
    x = np.random.rand(n).astype(np.float32)
    for p in [5, 14, 50, 200]:
        t_cpu = bench(cpu_momentum, x, p)
        t_gpu = bench(tacuda.momentum, x, p)
        speedup = t_cpu / max(t_gpu, 1e-9)
        print(f"p={p}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}")
