import numpy as np
import tacuda
from utils import bench


def cpu_macd_line(x, fast=12, slow=26):
    """Compute MACD line on CPU using exponential moving averages."""
    x = np.asarray(x, dtype=np.float32)
    macd = np.full_like(x, np.nan)
    k_fast = 2.0 / (fast + 1.0)
    k_slow = 2.0 / (slow + 1.0)
    ema_fast = np.empty_like(x)
    ema_slow = np.empty_like(x)
    ema_fast[0] = x[0]
    ema_slow[0] = x[0]
    for i in range(1, x.size):
        ema_fast[i] = ema_fast[i - 1] + (x[i] - ema_fast[i - 1]) * k_fast
        ema_slow[i] = ema_slow[i - 1] + (x[i] - ema_slow[i - 1]) * k_slow
        if i >= slow:
            macd[i] = ema_fast[i] - ema_slow[i]
    return macd


if __name__ == "__main__":
    n = 1_000_000
    x = np.random.rand(n).astype(np.float32)
    fast, slow = 12, 26
    t_cpu = bench(cpu_macd_line, x, fast, slow)
    t_gpu = bench(tacuda.macd_line, x, fast, slow)
    speedup = t_cpu / max(t_gpu, 1e-9)
    print(
        f"fast={fast}, slow={slow}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}"
    )
