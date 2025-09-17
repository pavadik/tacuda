import numpy as np
import tacuda
from utils import bench


def cpu_ema_window(x, period):
    """Reference EMA implementation matching tacuda.ema semantics."""
    x = np.asarray(x, dtype=np.float32)
    out = np.full_like(x, np.nan)
    k = 2.0 / (period + 1.0)
    valid = x.size - period + 1
    if valid <= 0:
        return out

    for start in range(valid):
        weight = 1.0
        weighted_sum = x[start + period - 1]
        weight_sum = 1.0
        for j in range(1, period):
            weight *= 1.0 - k
            weighted_sum += x[start + period - 1 - j] * weight
            weight_sum += weight
        out[start] = weighted_sum / weight_sum
    return out


if __name__ == "__main__":
    n = 1_000_000
    periods = [32, 4096]
    x = np.random.rand(n).astype(np.float32)

    for period in periods:
        t_cpu = bench(cpu_ema_window, x, period)
        t_gpu = bench(tacuda.ema, x, period)
        speedup = t_cpu / max(t_gpu, 1e-9)
        print(
            f"period={period}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}")
