import numpy as np
import tacuda
from utils import bench


def cpu_rsi(x, period=14):
    """Relative Strength Index using Wilder's smoothing."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    rsi = np.full(n, np.nan, dtype=np.float32)
    if n <= period:
        return rsi
    diff = np.diff(x)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = np.zeros(n, dtype=np.float32)
    avg_loss = np.zeros(n, dtype=np.float32)
    avg_gain[period] = gain[:period].mean()
    avg_loss[period] = loss[:period].mean()
    rs = avg_gain[period] / (avg_loss[period] + 1e-9)
    rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
        rs = avg_gain[i] / (avg_loss[i] + 1e-9)
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


if __name__ == "__main__":
    n = 1_000_000
    x = np.random.rand(n).astype(np.float32)
    period = 14
    t_cpu = bench(cpu_rsi, x, period)
    t_gpu = bench(tacuda.rsi, x, period)
    speedup = t_cpu / max(t_gpu, 1e-9)
    print(f"p={period}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}")
