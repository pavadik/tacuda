import numpy as np
import tacuda
from utils import bench


def cpu_atr(high, low, close, period=14):
    """Average True Range with Wilder's smoothing."""
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    n = high.size
    tr = np.empty(n, dtype=np.float32)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    atr = np.full(n, np.nan, dtype=np.float32)
    if n <= period:
        return atr
    atr_val = tr[1 : period + 1].mean()
    atr[period] = atr_val
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        atr[i] = atr_val
    return atr


if __name__ == "__main__":
    n = 1_000_000
    close = np.random.rand(n).astype(np.float32)
    high = close + np.random.rand(n).astype(np.float32) * 0.1
    low = close - np.random.rand(n).astype(np.float32) * 0.1
    period = 14
    t_cpu = bench(cpu_atr, high, low, close, period)
    t_gpu = bench(tacuda.atr, high, low, close, period)
    speedup = t_cpu / max(t_gpu, 1e-9)
    print(f"p={period}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}")
