import numpy as np
import tacuda
from utils import bench


def cpu_adx(high, low, close, period=14):
    """Average Directional Index using Wilder's smoothing."""
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    n = high.size
    plus_dm = np.zeros(n, dtype=np.float32)
    minus_dm = np.zeros(n, dtype=np.float32)
    tr = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    tr_sum = tr[1 : period + 1].sum()
    plus_dm_sum = plus_dm[1 : period + 1].sum()
    minus_dm_sum = minus_dm[1 : period + 1].sum()
    plus_di = np.zeros(n, dtype=np.float32)
    minus_di = np.zeros(n, dtype=np.float32)
    dx = np.zeros(n, dtype=np.float32)
    plus_di[period] = 100.0 * plus_dm_sum / tr_sum
    minus_di[period] = 100.0 * minus_dm_sum / tr_sum
    dx[period] = (
        100.0
        * abs(plus_di[period] - minus_di[period])
        / (plus_di[period] + minus_di[period] + 1e-9)
    )
    for i in range(period + 1, n):
        tr_sum = tr_sum - tr_sum / period + tr[i]
        plus_dm_sum = plus_dm_sum - plus_dm_sum / period + plus_dm[i]
        minus_dm_sum = minus_dm_sum - minus_dm_sum / period + minus_dm[i]
        plus_di[i] = 100.0 * plus_dm_sum / tr_sum
        minus_di[i] = 100.0 * minus_dm_sum / tr_sum
        dx[i] = (
            100.0
            * abs(plus_di[i] - minus_di[i])
            / (plus_di[i] + minus_di[i] + 1e-9)
        )
    adx = np.full(n, np.nan, dtype=np.float32)
    if n <= 2 * period:
        return adx
    adx_val = dx[period : 2 * period].mean()
    adx[2 * period - 1] = adx_val
    for i in range(2 * period, n):
        adx_val = (adx_val * (period - 1) + dx[i]) / period
        adx[i] = adx_val
    return adx


if __name__ == "__main__":
    n = 1_000_000
    close = np.random.rand(n).astype(np.float32)
    high = close + np.random.rand(n).astype(np.float32) * 0.1
    low = close - np.random.rand(n).astype(np.float32) * 0.1
    period = 14
    t_cpu = bench(cpu_adx, high, low, close, period)
    t_gpu = bench(tacuda.adx, high, low, close, period)
    speedup = t_cpu / max(t_gpu, 1e-9)
    print(f"p={period}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}")
