import numpy as np
import tacuda
from utils import bench


def cpu_obv(close, volume):
    """On-Balance Volume cumulative sum."""
    close = np.asarray(close, dtype=np.float32)
    volume = np.asarray(volume, dtype=np.float32)
    obv = np.zeros_like(close)
    obv[0] = volume[0]
    for i in range(1, close.size):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv


if __name__ == "__main__":
    n = 1_000_000
    close = np.random.rand(n).astype(np.float32)
    volume = np.random.rand(n).astype(np.float32) * 1_000
    t_cpu = bench(cpu_obv, close, volume)
    t_gpu = bench(tacuda.obv, close, volume)
    speedup = t_cpu / max(t_gpu, 1e-9)
    print(f"CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{speedup:.1f}")
