# tacuda

CUDA-accelerated technical analysis indicators (TA-style).

tacuda is a high-performance library of technical indicators executed on NVIDIA CUDA GPUs. It exposes a stable C API and bindings for Python and C#. Indicators are registered through a pluggable registry.

## Features

- **Indicators**: SMA, price transforms (AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, MIDPRICE), oscillators (MIDPOINT, MAXINDEX, MININDEX), range tools (MINMAX, MINMAXINDEX), rate-of-change variants (ROC, ROCP, ROCR, ROCR100) and momentum oscillators such as WILLR. The framework is ready for EMA, RSI, MACD, BBANDS, WMA, STDDEV, MIN/MAX and more.
- **Candlestick patterns**: Doji, Hammer, Inverted Hammer, Bullish Engulfing, Bearish Engulfing, Three White Soldiers, Abandoned Baby, Advance Block, Belt Hold, Breakaway, Two Crows, Three Black Crows, Three Inside, Three Line Strike, Three Stars In South, etc.
- **TA-style API**: procedural calls by indicator name with a unified dispatcher.
- **Stable C interface**: `extern "C"` functions (`tacuda_sma_host`, `tacuda_run_indicator_host_c`) with a fixed ABI.
- **Python bindings**: `tacuda.sma()` and a generic `tacuda.run()`.
- **C# binding**: thin .NET layer with `Tacuda.SMA(...)` and generic `Tacuda.Run<TParams>(...)`.
- **Tests**: GoogleTest covers registry, SMA correctness, helpers and error paths.
- **Performance**: GPU kernels with optional `cudaStream_t` for pipelined workloads.
- **Expected semantics**: `NaN` during warm-up periods (like TA-Lib).

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability ≥ 6.0 recommended).
- CUDA Toolkit 11/12 (12.x recommended) and a compatible driver.
- CMake ≥ 3.21 and a C++17 compiler.
- Optional: Python ≥ 3.8 (for bindings).
- Optional: .NET 7.0+ (for the C# binding).

## Build

### C++/CUDA library

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run an example:

```bash
./build/examples/tacuda_example
```

### Tests

```bash
ctest --test-dir build --output-on-failure
```

On CI runners without a GPU, some tests may be skipped or marked XFAIL — this is expected.

### Python bindings

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH=build python -c "import numpy as np, tacuda; x=np.arange(1,11,dtype=np.float32); print(tacuda.sma(x, window=5))"
```

### C# binding

```bash
dotnet build bindings/csharp/ConsoleExample -c Release

# Ensure the native library is discoverable by the process:
# - Windows: place tacuda.dll next to the .exe or add its folder to PATH
# - Linux: place libtacuda.so next to the binary or add to LD_LIBRARY_PATH
# - macOS: place libtacuda.dylib next to the binary or add to DYLD_LIBRARY_PATH

dotnet run --project bindings/csharp/ConsoleExample
```

## Quick start

### C++ (host helper)

```cpp
#include "tacuda/api.h"
#include "tacuda/indicators/sma.h"
#include <vector>

int main() {
  std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, out;
  tacuda::SMAParams p{5};
  auto st = tacuda::run_indicator_host("SMA", in, out, p);
  return st == tacuda::Status::OK ? 0 : 1;
}
```

### C API

```c
#include "tacuda/c_api.h"
...
```

## Architecture (brief)

- **Registry**: thread-safe, lazy map from indicator name to `{ function pointer, params_size }`.
- **IndicatorFn**: unified kernel signature `Status (*)(const float* d_in, float* d_out, int n, const void* params_blob, cudaStream_t stream)`.
- **DeviceBuffer<T>**: RAII for `cudaMalloc/cudaFree` plus async `copy_from_host/copy_to_host`.
- **API**: `run_indicator(...)`, `run_indicator_typed(...)`, and `run_indicator_host(...)` for host arrays.
- **Bindings**: `tacuda.sma(x, window)`, `tacuda.run(name, x, **params)` for Python; `Tacuda.SMA(float[], int)`, `Tacuda.Run<TParams>(string, float[], in TParams)` for C#.

## Benchmarks

See `benchmarks/` for CPU vs GPU comparisons. Recommended practice:

- Report the median of multiple runs; show warm vs hot timings.
- Pin the environment (GPU model, driver, CUDA version).
- Typical sizes: `N = 1e6`, windows `5, 14, 50, 200`.

Example (`benchmarks/bench_sma.py`):

```python
import time, numpy as np, tacuda

def bench(f, *args, repeats=5):
    xs=[]
    for _ in range(repeats):
        t0=time.perf_counter(); f(*args); xs.append(time.perf_counter()-t0)
    return np.median(xs)

def cpu_sma(x, w):
    y=np.full_like(x, np.nan)
    c=np.cumsum(np.insert(x,0,0.0))
    y[w-1:] = (c[w:]-c[:-w])/w
    return y

if __name__=="__main__":
    n=1_000_000
    x=np.random.rand(n).astype(np.float32)
    for w in [5,14,50,200]:
        t_cpu=bench(cpu_sma, x, w)
        t_gpu=bench(tacuda.sma, x, w)
        print(f"w={w}: CPU {t_cpu:.4f}s | GPU {t_gpu:.4f}s | x{t_cpu/max(t_gpu,1e-9):.1f}")
```

## Adding a new indicator

1. Create `include/indicators/<name>.h` with `Params` and `Status <name>_dispatch(...)`.
2. Implement kernels in `src/indicators/<name>.cu`.
3. Register it: `REGISTER_INDICATOR("<NAME>", <name>_dispatch, <ParamsType>)`.
4. **Python**: add a branch in `tacuda_pybind.cpp` (`run_py`) or a dedicated function.
4. **C#**: declare a `struct` with `LayoutKind.Sequential` matching the C++ params and call `Tacuda.Run("<NAME>", ...)`.
5. Add tests (CPU reference, property checks) and a small benchmark.

## Contributing

- Use `clang-format` and keep warnings clean.
- New code must come with tests.
- For performance-affecting changes, attach before/after benchmark numbers.
- Discuss API/architecture changes in Issues first.

## Roadmap

- **0.1**: `EMA`, `RSI`, `MACD`, `BBANDS`; benchmark suite; SMA kernel with prefix-scan optimization.
- **0.2**: OHLCV interfaces, batched execution, cuDF interop.
- **0.3**: Windows artifacts, NuGet package.
- **1.0**: frozen C ABI, PyPI / Conda packages.

## License

Apache-2.0 (see `LICENSE`).

## Disclaimer

This project is not affiliated with TA-Lib or any other trademark. “TA-style API” refers to surface-level interface similarity only.
