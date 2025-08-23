#tacuda — CUDA - accelerated technical analysis indicators(TA - style)

A high - performance library of technical indicators “in the spirit of TA -
    Lib”,
    executed on **NVIDIA CUDA GPUs **.It provides a **stable C API **,
    **Python bindings **(pybind11),
    and a **C #binding **(DllImport).Indicators are pluggable via a registry.

        -- -

        ##Features

        - ⚙️ **Indicators ** : `SMA`,
    price
    transforms(`AVGPRICE`, `MEDPRICE`, `TYPPRICE`, `WCLPRICE`, `MIDPRICE`),
    oscillators like `MIDPOINT`, `MAXINDEX` and `MININDEX`,
    range tools `MINMAX`/`MINMAXINDEX`,
    rate - of - change variants `ROC`, `ROCP`, `ROCR`, `ROCR100`,
    and momentum oscillators such as `WILLR`,
    alongside advanced features like `MACDFIX` and `HT_TRENDLINE`; the framework is ready for `EMA`, `RSI`, `MACD`, `BBANDS`, `WMA`, `STDDEV`, `MIN/MAX`, etc.
 - 🕯️ **Candlestick patterns**: Doji, Hammer, Inverted Hammer, Bullish Engulfing, Bearish Engulfing, Three White Soldiers, Abandoned Baby, Advance Block, Belt Hold, Breakaway, Two Crows, Three Black Crows, Three Inside, Three Line Strike, Three Stars In South.
- 🧩 **TA-style API**: procedural calls by indicator name with a unified dispatcher.
- 🧱 **Stable C interface**: `extern "C"` functions (`tacuda_sma_host`, `tacuda_run_indicator_host_c`) with a fixed ABI.
- 🐍 **Python bindings**: `tacuda.sma()` and a generic `tacuda.run()`.
- 💠 **C# binding**: thin .NET layer with `Tacuda.SMA(...)` and generic `Tacuda.Run<TParams>(...)`.
- 🧪 **Tests**: GoogleTest covers registry, `SMA` correctness, helpers, and error paths.
- 🚀 **Performance**: GPU kernels with optional `cudaStream_t` for pipelined workloads.
- ✅ **Expected semantics**: `NaN` during warm-up periods (like TA-Lib).

---

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability ≥ 6.0 recommended).
- **CUDA Toolkit** 11/12 (12.x recommended) and a compatible driver.
- **CMake ≥ 3.21**, C++17 compiler.
- Optional: **Python ≥ 3.8** (for bindings).
- Optional: **.NET 7.0+** (for the C# binding).

---

## Build & Install

### A) C++/CUDA library and examples

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Artifacts:
- Library: `build/` (`libtacuda.so` on Linux, `tacuda.dll` on Windows, `libtacuda.dylib` on macOS).
- Examples: `build/examples/`.

Run an example:
```bash
./build/examples/tacuda_example
```

### B) Run tests

```bash
ctest --test-dir build --output-on-failure
```

> On CI runners without a GPU, some tests may be skipped or marked XFAIL — this is expected.

### C) Python bindings (local development)

Build with Python bindings enabled:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
cmake --build build -j
```

Use the built module by pointing `PYTHONPATH` to the build directory (adjust path if needed):
```bash
PYTHONPATH=build python -c "import numpy as np, tacuda; x=np.arange(1,11,dtype=np.float32); print(tacuda.sma(x, window=5))"
```

> Packaging for PyPI via `scikit-build-core` is supported in the project layout; publish when ready.

### D) C# binding

Build the native library (see A), then:

```bash
dotnet build bindings/csharp/ConsoleExample -c Release

#Ensure the native library is discoverable by the process:
#- Windows : place tacuda.dll next to the.exe or add its folder to PATH
#- Linux : place libtacuda.so next to the binary or add to LD_LIBRARY_PATH
#- macOS : place libtacuda.dylib next to the binary or add to DYLD_LIBRARY_PATH

dotnet run --project bindings/csharp/ConsoleExample
```

---

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

    ## #C API

```c
#include "tacuda/c_api.h"
#include <stdio.h>

    int
    main() {
  float in[10];
  for (int i = 0; i < 10; i++)
    in[i] = (float)(i + 1);
  float out[10];
  int rc = tacuda_sma_host(in, 10, 5, out);
  if (rc != 0) {
    printf("Error: %s\n", tacuda_status_str(rc));
    return 1;
  }
  for (int i = 0; i < 10; i++)
    printf("%g\n", out[i]);
  return 0;
}
```

    ## #Python

```python import numpy as np,
    tacuda x = np.arange(1, 11, dtype = np.float32)
                   print(tacuda.sma(x, window = 5))
# or via the generic dispatcher:
                       print(tacuda.run("SMA", x, window = 3))
```

               ## #C #(.NET)

```csharp using System;
using TacudaNet;

class Demo {
  static void Main() {
    var x = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Console.WriteLine(string.Join(", ", Tacuda.SMA(x, 5)));
    var y = Tacuda.Run("SMA", x, new SMAParams(3));
    Console.WriteLine(string.Join(", ", y));
  }
}
```

    -- -

    ##Architecture(brief)

    - **Registry ** : thread - safe,
    lazy map from indicator name to `{
  function pointer, params_size
}`.
- **IndicatorFn**: unified kernel signature  
  `Status (*)(const float* d_in, float* d_out, int n, const void* params_blob, cudaStream_t stream)`.
- **DeviceBuffer<T>**: RAII for `cudaMalloc/cudaFree` plus async `copy_from_host/copy_to_host`.
- **API**: `run_indicator(...)`, `run_indicator_typed(...)`, and `run_indicator_host(...)` for host arrays.
- **C API**: `tacuda_sma_host`, `tacuda_run_indicator_host_c` (stable ABI).
- **Python**: `tacuda.sma(x, window)`, `tacuda.run(name, x, **params)`.
- **C#**: `Tacuda.SMA(float[], int)`, `Tacuda.Run<TParams>(string, float[], in TParams)`.

---

## Tests

Framework: **GoogleTest**.

Covered cases:
- Indicator registration (`SMA`) is visible with correct `params_size`.
- `SMA` correctness on `1..N` with `NaN` during warm-up.
- API paths: `run_indicator_host`, `run_indicator_typed` with `cudaStream_t`.
- Error handling: `NotFound`, `ParamSizeMismatch`, `InvalidArgument`.
- Infrastructure: `DeviceBuffer` move semantics, `launch_1d`.

Run:
```bash
ctest --test-dir build --output-on-failure
```

---

## Benchmarks

See `benchmarks/` for CPU vs GPU comparisons. Recommended practice:

- Report the **median of multiple runs**; show warm vs hot timings.
- Pin the environment (GPU model, driver, CUDA version).
- Typical sizes: `N = 1e6`, windows `5, 14, 50, 200`.

Example (`benchmarks/bench_sma.py`):

```python
import time, numpy as np, tacuda

def bench(f, *args, repeats=5):
    xs=[]
    for _ in range(repeats):
        t0=time.perf_counter();
f(*args); xs.append(time.perf_counter()-t0)
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

---

## Adding a new indicator

1. Create `include/tacuda/indicators/<name>.h` with `Params` and `Status <name>_dispatch(...)`.
2. Implement kernels in `src/indicators/<name>.cu`.
3. Register it: `REGISTER_INDICATOR("<NAME>", <name>_dispatch, <ParamsType>)`.
4. **Python**: add a branch in `tacuda_pybind.cpp` (`run_py`) or a dedicated function.
5. **C#**: declare a `struct` with `LayoutKind.Sequential` matching the C++ params and call `Tacuda.Run("<NAME>", ...)`.
6. Add tests (CPU reference, property checks) and a small benchmark.

---

## Contributing

- Use `clang-format` and keep warnings clean.
- New code must come with tests.
- For performance-affecting changes, attach before/after benchmark numbers.
- Discuss API/architecture changes in Issues first.

---

## Roadmap

- **0.1**: `EMA`, `RSI`, `MACD`, `BBANDS`;
benchmark suite;
SMA kernel with prefix - scan optimization.- **0.2 * * : OHLCV interfaces,
    batched execution, cuDF interop.- **0.3 * * : Windows artifacts,
    NuGet package.- **1.0 * * : frozen C ABI,
    PyPI / Conda packages.

                -- -

                ##License

                    **Apache -
            2.0 *
                *(see `LICENSE`)
                     .

                 -- -

                 ##Disclaimer

                 This project is * *
                not affiliated * *with TA
            - Lib
        or any other trademark.  
“TA - style API” refers to surface - level interface similarity only.
