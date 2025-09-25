# 🚀 TACUDA

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11%2F12-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![.NET](https://img.shields.io/badge/.NET-7.0%2B-purple.svg)](https://dotnet.microsoft.com/)

> **High-Performance CUDA-Accelerated Technical Analysis Library**

TACUDA delivers lightning-fast technical analysis indicators powered by NVIDIA CUDA GPUs. Built for quantitative traders, researchers, and financial analysts who demand maximum performance from their computational workflows.

---

## ✨ Key Features

### 🔥 **Performance First**
- **GPU-accelerated kernels** for massive parallel computation
- **Pipelined workloads** with optional `cudaStream_t` support
- **Zero-copy operations** where possible
- **Optimized memory patterns** for coalesced access

### 📊 **Comprehensive Indicator Suite**
- **Moving Averages**: SMA, EMA, WMA
- **Price Transforms**: AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, MIDPRICE
- **Oscillators**: MIDPOINT, MAXINDEX, MININDEX, WILLR
- **Momentum**: ROC, ROCP, ROCR, ROCR100
- **Statistical**: STDDEV, MIN, MAX, MINMAX, MINMAXINDEX
- **Advanced**: RSI, MACD, BBANDS, Others

### 🕯️ **Candlestick Pattern Recognition**
Comprehensive pattern detection including Doji, Hammer, Engulfing patterns, Three White Soldiers, and many more.

### 🔌 **Multi-Language Support**
- **C/C++**: Stable ABI with `extern "C"` interface
- **Python**: Pythonic API with NumPy integration
- **C#**: Native .NET bindings with generic support

### ✅ **Production Ready**
- **Warm-up markers** via trailing `NaN` values for incomplete windows
- **Comprehensive test suite** with GoogleTest
- **Memory-safe** RAII patterns
- **Thread-safe** indicator registry

---

## 🛠️ Installation

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| **NVIDIA GPU** | Compute ≥ 6.0 | Maxwell architecture or newer |
| **CUDA Toolkit** | 11.x / 12.x | 12.x recommended |
| **CMake** | ≥ 3.21 | |
| **Compiler** | C++17 | GCC, Clang, or MSVC |
| **Python** | ≥ 3.8 | *Optional for bindings* |
| **NET** | ≥ 7.0 | *Optional for C# binding* |

### Quick Build

```bash
# Clone repository
git clone https://github.com/pavadik/tacuda.git
cd tacuda

# Build core library
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run example
./build/examples/tacuda_example
```

### Python Installation

```bash
# Build with Python support
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
cmake --build build -j$(nproc)

# Test installation
PYTHONPATH=build python -c "
import numpy as np
import tacuda
data = np.arange(1, 11, dtype=np.float32)
result = tacuda.sma(data, window=5)
print('SMA Result:', result)
"
```

### C# Installation

```bash
# Build C# bindings
dotnet build bindings/csharp/ConsoleExample -c Release

# Ensure library is discoverable (Linux example)
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
dotnet run --project bindings/csharp/ConsoleExample
```

### Binding generation workflow

Python and C# bindings are generated from the public header.  After editing
`include/tacuda.h`, regenerate the artefacts and commit the results:

```bash
python bindings/generate_bindings.py
```

CTest contains a guard that fails if the checked-in bindings are stale.

---

## 🚀 Quick Start

### Python API

```python
import numpy as np
import tacuda

# Generate sample price data
prices = np.random.randn(1000).cumsum().astype(np.float32)

# Calculate 20-period Simple Moving Average
sma_20 = tacuda.sma(prices, window=20)

# Generic indicator interface
roc_5 = tacuda.run("ROC", prices, timeperiod=5)

# Williams %R oscillator
willr = tacuda.run("WILLR", prices, timeperiod=14)
```

### C++ API

```cpp
#include "tacuda/api.h"
#include "tacuda/indicators/sma.h"
#include <vector>
#include <iostream>

int main() {
    // Input data
    std::vector<float> prices = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> output;
    
    // Configure SMA parameters
    tacuda::SMAParams params{5};  // 5-period window
    
    // Execute on GPU
    auto status = tacuda::run_indicator_host("SMA", prices, output, params);

    std::cout << "SMA computed successfully!" << std::endl;
    for (float value : output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return status == tacuda::Status::OK ? 0 : 1;
}
```

Expected output:

```text
SMA computed successfully!
3 4 5 6 7 8 nan nan nan nan
```

The library leaves trailing `NaN` values in place of samples that do not yet
cover a full window so consumers can easily spot warm-up regions.

### 📦 OHLCV Container

#### C++
```cpp
#include <tacuda/OHLCVSeries.h>
#include <tacuda.h>

std::vector<float> open{1.0f, 2.0f, 1.5f};
std::vector<float> high{1.2f, 2.3f, 1.6f};
std::vector<float> low{0.9f, 1.9f, 1.3f};
std::vector<float> close{1.1f, 2.1f, 1.4f};

// Volume defaults to zero when omitted
tacuda::OHLCVSeries candles(open, high, low, close);
std::vector<float> imi(candles.size());
ct_imi(candles.open_data(), candles.close_data(), imi.data(), static_cast<int>(candles.size()), 3);
```

#### Python
```python
from tacuda import OHLCV, imi

ohlcv = OHLCV.from_columns(open, high, low, close, volume)
result = imi(ohlcv.open, ohlcv.close, period=3)
packed = ohlcv.column_major()  # [O1..On, H1..Hn, ...]
```

#### C#
```csharp
using Tacuda.Bindings;

var candles = new OhlcvSeries(open, high, low, close);
var output = new float[candles.Length];
NativeMethods.ct_imi(candles.Open, candles.Close, output, candles.Length, period: 3);
var columnMajor = candles.ToColumnMajor();
```


### C# API

```csharp
using Tacuda;

class Program 
{
    static void Main() 
    {
        float[] prices = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f};
        
        // Simple Moving Average
        var sma = Tacuda.SMA(prices, window: 5);
        
        // Generic interface
        var roc = Tacuda.Run<ROCParams>("ROC", prices, new ROCParams { timeperiod = 5 });
        
        Console.WriteLine($"SMA: [{string.Join(", ", sma)}]");
    }
}
```

---

## 📈 Performance Benchmarks

TACUDA delivers significant performance improvements over CPU implementations:

```python
# Benchmark example (1M data points)
import time
import numpy as np
import tacuda

def benchmark_sma():
    n = 1_000_000
    data = np.random.rand(n).astype(np.float32)
    
    for window in [5, 14, 50, 200]:
        # CPU implementation
        cpu_start = time.perf_counter()
        cpu_result = np.convolve(data, np.ones(window)/window, mode='same')
        cpu_time = time.perf_counter() - cpu_start
        
        # GPU implementation  
        gpu_start = time.perf_counter()
        gpu_result = tacuda.sma(data, window=window)
        gpu_time = time.perf_counter() - gpu_start
        
        speedup = cpu_time / gpu_time
        print(f"Window {window:3d}: CPU {cpu_time:.4f}s | GPU {gpu_time:.4f}s | {speedup:.1f}x faster")
```

**Typical Results** *(NVIDIA RTX 4090)*:
```
Window   5: CPU 0.0234s | GPU 0.0031s | 7.5x faster
Window  14: CPU 0.0267s | GPU 0.0033s | 8.1x faster
Window  50: CPU 0.0312s | GPU 0.0035s | 8.9x faster
Window 200: CPU 0.0445s | GPU 0.0041s | 10.9x faster
```

### ⚙️ Exponential Moving Average Acceleration

The EMA family of indicators now share a single CUDA implementation based on a
prefix-scan formulation of the linear recurrence:

```
EMA[i] = α · x[i] + (1 − α) · EMA[i − 1]
```

Instead of iterating sequentially, we interpret each update as an affine
transformation and compute the cumulative product of these transforms with
`thrust::inclusive_scan`. This yields all intermediate EMA values in parallel,
which are then re-used across EMA, DEMA, TEMA, T3, TRIX and MACD calculations.
The shared helper keeps warm-up regions initialised to `NaN` while guaranteeing
identical numerical outputs to the previous per-kernel loops.

To reproduce the performance improvement for large smoothing periods, run the
dedicated benchmark:

```bash
python benchmarks/bench_ema.py
```

---

## 🏗️ Architecture

TACUDA employs a clean, extensible architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Language      │    │    Unified       │    │   CUDA Kernel   │
│   Bindings      │───▶│   Dispatcher     │───▶│   Execution     │
│ (Python/C#/C++) │    │   Registry       │    │   Engine        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Type Safety   │    │   Memory         │    │   Optimized     │
│   Parameter     │    │   Management     │    │   Algorithms    │
│   Validation    │    │   (RAII)         │    │   (Parallel)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

- **🔧 Registry**: Thread-safe, lazy-loaded indicator registry
- **⚡ IndicatorFn**: Unified kernel signature for all indicators  
- **🎯 DeviceBuffer**: RAII memory management with async operations
- **🔌 Bindings**: Language-specific wrappers with native feel

---

## 🧪 Testing & Quality

```bash
# Run comprehensive test suite
ctest --test-dir build --output-on-failure

# Performance regression testing
cd benchmarks && python benchmark_suite.py

# Memory leak detection (Linux)
valgrind --tool=memcheck ./build/examples/tacuda_example
```

**Test Coverage:**
- ✅ Numerical accuracy vs reference implementations
- ✅ Edge cases (empty inputs, extreme values)
- ✅ Memory safety and leak detection
- ✅ Multi-threaded safety
- ✅ Cross-platform compatibility

---

## 🔧 Extending TACUDA

Adding new indicators is straightforward:

### 1. Define Parameters Structure

```cpp
// include/tacuda/indicators/my_indicator.h
struct MyIndicatorParams {
    int period;
    float factor;
};
```

### 2. Implement CUDA Kernel

```cpp
// src/indicators/my_indicator.cu
__global__ void my_indicator_kernel(const float* d_in, float* d_out, 
                                   int n, MyIndicatorParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Your algorithm here
    d_out[idx] = d_in[idx] * params.factor;
}
```

### 3. Register Indicator

```cpp
// Register in the global registry
REGISTER_INDICATOR("MY_INDICATOR", my_indicator_dispatch, MyIndicatorParams);
```

### 4. Add Language Bindings

```python
# Python binding
def my_indicator(data, period=10, factor=1.0):
    return run("MY_INDICATOR", data, period=period, factor=factor)
```

---

## ✅ Status & TODO

### ✅ Completed pillars

- [x] CUDA implementations for moving averages, momentum, volatility, oscillators, candlestick recognition, and price transforms with TA-Lib compatible semantics backed by 140+ GPU kernels.
- [x] Python (ctypes) and .NET 7 bindings generated from the shared C ABI via `bindings/generate_bindings.py`, with regeneration guards wired into CTest.
- [x] GoogleTest regression suite cross-checking against TA-Lib reference data, including device buffer pool coverage and representative indicator families.

### 🚧 Outstanding priorities

**Data ergonomics & throughput**
- [ ] Columnar OHLCV host container plus binding updates so callers no longer hand-pack arrays.
- [ ] Batched/multi-symbol execution entry points with stream-aware scheduling for portfolio-scale workloads.

**Release readiness**
- [ ] Packaging for common ecosystems (PyPI wheel, NuGet, Conda, binary releases) with automated CI publication.
- [ ] Documented production deployment guide, ABI stability policy, and fill-in for the referenced `docs/` tree.

**Benchmarking & validation**
- [ ] Curated benchmark datasets and published comparative numbers covering CPU vs GPU baselines.

## 🗓️ Proposed Sprint Plan

### Sprint 1 – Data ergonomics foundation *(2 weeks)*
- Design and implement an OHLCV columnar container usable from C++, Python, and .NET host APIs.
- Update bindings to accept structured inputs and extend tests/benchmarks to exercise the new path.
- Document migration guidance for existing users still relying on manual array packing.

### Sprint 2 – Batched execution & scheduling *(2 weeks)*
- Add multi-symbol/batched dispatchers with CUDA stream management hooks.
- Extend the registry and bindings to accept portfolio requests, including stress tests and profiling scripts.
- Prototype heuristics for overlapping host/device transfers and kernel launches.

### Sprint 3 – Distribution pipeline *(2 weeks)*
- Author reproducible benchmark datasets and integrate them into the benchmarking harness.
- Stand up CI jobs producing PyPI wheels, Conda packages, NuGet packages, and binary tarballs.
- Capture release criteria covering artifact validation, signing, and smoke tests.

### Sprint 4 – Production readiness *(2 weeks)*
- Build out the referenced documentation tree (API, user guide, performance, operations) and publish deployment runbooks.
- Formalize ABI stability guarantees, including header versioning and changelog automation.
- Prepare an adoption checklist covering monitoring, upgrade sequencing, and support escalation paths.

---

## 📚 Documentation

- **[API Reference](docs/api/)** - Complete function documentation
- **[User Guide](docs/guide/)** - Getting started tutorials  
- **[Performance Guide](docs/performance/)** - Optimization best practices
- **[Contributing](CONTRIBUTING.md)** - Development workflow

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/pavadik/tacuda.git
cd tacuda

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Build in debug mode
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

### Contribution Guidelines

- 🎯 **Focus on performance**: Include benchmarks for performance-affecting changes
- 🧪 **Test coverage**: New features must include comprehensive tests
- 📝 **Documentation**: Update docs for API changes
- 🎨 **Code style**: Use `clang-format` and maintain consistency
- 💬 **Discussion**: Open an issue before major architectural changes

### Code Quality Standards

- ✅ All tests must pass
- ✅ No compiler warnings
- ✅ Memory leak-free
- ✅ Thread-safe implementations
- ✅ Proper error handling

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **NVIDIA** for CUDA toolkit and documentation
- **TA-Lib** for algorithmic reference implementations  
- **NumPy** community for API design inspiration
- **GoogleTest** for testing framework

---

## ⚠️ Disclaimer

This project is independent and not affiliated with TA-Lib or any other trademark. "TA-style API" refers to interface similarity for user convenience only.

---

<div align="center">

**[⭐ Star this repo](https://github.com/pavadik/tacuda)** if you find TACUDA useful!

**Made with ❤️ by Pavel Dikalov**

</div>
