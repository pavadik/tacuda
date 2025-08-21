#include <algorithm>
#include <stdexcept>
#include <indicators/MACD.h>
#include <utils/CudaUtils.h>

// Compute EMA of x at index idx using period. The optional start parameter
// specifies the lowest index to look back to (useful when the beginning of the
// array contains NaNs).
__device__ float ema_at(const float* __restrict__ x, int idx, int period, int start) {
    float k = 2.0f / (period + 1.0f);
    float ema = x[idx];
    int steps = idx - start;
    steps = steps < 0 ? 0 : steps;
    if (steps > period * 4) steps = period * 4;
    for (int i = 1; i <= steps; ++i) {
        ema = x[idx - i] * k + ema * (1.0f - k);
    }
    return ema;
}

__global__ void macdLineKernel(const float* __restrict__ input,
                               float* __restrict__ macdOut,
                               int fastP, int slowP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        float emaFast = ema_at(input, idx, fastP, 0);
        float emaSlow = ema_at(input, idx, slowP, 0);
        macdOut[idx] = emaFast - emaSlow;
    }
}

__global__ void macdSignalKernel(const float* __restrict__ macdIn,
                                 float* __restrict__ signalOut,
                                 float* __restrict__ histOut,
                                 int slowP, int signalP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = slowP;
    if (idx >= slowP + signalP - 1 && idx < size) {
        float signal = ema_at(macdIn, idx, signalP, start);
        signalOut[idx] = signal;
        histOut[idx] = macdIn[idx] - signal;
    }
}

MACD::MACD(int fastPeriod, int slowPeriod, int signalPeriod)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod), signalPeriod(signalPeriod) {}

void MACD::calculate(const float* input, float* lineOut, float* signalOut,
                     float* histOut, std::size_t size) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0) {
        throw std::invalid_argument("MACD: invalid periods");
    }
    if (fastPeriod >= slowPeriod) {
        throw std::invalid_argument("MACD: fastPeriod must be < slowPeriod");
    }

    // Initialize outputs with NaNs so regions without valid data remain NaN.
    CUDA_CHECK(cudaMemset(lineOut, 0xFF, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(signalOut, 0xFF, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(histOut, 0xFF, size * sizeof(float)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    macdLineKernel<<<grid, block>>>(input, lineOut, fastPeriod, slowPeriod, (int)size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    macdSignalKernel<<<grid, block>>>(lineOut, signalOut, histOut, slowPeriod, signalPeriod, (int)size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
