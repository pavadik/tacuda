#include <algorithm>
#include <stdexcept>
#include <indicators/MACD.h>
#include <utils/CudaUtils.h>

__device__ float ema_at(const float* __restrict__ x, int idx, int period) {
    float k = 2.0f / (period + 1.0f);
    float ema = x[idx];
    int steps = min(period * 4, idx);
    for (int i = 1; i <= steps; ++i) {
        ema = x[idx - i] * k + ema * (1.0f - k);
    }
    return ema;
}

__global__ void macdKernel(const float* __restrict__ input,
                           float* __restrict__ macdOut,
                           int fastP, int slowP, int signalP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        float emaFast = ema_at(input, idx, fastP);
        float emaSlow = ema_at(input, idx, slowP);
        macdOut[idx] = emaFast - emaSlow;
    }
}

MACD::MACD(int fastPeriod, int slowPeriod, int signalPeriod)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod), signalPeriod(signalPeriod) {}

void MACD::calculate(const float* input, float* output, int size) {
    if (fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0) {
        throw std::invalid_argument("MACD: invalid periods");
    }
    if (fastPeriod >= slowPeriod) {
        throw std::invalid_argument("MACD: fastPeriod must be < slowPeriod");
    }
    // Warm-up region at the beginning should remain NaN. Initialize the
    // entire output with NaNs and only compute values for indices beyond the
    // slowPeriod.
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    macdKernel<<<grid, block>>>(input, output, fastPeriod, slowPeriod, signalPeriod, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
