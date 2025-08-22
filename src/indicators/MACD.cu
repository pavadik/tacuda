#include <algorithm>
#include <stdexcept>
#include <indicators/MACD.h>
#include <utils/CudaUtils.h>

// Compute the exponential moving average for a given index using a
// sliding‑window formulation.  The previous implementation walked a
// hardcoded "period * 4" window which wasted work for small periods and
// caused excessive global memory traffic.  Here we restrict the walk to
// the actual period and accumulate weighted sums, effectively mimicking a
// prefix-sum of exponentially decaying weights.  This improves cache
// locality and avoids touching values outside the required window.
static __device__ float ema_at(const float* __restrict__ x, int idx, int period) {
    const float k = 2.0f / (period + 1.0f);
    float weight = 1.0f;        // Current weight for x[idx - i]
    float weightedSum = x[idx]; // Accumulated weighted input values
    float weightSum   = 1.0f;   // Sum of weights for normalisation

    // Only scan as far back as the actual period or the available history.
    int steps = min(period, idx);
#pragma unroll
    for (int i = 1; i <= steps; ++i) {
        weight *= (1.0f - k);               // Exponential decay
        weightedSum += x[idx - i] * weight; // Accumulate weighted sample
        weightSum   += weight;              // Track total weight
    }

    return weightedSum / weightSum;
}

__global__ void macdKernel(const float* __restrict__ input,
                           float* __restrict__ macdOut,
                           int fastP, int slowP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        float emaFast = ema_at(input, idx, fastP);
        float emaSlow = ema_at(input, idx, slowP);
        macdOut[idx] = emaFast - emaSlow;
    }
}

MACD::MACD(int fastPeriod, int slowPeriod)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod) {}

void MACD::calculate(const float* input, float* output, int size) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0) {
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
    macdKernel<<<grid, block>>>(input, output, fastPeriod, slowPeriod, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
