#include <indicators/ADOSC.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <algorithm>

__global__ void adLineKernel(const float* __restrict__ high,
                             const float* __restrict__ low,
                             const float* __restrict__ close,
                             const float* __restrict__ volume,
                             float* __restrict__ ad,
                             int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float cum = 0.0f;
        for (int i = 0; i < size; ++i) {
            float h = high[i];
            float l = low[i];
            float c = close[i];
            float v = volume[i];
            float denom = h - l;
            float clv = denom == 0.0f ? 0.0f : ((c - l) - (h - c)) / denom;
            cum += clv * v;
            ad[i] = cum;
        }
    }
}

static __device__ float ema_at(const float* __restrict__ x, int idx, int period) {
    const float k = 2.0f / (period + 1.0f);
    float weight = 1.0f;
    float weightedSum = x[idx];
    float weightSum = 1.0f;
    int steps = min(period, idx);
#pragma unroll
    for (int i = 1; i <= steps; ++i) {
        weight *= (1.0f - k);
        weightedSum += x[idx - i] * weight;
        weightSum += weight;
    }
    return weightedSum / weightSum;
}

__global__ void adoscKernel(const float* __restrict__ ad,
                            float* __restrict__ output,
                            int shortP, int longP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= longP && idx < size) {
        float emaShort = ema_at(ad, idx, shortP);
        float emaLong = ema_at(ad, idx, longP);
        output[idx] = emaShort - emaLong;
    }
}

ADOSC::ADOSC(int shortPeriod, int longPeriod)
    : shortPeriod(shortPeriod), longPeriod(longPeriod) {}

void ADOSC::calculate(const float* high, const float* low, const float* close,
                      const float* volume, float* output, int size) noexcept(false) {
    if (shortPeriod <= 0 || longPeriod <= 0) {
        throw std::invalid_argument("ADOSC: invalid periods");
    }
    if (shortPeriod >= longPeriod) {
        throw std::invalid_argument("ADOSC: shortPeriod must be < longPeriod");
    }
    float* ad = nullptr;
    CUDA_CHECK(cudaMalloc(&ad, size * sizeof(float)));

    adLineKernel<<<1,1>>>(high, low, close, volume, ad, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    adoscKernel<<<grid, block>>>(ad, output, shortPeriod, longPeriod, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(ad));
}

void ADOSC::calculate(const float* input, float* output, int size) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    const float* volume = input + 3 * size;
    calculate(high, low, close, volume, output, size);
}
