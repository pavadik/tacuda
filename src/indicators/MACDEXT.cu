#include <algorithm>
#include <stdexcept>
#include <indicators/MACDEXT.h>
#include <utils/CudaUtils.h>

static __device__ float ema_at(const float* __restrict__ x, int idx, int period, int start) {
    const float k = 2.0f / (period + 1.0f);
    float weight = 1.0f;
    float weightedSum = x[idx];
    float weightSum = 1.0f;
    int steps = min(period, idx - start);
    for (int i = 1; i <= steps; ++i) {
        weight *= (1.0f - k);
        weightedSum += x[idx - i] * weight;
        weightSum += weight;
    }
    return weightedSum / weightSum;
}

static __device__ float sma_at(const float* __restrict__ x, int idx, int period, int start) {
    float sum = 0.0f;
    int steps = min(period, idx - start + 1);
    for (int i = 0; i < steps; ++i)
        sum += x[idx - i];
    return sum / steps;
}

__global__ void macdLineKernel(const float* __restrict__ input,
                               float* __restrict__ macd,
                               int fastP, int slowP, int size,
                               MAType type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        float maFast = (type == MAType::EMA)
            ? ema_at(input, idx, fastP, 0)
            : sma_at(input, idx, fastP, 0);
        float maSlow = (type == MAType::EMA)
            ? ema_at(input, idx, slowP, 0)
            : sma_at(input, idx, slowP, 0);
        macd[idx] = maFast - maSlow;
    }
}

__global__ void macdSignalKernel(const float* __restrict__ macd,
                                 float* __restrict__ signal,
                                 float* __restrict__ hist,
                                 int slowP, int signalP, int size,
                                 MAType type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        float sig = (type == MAType::EMA)
            ? ema_at(macd, idx, signalP, slowP)
            : sma_at(macd, idx, signalP, slowP);
        signal[idx] = sig;
        hist[idx] = macd[idx] - sig;
    }
}

MACDEXT::MACDEXT(int fastPeriod, int slowPeriod, int signalPeriod, MAType type)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod), signalPeriod(signalPeriod), type(type) {}

void MACDEXT::calculate(const float* input, float* output, int size) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0) {
        throw std::invalid_argument("MACD: invalid periods");
    }
    if (fastPeriod >= slowPeriod) {
        throw std::invalid_argument("MACD: fastPeriod must be < slowPeriod");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, 3 * size * sizeof(float)));
    float* macd = output;
    float* signal = output + size;
    float* hist = output + 2 * size;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    macdLineKernel<<<grid, block>>>(input, macd, fastPeriod, slowPeriod, size, type);
    CUDA_CHECK(cudaGetLastError());
    macdSignalKernel<<<grid, block>>>(macd, signal, hist, slowPeriod, signalPeriod, size, type);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
