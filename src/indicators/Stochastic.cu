#include <indicators/Stochastic.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void stochasticKernel(const float* __restrict__ high,
                                 const float* __restrict__ low,
                                 const float* __restrict__ close,
                                 float* __restrict__ kOut,
                                 float* __restrict__ dOut,
                                 int kPeriod, int dPeriod, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float nan = nanf("");
        for (int i = 0; i < size; ++i) {
            kOut[i] = nan;
            dOut[i] = nan;
        }
        // compute fast %K for all indices where possible
        for (int i = kPeriod - 1; i < size; ++i) {
            float highest = high[i];
            float lowest = low[i];
            for (int j = 1; j < kPeriod; ++j) {
                float h = high[i - j];
                float l = low[i - j];
                if (h > highest) highest = h;
                if (l < lowest)  lowest = l;
            }
            float denom = highest - lowest;
            kOut[i] = denom == 0.0f ? 0.0f : (close[i] - lowest) / denom * 100.0f;
        }
        int start = kPeriod + dPeriod - 2;
        for (int i = start; i < size; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < dPeriod; ++j) {
                sum += kOut[i - j];
            }
            dOut[i] = sum / dPeriod;
        }
        for (int i = kPeriod - 1; i < start && i < size; ++i) {
            kOut[i] = nan;
        }
    }
}

Stochastic::Stochastic(int kPeriod, int dPeriod) : kPeriod(kPeriod), dPeriod(dPeriod) {}

void Stochastic::calculate(const float* high, const float* low, const float* close,
                           float* output, int size, cudaStream_t stream) noexcept(false) {
    if (kPeriod <= 0 || dPeriod <= 0 || kPeriod + dPeriod - 1 > size) {
        throw std::invalid_argument("Stochastic: invalid periods");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * size * sizeof(float), stream));
    float* kOut = output;
    float* dOut = output + size;
    stochasticKernel<<<1, 1, 0, stream>>>(high, low, close, kOut, dOut, kPeriod, dPeriod, size);
    CUDA_CHECK(cudaGetLastError());
}

void Stochastic::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
