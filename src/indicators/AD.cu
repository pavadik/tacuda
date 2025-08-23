#include <indicators/AD.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void adKernel(const float* __restrict__ high,
                         const float* __restrict__ low,
                         const float* __restrict__ close,
                         const float* __restrict__ volume,
                         float* __restrict__ output,
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
            output[i] = cum;
        }
    }
}

void AD::calculate(const float* high, const float* low, const float* close,
                   const float* volume, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("AD: invalid size");
    }
    adKernel<<<1, 1, 0, stream>>>(high, low, close, volume, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void AD::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    const float* volume = input + 3 * size;
    calculate(high, low, close, volume, output, size, stream);
}
