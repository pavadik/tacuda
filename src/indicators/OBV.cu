#include <indicators/OBV.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void obvKernel(const float* __restrict__ price,
                          const float* __restrict__ volume,
                          float* __restrict__ output,
                          int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (size <= 0) return;
        float prevPrice = price[0];
        float obv = volume[0];
        output[0] = obv;
        for (int i = 1; i < size; ++i) {
            float p = price[i];
            float v = volume[i];
            if (p > prevPrice) {
                obv += v;
            } else if (p < prevPrice) {
                obv -= v;
            }
            output[i] = obv;
            prevPrice = p;
        }
    }
}

void tacuda::OBV::calculate(const float* price, const float* volume,
                    float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("OBV: invalid size");
    }
    obvKernel<<<1, 1, 0, stream>>>(price, volume, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::OBV::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* price = input;
    const float* volume = input + size;
    calculate(price, volume, output, size, stream);
}

