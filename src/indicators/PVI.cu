#include <indicators/PVI.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

namespace {
__global__ void pviKernel(const float* __restrict__ close,
                          const float* __restrict__ volume,
                          float* __restrict__ output,
                          int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (size <= 0) {
            return;
        }
        float prevIndex = 1000.0f;
        output[0] = prevIndex;
        for (int i = 1; i < size; ++i) {
            float prevVol = volume[i - 1];
            float currVol = volume[i];
            float prevClose = close[i - 1];
            float value = prevIndex;
            if (currVol > prevVol && prevClose != 0.0f) {
                value = prevIndex * (1.0f + (close[i] - prevClose) / prevClose);
            }
            output[i] = value;
            prevIndex = value;
        }
    }
}
} // namespace

void tacuda::PVI::calculate(const float* close, const float* volume, float* output,
                            int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("PVI: invalid size");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    pviKernel<<<1, 1, 0, stream>>>(close, volume, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::PVI::calculate(const float* input, float* output, int size,
                            cudaStream_t stream) noexcept(false) {
    const float* close = input;
    const float* volume = input + size;
    calculate(close, volume, output, size, stream);
}
