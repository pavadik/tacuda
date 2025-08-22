#include <indicators/MFI.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void mfiKernel(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          const float* __restrict__ volume,
                          float* __restrict__ signedMF,
                          float* __restrict__ output,
                          int period, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float prevTP = (high[0] + low[0] + close[0]) / 3.0f;
        signedMF[0] = 0.0f;
        float posSum = 0.0f;
        float negSum = 0.0f;
        for (int i = 1; i < size; ++i) {
            float tp = (high[i] + low[i] + close[i]) / 3.0f;
            float mf = tp * volume[i];
            float sf = 0.0f;
            if (tp > prevTP) {
                sf = mf;
                posSum += mf;
            } else if (tp < prevTP) {
                sf = -mf;
                negSum += mf;
            }
            signedMF[i] = sf;
            prevTP = tp;
            if (i >= period) {
                float old = signedMF[i - period];
                if (old > 0.0f) {
                    posSum -= old;
                } else {
                    negSum += old; // old is negative
                }
            }
            if (i >= period) {
                float mfi;
                if (negSum == 0.0f) {
                    mfi = (posSum == 0.0f) ? 0.0f : 100.0f;
                } else {
                    float ratio = posSum / negSum;
                    mfi = 100.0f - 100.0f / (1.0f + ratio);
                }
                output[i] = mfi;
            }
        }
    }
}

MFI::MFI(int period) : period(period) {}

void MFI::calculate(const float* high, const float* low, const float* close,
                    const float* volume, float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MFI: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    float* signedMF = nullptr;
    CUDA_CHECK(cudaMalloc(&signedMF, size * sizeof(float)));
    mfiKernel<<<1,1>>>(high, low, close, volume, signedMF, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(signedMF));
}

void MFI::calculate(const float* input, float* output, int size) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    const float* volume = input + 3 * size;
    calculate(high, low, close, volume, output, size);
}
