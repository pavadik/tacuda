#include <indicators/SAR.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void sarKernel(const float* __restrict__ high,
                          const float* __restrict__ low,
                          float* __restrict__ output,
                          float step,
                          float maxAcc,
                          int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float af = step;
        float ep = high[0];
        float sar = low[0];
        bool longPos = true;
        output[0] = sar;
        for (int i = 1; i < size; ++i) {
            sar = sar + af * (ep - sar);
            if (longPos) {
                sar = fminf(sar, low[i - 1]);
                if (low[i] < sar) {
                    longPos = false;
                    sar = ep;
                    ep = low[i];
                    af = step;
                    sar = fmaxf(sar, high[i - 1]);
                } else {
                    if (high[i] > ep) {
                        ep = high[i];
                        af = fminf(af + step, maxAcc);
                    }
                }
            } else {
                sar = fmaxf(sar, high[i - 1]);
                if (high[i] > sar) {
                    longPos = true;
                    sar = ep;
                    ep = high[i];
                    af = step;
                    sar = fminf(sar, low[i - 1]);
                } else {
                    if (low[i] < ep) {
                        ep = low[i];
                        af = fminf(af + step, maxAcc);
                    }
                }
            }
            output[i] = sar;
        }
    }
}

SAR::SAR(float step, float maxAcceleration)
    : step(step), maxAcceleration(maxAcceleration) {}

void SAR::calculate(const float* high, const float* low,
                    float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("SAR: invalid size");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    sarKernel<<<1, 1, 0, stream>>>(high, low, output, step, maxAcceleration, size);
    CUDA_CHECK(cudaGetLastError());
}

void SAR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}
