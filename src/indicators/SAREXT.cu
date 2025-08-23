#include <indicators/SAREXT.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void sarextKernel(const float* __restrict__ high,
                             const float* __restrict__ low,
                             float* __restrict__ output,
                             float startValue, float offset,
                             float accInitLong, float accLong, float accMaxLong,
                             float accInitShort, float accShort, float accMaxShort,
                             int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sar = (startValue != 0.0f) ? startValue : low[0];
        bool longPos = true;
        float af = accInitLong;
        float ep = high[0];
        output[0] = sar;
        for (int i = 1; i < size; ++i) {
            sar = sar + af * (ep - sar);
            if (longPos) {
                sar = fminf(sar, low[i - 1]);
                if (low[i] < sar) {
                    longPos = false;
                    sar = ep + offset;
                    ep = low[i];
                    af = accInitShort;
                    sar = fmaxf(sar, high[i - 1]);
                } else {
                    if (high[i] > ep) {
                        ep = high[i];
                        af = fminf(af + accLong, accMaxLong);
                    }
                }
            } else {
                sar = fmaxf(sar, high[i - 1]);
                if (high[i] > sar) {
                    longPos = true;
                    sar = ep - offset;
                    ep = high[i];
                    af = accInitLong;
                    sar = fminf(sar, low[i - 1]);
                } else {
                    if (low[i] < ep) {
                        ep = low[i];
                        af = fminf(af + accShort, accMaxShort);
                    }
                }
            }
            output[i] = sar;
        }
    }
}

SAREXT::SAREXT(float startValue, float offsetOnReverse,
               float accInitLong, float accLong, float accMaxLong,
               float accInitShort, float accShort, float accMaxShort)
    : startValue(startValue), offsetOnReverse(offsetOnReverse),
      accInitLong(accInitLong), accLong(accLong), accMaxLong(accMaxLong),
      accInitShort(accInitShort), accShort(accShort), accMaxShort(accMaxShort) {}

void SAREXT::calculate(const float* high, const float* low,
                       float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("SAREXT: invalid size");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    sarextKernel<<<1, 1, 0, stream>>>(high, low, output, startValue, offsetOnReverse,
                          accInitLong, accLong, accMaxLong,
                          accInitShort, accShort, accMaxShort, size);
    CUDA_CHECK(cudaGetLastError());
}

void SAREXT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}
