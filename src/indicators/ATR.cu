#include <indicators/ATR.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void atrKernel(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          float* __restrict__ output,
                          int period, float initial, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float prevClose = close[0];
        float atr = initial;
        if (initial <= 0.0f) {
            float sum = 0.0f;
            for (int i = 0; i < period; ++i) {
                float tr = fmaxf(high[i] - low[i],
                                 fmaxf(fabsf(high[i] - prevClose),
                                       fabsf(low[i] - prevClose)));
                sum += tr;
                prevClose = close[i];
            }
            atr = sum / period;
        } else {
            for (int i = 0; i < period; ++i) {
                float tr = fmaxf(high[i] - low[i],
                                 fmaxf(fabsf(high[i] - prevClose),
                                       fabsf(low[i] - prevClose)));
                prevClose = close[i];
            }
        }
        output[period - 1] = atr;
        for (int i = period; i < size; ++i) {
            float tr = fmaxf(high[i] - low[i],
                             fmaxf(fabsf(high[i] - prevClose),
                                   fabsf(low[i] - prevClose)));
            atr = (atr * (period - 1) + tr) / period;
            output[i] = atr;
            prevClose = close[i];
        }
    }
}

ATR::ATR(int period, float initial) : period(period), initial(initial) {}

void ATR::calculate(const float* high, const float* low, const float* close,
                    float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("ATR: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    atrKernel<<<1, 1>>>(high, low, close, output, period, initial, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ATR::calculate(const float* input, float* output, int size) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size);
}
