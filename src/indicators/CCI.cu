#include <indicators/CCI.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void cciKernel(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          float* __restrict__ output,
                          int period, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float nan = nanf("");
        for (int i = 0; i < size; ++i) {
            output[i] = nan;
        }
        for (int i = period - 1; i < size; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < period; ++j) {
                int idx = i - j;
                sum += (high[idx] + low[idx] + close[idx]) / 3.0f;
            }
            float sma = sum / period;
            float dev = 0.0f;
            for (int j = 0; j < period; ++j) {
                int idx = i - j;
                float tp = (high[idx] + low[idx] + close[idx]) / 3.0f;
                dev += fabsf(tp - sma);
            }
            float md = dev / period;
            float tp_cur = (high[i] + low[i] + close[i]) / 3.0f;
            output[i] = (md == 0.0f) ? 0.0f : (tp_cur - sma) / (0.015f * md);
        }
    }
}

CCI::CCI(int period) : period(period) {}

void CCI::calculate(const float* high, const float* low, const float* close,
                    float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("CCI: invalid period");
    }
    cciKernel<<<1, 1, 0, stream>>>(high, low, close, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void CCI::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
