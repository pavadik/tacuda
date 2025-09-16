#include <indicators/CCI.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void cciKernel(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= period - 1 && idx < size) {
        float sum = 0.0f;
        for (int j = 0; j < period; ++j) {
            int cur = idx - j;
            sum += (high[cur] + low[cur] + close[cur]) / 3.0f;
        }
        float sma = sum / period;
        float dev = 0.0f;
        for (int j = 0; j < period; ++j) {
            int cur = idx - j;
            float tp = (high[cur] + low[cur] + close[cur]) / 3.0f;
            dev += fabsf(tp - sma);
        }
        float md = dev / period;
        float tp_cur = (high[idx] + low[idx] + close[idx]) / 3.0f;
        output[idx] = (md == 0.0f) ? 0.0f : (tp_cur - sma) / (0.015f * md);
    }
}

tacuda::CCI::CCI(int period) : period(period) {}

void tacuda::CCI::calculate(const float* high, const float* low, const float* close,
                    float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("CCI: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    cciKernel<<<grid, block, 0, stream>>>(high, low, close, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::CCI::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
