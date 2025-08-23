#include <indicators/MIDPRICE.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void midpriceKernel(const float* __restrict__ high,
                               const float* __restrict__ low,
                               float* __restrict__ output,
                               int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float maxH = high[idx];
        float minL = low[idx];
        for (int k = 1; k < period; ++k) {
            maxH = fmaxf(maxH, high[idx + k]);
            minL = fminf(minL, low[idx + k]);
        }
        output[idx] = 0.5f * (maxH + minL);
    }
}

MIDPRICE::MIDPRICE(int period) : period(period) {}

void MIDPRICE::calculate(const float* high, const float* low, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MIDPRICE: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    midpriceKernel<<<grid, block, 0, stream>>>(high, low, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void MIDPRICE::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}

