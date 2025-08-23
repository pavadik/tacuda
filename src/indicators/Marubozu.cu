#include <indicators/Marubozu.h>
#include <utils/CandleUtils.h>
#include <utils/CudaUtils.h>

__global__ void marubozuKernel(const float* __restrict__ open,
                               const float* __restrict__ high,
                               const float* __restrict__ low,
                               const float* __restrict__ close,
                               float* __restrict__ output,
                               int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = is_marubozu(open[idx], high[idx], low[idx], close[idx]) ? 1.0f : 0.0f;
    }
}

void Marubozu::calculate(const float* open, const float* high,
                         const float* low, const float* close,
                         float* output, int size) noexcept(false) {
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    marubozuKernel<<<grid, block>>>(open, high, low, close, output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Marubozu::calculate(const float* input, float* output,
                         int size) noexcept(false) {
    const float* open = input;
    const float* high = input + size;
    const float* low  = input + 2 * size;
    const float* close= input + 3 * size;
    calculate(open, high, low, close, output, size);
}

