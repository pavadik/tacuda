#include <indicators/LadderBottom.h>
#include <utils/CandleUtils.h>
#include <utils/CudaUtils.h>

__global__ void ladderBottomKernel(const float* __restrict__ open,
                                   const float* __restrict__ high,
                                   const float* __restrict__ low,
                                   const float* __restrict__ close,
                                   float* __restrict__ output,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 3 && idx < size) {
        output[idx] = is_ladder_bottom(open[idx - 4], high[idx - 4], low[idx - 4], close[idx - 4],
                                       open[idx - 3], high[idx - 3], low[idx - 3], close[idx - 3],
                                       open[idx - 2], high[idx - 2], low[idx - 2], close[idx - 2],
                                       open[idx - 1], high[idx - 1], low[idx - 1], close[idx - 1],
                                       open[idx], high[idx], low[idx], close[idx])
                          ? 1.0f
                          : 0.0f;
    }
}

void tacuda::LadderBottom::calculate(const float* open, const float* high,
                             const float* low, const float* close,
                             float* output, int size, cudaStream_t stream) noexcept(false) {
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    ladderBottomKernel<<<grid, block, 0, stream>>>(open, high, low, close, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::LadderBottom::calculate(const float* input, float* output,
                             int size, cudaStream_t stream) noexcept(false) {
    const float* open = input;
    const float* high = input + size;
    const float* low  = input + 2 * size;
    const float* close= input + 3 * size;
    calculate(open, high, low, close, output, size, stream);
}

