#include <indicators/HomingPigeon.h>
#include <utils/CandleUtils.h>
#include <utils/CudaUtils.h>

__global__ void homingPigeonKernel(const float* __restrict__ open,
                                   const float* __restrict__ high,
                                   const float* __restrict__ low,
                                   const float* __restrict__ close,
                                   float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < size) {
        output[idx] = is_homing_pigeon(open[idx - 1], high[idx - 1],
                                       low[idx - 1], close[idx - 1], open[idx],
                                       high[idx], low[idx], close[idx])
                          ? 1.0f
                          : 0.0f;
    }
}

void HomingPigeon::calculate(const float* open, const float* high,
                             const float* low, const float* close,
                             float* output, int size, cudaStream_t stream) noexcept(false) {
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    homingPigeonKernel<<<grid, block, 0, stream>>>(open, high, low, close, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void HomingPigeon::calculate(const float* input, float* output,
                             int size, cudaStream_t stream) noexcept(false) {
    const float* open = input;
    const float* high = input + size;
    const float* low  = input + 2 * size;
    const float* close= input + 3 * size;
    calculate(open, high, low, close, output, size, stream);
}

