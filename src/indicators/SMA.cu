#include "../../include/indicators/SMA.h"
#include "../../include/utils/CudaUtils.h"
#include <stdexcept>

__global__ void smaKernel(const float* __restrict__ input, float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 1024; ++i) {
            if (i >= period) break;
            sum += input[idx + i];
        }
        output[idx] = sum / period;
    }
}

SMA::SMA(int period) : period(period) {}

void SMA::calculate(const float* input, float* output, int size) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("SMA: invalid period");
    }
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    smaKernel<<<grid, block>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
