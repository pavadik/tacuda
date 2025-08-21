#include <indicators/Momentum.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void momentumKernel(const float* __restrict__ input, float* __restrict__ output,
                               int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - period) {
        output[idx] = input[idx + period] - input[idx];
    }
}

Momentum::Momentum(int period) : period(period) {}

void Momentum::calculate(const float* input, float* output, int size) {
    if (period <= 0 || period >= size) {
        throw std::invalid_argument("Momentum: invalid period");
    }
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    momentumKernel<<<grid, block>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
