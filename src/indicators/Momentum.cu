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

void Momentum::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period >= size) {
        throw std::invalid_argument("Momentum: invalid period");
    }
    // Pre-fill the output buffer with NaNs so that the unwritten tail
    // represents the warm-up region.
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    momentumKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
