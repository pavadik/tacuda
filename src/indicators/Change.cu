#include <indicators/Change.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void changeKernel(const float* __restrict__ input, float* __restrict__ output,
                             int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= period && idx < size) {
        output[idx] = input[idx] - input[idx - period];
    }
}

Change::Change(int period) : period(period) {}

void Change::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period >= size) {
        throw std::invalid_argument("Change: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    changeKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
