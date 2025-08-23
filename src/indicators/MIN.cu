#include <indicators/MIN.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void minKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float m = input[idx];
        for (int k = 1; k < period; ++k)
            m = fminf(m, input[idx + k]);
        output[idx] = m;
    }
}

MIN::MIN(int period) : period(period) {}

void MIN::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MIN: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    minKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
