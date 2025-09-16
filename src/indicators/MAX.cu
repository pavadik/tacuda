#include <indicators/MAX.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void maxKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float m = input[idx];
        for (int k = 1; k < period; ++k)
            m = fmaxf(m, input[idx + k]);
        output[idx] = m;
    }
}

MAX::MAX(int period) : period(period) {}

void MAX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MAX: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    maxKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
