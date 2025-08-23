#include <indicators/MAXINDEX.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void maxIndexKernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float maxVal = input[idx];
        int maxIdx = 0;
        for (int k = 1; k < period; ++k) {
            float v = input[idx + k];
            if (v > maxVal) {
                maxVal = v;
                maxIdx = k;
            }
        }
        output[idx] = static_cast<float>(maxIdx);
    }
}

MAXINDEX::MAXINDEX(int period) : period(period) {}

void MAXINDEX::calculate(const float* input, float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MAXINDEX: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    maxIndexKernel<<<grid, block>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

