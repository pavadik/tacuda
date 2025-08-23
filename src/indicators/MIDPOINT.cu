#include <indicators/MIDPOINT.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void midpointKernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float mn = input[idx];
        float mx = input[idx];
        for (int k = 1; k < period; ++k) {
            float v = input[idx + k];
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        output[idx] = 0.5f * (mx + mn);
    }
}

MIDPOINT::MIDPOINT(int period) : period(period) {}

void MIDPOINT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MIDPOINT: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    midpointKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

