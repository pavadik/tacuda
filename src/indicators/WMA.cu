#include <indicators/WMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void wmaKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float weightedSum = 0.0f;
        for (int i = 0; i < period; ++i) {
            weightedSum += input[idx + i] * (period - i);
        }
        float denom = 0.5f * period * (period + 1);
        output[idx] = weightedSum / denom;
    }
}

WMA::WMA(int period) : period(period) {}

void WMA::calculate(const float* input, float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("WMA: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    wmaKernel<<<grid, block>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
