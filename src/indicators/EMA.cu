#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void emaKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        const float k = 2.0f / (period + 1.0f);
        float weight = 1.0f;
        float weightedSum = input[idx + period - 1];
        float weightSum = 1.0f;
        for (int i = 1; i < period; ++i) {
            weight *= (1.0f - k);
            weightedSum += input[idx + period - 1 - i] * weight;
            weightSum += weight;
        }
        output[idx] = weightedSum / weightSum;
    }
}

EMA::EMA(int period) : period(period) {}

void EMA::calculate(const float* input, float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("EMA: invalid period");
    }
    // Initialize output with NaNs so unwritten tail remains NaN
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    emaKernel<<<grid, block>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
