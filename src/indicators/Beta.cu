#include <indicators/Beta.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void betaKernel(const float* __restrict__ x,
                           const float* __restrict__ y,
                           float* __restrict__ output,
                           int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - period) {
        float sumX = 0.0f, sumY = 0.0f;
        for (int i = 0; i < period; ++i) {
            sumX += x[idx + i];
            sumY += y[idx + i];
        }
        float meanX = sumX / period;
        float meanY = sumY / period;
        float cov = 0.0f, varY = 0.0f;
        for (int i = 0; i < period; ++i) {
            float dx = x[idx + i] - meanX;
            float dy = y[idx + i] - meanY;
            cov += dx * dy;
            varY += dy * dy;
        }
        output[idx] = (varY == 0.0f) ? 0.0f : cov / varY;
    }
}

Beta::Beta(int period) : period(period) {}

void Beta::calculate(const float* x, const float* y, float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("Beta: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    betaKernel<<<grid, block>>>(x, y, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Beta::calculate(const float* input, float* output, int size) noexcept(false) {
    const float* x = input;
    const float* y = input + size;
    calculate(x, y, output, size);
}

