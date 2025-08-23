#include <indicators/Correl.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void correlKernel(const float* __restrict__ x,
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
        float cov = 0.0f, varX = 0.0f, varY = 0.0f;
        for (int i = 0; i < period; ++i) {
            float dx = x[idx + i] - meanX;
            float dy = y[idx + i] - meanY;
            cov += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }
        float denom = sqrtf(varX * varY);
        output[idx] = (denom == 0.0f) ? 0.0f : cov / denom;
    }
}

Correl::Correl(int period) : period(period) {}

void Correl::calculate(const float* x, const float* y, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("Correl: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    correlKernel<<<grid, block, 0, stream>>>(x, y, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void Correl::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* x = input;
    const float* y = input + size;
    calculate(x, y, output, size, stream);
}

