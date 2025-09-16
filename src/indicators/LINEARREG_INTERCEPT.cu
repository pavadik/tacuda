#include <indicators/LINEARREG_INTERCEPT.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void linearregInterceptKernel(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float sumY = 0.0f, sumXY = 0.0f;
        for (int i = 0; i < period; ++i) {
            float y = in[idx + i];
            sumY += y;
            sumXY += i * y;
        }
        float sumX = 0.5f * period * (period - 1);
        float sumX2 = (period - 1) * period * (2 * period - 1) / 6.0f;
        float denom = period * sumX2 - sumX * sumX;
        float slope = (period * sumXY - sumX * sumY) / denom;
        float intercept = (sumY - slope * sumX) / period;
        out[idx] = intercept;
    }
}

LINEARREG_INTERCEPT::LINEARREG_INTERCEPT(int period) : period(period) {}

void LINEARREG_INTERCEPT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("LINEARREG_INTERCEPT: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    linearregInterceptKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

