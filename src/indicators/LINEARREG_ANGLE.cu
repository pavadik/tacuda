#include <indicators/LINEARREG_ANGLE.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void linearregAngleKernel(const float* __restrict__ in,
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
        const float rad2deg = 180.0f / 3.14159265358979323846f;
        out[idx] = atanf(slope) * rad2deg;
    }
}

LINEARREG_ANGLE::LINEARREG_ANGLE(int period) : period(period) {}

void LINEARREG_ANGLE::calculate(const float* input, float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("LINEARREG_ANGLE: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    linearregAngleKernel<<<grid, block>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

