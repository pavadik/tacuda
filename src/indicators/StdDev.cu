#include <indicators/StdDev.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void stddevKernel(const float* __restrict__ input, float* __restrict__ output,
                             int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= period - 1 && idx < size) {
        float mean = 0.0f;
        for (int j = 0; j < period; ++j)
            mean += input[idx - j];
        mean /= period;
        float sumsq = 0.0f;
        for (int j = 0; j < period; ++j) {
            float diff = input[idx - j] - mean;
            sumsq += diff * diff;
        }
        output[idx] = sqrtf(sumsq / period);
    }
}

tacuda::StdDev::StdDev(int period) : period(period) {}

void tacuda::StdDev::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("StdDev: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    stddevKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
