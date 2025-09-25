#include <indicators/AVGDEV.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

namespace {
__global__ void avgdevKernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float sum = 0.0f;
        for (int i = 0; i < period; ++i) {
            sum += input[idx + i];
        }
        float mean = sum / period;
        float dev = 0.0f;
        for (int i = 0; i < period; ++i) {
            dev += fabsf(input[idx + i] - mean);
        }
        output[idx] = dev / period;
    }
}
} // namespace

tacuda::AVGDEV::AVGDEV(int period) : period(period) {}

void tacuda::AVGDEV::calculate(const float* input, float* output, int size,
                               cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("AVGDEV: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    avgdevKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
