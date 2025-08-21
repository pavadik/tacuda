#include <indicators/SMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void smaKernel(const float* __restrict__ input, float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 1024; ++i) {
            if (i >= period) break;
            sum += input[idx + i];
        }
        output[idx] = sum / period;
    }
}

SMA::SMA(int period) : period(period) {}

void SMA::calculate(const float* input, float* output, int size) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("SMA: invalid period");
    }
    // Initialize the entire output array with NaNs so that any unwritten
    // warm-up region retains the expected NaN semantics.
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    smaKernel<<<grid, block>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
