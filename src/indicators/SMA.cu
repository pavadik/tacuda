#include <indicators/SMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__global__ void smaKernelPrefix(const float* __restrict__ prefix,
                                float* __restrict__ output,
                                int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float prev = (idx == 0) ? 0.0f : prefix[idx - 1];
        float sum = prefix[idx + period - 1] - prev;
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

    // Compute prefix sums of the input using Thrust.
    float* prefix = nullptr;
    CUDA_CHECK(cudaMalloc(&prefix, size * sizeof(float)));
    thrust::device_ptr<const float> inPtr(input);
    thrust::device_ptr<float> prePtr(prefix);
    thrust::inclusive_scan(inPtr, inPtr + size, prePtr);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    smaKernelPrefix<<<grid, block>>>(prefix, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(prefix));
}
