#include <indicators/BBANDS.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cmath>

__global__ void squareKernel(const float* __restrict__ input,
                             float* __restrict__ squared,
                             int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = input[idx];
        squared[idx] = v * v;
    }
}

__global__ void bbandsKernel(const float* __restrict__ prefix,
                             const float* __restrict__ prefixSq,
                             float* __restrict__ upper,
                             float* __restrict__ middle,
                             float* __restrict__ lower,
                             int period, int size,
                             float upMult, float downMult) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        float prev = (idx == 0) ? 0.0f : prefix[idx - 1];
        float prevSq = (idx == 0) ? 0.0f : prefixSq[idx - 1];
        float sum = prefix[idx + period - 1] - prev;
        float sumSq = prefixSq[idx + period - 1] - prevSq;
        float mean = sum / period;
        float variance = sumSq / period - mean * mean;
        variance = variance > 0.0f ? variance : 0.0f;
        float stddev = sqrtf(variance);
        middle[idx] = mean;
        upper[idx] = mean + upMult * stddev;
        lower[idx] = mean - downMult * stddev;
    }
}

BBANDS::BBANDS(int period, float upperMultiplier, float lowerMultiplier)
    : period(period), upperMultiplier(upperMultiplier), lowerMultiplier(lowerMultiplier) {}

void BBANDS::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("BBANDS: invalid period");
    }
    // Initialize outputs with NaNs so unwritten tail retains NaN semantics
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 3 * size * sizeof(float), stream));

    auto prefix = acquireDeviceBuffer<float>(size);
    auto squared = acquireDeviceBuffer<float>(size);
    auto prefixSq = acquireDeviceBuffer<float>(size);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    squareKernel<<<grid, block, 0, stream>>>(input, squared.get(), size);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<const float> inPtr(input);
    thrust::device_ptr<float> prePtr(prefix.get());
    thrust::inclusive_scan(inPtr, inPtr + size, prePtr);

    thrust::device_ptr<float> sqPtr(squared.get());
    thrust::device_ptr<float> preSqPtr(prefixSq.get());
    thrust::inclusive_scan(sqPtr, sqPtr + size, preSqPtr);

    float* upper = output;
    float* middle = output + size;
    float* lower = output + 2 * size;
    bbandsKernel<<<grid, block, 0, stream>>>(prefix.get(), prefixSq.get(), upper, middle, lower,
                                  period, size, upperMultiplier, lowerMultiplier);
    CUDA_CHECK(cudaGetLastError());
}

