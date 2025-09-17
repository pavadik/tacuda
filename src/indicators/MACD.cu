#include <stdexcept>
#include <indicators/MACD.h>
#include <indicators/detail/ema_common.cuh>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
using tacuda::indicators::detail::computeEmaDevice;

__global__ void macdKernel(const float* __restrict__ emaFast,
                           const float* __restrict__ emaSlow,
                           float* __restrict__ macdOut,
                           int slowP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        macdOut[idx] = emaFast[idx] - emaSlow[idx];
    }
}

tacuda::MACD::MACD(int fastPeriod, int slowPeriod)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod) {}

void tacuda::MACD::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0) {
        throw std::invalid_argument("MACD: invalid periods");
    }
    if (fastPeriod >= slowPeriod) {
        throw std::invalid_argument("MACD: fastPeriod must be < slowPeriod");
    }
    // Warm-up region at the beginning should remain NaN. Initialize the
    // entire output with NaNs and only compute values for indices beyond the
    // slowPeriod.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    auto emaFast = acquireDeviceBuffer<float>(size);
    auto emaSlow = acquireDeviceBuffer<float>(size);

    computeEmaDevice(input, emaFast.get(), size, fastPeriod, stream);
    computeEmaDevice(input, emaSlow.get(), size, slowPeriod, stream);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    macdKernel<<<grid, block, 0, stream>>>(emaFast.get(), emaSlow.get(), output, slowPeriod, size);
    CUDA_CHECK(cudaGetLastError());
}
