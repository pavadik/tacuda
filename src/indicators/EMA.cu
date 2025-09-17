#include <cmath>
#include <indicators/EMA.h>
#include <indicators/detail/ema_common.cuh>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

namespace {

__global__ void emaWindowKernel(const float* __restrict__ input,
                                const float* __restrict__ emaSeries,
                                float* __restrict__ output,
                                float kPow,
                                float denom,
                                int period,
                                int valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < valid) {
        float prev = (idx == 0) ? input[0] : emaSeries[idx - 1];
        float curr = emaSeries[idx + period - 1];
        output[idx] = (curr - kPow * prev) / denom;
    }
}

}  // namespace

tacuda::EMA::EMA(int period) : period(period) {}

void tacuda::EMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("EMA: invalid period");
    }
    // Initialize output with NaNs so unwritten tail remains NaN
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    auto emaSeries = acquireDeviceBuffer<float>(size);
    tacuda::indicators::detail::computeEmaDevice(input, emaSeries.get(), size, period, stream);

    int valid = size - period + 1;
    float alpha = 2.0f / (period + 1.0f);
    float k = 1.0f - alpha;
    float kPow = static_cast<float>(std::pow(k, period));
    float denom = 1.0f - kPow;

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    emaWindowKernel<<<grid, block, 0, stream>>>(input, emaSeries.get(), output, kPow, denom, period, valid);
    CUDA_CHECK(cudaGetLastError());
}
