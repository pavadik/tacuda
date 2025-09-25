#include <indicators/IMI.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

namespace {
__global__ void intradayDiffKernel(const float* __restrict__ open,
                                   const float* __restrict__ close,
                                   float* __restrict__ up,
                                   float* __restrict__ down,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = close[idx] - open[idx];
        if (diff > 0.0f) {
            up[idx] = diff;
            down[idx] = 0.0f;
        } else {
            up[idx] = 0.0f;
            down[idx] = -diff;
        }
    }
}

__global__ void imiKernel(const float* __restrict__ prefixUp,
                          const float* __restrict__ prefixDown,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size - period) {
        int end = idx + period - 1;
        float upPrev = (idx == 0) ? 0.0f : prefixUp[idx - 1];
        float downPrev = (idx == 0) ? 0.0f : prefixDown[idx - 1];
        float sumUp = prefixUp[end] - upPrev;
        float sumDown = prefixDown[end] - downPrev;
        float denom = sumUp + sumDown;
        float value = (denom == 0.0f) ? 0.0f : 100.0f * (sumUp / denom);
        output[idx] = value;
    }
}
} // namespace

tacuda::IMI::IMI(int period) : period(period) {}

void tacuda::IMI::calculate(const float* open, const float* close, float* output,
                            int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("IMI: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    auto up = acquireDeviceBuffer<float>(size);
    auto down = acquireDeviceBuffer<float>(size);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    intradayDiffKernel<<<grid, block, 0, stream>>>(open, close, up.get(), down.get(), size);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<float> upPtr(up.get());
    thrust::device_ptr<float> downPtr(down.get());
    thrust::inclusive_scan(thrust::cuda::par.on(stream), upPtr, upPtr + size, upPtr);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), downPtr, downPtr + size, downPtr);

    imiKernel<<<grid, block, 0, stream>>>(up.get(), down.get(), output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::IMI::calculate(const float* input, float* output, int size,
                            cudaStream_t stream) noexcept(false) {
    const float* open = input;
    // Expects input to be a contiguous array of floats in column-major OHLC layout:
    // [O1, O2, ..., On, H1, ..., Hn, L1, ..., Ln, C1, ..., Cn], each section of length 'size'.
    // Only Open and Close are used: 'open' points to the start, 'close' points to input + 3 * size.
    const float* close = input + 3 * size;
    calculate(open, close, output, size, stream);
}
