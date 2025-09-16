#include <indicators/DEMA.h>
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

__global__ void demaKernel(const float* __restrict__ ema1,
                           const float* __restrict__ ema2,
                           float* __restrict__ output,
                           int period,
                           int valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < valid) {
        output[idx] = 2.0f * ema1[idx + period - 1] - ema2[idx];
    }
}

tacuda::DEMA::DEMA(int period) : period(period) {}

void tacuda::DEMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || size < 2 * period - 1) {
        throw std::invalid_argument("DEMA: invalid period");
    }

    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    auto ema1 = acquireDeviceBuffer<float>(size);
    auto ema2 = acquireDeviceBuffer<float>(size);

    tacuda::EMA ema(period);
    ema.calculate(input, ema1.get(), size, stream);
    int size2 = size - period + 1;
    ema.calculate(ema1.get(), ema2.get(), size2);

    int valid = size - 2 * period + 2;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    demaKernel<<<grid, block, 0, stream>>>(ema1.get(), ema2.get(), output, period, valid);
    CUDA_CHECK(cudaGetLastError());
}
