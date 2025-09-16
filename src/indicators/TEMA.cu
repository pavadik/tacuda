#include <indicators/TEMA.h>
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

__global__ void temaKernel(const float* __restrict__ ema1,
                           const float* __restrict__ ema2,
                           const float* __restrict__ ema3,
                           float* __restrict__ output,
                           int period,
                           int valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < valid) {
        output[idx] = 3.0f * ema1[idx + 2 * (period - 1)]
                    - 3.0f * ema2[idx + (period - 1)]
                    + ema3[idx];
    }
}

TEMA::TEMA(int period) : period(period) {}

void TEMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || size < 3 * period - 2) {
        throw std::invalid_argument("TEMA: invalid period");
    }

    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    auto ema1 = acquireDeviceBuffer<float>(size);
    auto ema2 = acquireDeviceBuffer<float>(size);
    auto ema3 = acquireDeviceBuffer<float>(size);

    EMA ema(period);
    ema.calculate(input, ema1.get(), size, stream);
    int size2 = size - period + 1;
    ema.calculate(ema1.get(), ema2.get(), size2);
    int size3 = size2 - period + 1;
    ema.calculate(ema2.get(), ema3.get(), size3);

    int valid = size - 3 * period + 3;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    temaKernel<<<grid, block, 0, stream>>>(ema1.get(), ema2.get(), ema3.get(), output, period, valid);
    CUDA_CHECK(cudaGetLastError());
}
