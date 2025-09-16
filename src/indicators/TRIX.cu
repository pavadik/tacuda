#include <indicators/TRIX.h>
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

__global__ void trixKernel(const float* __restrict__ ema3,
                           float* __restrict__ output,
                           int valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < valid && ema3[idx] != 0.0f) {
        float prev = ema3[idx];
        float curr = ema3[idx + 1];
        output[idx] = ((curr - prev) / prev) * 100.0f;
    }
}

TRIX::TRIX(int period) : period(period) {}

void TRIX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || size < 3 * period - 1) {
        throw std::invalid_argument("TRIX: invalid period");
    }

    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    auto ema1 = acquireDeviceBuffer<float>(size);
    auto ema2 = acquireDeviceBuffer<float>(size);
    auto ema3 = acquireDeviceBuffer<float>(size);

    EMA ema(period);
    ema.calculate(input, ema1.get(), size, stream);
    int size2 = size - period + 1;
    ema.calculate(ema1.get(), ema2.get(), size2);
    int size3 = size2 - period + 1;
    ema.calculate(ema2.get(), ema3.get(), size3);

    int valid = size3 - 1;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    trixKernel<<<grid, block, 0, stream>>>(ema3.get(), output, valid);
    CUDA_CHECK(cudaGetLastError());
}
