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

DEMA::DEMA(int period) : period(period) {}

void DEMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || size < 2 * period - 1) {
        throw std::invalid_argument("DEMA: invalid period");
    }

    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    float* ema1 = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));
    float* ema2 = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));

    EMA ema(period);
    ema.calculate(input, ema1, size, stream);
    int size2 = size - period + 1;
    ema.calculate(ema1, ema2, size2);

    int valid = size - 2 * period + 2;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    demaKernel<<<grid, block, 0, stream>>>(ema1, ema2, output, period, valid);
    CUDA_CHECK(cudaGetLastError());

    DeviceBufferPool::instance().release(ema1);
    DeviceBufferPool::instance().release(ema2);
}
