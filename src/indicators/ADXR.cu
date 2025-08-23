#include <indicators/ADXR.h>
#include <indicators/ADX.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

__global__ void adxrKernel(const float* __restrict__ adx,
                           float* __restrict__ output,
                           int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = 3 * period - 1;
    if (idx >= start && idx < size) {
        output[idx] = 0.5f * (adx[idx] + adx[idx - period]);
    }
}

ADXR::ADXR(int period) : period(period) {}

void ADXR::calculate(const float* high, const float* low, const float* close,
                     float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("ADXR: invalid period");
    }
    float* adx = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));

    ADX adxInd(period);
    adxInd.calculate(high, low, close, adx, size, stream);

    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    adxrKernel<<<grid, block, 0, stream>>>(adx, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    DeviceBufferPool::instance().release(adx);
}

void ADXR::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
