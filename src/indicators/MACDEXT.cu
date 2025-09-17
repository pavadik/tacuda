#include <algorithm>
#include <stdexcept>
#include <indicators/MACDEXT.h>
#include <indicators/detail/ema_common.cuh>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

using tacuda::indicators::detail::computeEmaDevice;

__global__ void smaKernelEnd(const float* __restrict__ prefix,
                             float* __restrict__ output,
                             int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float prev = (idx >= period) ? prefix[idx - period] : 0.0f;
        float sum = prefix[idx] - prev;
        int steps = (idx >= period) ? period : (idx + 1);
        output[idx] = sum / steps;
    }
}

static void computeSma(const float* input, float* output, int size, int period, cudaStream_t stream) {
    if (size <= 0) {
        return;
    }
    auto prefix = acquireDeviceBuffer<float>(size);
    thrust::device_ptr<const float> inPtr(input);
    thrust::device_ptr<float> prePtr(prefix.get());
    thrust::inclusive_scan(thrust::cuda::par.on(stream), inPtr, inPtr + size, prePtr);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    smaKernelEnd<<<grid, block, 0, stream>>>(prefix.get(), output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void macdLineKernel(const float* __restrict__ maFast,
                               const float* __restrict__ maSlow,
                               float* __restrict__ macdOut,
                               int slowP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        macdOut[idx] = maFast[idx] - maSlow[idx];
    }
}

__global__ void histKernel(const float* __restrict__ macd,
                           const float* __restrict__ signal,
                           float* __restrict__ hist,
                           int slowP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        hist[idx] = macd[idx] - signal[idx];
    }
}

tacuda::MACDEXT::MACDEXT(int fastPeriod, int slowPeriod, int signalPeriod, tacuda::MAType type)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod), signalPeriod(signalPeriod), type(type) {}

void tacuda::MACDEXT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0) {
        throw std::invalid_argument("MACD: invalid periods");
    }
    if (fastPeriod >= slowPeriod) {
        throw std::invalid_argument("MACD: fastPeriod must be < slowPeriod");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 3 * size * sizeof(float), stream));
    float* macd = output;
    float* signal = output + size;
    float* hist = output + 2 * size;

    auto maFast = acquireDeviceBuffer<float>(size);
    auto maSlow = acquireDeviceBuffer<float>(size);

    if (type == tacuda::MAType::EMA) {
        computeEmaDevice(input, maFast.get(), size, fastPeriod, stream);
        computeEmaDevice(input, maSlow.get(), size, slowPeriod, stream);
    } else {
        computeSma(input, maFast.get(), size, fastPeriod, stream);
        computeSma(input, maSlow.get(), size, slowPeriod, stream);
    }

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    macdLineKernel<<<grid, block, 0, stream>>>(maFast.get(), maSlow.get(), macd, slowPeriod, size);
    CUDA_CHECK(cudaGetLastError());

    if (type == tacuda::MAType::EMA) {
        computeEmaDevice(macd + slowPeriod, signal + slowPeriod, size - slowPeriod, signalPeriod, stream);
    } else {
        computeSma(macd + slowPeriod, signal + slowPeriod, size - slowPeriod, signalPeriod, stream);
    }

    histKernel<<<grid, block, 0, stream>>>(macd, signal, hist, slowPeriod, size);
    CUDA_CHECK(cudaGetLastError());
}
