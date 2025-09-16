#include <indicators/RSI.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <stdexcept>

__global__ void diffKernel(const float* __restrict__ input,
                           float* __restrict__ gainPrefix,
                           float* __restrict__ lossPrefix,
                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        float diff = input[idx + 1] - input[idx];
        gainPrefix[idx + 1] = diff > 0.0f ? diff : 0.0f;
        lossPrefix[idx + 1] = diff < 0.0f ? -diff : 0.0f;
    }
}

__global__ void rsiKernelPrefix(const float* __restrict__ gainPrefix,
                                const float* __restrict__ lossPrefix,
                                float* __restrict__ output,
                                int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - period) {
        float sumGain = gainPrefix[idx + period] - gainPrefix[idx];
        float sumLoss = lossPrefix[idx + period] - lossPrefix[idx];
        float avgGain = sumGain / period;
        float avgLoss = sumLoss / period;
        float rsi;
        if (avgLoss == 0.0f) {
            rsi = (avgGain == 0.0f) ? 50.0f : 100.0f;
        } else if (avgGain == 0.0f) {
            rsi = 0.0f;
        } else {
            float rs = avgGain / avgLoss;
            rsi = 100.0f - 100.0f / (1.0f + rs);
        }
        output[idx] = rsi;
    }
}

RSI::RSI(int period) : period(period) {}

void RSI::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period >= size) {
        throw std::invalid_argument("RSI: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    auto gain = acquireDeviceBuffer<float>(size);
    auto loss = acquireDeviceBuffer<float>(size);

    CUDA_CHECK(cudaMemsetAsync(gain.get(), 0, size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(loss.get(), 0, size * sizeof(float), stream));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    diffKernel<<<grid, block, 0, stream>>>(input, gain.get(), loss.get(), size);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<float> gainPtr(gain.get());
    thrust::device_ptr<float> lossPtr(loss.get());
    thrust::inclusive_scan(thrust::cuda::par.on(stream), gainPtr, gainPtr + size, gainPtr);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), lossPtr, lossPtr + size, lossPtr);

    rsiKernelPrefix<<<grid, block, 0, stream>>>(gain.get(), loss.get(), output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
