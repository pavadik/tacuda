#include <algorithm>
#include <indicators/MAVP.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>
#include <vector>

namespace {
__global__ void clampPeriodsKernel(const float* __restrict__ periods,
                                   int* __restrict__ clamped,
                                   int size, int minP, int maxP) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int p = static_cast<int>(periods[idx]);
        if (p < minP) p = minP;
        if (p > maxP) p = maxP;
        clamped[idx] = p;
    }
}

__global__ void scatterMavpKernel(const float* __restrict__ values,
                                  const int* __restrict__ periods,
                                  float* __restrict__ output,
                                  int size, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && periods[idx] == period) {
        if (idx <= size - period) {
            output[idx] = values[idx];
        }
    }
}
} // namespace

tacuda::MAVP::MAVP(int minPeriod, int maxPeriod, MAType type)
    : minPeriod(minPeriod), maxPeriod(maxPeriod), type(type) {
    if (minPeriod <= 0 || maxPeriod <= 0 || minPeriod > maxPeriod) {
        throw std::invalid_argument("MAVP: invalid period range");
    }
}

void tacuda::MAVP::calculate(const float* values, const float* periods, float* output,
                             int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("MAVP: invalid size");
    }
    if (maxPeriod > size) {
        throw std::invalid_argument("MAVP: maxPeriod exceeds size");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    auto devicePeriods = acquireDeviceBuffer<int>(size);
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    clampPeriodsKernel<<<grid, block, 0, stream>>>(periods, devicePeriods.get(), size, minPeriod, maxPeriod);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int> hostPeriods(size);
    CUDA_CHECK(cudaMemcpy(hostPeriods.data(), devicePeriods.get(),
                          size * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> uniquePeriods = hostPeriods;
    std::sort(uniquePeriods.begin(), uniquePeriods.end());
    uniquePeriods.erase(std::unique(uniquePeriods.begin(), uniquePeriods.end()), uniquePeriods.end());

    for (int period : uniquePeriods) {
        tacuda::MA ma(period, type);
        auto tmp = acquireDeviceBuffer<float>(size);
        ma.calculate(values, tmp.get(), size, stream);
        scatterMavpKernel<<<grid, block, 0, stream>>>(tmp.get(), devicePeriods.get(), output, size, period);
        CUDA_CHECK(cudaGetLastError());
    }
}

void tacuda::MAVP::calculate(const float* input, float* output, int size,
                             cudaStream_t stream) noexcept(false) {
    const float* values = input;
    const float* periods = input + size;
    calculate(values, periods, output, size, stream);
}
