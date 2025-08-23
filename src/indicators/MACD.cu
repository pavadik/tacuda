#include <algorithm>
#include <stdexcept>
#include <indicators/MACD.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

struct EmaCombine {
    __host__ __device__ thrust::complex<float> operator()(const thrust::complex<float>& prev,
                                                          const thrust::complex<float>& curr) const {
        // curr ⊗ prev = (curr.a * prev.a, curr.a * prev.b + curr.b)
        float a = curr.real() * prev.real();
        float b = curr.real() * prev.imag() + curr.imag();
        return {a, b};
    }
};

__global__ void emaPrepKernel(const float* __restrict__ input,
                              thrust::complex<float>* __restrict__ trans,
                              float alpha, float k, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (idx < size) {
        trans[idx - 1] = thrust::complex<float>(k, alpha * input[idx]);
    }
}

__global__ void emaFinalizeKernel(const float* __restrict__ input,
                                  const thrust::complex<float>* __restrict__ trans,
                                  float* __restrict__ ema,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float first = input[0];
    if (idx == 0) {
        ema[0] = first;
    } else if (idx < size) {
        auto t = trans[idx - 1];
        ema[idx] = t.real() * first + t.imag();
    }
}

static void computeEma(const float* input, float* output, int size, int period, cudaStream_t stream) {
    float alpha = 2.0f / (period + 1.0f);
    float k = 1.0f - alpha;
    thrust::complex<float>* trans = static_cast<thrust::complex<float>*>(
        DeviceBufferPool::instance().acquire((size - 1) * sizeof(thrust::complex<float>)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    emaPrepKernel<<<grid, block, 0, stream>>>(input, trans, alpha, k, size);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<thrust::complex<float>> tPtr(trans);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), tPtr, tPtr + size - 1, tPtr, EmaCombine());

    emaFinalizeKernel<<<grid, block, 0, stream>>>(input, trans, output, size);
    CUDA_CHECK(cudaGetLastError());

    DeviceBufferPool::instance().release(trans);
}

__global__ void macdKernel(const float* __restrict__ emaFast,
                           const float* __restrict__ emaSlow,
                           float* __restrict__ macdOut,
                           int slowP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slowP && idx < size) {
        macdOut[idx] = emaFast[idx] - emaSlow[idx];
    }
}

MACD::MACD(int fastPeriod, int slowPeriod)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod) {}

void MACD::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0) {
        throw std::invalid_argument("MACD: invalid periods");
    }
    if (fastPeriod >= slowPeriod) {
        throw std::invalid_argument("MACD: fastPeriod must be < slowPeriod");
    }
    // Warm-up region at the beginning should remain NaN. Initialize the
    // entire output with NaNs and only compute values for indices beyond the
    // slowPeriod.
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    float* emaFast = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));
    float* emaSlow = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));

    computeEma(input, emaFast, size, fastPeriod, stream);
    computeEma(input, emaSlow, size, slowPeriod, stream);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    macdKernel<<<grid, block, 0, stream>>>(emaFast, emaSlow, output, slowPeriod, size);
    CUDA_CHECK(cudaGetLastError());

    DeviceBufferPool::instance().release(emaFast);
    DeviceBufferPool::instance().release(emaSlow);
}
