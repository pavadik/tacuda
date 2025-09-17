#include <algorithm>
#include <indicators/detail/ema_common.cuh>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

namespace tacuda::indicators::detail {
namespace {

struct EmaCombine {
    __host__ __device__ thrust::complex<float> operator()(const thrust::complex<float>& prev,
                                                          const thrust::complex<float>& curr) const {
        float a = curr.real() * prev.real();
        float b = curr.real() * prev.imag() + curr.imag();
        return {a, b};
    }
};

__global__ void emaPrepKernel(const float* __restrict__ input,
                              thrust::complex<float>* __restrict__ trans,
                              float alpha,
                              float k,
                              int size) {
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

}  // namespace

void computeEmaDevice(const float* input,
                      float* output,
                      int size,
                      int period,
                      cudaStream_t stream) {
    if (size <= 0) {
        return;
    }

    float alpha = 2.0f / (period + 1.0f);
    float k = 1.0f - alpha;
    auto trans =
        acquireDeviceBuffer<thrust::complex<float>>(static_cast<size_t>(std::max(0, size - 1)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    emaPrepKernel<<<grid, block, 0, stream>>>(input, trans.get(), alpha, k, size);
    CUDA_CHECK(cudaGetLastError());

    if (size > 1) {
        thrust::device_ptr<thrust::complex<float>> tPtr(trans.get());
        thrust::inclusive_scan(thrust::cuda::par.on(stream), tPtr, tPtr + size - 1, tPtr, EmaCombine());
    }

    emaFinalizeKernel<<<grid, block, 0, stream>>>(input, trans.get(), output, size);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace tacuda::indicators::detail

