#include <algorithm>
#include <stdexcept>
#include <indicators/MACDEXT.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

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
    float* prefix = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));
    thrust::device_ptr<const float> inPtr(input);
    thrust::device_ptr<float> prePtr(prefix);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), inPtr, inPtr + size, prePtr);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    smaKernelEnd<<<grid, block, 0, stream>>>(prefix, output, period, size);
    CUDA_CHECK(cudaGetLastError());

    DeviceBufferPool::instance().release(prefix);
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

MACDEXT::MACDEXT(int fastPeriod, int slowPeriod, int signalPeriod, MAType type)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod), signalPeriod(signalPeriod), type(type) {}

void MACDEXT::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0) {
        throw std::invalid_argument("MACD: invalid periods");
    }
    if (fastPeriod >= slowPeriod) {
        throw std::invalid_argument("MACD: fastPeriod must be < slowPeriod");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, 3 * size * sizeof(float)));
    float* macd = output;
    float* signal = output + size;
    float* hist = output + 2 * size;

    float* maFast = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));
    float* maSlow = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));

    if (type == MAType::EMA) {
        computeEma(input, maFast, size, fastPeriod, stream);
        computeEma(input, maSlow, size, slowPeriod, stream);
    } else {
        computeSma(input, maFast, size, fastPeriod, stream);
        computeSma(input, maSlow, size, slowPeriod, stream);
    }

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    macdLineKernel<<<grid, block, 0, stream>>>(maFast, maSlow, macd, slowPeriod, size);
    CUDA_CHECK(cudaGetLastError());

    if (type == MAType::EMA) {
        computeEma(macd + slowPeriod, signal + slowPeriod, size - slowPeriod, signalPeriod, stream);
    } else {
        computeSma(macd + slowPeriod, signal + slowPeriod, size - slowPeriod, signalPeriod, stream);
    }

    histKernel<<<grid, block, 0, stream>>>(macd, signal, hist, slowPeriod, size);
    CUDA_CHECK(cudaGetLastError());

    DeviceBufferPool::instance().release(maFast);
    DeviceBufferPool::instance().release(maSlow);
}
