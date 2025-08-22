#include <indicators/SUM.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <utils/CudaUtils.h>

__global__ void sumKernelPrefix(const float *__restrict__ prefix,
                                float *__restrict__ output, int period,
                                int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx <= size - period) {
    float prev = (idx == 0) ? 0.0f : prefix[idx - 1];
    output[idx] = prefix[idx + period - 1] - prev;
  }
}

SUM::SUM(int period) : period(period) {}

void SUM::calculate(const float *input, float *output,
                    int size) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("SUM: invalid period");
  }
  CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

  float *prefix = nullptr;
  CUDA_CHECK(cudaMalloc(&prefix, size * sizeof(float)));
  thrust::device_ptr<const float> inPtr(input);
  thrust::device_ptr<float> prePtr(prefix);
  thrust::inclusive_scan(inPtr, inPtr + size, prePtr);

  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  sumKernelPrefix<<<grid, block>>>(prefix, output, period, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(prefix));
}
