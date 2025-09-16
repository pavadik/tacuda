#include <cmath>
#include <indicators/MININDEX.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void minIndexKernel(const float *__restrict__ input,
                               float *__restrict__ output, int period,
                               int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx <= size - period) {
    float minVal = input[idx];
    int minIdx = 0;
    for (int k = 1; k < period; ++k) {
      float v = input[idx + k];
      if (v < minVal) {
        minVal = v;
        minIdx = k;
      }
    }
    output[idx] = static_cast<float>(minIdx);
  }
}

MININDEX::MININDEX(int period) : period(period) {}

void MININDEX::calculate(const float *input, float *output,
                         int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("MININDEX: invalid period");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  minIndexKernel<<<grid, block, 0, stream>>>(input, output, period, size);
  CUDA_CHECK(cudaGetLastError());
}
