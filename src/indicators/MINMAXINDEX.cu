#include <cmath>
#include <indicators/MINMAXINDEX.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void minmaxIndexKernel(const float *__restrict__ input,
                                  float *__restrict__ minOut,
                                  float *__restrict__ maxOut, int period,
                                  int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx <= size - period) {
    float minVal = input[idx];
    float maxVal = input[idx];
    int minIdx = 0;
    int maxIdx = 0;
    for (int k = 1; k < period; ++k) {
      float v = input[idx + k];
      if (v < minVal) {
        minVal = v;
        minIdx = k;
      }
      if (v > maxVal) {
        maxVal = v;
        maxIdx = k;
      }
    }
    minOut[idx] = static_cast<float>(minIdx);
    maxOut[idx] = static_cast<float>(maxIdx);
  }
}

MINMAXINDEX::MINMAXINDEX(int period) : period(period) {}

void MINMAXINDEX::calculate(const float *input, float *output,
                            int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("MINMAXINDEX: invalid period");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * size * sizeof(float), stream));
  float *minOut = output;
  float *maxOut = output + size;
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  minmaxIndexKernel<<<grid, block, 0, stream>>>(input, minOut, maxOut, period, size);
  CUDA_CHECK(cudaGetLastError());
}
