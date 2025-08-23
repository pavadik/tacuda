#include <cmath>
#include <indicators/MINMAX.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void minmaxKernel(const float *__restrict__ input,
                             float *__restrict__ minOut,
                             float *__restrict__ maxOut, int period, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx <= size - period) {
    float minVal = input[idx];
    float maxVal = input[idx];
    for (int k = 1; k < period; ++k) {
      float v = input[idx + k];
      minVal = fminf(minVal, v);
      maxVal = fmaxf(maxVal, v);
    }
    minOut[idx] = minVal;
    maxOut[idx] = maxVal;
  }
}

MINMAX::MINMAX(int period) : period(period) {}

void MINMAX::calculate(const float *input, float *output,
                       int size) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("MINMAX: invalid period");
  }
  CUDA_CHECK(cudaMemset(output, 0xFF, 2 * size * sizeof(float)));
  float *minOut = output;
  float *maxOut = output + size;
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  minmaxKernel<<<grid, block>>>(input, minOut, maxOut, period, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
