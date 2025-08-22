#include <indicators/TRANGE.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void trangeKernel(const float *__restrict__ high,
                             const float *__restrict__ low,
                             const float *__restrict__ close,
                             float *__restrict__ output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (idx == 0) {
      output[idx] = high[0] - low[0];
    } else {
      float tr1 = high[idx] - low[idx];
      float tr2 = fabsf(high[idx] - close[idx - 1]);
      float tr3 = fabsf(low[idx] - close[idx - 1]);
      float m = tr1 > tr2 ? tr1 : tr2;
      output[idx] = m > tr3 ? m : tr3;
    }
  }
}

void TRANGE::calculate(const float *high, const float *low, const float *close,
                       float *output, int size) noexcept(false) {
  if (size <= 0) {
    throw std::invalid_argument("TRANGE: invalid size");
  }
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  trangeKernel<<<grid, block>>>(high, low, close, output, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void TRANGE::calculate(const float *input, float *output,
                       int size) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  const float *close = input + 2 * size;
  calculate(high, low, close, output, size);
}
