#include <indicators/WILLR.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void willrKernel(const float *__restrict__ high,
                            const float *__restrict__ low,
                            const float *__restrict__ close,
                            float *__restrict__ output, int period, int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = period - 1; i < size; ++i) {
      float highest = high[i];
      float lowest = low[i];
      for (int j = 1; j < period; ++j) {
        float h = high[i - j];
        float l = low[i - j];
        if (h > highest)
          highest = h;
        if (l < lowest)
          lowest = l;
      }
      float denom = highest - lowest;
      output[i] =
          denom == 0.0f ? 0.0f : ((highest - close[i]) / denom) * -100.0f;
    }
  }
}

WILLR::WILLR(int period) : period(period) {}

void WILLR::calculate(const float *high, const float *low, const float *close,
                      float *output, int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("WILLR: invalid period");
  }
  CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
  willrKernel<<<1, 1, 0, stream>>>(high, low, close, output, period, size);
  CUDA_CHECK(cudaGetLastError());
}

void WILLR::calculate(const float *input, float *output,
                      int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  const float *close = input + 2 * size;
  calculate(high, low, close, output, size, stream);
}
