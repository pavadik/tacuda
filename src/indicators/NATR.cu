#include <indicators/NATR.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void natrKernel(const float *__restrict__ high,
                           const float *__restrict__ low,
                           const float *__restrict__ close,
                           float *__restrict__ output, int period, int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float prevClose = close[0];
    float sum = 0.0f;
    for (int i = 0; i < period; ++i) {
      float tr = fmaxf(high[i] - low[i],
                       fmaxf(fabsf(high[i] - prevClose),
                             fabsf(low[i] - prevClose)));
      sum += tr;
      prevClose = close[i];
    }
    float atr = sum / period;
    output[period - 1] = close[period - 1] == 0.0f
                             ? 0.0f
                             : 100.0f * atr / close[period - 1];
    for (int i = period; i < size; ++i) {
      float tr = fmaxf(high[i] - low[i],
                       fmaxf(fabsf(high[i] - prevClose),
                             fabsf(low[i] - prevClose)));
      atr = (atr * (period - 1) + tr) / period;
      output[i] = close[i] == 0.0f ? 0.0f : 100.0f * atr / close[i];
      prevClose = close[i];
    }
  }
}

tacuda::NATR::NATR(int period) : period(period) {}

void tacuda::NATR::calculate(const float *high, const float *low, const float *close,
                     float *output, int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("NATR: invalid period");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  natrKernel<<<1, 1, 0, stream>>>(high, low, close, output, period, size);
  CUDA_CHECK(cudaGetLastError());
}

void tacuda::NATR::calculate(const float *input, float *output,
                     int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  const float *close = input + 2 * size;
  calculate(high, low, close, output, size, stream);
}
