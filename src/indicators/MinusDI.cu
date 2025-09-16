#include <indicators/MinusDI.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void minusDIKernel(const float *__restrict__ high,
                              const float *__restrict__ low,
                              const float *__restrict__ close,
                              float *__restrict__ output, int period,
                              int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float prevHigh = high[0];
    float prevLow = low[0];
    float prevClose = close[0];
    float dmm_s = 0.0f;
    float tr_s = 0.0f;
    for (int i = 1; i < size; ++i) {
      float upMove = high[i] - prevHigh;
      float downMove = prevLow - low[i];
      float dmMinus = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
      float tr = fmaxf(high[i] - low[i], fmaxf(fabsf(high[i] - prevClose),
                                               fabsf(low[i] - prevClose)));
      prevHigh = high[i];
      prevLow = low[i];
      prevClose = close[i];
      if (i <= period) {
        dmm_s += dmMinus;
        tr_s += tr;
        if (i == period)
          output[i] = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
      } else {
        dmm_s = dmm_s - dmm_s / period + dmMinus;
        tr_s = tr_s - tr_s / period + tr;
        output[i] = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
      }
    }
  }
}

tacuda::MinusDI::MinusDI(int period) : period(period) {}

void tacuda::MinusDI::calculate(const float *high, const float *low, const float *close,
                        float *output, int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("MinusDI: invalid period");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  minusDIKernel<<<1, 1, 0, stream>>>(high, low, close, output, period, size);
  CUDA_CHECK(cudaGetLastError());
}

void tacuda::MinusDI::calculate(const float *input, float *output,
                        int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  const float *close = input + 2 * size;
  calculate(high, low, close, output, size, stream);
}
