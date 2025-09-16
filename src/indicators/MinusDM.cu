#include <indicators/MinusDM.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void minusDMKernel(const float *__restrict__ high,
                              const float *__restrict__ low,
                              float *__restrict__ output, int period,
                              int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float prevHigh = high[0];
    float prevLow = low[0];
    float prevDM = 0.0f;
    for (int i = 1; i < period; ++i) {
      float diffP = high[i] - prevHigh;
      prevHigh = high[i];
      float diffM = prevLow - low[i];
      prevLow = low[i];
      if (diffM > 0.0f && diffM > diffP)
        prevDM += diffM;
    }
    output[period - 1] = prevDM;
    for (int i = period; i < size; ++i) {
      float diffP = high[i] - prevHigh;
      prevHigh = high[i];
      float diffM = prevLow - low[i];
      prevLow = low[i];
      if (diffM > 0.0f && diffM > diffP)
        prevDM = prevDM - prevDM / period + diffM;
      else
        prevDM = prevDM - prevDM / period;
      output[i] = prevDM;
    }
  }
}

tacuda::MinusDM::MinusDM(int period) : period(period) {}

void tacuda::MinusDM::calculate(const float *high, const float *low, float *output,
                        int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("MinusDM: invalid period");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  minusDMKernel<<<1, 1, 0, stream>>>(high, low, output, period, size);
  CUDA_CHECK(cudaGetLastError());
}

void tacuda::MinusDM::calculate(const float *input, float *output,
                        int size, cudaStream_t stream) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  calculate(high, low, output, size, stream);
}
