#include <indicators/PlusDM.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void plusDMKernel(const float *__restrict__ high,
                             const float *__restrict__ low,
                             float *__restrict__ output, int period, int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float prevHigh = high[0];
    float prevLow = low[0];
    float prevDM = 0.0f;
    for (int i = 1; i < period; ++i) {
      float diffP = high[i] - prevHigh;
      prevHigh = high[i];
      float diffM = prevLow - low[i];
      prevLow = low[i];
      if (diffP > 0.0f && diffP > diffM)
        prevDM += diffP;
    }
    output[period - 1] = prevDM;
    for (int i = period; i < size; ++i) {
      float diffP = high[i] - prevHigh;
      prevHigh = high[i];
      float diffM = prevLow - low[i];
      prevLow = low[i];
      if (diffP > 0.0f && diffP > diffM)
        prevDM = prevDM - prevDM / period + diffP;
      else
        prevDM = prevDM - prevDM / period;
      output[i] = prevDM;
    }
  }
}

PlusDM::PlusDM(int period) : period(period) {}

void PlusDM::calculate(const float *high, const float *low, float *output,
                       int size) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("PlusDM: invalid period");
  }
  CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
  plusDMKernel<<<1, 1>>>(high, low, output, period, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void PlusDM::calculate(const float *input, float *output,
                       int size) noexcept(false) {
  const float *high = input;
  const float *low = input + size;
  calculate(high, low, output, size);
}
