#include <indicators/KAMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

__global__ void kamaKernel(const float *__restrict__ input,
                           float *__restrict__ output, int period, float fastSC,
                           float slowSC, int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (size <= period)
      return;
    float prevKama = input[period - 1];
    for (int i = period; i < size; ++i) {
      float change = fabsf(input[i] - input[i - period]);
      float volatility = 0.0f;
      for (int j = i - period + 1; j <= i; ++j) {
        volatility += fabsf(input[j] - input[j - 1]);
      }
      float er = volatility > 0.0f ? (change / volatility) : 0.0f;
      float sc = er * (fastSC - slowSC) + slowSC;
      sc *= sc;
      prevKama = prevKama + sc * (input[i] - prevKama);
      output[i - period] = prevKama;
    }
  }
}

KAMA::KAMA(int period, int fastPeriod, int slowPeriod)
    : period(period), fastSC(2.0f / (fastPeriod + 1.0f)),
      slowSC(2.0f / (slowPeriod + 1.0f)) {}

void KAMA::calculate(const float *input, float *output,
                     int size) noexcept(false) {
  if (period <= 0 || period > size) {
    throw std::invalid_argument("KAMA: invalid period");
  }
  // Initialize output with NaNs so unwritten tail remains NaN
  CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

  kamaKernel<<<1, 1>>>(input, output, period, fastSC, slowSC, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
