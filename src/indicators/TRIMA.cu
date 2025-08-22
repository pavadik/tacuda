#include <indicators/SMA.h>
#include <indicators/TRIMA.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

TRIMA::TRIMA(int period) : period(period) {}

void TRIMA::calculate(const float *input, float *output,
                      int size) noexcept(false) {
  if (period <= 0 || size < period) {
    throw std::invalid_argument("TRIMA: invalid period");
  }
  CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
  int p1 = (period + 1) / 2;
  int p2 = (period % 2 == 0) ? (p1 + 1) : p1;

  float *tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&tmp, size * sizeof(float)));

  SMA sma1(p1);
  sma1.calculate(input, tmp, size);
  int size2 = size - p1 + 1;
  SMA sma2(p2);
  sma2.calculate(tmp, output, size2);

  CUDA_CHECK(cudaFree(tmp));
}
