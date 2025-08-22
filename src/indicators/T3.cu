#include <indicators/EMA.h>
#include <indicators/T3.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void t3Kernel(const float *__restrict__ e3,
                         const float *__restrict__ e4,
                         const float *__restrict__ e5,
                         const float *__restrict__ e6,
                         float *__restrict__ output, float c1, float c2,
                         float c3, float c4, int period, int valid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < valid) {
    int p1 = period - 1;
    output[idx] = c1 * e6[idx + 3 * p1] + c2 * e5[idx + 2 * p1] +
                  c3 * e4[idx + p1] + c4 * e3[idx];
  }
}

T3::T3(int period, float vFactor) : period(period), vFactor(vFactor) {}

void T3::calculate(const float *input, float *output,
                   int size) noexcept(false) {
  if (period <= 0 || size < 3 * period - 2) {
    throw std::invalid_argument("T3: invalid period");
  }
  CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

  float *e1 = nullptr, *e2 = nullptr, *e3 = nullptr, *e4 = nullptr,
        *e5 = nullptr, *e6 = nullptr;
  CUDA_CHECK(cudaMalloc(&e1, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&e2, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&e3, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&e4, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&e5, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&e6, size * sizeof(float)));

  EMA ema(period);
  ema.calculate(input, e1, size);
  int size2 = size - period + 1;
  ema.calculate(e1, e2, size2);
  int size3 = size2 - period + 1;
  ema.calculate(e2, e3, size3);
  int size4 = size3 - period + 1;
  ema.calculate(e3, e4, size4);
  int size5 = size4 - period + 1;
  ema.calculate(e4, e5, size5);
  int size6 = size5 - period + 1;
  ema.calculate(e5, e6, size6);

  float b = vFactor;
  float c1 = -b * b * b;
  float c2 = 3 * b * b + 3 * b * b * b;
  float c3 = -3 * b - 6 * b * b - 3 * b * b * b;
  float c4 = 1 + 3 * b + 3 * b * b + b * b * b;

  int valid = size - 3 * period + 3;
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(valid);
  t3Kernel<<<grid, block>>>(e3, e4, e5, e6, output, c1, c2, c3, c4, period,
                            valid);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(e1);
  cudaFree(e2);
  cudaFree(e3);
  cudaFree(e4);
  cudaFree(e5);
  cudaFree(e6);
}
