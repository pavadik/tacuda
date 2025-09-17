#include <indicators/EMA.h>
#include <indicators/T3.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

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

tacuda::T3::T3(int period, float vFactor) : period(period), vFactor(vFactor) {}

void tacuda::T3::calculate(const float *input, float *output,
                   int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0 || size < 3 * period - 2) {
    throw std::invalid_argument("T3: invalid period");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

  auto e1 = acquireDeviceBuffer<float>(size);
  auto e2 = acquireDeviceBuffer<float>(size);
  auto e3 = acquireDeviceBuffer<float>(size);
  auto e4 = acquireDeviceBuffer<float>(size);
  auto e5 = acquireDeviceBuffer<float>(size);
  auto e6 = acquireDeviceBuffer<float>(size);

  tacuda::EMA ema(period);
  ema.calculate(input, e1.get(), size, stream);
  int size2 = size - period + 1;
  ema.calculate(e1.get(), e2.get(), size2, stream);
  int size3 = size2 - period + 1;
  ema.calculate(e2.get(), e3.get(), size3, stream);
  int size4 = size3 - period + 1;
  ema.calculate(e3.get(), e4.get(), size4, stream);
  int size5 = size4 - period + 1;
  ema.calculate(e4.get(), e5.get(), size5, stream);
  int size6 = size5 - period + 1;
  ema.calculate(e5.get(), e6.get(), size6, stream);

  float b = vFactor;
  float c1 = -b * b * b;
  float c2 = 3 * b * b + 3 * b * b * b;
  float c3 = -3 * b - 6 * b * b - 3 * b * b * b;
  float c4 = 1 + 3 * b + 3 * b * b + b * b * b;

  int valid = size - 3 * period + 3;
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(valid);
  t3Kernel<<<grid, block, 0, stream>>>(e3.get(), e4.get(), e5.get(), e6.get(), output, c1, c2, c3, c4, period,
                            valid);
  CUDA_CHECK(cudaGetLastError());
}
