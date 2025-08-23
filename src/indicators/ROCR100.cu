#include <indicators/ROCR100.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void rocr100Kernel(const float *__restrict__ input,
                              float *__restrict__ output, int period,
                              int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size - period && input[idx] != 0.0f) {
    float prev = input[idx];
    float curr = input[idx + period];
    output[idx] = (curr / prev) * 100.0f;
  }
}

ROCR100::ROCR100(int period) : period(period) {}

void ROCR100::calculate(const float *input, float *output,
                        int size) noexcept(false) {
  if (period <= 0 || period >= size) {
    throw std::invalid_argument("ROCR100: invalid period");
  }
  CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  rocr100Kernel<<<grid, block>>>(input, output, period, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
