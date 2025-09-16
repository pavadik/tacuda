#include <indicators/ROCR.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void rocrKernel(const float *__restrict__ input,
                           float *__restrict__ output, int period, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size - period && input[idx] != 0.0f) {
    float prev = input[idx];
    float curr = input[idx + period];
    output[idx] = curr / prev;
  }
}

ROCR::ROCR(int period) : period(period) {}

void ROCR::calculate(const float *input, float *output,
                     int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0 || period >= size) {
    throw std::invalid_argument("ROCR: invalid period");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  rocrKernel<<<grid, block, 0, stream>>>(input, output, period, size);
  CUDA_CHECK(cudaGetLastError());
}
