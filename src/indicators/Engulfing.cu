#include <indicators/Engulfing.h>
#include <utils/CandleUtils.h>
#include <utils/CudaUtils.h>

__global__ void engulfingKernel(const float *__restrict__ open,
                                const float *__restrict__ high,
                                const float *__restrict__ low,
                                const float *__restrict__ close,
                                float *__restrict__ output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > 0 && idx < size) {
    output[idx] =
        engulfing(open[idx - 1], close[idx - 1], open[idx], close[idx]);
  }
}

void tacuda::Engulfing::calculate(const float *open, const float *high,
                          const float *low, const float *close, float *output,
                          int size, cudaStream_t stream) noexcept(false) {
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  engulfingKernel<<<grid, block, 0, stream>>>(open, high, low, close, output, size);
  CUDA_CHECK(cudaGetLastError());
}

void tacuda::Engulfing::calculate(const float *input, float *output,
                          int size, cudaStream_t stream) noexcept(false) {
  const float *open = input;
  const float *high = input + size;
  const float *low = input + 2 * size;
  const float *close = input + 3 * size;
  calculate(open, high, low, close, output, size, stream);
}
