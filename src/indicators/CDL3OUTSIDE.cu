#include <indicators/CDL3OUTSIDE.h>
#include <utils/CandleUtils.h>
#include <utils/CudaUtils.h>

namespace {
__device__ inline int candleColor(float open, float close) {
    if (close > open) return 1;
    if (close < open) return -1;
    return 0;
}

__global__ void cdl3outsideKernel(const float* __restrict__ open,
                                  const float* __restrict__ high,
                                  const float* __restrict__ low,
                                  const float* __restrict__ close,
                                  float* __restrict__ output,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx < size) {
        int colorPrev1 = candleColor(open[idx - 1], close[idx - 1]);
        int colorPrev2 = candleColor(open[idx - 2], close[idx - 2]);
        bool bullish = colorPrev1 == 1 && colorPrev2 == -1 &&
                       close[idx - 1] > open[idx - 2] &&
                       open[idx - 1] < close[idx - 2] &&
                       close[idx] > close[idx - 1];
        bool bearish = colorPrev1 == -1 && colorPrev2 == 1 &&
                       open[idx - 1] > close[idx - 2] &&
                       close[idx - 1] < open[idx - 2] &&
                       close[idx] < close[idx - 1];
        if (bullish) {
            output[idx] = 100.0f;
        } else if (bearish) {
            output[idx] = -100.0f;
        } else {
            output[idx] = 0.0f;
        }
    }
}
} // namespace

void tacuda::CDL3OUTSIDE::calculate(const float* open, const float* high,
                                    const float* low, const float* close,
                                    float* output, int size,
                                    cudaStream_t stream) noexcept(false) {
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    cdl3outsideKernel<<<grid, block, 0, stream>>>(open, high, low, close, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::CDL3OUTSIDE::calculate(const float* input, float* output, int size,
                                    cudaStream_t stream) noexcept(false) {
    const float* open = input;
    const float* high = input + size;
    const float* low = input + 2 * size;
    const float* close = input + 3 * size;
    calculate(open, high, low, close, output, size, stream);
}
