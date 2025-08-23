#include <indicators/RSI.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void rsiKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - period) {
        float gain = 0.0f;
        float loss = 0.0f;
        for (int i = 0; i < period; ++i) {
            float diff = input[idx + i + 1] - input[idx + i];
            if (diff > 0.0f) {
                gain += diff;
            } else {
                loss -= diff; // diff is negative
            }
        }
        float avgGain = gain / period;
        float avgLoss = loss / period;
        float rsi;
        if (avgLoss == 0.0f) {
            rsi = (avgGain == 0.0f) ? 50.0f : 100.0f;
        } else if (avgGain == 0.0f) {
            rsi = 0.0f;
        } else {
            float rs = avgGain / avgLoss;
            rsi = 100.0f - 100.0f / (1.0f + rs);
        }
        output[idx] = rsi;
    }
}

RSI::RSI(int period) : period(period) {}

void RSI::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period >= size) {
        throw std::invalid_argument("RSI: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    rsiKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
