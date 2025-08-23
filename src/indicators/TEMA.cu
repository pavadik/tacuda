#include <indicators/TEMA.h>
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void temaKernel(const float* __restrict__ ema1,
                           const float* __restrict__ ema2,
                           const float* __restrict__ ema3,
                           float* __restrict__ output,
                           int period,
                           int valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < valid) {
        output[idx] = 3.0f * ema1[idx + 2 * (period - 1)]
                    - 3.0f * ema2[idx + (period - 1)]
                    + ema3[idx];
    }
}

TEMA::TEMA(int period) : period(period) {}

void TEMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || size < 3 * period - 2) {
        throw std::invalid_argument("TEMA: invalid period");
    }

    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    float* ema1 = nullptr;
    float* ema2 = nullptr;
    float* ema3 = nullptr;
    CUDA_CHECK(cudaMalloc(&ema1, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ema2, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ema3, size * sizeof(float)));

    EMA ema(period);
    ema.calculate(input, ema1, size, stream);
    int size2 = size - period + 1;
    ema.calculate(ema1, ema2, size2);
    int size3 = size2 - period + 1;
    ema.calculate(ema2, ema3, size3);

    int valid = size - 3 * period + 3;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    temaKernel<<<grid, block, 0, stream>>>(ema1, ema2, ema3, output, period, valid);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(ema1);
    cudaFree(ema2);
    cudaFree(ema3);
}
