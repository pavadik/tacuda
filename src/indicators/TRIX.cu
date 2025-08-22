#include <indicators/TRIX.h>
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void trixKernel(const float* __restrict__ ema3,
                           float* __restrict__ output,
                           int valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < valid && ema3[idx] != 0.0f) {
        float prev = ema3[idx];
        float curr = ema3[idx + 1];
        output[idx] = ((curr - prev) / prev) * 100.0f;
    }
}

TRIX::TRIX(int period) : period(period) {}

void TRIX::calculate(const float* input, float* output, int size) noexcept(false) {
    if (period <= 0 || size < 3 * period - 1) {
        throw std::invalid_argument("TRIX: invalid period");
    }

    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    float* ema1 = nullptr;
    float* ema2 = nullptr;
    float* ema3 = nullptr;
    CUDA_CHECK(cudaMalloc(&ema1, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ema2, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ema3, size * sizeof(float)));

    EMA ema(period);
    ema.calculate(input, ema1, size);
    int size2 = size - period + 1;
    ema.calculate(ema1, ema2, size2);
    int size3 = size2 - period + 1;
    ema.calculate(ema2, ema3, size3);

    int valid = size3 - 1;
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(valid);
    trixKernel<<<grid, block>>>(ema3, output, valid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(ema1);
    cudaFree(ema2);
    cudaFree(ema3);
}
