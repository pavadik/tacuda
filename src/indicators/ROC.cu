#include <indicators/ROC.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void rocKernel(const float* __restrict__ input, float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - period && input[idx] != 0.0f) {
        float prev = input[idx];
        float curr = input[idx + period];
        output[idx] = ((curr - prev) / prev) * 100.0f;
    }
}

tacuda::ROC::ROC(int period) : period(period) {}

void tacuda::ROC::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period >= size) {
        throw std::invalid_argument("ROC: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    rocKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}
