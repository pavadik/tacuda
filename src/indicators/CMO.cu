#include <indicators/CMO.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void cmoKernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - period) {
        float up = 0.0f, down = 0.0f;
        for (int i = 0; i < period; ++i) {
            float diff = input[idx + i + 1] - input[idx + i];
            if (diff > 0.0f) {
                up += diff;
            } else {
                down -= diff;
            }
        }
        float denom = up + down;
        output[idx] = (denom == 0.0f) ? 0.0f : 100.0f * (up - down) / denom;
    }
}

tacuda::CMO::CMO(int period) : period(period) {}

void tacuda::CMO::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period >= size) {
        throw std::invalid_argument("CMO: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    cmoKernel<<<grid, block, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

