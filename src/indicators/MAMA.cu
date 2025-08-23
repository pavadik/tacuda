#include <indicators/MAMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

__global__ void mamaKernel(const float* __restrict__ input,
                           float* __restrict__ mama,
                           float* __restrict__ fama,
                           float fast, float slow, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float prevMama = input[0];
        float prevFama = input[0];
        mama[0] = prevMama;
        fama[0] = prevFama;
        for (int i = 1; i < size; ++i) {
            float m = fast * input[i] + (1.0f - fast) * prevMama;
            prevMama = m;
            mama[i] = m;
            float f = slow * m + (1.0f - slow) * prevFama;
            prevFama = f;
            fama[i] = f;
        }
    }
}

MAMA::MAMA(float fastLimit, float slowLimit)
    : fastLimit(fastLimit), slowLimit(slowLimit) {}

void MAMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("MAMA: invalid size");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, 2 * size * sizeof(float)));
    float* mama = output;
    float* fama = output + size;
    mamaKernel<<<1, 1, 0, stream>>>(input, mama, fama, fastLimit, slowLimit, size);
    CUDA_CHECK(cudaGetLastError());
}
