#include <indicators/AroonOscillator.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void aroonOscKernel(const float* __restrict__ high,
                               const float* __restrict__ low,
                               float* __restrict__ output,
                               int period, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < size; ++i) {
            if (i >= period) {
                int sinceHigh = 0;
                int sinceLow = 0;
                float maxVal = high[i];
                float minVal = low[i];
                for (int j = 1; j <= period; ++j) {
                    float h = high[i - j];
                    float l = low[i - j];
                    if (h >= maxVal) {
                        maxVal = h;
                        sinceHigh = j;
                    }
                    if (l <= minVal) {
                        minVal = l;
                        sinceLow = j;
                    }
                }
                float up = 100.0f * (period - sinceHigh) / period;
                float down = 100.0f * (period - sinceLow) / period;
                output[i] = up - down;
            }
        }
    }
}

AroonOscillator::AroonOscillator(int period) : period(period) {}

void AroonOscillator::calculate(const float* high, const float* low,
                                float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("AroonOscillator: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    aroonOscKernel<<<1,1>>>(high, low, output, period, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void AroonOscillator::calculate(const float* input, float* output, int size) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size);
}
