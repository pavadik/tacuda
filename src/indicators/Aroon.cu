#include <indicators/Aroon.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void aroonKernel(const float* __restrict__ high,
                            const float* __restrict__ low,
                            float* __restrict__ output,
                            int upPeriod,
                            int downPeriod,
                            int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float* upOut = output;
        float* downOut = output + size;
        float* oscOut = output + 2 * size;
        int maxPeriod = upPeriod > downPeriod ? upPeriod : downPeriod;
        for (int i = 0; i < size; ++i) {
            float up = NAN;
            float down = NAN;
            if (i >= upPeriod) {
                int sinceHigh = 0;
                float maxVal = high[i];
                for (int j = 1; j <= upPeriod; ++j) {
                    float val = high[i - j];
                    if (val >= maxVal) {
                        maxVal = val;
                        sinceHigh = j;
                    }
                }
                up = 100.0f * (upPeriod - sinceHigh) / upPeriod;
                upOut[i] = up;
            }
            if (i >= downPeriod) {
                int sinceLow = 0;
                float minVal = low[i];
                for (int j = 1; j <= downPeriod; ++j) {
                    float val = low[i - j];
                    if (val <= minVal) {
                        minVal = val;
                        sinceLow = j;
                    }
                }
                down = 100.0f * (downPeriod - sinceLow) / downPeriod;
                downOut[i] = down;
            }
            if (i >= maxPeriod) {
                oscOut[i] = up - down;
            }
        }
    }
}

tacuda::Aroon::Aroon(int upPeriod, int downPeriod) : upPeriod(upPeriod), downPeriod(downPeriod) {}

void tacuda::Aroon::calculate(const float* high, const float* low, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (upPeriod <= 0 || upPeriod > size || downPeriod <= 0 || downPeriod > size) {
        throw std::invalid_argument("Aroon: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 3 * size * sizeof(float), stream));
    aroonKernel<<<1, 1, 0, stream>>>(high, low, output, upPeriod, downPeriod, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::Aroon::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    calculate(high, low, output, size, stream);
}

