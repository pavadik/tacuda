#include <indicators/ULTOSC.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void ultoscKernel(const float* __restrict__ high,
                             const float* __restrict__ low,
                             const float* __restrict__ close,
                             float* __restrict__ output,
                             int shortP, int medP, int longP, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx >= longP) {
        float bp1 = 0.0f, tr1 = 0.0f;
        float bp2 = 0.0f, tr2 = 0.0f;
        float bp3 = 0.0f, tr3 = 0.0f;
        for (int j = 0; j < longP; ++j) {
            int i = idx - j;
            float prevClose = (i > 0) ? close[i - 1] : close[i];
            float bp = close[i] - fminf(low[i], prevClose);
            float tr = fmaxf(high[i], prevClose) - fminf(low[i], prevClose);
            if (j < shortP) { bp1 += bp; tr1 += tr; }
            if (j < medP)   { bp2 += bp; tr2 += tr; }
            bp3 += bp; tr3 += tr;
        }
        float avg1 = (tr1 == 0.0f) ? 0.0f : bp1 / tr1;
        float avg2 = (tr2 == 0.0f) ? 0.0f : bp2 / tr2;
        float avg3 = (tr3 == 0.0f) ? 0.0f : bp3 / tr3;
        output[idx] = 100.0f * (4.0f * avg1 + 2.0f * avg2 + avg3) / 7.0f;
    }
}

tacuda::ULTOSC::ULTOSC(int shortPeriod, int mediumPeriod, int longPeriod)
    : shortPeriod(shortPeriod), mediumPeriod(mediumPeriod), longPeriod(longPeriod) {}

void tacuda::ULTOSC::calculate(const float* high, const float* low, const float* close,
                       float* output, int size, cudaStream_t stream) noexcept(false) {
    if (shortPeriod <= 0 || mediumPeriod <= 0 || longPeriod <= 0 ||
        shortPeriod > mediumPeriod || mediumPeriod > longPeriod ||
        longPeriod >= size) {
        throw std::invalid_argument("ULTOSC: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    ultoscKernel<<<grid, block, 0, stream>>>(high, low, close, output,
                                  shortPeriod, mediumPeriod, longPeriod, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::ULTOSC::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
