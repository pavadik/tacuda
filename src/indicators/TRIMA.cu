#include <indicators/TRIMA.h>
#include <indicators/SMA.h>
#include <utils/CudaUtils.h>
#include <stdexcept>

TRIMA::TRIMA(int period) : period(period) {}

void TRIMA::calculate(const float* input, float* output, int size) noexcept(false) {
    if (period <= 0 || size < 2 * period - 1) {
        throw std::invalid_argument("TRIMA: invalid period");
    }

    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));

    float* tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp, size * sizeof(float)));

    SMA sma(period);
    sma.calculate(input, tmp, size);
    int size2 = size - period + 1;
    sma.calculate(tmp, output, size2);

    cudaFree(tmp);
}
