#include <indicators/MACDFIX.h>
#include <indicators/MACDEXT.h>

MACDFIX::MACDFIX(int signalPeriod) : signalPeriod(signalPeriod) {}

void MACDFIX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    MACDEXT macd(12, 26, signalPeriod, MAType::EMA);
    macd.calculate(input, output, size, stream);
}

