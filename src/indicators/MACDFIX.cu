#include <indicators/MACDFIX.h>
#include <indicators/MACDEXT.h>

tacuda::MACDFIX::MACDFIX(int signalPeriod) : signalPeriod(signalPeriod) {}

void tacuda::MACDFIX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    tacuda::MACDEXT macd(12, 26, signalPeriod, tacuda::MAType::EMA);
    macd.calculate(input, output, size, stream);
}

