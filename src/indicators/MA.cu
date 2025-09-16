#include <indicators/MA.h>
#include <indicators/SMA.h>
#include <indicators/EMA.h>
#include <stdexcept>

tacuda::MA::MA(int period, tacuda::MAType type) : period(period), type(type) {}

void tacuda::MA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MA: invalid period");
    }
    switch (type) {
    case tacuda::MAType::SMA: {
        tacuda::SMA sma(period);
        sma.calculate(input, output, size, stream);
        break;
    }
    case tacuda::MAType::EMA: {
        tacuda::EMA ema(period);
        ema.calculate(input, output, size, stream);
        break;
    }
    default:
        throw std::invalid_argument("MA: unsupported type");
    }
}
