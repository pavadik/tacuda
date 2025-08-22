#include <indicators/MA.h>
#include <indicators/SMA.h>
#include <indicators/EMA.h>
#include <stdexcept>

MA::MA(int period, MAType type) : period(period), type(type) {}

void MA::calculate(const float* input, float* output, int size) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("MA: invalid period");
    }
    switch (type) {
    case MAType::SMA: {
        SMA sma(period);
        sma.calculate(input, output, size);
        break;
    }
    case MAType::EMA: {
        EMA ema(period);
        ema.calculate(input, output, size);
        break;
    }
    default:
        throw std::invalid_argument("MA: unsupported type");
    }
}
