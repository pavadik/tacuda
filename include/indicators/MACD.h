#ifndef MACD_H
#define MACD_H

#include <cstddef>

// MACD indicator: computes the MACD line (fast EMA minus slow EMA),
// the signal line (EMA of the MACD line) and the histogram (line - signal).
// The calculate method expects three output arrays of length `size`.
class MACD {
public:
    MACD(int fastPeriod, int slowPeriod, int signalPeriod);
    void calculate(const float* input, float* lineOut, float* signalOut,
                   float* histOut, std::size_t size) noexcept(false);
private:
    int fastPeriod;
    int slowPeriod;
    int signalPeriod;
};

#endif
