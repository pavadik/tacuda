#ifndef SAREXT_H
#define SAREXT_H

#include "Indicator.h"

class SAREXT : public Indicator {
public:
    SAREXT(float startValue, float offsetOnReverse,
           float accInitLong, float accLong, float accMaxLong,
           float accInitShort, float accShort, float accMaxShort);
    void calculate(const float* high, const float* low,
                   float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    float startValue, offsetOnReverse;
    float accInitLong, accLong, accMaxLong;
    float accInitShort, accShort, accMaxShort;
};

#endif
