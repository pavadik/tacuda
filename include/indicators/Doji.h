#ifndef DOJI_H
#define DOJI_H

#include "Indicator.h"

class Doji : public Indicator {
public:
    explicit Doji(float threshold = 0.1f);
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    float threshold;
};

#endif
