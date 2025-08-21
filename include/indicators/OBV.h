#ifndef OBV_H
#define OBV_H

#include "Indicator.h"

class OBV : public Indicator {
public:
    OBV() = default;
    void calculate(const float* price, const float* volume,
                   float* output, int size) noexcept(false);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
};

#endif
