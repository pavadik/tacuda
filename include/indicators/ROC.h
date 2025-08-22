#ifndef ROC_H
#define ROC_H

#include "Indicator.h"

class ROC : public Indicator {
public:
    explicit ROC(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
