#ifndef ROC_H
#define ROC_H

#include "Indicator.h"

class ROC : public Indicator {
public:
    explicit ROC(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

#endif
