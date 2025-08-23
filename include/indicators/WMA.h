#ifndef WMA_H
#define WMA_H

#include "Indicator.h"

class WMA : public Indicator {
public:
    explicit WMA(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

#endif
