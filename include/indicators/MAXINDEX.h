#ifndef MAXINDEX_H
#define MAXINDEX_H

#include "Indicator.h"

class MAXINDEX : public Indicator {
public:
    explicit MAXINDEX(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

#endif
