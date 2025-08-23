#ifndef INDICATOR_H
#define INDICATOR_H

#include <cuda_runtime.h>

class Indicator {
public:
    virtual void calculate(const float* input, float* output, int size,
                           cudaStream_t stream = 0) noexcept(false) = 0;
    virtual ~Indicator() = default;
};

#endif
