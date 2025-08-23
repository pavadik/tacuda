#ifndef MFI_H
#define MFI_H

#include "Indicator.h"

class MFI : public Indicator {
public:
    explicit MFI(int period);
    void calculate(const float* high, const float* low, const float* close,
                   const float* volume, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

#endif
