#ifndef CMO_H
#define CMO_H

#include "Indicator.h"

class CMO : public Indicator {
public:
    explicit CMO(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

#endif
