#ifndef MIDPOINT_H
#define MIDPOINT_H

#include "Indicator.h"

namespace tacuda {
class MIDPOINT : public Indicator {
public:
    explicit MIDPOINT(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
