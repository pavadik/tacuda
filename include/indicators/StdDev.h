#ifndef STDDEV_H
#define STDDEV_H

#include "Indicator.h"

namespace tacuda {
class StdDev : public Indicator {
public:
    explicit StdDev(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
