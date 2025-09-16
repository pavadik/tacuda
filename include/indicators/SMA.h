#ifndef SMA_H
#define SMA_H

#include "Indicator.h"

namespace tacuda {
class SMA : public Indicator {
public:
    explicit SMA(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
