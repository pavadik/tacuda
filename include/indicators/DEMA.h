#ifndef DEMA_H
#define DEMA_H

#include "Indicator.h"

namespace tacuda {
class DEMA : public Indicator {
public:
    explicit DEMA(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
