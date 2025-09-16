#ifndef VAR_H
#define VAR_H

#include "Indicator.h"

namespace tacuda {
class VAR : public Indicator {
public:
    explicit VAR(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
