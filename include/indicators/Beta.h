#ifndef BETA_H
#define BETA_H

#include "Indicator.h"

namespace tacuda {
class Beta : public Indicator {
public:
    explicit Beta(int period);
    void calculate(const float* x, const float* y, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
