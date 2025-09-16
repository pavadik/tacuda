#ifndef CORREL_H
#define CORREL_H

#include "Indicator.h"

namespace tacuda {
class Correl : public Indicator {
public:
    explicit Correl(int period);
    void calculate(const float* x, const float* y, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
